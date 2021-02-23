#===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##

import copy
import os
import pkgutil
import pipes
import re
import shlex
import shutil
import sys

from libcxx.compiler import CXXCompiler
from libcxx.test.target_info import make_target_info
import libcxx.util
import libcxx.test.features
import libcxx.test.newconfig
import libcxx.test.params

def loadSiteConfig(lit_config, config, param_name, env_name):
    # We haven't loaded the site specific configuration (the user is
    # probably trying to run on a test file directly, and either the site
    # configuration hasn't been created by the build system, or we are in an
    # out-of-tree build situation).
    site_cfg = lit_config.params.get(param_name,
                                     os.environ.get(env_name))
    if not site_cfg:
        lit_config.warning('No site specific configuration file found!'
                           ' Running the tests in the default configuration.')
    elif not os.path.isfile(site_cfg):
        lit_config.fatal(
            "Specified site configuration file does not exist: '%s'" %
            site_cfg)
    else:
        lit_config.note('using site specific configuration at %s' % site_cfg)
        ld_fn = lit_config.load_config

        # Null out the load_config function so that lit.site.cfg doesn't
        # recursively load a config even if it tries.
        # TODO: This is one hell of a hack. Fix it.
        def prevent_reload_fn(*args, **kwargs):
            pass
        lit_config.load_config = prevent_reload_fn
        ld_fn(config, site_cfg)
        lit_config.load_config = ld_fn

# Extract the value of a numeric macro such as __cplusplus or a feature-test
# macro.
def intMacroValue(token):
    return int(token.rstrip('LlUu'))

class Configuration(object):
    # pylint: disable=redefined-outer-name
    def __init__(self, lit_config, config):
        self.lit_config = lit_config
        self.config = config
        self.cxx = None
        self.cxx_is_clang_cl = None
        self.cxx_stdlib_under_test = None
        self.project_obj_root = None
        self.libcxx_src_root = None
        self.libcxx_obj_root = None
        self.cxx_library_root = None
        self.cxx_runtime_root = None
        self.abi_library_root = None
        self.link_shared = self.get_lit_bool('enable_shared', default=True)
        self.debug_build = self.get_lit_bool('debug_build',   default=False)
        self.exec_env = dict()
        self.use_clang_verify = False

    def get_lit_conf(self, name, default=None):
        val = self.lit_config.params.get(name, None)
        if val is None:
            val = getattr(self.config, name, None)
            if val is None:
                val = default
        return val

    def get_lit_bool(self, name, default=None, env_var=None):
        def check_value(value, var_name):
            if value is None:
                return default
            if isinstance(value, bool):
                return value
            if not isinstance(value, str):
                raise TypeError('expected bool or string')
            if value.lower() in ('1', 'true'):
                return True
            if value.lower() in ('', '0', 'false'):
                return False
            self.lit_config.fatal(
                "parameter '{}' should be true or false".format(var_name))

        conf_val = self.get_lit_conf(name)
        if env_var is not None and env_var in os.environ and \
                os.environ[env_var] is not None:
            val = os.environ[env_var]
            if conf_val is not None:
                self.lit_config.warning(
                    'Environment variable %s=%s is overriding explicit '
                    '--param=%s=%s' % (env_var, val, name, conf_val))
            return check_value(val, env_var)
        return check_value(conf_val, name)

    def make_static_lib_name(self, name):
        """Return the full filename for the specified library name"""
        if self.target_info.is_windows():
            assert name == 'c++'  # Only allow libc++ to use this function for now.
            return 'lib' + name + '.lib'
        else:
            return 'lib' + name + '.a'

    def configure(self):
        self.target_info = make_target_info(self)
        self.executor = self.get_lit_conf('executor')
        self.configure_cxx()
        self.configure_triple()
        self.configure_src_root()
        self.configure_obj_root()
        self.cxx_stdlib_under_test = self.get_lit_conf('cxx_stdlib_under_test', 'libc++')
        self.cxx_library_root = self.get_lit_conf('cxx_library_root', self.libcxx_obj_root)
        self.abi_library_root = self.get_lit_conf('abi_library_path', None)
        self.cxx_runtime_root = self.get_lit_conf('cxx_runtime_root', self.cxx_library_root)
        self.configure_compile_flags()
        self.configure_link_flags()
        self.configure_env()
        self.configure_debug_mode()
        self.configure_sanitizer()
        self.configure_coverage()
        self.configure_modules()
        self.configure_substitutions()
        self.configure_features()

        libcxx.test.newconfig.configure(
            libcxx.test.params.DEFAULT_PARAMETERS,
            libcxx.test.features.DEFAULT_FEATURES,
            self.config,
            self.lit_config
        )

        self.lit_config.note("All available features: {}".format(self.config.available_features))

    def print_config_info(self):
        if self.cxx.use_modules:
            self.lit_config.note('Using modules flags: %s' %
                                 self.cxx.modules_flags)
        if len(self.cxx.warning_flags):
            self.lit_config.note('Using warnings: %s' % self.cxx.warning_flags)
        show_env_vars = {}
        for k,v in self.exec_env.items():
            if k not in os.environ or os.environ[k] != v:
                show_env_vars[k] = v
        self.lit_config.note('Adding environment variables: %r' % show_env_vars)
        self.lit_config.note("Linking against the C++ Library at {}".format(self.cxx_library_root))
        self.lit_config.note("Running against the C++ Library at {}".format(self.cxx_runtime_root))
        sys.stderr.flush()  # Force flushing to avoid broken output on Windows

    def get_test_format(self):
        from libcxx.test.format import LibcxxTestFormat
        return LibcxxTestFormat(
            self.cxx,
            self.use_clang_verify,
            self.executor,
            exec_env=self.exec_env)

    def configure_cxx(self):
        # Gather various compiler parameters.
        cxx = self.get_lit_conf('cxx_under_test')
        self.cxx_is_clang_cl = cxx is not None and \
                               os.path.basename(cxx) == 'clang-cl.exe'
        # If no specific cxx_under_test was given, attempt to infer it as
        # clang++.
        if cxx is None or self.cxx_is_clang_cl:
            search_paths = self.config.environment['PATH']
            if cxx is not None and os.path.isabs(cxx):
                search_paths = os.path.dirname(cxx)
            clangxx = libcxx.util.which('clang++', search_paths)
            if clangxx:
                cxx = clangxx
                self.lit_config.note(
                    "inferred cxx_under_test as: %r" % cxx)
            elif self.cxx_is_clang_cl:
                self.lit_config.fatal('Failed to find clang++ substitution for'
                                      ' clang-cl')
        if not cxx:
            self.lit_config.fatal('must specify user parameter cxx_under_test '
                                  '(e.g., --param=cxx_under_test=clang++)')
        self.cxx = CXXCompiler(self, cxx) if not self.cxx_is_clang_cl else \
                   self._configure_clang_cl(cxx)
        self.cxx.compile_env = dict(os.environ)

    def _configure_clang_cl(self, clang_path):
        def _split_env_var(var):
            return [p.strip() for p in os.environ.get(var, '').split(';') if p.strip()]

        def _prefixed_env_list(var, prefix):
            from itertools import chain
            return list(chain.from_iterable((prefix, path) for path in _split_env_var(var)))

        assert self.cxx_is_clang_cl
        flags = []
        compile_flags = _prefixed_env_list('INCLUDE', '-isystem')
        link_flags = _prefixed_env_list('LIB', '-L')
        for path in _split_env_var('LIB'):
            self.add_path(self.exec_env, path)
        return CXXCompiler(self, clang_path, flags=flags,
                           compile_flags=compile_flags,
                           link_flags=link_flags)

    def configure_src_root(self):
        self.libcxx_src_root = self.get_lit_conf(
            'libcxx_src_root', os.path.dirname(self.config.test_source_root))

    def configure_obj_root(self):
        self.project_obj_root = self.get_lit_conf('project_obj_root')
        self.libcxx_obj_root = self.get_lit_conf('libcxx_obj_root')
        if not self.libcxx_obj_root and self.project_obj_root is not None:
            possible_roots = [
                os.path.join(self.project_obj_root, 'libcxx'),
                os.path.join(self.project_obj_root, 'projects', 'libcxx'),
                os.path.join(self.project_obj_root, 'runtimes', 'libcxx'),
            ]
            for possible_root in possible_roots:
                if os.path.isdir(possible_root):
                    self.libcxx_obj_root = possible_root
                    break
            else:
                self.libcxx_obj_root = self.project_obj_root

    def configure_features(self):
        additional_features = self.get_lit_conf('additional_features')
        if additional_features:
            for f in additional_features.split(','):
                self.config.available_features.add(f.strip())

        if self.target_info.is_windows():
            if self.cxx_stdlib_under_test == 'libc++':
                # LIBCXX-WINDOWS-FIXME is the feature name used to XFAIL the
                # initial Windows failures until they can be properly diagnosed
                # and fixed. This allows easier detection of new test failures
                # and regressions. Note: New failures should not be suppressed
                # using this feature. (Also see llvm.org/PR32730)
                self.config.available_features.add('LIBCXX-WINDOWS-FIXME')

        target_triple = getattr(self.config, 'target_triple', None)
        if target_triple:
            if re.match(r'^x86_64.*-apple', target_triple):
                self.config.available_features.add('x86_64-apple')
            if re.match(r'^x86_64.*-linux', target_triple):
                self.config.available_features.add('x86_64-linux')
            if re.match(r'^i.86.*', target_triple):
                self.config.available_features.add('target-x86')
            elif re.match(r'^x86_64.*', target_triple):
                self.config.available_features.add('target-x86_64')
            elif re.match(r'^aarch64.*', target_triple):
                self.config.available_features.add('target-aarch64')
            elif re.match(r'^arm.*', target_triple):
                self.config.available_features.add('target-arm')

    def configure_compile_flags(self):
        self.configure_default_compile_flags()
        # Configure extra flags
        compile_flags_str = self.get_lit_conf('compile_flags', '')
        self.cxx.compile_flags += shlex.split(compile_flags_str)
        if self.target_info.is_windows():
            self.cxx.compile_flags += ['-D_CRT_SECURE_NO_WARNINGS']
            # Don't warn about using common but nonstandard unprefixed functions
            # like chdir, fileno.
            self.cxx.compile_flags += ['-D_CRT_NONSTDC_NO_WARNINGS']
            # Build the tests in the same configuration as libcxx itself,
            # to avoid mismatches if linked statically.
            self.cxx.compile_flags += ['-D_CRT_STDIO_ISO_WIDE_SPECIFIERS']
            # Required so that tests using min/max don't fail on Windows,
            # and so that those tests don't have to be changed to tolerate
            # this insanity.
            self.cxx.compile_flags += ['-DNOMINMAX']
        additional_flags = self.get_lit_conf('test_compiler_flags')
        if additional_flags:
            self.cxx.compile_flags += shlex.split(additional_flags)

    def configure_default_compile_flags(self):
        # Configure include paths
        self.configure_compile_flags_header_includes()
        self.target_info.add_cxx_compile_flags(self.cxx.compile_flags)
        self.target_info.add_cxx_flags(self.cxx.flags)
        # Configure feature flags.
        enable_32bit = self.get_lit_bool('enable_32bit', False)
        if enable_32bit:
            self.cxx.flags += ['-m32']
        # Use verbose output for better errors
        self.cxx.flags += ['-v']
        sysroot = self.get_lit_conf('sysroot')
        if sysroot:
            self.cxx.flags += ['--sysroot=' + sysroot]
        gcc_toolchain = self.get_lit_conf('gcc_toolchain')
        if gcc_toolchain:
            self.cxx.flags += ['--gcc-toolchain=' + gcc_toolchain]
        # NOTE: the _DEBUG definition must preceed the triple check because for
        # the Windows build of libc++, the forced inclusion of a header requires
        # that _DEBUG is defined.  Incorrect ordering will result in -target
        # being elided.
        if self.target_info.is_windows() and self.debug_build:
            self.cxx.compile_flags += ['-D_DEBUG']
        if not self.cxx.addFlagIfSupported(['--target=' + self.config.target_triple]):
            self.lit_config.warning('Not adding any target triple -- the compiler does '
                                    'not support --target=<triple>')

        # Add includes for support headers used in the tests.
        support_path = os.path.join(self.libcxx_src_root, 'test/support')
        self.cxx.compile_flags += ['-I' + support_path]

        # On GCC, the libc++ headers cause errors due to throw() decorators
        # on operator new clashing with those from the test suite, so we
        # don't enable warnings in system headers on GCC.
        if self.cxx.type != 'gcc':
            self.cxx.compile_flags += ['-D_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER']

        # Add includes for the PSTL headers
        pstl_src_root = self.get_lit_conf('pstl_src_root')
        pstl_obj_root = self.get_lit_conf('pstl_obj_root')
        if pstl_src_root is not None and pstl_obj_root is not None:
            self.cxx.compile_flags += ['-I' + os.path.join(pstl_src_root, 'include')]
            self.cxx.compile_flags += ['-I' + os.path.join(pstl_obj_root, 'generated_headers')]
            self.cxx.compile_flags += ['-I' + os.path.join(pstl_src_root, 'test')]
            self.config.available_features.add('parallel-algorithms')

    def configure_compile_flags_header_includes(self):
        support_path = os.path.join(self.libcxx_src_root, 'test', 'support')
        self.configure_config_site_header()
        if self.cxx_stdlib_under_test != 'libstdc++' and \
           not self.target_info.is_windows():
            self.cxx.compile_flags += [
                '-include', os.path.join(support_path, 'nasty_macros.h')]
        if self.cxx_stdlib_under_test == 'msvc':
            self.cxx.compile_flags += [
                '-include', os.path.join(support_path,
                                         'msvc_stdlib_force_include.h')]
            pass
        if self.target_info.is_windows() and self.debug_build and \
                self.cxx_stdlib_under_test != 'msvc':
            self.cxx.compile_flags += [
                '-include', os.path.join(support_path,
                                         'set_windows_crt_report_mode.h')
            ]
        cxx_headers = self.get_lit_conf('cxx_headers')
        if cxx_headers == '' or (cxx_headers is None
                                 and self.cxx_stdlib_under_test != 'libc++'):
            self.lit_config.note('using the system cxx headers')
            return
        self.cxx.compile_flags += ['-nostdinc++']
        if cxx_headers is None:
            cxx_headers = os.path.join(self.libcxx_src_root, 'include')
        if not os.path.isdir(cxx_headers):
            self.lit_config.fatal("cxx_headers='%s' is not a directory."
                                  % cxx_headers)
        self.cxx.compile_flags += ['-I' + cxx_headers]
        if self.libcxx_obj_root is not None:
            cxxabi_headers = os.path.join(self.libcxx_obj_root, 'include',
                                          'c++build')
            if os.path.isdir(cxxabi_headers):
                self.cxx.compile_flags += ['-I' + cxxabi_headers]

    def configure_config_site_header(self):
        # Check for a possible __config_site in the build directory. We
        # use this if it exists.
        if self.libcxx_obj_root is None:
            return
        config_site_header = os.path.join(self.libcxx_obj_root, '__config_site')
        if not os.path.isfile(config_site_header):
            return
        self.cxx.compile_flags += ['-include', config_site_header]

    def configure_link_flags(self):
        # Configure library path
        self.configure_link_flags_cxx_library_path()
        self.configure_link_flags_abi_library_path()

        # Configure libraries
        if self.cxx_stdlib_under_test == 'libc++':
            self.cxx.link_flags += ['-nodefaultlibs']
            # FIXME: Handle MSVCRT as part of the ABI library handling.
            if self.target_info.is_windows():
                self.cxx.link_flags += ['-nostdlib']
            self.configure_link_flags_cxx_library()
            self.configure_link_flags_abi_library()
            self.configure_extra_library_flags()
        elif self.cxx_stdlib_under_test == 'libstdc++':
            self.cxx.link_flags += ['-lstdc++fs', '-lm', '-pthread']
        elif self.cxx_stdlib_under_test == 'msvc':
            # FIXME: Correctly setup debug/release flags here.
            pass
        elif self.cxx_stdlib_under_test == 'cxx_default':
            self.cxx.link_flags += ['-pthread']
        else:
            self.lit_config.fatal('invalid stdlib under test')

        link_flags_str = self.get_lit_conf('link_flags', '')
        self.cxx.link_flags += shlex.split(link_flags_str)

    def configure_link_flags_cxx_library_path(self):
        if self.cxx_library_root:
            self.cxx.link_flags += ['-L' + self.cxx_library_root]
            if self.target_info.is_windows() and self.link_shared:
                self.add_path(self.cxx.compile_env, self.cxx_library_root)
        if self.cxx_runtime_root:
            if not self.target_info.is_windows():
                self.cxx.link_flags += ['-Wl,-rpath,' +
                                        self.cxx_runtime_root]
            elif self.target_info.is_windows() and self.link_shared:
                self.add_path(self.exec_env, self.cxx_runtime_root)
        additional_flags = self.get_lit_conf('test_linker_flags')
        if additional_flags:
            self.cxx.link_flags += shlex.split(additional_flags)

    def configure_link_flags_abi_library_path(self):
        # Configure ABI library paths.
        if self.abi_library_root:
            self.cxx.link_flags += ['-L' + self.abi_library_root]
            if not self.target_info.is_windows():
                self.cxx.link_flags += ['-Wl,-rpath,' + self.abi_library_root]
            else:
                self.add_path(self.exec_env, self.abi_library_root)

    def configure_link_flags_cxx_library(self):
        if self.link_shared:
            self.cxx.link_flags += ['-lc++']
        else:
            if self.cxx_library_root:
                libname = self.make_static_lib_name('c++')
                abs_path = os.path.join(self.cxx_library_root, libname)
                assert os.path.exists(abs_path) and \
                       "static libc++ library does not exist"
                self.cxx.link_flags += [abs_path]
            else:
                self.cxx.link_flags += ['-lc++']

    def configure_link_flags_abi_library(self):
        cxx_abi = self.get_lit_conf('cxx_abi', 'libcxxabi')
        if cxx_abi == 'libstdc++':
            self.cxx.link_flags += ['-lstdc++']
        elif cxx_abi == 'libsupc++':
            self.cxx.link_flags += ['-lsupc++']
        elif cxx_abi == 'libcxxabi':
            # If the C++ library requires explicitly linking to libc++abi, or
            # if we're testing libc++abi itself (the test configs are shared),
            # then link it.
            testing_libcxxabi = self.get_lit_conf('name', '') == 'libc++abi'
            if self.target_info.allow_cxxabi_link() or testing_libcxxabi:
                libcxxabi_shared = self.get_lit_bool('libcxxabi_shared', default=True)
                if libcxxabi_shared:
                    self.cxx.link_flags += ['-lc++abi']
                else:
                    if self.abi_library_root:
                        libname = self.make_static_lib_name('c++abi')
                        abs_path = os.path.join(self.abi_library_root, libname)
                        self.cxx.link_libcxxabi_flag = abs_path
                        self.cxx.link_flags += [abs_path]
                    else:
                        self.cxx.link_flags += ['-lc++abi']
        elif cxx_abi == 'libcxxrt':
            self.cxx.link_flags += ['-lcxxrt']
        elif cxx_abi == 'vcruntime':
            debug_suffix = 'd' if self.debug_build else ''
            self.cxx.link_flags += ['-l%s%s' % (lib, debug_suffix) for lib in
                                    ['vcruntime', 'ucrt', 'msvcrt']]
        elif cxx_abi == 'none' or cxx_abi == 'default':
            if self.target_info.is_windows():
                debug_suffix = 'd' if self.debug_build else ''
                self.cxx.link_flags += ['-lmsvcrt%s' % debug_suffix]
        else:
            self.lit_config.fatal(
                'C++ ABI setting %s unsupported for tests' % cxx_abi)

    def configure_extra_library_flags(self):
        if self.get_lit_bool('cxx_ext_threads', default=False):
            self.cxx.link_flags += ['-lc++external_threads']
        self.target_info.add_cxx_link_flags(self.cxx.link_flags)

    def configure_debug_mode(self):
        debug_level = self.get_lit_conf('debug_level', None)
        if not debug_level:
            return
        if debug_level not in ['0', '1']:
            self.lit_config.fatal('Invalid value for debug_level "%s".'
                                  % debug_level)
        self.cxx.compile_flags += ['-D_LIBCPP_DEBUG=%s' % debug_level]

    def configure_sanitizer(self):
        san = self.get_lit_conf('use_sanitizer', '').strip()
        if san:
            # Search for llvm-symbolizer along the compiler path first
            # and then along the PATH env variable.
            symbolizer_search_paths = os.environ.get('PATH', '')
            cxx_path = libcxx.util.which(self.cxx.path)
            if cxx_path is not None:
                symbolizer_search_paths = (
                    os.path.dirname(cxx_path) +
                    os.pathsep + symbolizer_search_paths)
            llvm_symbolizer = libcxx.util.which('llvm-symbolizer',
                                                symbolizer_search_paths)

            def add_ubsan():
                self.cxx.flags += ['-fsanitize=undefined',
                                   '-fno-sanitize=float-divide-by-zero',
                                   '-fno-sanitize-recover=all']
                self.exec_env['UBSAN_OPTIONS'] = 'print_stacktrace=1'
                self.config.available_features.add('ubsan')

            # Setup the sanitizer compile flags
            self.cxx.flags += ['-g', '-fno-omit-frame-pointer']
            if san == 'Address' or san == 'Address;Undefined' or san == 'Undefined;Address':
                self.cxx.flags += ['-fsanitize=address']
                if llvm_symbolizer is not None:
                    self.exec_env['ASAN_SYMBOLIZER_PATH'] = llvm_symbolizer
                # FIXME: Turn ODR violation back on after PR28391 is resolved
                # https://bugs.llvm.org/show_bug.cgi?id=28391
                self.exec_env['ASAN_OPTIONS'] = 'detect_odr_violation=0'
                self.config.available_features.add('asan')
                self.config.available_features.add('sanitizer-new-delete')
                self.cxx.compile_flags += ['-O1']
                if san == 'Address;Undefined' or san == 'Undefined;Address':
                    add_ubsan()
            elif san == 'Memory' or san == 'MemoryWithOrigins':
                self.cxx.flags += ['-fsanitize=memory']
                if san == 'MemoryWithOrigins':
                    self.cxx.compile_flags += [
                        '-fsanitize-memory-track-origins']
                if llvm_symbolizer is not None:
                    self.exec_env['MSAN_SYMBOLIZER_PATH'] = llvm_symbolizer
                self.config.available_features.add('msan')
                self.config.available_features.add('sanitizer-new-delete')
                self.cxx.compile_flags += ['-O1']
            elif san == 'Undefined':
                add_ubsan()
                self.cxx.compile_flags += ['-O2']
            elif san == 'Thread':
                self.cxx.flags += ['-fsanitize=thread']
                self.config.available_features.add('tsan')
                self.config.available_features.add('sanitizer-new-delete')
            elif san == 'DataFlow':
                self.cxx.flags += ['-fsanitize=dataflow']
            else:
                self.lit_config.fatal('unsupported value for '
                                      'use_sanitizer: {0}'.format(san))
            san_lib = self.get_lit_conf('sanitizer_library')
            if san_lib:
                self.cxx.link_flags += [
                    san_lib, '-Wl,-rpath,%s' % os.path.dirname(san_lib)]

    def configure_coverage(self):
        self.generate_coverage = self.get_lit_bool('generate_coverage', False)
        if self.generate_coverage:
            self.cxx.flags += ['-g', '--coverage']
            self.cxx.compile_flags += ['-O0']

    def configure_modules(self):
        modules_flags = ['-fmodules', '-Xclang', '-fmodules-local-submodule-visibility']
        supports_modules = self.cxx.hasCompileFlag(modules_flags)
        enable_modules = self.get_lit_bool('enable_modules', default=False,
                                                             env_var='LIBCXX_ENABLE_MODULES')
        if enable_modules and not supports_modules:
            self.lit_config.fatal(
                '-fmodules is enabled but not supported by the compiler')
        if not supports_modules:
            return
        module_cache = os.path.join(self.config.test_exec_root,
                                   'modules.cache')
        module_cache = os.path.realpath(module_cache)
        if os.path.isdir(module_cache):
            shutil.rmtree(module_cache)
        os.makedirs(module_cache)
        self.cxx.modules_flags += modules_flags + \
            ['-fmodules-cache-path=' + module_cache]
        if enable_modules:
            self.config.available_features.add('-fmodules')
            self.cxx.useModules()

    def configure_substitutions(self):
        sub = self.config.substitutions
        sub.append(('%{cxx}', pipes.quote(self.cxx.path)))
        flags = self.cxx.flags + (self.cxx.modules_flags if self.cxx.use_modules else [])
        compile_flags = self.cxx.compile_flags + (self.cxx.warning_flags if self.cxx.use_warnings else [])
        sub.append(('%{flags}',         ' '.join(map(pipes.quote, flags))))
        sub.append(('%{compile_flags}', ' '.join(map(pipes.quote, compile_flags))))
        sub.append(('%{link_flags}',    ' '.join(map(pipes.quote, self.cxx.link_flags))))
        sub.append(('%{link_libcxxabi}', pipes.quote(self.cxx.link_libcxxabi_flag)))
        codesign_ident = self.get_lit_conf('llvm_codesign_identity', '')
        env_vars = ' '.join('%s=%s' % (k, pipes.quote(v)) for (k, v) in self.exec_env.items())
        exec_args = [
            '--execdir %T',
            '--codesign_identity "{}"'.format(codesign_ident),
            '--env {}'.format(env_vars)
        ]
        sub.append(('%{exec}', '{} {} -- '.format(self.executor, ' '.join(exec_args))))

    def configure_triple(self):
        # Get or infer the target triple.
        target_triple = self.get_lit_conf('target_triple')

        # If no target triple was given, try to infer it from the compiler
        # under test.
        if not target_triple:
            self.lit_config.note('Trying to infer the target_triple because none was specified')

            target_triple = self.cxx.getTriple()
            # Drop sub-major version components from the triple, because the
            # current XFAIL handling expects exact matches for feature checks.
            # Example: x86_64-apple-darwin14.0.0 -> x86_64-apple-darwin14
            # The 5th group handles triples greater than 3 parts
            # (ex x86_64-pc-linux-gnu).
            target_triple = re.sub(r'([^-]+)-([^-]+)-([^.]+)([^-]*)(.*)',
                                   r'\1-\2-\3\5', target_triple)
            # linux-gnu is needed in the triple to properly identify linuxes
            # that use GLIBC. Handle redhat and opensuse triples as special
            # cases and append the missing `-gnu` portion.
            if (target_triple.endswith('redhat-linux') or
                target_triple.endswith('suse-linux')):
                target_triple += '-gnu'

        # Save the triple
        self.lit_config.note("Setting target_triple to {}".format(target_triple))
        self.config.target_triple = target_triple

    def configure_env(self):
        self.config.environment = dict(os.environ)

    def add_path(self, dest_env, new_path):
        self.target_info.add_path(dest_env, new_path)
