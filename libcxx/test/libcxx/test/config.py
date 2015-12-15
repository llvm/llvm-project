import importlib
import locale
import os
import platform
import pkgutil
import re
import shlex
import sys

import lit.Test  # pylint: disable=import-error,no-name-in-module
import lit.util  # pylint: disable=import-error,no-name-in-module

from libcxx.test.format import LibcxxTestFormat
from libcxx.compiler import CXXCompiler
from libcxx.test.executor import *
from libcxx.test.tracing import *

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

def getSysrootFlagsOnDarwin(config, lit_config):
    # On Darwin, support relocatable SDKs by providing Clang with a
    # default system root path.
    if 'darwin' in config.target_triple:
        try:
            out = lit.util.capture(['xcrun', '--show-sdk-path']).strip()
            res = 0
        except OSError:
            res = -1
        if res == 0 and out:
            sdk_path = out
            lit_config.note('using SDKROOT: %r' % sdk_path)
            return ["-isysroot", sdk_path]
    return []


class Configuration(object):
    # pylint: disable=redefined-outer-name
    def __init__(self, lit_config, config):
        self.lit_config = lit_config
        self.config = config
        self.cxx = None
        self.project_obj_root = None
        self.libcxx_src_root = None
        self.libcxx_obj_root = None
        self.cxx_library_root = None
        self.abi_library_root = None
        self.env = {}
        self.use_target = False
        self.use_system_cxx_lib = False
        self.use_clang_verify = False
        self.long_tests = None
        self.execute_external = False

    def get_lit_conf(self, name, default=None):
        val = self.lit_config.params.get(name, None)
        if val is None:
            val = getattr(self.config, name, None)
            if val is None:
                val = default
        return val

    def get_lit_bool(self, name, default=None):
        conf = self.get_lit_conf(name)
        if conf is None:
            return default
        if conf.lower() in ('1', 'true'):
            return True
        if conf.lower() in ('', '0', 'false'):
            return False
        self.lit_config.fatal(
            "parameter '{}' should be true or false".format(name))

    def configure(self):
        self.configure_executor()
        self.configure_target_info()
        self.configure_cxx()
        self.configure_triple()
        self.configure_src_root()
        self.configure_obj_root()
        self.configure_cxx_library_root()
        self.configure_use_system_cxx_lib()
        self.configure_use_clang_verify()
        self.configure_execute_external()
        self.configure_ccache()
        self.configure_compile_flags()
        self.configure_link_flags()
        self.configure_env()
        self.configure_color_diagnostics()
        self.configure_debug_mode()
        self.configure_warnings()
        self.configure_sanitizer()
        self.configure_coverage()
        self.configure_substitutions()
        self.configure_features()

    def print_config_info(self):
        # Print the final compile and link flags.
        self.lit_config.note('Using compiler: %s' % self.cxx.path)
        self.lit_config.note('Using flags: %s' % self.cxx.flags)
        self.lit_config.note('Using compile flags: %s'
                             % self.cxx.compile_flags)
        self.lit_config.note('Using link flags: %s' % self.cxx.link_flags)
        # Print as list to prevent "set([...])" from being printed.
        self.lit_config.note('Using available_features: %s' %
                             list(self.config.available_features))
        self.lit_config.note('Using environment: %r' % self.env)

    def get_test_format(self):
        return LibcxxTestFormat(
            self.cxx,
            self.use_clang_verify,
            self.execute_external,
            self.executor,
            exec_env=self.env)

    def configure_executor(self):
        exec_str = self.get_lit_conf('executor', "None")
        te = eval(exec_str)
        if te:
            self.lit_config.note("Using executor: %r" % exec_str)
            if self.lit_config.useValgrind:
                # We have no way of knowing where in the chain the
                # ValgrindExecutor is supposed to go. It is likely
                # that the user wants it at the end, but we have no
                # way of getting at that easily.
                selt.lit_config.fatal("Cannot infer how to create a Valgrind "
                                      " executor.")
        else:
            te = LocalExecutor()
            if self.lit_config.useValgrind:
                te = ValgrindExecutor(self.lit_config.valgrindArgs, te)
        self.executor = te

    def configure_target_info(self):
        default = "libcxx.test.target_info.LocalTI"
        info_str = self.get_lit_conf('target_info', default)
        mod_path, _, info = info_str.rpartition('.')
        mod = importlib.import_module(mod_path)
        self.target_info = getattr(mod, info)()
        if info_str != default:
            self.lit_config.note("inferred target_info as: %r" % info_str)

    def configure_cxx(self):
        # Gather various compiler parameters.
        cxx = self.get_lit_conf('cxx_under_test')

        # If no specific cxx_under_test was given, attempt to infer it as
        # clang++.
        if cxx is None:
            clangxx = lit.util.which('clang++',
                                     self.config.environment['PATH'])
            if clangxx:
                cxx = clangxx
                self.lit_config.note(
                    "inferred cxx_under_test as: %r" % cxx)
        if not cxx:
            self.lit_config.fatal('must specify user parameter cxx_under_test '
                                  '(e.g., --param=cxx_under_test=clang++)')
        self.cxx = CXXCompiler(cxx)
        cxx_type = self.cxx.type
        if cxx_type is not None:
            assert self.cxx.version is not None
            maj_v, min_v, _ = self.cxx.version
            self.config.available_features.add(cxx_type)
            self.config.available_features.add('%s-%s.%s' % (
                cxx_type, maj_v, min_v))

    def configure_src_root(self):
        self.libcxx_src_root = self.get_lit_conf(
            'libcxx_src_root', os.path.dirname(self.config.test_source_root))

    def configure_obj_root(self):
        self.project_obj_root = self.get_lit_conf('project_obj_root')
        self.libcxx_obj_root = self.get_lit_conf('libcxx_obj_root')
        if not self.libcxx_obj_root and self.project_obj_root is not None:
            possible_root = os.path.join(self.project_obj_root, 'projects', 'libcxx')
            if os.path.isdir(possible_root):
                self.libcxx_obj_root = possible_root
            else:
                self.libcxx_obj_root = self.project_obj_root

    def configure_cxx_library_root(self):
        self.cxx_library_root = self.get_lit_conf('cxx_library_root',
                                                  self.libcxx_obj_root)

    def configure_use_system_cxx_lib(self):
        # This test suite supports testing against either the system library or
        # the locally built one; the former mode is useful for testing ABI
        # compatibility between the current headers and a shipping dynamic
        # library.
        self.use_system_cxx_lib = self.get_lit_bool('use_system_cxx_lib')
        if self.use_system_cxx_lib is None:
            # Default to testing against the locally built libc++ library.
            self.use_system_cxx_lib = False
            self.lit_config.note(
                "inferred use_system_cxx_lib as: %r" % self.use_system_cxx_lib)

    def configure_use_clang_verify(self):
        '''If set, run clang with -verify on failing tests.'''
        self.use_clang_verify = self.get_lit_bool('use_clang_verify')
        if self.use_clang_verify is None:
            # NOTE: We do not test for the -verify flag directly because
            #   -verify will always exit with non-zero on an empty file.
            self.use_clang_verify = self.cxx.hasCompileFlag(
                ['-Xclang', '-verify-ignore-unexpected'])
            self.lit_config.note(
                "inferred use_clang_verify as: %r" % self.use_clang_verify)

    def configure_execute_external(self):
        # Choose between lit's internal shell pipeline runner and a real shell.
        # If LIT_USE_INTERNAL_SHELL is in the environment, we use that as the
        # default value. Otherwise we default to internal on Windows and
        # external elsewhere, as bash on Windows is usually very slow.
        use_lit_shell_default = os.environ.get('LIT_USE_INTERNAL_SHELL')
        if use_lit_shell_default is not None:
            use_lit_shell_default = use_lit_shell_default != '0'
        else:
            use_lit_shell_default = sys.platform == 'win32'
        # Check for the command line parameter using the default value if it is
        # not present.
        use_lit_shell = self.get_lit_bool('use_lit_shell',
                                          use_lit_shell_default)
        self.execute_external = not use_lit_shell

    def configure_ccache(self):
        use_ccache_default = os.environ.get('LIBCXX_USE_CCACHE') is not None
        use_ccache = self.get_lit_bool('use_ccache', use_ccache_default)
        if use_ccache:
            self.cxx.use_ccache = True
            self.lit_config.note('enabling ccache')

    def configure_features(self):
        additional_features = self.get_lit_conf('additional_features')
        if additional_features:
            for f in additional_features.split(','):
                self.config.available_features.add(f.strip())

        # Figure out which of the required locales we support
        locales = {
            'Darwin': {
                'en_US.UTF-8': 'en_US.UTF-8',
                'cs_CZ.ISO8859-2': 'cs_CZ.ISO8859-2',
                'fr_FR.UTF-8': 'fr_FR.UTF-8',
                'fr_CA.ISO8859-1': 'fr_CA.ISO8859-1',
                'ru_RU.UTF-8': 'ru_RU.UTF-8',
                'zh_CN.UTF-8': 'zh_CN.UTF-8',
            },
            'FreeBSD': {
                'en_US.UTF-8': 'en_US.UTF-8',
                'cs_CZ.ISO8859-2': 'cs_CZ.ISO8859-2',
                'fr_FR.UTF-8': 'fr_FR.UTF-8',
                'fr_CA.ISO8859-1': 'fr_CA.ISO8859-1',
                'ru_RU.UTF-8': 'ru_RU.UTF-8',
                'zh_CN.UTF-8': 'zh_CN.UTF-8',
            },
            'Linux': {
                'en_US.UTF-8': 'en_US.UTF-8',
                'cs_CZ.ISO8859-2': 'cs_CZ.ISO-8859-2',
                'fr_FR.UTF-8': 'fr_FR.UTF-8',
                'fr_CA.ISO8859-1': 'fr_CA.ISO-8859-1',
                'ru_RU.UTF-8': 'ru_RU.UTF-8',
                'zh_CN.UTF-8': 'zh_CN.UTF-8',
            },
            'Windows': {
                'en_US.UTF-8': 'English_United States.1252',
                'cs_CZ.ISO8859-2': 'Czech_Czech Republic.1250',
                'fr_FR.UTF-8': 'French_France.1252',
                'fr_CA.ISO8859-1': 'French_Canada.1252',
                'ru_RU.UTF-8': 'Russian_Russia.1251',
                'zh_CN.UTF-8': 'Chinese_China.936',
            },
        }

        target_system = self.target_info.system()
        target_platform = self.target_info.platform()

        if target_system in locales:
            default_locale = locale.setlocale(locale.LC_ALL)
            for feature, loc in locales[target_system].items():
                try:
                    locale.setlocale(locale.LC_ALL, loc)
                    self.config.available_features.add(
                        'locale.{0}'.format(feature))
                except locale.Error:
                    self.lit_config.warning('The locale {0} is not supported by '
                                            'your platform. Some tests will be '
                                            'unsupported.'.format(loc))
            locale.setlocale(locale.LC_ALL, default_locale)
        else:
            # Warn that the user doesn't get any free XFAILs for locale issues
            self.lit_config.warning("No locales entry for target_system: %s" %
                                    target_system)

        # Write an "available feature" that combines the triple when
        # use_system_cxx_lib is enabled. This is so that we can easily write
        # XFAIL markers for tests that are known to fail with versions of
        # libc++ as were shipped with a particular triple.
        if self.use_system_cxx_lib:
            self.config.available_features.add(
                'with_system_cxx_lib=%s' % self.config.target_triple)

        # Insert the platform name into the available features as a lower case.
        self.config.available_features.add(target_platform)

        # Some linux distributions have different locale data than others.
        # Insert the distributions name and name-version into the available
        # features to allow tests to XFAIL on them.
        if target_platform == 'linux':
            name = self.target_info.platform_name()
            ver = self.target_info.platform_ver()
            if name:
                self.config.available_features.add(name)
            if name and ver:
                self.config.available_features.add('%s-%s' % (name, ver))

        # Simulator testing can take a really long time for some of these tests
        # so add a feature check so we can REQUIRES: long_tests in them
        self.long_tests = self.get_lit_bool('long_tests')
        if self.long_tests is None:
            # Default to running long tests.
            self.long_tests = True
            self.lit_config.note(
                "inferred long_tests as: %r" % self.long_tests)

        if self.long_tests:
            self.config.available_features.add('long_tests')

        # Run a compile test for the -fsized-deallocation flag. This is needed
        # in test/std/language.support/support.dynamic/new.delete
        if self.cxx.hasCompileFlag('-fsized-deallocation'):
            self.config.available_features.add('fsized-deallocation')

    def configure_compile_flags(self):
        no_default_flags = self.get_lit_bool('no_default_flags', False)
        if not no_default_flags:
            self.configure_default_compile_flags()
        # Configure extra flags
        compile_flags_str = self.get_lit_conf('compile_flags', '')
        self.cxx.compile_flags += shlex.split(compile_flags_str)
        sysroot_flags = getSysrootFlagsOnDarwin(self.config, self.lit_config)
        self.cxx.compile_flags.extend(sysroot_flags)

    def configure_default_compile_flags(self):
        # Try and get the std version from the command line. Fall back to
        # default given in lit.site.cfg is not present. If default is not
        # present then force c++11.
        std = self.get_lit_conf('std')
        if not std:
            # Choose the newest possible language dialect if none is given.
            possible_stds = ['c++1z', 'c++14', 'c++11', 'c++03']
            for s in possible_stds:
                if self.cxx.hasCompileFlag('-std=%s' % s):
                    std = s
                    self.lit_config.note(
                        'inferred language dialect as: %s' % std)
                    break
            if not std:
                self.lit_config.fatal(
                    'Failed to infer a supported language dialect from one of %r'
                    % possible_stds)
        self.cxx.compile_flags += ['-std={0}'.format(std)]
        self.config.available_features.add(std)
        # Configure include paths
        self.cxx.compile_flags += ['-nostdinc++']
        self.configure_compile_flags_header_includes()
        if self.target_info.platform() == 'linux':
            self.cxx.compile_flags += ['-D__STDC_FORMAT_MACROS',
                                       '-D__STDC_LIMIT_MACROS',
                                       '-D__STDC_CONSTANT_MACROS']
        # Configure feature flags.
        self.configure_compile_flags_exceptions()
        self.configure_compile_flags_rtti()
        self.configure_compile_flags_abi_version()
        enable_32bit = self.get_lit_bool('enable_32bit', False)
        if enable_32bit:
            self.cxx.flags += ['-m32']
        # Use verbose output for better errors
        self.cxx.flags += ['-v']
        sysroot = self.get_lit_conf('sysroot')
        if sysroot:
            self.cxx.flags += ['--sysroot', sysroot]
        gcc_toolchain = self.get_lit_conf('gcc_toolchain')
        if gcc_toolchain:
            self.cxx.flags += ['-gcc-toolchain', gcc_toolchain]
        if self.use_target:
            self.cxx.flags += ['-target', self.config.target_triple]

    def configure_compile_flags_header_includes(self):
        support_path = os.path.join(self.libcxx_src_root, 'test/support')
        self.cxx.compile_flags += ['-I' + support_path]
        self.cxx.compile_flags += ['-include', os.path.join(support_path, 'nasty_macros.hpp')]
        self.configure_config_site_header()
        libcxx_headers = self.get_lit_conf(
            'libcxx_headers', os.path.join(self.libcxx_src_root, 'include'))
        if not os.path.isdir(libcxx_headers):
            self.lit_config.fatal("libcxx_headers='%s' is not a directory."
                                  % libcxx_headers)
        self.cxx.compile_flags += ['-I' + libcxx_headers]

    def configure_config_site_header(self):
        # Check for a possible __config_site in the build directory. We
        # use this if it exists.
        if self.libcxx_obj_root is None:
            return
        config_site_header = os.path.join(self.libcxx_obj_root, '__config_site')
        if not os.path.isfile(config_site_header):
            return
        contained_macros = self.parse_config_site_and_add_features(
            config_site_header)
        self.lit_config.note('Using __config_site header %s with macros: %r'
            % (config_site_header, contained_macros))
        # FIXME: This must come after the call to
        # 'parse_config_site_and_add_features(...)' in order for it to work.
        self.cxx.compile_flags += ['-include', config_site_header]

    def parse_config_site_and_add_features(self, header):
        """ parse_config_site_and_add_features - Deduce and add the test
            features that that are implied by the #define's in the __config_site
            header. Return a dictionary containing the macros found in the
            '__config_site' header.
        """
        # Parse the macro contents of __config_site by dumping the macros
        # using 'c++ -dM -E' and filtering the predefines.
        predefines = self.cxx.dumpMacros()
        macros = self.cxx.dumpMacros(header)
        feature_macros_keys = set(macros.keys()) - set(predefines.keys())
        feature_macros = {}
        for k in feature_macros_keys:
            feature_macros[k] = macros[k]
        # We expect the header guard to be one of the definitions
        assert '_LIBCPP_CONFIG_SITE' in feature_macros
        del feature_macros['_LIBCPP_CONFIG_SITE']
        # The __config_site header should be non-empty. Otherwise it should
        # have never been emitted by CMake.
        assert len(feature_macros) > 0
        # Transform each macro name into the feature name used in the tests.
        # Ex. _LIBCPP_HAS_NO_THREADS -> libcpp-has-no-threads
        for m in feature_macros:
            if m == '_LIBCPP_ABI_VERSION':
                self.config.available_features.add('libcpp-abi-version-v%s'
                    % feature_macros[m])
                continue
            assert m.startswith('_LIBCPP_HAS_') or m == '_LIBCPP_ABI_UNSTABLE'
            m = m.lower()[1:].replace('_', '-')
            self.config.available_features.add(m)
        return feature_macros



    def configure_compile_flags_exceptions(self):
        enable_exceptions = self.get_lit_bool('enable_exceptions', True)
        if not enable_exceptions:
            self.config.available_features.add('libcpp-no-exceptions')
            self.cxx.compile_flags += ['-fno-exceptions']

    def configure_compile_flags_rtti(self):
        enable_rtti = self.get_lit_bool('enable_rtti', True)
        if not enable_rtti:
            self.config.available_features.add('libcpp-no-rtti')
            self.cxx.compile_flags += ['-fno-rtti', '-D_LIBCPP_NO_RTTI']

    def configure_compile_flags_abi_version(self):
        abi_version = self.get_lit_conf('abi_version', '').strip()
        abi_unstable = self.get_lit_bool('abi_unstable')
        # Only add the ABI version when it is non-default.
        # FIXME(EricWF): Get the ABI version from the "__config_site".
        if abi_version and abi_version != '1':
          self.cxx.compile_flags += ['-D_LIBCPP_ABI_VERSION=' + abi_version]
        if abi_unstable:
          self.config.available_features.add('libcpp-abi-unstable')
          self.cxx.compile_flags += ['-D_LIBCPP_ABI_UNSTABLE']

    def configure_link_flags(self):
        no_default_flags = self.get_lit_bool('no_default_flags', False)
        if not no_default_flags:
            self.cxx.link_flags += ['-nodefaultlibs']

            # Configure library path
            self.configure_link_flags_cxx_library_path()
            self.configure_link_flags_abi_library_path()

            # Configure libraries
            self.configure_link_flags_cxx_library()
            self.configure_link_flags_abi_library()
            self.configure_extra_library_flags()

        link_flags_str = self.get_lit_conf('link_flags', '')
        self.cxx.link_flags += shlex.split(link_flags_str)

    def configure_link_flags_cxx_library_path(self):
        libcxx_library = self.get_lit_conf('libcxx_library')
        # Configure libc++ library paths.
        if libcxx_library is not None:
            # Check that the given value for libcxx_library is valid.
            if not os.path.isfile(libcxx_library):
                self.lit_config.fatal(
                    "libcxx_library='%s' is not a valid file." %
                    libcxx_library)
            if self.use_system_cxx_lib:
                self.lit_config.fatal(
                    "Conflicting options: 'libcxx_library' cannot be used "
                    "with 'use_system_cxx_lib=true'")
            self.cxx.link_flags += ['-Wl,-rpath,' +
                                    os.path.dirname(libcxx_library)]
        elif not self.use_system_cxx_lib and self.cxx_library_root:
            self.cxx.link_flags += ['-L' + self.cxx_library_root,
                                    '-Wl,-rpath,' + self.cxx_library_root]

    def configure_link_flags_abi_library_path(self):
        # Configure ABI library paths.
        self.abi_library_root = self.get_lit_conf('abi_library_path')
        if self.abi_library_root:
            self.cxx.link_flags += ['-L' + self.abi_library_root,
                                    '-Wl,-rpath,' + self.abi_library_root]

    def configure_link_flags_cxx_library(self):
        libcxx_library = self.get_lit_conf('libcxx_library')
        if libcxx_library:
            self.cxx.link_flags += [libcxx_library]
        else:
            self.cxx.link_flags += ['-lc++']

    def configure_link_flags_abi_library(self):
        cxx_abi = self.get_lit_conf('cxx_abi', 'libcxxabi')
        if cxx_abi == 'libstdc++':
            self.cxx.link_flags += ['-lstdc++']
        elif cxx_abi == 'libsupc++':
            self.cxx.link_flags += ['-lsupc++']
        elif cxx_abi == 'libcxxabi':
            # Don't link libc++abi explicitly on OS X because the symbols
            # should be available in libc++ directly.
            if self.target_info.platform() != 'darwin':
                self.cxx.link_flags += ['-lc++abi']
        elif cxx_abi == 'libcxxrt':
            self.cxx.link_flags += ['-lcxxrt']
        elif cxx_abi == 'none':
            pass
        else:
            self.lit_config.fatal(
                'C++ ABI setting %s unsupported for tests' % cxx_abi)

    def configure_extra_library_flags(self):
        enable_threads = ('libcpp-has-no-threads' not in
                          self.config.available_features)
        llvm_unwinder = self.get_lit_bool('llvm_unwinder', False)
        target_platform = self.target_info.platform()
        if target_platform == 'darwin':
            self.cxx.link_flags += ['-lSystem']
        elif target_platform == 'linux':
            self.cxx.link_flags += ['-lm']
            if not llvm_unwinder:
                self.cxx.link_flags += ['-lgcc_s', '-lgcc']
            if enable_threads:
                self.cxx.link_flags += ['-lpthread']
            self.cxx.link_flags += ['-lc']
            if llvm_unwinder:
                self.cxx.link_flags += ['-lunwind', '-ldl']
            else:
                self.cxx.link_flags += ['-lgcc_s', '-lgcc']
        elif target_platform.startswith('freebsd'):
            self.cxx.link_flags += ['-lc', '-lm', '-lpthread', '-lgcc_s', '-lcxxrt']
        else:
            self.lit_config.fatal("unrecognized system: %r" % target_platform)

    def configure_color_diagnostics(self):
        use_color = self.get_lit_conf('color_diagnostics')
        if use_color is None:
            use_color = os.environ.get('LIBCXX_COLOR_DIAGNOSTICS')
        if use_color is None:
            return
        if use_color != '':
            self.lit_config.fatal('Invalid value for color_diagnostics "%s".'
                                  % use_color)
        color_flag = '-fdiagnostics-color=always'
        # Check if the compiler supports the color diagnostics flag. Issue a
        # warning if it does not since color diagnostics have been requested.
        if not self.cxx.hasCompileFlag(color_flag):
            self.lit_config.warning(
                'color diagnostics have been requested but are not supported '
                'by the compiler')
        else:
            self.cxx.flags += [color_flag]

    def configure_debug_mode(self):
        debug_level = self.get_lit_conf('debug_level', None)
        if not debug_level:
            return
        if debug_level not in ['0', '1']:
            self.lit_config.fatal('Invalid value for debug_level "%s".'
                                  % debug_level)
        self.cxx.compile_flags += ['-D_LIBCPP_DEBUG=%s' % debug_level]

    def configure_warnings(self):
        enable_warnings = self.get_lit_bool('enable_warnings', False)
        if enable_warnings:
            self.cxx.compile_flags += [
                '-D_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER',
                '-Wall', '-Werror'
            ]
            self.cxx.addWarningFlagIfSupported('-Wno-attributes')
            self.cxx.addWarningFlagIfSupported('-Wno-pessimizing-move')
            self.cxx.addWarningFlagIfSupported('-Wno-c++11-extensions')
            self.cxx.addWarningFlagIfSupported('-Wno-user-defined-literals')
            # TODO(EricWF) Remove the unused warnings once the test suite
            # compiles clean with them.
            self.cxx.addWarningFlagIfSupported('-Wno-unused-local-typedef')
            self.cxx.addWarningFlagIfSupported('-Wno-unused-variable')
            std = self.get_lit_conf('std', None)
            if std in ['c++98', 'c++03']:
                # The '#define static_assert' provided by libc++ in C++03 mode
                # causes an unused local typedef whenever it is used.
                self.cxx.addWarningFlagIfSupported('-Wno-unused-local-typedef')

    def configure_sanitizer(self):
        san = self.get_lit_conf('use_sanitizer', '').strip()
        if san:
            # Search for llvm-symbolizer along the compiler path first
            # and then along the PATH env variable.
            symbolizer_search_paths = os.environ.get('PATH', '')
            cxx_path = lit.util.which(self.cxx.path)
            if cxx_path is not None:
                symbolizer_search_paths = (
                    os.path.dirname(cxx_path) +
                    os.pathsep + symbolizer_search_paths)
            llvm_symbolizer = lit.util.which('llvm-symbolizer',
                                             symbolizer_search_paths)
            # Setup the sanitizer compile flags
            self.cxx.flags += ['-g', '-fno-omit-frame-pointer']
            if self.target_info.platform() == 'linux':
                # The libraries and their order are taken from the
                # linkSanitizerRuntimeDeps function in
                # clang/lib/Driver/Tools.cpp
                self.cxx.link_flags += ['-lpthread', '-lrt', '-lm', '-ldl']
            if san == 'Address':
                self.cxx.flags += ['-fsanitize=address']
                if llvm_symbolizer is not None:
                    self.env['ASAN_SYMBOLIZER_PATH'] = llvm_symbolizer
                self.config.available_features.add('asan')
                self.config.available_features.add('sanitizer-new-delete')
            elif san == 'Memory' or san == 'MemoryWithOrigins':
                self.cxx.flags += ['-fsanitize=memory']
                if san == 'MemoryWithOrigins':
                    self.cxx.compile_flags += [
                        '-fsanitize-memory-track-origins']
                if llvm_symbolizer is not None:
                    self.env['MSAN_SYMBOLIZER_PATH'] = llvm_symbolizer
                self.config.available_features.add('msan')
                self.config.available_features.add('sanitizer-new-delete')
            elif san == 'Undefined':
                self.cxx.flags += ['-fsanitize=undefined',
                                   '-fno-sanitize=vptr,function',
                                   '-fno-sanitize-recover']
                self.cxx.compile_flags += ['-O3']
                self.config.available_features.add('ubsan')
                if self.target_info.platform() == 'darwin':
                    self.config.available_features.add('sanitizer-new-delete')
            elif san == 'Thread':
                self.cxx.flags += ['-fsanitize=thread']
                self.config.available_features.add('tsan')
                self.config.available_features.add('sanitizer-new-delete')
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

    def configure_substitutions(self):
        sub = self.config.substitutions
        # Configure compiler substitions
        sub.append(('%cxx', self.cxx.path))
        # Configure flags substitutions
        flags_str = ' '.join(self.cxx.flags)
        compile_flags_str = ' '.join(self.cxx.compile_flags)
        link_flags_str = ' '.join(self.cxx.link_flags)
        all_flags = '%s %s %s' % (flags_str, compile_flags_str, link_flags_str)
        sub.append(('%flags', flags_str))
        sub.append(('%compile_flags', compile_flags_str))
        sub.append(('%link_flags', link_flags_str))
        sub.append(('%all_flags', all_flags))
        # Add compile and link shortcuts
        compile_str = (self.cxx.path + ' -o %t.o %s -c ' + flags_str
                       + compile_flags_str)
        link_str = (self.cxx.path + ' -o %t.exe %t.o ' + flags_str
                    + link_flags_str)
        assert type(link_str) is str
        build_str = self.cxx.path + ' -o %t.exe %s ' + all_flags
        sub.append(('%compile', compile_str))
        sub.append(('%link', link_str))
        sub.append(('%build', build_str))
        # Configure exec prefix substitutions.
        exec_env_str = 'env ' if len(self.env) != 0 else ''
        for k, v in self.env.items():
            exec_env_str += ' %s=%s' % (k, v)
        # Configure run env substitution.
        exec_str = ''
        if self.lit_config.useValgrind:
            exec_str = ' '.join(self.lit_config.valgrindArgs) + exec_env_str
        sub.append(('%exec', exec_str))
        # Configure run shortcut
        sub.append(('%run', exec_str + ' %t.exe'))
        # Configure not program substitions
        not_py = os.path.join(self.libcxx_src_root, 'utils', 'not', 'not.py')
        not_str = '%s %s' % (sys.executable, not_py)
        sub.append(('not', not_str))

    def configure_triple(self):
        # Get or infer the target triple.
        self.config.target_triple = self.get_lit_conf('target_triple')
        self.use_target = bool(self.config.target_triple)
        # If no target triple was given, try to infer it from the compiler
        # under test.
        if not self.use_target:
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
            self.config.target_triple = target_triple
            self.lit_config.note(
                "inferred target_triple as: %r" % self.config.target_triple)

    def configure_env(self):
        if self.target_info.platform() == 'darwin':
            library_paths = []
            # Configure the library path for libc++
            libcxx_library = self.get_lit_conf('libcxx_library')
            if self.use_system_cxx_lib:
                pass
            elif libcxx_library:
                library_paths += [os.path.dirname(libcxx_library)]
            elif self.cxx_library_root:
                library_paths += [self.cxx_library_root]
            # Configure the abi library path
            if self.abi_library_root:
                library_paths += [self.abi_library_root]
            if library_paths:
                self.env['DYLD_LIBRARY_PATH'] = ':'.join(library_paths)
