import locale
import os
import platform
import re
import shlex
import sys

import lit.Test  # pylint: disable=import-error,no-name-in-module
import lit.util  # pylint: disable=import-error,no-name-in-module

from libcxx.test.format import LibcxxTestFormat


class Configuration(object):
    # pylint: disable=redefined-outer-name
    def __init__(self, lit_config, config):
        self.lit_config = lit_config
        self.config = config
        self.cxx = None
        self.src_root = None
        self.obj_root = None
        self.library_root = None
        self.env = {}
        self.compile_flags = []
        self.link_flags = []
        self.use_system_lib = False
        self.use_clang_verify = False
        self.use_ccache = False
        self.long_tests = None

        if platform.system() not in ('Darwin', 'FreeBSD', 'Linux'):
            self.lit_config.fatal("unrecognized system")

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
        self.configure_cxx()
        self.probe_cxx()
        self.configure_triple()
        self.configure_src_root()
        self.configure_obj_root()
        self.configure_library_root()
        self.configure_use_system_lib()
        self.configure_use_clang_verify()
        self.configure_ccache()
        self.configure_env()
        self.configure_compile_flags()
        self.configure_link_flags()
        self.configure_sanitizer()
        self.configure_features()
        # Print the final compile and link flags.
        self.lit_config.note('Using compile flags: %s' % self.compile_flags)
        self.lit_config.note('Using link flags: %s' % self.link_flags)
        # Print as list to prevent "set([...])" from being printed.
        self.lit_config.note('Using available_features: %s' %
                             list(self.config.available_features))

    def get_test_format(self):
        return LibcxxTestFormat(
            self.cxx,
            self.use_clang_verify,
            cpp_flags=self.compile_flags,
            ld_flags=self.link_flags,
            exec_env=self.env,
            use_ccache=self.use_ccache)

    def configure_cxx(self):
        # Gather various compiler parameters.
        self.cxx = self.get_lit_conf('cxx_under_test')

        # If no specific cxx_under_test was given, attempt to infer it as
        # clang++.
        if self.cxx is None:
            clangxx = lit.util.which('clang++',
                                     self.config.environment['PATH'])
            if clangxx:
                self.cxx = clangxx
                self.lit_config.note(
                    "inferred cxx_under_test as: %r" % self.cxx)
        if not self.cxx:
            self.lit_config.fatal('must specify user parameter cxx_under_test '
                                  '(e.g., --param=cxx_under_test=clang++)')

    def probe_cxx(self):
        # Dump all of the predefined macros
        dump_macro_cmd = [self.cxx, '-dM', '-E', '-x', 'c++', '/dev/null']
        out, _, rc = lit.util.executeCommand(dump_macro_cmd)
        if rc != 0:
            self.lit_config.warning('Failed to dump macros for compiler: %s' %
                                    self.cxx)
            return
        # Create a dict containing all the predefined macros.
        macros = {}
        lines = [l.strip() for l in out.split('\n') if l.strip()]
        for l in lines:
            assert l.startswith('#define ')
            l = l[len('#define '):]
            macro, _, value = l.partition(' ')
            macros[macro] = value
        # Add compiler information to available features.
        compiler_name = None
        major_ver = minor_ver = None
        if '__clang__' in macros.keys():
            compiler_name = 'clang'
            # Treat apple's llvm fork differently.
            if '__apple_build_version__' in macros.keys():
                compiler_name = 'apple-clang'
            major_ver = macros['__clang_major__']
            minor_ver = macros['__clang_minor__']
        elif '__GNUC__' in macros.keys():
            compiler_name = 'gcc'
            major_ver = macros['__GNUC__']
            minor_ver = macros['__GNUC_MINOR__']
        else:
            self.lit_config.warning('Failed to detect compiler for cxx: %s' %
                                    self.cxx)
        if compiler_name is not None:
            self.config.available_features.add(compiler_name)
            self.config.available_features.add('%s-%s.%s' % (
                compiler_name, major_ver, minor_ver))

    def configure_src_root(self):
        self.src_root = self.get_lit_conf(
            'libcxx_src_root', os.path.dirname(self.config.test_source_root))

    def configure_obj_root(self):
        self.obj_root = self.get_lit_conf('libcxx_obj_root', self.src_root)

    def configure_library_root(self):
        self.library_root = self.get_lit_conf('libcxx_library_root',
                                              self.obj_root)

    def configure_use_system_lib(self):
        # This test suite supports testing against either the system library or
        # the locally built one; the former mode is useful for testing ABI
        # compatibility between the current headers and a shipping dynamic
        # library.
        self.use_system_lib = self.get_lit_bool('use_system_lib')
        if self.use_system_lib is None:
            # Default to testing against the locally built libc++ library.
            self.use_system_lib = False
            self.lit_config.note(
                "inferred use_system_lib as: %r" % self.use_system_lib)

    def configure_use_clang_verify(self):
        '''If set, run clang with -verify on failing tests.'''
        self.use_clang_verify = self.get_lit_bool('use_clang_verify')
        if self.use_clang_verify is None:
            # TODO: Default this to True when using clang.
            self.use_clang_verify = False
            self.lit_config.note(
                "inferred use_clang_verify as: %r" % self.use_clang_verify)

    def configure_ccache(self):
        self.use_ccache = self.get_lit_bool('use_ccache', False)
        if self.use_ccache:
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
                'fr_CA.ISO8859-1': 'cs_CZ.ISO8859-1',
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

        default_locale = locale.setlocale(locale.LC_ALL)
        for feature, loc in locales[platform.system()].items():
            try:
                locale.setlocale(locale.LC_ALL, loc)
                self.config.available_features.add(
                    'locale.{0}'.format(feature))
            except locale.Error:
                self.lit_config.warning('The locale {0} is not supported by '
                                        'your platform. Some tests will be '
                                        'unsupported.'.format(loc))
        locale.setlocale(locale.LC_ALL, default_locale)

        # Write an "available feature" that combines the triple when
        # use_system_lib is enabled. This is so that we can easily write XFAIL
        # markers for tests that are known to fail with versions of libc++ as
        # were shipped with a particular triple.
        if self.use_system_lib:
            self.config.available_features.add(
                'with_system_lib=%s' % self.config.target_triple)

        # Some linux distributions have different locale data than others.
        # Insert the distributions name and name-version into the available
        # features to allow tests to XFAIL on them.
        if sys.platform.startswith('linux'):
            name, ver, _ = platform.linux_distribution()
            name = name.lower().strip()
            ver = ver.lower().strip()
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

    def configure_compile_flags(self):
        # Try and get the std version from the command line. Fall back to
        # default given in lit.site.cfg is not present. If default is not
        # present then force c++11.
        std = self.get_lit_conf('std', 'c++11')
        self.compile_flags += ['-std={0}'.format(std)]
        self.config.available_features.add(std)
        # Configure include paths
        self.compile_flags += ['-nostdinc++']
        support_path = os.path.join(self.src_root, 'test/support')
        self.cxx.compile_flags += ['-I' + support_path]
        self.cxx.compile_flags += ['-include', os.path.join(support_path, 'nasty_macros.hpp')]
        libcxx_headers = self.get_lit_conf('libcxx_headers',
                                           self.src_root + '/include')
        if not os.path.isdir(libcxx_headers):
            self.lit_config.fatal("libcxx_headers='%s' is not a directory."
                                  % libcxx_headers)
        self.compile_flags += ['-I' + libcxx_headers]
        if sys.platform.startswith('linux'):
            self.compile_flags += ['-D__STDC_FORMAT_MACROS',
                                   '-D__STDC_LIMIT_MACROS',
                                   '-D__STDC_CONSTANT_MACROS']
        # Configure feature flags.
        enable_exceptions = self.get_lit_bool('enable_exceptions', True)
        if enable_exceptions:
            self.config.available_features.add('exceptions')
        else:
            self.compile_flags += ['-fno-exceptions']
        enable_rtti = self.get_lit_bool('enable_rtti', True)
        if enable_rtti:
            self.config.available_features.add('rtti')
        else:
            self.compile_flags += ['-fno-rtti', '-D_LIBCPP_NO_RTTI']
        enable_32bit = self.get_lit_bool('enable_32bit', False)
        if enable_32bit:
            self.compile_flags += ['-m32']
        # Configure threading features.
        enable_threads = self.get_lit_bool('enable_threads', True)
        enable_monotonic_clock = self.get_lit_bool('enable_monotonic_clock',
                                                   True)
        if not enable_threads:
            self.compile_flags += ['-D_LIBCPP_HAS_NO_THREADS']
            self.config.available_features.add('libcpp-has-no-threads')
            if not enable_monotonic_clock:
                self.compile_flags += ['-D_LIBCPP_HAS_NO_MONOTONIC_CLOCK']
                self.config.available_features.add(
                    'libcpp-has-no-monotonic-clock')
        elif not enable_monotonic_clock:
            self.lit_config.fatal('enable_monotonic_clock cannot be false when'
                                  ' enable_threads is true.')
        # Use verbose output for better errors
        self.compile_flags += ['-v']
        # Configure extra compile flags.
        compile_flags_str = self.get_lit_conf('compile_flags', '')
        self.compile_flags += shlex.split(compile_flags_str)

    def configure_link_flags(self):
        self.link_flags += ['-nodefaultlibs']
        libcxx_library = self.get_lit_conf('libcxx_library')
        # Configure libc++ library paths.
        if libcxx_library is not None:
            # Check that the given value for libcxx_library is valid.
            if not os.path.isfile(libcxx_library):
                self.lit_config.fatal(
                    "libcxx_library='%s' is not a valid file." %
                    libcxx_library)
            if self.use_system_lib:
                self.lit_config.fatal(
                    "Conflicting options: 'libcxx_library' cannot be used "
                    "with 'use_system_lib=true'")
            self.link_flags += ['-Wl,-rpath,' +
                                os.path.dirname(libcxx_library)]
        elif not self.use_system_lib:
            self.link_flags += ['-L' + self.library_root,
                                '-Wl,-rpath,' + self.library_root]
        # Configure ABI library paths.
        abi_library_path = self.get_lit_conf('abi_library_path', '')
        if abi_library_path:
            self.link_flags += ['-L' + abi_library_path,
                                '-Wl,-rpath,' + abi_library_path]

        # Configure libraries
        self.configure_link_flags_cxx_library()
        self.configure_link_flags_abi_library()
        self.configure_extra_library_flags()

        link_flags_str = self.get_lit_conf('link_flags', '')
        self.link_flags += shlex.split(link_flags_str)

    def configure_link_flags_cxx_library(self):
        libcxx_library = self.get_lit_conf('libcxx_library')
        if libcxx_library:
            self.link_flags += [libcxx_library]
        else:
            self.link_flags += ['-lc++']

    def configure_link_flags_abi_library(self):
        cxx_abi = self.get_lit_conf('cxx_abi', 'libcxxabi')
        if cxx_abi == 'libstdc++':
            self.link_flags += ['-lstdc++']
        elif cxx_abi == 'libsupc++':
            self.link_flags += ['-lsupc++']
        elif cxx_abi == 'libcxxabi':
            self.link_flags += ['-lc++abi']
        elif cxx_abi == 'libcxxrt':
            self.link_flags += ['-lcxxrt']
        elif cxx_abi == 'none':
            pass
        else:
            self.lit_config.fatal(
                'C++ ABI setting %s unsupported for tests' % cxx_abi)

    def configure_extra_library_flags(self):
        if sys.platform == 'darwin':
            self.link_flags += ['-lSystem']
        elif sys.platform.startswith('linux'):
            self.link_flags += ['-lgcc_eh', '-lc', '-lm', '-lpthread',
                                '-lrt', '-lgcc_s']
        elif sys.platform.startswith('freebsd'):
            self.link_flags += ['-lc', '-lm', '-pthread', '-lgcc_s']
        else:
            self.lit_config.fatal("unrecognized system: %r" % sys.platform)

    def configure_sanitizer(self):
        san = self.get_lit_conf('use_sanitizer', '').strip()
        if san:
            # Search for llvm-symbolizer along the compiler path first
            # and then along the PATH env variable.
            symbolizer_search_paths = os.environ.get('PATH', '')
            cxx_path = lit.util.which(self.cxx)
            if cxx_path is not None:
                symbolizer_search_paths = (
                    os.path.dirname(cxx_path) +
                    os.pathsep + symbolizer_search_paths)
            llvm_symbolizer = lit.util.which('llvm-symbolizer',
                                             symbolizer_search_paths)
            # Setup the sanitizer compile flags
            self.compile_flags += ['-g', '-fno-omit-frame-pointer']
            if sys.platform.startswith('linux'):
                self.link_flags += ['-ldl']
            if san == 'Address':
                self.compile_flags += ['-fsanitize=address']
                if llvm_symbolizer is not None:
                    self.env['ASAN_SYMBOLIZER_PATH'] = llvm_symbolizer
                self.config.available_features.add('asan')
            elif san == 'Memory' or san == 'MemoryWithOrigins':
                self.compile_flags += ['-fsanitize=memory']
                if san == 'MemoryWithOrigins':
                    self.compile_flags += ['-fsanitize-memory-track-origins']
                if llvm_symbolizer is not None:
                    self.env['MSAN_SYMBOLIZER_PATH'] = llvm_symbolizer
                self.config.available_features.add('msan')
            elif san == 'Undefined':
                self.compile_flags += ['-fsanitize=undefined',
                                       '-fno-sanitize=vptr,function',
                                       '-fno-sanitize-recover', '-O3']
                self.config.available_features.add('ubsan')
            elif san == 'Thread':
                self.compile_flags += ['-fsanitize=thread']
                self.config.available_features.add('tsan')
            else:
                self.lit_config.fatal('unsupported value for '
                                      'use_sanitizer: {0}'.format(san))

    def configure_triple(self):
        # Get or infer the target triple.
        self.config.target_triple = self.get_lit_conf('target_triple')
        # If no target triple was given, try to infer it from the compiler
        # under test.
        if not self.config.target_triple:
            target_triple = lit.util.capture(
                [self.cxx, '-dumpmachine']).strip()
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
            if target_triple.endswith('redhat-linux') or \
               target_triple.endswith('suse-linux'):
                target_triple += '-gnu'
            self.config.target_triple = target_triple
            self.lit_config.note(
                "inferred target_triple as: %r" % self.config.target_triple)

    def configure_env(self):
        if sys.platform == 'darwin' and not self.use_system_lib:
            libcxx_library = self.get_lit_conf('libcxx_library')
            if libcxx_library:
                library_root = os.path.dirname(libcxx_library)
            else:
                library_root = self.library_root
            self.env['DYLD_LIBRARY_PATH'] = library_root
