#===----------------------------------------------------------------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===//

import importlib
import locale
import os
import platform
import re
import subprocess
import sys

from libcxx.util import executeCommand

class DefaultTargetInfo(object):
    def __init__(self, full_config):
        self.full_config = full_config
        self.executor = None

    def platform(self):
        return sys.platform.lower().strip()

    def is_windows(self):
        return self.platform() == 'win32'

    def is_darwin(self):
        return self.platform() == 'darwin'

    def add_locale_features(self, features):
        self.full_config.lit_config.warning(
            "No locales entry for target_system: %s" % self.platform())

    def add_cxx_compile_flags(self, flags): pass
    def add_cxx_link_flags(self, flags): pass
    def configure_env(self, env): pass
    def allow_cxxabi_link(self): return True
    def use_lit_shell_default(self): return False

    def add_platform_features(self, features):
        features.add(self.platform())

    def add_path(self, dest_env, new_path):
        if not new_path:
            return
        if 'PATH' not in dest_env:
            dest_env['PATH'] = new_path
        else:
            split_char = ';' if self.is_windows() else ':'
            dest_env['PATH'] = '%s%s%s' % (new_path, split_char,
                                           dest_env['PATH'])

    def _test_locale(self, loc):
        assert loc is not None
        default_locale = locale.setlocale(locale.LC_ALL)
        try:
            locale.setlocale(locale.LC_ALL, loc)
            return True
        except locale.Error:
            return False
        finally:
            locale.setlocale(locale.LC_ALL, default_locale)

    def add_common_locales(self, features, is_windows=False):
        # A list of locales needed by the test-suite.
        # The list uses the canonical name for the locale used in the test-suite
        # TODO: On Linux ISO8859 *may* needs to hyphenated.
        locales = [
            ('en_US.UTF-8', 'English_United States.1252'),
            ('fr_FR.UTF-8', 'French_France.1252'),
            ('ru_RU.UTF-8', 'Russian_Russia.1251'),
            ('zh_CN.UTF-8', 'Chinese_China.936'),
            ('fr_CA.ISO8859-1', 'French_Canada.1252'),
            ('cs_CZ.ISO8859-2', 'Czech_Czech Republic.1250')
        ]
        for loc_id, windows_loc_name in locales:
            loc_name = windows_loc_name if is_windows else loc_id
            if self._test_locale(loc_name):
                features.add('locale.{0}'.format(loc_id))
            else:
                self.full_config.lit_config.warning('The locale {0} is not supported by '
                                'your platform. Some tests will be '
                                'unsupported.'.format(loc_name))


class DarwinLocalTI(DefaultTargetInfo):
    def __init__(self, full_config):
        super(DarwinLocalTI, self).__init__(full_config)

    def is_host_macosx(self):
        name = subprocess.check_output(['sw_vers', '-productName']).decode().strip()
        return name == "Mac OS X"

    def get_macosx_version(self):
        assert self.is_host_macosx()
        version = subprocess.check_output(['sw_vers', '-productVersion']).decode().strip()
        version = re.sub(r'([0-9]+\.[0-9]+)(\..*)?', r'\1', version)
        return version

    def get_sdk_version(self, name):
        assert self.is_host_macosx()
        cmd = ['xcrun', '--sdk', name, '--show-sdk-path']
        try:
            out = subprocess.check_output(cmd).strip()
        except OSError:
            pass

        if not out:
            self.full_config.lit_config.fatal(
                    "cannot infer sdk version with: %r" % cmd)

        return re.sub(r'.*/[^0-9]+([0-9.]+)\.sdk', r'\1', out)

    def get_platform(self):
        platform = self.full_config.get_lit_conf('platform')
        if platform:
            platform = re.sub(r'([^0-9]+)([0-9\.]*)', r'\1-\2', platform)
            name, version = tuple(platform.split('-', 1))
        else:
            name = 'macosx'
            version = None

        if version:
            return (False, name, version)

        # Infer the version, either from the SDK or the system itself.  For
        # macosx, ignore the SDK version; what matters is what's at
        # /usr/lib/libc++.dylib.
        if name == 'macosx':
            version = self.get_macosx_version()
        else:
            version = self.get_sdk_version(name)
        return (True, name, version)

    def add_locale_features(self, features):
        self.add_common_locales(features)

    def add_cxx_compile_flags(self, flags):
        if self.full_config.use_deployment:
            _, name, _ = self.full_config.config.deployment
            cmd = ['xcrun', '--sdk', name, '--show-sdk-path']
        else:
            cmd = ['xcrun', '--show-sdk-path']
        out, err, exit_code = executeCommand(cmd)
        if exit_code != 0:
            self.full_config.lit_config.warning("Could not determine macOS SDK path! stderr was " + err)
        if exit_code == 0 and out:
            sdk_path = out.strip()
            self.full_config.lit_config.note('using SDKROOT: %r' % sdk_path)
            assert isinstance(sdk_path, str)
            flags += ["-isysroot", sdk_path]

    def add_cxx_link_flags(self, flags):
        flags += ['-lSystem']

    def configure_env(self, env):
        library_paths = []
        # Configure the library path for libc++
        if self.full_config.cxx_runtime_root:
            library_paths += [self.full_config.cxx_runtime_root]
        elif self.full_config.use_system_cxx_lib:
            if (os.path.isdir(str(self.full_config.use_system_cxx_lib))):
                library_paths += [self.full_config.use_system_cxx_lib]

        # Configure the abi library path
        if self.full_config.abi_library_root:
            library_paths += [self.full_config.abi_library_root]
        if library_paths:
            env['DYLD_LIBRARY_PATH'] = ':'.join(library_paths)

    def allow_cxxabi_link(self):
        # Don't link libc++abi explicitly on OS X because the symbols
        # should be available in libc++ directly.
        return False


class FreeBSDLocalTI(DefaultTargetInfo):
    def __init__(self, full_config):
        super(FreeBSDLocalTI, self).__init__(full_config)

    def add_locale_features(self, features):
        self.add_common_locales(features)

    def add_cxx_link_flags(self, flags):
        flags += ['-lc', '-lm', '-lpthread', '-lgcc_s', '-lcxxrt']


class NetBSDLocalTI(DefaultTargetInfo):
    def __init__(self, full_config):
        super(NetBSDLocalTI, self).__init__(full_config)

    def add_locale_features(self, features):
        self.add_common_locales(features)

    def add_cxx_link_flags(self, flags):
        flags += ['-lc', '-lm', '-lpthread', '-lgcc_s', '-lc++abi',
                  '-lunwind']


class LinuxLocalTI(DefaultTargetInfo):
    def __init__(self, full_config):
        super(LinuxLocalTI, self).__init__(full_config)

    def platform(self):
        return 'linux'

    def _distribution(self):
        try:
            # linux_distribution is not available since Python 3.8
            # However, this function is only used to detect SLES 11,
            # which is quite an old distribution that doesn't have
            # Python 3.8.
            return platform.linux_distribution()
        except AttributeError:
            return '', '', ''

    def platform_name(self):
        name, _, _ = self._distribution()
        # Some distros have spaces, e.g. 'SUSE Linux Enterprise Server'
        # lit features can't have spaces
        name = name.lower().strip().replace(' ', '-')
        return name # Permitted to be None

    def platform_ver(self):
        _, ver, _ = self._distribution()
        ver = ver.lower().strip().replace(' ', '-')
        return ver # Permitted to be None.

    def add_locale_features(self, features):
        self.add_common_locales(features)

    def add_platform_features(self, features):
        super(LinuxLocalTI, self).add_platform_features(features)

        # Some linux distributions have different locale data than others.
        # Insert the distributions name and name-version into the available
        # features to allow tests to XFAIL on them.
        name = self.platform_name()
        ver = self.platform_ver()
        if name:
            features.add(name)
        if name and ver:
            features.add('%s-%s' % (name, ver))

    def add_cxx_compile_flags(self, flags):
        flags += ['-D__STDC_FORMAT_MACROS',
                  '-D__STDC_LIMIT_MACROS',
                  '-D__STDC_CONSTANT_MACROS']

    def add_cxx_link_flags(self, flags):
        enable_threads = ('libcpp-has-no-threads' not in
                          self.full_config.config.available_features)
        llvm_unwinder = self.full_config.get_lit_bool('llvm_unwinder', False)
        shared_libcxx = self.full_config.get_lit_bool('enable_shared', True)
        flags += ['-lm']
        if not llvm_unwinder:
            flags += ['-lgcc_s', '-lgcc']
        if enable_threads:
            flags += ['-lpthread']
            if not shared_libcxx:
                flags += ['-lrt']
        flags += ['-lc']
        if llvm_unwinder:
            flags += ['-lunwind', '-ldl']
        else:
            flags += ['-lgcc_s']
        builtins_lib = self.full_config.get_lit_conf('builtins_library')
        if builtins_lib:
            flags += [builtins_lib]
        else:
            flags += ['-lgcc']
        use_libatomic = self.full_config.get_lit_bool('use_libatomic', False)
        if use_libatomic:
            flags += ['-latomic']
        san = self.full_config.get_lit_conf('use_sanitizer', '').strip()
        if san:
            # The libraries and their order are taken from the
            # linkSanitizerRuntimeDeps function in
            # clang/lib/Driver/Tools.cpp
            flags += ['-lpthread', '-lrt', '-lm', '-ldl']

class LinuxRemoteTI(LinuxLocalTI):
    def __init__(self, full_config):
        super(LinuxRemoteTI, self).__init__(full_config)
        self.__cached_locales = None

    def _test_locale(self, loc):
        if self.__cached_locales is None:
            self.full_config.lit_config.note('asking the remote host for supported locales...')
            _, out, err, exit_code = self.executor.execute_command_remote(['locale', '-a'])
            if exit_code != 0:
                raise RuntimeError(err)
            self.__cached_locales = out.splitlines()

        # It is possible that the output will use 'en_US.utf8' instead of 'en_US.UTF-8',
        # so we check both variants.
        return loc in self.__cached_locales or \
               loc.replace('UTF-8', 'utf8') in self.__cached_locales

class WindowsLocalTI(DefaultTargetInfo):
    def __init__(self, full_config):
        super(WindowsLocalTI, self).__init__(full_config)

    def add_locale_features(self, features):
        self.add_common_locales(features, is_windows=True)

    def use_lit_shell_default(self):
        # Default to the internal shell on Windows, as bash on Windows is
        # usually very slow.
        return True


def make_target_info(full_config):
    default = "libcxx.test.target_info.LocalTI"
    info_str = full_config.get_lit_conf('target_info', default)
    if info_str != default:
        mod_path, _, info = info_str.rpartition('.')
        mod = importlib.import_module(mod_path)
        target_info = getattr(mod, info)(full_config)
        full_config.lit_config.note("inferred target_info as: %r" % info_str)
        return target_info
    target_system = platform.system()
    if target_system == 'Darwin':  return DarwinLocalTI(full_config)
    if target_system == 'FreeBSD': return FreeBSDLocalTI(full_config)
    if target_system == 'NetBSD':  return NetBSDLocalTI(full_config)
    if target_system == 'Linux':   return LinuxLocalTI(full_config)
    if target_system == 'Windows': return WindowsLocalTI(full_config)
    return DefaultTargetInfo(full_config)
