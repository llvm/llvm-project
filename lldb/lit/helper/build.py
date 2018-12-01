from __future__ import print_function

import argparse
import os
import signal
import subprocess
import sys

if sys.platform == 'win32':
    # This module was renamed in Python 3.  Make sure to import it using a
    # consistent name regardless of python version.
    try:
        import winreg
    except:
        import _winreg as winreg

if __name__ != "__main__":
    raise RuntimeError("Do not import this script, run it instead")


parser = argparse.ArgumentParser(description='LLDB compilation wrapper')
parser.add_argument('--arch',
                    metavar='arch',
                    dest='arch',
                    required=True,
                    help='Specify the architecture to target.  Valid values=[32,64]')

parser.add_argument('--compiler',
                    metavar='compiler',
                    dest='compiler',
                    required=True,
                    help='Path to a compiler executable, or one of the values [any, msvc, clang-cl, gcc, clang]')

parser.add_argument('--tools-dir',
                    metavar='directory',
                    dest='tools_dir',
                    required=False,
                    action='append',
                    help='If specified, a path to search in addition to PATH when --compiler is not an exact path')

if sys.platform == 'darwin':
    parser.add_argument('--apple-sdk',
                        metavar='apple_sdk',
                        dest='apple_sdk',
                        default="macosx",
                        help='Specify the name of the Apple SDK (macosx, macosx.internal, iphoneos, iphoneos.internal, or path to SDK) and use the appropriate tools from that SDK\'s toolchain.')

parser.add_argument('--output', '-o',
                    dest='output',
                    metavar='file',
                    required=True,
                    help='Path to output file')

parser.add_argument('--nodefaultlib',
                    dest='nodefaultlib',
                    action='store_true',
                    default=False,
                    help='When specified, the resulting image should not link against system libraries or include system headers.  Useful when writing cross-targeting tests.')

parser.add_argument('--opt',
                    dest='opt',
                    default='none',
                    choices=['none', 'basic', 'lto'],
                    help='Optimization level')

parser.add_argument('--mode',
                    dest='mode',
                    default='compile-and-link',
                    choices=['compile', 'link', 'compile-and-link'],
                    help='Specifies whether to compile, link, or both')

parser.add_argument('--noclean',
                    dest='clean',
                    action='store_false',
                    default=True,
                    help='Dont clean output file before building')

parser.add_argument('--verbose',
                    dest='verbose',
                    action='store_true',
                    default=False,
                    help='Print verbose output')

parser.add_argument('input',
                    metavar='file',
                    help='Source file to compile / object file to link')


args = parser.parse_args(args=sys.argv[1:])


def to_string(b):
    """Return the parameter as type 'str', possibly encoding it.

    In Python2, the 'str' type is the same as 'bytes'. In Python3, the
    'str' type is (essentially) Python2's 'unicode' type, and 'bytes' is
    distinct.

    This function is copied from llvm/utils/lit/lit/util.py
    """
    if isinstance(b, str):
        # In Python2, this branch is taken for types 'str' and 'bytes'.
        # In Python3, this branch is taken only for 'str'.
        return b
    if isinstance(b, bytes):
        # In Python2, this branch is never taken ('bytes' is handled as 'str').
        # In Python3, this is true only for 'bytes'.
        try:
            return b.decode('utf-8')
        except UnicodeDecodeError:
            # If the value is not valid Unicode, return the default
            # repr-line encoding.
            return str(b)

    # By this point, here's what we *don't* have:
    #
    #  - In Python2:
    #    - 'str' or 'bytes' (1st branch above)
    #  - In Python3:
    #    - 'str' (1st branch above)
    #    - 'bytes' (2nd branch above)
    #
    # The last type we might expect is the Python2 'unicode' type. There is no
    # 'unicode' type in Python3 (all the Python3 cases were already handled). In
    # order to get a 'str' object, we need to encode the 'unicode' object.
    try:
        return b.encode('utf-8')
    except AttributeError:
        raise TypeError('not sure how to convert %s to %s' % (type(b), str))

def print_environment(env):
    for e in env:
        value = env[e]
        split = value.split(os.pathsep)
        print('    {0} = {1}'.format(e, split[0]))
        prefix_width = 3 + len(e)
        for next in split[1:]:
            print('    {0}{1}'.format(' ' * prefix_width, next))

def find_executable(binary_name, search_paths):
    if sys.platform == 'win32':
        binary_name = binary_name + '.exe'

    search_paths = os.pathsep.join(search_paths)
    paths = search_paths + os.pathsep + os.environ.get('PATH', '')
    for path in paths.split(os.pathsep):
        p = os.path.join(path, binary_name)
        if os.path.exists(p) and not os.path.isdir(p):
            return os.path.normpath(p)
    return None

def find_toolchain(compiler, tools_dir):
    if compiler == 'msvc':
        return ('msvc', find_executable('cl', tools_dir))
    if compiler == 'clang-cl':
        return ('clang-cl', find_executable('clang-cl', tools_dir))
    if compiler == 'gcc':
        return ('gcc', find_executable('g++', tools_dir))
    if compiler == 'clang':
        return ('clang', find_executable('clang++', tools_dir))
    if compiler == 'any':
        priorities = []
        if sys.platform == 'win32':
            priorities = ['clang-cl', 'msvc', 'clang', 'gcc']
        else:
            priorities = ['clang', 'gcc', 'clang-cl']
        for toolchain in priorities:
            (type, dir) = find_toolchain(toolchain, tools_dir)
            if type and dir:
                return (type, dir)
        # Could not find any toolchain.
        return (None, None)

    # From here on, assume that |compiler| is a path to a file.
    file = os.path.basename(compiler)
    name, ext = os.path.splitext(file)
    if file.lower() == 'cl.exe':
        return 'msvc'
    if name == 'clang-cl':
        return 'clang-cl'
    if name.startswith('clang'):
        return 'clang'
    if name.startswith('gcc') or name.startswith('g++'):
        return 'gcc'
    if name == 'cc' or name == 'c++':
        return 'generic'
    return 'unknown'

class Builder(object):
    def __init__(self, toolchain_type, args):
        self.toolchain_type = toolchain_type
        self.input = args.input
        self.arch = args.arch
        self.opt = args.opt
        self.compiler = args.compiler
        self.clean = args.clean
        self.output = args.output
        self.mode = args.mode
        self.nodefaultlib = args.nodefaultlib
        self.verbose = args.verbose

class MsvcBuilder(Builder):
    def __init__(self, toolchain_type, args):
        Builder.__init__(self, toolchain_type, args)

        self.msvc_arch_str = 'x86' if self.arch == '32' else 'x64'

        if toolchain_type == 'msvc':
            # Make sure we're using the appropriate toolchain for the desired
            # target type.
            compiler_parent_dir = os.path.dirname(self.compiler)
            selected_target_version = os.path.basename(compiler_parent_dir)
            if selected_target_version != self.msvc_arch_str:
                host_dir = os.path.dirname(compiler_parent_dir)
                self.compiler = os.path.join(host_dir, self.msvc_arch_str, 'cl.exe')
                if self.verbose:
                    print('Using alternate compiler "{0}" to match selected target.'.format(self.compiler))

        if self.mode == 'link' or self.mode == 'compile-and-link':
            self.linker = self._find_linker('link') if toolchain_type == 'msvc' else self._find_linker('lld-link')
            if not self.linker:
                raise ValueError('Unable to find an appropriate linker.')

        self.compile_env, self.link_env = self._get_visual_studio_environment()

    def _find_linker(self, name):
        if sys.platform == 'win32':
            name = name + '.exe'
        compiler_dir = os.path.dirname(self.compiler)
        linker_path = os.path.join(compiler_dir, name)
        if not os.path.exists(linker_path):
            raise ValueError('Could not find \'{}\''.format(linker_path))
        return linker_path

    def _get_vc_install_dir(self):
        dir = os.getenv('VCINSTALLDIR', None)
        if dir:
            if self.verbose:
                print('Using %VCINSTALLDIR% {}'.format(dir))
            return dir

        dir = os.getenv('VSINSTALLDIR', None)
        if dir:
            if self.verbose:
                print('Using %VSINSTALLDIR% {}'.format(dir))
            return os.path.join(dir, 'VC')

        dir = os.getenv('VS2019INSTALLDIR', None)
        if dir:
            if self.verbose:
                print('Using %VS2019INSTALLDIR% {}'.format(dir))
            return os.path.join(dir, 'VC')

        dir = os.getenv('VS2017INSTALLDIR', None)
        if dir:
            if self.verbose:
                print('Using %VS2017INSTALLDIR% {}'.format(dir))
            return os.path.join(dir, 'VC')

        dir = os.getenv('VS2015INSTALLDIR', None)
        if dir:
            if self.verbose:
                print('Using %VS2015INSTALLDIR% {}'.format(dir))
            return os.path.join(dir, 'VC')
        return None

    def _get_vctools_version(self):
        ver = os.getenv('VCToolsVersion', None)
        if ver:
            if self.verbose:
                print('Using %VCToolsVersion% {}'.format(ver))
            return ver

        vcinstalldir = self._get_vc_install_dir()
        vcinstalldir = os.path.join(vcinstalldir, 'Tools', 'MSVC')
        subdirs = next(os.walk(vcinstalldir))[1]
        if not subdirs:
            return None

        from distutils.version import StrictVersion
        subdirs.sort(key=lambda x : StrictVersion(x))

        if self.verbose:
            full_path = os.path.join(vcinstalldir, subdirs[-1])
            print('Using VC tools version directory {0} found by directory walk.'.format(full_path))
        return subdirs[-1]

    def _get_vctools_install_dir(self):
        dir = os.getenv('VCToolsInstallDir', None)
        if dir:
            if self.verbose:
                print('Using %VCToolsInstallDir% {}'.format(dir))
            return dir

        vcinstalldir = self._get_vc_install_dir()
        if not vcinstalldir:
            return None
        vctoolsver = self._get_vctools_version()
        if not vctoolsver:
            return None
        result = os.path.join(vcinstalldir, 'Tools', 'MSVC', vctoolsver)
        if not os.path.exists(result):
            return None
        if self.verbose:
            print('Using VC tools install dir {} found by directory walk'.format(result))
        return result

    def _find_windows_sdk_in_registry_view(self, view):
        products_key = None
        roots_key = None
        installed_options_keys = []
        try:
            sam = view | winreg.KEY_READ
            products_key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                          r'Software\Microsoft\Windows Kits\Installed Products',
                                          0,
                                          sam)

            # This is the GUID for the desktop component.  If this is present
            # then the components required for the Desktop SDK are installed.
            # If not it will throw an exception.
            winreg.QueryValueEx(products_key, '{5A3D81EC-D870-9ECF-D997-24BDA6644752}')

            roots_key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                       r'Software\Microsoft\Windows Kits\Installed Roots',
                                       0,
                                       sam)
            root_dir = winreg.QueryValueEx(roots_key, 'KitsRoot10')
            root_dir = to_string(root_dir[0])
            sdk_versions = []
            index = 0
            while True:
                # Installed SDK versions are stored as sub-keys of the
                # 'Installed Roots' key.  Find all of their names, then sort
                # them by version
                try:
                    ver_key = winreg.EnumKey(roots_key, index)
                    sdk_versions.append(ver_key)
                    index = index + 1
                except WindowsError:
                    break
            if not sdk_versions:
                return (None, None)

            # Windows SDK version numbers consist of 4 dotted components, so we
            # have to use LooseVersion, as StrictVersion supports 3 or fewer.
            from distutils.version import LooseVersion
            sdk_versions.sort(key=lambda x : LooseVersion(x), reverse=True)
            option_value_name = 'OptionId.DesktopCPP' + self.msvc_arch_str
            for v in sdk_versions:
                try:
                    version_subkey = v + r'\Installed Options'
                    key = winreg.OpenKey(roots_key, version_subkey)
                    installed_options_keys.append(key)
                    (value, value_type) = winreg.QueryValueEx(key, option_value_name)
                    if value == 1:
                        # The proper architecture is installed.  Return the
                        # associated paths.
                        if self.verbose:
                            print('Found Installed Windows SDK v{0} at {1}'.format(v, root_dir))
                        return (root_dir, v)
                except:
                    continue
        except:
            return (None, None)
        finally:
            del products_key
            del roots_key
            for k in installed_options_keys:
                del k
        return (None, None)

    def _find_windows_sdk_in_registry(self):
        # This could be a clang-cl cross-compile.  If so, there's no registry
        # so just exit.
        if sys.platform != 'win32':
            return (None, None)
        if self.verbose:
            print('Looking for Windows SDK in 64-bit registry.')
        dir, ver = self._find_windows_sdk_in_registry_view(winreg.KEY_WOW64_64KEY)
        if not dir or not ver:
            if self.verbose:
                print('Looking for Windows SDK in 32-bit registry.')
            dir, ver = self._find_windows_sdk_in_registry_view(winreg.KEY_WOW64_32KEY)

        return (dir, ver)

    def _get_winsdk_dir(self):
        # If a Windows SDK is specified in the environment, use that.  Otherwise
        # try to find one in the Windows registry.
        dir = os.getenv('WindowsSdkDir', None)
        if not dir or not os.path.exists(dir):
            return self._find_windows_sdk_in_registry()
        ver = os.getenv('WindowsSDKLibVersion', None)
        if not ver:
            return self._find_windows_sdk_in_registry()

        ver = ver.rstrip('\\')
        if self.verbose:
            print('Using %WindowsSdkDir% {}'.format(dir))
            print('Using %WindowsSDKLibVersion% {}'.format(ver))
        return (dir, ver)

    def _get_msvc_native_toolchain_dir(self):
        assert self.toolchain_type == 'msvc'
        compiler_dir = os.path.dirname(self.compiler)
        target_dir = os.path.dirname(compiler_dir)
        host_name = os.path.basename(target_dir)
        host_name = host_name[4:].lower()
        return os.path.join(target_dir, host_name)

    def _get_visual_studio_environment(self):
        vctools = self._get_vctools_install_dir()
        winsdk, winsdkver = self._get_winsdk_dir()

        if not vctools and self.verbose:
            print('Unable to find VC tools installation directory.')
        if (not winsdk or not winsdkver) and self.verbose:
            print('Unable to find Windows SDK directory.')

        vcincludes = []
        vclibs = []
        sdkincludes = []
        sdklibs = []
        if vctools is not None:
            includes = [['ATLMFC', 'include'], ['include']]
            libs = [['ATLMFC', 'lib'], ['lib']]
            vcincludes = [os.path.join(vctools, *y) for y in includes]
            vclibs = [os.path.join(vctools, *y) for y in libs]
        if winsdk is not None:
            includes = [['include', winsdkver, 'ucrt'],
                        ['include', winsdkver, 'shared'],
                        ['include', winsdkver, 'um'],
                        ['include', winsdkver, 'winrt'],
                        ['include', winsdkver, 'cppwinrt']]
            libs = [['lib', winsdkver, 'ucrt'],
                    ['lib', winsdkver, 'um']]
            sdkincludes = [os.path.join(winsdk, *y) for y in includes]
            sdklibs = [os.path.join(winsdk, *y) for y in libs]

        includes = vcincludes + sdkincludes
        libs = vclibs + sdklibs
        libs = [os.path.join(x, self.msvc_arch_str) for x in libs]
        compileenv = None
        linkenv = None
        defaultenv = {}
        if sys.platform == 'win32':
            defaultenv = { x : os.environ[x] for x in
                          ['SystemDrive', 'SystemRoot', 'TMP', 'TEMP'] }
            # The directory to mspdbcore.dll needs to be in PATH, but this is
            # always in the native toolchain path, not the cross-toolchain
            # path.  So, for example, if we're using HostX64\x86 then we need
            # to add HostX64\x64 to the path, and if we're using HostX86\x64
            # then we need to add HostX86\x86 to the path.
            if self.toolchain_type == 'msvc':
                defaultenv['PATH'] = self._get_msvc_native_toolchain_dir()

        if includes:
            compileenv = {}
            compileenv['INCLUDE'] = os.pathsep.join(includes)
            compileenv.update(defaultenv)
        if libs:
            linkenv = {}
            linkenv['LIB'] = os.pathsep.join(libs)
            linkenv.update(defaultenv)
        return (compileenv, linkenv)

    def _ilk_file_name(self):
        if self.mode == 'link':
            return None
        return os.path.splitext(self.output)[0] + '.ilk'

    def _obj_file_name(self):
        if self.mode == 'compile':
            return self.output
        return os.path.splitext(self.output)[0] + '.obj'

    def _pdb_file_name(self):
        if self.mode == 'compile':
            return None
        return os.path.splitext(self.output)[0] + '.pdb'

    def _exe_file_name(self):
        if self.mode == 'compile':
            return None
        return self.output

    def _get_compilation_command(self):
        args = []

        args.append(self.compiler)
        if self.toolchain_type == 'clang-cl':
            args.append('-m' + self.arch)

        if self.opt == 'none':
            args.append('/Od')
        elif self.opt == 'basic':
            args.append('/O2')
        elif self.opt == 'lto':
            if self.toolchain_type == 'msvc':
                args.append('/GL')
                args.append('/Gw')
            else:
                args.append('-flto=thin')
        if self.nodefaultlib:
            args.append('/GS-')
            args.append('/GR-')
        args.append('/Z7')
        if self.toolchain_type == 'clang-cl':
            args.append('-Xclang')
            args.append('-fkeep-static-consts')
        args.append('/c')

        args.append('/Fo' + self._obj_file_name())
        args.append(self.input)
        input = os.path.basename(self.input)
        output = os.path.basename(self._obj_file_name())
        return ('compiling {0} -> {1}'.format(input, output),
                self.compile_env,
                args)

    def _get_link_command(self):
        args = []
        args.append(self.linker)
        args.append('/DEBUG:FULL')
        args.append('/INCREMENTAL:NO')
        if self.nodefaultlib:
            args.append('/nodefaultlib')
            args.append('/entry:main')
        args.append('/PDB:' + self._pdb_file_name())
        args.append('/OUT:' + self._exe_file_name())
        args.append(self._obj_file_name())

        input = os.path.basename(self._obj_file_name())
        output = os.path.basename(self._exe_file_name())
        return ('linking {0} -> {1}'.format(input, output),
                self.link_env,
                args)

    def build_commands(self):
        commands = []
        if self.mode == 'compile' or self.mode == 'compile-and-link':
            commands.append(self._get_compilation_command())
        if self.mode == 'link' or self.mode == 'compile-and-link':
            commands.append(self._get_link_command())
        return commands

    def output_files(self):
        outdir = os.path.dirname(self.output)
        file = os.path.basename(self.output)
        name, ext = os.path.splitext(file)

        outputs = []
        if self.mode == 'compile' or self.mode == 'compile-and-link':
            outputs.append(self._ilk_file_name())
            outputs.append(self._obj_file_name())
        if self.mode == 'link' or self.mode == 'compile-and-link':
            outputs.append(self._pdb_file_name())
            outputs.append(self._exe_file_name())

        return [x for x in outputs if x is not None]

class GccBuilder(Builder):
    def __init__(self, toolchain_type, args):
        Builder.__init__(self, toolchain_type, args)

    def build_commands(self):
        pass

    def output_files(self):
        pass

def indent(text, spaces):
    def prefixed_lines():
        prefix = ' ' * spaces
        for line in text.splitlines(True):
            yield prefix + line
    return ''.join(prefixed_lines())

def build(commands):
    global args
    for (status, env, child_args) in commands:
        print('\n\n')
        print(status)
        if args.verbose:
            print('  Command Line: ' + ' '.join(child_args))
            print('  Env:')
            print_environment(env)
        popen = subprocess.Popen(child_args,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 env=env,
                                 universal_newlines=True)
        stdout, stderr = popen.communicate()
        res = popen.wait()
        if res == -signal.SIGINT:
            raise KeyboardInterrupt
        print('  STDOUT:')
        print(indent(stdout, 4))
        if res != 0:
            print('  STDERR:')
            print(indent(stderr, 4))
            sys.exit(res)

def clean(files):
    global args
    for o in files:
        file = o if args.verbose else os.path.basename(o)
        print('Cleaning {0}'.format(file))
        try:
            if os.path.exists(o):
                os.remove(o)
                if args.verbose:
                    print('  The file was successfully cleaned.')
            elif args.verbose:
                print('  The file does not exist.')
        except:
            if args.verbose:
                print('  The file could not be removed.')

(toolchain_type, toolchain_path) = find_toolchain(args.compiler, args.tools_dir)
if not toolchain_path or not toolchain_type:
    print('Unable to find toolchain {0}'.format(args.compiler))
    sys.exit(1)

if args.verbose:
    print("Script Environment:")
    print_environment(os.environ)

args.compiler = toolchain_path
if not os.path.exists(args.compiler):
    raise ValueError('The toolchain {} does not exist.'.format(args.compiler))

if toolchain_type == 'msvc' or toolchain_type=='clang-cl':
    builder = MsvcBuilder(toolchain_type, args)
else:
    builder = GccBuilder(toolchain_type, args)

if args.clean:
    clean(builder.output_files())

cmds = builder.build_commands()

build(cmds)
