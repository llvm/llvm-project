#===----------------------------------------------------------------------===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is dual licensed under the MIT and the University of Illinois Open
# Source Licenses. See LICENSE.TXT for details.
#
#===----------------------------------------------------------------------===##

import os
import lit.util
import libcxx.util


class CXXCompiler(object):
    def __init__(self, path, flags=None, compile_flags=None, link_flags=None,
                 use_ccache=False):
        self.path = path
        self.flags = list(flags or [])
        self.compile_flags = list(compile_flags or [])
        self.link_flags = list(link_flags or [])
        self.use_ccache = use_ccache
        self.type = None
        self.version = None
        self._initTypeAndVersion()

    def _initTypeAndVersion(self):
        # Get compiler type and version
        macros = self.dumpMacros()
        if macros is None:
            return
        compiler_type = None
        major_ver = minor_ver = patchlevel = None
        if '__clang__' in macros.keys():
            compiler_type = 'clang'
            # Treat apple's llvm fork differently.
            if '__apple_build_version__' in macros.keys():
                compiler_type = 'apple-clang'
            major_ver = macros['__clang_major__']
            minor_ver = macros['__clang_minor__']
            patchlevel = macros['__clang_patchlevel__']
        elif '__GNUC__' in macros.keys():
            compiler_type = 'gcc'
            major_ver = macros['__GNUC__']
            minor_ver = macros['__GNUC_MINOR__']
            patchlevel = macros['__GNUC_PATCHLEVEL__']
        self.type = compiler_type
        self.version = (major_ver, minor_ver, patchlevel)

    def _basicCmd(self, source_files, out, is_link=False, input_is_cxx=False):
        cmd = []
        if self.use_ccache and not is_link:
            cmd += ['ccache']
        cmd += [self.path]
        if out is not None:
            cmd += ['-o', out]
        if input_is_cxx:
            cmd += ['-x', 'c++']
        if isinstance(source_files, list):
            cmd += source_files
        elif isinstance(source_files, str):
            cmd += [source_files]
        else:
            raise TypeError('source_files must be a string or list')
        return cmd

    def preprocessCmd(self, source_files, out=None, flags=[]):
        cmd = self._basicCmd(source_files, out, input_is_cxx=True) + ['-E']
        cmd += self.flags + self.compile_flags + flags
        return cmd

    def compileCmd(self, source_files, out=None, flags=[]):
        cmd = self._basicCmd(source_files, out, input_is_cxx=True) + ['-c']
        cmd += self.flags + self.compile_flags + flags
        return cmd

    def linkCmd(self, source_files, out=None, flags=[]):
        cmd = self._basicCmd(source_files, out, is_link=True)
        cmd += self.flags + self.link_flags + flags
        return cmd

    def compileLinkCmd(self, source_files, out=None, flags=[]):
        cmd = self._basicCmd(source_files, out, is_link=True)
        cmd += self.flags + self.compile_flags + self.link_flags + flags
        return cmd

    def preprocess(self, source_files, out=None, flags=[], env=None, cwd=None):
        cmd = self.preprocessCmd(source_files, out, flags)
        out, err, rc = lit.util.executeCommand(cmd, env=env, cwd=cwd)
        return cmd, out, err, rc

    def compile(self, source_files, out=None, flags=[], env=None, cwd=None):
        cmd = self.compileCmd(source_files, out, flags)
        out, err, rc = lit.util.executeCommand(cmd, env=env, cwd=cwd)
        return cmd, out, err, rc

    def link(self, source_files, out=None, flags=[], env=None, cwd=None):
        cmd = self.linkCmd(source_files, out, flags)
        out, err, rc = lit.util.executeCommand(cmd, env=env, cwd=cwd)
        return cmd, out, err, rc

    def compileLink(self, source_files, out=None, flags=[], env=None,
                    cwd=None):
        cmd = self.compileLinkCmd(source_files, out, flags)
        out, err, rc = lit.util.executeCommand(cmd, env=env, cwd=cwd)
        return cmd, out, err, rc

    def compileLinkTwoSteps(self, source_file, out=None, object_file=None,
                            flags=[], env=None, cwd=None):
        if not isinstance(source_file, str):
            raise TypeError('This function only accepts a single input file')
        if object_file is None:
            # Create, use and delete a temporary object file if none is given.
            with_fn = lambda: libcxx.util.guardedTempFilename(suffix='.o')
        else:
            # Otherwise wrap the filename in a context manager function.
            with_fn = lambda: libcxx.util.nullContext(object_file)
        with with_fn() as object_file:
            cc_cmd, cc_stdout, cc_stderr, rc = self.compile(
                    source_file, object_file, flags=flags, env=env, cwd=cwd)
            if rc != 0:
                return cc_cmd, cc_stdout, cc_stderr, rc

            link_cmd, link_stdout, link_stderr, rc = self.link(
                    object_file, out=out, flags=flags, env=env, cwd=cwd)
            return (cc_cmd + ['&&'] + link_cmd, cc_stdout + link_stdout,
                    cc_stderr + link_stderr, rc)

    def dumpMacros(self, source_files=None, flags=[], env=None, cwd=None):
        if source_files is None:
            source_files = os.devnull
        flags = ['-dM'] + flags
        cmd, out, err, rc = self.preprocess(source_files, flags=flags, env=env,
                                            cwd=cwd)
        if rc != 0:
            return None
        parsed_macros = {}
        lines = [l.strip() for l in out.split('\n') if l.strip()]
        for l in lines:
            assert l.startswith('#define ')
            l = l[len('#define '):]
            macro, _, value = l.partition(' ')
            parsed_macros[macro] = value
        return parsed_macros

    def getTriple(self):
        cmd = [self.path] + self.flags + ['-dumpmachine']
        return lit.util.capture(cmd).strip()

    def hasCompileFlag(self, flag):
        if isinstance(flag, list):
            flags = list(flag)
        else:
            flags = [flag]
        # Add -Werror to ensure that an unrecognized flag causes a non-zero
        # exit code. -Werror is supported on all known compiler types.
        if self.type is not None:
            flags += ['-Werror']
        cmd, out, err, rc = self.compile(os.devnull, out=os.devnull,
                                         flags=flags)
        return rc == 0

    def addCompileFlagIfSupported(self, flag):
        if isinstance(flag, list):
            flags = list(flag)
        else:
            flags = [flag]
        if self.hasCompileFlag(flags):
            self.compile_flags += flags
            return True
        else:
            return False

    def addWarningFlagIfSupported(self, flag):
        """
        addWarningFlagIfSupported - Add a warning flag if the compiler
        supports it. Unlike addCompileFlagIfSupported, this function detects
        when "-Wno-<warning>" flags are unsupported. If flag is a
        "-Wno-<warning>" GCC will not emit an unknown option diagnostic unless
        another error is triggered during compilation.
        """
        assert isinstance(flag, str)
        if not flag.startswith('-Wno-'):
            return self.addCompileFlagIfSupported(flag)
        flags = ['-Werror', flag]
        cmd = self.compileCmd('-', os.devnull, flags)
        # Remove '-v' because it will cause the command line invocation
        # to be printed as part of the error output.
        # TODO(EricWF): Are there other flags we need to worry about?
        if '-v' in cmd:
            cmd.remove('-v')
        out, err, rc = lit.util.executeCommand(cmd, input='#error\n')
        assert rc != 0
        if flag in err:
            return False
        self.compile_flags += [flag]
        return True
