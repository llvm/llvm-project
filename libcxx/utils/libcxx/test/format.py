#===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##

import lit
import lit.formats
import os
import pipes
import re
import shutil
import subprocess

def _supportsVerify(config):
    """
    Determine whether clang-verify is supported by the given configuration.

    This is done by checking whether the %{cxx} substitution in that
    configuration supports certain compiler flags.
    """
    command = "%{{cxx}} -xc++ {} -Werror -fsyntax-only -Xclang -verify-ignore-unexpected".format(os.devnull)
    command = lit.TestRunner.applySubstitutions([command], config.substitutions,
                                                recursion_limit=config.recursiveExpansionLimit)[0]
    devNull = open(os.devnull, 'w')
    result = subprocess.call(command, shell=True, stdout=devNull, stderr=devNull)
    return result == 0

def _getTempPaths(test):
    """
    Return the values to use for the %T and %t substitutions, respectively.

    The difference between this and Lit's default behavior is that we guarantee
    that %T is a path unique to the test being run.
    """
    tmpDir, _ = lit.TestRunner.getTempPaths(test)
    _, testName = os.path.split(test.getExecPath())
    tmpDir = os.path.join(tmpDir, testName + '.dir')
    tmpBase = os.path.join(tmpDir, 't')
    return tmpDir, tmpBase

def _checkBaseSubstitutions(substitutions):
    substitutions = [s for (s, _) in substitutions]
    for s in ['%{cxx}', '%{compile_flags}', '%{link_flags}', '%{flags}', '%{exec}']:
        assert s in substitutions, "Required substitution {} was not provided".format(s)

def parseScript(test, preamble):
    """
    Extract the script from a test, with substitutions applied.

    Returns a list of commands ready to be executed.

    - test
        The lit.Test to parse.

    - preamble
        A list of commands to perform before any command in the test.
        These commands can contain unexpanded substitutions, but they
        must not be of the form 'RUN:' -- they must be proper commands
        once substituted.
    """

    # Get the default substitutions
    tmpDir, tmpBase = _getTempPaths(test)
    useExternalSh = True
    substitutions = lit.TestRunner.getDefaultSubstitutions(test, tmpDir, tmpBase,
                                                           normalize_slashes=useExternalSh)

    # Check base substitutions and add the %{build} and %{run} convenience substitutions
    _checkBaseSubstitutions(substitutions)
    substitutions.append(('%{build}', '%{cxx} %s %{flags} %{compile_flags} %{link_flags} -o %t.exe'))
    substitutions.append(('%{run}', '%{exec} %t.exe'))

    # Parse the test file, including custom directives
    additionalCompileFlags = []
    fileDependencies = []
    parsers = [
        lit.TestRunner.IntegratedTestKeywordParser('FILE_DEPENDENCIES:',
                                                   lit.TestRunner.ParserKind.LIST,
                                                   initial_value=fileDependencies),
        lit.TestRunner.IntegratedTestKeywordParser('ADDITIONAL_COMPILE_FLAGS:',
                                                   lit.TestRunner.ParserKind.LIST,
                                                   initial_value=additionalCompileFlags)
    ]

    scriptInTest = lit.TestRunner.parseIntegratedTestScript(test, additional_parsers=parsers,
                                                            require_script=not preamble)
    if isinstance(scriptInTest, lit.Test.Result):
        return scriptInTest

    script = []

    # For each file dependency in FILE_DEPENDENCIES, inject a command to copy
    # that file to the execution directory. Execute the copy from %S to allow
    # relative paths from the test directory.
    for dep in fileDependencies:
        script += ['%dbg(SETUP) cd %S && cp {} %T'.format(dep)]
    script += preamble
    script += scriptInTest

    # Add compile flags specified with ADDITIONAL_COMPILE_FLAGS.
    substitutions = [(s, x + ' ' + ' '.join(additionalCompileFlags)) if s == '%{compile_flags}'
                            else (s, x) for (s, x) in substitutions]

    # Perform substitutions in the script itself.
    script = lit.TestRunner.applySubstitutions(script, substitutions,
                                               recursion_limit=test.config.recursiveExpansionLimit)

    return script


class CxxStandardLibraryTest(lit.formats.TestFormat):
    """
    Lit test format for the C++ Standard Library conformance test suite.

    This test format is based on top of the ShTest format -- it basically
    creates a shell script performing the right operations (compile/link/run)
    based on the extension of the test file it encounters. It supports files
    with the following extensions:

    FOO.pass.cpp            - Compiles, links and runs successfully
    FOO.pass.mm             - Same as .pass.cpp, but for Objective-C++
    FOO.run.fail.cpp        - Compiles and links successfully, but fails at runtime

    FOO.compile.pass.cpp    - Compiles successfully, link and run not attempted
    FOO.compile.fail.cpp    - Does not compile successfully

    FOO.link.pass.cpp       - Compiles and links successfully, run not attempted
    FOO.link.fail.cpp       - Compiles successfully, but fails to link

    FOO.sh.<anything>       - A builtin Lit Shell test

    FOO.verify.cpp          - Compiles with clang-verify. This type of test is
                              automatically marked as UNSUPPORTED if the compiler
                              does not support Clang-verify.

    FOO.fail.cpp            - Compiled with clang-verify if clang-verify is
                              supported, and equivalent to a .compile.fail.cpp
                              test otherwise. This is supported only for backwards
                              compatibility with the test suite.


    Substitution requirements
    ===============================
    The test format operates by assuming that each test's configuration provides
    the following substitutions, which it will reuse in the shell scripts it
    constructs:
        %{cxx}           - A command that can be used to invoke the compiler
        %{compile_flags} - Flags to use when compiling a test case
        %{link_flags}    - Flags to use when linking a test case
        %{flags}         - Flags to use either when compiling or linking a test case
        %{exec}          - A command to prefix the execution of executables

    Note that when building an executable (as opposed to only compiling a source
    file), all three of %{flags}, %{compile_flags} and %{link_flags} will be used
    in the same command line. In other words, the test format doesn't perform
    separate compilation and linking steps in this case.


    Additional supported directives
    ===============================
    In addition to everything that's supported in Lit ShTests, this test format
    also understands the following directives inside test files:

        // FILE_DEPENDENCIES: file, directory, /path/to/file

            This directive expresses that the test requires the provided files
            or directories in order to run. An example is a test that requires
            some test input stored in a data file. When a test file contains
            such a directive, this test format will collect them and copy them
            to the directory represented by %T. The intent is that %T contains
            all the inputs necessary to run the test, such that e.g. execution
            on a remote host can be done by simply copying %T to the host.

        // ADDITIONAL_COMPILE_FLAGS: flag1, flag2, flag3

            This directive will cause the provided flags to be added to the
            %{compile_flags} substitution for the test that contains it. This
            allows adding special compilation flags without having to use a
            .sh.cpp test, which would be more powerful but perhaps overkill.


    Additional provided substitutions and features
    ==============================================
    The test format will define the following substitutions for use inside tests:

        %{build}
            Expands to a command-line that builds the current source
            file with the %{flags}, %{compile_flags} and %{link_flags}
            substitutions, and that produces an executable named %t.exe.

        %{run}
            Equivalent to `%{exec} %t.exe`. This is intended to be used
            in conjunction with the %{build} substitution.
    """
    def getTestsInDirectory(self, testSuite, pathInSuite, litConfig, localConfig):
        SUPPORTED_SUFFIXES = ['[.]pass[.]cpp$', '[.]pass[.]mm$', '[.]run[.]fail[.]cpp$',
                              '[.]compile[.]pass[.]cpp$', '[.]compile[.]fail[.]cpp$',
                              '[.]link[.]pass[.]cpp$', '[.]link[.]fail[.]cpp$',
                              '[.]sh[.][^.]+$',
                              '[.]verify[.]cpp$',
                              '[.]fail[.]cpp$']
        sourcePath = testSuite.getSourcePath(pathInSuite)
        for filename in os.listdir(sourcePath):
            # Ignore dot files and excluded tests.
            if filename.startswith('.') or filename in localConfig.excludes:
                continue

            filepath = os.path.join(sourcePath, filename)
            if not os.path.isdir(filepath):
                if any([re.search(ext, filename) for ext in SUPPORTED_SUFFIXES]):
                    yield lit.Test.Test(testSuite, pathInSuite + (filename,), localConfig)

    def _disableWithModules(self, test):
        with open(test.getSourcePath(), 'rb') as f:
            contents = f.read()
        return b'#define _LIBCPP_ASSERT' in contents

    def execute(self, test, litConfig):
        VERIFY_FLAGS = '-Xclang -verify -Xclang -verify-ignore-unexpected=note -ferror-limit=0'
        supportsVerify = _supportsVerify(test.config)
        filename = test.path_in_suite[-1]

        # TODO(ldionne): We currently disable tests that re-define _LIBCPP_ASSERT
        #                when we run with modules enabled. Instead, we should
        #                split the part that does a death test outside of the
        #                test, and only disable that part when modules are
        #                enabled.
        if '-fmodules' in test.config.available_features and self._disableWithModules(test):
            return lit.Test.Result(lit.Test.UNSUPPORTED, 'Test {} is unsupported when modules are enabled')

        if re.search('[.]sh[.][^.]+$', filename):
            steps = [ ] # The steps are already in the script
            return self._executeShTest(test, litConfig, steps)
        elif filename.endswith('.compile.pass.cpp'):
            steps = [
                "%dbg(COMPILED WITH) %{cxx} %s %{flags} %{compile_flags} -fsyntax-only"
            ]
            return self._executeShTest(test, litConfig, steps)
        elif filename.endswith('.compile.fail.cpp'):
            steps = [
                "%dbg(COMPILED WITH) ! %{cxx} %s %{flags} %{compile_flags} -fsyntax-only"
            ]
            return self._executeShTest(test, litConfig, steps)
        elif filename.endswith('.link.pass.cpp'):
            steps = [
                "%dbg(COMPILED WITH) %{cxx} %s %{flags} %{compile_flags} %{link_flags} -o %t.exe"
            ]
            return self._executeShTest(test, litConfig, steps)
        elif filename.endswith('.link.fail.cpp'):
            steps = [
                "%dbg(COMPILED WITH) %{cxx} %s %{flags} %{compile_flags} -c -o %t.o",
                "%dbg(LINKED WITH) ! %{cxx} %t.o %{flags} %{link_flags} -o %t.exe"
            ]
            return self._executeShTest(test, litConfig, steps)
        elif filename.endswith('.run.fail.cpp'):
            steps = [
                "%dbg(COMPILED WITH) %{cxx} %s %{flags} %{compile_flags} %{link_flags} -o %t.exe",
                "%dbg(EXECUTED AS) %{exec} ! %t.exe"
            ]
            return self._executeShTest(test, litConfig, steps)
        elif filename.endswith('.verify.cpp'):
            if not supportsVerify:
                return lit.Test.Result(lit.Test.UNSUPPORTED,
                    "Test {} requires support for Clang-verify, which isn't supported by the compiler".format(test.getFullName()))
            steps = [
                # Note: Use -Wno-error to make sure all diagnostics are not treated as errors,
                #       which doesn't make sense for clang-verify tests.
                "%dbg(COMPILED WITH) %{{cxx}} %s %{{flags}} %{{compile_flags}} -fsyntax-only -Wno-error {}".format(VERIFY_FLAGS)
            ]
            return self._executeShTest(test, litConfig, steps)
        # Make sure to check these ones last, since they will match other
        # suffixes above too.
        elif filename.endswith('.pass.cpp') or filename.endswith('.pass.mm'):
            steps = [
                "%dbg(COMPILED WITH) %{cxx} %s %{flags} %{compile_flags} %{link_flags} -o %t.exe",
                "%dbg(EXECUTED AS) %{exec} %t.exe"
            ]
            return self._executeShTest(test, litConfig, steps)
        # This is like a .verify.cpp test when clang-verify is supported,
        # otherwise it's like a .compile.fail.cpp test. This is only provided
        # for backwards compatibility with the test suite.
        elif filename.endswith('.fail.cpp'):
            if supportsVerify:
                steps = [
                    "%dbg(COMPILED WITH) %{{cxx}} %s %{{flags}} %{{compile_flags}} -fsyntax-only -Wno-error {}".format(VERIFY_FLAGS)
                ]
            else:
                steps = [
                    "%dbg(COMPILED WITH) ! %{cxx} %s %{flags} %{compile_flags} -fsyntax-only"
                ]
            return self._executeShTest(test, litConfig, steps)
        else:
            return lit.Test.Result(lit.Test.UNRESOLVED, "Unknown test suffix for '{}'".format(filename))

    # Utility function to add compile flags in lit.local.cfg files.
    def addCompileFlags(self, config, *flags):
        string = ' '.join(flags)
        config.substitutions = [(s, x + ' ' + string) if s == '%{compile_flags}' else (s, x) for (s, x) in config.substitutions]

    def _executeShTest(self, test, litConfig, steps):
        if test.config.unsupported:
            return lit.Test.Result(lit.Test.UNSUPPORTED, 'Test is unsupported')

        script = parseScript(test, steps)
        if isinstance(script, lit.Test.Result):
            return script

        if litConfig.noExecute:
            return lit.Test.Result(lit.Test.XFAIL if test.isExpectedToFail() else lit.Test.PASS)
        else:
            _, tmpBase = _getTempPaths(test)
            useExternalSh = True
            return lit.TestRunner._runShTest(test, litConfig, useExternalSh, script, tmpBase)
