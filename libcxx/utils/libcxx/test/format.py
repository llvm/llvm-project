# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

import lit
import lit.formats
import os


def _validateModuleDependencies(modules):
    for m in modules:
        if m not in ("std", "std.compat"):
            raise RuntimeError(
                f"Invalid module dependency '{m}', only 'std' and 'std.compat' are valid"
            )


def _buildModule(test, litConfig, commands):
    tmpDir, tmpBase = lit.formats.standardlibrarytest._getTempPaths(test)
    execDir = os.path.dirname(test.getExecPath())
    substitutions = lit.TestRunner.getDefaultSubstitutions(test, tmpDir, tmpBase)

    substituted = lit.TestRunner.applySubstitutions(
        commands, substitutions, recursion_limit=test.config.recursiveExpansionLimit
    )
    (out, err, exitCode, _) = lit.TestRunner.executeScriptInternal(test, litConfig, tmpBase, substituted, execDir)
    if exitCode != 0:
        return lit.Test.Result(
            lit.Test.UNRESOLVED, "Failed to build module std for '{}':\n{}\n{}".format(test.getFilePath(), out, err)
        )


class LibcxxTest(lit.formats.StandardLibraryTest):
    def execute(self, test, litConfig):
        if test.config.unsupported:
            return lit.Test.Result(lit.Test.UNSUPPORTED, "Test is unsupported")

        # Parse any MODULE_DEPENDENCIES in the test file, and also handle
        # UNSUPPORTED/REQUIRES lines in the test.
        modules = []
        parsers = [
            lit.TestRunner.IntegratedTestKeywordParser(
                "MODULE_DEPENDENCIES:",
                lit.TestRunner.ParserKind.SPACE_LIST,
                initial_value=modules,
            )
        ]
        res = lit.TestRunner.parseIntegratedTestScript(test, additional_parsers=parsers, require_script=False)
        if isinstance(res, lit.Test.Result):
            return res

        # Build the modules if needed and tweak the compiler flags of the rest of the test so
        # it knows about the just-built modules.
        moduleCompileFlags = []
        if modules:
            _validateModuleDependencies(modules)

            # Make sure the std module is built before std.compat. Libc++'s
            # std.compat module depends on the std module. It is not
            # known whether the compiler expects the modules in the order of
            # their dependencies. However it's trivial to provide them in
            # that order.
            commands = [
                "mkdir -p %T",
                "%dbg(MODULE std) %{cxx} %{flags} %{compile_flags} "
                "-Wno-reserved-module-identifier -Wno-reserved-user-defined-literal "
                "--precompile -o %T/std.pcm -c %{module-dir}/std.cppm",
            ]
            res = _buildModule(test, litConfig, commands)
            if isinstance(res, lit.Test.Result):
                return res
            moduleCompileFlags.extend(["-fmodule-file=std=%T/std.pcm", "%T/std.pcm"])

            if "std.compat" in modules:
                commands = [
                    "mkdir -p %T",
                    "%dbg(MODULE std.compat) %{cxx} %{flags} %{compile_flags} "
                    "-Wno-reserved-module-identifier -Wno-reserved-user-defined-literal "
                    "-fmodule-file=std=%T/std.pcm " # The std.compat module imports std.
                    "--precompile -o %T/std.compat.pcm -c %{module-dir}/std.compat.cppm",
                ]
                res = _buildModule(test, litConfig, commands)
                if isinstance(res, lit.Test.Result):
                    return res
                moduleCompileFlags.extend(["-fmodule-file=std.compat=%T/std.compat.pcm", "%T/std.compat.pcm"])

            # Add compile flags required for the test to use the just-built modules
            test.config.substitutions = lit.formats.standardlibrarytest._appendToSubstitution(
                test.config.substitutions, "%{compile_flags}", " ".join(moduleCompileFlags)
            )

        return super().execute(test, litConfig)
