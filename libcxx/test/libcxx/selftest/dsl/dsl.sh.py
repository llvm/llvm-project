#===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##

# XFAIL: LIBCXX-WINDOWS-FIXME

# Note: We prepend arguments with 'x' to avoid thinking there are too few
#       arguments in case an argument is an empty string.
# RUN: %{python} %s x%S x%T x%{substitutions}

import base64
import copy
import os
import pickle
import platform
import subprocess
import sys
import unittest
from os.path import dirname

# Allow importing 'lit' and the 'libcxx' module. Make sure we put the lit
# path first so we don't find any system-installed version.
monorepoRoot = dirname(dirname(dirname(dirname(dirname(dirname(__file__))))))
sys.path = [os.path.join(monorepoRoot, 'libcxx', 'utils'),
            os.path.join(monorepoRoot, 'llvm', 'utils', 'lit')] + sys.path
import libcxx.test.dsl as dsl
import lit.LitConfig
import lit.util

# Steal some parameters from the config running this test so that we can
# bootstrap our own TestingConfig.
args = list(map(lambda s: s[1:], sys.argv[1:8])) # Remove the leading 'x'
SOURCE_ROOT, EXEC_PATH, SUBSTITUTIONS = args
sys.argv[1:8] = []

# Decode the substitutions.
SUBSTITUTIONS = pickle.loads(base64.b64decode(SUBSTITUTIONS))
for s, sub in SUBSTITUTIONS:
    print("Substitution '{}' is '{}'".format(s, sub))

class SetupConfigs(unittest.TestCase):
    """
    Base class for the tests below -- it creates a fake TestingConfig.
    """
    def setUp(self):
        """
        Create a fake TestingConfig that can be populated however we wish for
        the purpose of running unit tests below. We pre-populate it with the
        minimum required substitutions.
        """
        self.litConfig = lit.LitConfig.LitConfig(
            progname='lit',
            path=[],
            quiet=False,
            useValgrind=False,
            valgrindLeakCheck=False,
            valgrindArgs=[],
            noExecute=False,
            debug=False,
            isWindows=platform.system() == 'Windows',
            params={})

        self.config = lit.TestingConfig.TestingConfig.fromdefaults(self.litConfig)
        self.config.test_source_root = SOURCE_ROOT
        self.config.test_exec_root = EXEC_PATH
        self.config.recursiveExpansionLimit = 10
        self.config.substitutions = copy.deepcopy(SUBSTITUTIONS)

    def getSubstitution(self, substitution):
        """
        Return a given substitution from the TestingConfig. It is an error if
        there is no such substitution.
        """
        found = [x for (s, x) in self.config.substitutions if s == substitution]
        assert len(found) == 1
        return found[0]


def findIndex(list, pred):
    """Finds the index of the first element satisfying 'pred' in a list, or
       'len(list)' if there is no such element."""
    index = 0
    for x in list:
        if pred(x):
            break
        else:
            index += 1
    return index


class TestHasCompileFlag(SetupConfigs):
    """
    Tests for libcxx.test.dsl.hasCompileFlag
    """
    def test_no_flag_should_work(self):
        self.assertTrue(dsl.hasCompileFlag(self.config, ''))

    def test_flag_exists(self):
        self.assertTrue(dsl.hasCompileFlag(self.config, '-O1'))

    def test_nonexistent_flag(self):
        self.assertFalse(dsl.hasCompileFlag(self.config, '-this_is_not_a_flag_any_compiler_has'))

    def test_multiple_flags(self):
        self.assertTrue(dsl.hasCompileFlag(self.config, '-O1 -Dhello'))


class TestSourceBuilds(SetupConfigs):
    """
    Tests for libcxx.test.dsl.sourceBuilds
    """
    def test_valid_program_builds(self):
        source = """int main(int, char**) { return 0; }"""
        self.assertTrue(dsl.sourceBuilds(self.config, source))

    def test_compilation_error_fails(self):
        source = """int main(int, char**) { this does not compile }"""
        self.assertFalse(dsl.sourceBuilds(self.config, source))

    def test_link_error_fails(self):
        source = """extern void this_isnt_defined_anywhere();
                    int main(int, char**) { this_isnt_defined_anywhere(); return 0; }"""
        self.assertFalse(dsl.sourceBuilds(self.config, source))


class TestProgramOutput(SetupConfigs):
    """
    Tests for libcxx.test.dsl.programOutput
    """
    def test_valid_program_returns_output(self):
        source = """
        #include <cstdio>
        int main(int, char**) { std::printf("FOOBAR"); return 0; }
        """
        self.assertEqual(dsl.programOutput(self.config, source), "FOOBAR")

    def test_valid_program_returns_output_newline_handling(self):
        source = """
        #include <cstdio>
        int main(int, char**) { std::printf("FOOBAR\\n"); return 0; }
        """
        self.assertEqual(dsl.programOutput(self.config, source), "FOOBAR\n")

    def test_valid_program_returns_no_output(self):
        source = """
        int main(int, char**) { return 0; }
        """
        self.assertEqual(dsl.programOutput(self.config, source), "")

    def test_program_that_fails_to_run_raises_runtime_error(self):
        # The program compiles, but exits with an error
        source = """
        int main(int, char**) { return 1; }
        """
        self.assertRaises(dsl.ConfigurationRuntimeError, lambda: dsl.programOutput(self.config, source))

    def test_program_that_fails_to_compile_raises_compilation_error(self):
        # The program doesn't compile
        source = """
        int main(int, char**) { this doesnt compile }
        """
        self.assertRaises(dsl.ConfigurationCompilationError, lambda: dsl.programOutput(self.config, source))

    def test_pass_arguments_to_program(self):
        source = """
        #include <cassert>
        #include <string>
        int main(int argc, char** argv) {
            assert(argc == 3);
            assert(argv[1] == std::string("first-argument"));
            assert(argv[2] == std::string("second-argument"));
            return 0;
        }
        """
        args = ["first-argument", "second-argument"]
        self.assertEqual(dsl.programOutput(self.config, source, args=args), "")

    def test_caching_is_not_too_aggressive(self):
        # Run a program, then change the substitutions and run it again.
        # Make sure the program is run the second time and the right result
        # is given, to ensure we're not incorrectly caching the result of the
        # first program run.
        source = """
        #include <cstdio>
        int main(int, char**) {
            std::printf("MACRO=%u\\n", MACRO);
            return 0;
        }
        """
        compileFlagsIndex = findIndex(self.config.substitutions, lambda x: x[0] == '%{compile_flags}')
        compileFlags = self.config.substitutions[compileFlagsIndex][1]

        self.config.substitutions[compileFlagsIndex] = ('%{compile_flags}',  compileFlags + ' -DMACRO=1')
        output1 = dsl.programOutput(self.config, source)
        self.assertEqual(output1, "MACRO=1\n")

        self.config.substitutions[compileFlagsIndex] = ('%{compile_flags}',  compileFlags + ' -DMACRO=2')
        output2 = dsl.programOutput(self.config, source)
        self.assertEqual(output2, "MACRO=2\n")

    def test_program_stderr_is_not_conflated_with_stdout(self):
        # Run a program that produces stdout output and stderr output too, making
        # sure the stderr output does not pollute the stdout output.
        source = """
        #include <cstdio>
        int main(int, char**) {
            std::fprintf(stdout, "STDOUT-OUTPUT");
            std::fprintf(stderr, "STDERR-OUTPUT");
            return 0;
        }
        """
        self.assertEqual(dsl.programOutput(self.config, source), "STDOUT-OUTPUT")


class TestProgramSucceeds(SetupConfigs):
    """
    Tests for libcxx.test.dsl.programSucceeds
    """
    def test_success(self):
        source = """
        int main(int, char**) { return 0; }
        """
        self.assertTrue(dsl.programSucceeds(self.config, source))

    def test_failure(self):
        source = """
        int main(int, char**) { return 1; }
        """
        self.assertFalse(dsl.programSucceeds(self.config, source))

    def test_compile_failure(self):
        source = """
        this does not compile
        """
        self.assertRaises(dsl.ConfigurationCompilationError, lambda: dsl.programSucceeds(self.config, source))

class TestHasLocale(SetupConfigs):
    """
    Tests for libcxx.test.dsl.hasLocale
    """
    def test_doesnt_explode(self):
        # It's really hard to test that a system has a given locale, so at least
        # make sure we don't explode when we try to check it.
        try:
            dsl.hasAnyLocale(self.config, ['en_US.UTF-8'])
        except subprocess.CalledProcessError:
            self.fail("checking for hasLocale should not explode")

    def test_nonexistent_locale(self):
        self.assertFalse(dsl.hasAnyLocale(self.config, ['for_sure_this_is_not_an_existing_locale']))

    def test_localization_program_doesnt_compile(self):
        compilerIndex = findIndex(self.config.substitutions, lambda x: x[0] == '%{cxx}')
        self.config.substitutions[compilerIndex] = ('%{cxx}', 'this-is-certainly-not-a-valid-compiler!!')
        self.assertRaises(dsl.ConfigurationCompilationError, lambda: dsl.hasAnyLocale(self.config, ['en_US.UTF-8']))


class TestCompilerMacros(SetupConfigs):
    """
    Tests for libcxx.test.dsl.compilerMacros
    """
    def test_basic(self):
        macros = dsl.compilerMacros(self.config)
        self.assertIsInstance(macros, dict)
        self.assertGreater(len(macros), 0)
        for (k, v) in macros.items():
            self.assertIsInstance(k, str)
            self.assertIsInstance(v, str)

    def test_no_flag(self):
        macros = dsl.compilerMacros(self.config)
        self.assertIn('__cplusplus', macros.keys())

    def test_empty_flag(self):
        macros = dsl.compilerMacros(self.config, '')
        self.assertIn('__cplusplus', macros.keys())

    def test_with_flag(self):
        macros = dsl.compilerMacros(self.config, '-DFOO=3')
        self.assertIn('__cplusplus', macros.keys())
        self.assertEqual(macros['FOO'], '3')

    def test_with_flags(self):
        macros = dsl.compilerMacros(self.config, '-DFOO=3 -DBAR=hello')
        self.assertIn('__cplusplus', macros.keys())
        self.assertEqual(macros['FOO'], '3')
        self.assertEqual(macros['BAR'], 'hello')


class TestFeatureTestMacros(SetupConfigs):
    """
    Tests for libcxx.test.dsl.featureTestMacros
    """
    def test_basic(self):
        macros = dsl.featureTestMacros(self.config)
        self.assertIsInstance(macros, dict)
        self.assertGreater(len(macros), 0)
        for (k, v) in macros.items():
            self.assertIsInstance(k, str)
            self.assertIsInstance(v, int)


class TestFeature(SetupConfigs):
    """
    Tests for libcxx.test.dsl.Feature
    """
    def test_trivial(self):
        feature = dsl.Feature(name='name')
        origSubstitutions = copy.deepcopy(self.config.substitutions)
        actions = feature.getActions(self.config)
        self.assertTrue(len(actions) == 1)
        for a in actions:
            a.applyTo(self.config)
        self.assertEqual(origSubstitutions, self.config.substitutions)
        self.assertIn('name', self.config.available_features)

    def test_name_can_be_a_callable(self):
        feature = dsl.Feature(name=lambda cfg: 'name')
        for a in feature.getActions(self.config):
            a.applyTo(self.config)
        self.assertIn('name', self.config.available_features)

    def test_name_is_not_a_string_1(self):
        feature = dsl.Feature(name=None)
        self.assertRaises(ValueError, lambda: feature.getActions(self.config))
        self.assertRaises(ValueError, lambda: feature.pretty(self.config))

    def test_name_is_not_a_string_2(self):
        feature = dsl.Feature(name=lambda cfg: None)
        self.assertRaises(ValueError, lambda: feature.getActions(self.config))
        self.assertRaises(ValueError, lambda: feature.pretty(self.config))

    def test_adding_action(self):
        feature = dsl.Feature(name='name', actions=[dsl.AddCompileFlag('-std=c++03')])
        origLinkFlags = copy.deepcopy(self.getSubstitution('%{link_flags}'))
        for a in feature.getActions(self.config):
            a.applyTo(self.config)
        self.assertIn('name', self.config.available_features)
        self.assertIn('-std=c++03', self.getSubstitution('%{compile_flags}'))
        self.assertEqual(origLinkFlags, self.getSubstitution('%{link_flags}'))

    def test_actions_can_be_a_callable(self):
        feature = dsl.Feature(name='name',
                              actions=lambda cfg: (
                                self.assertIs(self.config, cfg),
                                [dsl.AddCompileFlag('-std=c++03')]
                              )[1])
        for a in feature.getActions(self.config):
            a.applyTo(self.config)
        self.assertIn('-std=c++03', self.getSubstitution('%{compile_flags}'))

    def test_unsupported_feature(self):
        feature = dsl.Feature(name='name', when=lambda _: False)
        self.assertEqual(feature.getActions(self.config), [])

    def test_is_supported_gets_passed_the_config(self):
        feature = dsl.Feature(name='name', when=lambda cfg: (self.assertIs(self.config, cfg), True)[1])
        self.assertEqual(len(feature.getActions(self.config)), 1)


def _throw():
    raise ValueError()

class TestParameter(SetupConfigs):
    """
    Tests for libcxx.test.dsl.Parameter
    """
    def test_empty_name_should_blow_up(self):
        self.assertRaises(ValueError, lambda: dsl.Parameter(name='', choices=['c++03'], type=str, help='', actions=lambda _: []))

    def test_empty_choices_should_blow_up(self):
        self.assertRaises(ValueError, lambda: dsl.Parameter(name='std', choices=[], type=str, help='', actions=lambda _: []))

    def test_no_choices_is_ok(self):
        param = dsl.Parameter(name='triple', type=str, help='', actions=lambda _: [])
        self.assertEqual(param.name, 'triple')

    def test_name_is_set_correctly(self):
        param = dsl.Parameter(name='std', choices=['c++03'], type=str, help='', actions=lambda _: [])
        self.assertEqual(param.name, 'std')

    def test_no_value_provided_and_no_default_value(self):
        param = dsl.Parameter(name='std', choices=['c++03'], type=str, help='', actions=lambda _: [])
        self.assertRaises(ValueError, lambda: param.getActions(self.config, self.litConfig.params))

    def test_no_value_provided_and_default_value(self):
        param = dsl.Parameter(name='std', choices=['c++03'], type=str, help='', default='c++03',
                              actions=lambda std: [dsl.AddFeature(std)])
        for a in param.getActions(self.config, self.litConfig.params):
            a.applyTo(self.config)
        self.assertIn('c++03', self.config.available_features)

    def test_value_provided_on_command_line_and_no_default_value(self):
        self.litConfig.params['std'] = 'c++03'
        param = dsl.Parameter(name='std', choices=['c++03'], type=str, help='',
                              actions=lambda std: [dsl.AddFeature(std)])
        for a in param.getActions(self.config, self.litConfig.params):
            a.applyTo(self.config)
        self.assertIn('c++03', self.config.available_features)

    def test_value_provided_on_command_line_and_default_value(self):
        """The value provided on the command line should override the default value"""
        self.litConfig.params['std'] = 'c++11'
        param = dsl.Parameter(name='std', choices=['c++03', 'c++11'], type=str, default='c++03', help='',
                              actions=lambda std: [dsl.AddFeature(std)])
        for a in param.getActions(self.config, self.litConfig.params):
            a.applyTo(self.config)
        self.assertIn('c++11', self.config.available_features)
        self.assertNotIn('c++03', self.config.available_features)

    def test_value_provided_in_config_and_default_value(self):
        """The value provided in the config should override the default value"""
        self.config.std ='c++11'
        param = dsl.Parameter(name='std', choices=['c++03', 'c++11'], type=str, default='c++03', help='',
                              actions=lambda std: [dsl.AddFeature(std)])
        for a in param.getActions(self.config, self.litConfig.params):
            a.applyTo(self.config)
        self.assertIn('c++11', self.config.available_features)
        self.assertNotIn('c++03', self.config.available_features)

    def test_value_provided_in_config_and_on_command_line(self):
        """The value on the command line should override the one in the config"""
        self.config.std = 'c++11'
        self.litConfig.params['std'] = 'c++03'
        param = dsl.Parameter(name='std', choices=['c++03', 'c++11'], type=str, help='',
                              actions=lambda std: [dsl.AddFeature(std)])
        for a in param.getActions(self.config, self.litConfig.params):
            a.applyTo(self.config)
        self.assertIn('c++03', self.config.available_features)
        self.assertNotIn('c++11', self.config.available_features)

    def test_no_actions(self):
        self.litConfig.params['std'] = 'c++03'
        param = dsl.Parameter(name='std', choices=['c++03'], type=str, help='',
                              actions=lambda _: [])
        actions = param.getActions(self.config, self.litConfig.params)
        self.assertEqual(actions, [])

    def test_boolean_value_parsed_from_trueish_string_parameter(self):
        self.litConfig.params['enable_exceptions'] = "True"
        param = dsl.Parameter(name='enable_exceptions', choices=[True, False], type=bool, help='',
                              actions=lambda exceptions: [] if exceptions else _throw())
        self.assertEqual(param.getActions(self.config, self.litConfig.params), [])

    def test_boolean_value_from_true_boolean_parameter(self):
        self.litConfig.params['enable_exceptions'] = True
        param = dsl.Parameter(name='enable_exceptions', choices=[True, False], type=bool, help='',
                              actions=lambda exceptions: [] if exceptions else _throw())
        self.assertEqual(param.getActions(self.config, self.litConfig.params), [])

    def test_boolean_value_parsed_from_falseish_string_parameter(self):
        self.litConfig.params['enable_exceptions'] = "False"
        param = dsl.Parameter(name='enable_exceptions', choices=[True, False], type=bool, help='',
                              actions=lambda exceptions: [] if exceptions else [dsl.AddFeature("-fno-exceptions")])
        for a in param.getActions(self.config, self.litConfig.params):
            a.applyTo(self.config)
        self.assertIn('-fno-exceptions', self.config.available_features)

    def test_boolean_value_from_false_boolean_parameter(self):
        self.litConfig.params['enable_exceptions'] = False
        param = dsl.Parameter(name='enable_exceptions', choices=[True, False], type=bool, help='',
                              actions=lambda exceptions: [] if exceptions else [dsl.AddFeature("-fno-exceptions")])
        for a in param.getActions(self.config, self.litConfig.params):
            a.applyTo(self.config)
        self.assertIn('-fno-exceptions', self.config.available_features)

    def test_list_parsed_from_comma_delimited_string_empty(self):
        self.litConfig.params['additional_features'] = ""
        param = dsl.Parameter(name='additional_features', type=list, help='', actions=lambda f: f)
        self.assertEqual(param.getActions(self.config, self.litConfig.params), [])

    def test_list_parsed_from_comma_delimited_string_1(self):
        self.litConfig.params['additional_features'] = "feature1"
        param = dsl.Parameter(name='additional_features', type=list, help='', actions=lambda f: f)
        self.assertEqual(param.getActions(self.config, self.litConfig.params), ['feature1'])

    def test_list_parsed_from_comma_delimited_string_2(self):
        self.litConfig.params['additional_features'] = "feature1,feature2"
        param = dsl.Parameter(name='additional_features', type=list, help='', actions=lambda f: f)
        self.assertEqual(param.getActions(self.config, self.litConfig.params), ['feature1', 'feature2'])

    def test_list_parsed_from_comma_delimited_string_3(self):
        self.litConfig.params['additional_features'] = "feature1,feature2, feature3"
        param = dsl.Parameter(name='additional_features', type=list, help='', actions=lambda f: f)
        self.assertEqual(param.getActions(self.config, self.litConfig.params), ['feature1', 'feature2', 'feature3'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
