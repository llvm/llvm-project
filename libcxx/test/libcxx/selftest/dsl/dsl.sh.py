#===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##

# Note: We prepend arguments with 'x' to avoid thinking there are too few
#       arguments in case an argument is an empty string.
# RUN: %{python} %s x%S \
# RUN:              x%T \
# RUN:              x%{escaped_exec} \
# RUN:              x%{escaped_cxx} \
# RUN:              x%{escaped_flags} \
# RUN:              x%{escaped_compile_flags} \
# RUN:              x%{escaped_link_flags}
# END.

import base64
import copy
import os
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
SOURCE_ROOT, EXEC_PATH, EXEC, CXX, FLAGS, COMPILE_FLAGS, LINK_FLAGS = args
sys.argv[1:8] = []

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
        base64Decode = lambda s: lit.util.to_string(base64.b64decode(s))
        self.config.substitutions = [
            ('%{cxx}', base64Decode(CXX)),
            ('%{flags}', base64Decode(FLAGS)),
            ('%{compile_flags}', base64Decode(COMPILE_FLAGS)),
            ('%{link_flags}', base64Decode(LINK_FLAGS)),
            ('%{exec}', base64Decode(EXEC))
        ]

    def getSubstitution(self, substitution):
        """
        Return a given substitution from the TestingConfig. It is an error if
        there is no such substitution.
        """
        found = [x for (s, x) in self.config.substitutions if s == substitution]
        assert len(found) == 1
        return found[0]


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
        source = """int main(int, char**) { }"""
        self.assertTrue(dsl.sourceBuilds(self.config, source))

    def test_compilation_error_fails(self):
        source = """in main(int, char**) { }"""
        self.assertFalse(dsl.sourceBuilds(self.config, source))

    def test_link_error_fails(self):
        source = """extern void this_isnt_defined_anywhere();
                    int main(int, char**) { this_isnt_defined_anywhere(); }"""
        self.assertFalse(dsl.sourceBuilds(self.config, source))

class TestProgramOutput(SetupConfigs):
    """
    Tests for libcxx.test.dsl.programOutput
    """
    def test_valid_program_returns_output(self):
        source = """
        #include <cstdio>
        int main(int, char**) { std::printf("FOOBAR"); }
        """
        self.assertEqual(dsl.programOutput(self.config, source), "FOOBAR")

    def test_valid_program_returns_output_newline_handling(self):
        source = """
        #include <cstdio>
        int main(int, char**) { std::printf("FOOBAR\\n"); }
        """
        self.assertEqual(dsl.programOutput(self.config, source), "FOOBAR\n")

    def test_valid_program_returns_no_output(self):
        source = """
        int main(int, char**) { }
        """
        self.assertEqual(dsl.programOutput(self.config, source), "")

    def test_invalid_program_returns_None_1(self):
        # The program compiles, but exits with an error
        source = """
        int main(int, char**) { return 1; }
        """
        self.assertEqual(dsl.programOutput(self.config, source), None)

    def test_invalid_program_returns_None_2(self):
        # The program doesn't compile
        source = """
        int main(int, char**) { this doesnt compile }
        """
        self.assertEqual(dsl.programOutput(self.config, source), None)

    def test_pass_arguments_to_program(self):
        source = """
        #include <cassert>
        #include <string>
        int main(int argc, char** argv) {
            assert(argc == 3);
            assert(argv[1] == std::string("first-argument"));
            assert(argv[2] == std::string("second-argument"));
        }
        """
        args = ["first-argument", "second-argument"]
        self.assertEqual(dsl.programOutput(self.config, source, args=args), "")

class TestHasLocale(SetupConfigs):
    """
    Tests for libcxx.test.dsl.hasLocale
    """
    def test_doesnt_explode(self):
        # It's really hard to test that a system has a given locale, so at least
        # make sure we don't explode when we try to check it.
        try:
            dsl.hasLocale(self.config, 'en_US.UTF-8')
        except subprocess.CalledProcessError:
            self.fail("checking for hasLocale should not explode")

    def test_nonexistent_locale(self):
        self.assertFalse(dsl.hasLocale(self.config, 'for_sure_this_is_not_an_existing_locale'))


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
        self.assertTrue(feature.isSupported(self.config))
        feature.enableIn(self.config)
        self.assertEqual(origSubstitutions, self.config.substitutions)
        self.assertIn('name', self.config.available_features)

    def test_name_can_be_a_callable(self):
        feature = dsl.Feature(name=lambda cfg: (self.assertIs(self.config, cfg), 'name')[1])
        assert feature.isSupported(self.config)
        feature.enableIn(self.config)
        self.assertIn('name', self.config.available_features)

    def test_name_is_not_a_string_1(self):
        feature = dsl.Feature(name=None)
        assert feature.isSupported(self.config)
        self.assertRaises(ValueError, lambda: feature.enableIn(self.config))

    def test_name_is_not_a_string_2(self):
        feature = dsl.Feature(name=lambda cfg: None)
        assert feature.isSupported(self.config)
        self.assertRaises(ValueError, lambda: feature.enableIn(self.config))

    def test_adding_compile_flag(self):
        feature = dsl.Feature(name='name', compileFlag='-foo')
        origLinkFlags = copy.deepcopy(self.getSubstitution('%{link_flags}'))
        assert feature.isSupported(self.config)
        feature.enableIn(self.config)
        self.assertIn('name', self.config.available_features)
        self.assertIn('-foo', self.getSubstitution('%{compile_flags}'))
        self.assertEqual(origLinkFlags, self.getSubstitution('%{link_flags}'))

    def test_compile_flag_can_be_a_callable(self):
        feature = dsl.Feature(name='name',
                              compileFlag=lambda cfg: (self.assertIs(self.config, cfg), '-foo')[1])
        assert feature.isSupported(self.config)
        feature.enableIn(self.config)
        self.assertIn('-foo', self.getSubstitution('%{compile_flags}'))

    def test_adding_link_flag(self):
        feature = dsl.Feature(name='name', linkFlag='-foo')
        origCompileFlags = copy.deepcopy(self.getSubstitution('%{compile_flags}'))
        assert feature.isSupported(self.config)
        feature.enableIn(self.config)
        self.assertIn('name', self.config.available_features)
        self.assertIn('-foo', self.getSubstitution('%{link_flags}'))
        self.assertEqual(origCompileFlags, self.getSubstitution('%{compile_flags}'))

    def test_link_flag_can_be_a_callable(self):
        feature = dsl.Feature(name='name',
                              linkFlag=lambda cfg: (self.assertIs(self.config, cfg), '-foo')[1])
        assert feature.isSupported(self.config)
        feature.enableIn(self.config)
        self.assertIn('-foo', self.getSubstitution('%{link_flags}'))

    def test_adding_both_flags(self):
        feature = dsl.Feature(name='name', compileFlag='-hello', linkFlag='-world')
        assert feature.isSupported(self.config)
        feature.enableIn(self.config)
        self.assertIn('name', self.config.available_features)

        self.assertIn('-hello', self.getSubstitution('%{compile_flags}'))
        self.assertNotIn('-world', self.getSubstitution('%{compile_flags}'))

        self.assertIn('-world', self.getSubstitution('%{link_flags}'))
        self.assertNotIn('-hello', self.getSubstitution('%{link_flags}'))

    def test_unsupported_feature(self):
        feature = dsl.Feature(name='name', when=lambda _: False)
        self.assertFalse(feature.isSupported(self.config))
        # Also make sure we assert if we ever try to add it to a config
        self.assertRaises(AssertionError, lambda: feature.enableIn(self.config))

    def test_is_supported_gets_passed_the_config(self):
        feature = dsl.Feature(name='name', when=lambda cfg: (self.assertIs(self.config, cfg), True)[1])
        self.assertTrue(feature.isSupported(self.config))


class TestParameter(SetupConfigs):
    """
    Tests for libcxx.test.dsl.Parameter
    """
    def test_empty_name_should_blow_up(self):
        self.assertRaises(ValueError, lambda: dsl.Parameter(name='', choices=['c++03'], type=str, help='', feature=lambda _: None))

    def test_empty_choices_should_blow_up(self):
        self.assertRaises(ValueError, lambda: dsl.Parameter(name='std', choices=[], type=str, help='', feature=lambda _: None))

    def test_name_is_set_correctly(self):
        param = dsl.Parameter(name='std', choices=['c++03'], type=str, help='', feature=lambda _: None)
        self.assertEqual(param.name, 'std')

    def test_no_value_provided_and_no_default_value(self):
        param = dsl.Parameter(name='std', choices=['c++03'], type=str, help='', feature=lambda _: None)
        self.assertRaises(ValueError, lambda: param.getFeature(self.config, self.litConfig.params))

    def test_no_value_provided_and_default_value(self):
        param = dsl.Parameter(name='std', choices=['c++03'], type=str, help='', default='c++03',
                              feature=lambda std: dsl.Feature(name=std))
        param.getFeature(self.config, self.litConfig.params).enableIn(self.config)
        self.assertIn('c++03', self.config.available_features)

    def test_value_provided_on_command_line_and_no_default_value(self):
        self.litConfig.params['std'] = 'c++03'
        param = dsl.Parameter(name='std', choices=['c++03'], type=str, help='',
                              feature=lambda std: dsl.Feature(name=std))
        param.getFeature(self.config, self.litConfig.params).enableIn(self.config)
        self.assertIn('c++03', self.config.available_features)

    def test_value_provided_on_command_line_and_default_value(self):
        """The value provided on the command line should override the default value"""
        self.litConfig.params['std'] = 'c++11'
        param = dsl.Parameter(name='std', choices=['c++03', 'c++11'], type=str, default='c++03', help='',
                              feature=lambda std: dsl.Feature(name=std))
        param.getFeature(self.config, self.litConfig.params).enableIn(self.config)
        self.assertIn('c++11', self.config.available_features)
        self.assertNotIn('c++03', self.config.available_features)

    def test_value_provided_in_config_and_default_value(self):
        """The value provided in the config should override the default value"""
        self.config.std ='c++11'
        param = dsl.Parameter(name='std', choices=['c++03', 'c++11'], type=str, default='c++03', help='',
                              feature=lambda std: dsl.Feature(name=std))
        param.getFeature(self.config, self.litConfig.params).enableIn(self.config)
        self.assertIn('c++11', self.config.available_features)
        self.assertNotIn('c++03', self.config.available_features)

    def test_value_provided_in_config_and_on_command_line(self):
        """The value on the command line should override the one in the config"""
        self.config.std = 'c++11'
        self.litConfig.params['std'] = 'c++03'
        param = dsl.Parameter(name='std', choices=['c++03', 'c++11'], type=str, help='',
                              feature=lambda std: dsl.Feature(name=std))
        param.getFeature(self.config, self.litConfig.params).enableIn(self.config)
        self.assertIn('c++03', self.config.available_features)
        self.assertNotIn('c++11', self.config.available_features)

    def test_feature_is_None(self):
        self.litConfig.params['std'] = 'c++03'
        param = dsl.Parameter(name='std', choices=['c++03'], type=str, help='',
                              feature=lambda _: None)
        feature = param.getFeature(self.config, self.litConfig.params)
        self.assertIsNone(feature)

    def test_boolean_value_parsed_from_trueish_string_parameter(self):
        self.litConfig.params['enable_exceptions'] = "True"
        param = dsl.Parameter(name='enable_exceptions', choices=[True, False], type=bool, help='',
                              feature=lambda exceptions: None if exceptions else ValueError())
        self.assertIsNone(param.getFeature(self.config, self.litConfig.params))

    def test_boolean_value_from_true_boolean_parameter(self):
        self.litConfig.params['enable_exceptions'] = True
        param = dsl.Parameter(name='enable_exceptions', choices=[True, False], type=bool, help='',
                              feature=lambda exceptions: None if exceptions else ValueError())
        self.assertIsNone(param.getFeature(self.config, self.litConfig.params))

    def test_boolean_value_parsed_from_falseish_string_parameter(self):
        self.litConfig.params['enable_exceptions'] = "False"
        param = dsl.Parameter(name='enable_exceptions', choices=[True, False], type=bool, help='',
                              feature=lambda exceptions: None if exceptions else dsl.Feature(name="-fno-exceptions"))
        param.getFeature(self.config, self.litConfig.params).enableIn(self.config)
        self.assertIn('-fno-exceptions', self.config.available_features)

    def test_boolean_value_from_false_boolean_parameter(self):
        self.litConfig.params['enable_exceptions'] = False
        param = dsl.Parameter(name='enable_exceptions', choices=[True, False], type=bool, help='',
                              feature=lambda exceptions: None if exceptions else dsl.Feature(name="-fno-exceptions"))
        param.getFeature(self.config, self.litConfig.params).enableIn(self.config)
        self.assertIn('-fno-exceptions', self.config.available_features)


if __name__ == '__main__':
    unittest.main(verbosity=2)
