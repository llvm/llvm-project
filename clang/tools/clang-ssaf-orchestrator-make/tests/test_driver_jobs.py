# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import unittest

from clang_ssaf_orchestrator_make.shims import driver_jobs as under_test

PREAMBLE = (
    b"clang version 24.0.0git\n"
    b"Target: arm64-apple-darwin24.6.0\n"
    b"Thread model: posix\n"
    b"InstalledDir: /opt/llvm/bin\n"
)


def listing(*jobs):
    return PREAMBLE + b"".join(b" " + job + b"\n" for job in jobs)


def _program(job):
    return os.path.basename(job.argv[0])


CC1_COMPILE = (
    b'"/opt/llvm/bin/clang" "-cc1" "-triple" "arm64-apple-macosx15.0.0" '
    b'"-emit-obj" "-o" "build/a.o" "-x" "c++" "src/a.cpp"'
)
LINK = b'"/usr/bin/ld" "-o" "build/prg" "build/a.o" "-lc++"'


class Preamble(unittest.TestCase):
    def test_preamble_is_not_mistaken_for_jobs(self):
        self.assertEqual(under_test.parse_driver_job_listing(PREAMBLE), [])

    def test_build_config_line_some_builds_add_is_ignored(self):
        data = PREAMBLE + b"Build config: +assertions\n" + b" " + CC1_COMPILE + b"\n"
        self.assertEqual(len(under_test.parse_driver_job_listing(data)), 1)

    def test_in_process_marker_is_not_a_job(self):
        data = PREAMBLE + b" (in-process)\n" + b" " + CC1_COMPILE + b"\n"
        jobs = under_test.parse_driver_job_listing(data)
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].argv[1], "-cc1")

    def test_diagnostics_before_and_after_the_preamble_are_ignored(self):
        data = (
            b"clang: error: unknown argument: '--nope'\n"
            + PREAMBLE
            + b"clang: warning: argument unused [-Wunused-command-line-argument]\n"
            + b" "
            + CC1_COMPILE
            + b"\n"
        )
        self.assertEqual(len(under_test.parse_driver_job_listing(data)), 1)

    def test_empty_listing(self):
        self.assertEqual(under_test.parse_driver_job_listing(b""), [])


class Escaping(unittest.TestCase):
    def _value(self, quoted):
        data = listing(b'"/opt/llvm/bin/clang" "-cc1" "-D" ' + quoted)
        return under_test.parse_driver_job_listing(data)[0].argv[-1]

    def test_space_is_raw_inside_quotes(self):
        self.assertEqual(self._value(b'"A=x y"'), "A=x y")

    def test_escaped_double_quote(self):
        self.assertEqual(self._value(b'"A=\\"q\\""'), 'A="q"')

    def test_escaped_backslash(self):
        self.assertEqual(self._value(b'"A=a\\\\b"'), "A=a\\b")

    def test_escaped_dollar(self):
        self.assertEqual(self._value(b'"A=\\$(PWD)/x"'), "A=$(PWD)/x")

    def test_tab_is_raw(self):
        self.assertEqual(self._value(b'"A=a\tb"'), "A=a\tb")

    def test_apostrophe_and_backtick_are_raw(self):
        self.assertEqual(self._value(b'"A=it\'s `x`"'), "A=it's `x`")

    def test_shell_metacharacters_are_raw(self):
        self.assertEqual(self._value(b'"A=a;b|c&d"'), "A=a;b|c&d")

    def test_empty_argument(self):
        self.assertEqual(self._value(b'""'), "")

    def test_non_utf8_bytes_survive(self):
        value = self._value(b'"A=\xff\xfe"')
        self.assertEqual(value.encode("utf-8", "surrogateescape"), b"A=\xff\xfe")

    def test_newline_inside_an_argument_does_not_end_the_job(self):
        data = listing(b'"/opt/llvm/bin/clang" "-cc1" "-D" "A=a\nb" "-x" "c++" "a.cpp"')
        jobs = under_test.parse_driver_job_listing(data)
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].argv[3], "A=a\nb")
        self.assertEqual(jobs[0].argv[-1], "a.cpp")


class Classification(unittest.TestCase):
    def test_an_assembler_job_is_read_like_any_other(self):
        data = listing(b'"/opt/llvm/bin/clang" "-cc1as" "-triple" "arm64--" "a.s"')
        job = under_test.parse_driver_job_listing(data)[0]
        self.assertEqual("/opt/llvm/bin/clang", job.argv[0])
        self.assertEqual("-cc1as", job.argv[1])

    def test_the_output_is_the_key_the_artifact_takes(self):
        job = under_test.parse_driver_job_listing(listing(CC1_COMPILE))[0]
        self.assertEqual(job.output, "build/a.o")

    def test_the_repr_names_what_a_reader_of_a_verbose_run_needs(self):
        job = under_test.parse_driver_job_listing(listing(CC1_COMPILE))[0]
        self.assertEqual("Job(argv=%r, output='build/a.o')" % (job.argv,), repr(job))

    def test_missing_option_value(self):
        job = under_test.parse_driver_job_listing(
            listing(b'"/opt/llvm/bin/clang" "-cc1" "-o"')
        )[0]
        self.assertIsNone(job.output)


class Decomposition(unittest.TestCase):
    def test_compile_and_link(self):
        jobs = under_test.parse_driver_job_listing(listing(CC1_COMPILE, LINK))
        self.assertEqual(["clang", "ld"], [_program(j) for j in jobs])

    def test_multiple_architectures_have_distinct_outputs(self):
        arm = (
            b'"/opt/llvm/bin/clang" "-cc1" "-triple" "arm64-apple-macosx15.0.0" '
            b'"-o" "/tmp/p-aaa/p-arm64.o" "-x" "c++" "p.cpp"'
        )
        intel = (
            b'"/opt/llvm/bin/clang" "-cc1" "-triple" "x86_64-apple-macosx15.0.0" '
            b'"-o" "/tmp/p-bbb/p-x86_64.o" "-x" "c++" "p.cpp"'
        )
        lipo = b'"/usr/bin/lipo" "-create" "-output" "p.o" "/tmp/p-aaa/p-arm64.o"'
        jobs = under_test.parse_driver_job_listing(listing(arm, intel, lipo))
        self.assertEqual(["clang", "clang", "lipo"], [_program(j) for j in jobs])
        self.assertNotEqual(jobs[0].output, jobs[1].output)

    def test_link_only_has_no_frontend_job(self):
        jobs = under_test.parse_driver_job_listing(listing(LINK))
        self.assertEqual(["ld"], [_program(j) for j in jobs])


if __name__ == "__main__":
    unittest.main()
