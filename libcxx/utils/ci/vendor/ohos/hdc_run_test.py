#!/usr/bin/env python3
# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest


SCRIPT_DIR = Path(__file__).resolve().parent
HDC_RUN = SCRIPT_DIR / "hdc_run.py"


class HdcRunTest(unittest.TestCase):
    def make_fake_hdc(self, directory):
        fake_hdc = directory / "hdc"
        fake_hdc.write_text(
            textwrap.dedent(
                """\
                #!/usr/bin/env python3
                import sys

                EXIT_MARKER = "__llvm_hdc_exit__="

                def main():
                    args = sys.argv[1:]
                    if args[:2] == ["file", "send"]:
                        print("FileTransfer finish")
                        return 0

                    if args and args[0] == "shell":
                        command = args[1]
                        if "stderr.tmp.exe" in command:
                            sys.stdout.write("STDOUT-OUTPUT")
                            sys.stderr.write("STDERR-OUTPUT")
                            sys.stdout.write(f"\\n{EXIT_MARKER}0\\n")
                            return 0
                        if "stdin.tmp.exe" in command:
                            sys.stdout.write("IN:" + sys.stdin.read())
                            sys.stdout.write(f"\\n{EXIT_MARKER}0\\n")
                            return 0
                        sys.stdout.write(f"\\n{EXIT_MARKER}0\\n")
                        return 0

                    print("unexpected fake hdc invocation: " + repr(args), file=sys.stderr)
                    return 2

                if __name__ == "__main__":
                    raise SystemExit(main())
                """
            )
        )
        fake_hdc.chmod(fake_hdc.stat().st_mode | stat.S_IXUSR)
        return fake_hdc

    def run_hdc_run(self, temp_dir, executable_name, *, stdin_text=None):
        execdir = temp_dir / "execdir"
        execdir.mkdir()
        executable = execdir / executable_name
        executable.write_text("# fake executable marker\n")
        executable.chmod(executable.stat().st_mode | stat.S_IXUSR)

        fake_hdc = self.make_fake_hdc(temp_dir)
        return subprocess.run(
            [
                sys.executable,
                str(HDC_RUN),
                "--hdc",
                str(fake_hdc),
                "--execdir",
                str(execdir),
                "--host-staging-root",
                str(temp_dir / "staging"),
                str(executable),
            ],
            input=stdin_text,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

    def test_preserves_remote_stdout_and_stderr_separately(self):
        with tempfile.TemporaryDirectory() as temp:
            completed = self.run_hdc_run(Path(temp), "stderr.tmp.exe")

        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stdout, "STDOUT-OUTPUT")
        self.assertEqual(completed.stderr, "STDERR-OUTPUT")

    def test_forwards_stdin_to_remote_command(self):
        with tempfile.TemporaryDirectory() as temp:
            completed = self.run_hdc_run(
                Path(temp),
                "stdin.tmp.exe",
                stdin_text="hello from stdin",
            )

        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stdout, "IN:hello from stdin")
        self.assertEqual(completed.stderr, "")


if __name__ == "__main__":
    unittest.main()
