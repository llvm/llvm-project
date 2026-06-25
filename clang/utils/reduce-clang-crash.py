#!/usr/bin/env python3
"""Calls reduction tools to create minimal reproducers for clang crashes.

For frontend crashes, runs C-Vise/C-Reduce on the source file.
For middle-end/backend crashes, runs llvm-reduce on emitted LLVM IR:
  - If `clang -emit-llvm` succeeds and `llc` on the IR crashes, it's a
    backend crash and llvm-reduce is run with llc as the test tool.
  - Otherwise, if `clang -emit-llvm -Xclang -disable-llvm-passes` succeeds
    and `opt` on the IR crashes, it's a middle-end crash and llvm-reduce
    is run with opt as the test tool.

Output files:
  *.reduced.sh -- crash reproducer with minimal arguments
  *.reduced.cpp or *.reduced.ll -- the reduced file
  *.test.sh -- interestingness test for C-Vise or llvm-reduce
"""

from argparse import ArgumentParser, RawTextHelpFormatter
import os
import re
import shutil
import stat
import sys
import subprocess
import shlex
import tempfile
import shutil
import multiprocessing

verbose = False
creduce_cmd = None
clang_cmd = None
llc_cmd = None
opt_cmd = None
llvm_reduce_cmd = None
reduce_pipeline_cmd = None
llvm_symbolizer_cmd = None


def verbose_print(*args, **kwargs):
    if verbose:
        print(*args, **kwargs)


def check_file(fname):
    fname = os.path.normpath(fname)
    if not os.path.isfile(fname):
        sys.exit("ERROR: %s does not exist" % (fname))
    return fname


def check_cmd(cmd_name, cmd_dir, cmd_path=None, return_none_if_not_found=False):
    """
    Returns absolute path to cmd_path if it is given,
    or absolute path to cmd_dir/cmd_name.
    """
    if cmd_path:
        # Make the path absolute so the creduce test can be run from any directory.
        cmd_path = os.path.abspath(cmd_path)
        cmd = shutil.which(cmd_path)
        if cmd:
            return cmd
        sys.exit("ERROR: executable `%s` not found" % (cmd_path))

    cmd = shutil.which(cmd_name, path=cmd_dir)
    if cmd:
        return cmd

    if return_none_if_not_found:
        return None

    if not cmd_dir:
        cmd_dir = "$PATH"
    sys.exit("ERROR: `%s` not found in %s" % (cmd_name, cmd_dir))


def quote_cmd(cmd):
    return " ".join(shlex.quote(arg) for arg in cmd)


def write_to_script(text, filename):
    with open(filename, "w") as f:
        f.write(text)
    os.chmod(filename, os.stat(filename).st_mode | stat.S_IEXEC)


class Reduce(object):
    def __init__(self, crash_script, file_to_reduce, creduce_flags, llvm_reduce_flags):
        crash_script_name, crash_script_ext = os.path.splitext(crash_script)
        file_reduce_name, file_reduce_ext = os.path.splitext(file_to_reduce)

        self.testfile = file_reduce_name + ".test.sh"
        self.crash_script = crash_script_name + ".reduced" + crash_script_ext
        self.reduced_source_file = file_reduce_name + ".reduced" + file_reduce_ext
        self.file_to_reduce = file_to_reduce

        self.clang = clang_cmd
        self.clang_args = []
        self.expected_output = []
        self.needs_stack_trace = False
        self.creduce_flags = ["--tidy"] + creduce_flags
        if "--n" not in self.creduce_flags:
            self.creduce_flags += ["--n", str(max(4, multiprocessing.cpu_count() // 2))]
        self.llvm_reduce_flags = llvm_reduce_flags

        self.read_clang_args(crash_script, file_to_reduce)
        self.read_expected_output()

    def prepare_source_reduction(self):
        shutil.copy(self.file_to_reduce, self.reduced_source_file)
        self.file_to_reduce = self.reduced_source_file

    def get_crash_cmd(self, cmd=None, args=None, filename=None):
        if not cmd:
            cmd = self.clang
        if not args:
            args = self.clang_args
        if not filename:
            filename = self.file_to_reduce

        return [cmd] + args + [filename]

    def read_clang_args(self, crash_script, filename):
        print("\nReading arguments from crash script...")
        with open(crash_script) as f:
            # Assume clang call is the first non comment line.
            cmd = []
            for line in f:
                if not line.lstrip().startswith("#"):
                    cmd = shlex.split(line)
                    break
        if not cmd:
            sys.exit("Could not find command in the crash script.")

        # Remove clang and filename from the command
        # Assume the last occurrence of the filename is the clang input file
        del cmd[0]
        target_base = os.path.basename(filename)
        for i in range(len(cmd) - 1, -1, -1):
            if os.path.basename(cmd[i]) == target_base:
                del cmd[i]
                break

        if "-cc1" not in cmd:
            cmd = self.driver_to_cc1(cmd, filename)

        self.clang_args = cmd
        verbose_print("Clang arguments:", quote_cmd(self.clang_args))

    def driver_to_cc1(self, driver_args, filename):
        """Convert a driver-mode invocation to its cc1 form via `clang -###`."""
        print("Driver command detected; using `clang -###` to get cc1 invocation...")
        invocation = [self.clang, "-###"] + driver_args + [filename]
        verbose_print("Running:", quote_cmd(invocation))
        p = subprocess.Popen(
            invocation, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        _, output = p.communicate()
        if p.returncode != 0:
            sys.exit(
                "ERROR: `clang -###` failed with exit code %d:\n%s"
                % (p.returncode, output.decode("utf-8", errors="replace"))
            )

        cc1_cmd = None
        for line in output.decode("utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                parsed = shlex.split(line)
            except ValueError:
                continue
            if "-cc1" in parsed:
                cc1_cmd = parsed
                break
        if cc1_cmd is None:
            sys.exit("ERROR: could not extract a cc1 invocation from `clang -###`")

        # Drop the executable
        del cc1_cmd[0]
        # Drop the last arg that matches the input filename, skipping
        # `-main-file-name <basename>` since that takes the basename as a value.
        target_base = os.path.basename(filename)
        for i in range(len(cc1_cmd) - 1, -1, -1):
            if os.path.basename(cc1_cmd[i]) == target_base:
                if i > 0 and cc1_cmd[i - 1] == "-main-file-name":
                    continue
                del cc1_cmd[i]
                break
        return cc1_cmd

    def read_expected_output(self):
        print("\nGetting expected crash output...")
        p = subprocess.Popen(
            self.get_crash_cmd(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        crash_output, _ = p.communicate()
        result = []

        # Remove color codes
        ansi_escape = r"\x1b\[[0-?]*m"
        crash_output = re.sub(ansi_escape, "", crash_output.decode("utf-8"))

        # Look for specific error messages
        regexes = [
            r"Assertion .+ failed",  # Linux assert()
            r"Assertion failed: .+,",  # FreeBSD/Mac assert()
            r"fatal error: error in backend: .+",
            r"LLVM ERROR: .+",
            r"UNREACHABLE executed at .+?!",
            r"LLVM IR generation of declaration '.+'",
            r"Generating code for declaration '.+'",
            r"\*\*\* Bad machine code: .+ \*\*\*",
            r"ERROR: .*Sanitizer: [^ ]+ ",
        ]
        for msg_re in regexes:
            match = re.search(msg_re, crash_output)
            if match:
                msg = match.group(0)
                result = [msg]
                print("Found message:", msg)
                break

        # If no message was found, use the top five stack trace functions,
        # ignoring some common functions
        # Five is a somewhat arbitrary number; the goal is to get a small number
        # of identifying functions with some leeway for common functions
        if not result:
            self.needs_stack_trace = True
            stacktrace_re = r"[0-9]+\s+0[xX][0-9a-fA-F]+\s*([^\r\n(]+)\("
            filters = [
                "PrintStackTrace",
                "RunSignalHandlers",
                "CleanupOnSignal",
                "HandleCrash",
                "SignalHandler",
                "__restore_rt",
                "gsignal",
                "abort",
                "SignalHandlerTerminate",
            ]

            matches = re.findall(stacktrace_re, crash_output)

            # Find the last frame that matches any of the filters
            last_filter_idx = -1
            for idx, func_name in enumerate(matches):
                if any(name in func_name for name in filters):
                    last_filter_idx = idx

            # Slice the matches to ignore all frames up to and including the last filtered frame
            app_matches = matches[last_filter_idx + 1 :]

            result = [x.strip() for x in app_matches if x.strip()][:5]
            for msg in result:
                print("Found stack trace function:", msg)

        if not result:
            print("ERROR: no crash was found")
            print("The crash output was:\n========\n%s========" % crash_output)
            sys.exit(1)

        self.expected_output = result

    def check_expected_output(self, args=None, filename=None):
        if not args:
            args = self.clang_args
        if not filename:
            filename = self.file_to_reduce

        p = subprocess.Popen(
            self.get_crash_cmd(args=args, filename=filename),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        crash_output, _ = p.communicate()
        return all(msg in crash_output.decode("utf-8") for msg in self.expected_output)

    def write_interestingness_test(self):
        print("\nCreating the interestingness test...")

        # Disable symbolization if it's not required to avoid slow symbolization.
        symbolizer_env = ""
        if not self.needs_stack_trace:
            symbolizer_env = "export LLVM_DISABLE_SYMBOLIZATION=1"
        elif llvm_symbolizer_cmd:
            symbolizer_env = "export LLVM_SYMBOLIZER_PATH=%s" % shlex.quote(
                llvm_symbolizer_cmd
            )

        output = """#!/bin/bash
%s
if %s >& "$(dirname "$0")/t.log" ; then
  exit 1
fi
""" % (
            symbolizer_env,
            quote_cmd(self.get_crash_cmd()),
        )

        for msg in self.expected_output:
            output += 'grep -F %s "$(dirname "$0")/t.log" || exit 1\n' % shlex.quote(
                msg
            )

        write_to_script(output, self.testfile)
        self.check_interestingness()

    def check_interestingness(self):
        testfile = os.path.abspath(self.testfile)

        # Check that the test considers the original file interesting
        returncode = subprocess.call(testfile, stdout=subprocess.DEVNULL)
        if returncode:
            sys.exit("The interestingness test does not pass for the original file.")

        # Check that an empty file is not interesting
        # Instead of modifying the filename in the test file, just run the command
        with tempfile.NamedTemporaryFile() as empty_file:
            is_interesting = self.check_expected_output(filename=empty_file.name)
        if is_interesting:
            sys.exit("The interestingness test passes for an empty file.")

    def clang_preprocess(self):
        print("\nTrying to preprocess the source file...")
        with tempfile.NamedTemporaryFile() as tmpfile:
            cmd_preprocess = self.get_crash_cmd() + ["-E", "-o", tmpfile.name]
            cmd_preprocess_no_lines = cmd_preprocess + ["-P"]
            try:
                subprocess.check_call(cmd_preprocess_no_lines)
                if self.check_expected_output(filename=tmpfile.name):
                    print("Successfully preprocessed with line markers removed")
                    shutil.copy(tmpfile.name, self.file_to_reduce)
                else:
                    subprocess.check_call(cmd_preprocess)
                    if self.check_expected_output(filename=tmpfile.name):
                        print("Successfully preprocessed without removing line markers")
                        shutil.copy(tmpfile.name, self.file_to_reduce)
                    else:
                        print(
                            "No longer crashes after preprocessing -- "
                            "using original source"
                        )
            except subprocess.CalledProcessError:
                print("Preprocessing failed")

    def emit_llvm_ir(self, output_file, disable_passes=False):
        """Try to emit textual LLVM IR with `clang -cc1 -emit-llvm -S`.

        Returns True if clang exited successfully (no crash, IR was written).
        """
        args = []
        skip_next = False
        for arg in self.clang_args:
            if skip_next:
                skip_next = False
                continue
            if arg in {
                "-fsyntax-only",
                "-emit-llvm",
                "-emit-llvm-bc",
                "-emit-llvm-only",
                "-emit-obj",
                "-S",
            }:
                continue
            if arg == "-o":
                skip_next = True
                continue
            args.append(arg)

        extra = ["-emit-llvm", "-o", output_file]
        if disable_passes:
            extra.append("-disable-llvm-passes")
        cmd = [self.clang] + args + extra + [self.file_to_reduce]
        verbose_print("Emitting LLVM IR:", quote_cmd(cmd))
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        _, err = p.communicate()
        if p.returncode != 0:
            verbose_print("emit_llvm_ir failed with exit code", p.returncode)
            verbose_print("stderr:", err.decode("utf-8", errors="replace"))
        return p.returncode == 0

    def get_opt_llc_args(self):
        """Extract args from clang_args that should be forwarded to llc/opt."""
        opt_level = "-O2"
        for a in self.clang_args:
            if re.match(r"^-O[0-3sz]$", a):
                opt_level = a

        forwarded_args = [opt_level]
        i = 0
        while i < len(self.clang_args):
            if self.clang_args[i] == "-mllvm":
                if i + 1 < len(self.clang_args):
                    forwarded_args.append(self.clang_args[i + 1])
                    i += 2
                else:
                    i += 1
            elif self.clang_args[i].startswith("-mllvm="):
                forwarded_args.append(self.clang_args[i][len("-mllvm=") :])
                i += 1
            else:
                i += 1
        return forwarded_args

    def check_tool_crash(self, tool_cmd):
        """Run tool_cmd and check whether the expected crash output appears."""
        p = subprocess.Popen(tool_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out, _ = p.communicate()
        out_str = out.decode("utf-8", errors="replace")
        matched = all(msg in out_str for msg in self.expected_output)
        if not matched:
            verbose_print("Failed to match expected output in tool crash.")
            verbose_print("Expected:", self.expected_output)
            verbose_print("Actual output:\n", out_str)
        return matched

    def reduce_tool_args(self, tool_cmd, tool_args, ir_file, extra_args=[]):
        """Minimize the tool arguments by trying to remove them one by one."""
        print("\nReducing %s arguments..." % os.path.basename(tool_cmd))
        reduced_args = list(tool_args)
        i = 0
        while i < len(reduced_args):
            candidate_args = reduced_args[:i] + reduced_args[i + 1 :]
            cmd = [tool_cmd] + candidate_args + extra_args + [ir_file]
            if self.check_tool_crash(cmd):
                verbose_print("Removed argument: %s" % reduced_args[i])
                reduced_args = candidate_args
            else:
                i += 1
        return reduced_args

    def try_llvm_ir_crash(self, ir_file):
        """Try to reproduce the crash with llc or opt on emitted LLVM IR.

        Writes the IR to `ir_file` if successful.
        Returns (tool_path, tool_args) on success, or None.
        """
        orig_expected_output = self.expected_output
        self.expected_output = [
            x for x in orig_expected_output if "clang" not in x.lower()
        ]
        if not self.expected_output:
            verbose_print("No LLVM frames in expected output, skipping IR crash try")
            self.expected_output = orig_expected_output
            return None

        ir_dir = os.path.dirname(ir_file)
        tmp_ir = tempfile.NamedTemporaryFile(suffix=".ll", dir=ir_dir, delete=False)
        tmp_ir_name = tmp_ir.name
        tmp_ir.close()

        try:
            if llc_cmd:
                print("\nTrying to reproduce crash with llc on optimized LLVM IR...")
                if self.emit_llvm_ir(tmp_ir_name, disable_passes=False):
                    llc_args = self.get_opt_llc_args()
                    if self.check_tool_crash([llc_cmd] + llc_args + [tmp_ir_name]):
                        print("Crash reproduces with llc -- treating as backend crash")
                        llc_args = self.reduce_tool_args(llc_cmd, llc_args, tmp_ir_name)
                        shutil.copy(tmp_ir_name, ir_file)
                        return (llc_cmd, llc_args)
                    print("Crash does not reproduce with llc")
                else:
                    print("clang -emit-llvm did not complete")

            if opt_cmd:
                print("\nTrying to reproduce crash with opt on unoptimized LLVM IR...")
                if self.emit_llvm_ir(tmp_ir_name, disable_passes=True):
                    opt_args = self.get_opt_llc_args()
                    if self.check_tool_crash(
                        [opt_cmd] + opt_args + ["-disable-output", tmp_ir_name]
                    ):
                        print("Crash reproduces with opt -- treating as middle-end crash")
                        reduced = self.run_reduce_pipeline(tmp_ir_name, opt_args)
                        if reduced is not None and self.check_tool_crash(
                            [opt_cmd] + reduced + ["-disable-output", tmp_ir_name]
                        ):
                            opt_args = reduced
                        opt_args = self.reduce_tool_args(
                            opt_cmd, opt_args, tmp_ir_name, extra_args=["-disable-output"]
                        )
                        shutil.copy(tmp_ir_name, ir_file)
                        return (opt_cmd, opt_args)
                    print("Crash does not reproduce with opt")
                else:
                    print("clang -emit-llvm -disable-llvm-passes did not complete")
        finally:
            if os.path.exists(tmp_ir_name):
                os.remove(tmp_ir_name)

        self.expected_output = orig_expected_output
        return None

    def run_reduce_pipeline(self, ir_file, opt_args):
        """Run reduce_pipeline.py to narrow down the failing opt pipeline.

        Returns a new list of opt args (with -passes=...) on success, replacing
        the -O level, or None if reduce_pipeline is unavailable or failed.
        Overwrites `ir_file` with the (possibly reduced) intermediate IR.
        """
        if not reduce_pipeline_cmd:
            return None

        passes = "default<O2>"
        extra = []
        for a in opt_args:
            m = re.match(r"^-O([0-3sz])$", a)
            if m:
                passes = "default<O%s>" % m.group(1)
            else:
                extra.append(a)

        output_file = os.path.splitext(ir_file)[0] + ".pipeline.ll"
        print("\nRunning reduce_pipeline.py to reduce the opt pipeline...")
        cmd = [
            sys.executable,
            reduce_pipeline_cmd,
            "--opt-binary=" + opt_cmd,
            "--input=" + ir_file,
            "--output=" + output_file,
            "--passes=" + passes,
        ] + extra
        verbose_print(quote_cmd(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if verbose:
            sys.stdout.write(result.stdout)
            sys.stderr.write(result.stderr)
        if result.returncode != 0:
            print("reduce_pipeline.py failed; keeping original pipeline")
            return None

        reduced_passes = passes
        for line in reversed(result.stdout.splitlines()):
            m = re.match(r'^-passes="(.*)"$', line)
            if m:
                reduced_passes = m.group(1)
                break

        if os.path.isfile(output_file):
            shutil.move(output_file, ir_file)

        print("Reduced opt pipeline:", reduced_passes)
        return ["-passes=" + reduced_passes]

    def write_llvm_reduce_test(self, tool_cmd, tool_args):
        """Write an interestingness test for llvm-reduce.

        The test receives the candidate IR file as $1.
        """
        print("\nCreating llvm-reduce interestingness test...")

        symbolizer_env = ""
        if not self.needs_stack_trace:
            symbolizer_env = "export LLVM_DISABLE_SYMBOLIZATION=1"
        elif llvm_symbolizer_cmd:
            symbolizer_env = "export LLVM_SYMBOLIZER_PATH=%s" % shlex.quote(
                llvm_symbolizer_cmd
            )

        invocation = quote_cmd([tool_cmd] + tool_args) + ' "$1"'

        output = """#!/bin/bash
%s
if %s >& "$(dirname "$0")/t.log" ; then
  exit 1
fi
""" % (
            symbolizer_env,
            invocation,
        )

        for msg in self.expected_output:
            output += 'grep -F %s "$(dirname "$0")/t.log" || exit 1\n' % shlex.quote(
                msg
            )

        write_to_script(output, self.testfile)

    def run_llvm_reduce(self, tool_cmd, tool_args, ir_file):
        self.write_llvm_reduce_test(tool_cmd, tool_args)

        testfile_abs = os.path.abspath(self.testfile)
        returncode = subprocess.call([testfile_abs, ir_file], stdout=subprocess.DEVNULL)
        if returncode:
            sys.exit("The interestingness test does not pass for the original IR file.")

        print("\nRunning llvm-reduce...")
        extra_flags = []
        if self.needs_stack_trace:
            extra_flags.append("--preserve-debug-environment")

        full_cmd = (
            [
                llvm_reduce_cmd,
                "--test=" + testfile_abs,
                "-o",
                ir_file,
            ]
            + self.llvm_reduce_flags
            + extra_flags
            + [ir_file]
        )
        verbose_print(quote_cmd(full_cmd))
        try:
            subprocess.check_call(full_cmd)
        except KeyboardInterrupt:
            print("\n\nctrl-c detected, killed llvm-reduce")
        except subprocess.CalledProcessError as e:
            print("llvm-reduce failed:", e)
            return

        reduced_cmd = quote_cmd([tool_cmd] + tool_args + [ir_file])
        write_to_script(reduced_cmd, self.crash_script)
        print("Reduced command:", reduced_cmd)

    @staticmethod
    def filter_args(
        args, opts_equal=[], opts_startswith=[], opts_one_arg_startswith=[]
    ):
        result = []
        skip_next = False
        for arg in args:
            if skip_next:
                skip_next = False
                continue
            if any(arg == a for a in opts_equal):
                continue
            if any(arg.startswith(a) for a in opts_startswith):
                continue
            if any(arg.startswith(a) for a in opts_one_arg_startswith):
                skip_next = True
                continue
            result.append(arg)
        return result

    def try_remove_args(self, args, msg=None, extra_arg=None, **kwargs):
        new_args = self.filter_args(args, **kwargs)

        if extra_arg:
            if extra_arg in new_args:
                new_args.remove(extra_arg)
            new_args.append(extra_arg)

        if new_args != args and self.check_expected_output(args=new_args):
            if msg:
                verbose_print(msg)
            return new_args
        return args

    def try_remove_arg_by_index(self, args, index):
        new_args = args[:index] + args[index + 1 :]
        removed_arg = args[index]

        # Heuristic for grouping arguments:
        # remove next argument if it doesn't start with "-"
        if index < len(new_args) and not new_args[index].startswith("-"):
            del new_args[index]
            removed_arg += " " + args[index + 1]

        if self.check_expected_output(args=new_args):
            verbose_print("Removed", removed_arg)
            return new_args, index
        return args, index + 1

    def simplify_clang_args(self):
        """Simplify clang arguments before running C-Vise to reduce the time the
        interestingness test takes to run.
        """
        print("\nSimplifying the clang command...")
        new_args = self.clang_args

        # Remove the color diagnostics flag to make it easier to match error
        # text.
        new_args = self.try_remove_args(
            new_args,
            msg="Removed -fcolor-diagnostics",
            opts_equal=["-fcolor-diagnostics"],
        )

        # Remove some clang arguments to speed up the interestingness test
        new_args = self.try_remove_args(
            new_args,
            msg="Removed debug info options",
            opts_startswith=["-gcodeview", "-debug-info-kind=", "-debugger-tuning="],
        )

        new_args = self.try_remove_args(
            new_args, msg="Removed --show-includes", opts_startswith=["--show-includes"]
        )
        # Not suppressing warnings (-w) sometimes prevents the crash from occurring
        # after preprocessing
        new_args = self.try_remove_args(
            new_args,
            msg="Replaced -W options with -w",
            extra_arg="-w",
            opts_startswith=["-W"],
        )
        new_args = self.try_remove_args(
            new_args,
            msg="Replaced optimization level with -O0",
            extra_arg="-O0",
            opts_startswith=["-O"],
        )

        # Try to remove compilation steps
        new_args = self.try_remove_args(
            new_args, msg="Added -emit-llvm", extra_arg="-emit-llvm"
        )
        new_args = self.try_remove_args(
            new_args, msg="Added -fsyntax-only", extra_arg="-fsyntax-only"
        )

        # Try to make implicit int an error for more sensible test output
        new_args = self.try_remove_args(
            new_args,
            msg="Added -Werror=implicit-int",
            opts_equal=["-w"],
            extra_arg="-Werror=implicit-int",
        )

        self.clang_args = new_args
        verbose_print("Simplified command:", quote_cmd(self.get_crash_cmd()))

    def reduce_clang_args(self):
        """Minimize the clang arguments after running C-Vise, to get the smallest
        command that reproduces the crash on the reduced file.
        """
        print("\nReducing the clang crash command...")

        new_args = self.clang_args

        # Remove some often occurring args
        new_args = self.try_remove_args(
            new_args, msg="Removed -D options", opts_startswith=["-D"]
        )
        new_args = self.try_remove_args(
            new_args, msg="Removed -D options", opts_one_arg_startswith=["-D"]
        )
        new_args = self.try_remove_args(
            new_args, msg="Removed -I options", opts_startswith=["-I"]
        )
        new_args = self.try_remove_args(
            new_args, msg="Removed -I options", opts_one_arg_startswith=["-I"]
        )
        new_args = self.try_remove_args(
            new_args, msg="Removed -W options", opts_startswith=["-W"]
        )

        # Remove other cases that aren't covered by the heuristic
        new_args = self.try_remove_args(
            new_args, msg="Removed -mllvm", opts_one_arg_startswith=["-mllvm"]
        )

        i = 0
        while i < len(new_args):
            new_args, i = self.try_remove_arg_by_index(new_args, i)

        self.clang_args = new_args

        reduced_cmd = quote_cmd(self.get_crash_cmd())
        write_to_script(reduced_cmd, self.crash_script)
        print("Reduced command:", reduced_cmd)

    def run_creduce(self):
        full_creduce_cmd = (
            [creduce_cmd] + self.creduce_flags + [self.testfile, self.file_to_reduce]
        )
        print("\nRunning C reduction tool...")
        verbose_print(quote_cmd(full_creduce_cmd))
        try:
            p = subprocess.Popen(full_creduce_cmd)
            p.communicate()
        except KeyboardInterrupt:
            # Hack to kill C-Reduce because it jumps into its own pgid
            print("\n\nctrl-c detected, killed reduction tool")
            p.kill()


def main():
    global verbose
    global creduce_cmd
    global clang_cmd
    global llc_cmd
    global opt_cmd
    global llvm_reduce_cmd
    global reduce_pipeline_cmd

    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        "crash_script",
        type=str,
        nargs=1,
        help="Name of the script that generates the crash.",
    )
    parser.add_argument(
        "file_to_reduce", type=str, nargs=1, help="Name of the file to be reduced."
    )
    parser.add_argument(
        "--llvm-bin", dest="llvm_bin", type=str, help="Path to the LLVM bin directory."
    )
    parser.add_argument(
        "--clang",
        dest="clang",
        type=str,
        help="The path to the `clang` executable. "
        "By default uses the llvm-bin directory.",
    )
    parser.add_argument(
        "--llc",
        dest="llc",
        type=str,
        help="The path to the `llc` executable. "
        "By default uses the llvm-bin directory.",
    )
    parser.add_argument(
        "--opt",
        dest="opt",
        type=str,
        help="The path to the `opt` executable. "
        "By default uses the llvm-bin directory.",
    )
    parser.add_argument(
        "--llvm-reduce",
        dest="llvm_reduce",
        type=str,
        help="The path to the `llvm-reduce` executable. "
        "By default uses the llvm-bin directory. Required to reduce IR-level crashes.",
    )
    parser.add_argument(
        "--reduce-pipeline",
        dest="reduce_pipeline",
        type=str,
        help="The path to `reduce_pipeline.py`. "
        "Default: llvm/utils/reduce_pipeline.py relative to this script. "
        "Used to reduce the opt pass pipeline for middle-end crashes.",
    )
    parser.add_argument(
        "--creduce",
        dest="creduce",
        type=str,
        help="The path to the `creduce` or `cvise` executable. "
        "Required if neither `creduce` nor `cvise` are on PATH.",
    )
    parser.add_argument(
        "--no-llvm-reduce",
        dest="no_llvm_reduce",
        action="store_true",
        help="Skip IR-level reduction with llvm-reduce, always run C-Vise/creduce.",
    )
    parser.add_argument(
        "--creduce-flag",
        dest="extra_creduce_flags",
        action="append",
        default=[],
        help="Extra flags to pass to creduce/cvise. Can be specified multiple times.",
    )
    parser.add_argument(
        "--llvm-reduce-flag",
        dest="llvm_reduce_flags",
        action="append",
        default=[],
        help="Extra flags to pass to llvm-reduce. Can be specified multiple times.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    verbose = args.verbose
    llvm_bin = os.path.abspath(args.llvm_bin) if args.llvm_bin else None
    creduce_cmd = check_cmd("cvise", None, args.creduce, return_none_if_not_found=True)
    if not creduce_cmd:
        creduce_cmd = check_cmd("creduce", None, args.creduce)
    clang_cmd = check_cmd("clang", llvm_bin, args.clang)
    if not llvm_bin and clang_cmd:
        llvm_bin = os.path.dirname(clang_cmd)

    global llvm_symbolizer_cmd
    llvm_symbolizer_cmd = check_cmd(
        "llvm-symbolizer", llvm_bin, return_none_if_not_found=True
    )
    if llvm_symbolizer_cmd:
        os.environ["LLVM_SYMBOLIZER_PATH"] = llvm_symbolizer_cmd

    llc_cmd = check_cmd("llc", llvm_bin, args.llc, return_none_if_not_found=True)
    opt_cmd = check_cmd("opt", llvm_bin, args.opt, return_none_if_not_found=True)
    llvm_reduce_cmd = check_cmd(
        "llvm-reduce", llvm_bin, args.llvm_reduce, return_none_if_not_found=True
    )
    if args.reduce_pipeline:
        reduce_pipeline_cmd = os.path.abspath(args.reduce_pipeline)
        if not os.path.isfile(reduce_pipeline_cmd):
            sys.exit("ERROR: %s does not exist" % reduce_pipeline_cmd)
    else:
        default_rp = os.path.normpath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "..",
                "llvm",
                "utils",
                "reduce_pipeline.py",
            )
        )
        reduce_pipeline_cmd = default_rp if os.path.isfile(default_rp) else None

    crash_script = check_file(args.crash_script[0])
    file_to_reduce = check_file(args.file_to_reduce[0])

    r = Reduce(
        crash_script,
        file_to_reduce,
        args.extra_creduce_flags,
        args.llvm_reduce_flags,
    )

    if not args.no_llvm_reduce and llvm_reduce_cmd and (llc_cmd or opt_cmd):
        ir_file = os.path.splitext(file_to_reduce)[0] + ".reduced.ll"
        ir_crash = r.try_llvm_ir_crash(ir_file)
        if ir_crash:
            tool_cmd, tool_args = ir_crash
            r.run_llvm_reduce(tool_cmd, tool_args, ir_file)
            return
        print(
            "\nCould not reproduce crash at the IR level, "
            "falling back to source-level reduction."
        )
    elif not args.no_llvm_reduce and not llvm_reduce_cmd:
        verbose_print(
            "llvm-reduce not found; skipping IR-level reduction. "
            "Pass --llvm-reduce or --llvm-bin to enable it."
        )

    r.prepare_source_reduction()
    r.simplify_clang_args()
    r.write_interestingness_test()
    r.clang_preprocess()
    r.run_creduce()
    r.reduce_clang_args()


if __name__ == "__main__":
    main()
