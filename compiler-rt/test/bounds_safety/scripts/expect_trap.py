#!/usr/bin/env python3
"""Runs a binary and verify a trap occurs in it as expected.

It supports running in two modes:

Direct mode: Run binary directly. Succeeds if process is killed by a signal
             (exit code < 0).
Debugger mode: Run binary under debugger. Succeeds if process stops with a
               bounds-safety trap and the trap location and message match the
               verify comments in the provided source file.

Verify comment format for unique or soft traps:
  // prefix@+N{reason}   - trap expected N lines below
  // prefix@-N{reason}   - trap expected N lines above
  // prefix{reason}      - trap expected on the same line

Verify comment format for merged traps:
  // prefix-merged{function_name} - trap occurs in function `function_name`
"""

import argparse
import logging
import os
import re
import subprocess
import sys

log = logging.getLogger("expect_trap")

SOFT_TRAP_MARKER = "***HIT BOUNDS SAFETY SOFT TRAP***:"


def run_direct(binary, args):
    result = subprocess.run([binary] + args)
    if result.returncode < 0:
        return 0
    log.error(
        "expected trap but process exited with status %d", result.returncode
    )
    return 1


def run_direct_soft_trap(binary, args):
    """Verify a soft trap fires in direct mode."""
    result = subprocess.run([binary] + args, capture_output=True, text=True)
    if result.returncode != 0:
        log.error(
            "expected soft trap but process exited with status %d",
            result.returncode,
        )
        return 1
    if SOFT_TRAP_MARKER not in result.stderr:
        log.error("soft trap marker not found in stderr")
        log.error("stderr: %s", result.stderr)
        return 1
    return 0


# ---------------------------------------------------------------------------
# Verify-comment parsing
# ---------------------------------------------------------------------------

def parse_verify_comments(source_path, prefix):
    """Parse verify comments from a source file.

    Returns a list of (expected_line, expected_message) tuples.
    """
    pattern = re.compile(
        r"//\s*" + re.escape(prefix) + r"(?:@([+-]?\d+))?\{(.+)\}"
    )
    results = []
    with open(source_path, "r") as f:
        for line_no, line in enumerate(f, start=1):
            m = pattern.search(line)
            if m:
                offset = int(m.group(1)) if m.group(1) is not None else 0
                expected_line = line_no + offset
                expected_message = m.group(2)
                results.append((expected_line, expected_message))
    return results


def parse_merged_verify_comments(source_path, prefix):
    """Parse merged-trap verify comments from a source file.

    Format: // prefix-merged{function_name}
    Returns a list of function_name strings.
    """
    pattern = re.compile(
        r"//\s*" + re.escape(prefix) + r"-merged\{(.+)\}"
    )
    results = []
    with open(source_path, "r") as f:
        for line_no, line in enumerate(f, start=1):
            m = pattern.search(line)
            if m:
                results.append(m.group(1))
    return results


# ---------------------------------------------------------------------------
# LLDB helpers
# ---------------------------------------------------------------------------

BOUNDS_CHECK_PREFIX = "Bounds check failed: "
SOFT_BOUNDS_CHECK_PREFIX = "Soft Bounds check failed: "


def strip_trap_prefix(description):
    """Strip the bounds-safety trap prefix from a stop description.

    Returns (stripped_message, True) on success, (description, False) on failure.
    """
    for pfx in (BOUNDS_CHECK_PREFIX, SOFT_BOUNDS_CHECK_PREFIX):
        if description.startswith(pfx):
            return description[len(pfx):], True
    return description, False


# Expected trap instructions per architecture (mnemonic, operands).
EXPECTED_TRAP_INSTRUCTIONS = {
    "arm64":  ("brk", "#0x5519"),
    "arm64e": ("brk", "#0x5519"),
    "x86_64": ("ud1l", "0x19(%eax), %eax"),
    "x86_64h": ("ud1l", "0x19(%eax), %eax"),
}


def check_trap_instruction(lldb, thread, process, arch):
    """Verify frame 0 stopped on the expected trap instruction.

    Returns (True, "") on success, (False, detail_string) on failure.
    """
    expected = EXPECTED_TRAP_INSTRUCTIONS.get(arch)
    if expected is None:
        return False, "unsupported architecture '{}' for trap instruction check".format(arch)

    expected_mnemonic, expected_operands = expected
    target = process.GetTarget()
    frame0 = thread.GetFrameAtIndex(0)
    addr = frame0.GetPCAddress()

    inst_list = target.ReadInstructions(addr, 1)
    if inst_list.GetSize() == 0:
        return False, "could not disassemble instruction at PC"

    inst = inst_list.GetInstructionAtIndex(0)
    mnemonic = inst.GetMnemonic(target)
    operands = inst.GetOperands(target)

    if mnemonic != expected_mnemonic or operands != expected_operands:
        return False, (
            "unexpected trap instruction\n"
            "  expected: {} {}\n"
            "  actual:   {} {}".format(
                expected_mnemonic, expected_operands, mnemonic, operands)
        )

    return True, f"{mnemonic} {operands}"


def find_test_frame(thread, source_basename):
    """Walk stack frames to find first frame matching the test source.

    Skips frames with line number 0 (e.g. the trap function itself which
    has the source filename but no meaningful line info).
    """
    for frame in thread:
        line_entry = frame.GetLineEntry()
        if not line_entry.IsValid():
            continue
        if line_entry.GetLine() == 0:
            continue
        file_spec = line_entry.GetFileSpec()
        if file_spec.GetFilename() == source_basename:
            return frame, line_entry
    return None, None


def _stop_reason_name(lldb, reason):
    """Return a human-readable name for an LLDB stop reason enum value."""
    names = {getattr(lldb, a): a for a in dir(lldb) if a.startswith("eStopReason")}
    name = names.get(reason, "unknown")
    return "{} ({})".format(name, reason)


def log_stop_info(thread):
    """Log the stop reason and backtrace from a thread, if available."""
    if thread is None:
        return
    import lldb
    stop_reason = thread.GetStopReason()
    stop_desc = thread.GetStopDescription(256)
    log.info("Stop reason: %s", _stop_reason_name(lldb, stop_reason))
    log.info("Stop description: %s", stop_desc)
    log.info("Backtrace:")
    for f in thread:
        log.info("  %s", f)


def has_bs_instrumentation_plugin(lldb, debugger):
    """Check if the BoundsSafety instrumentation runtime plugin is available."""
    result = lldb.SBCommandReturnObject()
    debugger.GetCommandInterpreter().HandleCommand(
        "plugin list instrumentation-runtime.BoundsSafety", result)
    return result.Succeeded() and "[+] BoundsSafety" in result.GetOutput()


def extract_trap_msg_from_stack(thread, start_frame=0):
    """Extract trap message from __clang_trap_msg frame in the call stack.

    When the BoundsSafety instrumentation plugin or VerboseTrapFrameRecognizer
    is not available, the trap message can be recovered from a frame's
    function name, which has the format:
      __clang_trap_msg$<category>$<message>

    Searches from start_frame onwards.
    Returns (category, message) on success, (None, None) on failure.
    """
    num_frames = thread.GetNumFrames()
    # For hard traps the __clang_trap_msg should be on frame 0, for soft traps
    # it's frame 1.
    for i in range(start_frame, min(num_frames, start_frame + 2)):
        frame = thread.GetFrameAtIndex(i)
        if frame is None:
            continue
        func_name = frame.GetFunctionName()
        if func_name is None or not func_name.startswith("__clang_trap_msg$"):
            continue
        parts = func_name.split("$", 2)
        if len(parts) != 3:
            continue
        return parts[1], parts[2]
    return None, None


def check_soft_trap_no_plugin(lldb, process, expectation, verify_source):
    """Verify a soft trap when the instrumentation plugin is absent.

    The process is stopped at a manual breakpoint on __bounds_safety_soft_trap.
    Extract the trap message from the __clang_trap_msg frame in the stack.

    This can be removed once we no longer need to test using older LLDB's
    that are missing the instrumentation-runtime.BoundsSafety plugin.
    """
    state = process.GetState()

    if state == lldb.eStateExited:
        exit_status = process.GetExitStatus()
        log.error(
            "expected soft trap but process exited with status %d", exit_status
        )
        return 1

    if state != lldb.eStateStopped:
        log.error("unexpected process state: %s", state)
        return 1

    thread = process.GetSelectedThread()
    stop_reason = thread.GetStopReason()

    if stop_reason != lldb.eStopReasonBreakpoint:
        log.error("unexpected stop reason %s: %s",
                  stop_reason, thread.GetStopDescription(256))
        log_stop_info(thread)
        return 1

    category, message = extract_trap_msg_from_stack(thread)
    if category is None:
        log.error("could not extract trap message from call stack")
        log_stop_info(thread)
        return 1

    # Build a stop description matching the verify-comment format
    stop_desc = "Soft {}: {}".format(category, message)
    stripped_msg, ok = strip_trap_prefix(stop_desc)
    if not ok:
        log.error("could not strip trap prefix from: '%s'", stop_desc)
        log_stop_info(thread)
        return 1

    expected_line, expected_message = expectation

    source_basename = os.path.basename(verify_source)
    frame, line_entry = find_test_frame(thread, source_basename)
    if frame is None:
        log.error("no stack frame found in '%s'", source_basename)
        log_stop_info(thread)
        return 1

    actual_line = line_entry.GetLine()
    if actual_line != expected_line:
        log.error(
            "trap at line %d but expected line %d",
            actual_line, expected_line,
        )
        log_stop_info(thread)
        return 1

    if stripped_msg != expected_message:
        log.error(
            "trap message mismatch\n"
            "  expected: '%s'\n"
            "  actual:   '%s'",
            expected_message, stripped_msg,
        )
        log_stop_info(thread)
        return 1
    log.info("Found expected -fbounds-safety soft trap")
    log_stop_info(thread)
    return 0


def check_trap(lldb, process, expectation, verify_source, merged_traps=False,
               soft_traps=False):
    """Verify the process stopped with a bounds-safety trap. Returns 0 on success."""
    state = process.GetState()

    if state == lldb.eStateExited:
        exit_status = process.GetExitStatus()
        log.error(
            "expected trap but process exited with status %d", exit_status
        )
        return 1

    if state != lldb.eStateStopped:
        log.error("unexpected process state: %s", state)
        return 1

    # Process is stopped - verify it's a bounds-safety trap
    thread = process.GetSelectedThread()
    stop_reason = thread.GetStopReason()
    stop_desc = thread.GetStopDescription(256)
    arch = process.GetTarget().GetTriple().split("-")[0]

    # Determine the expected stop reason based on trap type
    if soft_traps:
        expected_stop_reason = lldb.eStopReasonInstrumentation
    else:
        expected_stop_reason = lldb.eStopReasonException

    if stop_reason != expected_stop_reason:
        log.error("unexpected stop reason %s: %s", stop_reason, stop_desc)
        log_stop_info(thread)
        return 1

    # For hard traps, verify the trap instruction encoding and handle merged
    # traps if necessary
    if not soft_traps:
        ok, detail = check_trap_instruction(lldb, thread, process, arch)
        if not ok:
            log.error("%s", detail)
            log_stop_info(thread)
            return 1
        else:
            log.info('Stopped at expected -fbounds-safety trap instruction: %s', detail)

        if merged_traps:
            # Check function name
            expected_func = expectation  # just a string
            source_basename = os.path.basename(verify_source)
            frame, line_entry = find_test_frame(thread, source_basename)
            if frame is None:
                log.error("no stack frame found in '%s'", source_basename)
                log_stop_info(thread)
                return 1
            actual_func = frame.GetFunctionName()
            if actual_func != expected_func:
                log.error(
                    "function name mismatch\n"
                    "  expected: '%s'\n"
                    "  actual:   '%s'",
                    expected_func, actual_func,
                )
                log_stop_info(thread)
                return 1
            log.info("Found expected -fbounds-safety trap in '%s'", actual_func)
            log_stop_info(thread)
            return 0

    assert not merged_traps

    # Verify the stop description indicates a bounds-safety hard or soft trap.
    # When using older LLDB versions the VerboseFrapFrameRecognizer might not
    # fire. When that happens the stop description will be a raw exception
    # description (e.g. "EXC_BAD_INSTRUCTION (code=EXC_I386_INVOP,
    # subcode=0x0)"). Fall back to extracting the trap message from the
    # __clang_trap_msg function name in the call stack.
    if not (
        stop_desc.startswith(BOUNDS_CHECK_PREFIX)
        or stop_desc.startswith(SOFT_BOUNDS_CHECK_PREFIX)
    ):
        # TODO: Remove this once we no longer need to support older LLDB
        # versions.
        category, message = extract_trap_msg_from_stack(thread)
        if category is None:
            log.error(
                "stop description does not indicate a bounds-safety trap: '%s'",
                stop_desc,
            )
            log_stop_info(thread)
            return 1
        log.warning(
            "stop description ('%s') does not indicate a bounds-safety trap. "
            "Extracting trap message from call stack instead. This likely "
            "means an older LLDB is being used with a VerboseTrapFrame "
            "recognizer that is out of sync with the compiler.",
            stop_desc,
        )
        if soft_traps:
            stop_desc = "Soft {}: {}".format(category, message)
        else:
            stop_desc = "{}: {}".format(category, message)

    # Check location and message
    stripped_msg, ok = strip_trap_prefix(stop_desc)
    if not ok:
        log.error("could not strip trap prefix from: '%s'", stop_desc)
        log_stop_info(thread)
        return 1

    expected_line, expected_message = expectation

    # Find the frame in the test source
    source_basename = os.path.basename(verify_source)
    frame, line_entry = find_test_frame(thread, source_basename)
    if frame is None:
        log.error("no stack frame found in '%s'", source_basename)
        log_stop_info(thread)
        return 1

    actual_line = line_entry.GetLine()
    if actual_line != expected_line:
        log.error(
            "trap at line %d but expected line %d",
            actual_line, expected_line,
        )
        log_stop_info(thread)
        return 1

    if stripped_msg != expected_message:
        log.error(
            "trap message mismatch\n"
            "  expected: '%s'\n"
            "  actual:   '%s'",
            expected_message, stripped_msg,
        )
        log_stop_info(thread)
        return 1

    log.info("Found expected -fbounds-safety %s trap", 'soft' if soft_traps else 'hard')
    log_stop_info(thread)
    return 0


def resume_and_expect_clean_exit(lldb, process):
    """Resume a process stopped at a soft trap and require a clean exit.

    Soft traps are non-fatal: after the trap is validated the process must run
    to completion and exit with status 0. This is what distinguishes a soft
    trap from a hard trap and is what catches miscompiles that turn a soft trap
    into a fatal one. Assumes exactly one soft trap fires before exit (we do not
    yet support multiple verify expectations); a second stop is reported as a
    failure rather than silently passing.
    """
    # LLDBProcessContextManager runs in synchronous mode (SetAsync(False)), so
    # Continue() blocks until the process next stops or exits.
    error = process.Continue()
    if error.Fail():
        log.error("failed to resume process after soft trap: %s", error)
        return 1

    state = process.GetState()
    if state != lldb.eStateExited:
        log.error(
            "process did not exit after soft trap (state=%s); the soft trap "
            "may have become fatal", state,
        )
        log_stop_info(process.GetSelectedThread())
        return 1

    exit_status = process.GetExitStatus()
    if exit_status != 0:
        log.error(
            "process exited with status %d after soft trap (expected 0)",
            exit_status,
        )
        return 1

    log.info("Process exited cleanly after soft trap")
    return 0


def run_debugger(binary, args, lldb_python_path, verify_source, verify_prefix,
                  merged_traps=False, soft_traps=False):
    # Parse and validate verify comments before launching the debugger so we
    # fail fast on malformed or missing expectations.
    if merged_traps:
        expectations = parse_merged_verify_comments(
            verify_source, verify_prefix)
        if not expectations:
            log.error(
                "no merged verify comments with prefix '%s' found in '%s'",
                verify_prefix, verify_source,
            )
            return 1
        if len(expectations) > 1:
            log.error(
                "multiple merged verify comments with prefix '%s' found "
                "in '%s' (expected exactly one)",
                verify_prefix, verify_source,
            )
            return 1
        expectation = expectations[0]  # function name string
    else:
        expectations = parse_verify_comments(verify_source, verify_prefix)
        if not expectations:
            log.error(
                "no verify comments with prefix '%s' found in '%s'",
                verify_prefix, verify_source,
            )
            return 1
        if len(expectations) > 1:
            log.error(
                "multiple verify comments with prefix '%s' found in '%s' "
                "(expected exactly one)",
                verify_prefix, verify_source,
            )
            return 1
        expectation = expectations[0]

    from lldb_utils import LLDBLaunchError, LLDBProcessContextManager, import_lldb

    lldb = import_lldb(lldb_python_path)

    with LLDBProcessContextManager(lldb, binary, args) as ctx:
        # For soft traps without the instrumentation plugin, set a manual
        # breakpoint before launching so we stop at the trap function.
        has_plugin = False
        if soft_traps:
            has_plugin = has_bs_instrumentation_plugin(lldb, ctx.debugger)
            if not has_plugin:
                log.info("BoundsSafety instrumentation plugin not available, "
                         "setting manual breakpoint")
                ctx.target.BreakpointCreateByName("__bounds_safety_soft_trap")

        try:
            ctx.launch()
        except LLDBLaunchError as e:
            log.error("%s", e)
            return 1

        if soft_traps and not has_plugin:
            rc = check_soft_trap_no_plugin(lldb, ctx.process, expectation,
                                           verify_source)
        else:
            rc = check_trap(lldb, ctx.process, expectation, verify_source,
                            merged_traps, soft_traps)
        if rc != 0:
            return rc

        # Hard/merged traps: stopping at the trap is the entire check.
        if not soft_traps:
            return 0

        # Soft traps are non-fatal: after validating the trap the process must
        # resume and run to a clean exit.
        return resume_and_expect_clean_exit(lldb, ctx.process)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--log-level", default="info",
                        choices=["debug", "info", "warning", "error"],
                        help="Logging level (default: warning)")
    parser.add_argument("--use-debugger", action="store_true")
    parser.add_argument("--lldb-python-path", default=None)
    parser.add_argument("--verify-prefix", default="expect-trap",
                        help="Prefix for verify comments (default: expect-trap)")
    parser.add_argument("--merged-traps", action="store_true",
                        help="Merged-traps mode: verify function name only")
    parser.add_argument("--soft-traps", action="store_true",
                        help="Soft-traps mode: expect non-fatal traps")
    parser.add_argument("source_file_path",
                        help="Source file with verify comments")
    parser.add_argument("binary")
    parser.add_argument("args", nargs="*")

    parsed = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, parsed.log_level.upper()),
        format="%(name)s: %(levelname)s: %(message)s",
    )

    if not os.path.isfile(parsed.binary):
        log.error("binary not found: '%s'", parsed.binary)
        return 1
    if not os.path.isfile(parsed.source_file_path):
        log.error("source file not found: '%s'", parsed.source_file_path)
        return 1

    if parsed.use_debugger:
        return run_debugger(
            parsed.binary,
            parsed.args,
            parsed.lldb_python_path,
            parsed.source_file_path,
            parsed.verify_prefix,
            parsed.merged_traps,
            parsed.soft_traps,
        )
    if parsed.soft_traps:
        return run_direct_soft_trap(parsed.binary, parsed.args)
    return run_direct(parsed.binary, parsed.args)


if __name__ == "__main__":
    sys.exit(main())
