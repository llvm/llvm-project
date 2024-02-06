# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

import lldb

def breakpoint_handler(frame, bp_loc, internal_dict):
    name = frame.GetFunctionName()
    print(f"======> LLDB: func: {name}")
    module = frame.GetModule()
    filename = module.file.GetFilename()
    print(f"======> LLDB: file: {filename}")
    line = frame.GetLineEntry().GetLine()
    print(f"======> LLDB: file: {line}")
    parent = frame.get_parent_frame()

    expectation_val = parent.FindVariable("isDebuggerPresent")
    print(f"------ var: {expectation_val}")
    expectation_val = parent.FindVariable("isDebuggerPresent")
    print(f"------ var val: {expectation_val.value}")

    # expectation_val = frame.FindVariable("isDebuggerPresent")
    # print(f"------ var: {frame.variables}")

    # for var in frame.variables:
    #     print(f"     far: {var.name}")

    # print(f"typeof: {type(expectation_val.value)}")
    # print(f"typeof: {type(expectation_val.type)}")
    # print(f"typeof: {type(expectation_val.value_type)}")
    # print(f"expectation_val.value: {expectation_val.value}")
    # print(f"expectation_val.value 2: {expectation_val}")
    # value_value = expectation_val.value

    # if value_value is None:
    #     print(" ---- None")
    # else:
    #     print(f"---- Yes {value_value}")


    if expectation_val.value == "true":
        print(" ---- yes")
    else:
        print("---- no")

    if expectation_val is None or expectation_val.value == "false":
        # global test_failures

        print("FAIL: " + filename + ":" + str(line))
        print("`isDebuggerPresent` value is `false`, value should be `true`")

        # test_failures += 1
    else:
        print("PASS: " + filename + ":" + str(line))


def __lldb_init_module(debugger, internal_dict):
    print("-------- START")
    target = debugger.GetSelectedTarget()
    test_bp = target.BreakpointCreateByName("StopForDebugger")
    test_bp.SetScriptCallbackFunction("is_debugger_present_with_debugger_lldb.breakpoint_handler")
    test_bp.enabled = True
    print("------- END")


# def main():
#     """Main entry point"""
#     print("==============> Hello LLDB Python")
#     # Disable terminal paging

#     # gdb.execute("set height 0")
#     # gdb.execute("set python print-stack full")

#     # CheckResult()
#     # test_bp = gdb.Breakpoint("StopForDebugger")
#     # test_bp.enabled = True
#     # test_bp.silent = True
#     # test_bp.commands = """check_is_debugger_present
#     # continue"""

#     # # "run" won't return if the program exits; ensure the script regains control.

#     # gdb.events.exited.connect(exit_handler)
#     # gdb.execute("run")

#     # # If the program didn't exit, something went wrong, but we don't
#     # # know what. Fail on exit.

#     # test_failures += 1
#     # exit_handler(None)

#     # print(f"Test failures count: {test_failures}")

# if __name__ == "__main__":
#     main()
