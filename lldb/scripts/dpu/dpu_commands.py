import re
import sys
import subprocess
import os
import lldb

def set_debug_mode(debugger, rank, debug_mode):
    debugger.HandleCommand(
        "print (void) hw_set_debug_mode((dpu_rank_t *)"
        + rank.GetValue()
        + ", "
        + str(debug_mode)
        + ")")

def get_dpu_from_command(command, debugger, target):
    dpu_addr = 0
    try:
        dpu_addr = int(command, 16)
    except:
        print("Interpreting command: '" + command + "'")
        return_obj = lldb.SBCommandReturnObject()
        debugger.GetCommandInterpreter().HandleCommand("p " + command,
                                                       return_obj)
        if return_obj.GetStatus() != lldb.eReturnStatusSuccessFinishResult:
            print("Could not interpret command '" + command + "'")
            sys.exit(1)
        dpu_addr = int(re.search('dpu_t \*.*= (.+)', return_obj.GetOutput())
                       .group(1),16)
    print("Attaching to dpu: '" + hex(dpu_addr) + "'")
    return target.CreateValueFromExpression(
        "dpu_to_attach","(struct dpu_t *)" + str(dpu_addr))


def dpu_attach(debugger, command, result, internal_dict):
    '''
    usage: dpu_attach <struct dpu_t *>
    '''
    target = debugger.GetSelectedTarget()
    dpu = get_dpu_from_command(command, debugger, target)
    assert dpu.IsValid()

    slice_id = dpu.GetChildMemberWithName("slice_id").GetValueAsUnsigned()
    dpu_id = dpu.GetChildMemberWithName("dpu_id").GetValueAsUnsigned()

    rank = dpu.GetChildMemberWithName("rank")
    assert rank.IsValid()
    slice_info = rank.GetChildMemberWithName("runtime") \
        .GetChildMemberWithName("control_interface") \
        .GetChildMemberWithName("slice_info").GetChildAtIndex(slice_id)
    assert slice_info.IsValid()
    slice_target = slice_info.GetChildMemberWithName("slice_target")
    assert slice_target.IsValid()

    structure_value = slice_info.GetChildMemberWithName("structure_value") \
        .GetValueAsUnsigned()
    slice_target_type = slice_target.GetChildMemberWithName("type") \
        .GetValueAsUnsigned()
    slice_target_dpu_group_id = slice_target.GetChildMemberWithName("dpu_id") \
        .GetValueAsUnsigned()
    slice_target = (slice_target_type << 32) + slice_target_dpu_group_id

    rank_path = str(
        rank.GetChildMemberWithName("description")
        .GetChildMemberWithName("_internals")
        .GetChildMemberWithName("data")
        .Cast(target.FindFirstType("hw_dpu_rank_allocation_parameters_t"))
        .GetChildMemberWithName("rank_fs")
        .GetChildMemberWithName("rank_path")
        )
    region_id = int(re.search('dpu_region(.+)/', rank_path).group(1))
    rank_id = int(re.search('dpu_rank(.+)"', rank_path).group(1))

    pid = dpu_id + (slice_id << 16) + (rank_id << 32) + (region_id << 48) \
        + 0x80000000

    program_path = \
        re.search('"(.+)"',
                  str(dpu.GetChildMemberWithName("runtime_context")
                      .GetChildMemberWithName("program_path"))).group(1)

    set_debug_mode(debugger, rank, 1)

    lldb_server_dpu_env = os.environ.copy()
    lldb_server_dpu_env["UPMEM_LLDB_STRUCTURE_VALUE"] = str(structure_value)
    lldb_server_dpu_env["UPMEM_LLDB_SLICE_TARGET"] = str(slice_target)
    subprocess.Popen(['lldb-server-dpu',
                      'gdbserver',
                      '--attach',
                      str(pid),
                      ':2066'],
                     env=lldb_server_dpu_env)

    target_dpu = \
        debugger.CreateTargetWithFileAndTargetTriple(program_path,
                                                     "dpu-upmem-dpurte")
    assert target_dpu.IsValid()

    listener = debugger.GetListener()
    error = lldb.SBError()
    process_dpu = target_dpu.ConnectRemote(listener,
                                           "connect://localhost:2066",
                                           "gdb-remote",
                                           error)
    assert process_dpu.IsValid()

    debugger.SetSelectedTarget(target)
    set_debug_mode(debugger, rank, 0)

    debugger.SetSelectedTarget(target_dpu)
