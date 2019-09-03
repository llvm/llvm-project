import re
import sys
import subprocess
import os
import lldb


def compute_dpu_pid(region_id, rank_id, slice_id, dpu_id):
    return dpu_id + (slice_id << 16) + (rank_id << 32) + (region_id << 48) \
        + 0x80000000


def get_value_from_command(debugger, command, base):
    return_obj = lldb.SBCommandReturnObject()
    debugger.GetCommandInterpreter().HandleCommand("p " + command, return_obj)
    if return_obj.GetStatus() != lldb.eReturnStatusSuccessFinishResult:
        return False, 0
    if return_obj.GetOutput() is None:
        return True, 0
    addr = int(re.search('.*= (.+)', return_obj.GetOutput()).group(1), base)
    return True, addr


def set_debug_mode(debugger, rank, debug_mode):
    success, unused = get_value_from_command(
        debugger,
        "hw_set_debug_mode((dpu_rank_t *)"
        + rank.GetValue() + ", " + str(debug_mode) + ")",
        10)
    return success


def get_object_from_command(command, debugger, target, name, object_type,
                            base):
    addr = 0
    try:
        addr = int(command, base)
    except:
        success, addr = get_value_from_command(debugger, command, base)
        if not(success):
            print("Could not interpret command '" + command + "'")
            sys.exit(1)
    return target.CreateValueFromExpression(
        name, "(" + object_type + ")" + str(addr))


def get_dpu_from_command(command, debugger, target):
    return get_object_from_command(
        command, debugger, target, "dpu", "struct dpu_t *", 16)


def get_rank_from_command(command, debugger, target):
    return get_object_from_command(
        command, debugger, target, "rank", "struct dpu_rank_t *", 16)


def get_region_id_and_rank_id(rank, target):
    hw_dpu_rank_allocation_parameters_type = \
        target.FindFirstType("hw_dpu_rank_allocation_parameters_t")
    if not(hw_dpu_rank_allocation_parameters_type.IsValid()):
        return -1, -1
    rank_path = str(
        rank.GetChildMemberWithName("description")
        .GetChildMemberWithName("_internals")
        .GetChildMemberWithName("data")
        .Cast(hw_dpu_rank_allocation_parameters_type)
        .GetChildMemberWithName("rank_fs")
        .GetChildMemberWithName("rank_path")
        )
    region_id = int(re.search('dpu_region(.+)/', rank_path).group(1))
    rank_id = int(re.search('dpu_rank(.+)"', rank_path).group(1))
    return region_id, rank_id


def get_dpu_program_path(dpu):
    program_path = dpu.GetChildMemberWithName("runtime_context") \
                      .GetChildMemberWithName("program_path")
    if program_path.GetChildAtIndex(0).GetValue() is None:
        return None
    return re.search('"(.+)"', str(program_path)).group(1)


def get_dpu_status(rank, slice_id, dpu_id):
    run_context = rank.GetChildMemberWithName("runtime") \
                      .GetChildMemberWithName("run_context")
    assert run_context.IsValid()
    dpus_running = run_context.GetChildMemberWithName("dpu_running")
    assert dpus_running.IsValid()
    dpus_in_fault = run_context.GetChildMemberWithName("dpu_in_fault")
    assert dpus_in_fault.IsValid()
    dpu_running = int(dpus_running.GetChildAtIndex(slice_id)
                      .GetValue())
    dpu_in_fault = int(dpus_in_fault.GetChildAtIndex(slice_id)
                       .GetValue())
    dpu_mask = 1 << dpu_id
    if (dpu_mask & dpu_running) != 0:
        return "RUNNING"
    elif (dpu_mask & dpu_in_fault) != 0:
        return "ERROR  "
    else:
        return "IDLE   "


def dpu_attach(debugger, command, result, internal_dict):
    '''
    usage: dpu_attach <struct dpu_t *>
    '''
    target = debugger.GetSelectedTarget()
    dpu = get_dpu_from_command(command, debugger, target)
    assert dpu.IsValid()
    print("Attaching to dpu '" + dpu.GetValue() + "'")

    program_path = get_dpu_program_path(dpu)
    if program_path is None:
        print("Could not find program loaded in dpu")
        sys.exit(1)

    rank = dpu.GetChildMemberWithName("rank")
    assert rank.IsValid()

    region_id, rank_id = get_region_id_and_rank_id(rank, target)
    if region_id == -1 or rank_id == -1:
        print("Could not attach to simulator (hardware only)")
        sys.exit(1)

    slice_id = dpu.GetChildMemberWithName("slice_id").GetValueAsUnsigned()
    dpu_id = dpu.GetChildMemberWithName("dpu_id").GetValueAsUnsigned()
    if get_dpu_status(rank, slice_id, dpu_id) != "RUNNING":
        print("Dpu is not running! Impossible to attach")
        sys.exit(1)

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

    pid = compute_dpu_pid(region_id, rank_id, slice_id, dpu_id)

    assert set_debug_mode(debugger, rank, 1)

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
    assert set_debug_mode(debugger, rank, 0)

    debugger.SetSelectedTarget(target_dpu)


def dpu_get(rank_addr, slice_id, dpu_id, debugger, target):
    return get_dpu_from_command("dpu_get((struct dpu_rank_t *)"
                                + str(rank_addr)
                                + ", " + str(slice_id)
                                + ", " + str(dpu_id) + ")",
                                debugger,
                                target)


def print_list(list, result):
    if result is None:
        return
    result.PutCString("\nADDR \t\t\tID \t\tSTATUS \t\tPROGRAM\n")
    for (dpu_addr, dpu_id, status, program) in list:
        result.PutCString(
            dpu_addr + " \t" + dpu_id + " \t" + status + " \t" + program)


def dpu_list(debugger, command, result, internal_dict):
    '''
    usage: dpu_list
    '''
    success, nb_allocated_rank = \
        get_value_from_command(
            debugger, "dpu_rank_handler_dpu_rank_list_size", 10)
    assert success

    target = debugger.GetSelectedTarget()
    result_list = []

    for each_rank in range(0, nb_allocated_rank):
        rank = get_rank_from_command(
            "dpu_rank_handler_dpu_rank_list[" + str(each_rank) + "]",
            debugger,
            target)
        assert rank.IsValid()
        rank_addr = rank.GetValue()
        if int(rank_addr, 16) == 0:
            continue

        region_id, rank_id = get_region_id_and_rank_id(rank, target)

        slice_id = 0
        dpu_id = 0
        dpu = dpu_get(rank_addr, slice_id, dpu_id, debugger, target)
        assert dpu.IsValid
        while int(dpu.GetValue(), 16) != 0:
            while int(dpu.GetValue(), 16) != 0:
                program_path = get_dpu_program_path(dpu)
                if program_path is None:
                    program_path = ""

                dpu_status = get_dpu_status(rank, slice_id, dpu_id)

                result_list.append(("'" + dpu.GetValue() + "'",
                                    str(region_id)
                                    + "/" + str(rank_id)
                                    + "/" + str(slice_id)
                                    + "/" + str(dpu_id),
                                    dpu_status,
                                    "'" + program_path + "'"))

                dpu_id = dpu_id + 1
                dpu = dpu_get(rank_addr, slice_id, dpu_id, debugger, target)

            dpu_id = 0
            slice_id = slice_id + 1
            dpu = dpu_get(rank_addr, slice_id, dpu_id, debugger, target)

        print_list(result_list, result)
        return result_list
