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
    if not(run_context.IsValid()):
        return "UNKNOWN"
    dpus_running = run_context.GetChildMemberWithName("dpu_running")
    if not(dpus_running.IsValid()):
        return "UNKNOWN"
    dpus_in_fault = run_context.GetChildMemberWithName("dpu_in_fault")
    if not(dpus_in_fault.IsValid()):
        return "UNKNOWN"
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


def break_to_next_boot_and_get_dpus(debugger, target):
    launch_rank_function = "dpu_launch_thread_on_rank"
    launch_dpu_function = "dpu_launch_thread_on_dpu"
    launch_rank_bkp = \
        target.BreakpointCreateByName(launch_rank_function)
    launch_dpu_bkp = target.BreakpointCreateByName(launch_dpu_function)

    process = target.GetProcess()
    process.Continue()

    target.BreakpointDelete(launch_rank_bkp.GetID())
    target.BreakpointDelete(launch_dpu_bkp.GetID())

    dpu_list = []
    thread = process.GetSelectedThread()
    current_frame = 0
    frame = thread.GetFrameAtIndex(current_frame)
    function_name = frame.GetFunctionName()
    if thread.GetStopReason() != lldb.eStopReasonBreakpoint:
        return dpu_list, frame

    # Look for frame from which the host needs to step out to complete the boot
    # of the dpu
    nb_frames = thread.GetNumFrames()
    while ((function_name != launch_rank_function)
           and (function_name != launch_dpu_function)):
        current_frame = current_frame + 1
        if current_frame == nb_frames:
            return dpu_list, frame
        frame = thread.GetFrameAtIndex(current_frame)
        function_name = frame.GetFunctionName()

    thread.SetSelectedFrame(current_frame)
    if function_name == launch_rank_function:
        nb_ci = get_dpu_from_command(
            "(int)(rank->description->topology.nr_of_control_interfaces)",
            debugger, target)
        nb_dpu_per_ci = get_dpu_from_command(
            "(int)(rank->description->topology.nr_of_dpus_per_control_interface)",
            debugger,target)
        nb_dpu = int(nb_ci.GetValue(), 16) * int(nb_dpu_per_ci.GetValue(), 16)
        for each_dpu in range(0, nb_dpu):
            dpu_list.append(get_dpu_from_command("&rank->dpus["
                                                 + str(each_dpu) + "]",
                                                 debugger, target))
    elif function_name == launch_dpu_function:
        dpu_list.append(get_dpu_from_command("dpu", debugger, target))

    return dpu_list, frame


def dpu_attach_on_boot(debugger, command, result, internal_dict):
    '''
    usage: dpu_attach_on_boot [<struct dpu_t *>]
    '''
    target = debugger.GetSelectedTarget()

    dpus_booting, host_frame = \
        break_to_next_boot_and_get_dpus(debugger, target)
    if len(dpus_booting) == 0:
        print("Could not find any dpu booting")
        sys.exit(1)

    # If a dpu is specified in the command, wait for this specific dpu to boot
    if command != "":
        dpu_to_attach = get_dpu_from_command(command, debugger, target)
        dpus_booting = filter(
            lambda dpu: dpu.GetValue() == dpu_to_attach.GetValue(),
            dpus_booting)
        while len(dpus_booting) == 0:
            dpus_booting, host_frame =\
                break_to_next_boot_and_get_dpus(debugger, target)
            if len(dpus_booting) == 0:
                print("Could not find the dpu booting")
                sys.exit(1)
            dpus_booting = filter(
                lambda dpu: dpu.GetValue() == dpu_to_attach.GetValue(),
                dpus_booting)

    dpu_addr = dpus_booting[0].GetValue()
    print("Setting up dpu '" + str(dpu_addr) + "' for attach on boot...")
    target_dpu = dpu_attach(debugger, dpu_addr, None, None)

    error = lldb.SBError()
    process_dpu = target_dpu.GetProcess()
    dpu_first_instruction_addr = 0x80000000
    dpu_first_instruction = process_dpu.ReadMemory(dpu_first_instruction_addr,
                                                   8, error)
    process_dpu.WriteMemory(dpu_first_instruction_addr,
                            bytearray([0x00, 0x00, 0x00, 0x20,
                                       0x63, 0x7e, 0x00, 0x00]), error)
    process_dpu.Detach()
    debugger.DeleteTarget(target_dpu)
    debugger.SetSelectedTarget(target)

    target.GetProcess().GetSelectedThread().StepOutOfFrame(host_frame, error)
    print("dpu '" + str(dpu_addr) + "' has booted")

    target_dpu = dpu_attach(debugger, dpu_addr, None, None)
    target_dpu.GetProcess().WriteMemory(dpu_first_instruction_addr,
                                        dpu_first_instruction, error)

    return target_dpu


def dpu_attach(debugger, command, result, internal_dict):
    '''
    usage: dpu_attach <struct dpu_t *>
    '''
    target = debugger.GetSelectedTarget()
    dpu = get_dpu_from_command(command, debugger, target)
    if not(dpu.IsValid()):
        print("Could not find dpu")
        sys.exit(1)
    print("Attaching to dpu '" + dpu.GetValue() + "'")

    program_path = get_dpu_program_path(dpu)

    rank = dpu.GetChildMemberWithName("rank")
    if not(rank.IsValid()):
        print("Could not find dpu rank")
        sys.exit(1)

    region_id, rank_id = get_region_id_and_rank_id(rank, target)
    if region_id == -1 or rank_id == -1:
        print("Could not attach to simulator (hardware only)")
        sys.exit(1)

    slice_id = dpu.GetChildMemberWithName("slice_id").GetValueAsUnsigned()
    dpu_id = dpu.GetChildMemberWithName("dpu_id").GetValueAsUnsigned()

    slice_info = rank.GetChildMemberWithName("runtime") \
        .GetChildMemberWithName("control_interface") \
        .GetChildMemberWithName("slice_info").GetChildAtIndex(slice_id)
    if not(slice_info.IsValid()):
        print("Could not find dpu slice_info")
        sys.exit(1)
    slice_target = slice_info.GetChildMemberWithName("slice_target")
    if not(slice_target.IsValid()):
        print("Could not find dpu slice_target")
        sys.exit(1)

    structure_value = slice_info.GetChildMemberWithName("structure_value") \
        .GetValueAsUnsigned()
    slice_target_type = slice_target.GetChildMemberWithName("type") \
        .GetValueAsUnsigned()
    slice_target_dpu_group_id = slice_target.GetChildMemberWithName("dpu_id") \
        .GetValueAsUnsigned()
    slice_target = (slice_target_type << 32) + slice_target_dpu_group_id

    pid = compute_dpu_pid(region_id, rank_id, slice_id, dpu_id)

    if not(set_debug_mode(debugger, rank, 1)):
        print("Could not set dpu in debug mode")
        sys.exit(1)

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
    if not(target_dpu.IsValid()):
        print("Could not create dpu target")
        sys.exit(1)

    listener = debugger.GetListener()
    error = lldb.SBError()
    process_dpu = target_dpu.ConnectRemote(listener,
                                           "connect://localhost:2066",
                                           "gdb-remote",
                                           error)
    if not(process_dpu.IsValid()):
        print("Could not connect to dpu")
        sys.exit(1)

    debugger.SetSelectedTarget(target)
    if not(set_debug_mode(debugger, rank, 0)):
        print("Could not unset dpu from debug mode")
        sys.exit(1)

    debugger.SetSelectedTarget(target_dpu)
    return target_dpu


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
    result.PutCString("ADDR \t\t\tID \t\tSTATUS \t\tPROGRAM")
    for (dpu_addr, region_id, rank_id, slice_id, dpu_id, status, program) \
        in list:
        result.PutCString(
            "'" + str(dpu_addr) + "' \t"
            + str(region_id) + ":" + str(rank_id) + ":"
            + str(slice_id) + ":" + str(dpu_id)
            + " \t" + status + " \t'" + program + "'")


def dpu_list(debugger, command, result, internal_dict):
    '''
    usage: dpu_list
    '''
    success, nb_allocated_rank = \
        get_value_from_command(
            debugger, "dpu_rank_handler_dpu_rank_list_size", 10)
    if not(success):
        print("dpu_list: internal error 1")
        sys.exit(1)

    target = debugger.GetSelectedTarget()
    result_list = []

    for each_rank in range(0, nb_allocated_rank):
        rank = get_rank_from_command(
            "dpu_rank_handler_dpu_rank_list[" + str(each_rank) + "]",
            debugger,
            target)
        if not(rank.IsValid()):
            print("dpu_list: internal error 2")
            sys.exit(1)
        rank_addr = rank.GetValue()
        if int(rank_addr, 16) == 0:
            continue

        region_id, rank_id = get_region_id_and_rank_id(rank, target)

        slice_id = 0
        dpu_id = 0
        dpu = dpu_get(rank_addr, slice_id, dpu_id, debugger, target)
        if not(dpu.IsValid):
            print("dpu_list: internal error 3")
            sys.exit(1)
        while int(dpu.GetValue(), 16) != 0:
            while int(dpu.GetValue(), 16) != 0:
                program_path = get_dpu_program_path(dpu)
                if program_path is None:
                    program_path = ""

                dpu_status = get_dpu_status(rank, slice_id, dpu_id)

                result_list.append((dpu.GetValue(),
                                    region_id, rank_id, slice_id, dpu_id,
                                    dpu_status, program_path))

                dpu_id = dpu_id + 1
                dpu = dpu_get(rank_addr, slice_id, dpu_id, debugger, target)

            dpu_id = 0
            slice_id = slice_id + 1
            dpu = dpu_get(rank_addr, slice_id, dpu_id, debugger, target)

        print_list(result_list, result)
        return result_list


def dpu_detach(debugger, command, result, internal_dict):
    '''
    usage: dpu_detach
    '''
    target = debugger.GetSelectedTarget()
    if target.GetTriple() != "dpu-upmem-dpurte":
        print("Current target is not a DPU target")
        sys.exit(1)
    target.GetProcess().Detach()
    debugger.DeleteTarget(target)
