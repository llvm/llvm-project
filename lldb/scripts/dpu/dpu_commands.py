import re
import sys
import subprocess
import os
import lldb


def check_target(target):
    if target.GetTriple() == "dpu-upmem-dpurte":
        print("Command not allowed on dpu target")
        return False
    return True


def decompute_dpu_pid(pid):
    rank_id = (pid / (100*100)) % 100
    slice_id = (pid / 100) % 100
    dpu_id = pid % 100
    return rank_id, slice_id, dpu_id


def compute_dpu_pid(rank_id, slice_id, dpu_id):
    return dpu_id + 100 * (slice_id + 100 * (rank_id + 100))


def get_value_from_command(debugger, command, base):
    return_obj = lldb.SBCommandReturnObject()
    debugger.GetCommandInterpreter().HandleCommand("p/x " + command, return_obj)
    if return_obj.GetStatus() != lldb.eReturnStatusSuccessFinishResult:
        return False, 0
    output = return_obj.GetOutput()
    if output is None or len(output) == 0:
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


def get_dpu_from_command(command, debugger, target):
    addr = 0
    try:
        addr = int(command, 16)
    except ValueError:
        success, addr = get_value_from_command(debugger, command, 16)
        if not success:
            command_values = command.split('.')
            if len(command_values) == 3:
                rank_id = command_values[0]
                dpus = dpu_list(debugger, None, None, None)
                if rank_id in dpus:
                    addr = next((dpu[0] for dpu in dpus[rank_id]
                                 if (command ==
                                     str(rank_id) + "." + str(dpu[1]) + "." +
                                     str(dpu[2]))),
                                0)
    if addr == 0:
        print("Could not interpret command '" + command + "'")
        return None
    return target.CreateValueFromExpression(
        "dpu", "(struct dpu_t *)" + str(addr))


def get_rank_id(rank, target):
    full_rank_id = rank.GetChildMemberWithName("rank_id").GetValueAsUnsigned()
    # See <backends>/api/include/lowlevel/dpu_target_macros.h:DPU_TARGET_SHIFT
    dpu_target_shift = 12
    dpu_target_mask = ((1 << dpu_target_shift) - 1)
    rank_id = full_rank_id & dpu_target_mask
    target_id = full_rank_id >> dpu_target_shift
    return rank_id, target_id


def get_nb_slices_and_nb_dpus_per_slice(rank, target):
    uint32_type = target.FindFirstType("uint32_t")
    topology = rank.GetChildMemberWithName("description") \
                   .GetChildMemberWithName("hw")		  \
                   .GetChildMemberWithName("topology")
    nb_dpus_per_slice = topology \
        .GetChildMemberWithName("nr_of_dpus_per_control_interface") \
        .Cast(uint32_type).GetValueAsUnsigned() & 0xff
    nb_slices = topology \
        .GetChildMemberWithName("nr_of_control_interfaces") \
        .Cast(uint32_type).GetValueAsUnsigned() & 0xff
    return nb_slices, nb_dpus_per_slice


def get_dpu_program_path(dpu):
    program_path = dpu.GetChildMemberWithName("program") \
                      .GetChildMemberWithName("program_path")
    if program_path.GetChildAtIndex(0).GetValue() is None:
        return None
    return re.search('"(.+)"', str(program_path)).group(1)


def get_dpu_status(dpus_running, dpus_in_fault, slice_id, dpu_id):
    dpu_mask = 1 << dpu_id
    if (dpu_mask & dpus_running) != 0:
        return "RUNNING"
    elif (dpu_mask & dpus_in_fault) != 0:
        return "ERROR"
    else:
        return "IDLE"


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
        rank = frame.FindVariable("rank")
        nb_ci, nb_dpu_per_ci = \
            get_nb_slices_and_nb_dpus_per_slice(rank, target)
        nb_dpu = nb_ci * nb_dpu_per_ci
        for each_dpu in range(0, nb_dpu):
            dpu_list.append(int(str(rank.GetValueForExpressionPath(
                "->dpus[" + str(each_dpu) + "]").GetAddress()), 16))
    elif function_name == launch_dpu_function:
        dpu_list.append(frame.FindVariable("dpu").GetValueAsUnsigned())

    return dpu_list, frame


def dpu_attach_on_boot(debugger, command, result, internal_dict):
    '''
    usage: dpu_attach_on_boot [<struct dpu_t *>]
    '''
    target = debugger.GetSelectedTarget()
    if not(check_target(target)):
        return None

    dpus_booting, host_frame = \
        break_to_next_boot_and_get_dpus(debugger, target)
    if dpus_booting is None or len(dpus_booting) == 0:
        print("Could not find any dpu booting")
        return None

    # If a dpu is specified in the command, wait for this specific dpu to boot
    if command != "":
        dpu_to_attach = get_dpu_from_command(command, debugger, target)
        if dpu_to_attach is None:
            print("Could not find the dpu to attach to")
            return None
        dpus_booting = list(filter(
            lambda dpu: dpu == dpu_to_attach.GetValueAsUnsigned(),
            dpus_booting))
        while len(dpus_booting) == 0:
            dpus_booting, host_frame =\
                break_to_next_boot_and_get_dpus(debugger, target)
            if dpus_booting is None or len(dpus_booting) == 0:
                print("Could not find the dpu booting")
                return None
            dpus_booting = filter(
                lambda dpu: dpu == dpu_to_attach.GetValueAsUnsigned(),
                dpus_booting)

    dpu_addr = str(hex(dpus_booting[0]))
    print("Setting up dpu '" + dpu_addr + "' for attach on boot...")
    target_dpu = dpu_attach(debugger, dpu_addr, None, None)
    if target_dpu is None:
        print("Could not attach to dpu")
        return None

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
    if target_dpu is None:
        print("Could not attach to dpu")
        return None
    target_dpu.GetProcess().WriteMemory(dpu_first_instruction_addr,
                                        dpu_first_instruction, error)

    return target_dpu


def dpu_attach(debugger, command, result, internal_dict):
    '''
    usage: dpu_attach <struct dpu_t *>
    '''
    target = debugger.GetSelectedTarget()
    if not(check_target(target)):
        return None

    dpu = get_dpu_from_command(command, debugger, target)
    if dpu is None or not(dpu.IsValid):
        print("Could not find dpu")
        return None
    print("Attaching to dpu '" + dpu.GetValue() + "'")

    program_path = get_dpu_program_path(dpu)

    rank = dpu.GetChildMemberWithName("rank")
    if not(rank.IsValid()):
        print("Could not find dpu rank")
        return None

    rank_id, target_id = get_rank_id(rank, target)
    if target_id != 3:  # 3 => dpu_type_t:HW
        print("Could not attach to DPU (hardware only)")
        return None
    slice_id = dpu.GetChildMemberWithName("slice_id").GetValueAsUnsigned()
    dpu_id = dpu.GetChildMemberWithName("dpu_id").GetValueAsUnsigned()
    pid = compute_dpu_pid(rank_id, slice_id, dpu_id)

    nb_slices, nb_dpus_per_slice = get_nb_slices_and_nb_dpus_per_slice(
        rank, target)
    structures_value_env = ""
    slices_target_env = ""
    host_muxs_mram_state_env = ""
    slices_info = rank.GetChildMemberWithName("runtime") \
                      .GetChildMemberWithName("control_interface") \
                      .GetChildMemberWithName("slice_info")
    for each_slice in range(nb_slices):
        structures_value_env += str(each_slice) + ":"
        slices_target_env += str(each_slice) + ":"
        host_muxs_mram_state_env += str(each_slice) + ":"

        slice_info = slices_info.GetChildAtIndex(each_slice)
        if not(slice_info.IsValid()):
            print("Could not find dpu slice_info")
            return None
        slice_target = slice_info.GetChildMemberWithName("slice_target")
        if not(slice_target.IsValid()):
            print("Could not find dpu slice_target")
            return None

        structures_value_env += str(slice_info.GetChildMemberWithName(
            "structure_value") .GetValueAsUnsigned()) + "&"
        host_muxs_mram_state_env += \
            str(slice_info.GetChildMemberWithName("host_mux_mram_state").GetValueAsUnsigned()) + "&"
        slice_target_type = slice_target.GetChildMemberWithName("type") \
                                        .GetValueAsUnsigned()
        slice_target_dpu_group_id = slice_target.GetChildMemberWithName(
            "dpu_id") .GetValueAsUnsigned()
        slices_target_env += str((slice_target_dpu_group_id <<
                                  32) + slice_target_type) + "&"

    if not(set_debug_mode(debugger, rank, 1)):
        print("Could not set dpu in debug mode")
        return None

    lldb_server_dpu_env = os.environ.copy()
    lldb_server_dpu_env["UPMEM_LLDB_STRUCTURES_VALUE"] = structures_value_env
    lldb_server_dpu_env["UPMEM_LLDB_SLICES_TARGET"] = slices_target_env
    lldb_server_dpu_env["UPMEM_LLDB_HOST_MUXS_MRAM_STATE"] = \
        host_muxs_mram_state_env

    if not(program_path is None):
        module_spec = lldb.SBModuleSpec()
        module_spec.SetFileSpec(lldb.SBFileSpec(program_path))

        nr_tasklets = lldb.SBModule(module_spec).FindSymbol('NR_TASKLETS')
        if nr_tasklets.IsValid():
            lldb_server_dpu_env["UPMEM_LLDB_NR_TASKLETS"] = \
                str(nr_tasklets.GetIntegerValue())

    if program_path is not None and not os.path.exists(program_path):
        program_path = None
    target_dpu = \
        debugger.CreateTargetWithFileAndTargetTriple(program_path,
                                                     "dpu-upmem-dpurte")
    if not(target_dpu.IsValid()):
        print("Could not create dpu target")
        return None

    # Get the address of the error_storage variable.
    # As we need to send this address *before* the lldb server has started we cannot send it as a
    # normal lldb GDBRemote packet. Instead, we set an environment variable with the value that we
    # have looked up from the loaded bianry. If we cannot find one, then this environment variable
    # remains unset, and we cannot detach and re-attach within different processes.
    storage = target_dpu.FindFirstGlobalVariable("error_storage")
    if storage.IsValid():
        lldb_server_dpu_env["UPMEM_LLDB_ERROR_STORE_ADDR"] = str(storage.location)

    subprocess.Popen(['lldb-server-dpu',
                      'gdbserver',
                      '--attach',
                      str(pid),
                      ':2066'],
                     env=lldb_server_dpu_env)

    listener = debugger.GetListener()
    error = lldb.SBError()
    process_dpu = target_dpu.ConnectRemote(listener,
                                           "connect://localhost:2066",
                                           "gdb-remote",
                                           error)
    if not(process_dpu.IsValid()):
        print("Could not connect to dpu")
        return None

    open_print_sequence_fct_ctx = target_dpu \
        .FindFunctions("__open_print_sequence")
    close_print_sequence_fct_ctx = target_dpu \
        .FindFunctions("__close_print_sequence")
    if (open_print_sequence_fct_ctx.IsValid()
        and close_print_sequence_fct_ctx.IsValid()):
        open_print_sequence_fct = open_print_sequence_fct_ctx \
            .GetContextAtIndex(0).GetFunction()
        close_print_sequence_fct = close_print_sequence_fct_ctx \
            .GetContextAtIndex(0).GetFunction()
        if (open_print_sequence_fct.IsValid()
            and close_print_sequence_fct.IsValid()):
            print_buffer_var = target_dpu \
                .FindFirstGlobalVariable("__stdout_buffer")
            print_buffer_size_var = target_dpu \
                .FindFirstGlobalVariable("__stdout_buffer_size")
            print_buffer_var_var = target_dpu \
                .FindFirstGlobalVariable("__stdout_buffer_state")

            if (print_buffer_var.IsValid()
                and print_buffer_size_var.IsValid()
                and print_buffer_var_var.IsValid()):

                open_print_sequence_addr = open_print_sequence_fct \
                    .GetStartAddress().GetFileAddress()
                close_print_sequence_addr = close_print_sequence_fct \
                    .GetStartAddress().GetFileAddress()
                print_buffer_addr = print_buffer_var \
                    .GetAddress().GetFileAddress()
                print_buffer_size = print_buffer_size_var.GetValueAsUnsigned()
                print_buffer_var_addr = print_buffer_var_var.GetAddress() \
                                                            .GetFileAddress()

                process_dpu.SetDpuPrintInfo(open_print_sequence_addr,
                                            close_print_sequence_addr,
                                            print_buffer_addr,
                                            print_buffer_size,
                                            print_buffer_var_addr)

    debugger.SetSelectedTarget(target)
    if not(set_debug_mode(debugger, rank, 0)):
        print("Could not unset dpu from debug mode")
        return None

    debugger.SetSelectedTarget(target_dpu)
    return target_dpu


def print_list_rank(rank_id, dpus_info, verbose, status_filter, result):
    if verbose:
        for dpu_addr, slice_id, dpu_id, status, program in dpus_info:
            if status_filter is None or status_filter == status:
                result.PutCString(
                    "'%s'  %2u.%u.%u  %7s  '%s'" %
                    (
                        str(hex(dpu_addr)),
                        int(rank_id),
                        slice_id,
                        dpu_id,
                        status,
                        program
                    )
                )
    else:
        default_program = dpus_info[0][4]
        idle = 0
        running = 0
        error = 0
        for dpu_addr, slice_id, dpu_id, status, program in dpus_info:
            if status == "IDLE":
                idle += 1
            if status == "RUNNING":
                running += 1
            if status == "ERROR":
                error += 1
            if program != default_program:
                default_program = None
        result.PutCString(
            "RANK#%2u: %2u DPUs ( %2u IDLE - %2u RUNNING - %2u ERROR ) '%s'" %
            (int(rank_id),
             len(dpus_info),
                idle,
                running,
                error,
                (default_program if default_program is not None else "DPUs are not loaded with the same program")))


def print_list(rank_list, result, command):
    if result is None:
        return
    verbose = False
    rank_filter = None
    status_filter = None
    if command is not None:
        args = command.split()
        nb_args = len(args)
        for arg_id, arg in enumerate(args):
            if arg == '-v':
                verbose = True
            elif arg == '-r' and nb_args > arg_id + 1:
                rank_filter = args[arg_id + 1]
            elif arg[:2] == '-r':
                rank_filter = arg[2:]
            elif arg == '-s' and nb_args > arg_id + 1:
                status_filter = args[arg_id + 1]
            elif arg[:2] == '-s':
                status_filter = arg[2:]

    for rank_id, dpus_info in rank_list.items():
        if rank_filter is None or rank_filter == rank_id:
            print_list_rank(
                rank_id,
                dpus_info,
                verbose or rank_filter is not None or status_filter is not None,
                status_filter,
                result)


def dpu_list(debugger, command, result, internal_dict):
    '''
    usage: dpu_list [-v] [-r <rank_id>] [-s <status:IDLE|RUNNING|ERROR>]
    options:
    \t-v \tVerbose mode, print detail information for all DPUs
    \t-r \tFilter DPUs of the specified rank_id
    \t-s \tFilter DPUs of the specified status
    '''
    target = debugger.GetSelectedTarget()
    if not(check_target(target)):
        return None

    nb_allocated_rank = \
        target.FindFirstGlobalVariable("dpu_rank_handler_dpu_rank_list_size")
    if nb_allocated_rank is None:
        print("dpu_list: internal error 1 (can't get number of ranks)")
        return None

    rank_list = target.FindFirstGlobalVariable(
        "dpu_rank_handler_dpu_rank_list")
    if rank_list is None:
        print("dpu_list: internal error 2 (can't get rank list)")
        return None

    result_list = {}
    for each_rank in range(0, nb_allocated_rank.GetValueAsUnsigned()):
        rank = rank_list.GetValueForExpressionPath("[" + str(each_rank) + "]")
        if rank.GetValueAsUnsigned() == 0:
            continue

        rank_id, _ = get_rank_id(rank, target)

        nb_slices, nb_dpus_per_slice = \
            get_nb_slices_and_nb_dpus_per_slice(rank, target)

        run_context = rank.GetChildMemberWithName("runtime") \
                          .GetChildMemberWithName("run_context")
        dpus_running = run_context.GetChildMemberWithName("dpu_running")
        dpus_in_fault = run_context.GetChildMemberWithName("dpu_in_fault")
        dpus = rank.GetChildMemberWithName("dpus")

        for slice_id in range(0, nb_slices):
            dpus_running_in_slice = dpus_running.GetChildAtIndex(slice_id) \
                                                .GetValueAsUnsigned()
            dpus_in_fault_in_slice = dpus_in_fault.GetChildAtIndex(slice_id) \
                                                  .GetValueAsUnsigned()
            for dpu_id in range(0, nb_dpus_per_slice):
                dpu = rank.GetValueForExpressionPath(
                    "->dpus["
                    + str(slice_id * nb_dpus_per_slice + dpu_id)
                    + "]")

                if dpu.GetChildMemberWithName("enabled").GetValue() != 'true':
                    continue

                program_path = get_dpu_program_path(dpu)
                if program_path is None:
                    program_path = ""

                dpu_status = get_dpu_status(dpus_running_in_slice,
                                            dpus_in_fault_in_slice,
                                            slice_id,
                                            dpu_id)

                result_list.setdefault(str(rank_id), []).append((int(str(dpu.GetAddress()), 16),
                                                                 slice_id, dpu_id,
                                                                 dpu_status, program_path))

    print_list(result_list, result, command)
    return result_list


def exec_ufi_identity(debugger, rank):
    success, unused = get_value_from_command(
        debugger,
        "ufi_identity((dpu_rank_t *)"
        + rank.GetValue() + ", 0xff, lldb_dummy_results)",
        16)
    return success


def get_rank_from_pid(debugger, pid):
    target_rank_id, _, _ = decompute_dpu_pid(pid)
    target = debugger.GetSelectedTarget()
    if not(check_target(target)):
        return None

    nb_allocated_rank = \
        target.FindFirstGlobalVariable("dpu_rank_handler_dpu_rank_list_size")
    if nb_allocated_rank is None:
        print("get_rank_from_pid: internal error 1 (can't get number of ranks)")
        return None

    rank_list = target.FindFirstGlobalVariable(
        "dpu_rank_handler_dpu_rank_list")
    if rank_list is None:
        print("get_rank_from_pid: internal error 2 (can't get rank list)")
        return None

    for each_rank in range(0, nb_allocated_rank.GetValueAsUnsigned()):
        rank = rank_list.GetValueForExpressionPath("[" + str(each_rank) + "]")
        if rank.GetValueAsUnsigned() == 0:
            continue
        rank_id, _ = get_rank_id(rank, target)
        if rank_id != target_rank_id:
            continue
        return rank


def dpu_detach(debugger, command, result, internal_dict):
    '''
    usage: dpu_detach
    '''
    target = debugger.GetSelectedTarget()
    if target.GetTriple() != "dpu-upmem-dpurte":
        print("Current target is not a DPU target")
        return None
    pid = target.GetProcess().GetProcessID()
    target.GetProcess().Detach()
    debugger.DeleteTarget(target)
    rank = get_rank_from_pid(debugger, pid)
    if not exec_ufi_identity(debugger, rank):
        print("Could not execute ufi_identity during detach")
