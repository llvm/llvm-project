import re
import subprocess, os
import lldb

def dpu_attach(debugger, command, result, internal_dict):
  '''
  usage: dpu_attach <struct dpu_t *var>
  '''
  target = debugger.GetSelectedTarget()
  dpu = target.GetProcess().GetSelectedThread().GetSelectedFrame().FindVariable(command)
  assert dpu.IsValid()

  slice_id = dpu.GetChildMemberWithName("slice_id").GetValueAsUnsigned()
  dpu_id = dpu.GetChildMemberWithName("dpu_id").GetValueAsUnsigned()

  rank = dpu.GetChildMemberWithName("rank")
  slice_info = rank.GetChildMemberWithName("runtime").GetChildMemberWithName("control_interface").GetChildMemberWithName("slice_info").GetChildAtIndex(slice_id)
  slice_target = slice_info.GetChildMemberWithName("slice_target")

  structure_value = slice_info.GetChildMemberWithName("structure_value").GetValueAsUnsigned()
  slice_target_type = slice_target.GetChildMemberWithName("type").GetValueAsUnsigned()
  slice_target_dpu_group_id = slice_target.GetChildMemberWithName("dpu_id").GetValueAsUnsigned()
  slice_target = (slice_target_type << 32) + slice_target_dpu_group_id

  rank_path = str(rank.GetChildMemberWithName("description").GetChildMemberWithName("_internals").GetChildMemberWithName("data").Cast(target.FindFirstType("hw_dpu_rank_allocation_parameters_t")).GetChildMemberWithName("rank_fs").GetChildMemberWithName("rank_path"))
  region_id = int(re.search('dpu_region(.+)/', rank_path).group(1))
  rank_id = int(re.search('dpu_rank(.+)"', rank_path).group(1))

  pid = dpu_id + (slice_id << 16) + (rank_id << 32) + (region_id << 48)

  program_path = re.search('"(.+)"', str(dpu.GetChildMemberWithName("runtime_context").GetChildMemberWithName("program_path"))).group(1)

  lldb_server_dpu_env = os.environ.copy()
  lldb_server_dpu_env["UPMEM_LLDB_STRUCTURE_VALUE"] = str(structure_value)
  lldb_server_dpu_env["UPMEM_LLDB_SLICE_TARGET"] = str(slice_target)
  subprocess.Popen(['lldb-server-dpu', 'gdbserver', '--attach', str(pid), ':2066'], env=lldb_server_dpu_env)

  target_dpu = debugger.CreateTargetWithFileAndTargetTriple(program_path, "dpu-upmem-dpurte")
  assert dpu.IsValid()

  listener = debugger.GetListener()
  error = lldb.SBError()
  process_dpu = target_dpu.ConnectRemote(listener, "connect://localhost:2066", "gdb-remote", error)
  assert process_dpu.IsValid()

  debugger.SetSelectedTarget(target_dpu)
