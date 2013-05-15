# This implements the "diagnose-unwind" command, usually installed in the debug session like
#   script import lldb.macosx
# it is used when lldb's backtrace fails -- it collects and prints information about the stack frames,
# and tries an alternate unwind algorithm, that will help to understand why lldb's unwind algorithm did
# not succeed.

import optparse
import lldb
import re
import shlex

# Print the frame number, pc, frame pointer, module UUID and function name
def backtrace_print_frame (target, frame_num, addr, fp):
  process = target.GetProcess()
  addr_for_printing = addr
  addr_width = process.GetAddressByteSize() * 2
  if frame_num > 0:
    addr = addr - 1

  sbaddr = lldb.SBAddress()
  try:
    sbaddr.SetLoadAddress(addr, target)
    module_description = ""
    if sbaddr.GetModule():
      module_filename = ""
      module_uuid_str = sbaddr.GetModule().GetUUIDString()
      if module_uuid_str == None:
        module_uuid_str = ""
      if sbaddr.GetModule().GetFileSpec():
        module_filename = sbaddr.GetModule().GetFileSpec().GetFilename()
        if module_filename == None:
          module_filename = ""
      if module_uuid_str != "" or module_filename != "":
        module_description = '%s %s' % (module_filename, module_uuid_str)
  except Exception:
    print '%2d: pc==0x%-*x fp==0x%-*x' % (frame_num, addr_width, addr_for_printing, addr_width, fp)
    return

  sym_ctx = target.ResolveSymbolContextForAddress(sbaddr, lldb.eSymbolContextEverything)
  if sym_ctx.IsValid() and sym_ctx.GetSymbol().IsValid():
    function_start = sym_ctx.GetSymbol().GetStartAddress().GetLoadAddress(target)
    offset = addr - function_start
    print '%2d: pc==0x%-*x fp==0x%-*x %s %s + %d' % (frame_num, addr_width, addr_for_printing, addr_width, fp, module_description, sym_ctx.GetSymbol().GetName(), offset)
  else:
    print '%2d: pc==0x%-*x fp==0x%-*x %s' % (frame_num, addr_width, addr_for_printing, addr_width, fp, module_description)

# A simple stack walk algorithm that follows the frame chain after the first two frames.
def simple_backtrace(debugger):
  target = debugger.GetSelectedTarget()
  process = target.GetProcess()
  cur_thread = process.GetSelectedThread()

  backtrace_print_frame (target, 0, cur_thread.GetFrameAtIndex(0).GetPC(), cur_thread.GetFrameAtIndex(0).GetFP())
  if cur_thread.GetNumFrames() < 2:
    return

  cur_fp = cur_thread.GetFrameAtIndex(1).GetFP()
  cur_pc = cur_thread.GetFrameAtIndex(1).GetPC()

  # If the pseudoreg "fp" isn't recognized, on arm hardcode to r7 which is correct for Darwin programs.
  if cur_fp == lldb.LLDB_INVALID_ADDRESS and target.triple[0:3] == "arm":
    for reggroup in cur_thread.GetFrameAtIndex(1).registers:
      if reggroup.GetName() == "General Purpose Registers":
        for reg in reggroup:
          if reg.GetName() == "r7":
            cur_fp = int (reg.GetValue(), 16)

  frame_num = 1

  while cur_pc != 0 and cur_fp != 0 and cur_pc != lldb.LLDB_INVALID_ADDRESS and cur_fp != lldb.LLDB_INVALID_ADDRESS:
    backtrace_print_frame (target, frame_num, cur_pc, cur_fp)
    frame_num = frame_num + 1
    next_pc = 0
    next_fp = 0
    if target.triple[0:6] == "x86_64" or target.triple[0:4] == "i386" or target.triple[0:3] == "arm":
      error = lldb.SBError()
      next_pc = process.ReadPointerFromMemory(cur_fp + process.GetAddressByteSize(), error)
      if not error.Success():
        next_pc = 0
      next_fp = process.ReadPointerFromMemory(cur_fp, error)
      if not error.Success():
        next_fp = 0
    # Clear the 0th bit for arm frames - this indicates it is a thumb frame
    if target.triple[0:3] == "arm" and (next_pc & 1) == 1:
      next_pc = next_pc & ~1
    cur_pc = next_pc
    cur_fp = next_fp
  backtrace_print_frame (target, frame_num, cur_pc, cur_fp)

def diagnose_unwind(debugger, command, result, dict):
  # Use the Shell Lexer to properly parse up command options just like a
  # shell would
  command_args = shlex.split(command)
  parser = create_diagnose_unwind_options()
  try:
    (options, args) = parser.parse_args(command_args)
  except:
   return
  target = debugger.GetSelectedTarget()
  if target:
    process = target.GetProcess()
    if process:
      thread = process.GetSelectedThread()
      if thread:
        lldb_versions_match = re.search(r'[lL][lL][dD][bB]-(\d+)([.](\d+))?([.](\d+))?', debugger.GetVersionString())
        lldb_version = 0
        lldb_minor = 0
        if len(lldb_versions_match.groups()) >= 1 and lldb_versions_match.groups()[0]:
          lldb_major = int(lldb_versions_match.groups()[0])
        if len(lldb_versions_match.groups()) >= 5 and lldb_versions_match.groups()[4]:
          lldb_minor = int(lldb_versions_match.groups()[4])

        print 'Unwind diagnostics for thread %d' % thread.GetIndexID()
        print ""
        print "lldb's unwind algorithm:"
        print ""
        frame_num = 0
        for frame in thread.frames:
          if not frame.IsInlined():
            backtrace_print_frame (target, frame_num, frame.GetPC(), frame.GetFP())
            frame_num = frame_num + 1
        print ""
        print "============================================================================================="
        print ""
        print "Simple stack walk algorithm:"
        print ""
        simple_backtrace(debugger)
        print ""
        print "============================================================================================="
        print ""
        for frame in thread.frames:
          if not frame.IsInlined():
            print "--------------------------------------------------------------------------------------"
            print ""
            print "Disassembly of %s, frame %d" % (frame.GetFunctionName(), frame.GetFrameID())
            print ""
            if lldb_major > 300 or (lldb_major == 300 and lldb_minor >= 18):
                if target.triple[0:6] == "x86_64" or target.triple[0:4] == "i386":
                  debugger.HandleCommand('disassemble -F att -a 0x%x' % frame.GetPC())
                else:
                  debugger.HandleCommand('disassemble -a 0x%x' % frame.GetPC())
            else:
              debugger.HandleCommand('disassemble -n "%s"' % frame.GetFunctionName())
        print ""
        print "============================================================================================="
        print ""
        for frame in thread.frames:
          if not frame.IsInlined():
            print "--------------------------------------------------------------------------------------"
            print ""
            print "Unwind instructions for %s, frame %d" % (frame.GetFunctionName(), frame.GetFrameID())
            print ""
            if lldb_major > 300 or (lldb_major == 300 and lldb_minor >= 20):
              debugger.HandleCommand('image show-unwind -a "0x%x"' % frame.GetPC())
            else:
              debugger.HandleCommand('image show-unwind -n "%s"' % frame.GetFunctionName())

def create_diagnose_unwind_options():
  usage = "usage: %prog"
  description='''Print diagnostic information about a thread backtrace which will help to debug unwind problems'''
  parser = optparse.OptionParser(description=description, prog='diagnose_unwind',usage=usage)
  return parser

lldb.debugger.HandleCommand('command script add -f %s.diagnose_unwind diagnose-unwind' % __name__)
print 'The "diagnose-unwind" command has been installed, type "help diagnose-unwind" for detailed help.'
