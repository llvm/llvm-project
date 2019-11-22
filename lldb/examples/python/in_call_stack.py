#!/usr/bin/env python


def __lldb_init_module(debugger, internal_dict):
  debugger.HandleCommand(
      'command alias in_call_stack breakpoint command add --python-function in_call_stack.in_call_stack -k name -v %1'
  )


def in_call_stack(frame, bp_loc, arg_dict, _):
  """Only break if the given name is in the current call stack."""
  thread = frame.GetThread()
  found = False
  for frame in thread.frames:
    name = arg_dict.GetValueForKey('name').GetStringValue(1000)
    # Check the symbol.
    symbol = frame.GetSymbol()
    if symbol and name in frame.GetSymbol().GetName():
      return True
    # Check the function.
    function = frame.GetFunction()
    if function and name in function.GetName():
      return True
  return False
