#!/usr/bin/python

import lldb
import shlex

@lldb.command("shadow")
def check_shadow_command(debugger, command, result, dict):
    target = debugger.GetSelectedTarget()
    if not target:
        print >>result, "invalid target"
        return
    process = target.GetProcess()
    if not process:
        print >>result, "invalid process"
        return
    thread = process.GetSelectedThread()
    if not thread:
        print >>result, "invalid thread"
        return
    frame = thread.GetSelectedFrame()
    if not frame:
        print >>result, "invalid frame"
        return
    # Parse command line args
    command_args = shlex.split(command)
    # TODO: add support for using arguments that are passed to this command...
    
    # Make a dictionary of variable name to "SBBlock and SBValue"
    var_dict = {}
    
    # Get the deepest most block from the current frame
    block = frame.GetBlock()
    # Iterate through the block and all of its parents
    while block.IsValid():
        # Get block variables from the current block only
        block_vars = block.GetVariables(frame, True, True, True, 0)
        # Iterate through all variables in the current block
        for block_var in block_vars:
            # Get the variable name and see if we already have a variable by this name?
            block_var_name = block_var.GetName()
            if block_var_name in var_dict:
                # We already have seen a variable with this name, so it is shadowed
                shadow_block_and_vars = var_dict[block_var_name]
                for shadow_block_and_var in shadow_block_and_vars:
                    print shadow_block_and_var[0], shadow_block_and_var[1]
                print 'is shadowed by:'
                print block, block_var
            # Since we can have multiple shadowed variables, we our variable
            # name dictionary to have an array or "block + variable" pairs so
            # We can correctly print out all shadowed variables and whow which
            # blocks they come from
            if block_var_name in var_dict:
                var_dict[block_var_name].append([block, block_var])
            else:
                var_dict[block_var_name] = [[block, block_var]]
        # Get the parent block and continue 
        block = block.GetParent()
    

