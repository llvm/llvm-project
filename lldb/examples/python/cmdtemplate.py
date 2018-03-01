#!/usr/bin/python

#----------------------------------------------------------------------
# Be sure to add the python path that points to the LLDB shared library.
#
# # To use this in the embedded python interpreter using "lldb" just
# import it with the full path using the "command script import"
# command
#   (lldb) command script import /path/to/cmdtemplate.py
#----------------------------------------------------------------------

import lldb
import commands
import optparse
import shlex

class FrameStatCommand:
    def create_options(self):

        usage = "usage: %prog [options]"
        description = '''This command is meant to be an example of how to make an LLDB command that
does something useful, follows best practices, and exploits the SB API.
Specifically, this command computes the aggregate and average size of the variables in the current frame
and allows you to tweak exactly which variables are to be accounted in the computation.
'''

        # Pass add_help_option = False, since this keeps the command in line with lldb commands, 
        # and we wire up "help command" to work by providing the long & short help methods below.
        self.parser = optparse.OptionParser(
            description = description,
            prog = 'framestats',
            usage = usage,
            add_help_option = False)

        self.parser.add_option(
            '-i',
            '--in-scope',
            action = 'store_true',
            dest = 'inscope',
            help = 'in_scope_only = True',
            default = True)

        self.parser.add_option(
            '-a',
            '--arguments',
            action = 'store_true',
            dest = 'arguments',
            help = 'arguments = True',
            default = True)

        self.parser.add_option(
            '-l',
            '--locals',
            action = 'store_true',
            dest = 'locals',
            help = 'locals = True',
            default = True)

        self.parser.add_option(
            '-s',
            '--statics',
            action = 'store_true',
            dest = 'statics',
            help = 'statics = True',
            default = True)
 
    def get_short_help(self):
        return "Example command for use in debugging"

    def get_long_help(self):
        return self.help_string

    def __init__(self, debugger, unused):
        self.create_options()
        self.help_string = self.parser.format_help()

    def __call__(self, debugger, command, exe_ctx, result):
        # Use the Shell Lexer to properly parse up command options just like a
        # shell would
        command_args = shlex.split(command)
        
        try:
            (options, args) = self.parser.parse_args(command_args)
        except:
            # if you don't handle exceptions, passing an incorrect argument to the OptionParser will cause LLDB to exit
            # (courtesy of OptParse dealing with argument errors by throwing SystemExit)
            result.SetError("option parsing failed")
            return

        # Always get program state from the SBExecutionContext passed in as exe_ctx
        frame = exe_ctx.GetFrame()
        if not frame.IsValid():
            result.SetError("invalid frame")
            return

        variables_list = frame.GetVariables(
            options.arguments,
            options.locals,
            options.statics,
            options.inscope)
        variables_count = variables_list.GetSize()
        if variables_count == 0:
            print >> result, "no variables here"
            return
        total_size = 0
        for i in range(0, variables_count):
            variable = variables_list.GetValueAtIndex(i)
            variable_type = variable.GetType()
            total_size = total_size + variable_type.GetByteSize()
            average_size = float(total_size) / variables_count
            print >>result, "Your frame has %d variables. Their total size is %d bytes. The average size is %f bytes" % (
                variables_count, total_size, average_size)
         # not returning anything is akin to returning success


def __lldb_init_module(debugger, dict):
    # This initializer is being run from LLDB in the embedded command interpreter

    # Add any commands contained in this module to LLDB
    debugger.HandleCommand(
        'command script add -c cmdtemplate.FrameStatCommand framestats')
    print 'The "framestats" command has been installed, type "help framestats" for detailed help.'
