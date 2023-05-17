#!/usr/bin/env python

# ---------------------------------------------------------------------
# Be sure to add the python path that points to the LLDB shared library.
#
# # To use this in the embedded python interpreter using "lldb" just
# import it with the full path using the "command script import"
# command
#   (lldb) command script import /path/to/cmdtemplate.py
# ---------------------------------------------------------------------

import inspect
import lldb
import optparse
import shlex
import sys


class FrameStatCommand:
    program = 'framestats'

    @classmethod
    def register_lldb_command(cls, debugger, module_name):
        parser = cls.create_options()
        cls.__doc__ = parser.format_help()
        # Add any commands contained in this module to LLDB
        command = 'command script add -o -c %s.%s %s' % (module_name,
                                                      cls.__name__,
                                                      cls.program)
        debugger.HandleCommand(command)
        print('The "{0}" command has been installed, type "help {0}" or "{0} '
              '--help" for detailed help.'.format(cls.program))

    @classmethod
    def create_options(cls):

        usage = "usage: %prog [options]"
        description = ('This command is meant to be an example of how to make '
                       'an LLDB command that does something useful, follows '
                       'best practices, and exploits the SB API. '
                       'Specifically, this command computes the aggregate '
                       'and average size of the variables in the current '
                       'frame and allows you to tweak exactly which variables '
                       'are to be accounted in the computation.')

        # Pass add_help_option = False, since this keeps the command in line
        #  with lldb commands, and we wire up "help command" to work by
        # providing the long & short help methods below.
        parser = optparse.OptionParser(
            description=description,
            prog=cls.program,
            usage=usage,
            add_help_option=False)

        parser.add_option(
            '-i',
            '--in-scope',
            action='store_true',
            dest='inscope',
            help='in_scope_only = True',
            default=True)

        parser.add_option(
            '-a',
            '--arguments',
            action='store_true',
            dest='arguments',
            help='arguments = True',
            default=True)

        parser.add_option(
            '-l',
            '--locals',
            action='store_true',
            dest='locals',
            help='locals = True',
            default=True)

        parser.add_option(
            '-s',
            '--statics',
            action='store_true',
            dest='statics',
            help='statics = True',
            default=True)

        return parser

    def get_short_help(self):
        return "Example command for use in debugging"

    def get_long_help(self):
        return self.help_string

    def __init__(self, debugger, unused):
        self.parser = self.create_options()
        self.help_string = self.parser.format_help()

    def __call__(self, debugger, command, exe_ctx, result):
        # Use the Shell Lexer to properly parse up command options just like a
        # shell would
        command_args = shlex.split(command)

        try:
            (options, args) = self.parser.parse_args(command_args)
        except:
            # if you don't handle exceptions, passing an incorrect argument to
            # the OptionParser will cause LLDB to exit (courtesy of OptParse
            # dealing with argument errors by throwing SystemExit)
            result.SetError("option parsing failed")
            return

        # Always get program state from the lldb.SBExecutionContext passed
        # in as exe_ctx
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
            print("no variables here", file=result)
            return
        total_size = 0
        for i in range(0, variables_count):
            variable = variables_list.GetValueAtIndex(i)
            variable_type = variable.GetType()
            total_size = total_size + variable_type.GetByteSize()
            average_size = float(total_size) / variables_count
            print("Your frame has %d variables. Their total size "
                             "is %d bytes. The average size is %f bytes" % (
                                    variables_count, total_size, average_size), file=result)
        # not returning anything is akin to returning success


def __lldb_init_module(debugger, dict):
    # Register all classes that have a register_lldb_command method
    for _name, cls in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(cls) and callable(getattr(cls,
                                                     "register_lldb_command",
                                                     None)):
            cls.register_lldb_command(debugger, __name__)
