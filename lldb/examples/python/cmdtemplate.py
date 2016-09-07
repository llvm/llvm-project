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


def create_framestats_options():
    usage = "usage: %prog [options]"
    description = '''This command is meant to be an example of how to make an LLDB command that
does something useful, follows best practices, and exploits the SB API.
Specifically, this command computes the aggregate and average size of the variables in the current frame
and allows you to tweak exactly which variables are to be accounted in the computation.
'''
    parser = optparse.OptionParser(
        description=description,
        prog='framestats',
        usage=usage)
    parser.add_option(
        '-i',
        '--in-scope',
        action='store_true',
        dest='inscope',
        help='in_scope_only = True',
        default=False)
    parser.add_option(
        '-a',
        '--arguments',
        action='store_true',
        dest='arguments',
        help='arguments = True',
        default=False)
    parser.add_option(
        '-l',
        '--locals',
        action='store_true',
        dest='locals',
        help='locals = True',
        default=False)
    parser.add_option(
        '-s',
        '--statics',
        action='store_true',
        dest='statics',
        help='statics = True',
        default=False)
    return parser


def the_framestats_command(debugger, command, result, dict):
    # Use the Shell Lexer to properly parse up command options just like a
    # shell would
    command_args = shlex.split(command)
    parser = create_framestats_options()
    try:
        (options, args) = parser.parse_args(command_args)
    except:
        # if you don't handle exceptions, passing an incorrect argument to the OptionParser will cause LLDB to exit
        # (courtesy of OptParse dealing with argument errors by throwing SystemExit)
        result.SetError("option parsing failed")
        return

    # in a command - the lldb.* convenience variables are not to be used
    # and their values (if any) are undefined
    # this is the best practice to access those objects from within a command
    target = debugger.GetSelectedTarget()
    process = target.GetProcess()
    thread = process.GetSelectedThread()
    frame = thread.GetSelectedFrame()
    if not frame.IsValid():
        return "no frame here"
    # from now on, replace lldb.<thing>.whatever with <thing>.whatever
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
    # Make the options so we can generate the help text for the new LLDB
    # command line command prior to registering it with LLDB below
    parser = create_framestats_options()
    the_framestats_command.__doc__ = parser.format_help()
    # Add any commands contained in this module to LLDB
    debugger.HandleCommand(
        'command script add -f cmdtemplate.the_framestats_command framestats')
    print 'The "framestats" command has been installed, type "help framestats" or "framestats --help" for detailed help.'
