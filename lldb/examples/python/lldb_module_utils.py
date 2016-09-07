#!/usr/bin/python

import lldb
import optparse
import shlex
import string
import sys


def create_dump_module_line_tables_options():
    usage = "usage: dump_module_line_tables [options] MODULE1 [MODULE2 ...]"
    description = '''Dumps all line tables from all compile units for any modules specified as arguments. Specifying the --verbose flag will output address ranges for each line entry.'''
    parser = optparse.OptionParser(
        description=description,
        prog='start_gdb_log',
        usage=usage)
    parser.add_option(
        '-v',
        '--verbose',
        action='store_true',
        dest='verbose',
        help='Display verbose output.',
        default=False)
    return parser


def dump_module_line_tables(debugger, command, result, dict):
    '''Dumps all line tables from all compile units for any modules specified as arguments.'''
    command_args = shlex.split(command)

    parser = create_dump_module_line_tables_options()
    try:
        (options, args) = parser.parse_args(command_args)
    except:
        return
    if command_args:
        target = debugger.GetSelectedTarget()
        lldb.target = target
        for module_name in command_args:
            result.PutCString('Searching for module "%s"' % (module_name,))
            module_fspec = lldb.SBFileSpec(module_name, False)
            module = target.FindModule(module_fspec)
            if module:
                for cu_idx in range(module.GetNumCompileUnits()):
                    cu = module.GetCompileUnitAtIndex(cu_idx)
                    result.PutCString("\n%s:" % (cu.file))
                    for line_idx in range(cu.GetNumLineEntries()):
                        line_entry = cu.GetLineEntryAtIndex(line_idx)
                        start_file_addr = line_entry.addr.file_addr
                        end_file_addr = line_entry.end_addr.file_addr
                        # If the two addresses are equal, this line table entry
                        # is a termination entry
                        if options.verbose:
                            if start_file_addr != end_file_addr:
                                result.PutCString(
                                    '[%#x - %#x): %s' %
                                    (start_file_addr, end_file_addr, line_entry))
                        else:
                            if start_file_addr == end_file_addr:
                                result.PutCString('%#x: END' %
                                                  (start_file_addr))
                            else:
                                result.PutCString(
                                    '%#x: %s' %
                                    (start_file_addr, line_entry))
                        if start_file_addr == end_file_addr:
                            result.Printf("\n")
            else:
                result.PutCString("no module for '%s'" % module)
    else:
        result.PutCString("error: invalid target")

parser = create_dump_module_line_tables_options()
dump_module_line_tables.__doc__ = parser.format_help()
lldb.debugger.HandleCommand(
    'command script add -f %s.dump_module_line_tables dump_module_line_tables' %
    __name__)
print 'Installed "dump_module_line_tables" command'
