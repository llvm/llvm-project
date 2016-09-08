#!/usr/bin/python

import lldb
import shlex


def dump_module_sources(module, result):
    if module:
        print >> result, "Module: %s" % (module.file)
        for compile_unit in module.compile_units:
            if compile_unit.file:
                print >> result, "  %s" % (compile_unit.file)


def info_sources(debugger, command, result, dict):
    description = '''This command will dump all compile units in any modules that are listed as arguments, or for all modules if no arguments are supplied.'''
    module_names = shlex.split(command)
    target = debugger.GetSelectedTarget()
    if module_names:
        for module_name in module_names:
            dump_module_sources(target.module[module_name], result)
    else:
        for module in target.modules:
            dump_module_sources(module, result)


def __lldb_init_module(debugger, dict):
    # Add any commands contained in this module to LLDB
    debugger.HandleCommand(
        'command script add -f sources.info_sources info_sources')
    print 'The "info_sources" command has been installed, type "help info_sources" or "info_sources --help" for detailed help.'
