#!/usr/bin/env python3

import lldb
import shlex


def dump_module_sources(module, result):
    if module:
        print("Module: %s" % (module.file), file=result)
        for compile_unit in module.compile_units:
            if compile_unit.file:
                print("  %s" % (compile_unit.file), file=result)


def info_sources(debugger, command, exe_ctx, result, internal_dict):
    """This command will dump all compile units in any modules that are listed as arguments, or for all modules if no arguments are supplied."""
    module_names = shlex.split(command)
    target = exe_ctx.target
    if module_names:
        for module_name in module_names:
            dump_module_sources(target.module[module_name], result)
    else:
        for module in target.modules:
            dump_module_sources(module, result)


def __lldb_init_module(debugger, internal_dict):
    # Add any commands contained in this module to LLDB
    debugger.HandleCommand("command script add -o -f sources.info_sources info_sources")
    print(
        'The "info_sources" command has been installed, type "help info_sources" or "info_sources --help" for detailed help.'
    )
