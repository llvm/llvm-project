from __future__ import print_function


def bar_function(debugger, args, result, dict):
    global UtilityModule
    print(UtilityModule.barutil_function("bar told me " + args), file=result)
    return None


def __lldb_init_module(debugger, session_dict):
    global UtilityModule
    UtilityModule = __import__("barutil")
    debugger.HandleCommand(
        "command script add -f bar.bar_function barothercmd")
    return None
