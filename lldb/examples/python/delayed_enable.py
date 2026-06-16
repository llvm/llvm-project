import lldb
from lldb.plugins.parsed_cmd import ParsedCommand
import time


class EnableThemLater(ParsedCommand):
    program = "delayed_enable"

    def __init__(self, debugger: lldb.SBDebugger, internal_dict):
        super().__init__(debugger, internal_dict)

    def setup_command_definition(self):
        ov_parser = self.get_parser()
        ov_parser.add_argument_set(
            [
                ov_parser.make_argument_element(
                    lldb.eArgTypeUnsignedInteger, repeat="plain", groups=None
                )
            ]
        )

    def __call__(self, debugger, args_array, exe_ctx, result):
        target = exe_ctx.target

        breakpoints = []
        for breakpoint in target.breakpoints:
            if breakpoint.enabled:
                breakpoints.append(breakpoint)
                breakpoint.SetEnabled(False)

        interval = 5
        process = exe_ctx.process
        # If the process is running we don't need to do anything, lldb
        # will auto-pause/continue around disabling the breakpoints,
        # but if we're stopped we want to continue.
        if process.IsValid() and process.state == lldb.eStateStopped:
            old_async = self.debugger.GetAsync()
            self.debugger.SetAsync(True)
            process.Continue()
            self.debugger.SetAsync(old_async)

        # num_args can only be 0 or 1 due to the command definition.
        num_args = args_array.GetSize()
        if num_args == 1:
            interval = args_array.GetItemAtIndex(0).GetUnsignedIntegerValue(5)
        time.sleep(interval)

        for breakpoint in breakpoints:
            breakpoint.SetEnabled(True)

    def get_short_help(self):
        return "Disable all enabled breakpoints, sleep and re-enable"

    def get_long_help(self):
        return (
            "Disable all enabled breakpoints in the target passed to the "
            "command; sleep for 5 seconds - or the interval passed as the "
            "argument command; then reenable them.  "
            "\nThis is useful when you need to set up a situation in a GUI app w/o "
            "hitting breakpoints, and then start hitting them again w/o disturbing "
            "the GUI state.  For instance, if you want to hit a breakpoint "
            "while testing a drag and drop action, you can't interact with "
            "the debugger once you've started the drag so you need a delayed "
            "enable."
        )


def __lldb_init_module(debugger, internal_dict):
    ParsedCommand.do_register_cmd(EnableThemLater, debugger, __name__)
