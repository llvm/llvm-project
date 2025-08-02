import os
import os.path
from enum import Enum
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.gdbclientutils import *


class WatchpointType(list[str], Enum):
    READ = ["r", "read"]
    WRITE = ["w", "write"]
    READ_WRITE = ["rw", "read_write"]
    MODIFY = ["m", "modify"]


class WatchpointCLICommandVariant(str, Enum):
    EXPRESSION = "expression"
    VARIABLE = "variable"


def _get_SBWatchpointOptions(wp_type, wp_mode):
    wp_opts = lldb.SBWatchpointOptions()

    if wp_type == WatchpointType.READ or wp_type == WatchpointType.READ_WRITE:
        wp_opts.SetWatchpointTypeRead(True)
    if wp_type == WatchpointType.WRITE or wp_type == WatchpointType.READ_WRITE:
        wp_opts.SetWatchpointTypeWrite(lldb.eWatchpointWriteTypeAlways)
    if wp_type == WatchpointType.MODIFY:
        wp_opts.SetWatchpointTypeWrite(lldb.eWatchpointWriteTypeOnModify)

    wp_opts.SetWatchpointMode(wp_mode)

    return wp_opts


def get_set_watchpoint_CLI_command(command_variant, wp_type, wp_mode):
    cmd = ["watchpoint set", command_variant.value, "-w", wp_type.value[1]]
    if wp_mode == lldb.eWatchpointModeSoftware:
        cmd.append("-S")
    return " ".join(map(str, cmd))


def set_watchpoint_at_value(value, wp_type, wp_mode, error):
    wp_opts = _get_SBWatchpointOptions(wp_type, wp_mode)
    return value.Watch(True, wp_opts, error)


def set_watchpoint_at_pointee(value, wp_type, wp_mode, error):
    wp_opts = _get_SBWatchpointOptions(wp_type, wp_mode)
    return value.WatchPointee(True, wp_opts, error)


def set_watchpoint_by_address(target, addr, size, wp_type, wp_mode, error):
    wp_opts = _get_SBWatchpointOptions(wp_type, wp_mode)
    return target.WatchpointCreateByAddress(addr, size, wp_opts, error)
