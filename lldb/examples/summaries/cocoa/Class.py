"""
LLDB AppKit formatters

part of The LLVM Compiler Infrastructure
This file is distributed under the University of Illinois Open Source
License. See LICENSE.TXT for details.
"""
import lldb
import lldb.runtime.objc.objc_runtime
import lldb.formatters.Logger


def Class_Summary(valobj, dict):
    logger = lldb.formatters.Logger.Logger()
    runtime = lldb.runtime.objc.objc_runtime.ObjCRuntime.runtime_from_isa(
        valobj)
    if runtime is None or not runtime.is_valid():
        return '<error: unknown Class>'
    class_data = runtime.read_class_data()
    if class_data is None or not class_data.is_valid():
        return '<error: unknown Class>'
    return class_data.class_name()
