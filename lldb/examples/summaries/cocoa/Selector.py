"""
LLDB AppKit formatters

part of The LLVM Compiler Infrastructure
This file is distributed under the University of Illinois Open Source
License. See LICENSE.TXT for details.
"""
import lldb


def SEL_Summary(valobj, dict):
    return valobj.Cast(valobj.GetType().GetBasicType(
        lldb.eBasicTypeChar).GetPointerType()).GetSummary()


def SELPointer_Summary(valobj, dict):
    return valobj.CreateValueFromAddress(
        'text', valobj.GetValueAsUnsigned(0), valobj.GetType().GetBasicType(
            lldb.eBasicTypeChar)).AddressOf().GetSummary()
