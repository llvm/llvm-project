"""
This is the llvm::Optional data formatter from llvm/utils/lldbDataFormatters.py
with the implementation replaced by bytecode.
"""
from __future__ import annotations
from compiler import *
import lldb


def __lldb_init_module(debugger, internal_dict):
    debugger.HandleCommand(
        "type synthetic add -w llvm "
        f"-l {__name__}.MyOptionalSynthProvider "
        '-x "^MyOptional<.+>$"'
    )
    debugger.HandleCommand(
        "type summary add -w llvm "
        f"-e -F {__name__}.MyOptionalSummaryProvider "
        '-x "^MyOptional<.+>$"'
    )


def evaluate(assembler: str, data: list):
    bytecode = compile(assembler)
    trace = True
    if trace:
        print(
            "Compiled to {0} bytes of bytecode:\n0x{1}".format(
                len(bytecode), bytecode.hex()
            )
        )
    result = interpret(bytecode, [], data, False)  # trace)
    if trace:
        print("--> {0}".format(result))
    return result


# def GetOptionalValue(valobj):
#    storage = valobj.GetChildMemberWithName("Storage")
#    if not storage:
#        storage = valobj
#
#    failure = 2
#    hasVal = storage.GetChildMemberWithName("hasVal").GetValueAsUnsigned(failure)
#    if hasVal == failure:
#        return "<could not read MyOptional>"
#
#    if hasVal == 0:
#        return None
#
#    underlying_type = storage.GetType().GetTemplateArgumentType(0)
#    storage = storage.GetChildMemberWithName("value")
#    return storage.Cast(underlying_type)


def MyOptionalSummaryProvider(valobj, internal_dict):
    #    val = GetOptionalValue(valobj)
    #    if val is None:
    #        return "None"
    #    if val.summary:
    #        return val.summary
    #    return val.GetValue()
    summary = ""
    summary += ' dup "Storage" @get_child_with_name call'  # valobj storage
    summary += " dup { swap } if drop"  # storage
    summary += ' dup "hasVal" @get_child_with_name call'  # storage
    summary += " @get_value_as_unsigned call"  # storage int(hasVal)
    summary += ' dup 2 = { drop "<could not read MyOptional>" } {'
    summary += '   0 = { "None" } {'
    summary += (
        "     dup @get_type call 0 @get_template_argument_type call"  # storage type
    )
    summary += "     swap"  # type storage
    summary += '     "value" @get_child_with_name call'  # type value
    summary += "     swap @cast call"  # type(value)
    summary += '     dup 0 = { "None" } {'
    summary += "       dup @summary call { @summary call } { @get_value call } ifelse"
    summary += "     } ifelse"
    summary += "   } ifelse"
    summary += " } ifelse"
    return evaluate(summary, [valobj])


class MyOptionalSynthProvider:
    """Provides deref support to llvm::Optional<T>"""

    def __init__(self, valobj, internal_dict):
        self.valobj = valobj

    def num_children(self):
        # return self.valobj.num_children
        num_children = " @get_num_children call"
        return evaluate(num_children, [self.valobj])

    def get_child_index(self, name):
        # if name == "$$dereference$$":
        #    return self.valobj.num_children
        # return self.valobj.GetIndexOfChildWithName(name)
        get_child_index = ' dup "$$dereference$$" ='
        get_child_index += " { drop @get_num_children call } {"  # obj name
        get_child_index += "   @get_child_index call"  # index
        get_child_index += " } ifelse"
        return evaluate(get_child_index, [self.valobj, name])

    def get_child_at_index(self, index):
        # if index < self.valobj.num_children:
        #    return self.valobj.GetChildAtIndex(index)
        # return GetOptionalValue(self.valobj) or lldb.SBValue()
        get_child_at_index = " over over swap"  # obj index index obj
        get_child_at_index += " @get_num_children call"  # obj index index n
        get_child_at_index += " < { @get_child_at_index call } {"  # obj index

        get_opt_val = ' dup "Storage" @get_child_with_name call'  # valobj storage
        get_opt_val += " dup { swap } if drop"  # storage
        get_opt_val += ' dup "hasVal" @get_child_with_name call'  # storage
        get_opt_val += " @get_value_as_unsigned call"  # storage int(hasVal)
        get_opt_val += ' dup 2 = { drop "<could not read MyOptional>" } {'
        get_opt_val += '   0 = { "None" } {'
        get_opt_val += (
            "     dup @get_type call 0 @get_template_argument_type call"  # storage type
        )
        get_opt_val += "     swap"  # type storage
        get_opt_val += '     "value" @get_child_with_name call'  # type value
        get_opt_val += "     swap @cast call"  # type(value)
        get_opt_val += "   } ifelse"
        get_opt_val += " } ifelse"

        get_child_at_index += get_opt_val
        get_child_at_index += " } ifelse"

        return evaluate(get_child_at_index, [self.valobj, index])
