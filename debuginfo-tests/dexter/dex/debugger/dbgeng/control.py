# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ctypes import *
from functools import partial

from .utils import *
from .breakpoint import *

class DEBUG_STACK_FRAME_EX(Structure):
  _fields_ = [
      ("InstructionOffset", c_ulonglong),
      ("ReturnOffset", c_ulonglong),
      ("FrameOffset", c_ulonglong),
      ("StackOffset", c_ulonglong),
      ("FuncTableEntry", c_ulonglong),
      ("Params", c_ulonglong * 4),
      ("Reserved", c_ulonglong * 6),
      ("Virtual", c_bool),
      ("FrameNumber", c_ulong),
      ("InlineFrameContext", c_ulong),
      ("Reserved1", c_ulong)
    ]
PDEBUG_STACK_FRAME_EX = POINTER(DEBUG_STACK_FRAME_EX)

class DEBUG_VALUE_U(Union):
  _fields_ = [
      ("I8", c_byte),
      ("I16", c_short),
      ("I32", c_int),
      ("I64", c_long),
      ("F32", c_float),
      ("F64", c_double),
      ("RawBytes", c_ubyte * 24) # Force length to 24b.
    ]

class DEBUG_VALUE(Structure):
  _fields_ = [
      ("U", DEBUG_VALUE_U),
      ("TailOfRawBytes", c_ulong),
      ("Type", c_ulong)
    ]
PDEBUG_VALUE = POINTER(DEBUG_VALUE)

class DebugValueType(IntEnum):
  DEBUG_VALUE_INVALID      = 0
  DEBUG_VALUE_INT8         = 1
  DEBUG_VALUE_INT16        = 2
  DEBUG_VALUE_INT32        = 3
  DEBUG_VALUE_INT64        = 4
  DEBUG_VALUE_FLOAT32      = 5
  DEBUG_VALUE_FLOAT64      = 6
  DEBUG_VALUE_FLOAT80      = 7
  DEBUG_VALUE_FLOAT82      = 8
  DEBUG_VALUE_FLOAT128     = 9
  DEBUG_VALUE_VECTOR64     = 10
  DEBUG_VALUE_VECTOR128    = 11
  DEBUG_VALUE_TYPES        = 12

# UUID for DebugControl7 interface.
DebugControl7IID = IID(0xb86fb3b1, 0x80d4, 0x475b, IID_Data4_Type(0xae, 0xa3, 0xcf, 0x06, 0x53, 0x9c, 0xf6, 0x3a))

class IDebugControl7(Structure):
  pass

class IDebugControl7Vtbl(Structure):
  wrp = partial(WINFUNCTYPE, c_long, POINTER(IDebugControl7))
  idc_getnumbereventfilters = wrp(c_ulong_p, c_ulong_p, c_ulong_p)
  idc_setexceptionfiltersecondcommand = wrp(c_ulong, c_char_p)
  idc_waitforevent = wrp(c_long, c_long)
  idc_execute = wrp(c_long, c_char_p, c_long)
  idc_setexpressionsyntax = wrp(c_ulong)
  idc_addbreakpoint2 = wrp(c_ulong, c_ulong, POINTER(POINTER(DebugBreakpoint2)))
  idc_setexecutionstatus = wrp(c_ulong)
  idc_getexecutionstatus = wrp(c_ulong_p)
  idc_getstacktraceex = wrp(c_ulonglong, c_ulonglong, c_ulonglong, PDEBUG_STACK_FRAME_EX, c_ulong, c_ulong_p)
  idc_evaluate = wrp(c_char_p, c_ulong, PDEBUG_VALUE, c_ulong_p)
  idc_setengineoptions = wrp(c_ulong)
  _fields_ = [
      ("QueryInterface", c_void_p),
      ("AddRef", c_void_p),
      ("Release", c_void_p),
      ("GetInterrupt", c_void_p),
      ("SetInterrupt", c_void_p),
      ("GetInterruptTimeout", c_void_p),
      ("SetInterruptTimeout", c_void_p),
      ("GetLogFile", c_void_p),
      ("OpenLogFile", c_void_p),
      ("CloseLogFile", c_void_p),
      ("GetLogMask", c_void_p),
      ("SetLogMask", c_void_p),
      ("Input", c_void_p),
      ("ReturnInput", c_void_p),
      ("Output", c_void_p),
      ("OutputVaList", c_void_p),
      ("ControlledOutput", c_void_p),
      ("ControlledOutputVaList", c_void_p),
      ("OutputPrompt", c_void_p),
      ("OutputPromptVaList", c_void_p),
      ("GetPromptText", c_void_p),
      ("OutputCurrentState", c_void_p),
      ("OutputVersionInformation", c_void_p),
      ("GetNotifyEventHandle", c_void_p),
      ("SetNotifyEventHandle", c_void_p),
      ("Assemble", c_void_p),
      ("Disassemble", c_void_p),
      ("GetDisassembleEffectiveOffset", c_void_p),
      ("OutputDisassembly", c_void_p),
      ("OutputDisassemblyLines", c_void_p),
      ("GetNearInstruction", c_void_p),
      ("GetStackTrace", c_void_p),
      ("GetReturnOffset", c_void_p),
      ("OutputStackTrace", c_void_p),
      ("GetDebuggeeType", c_void_p),
      ("GetActualProcessorType", c_void_p),
      ("GetExecutingProcessorType", c_void_p),
      ("GetNumberPossibleExecutingProcessorTypes", c_void_p),
      ("GetPossibleExecutingProcessorTypes", c_void_p),
      ("GetNumberProcessors", c_void_p),
      ("GetSystemVersion", c_void_p),
      ("GetPageSize", c_void_p),
      ("IsPointer64Bit", c_void_p),
      ("ReadBugCheckData", c_void_p),
      ("GetNumberSupportedProcessorTypes", c_void_p),
      ("GetSupportedProcessorTypes", c_void_p),
      ("GetProcessorTypeNames", c_void_p),
      ("GetEffectiveProcessorType", c_void_p),
      ("SetEffectiveProcessorType", c_void_p),
      ("GetExecutionStatus", idc_getexecutionstatus),
      ("SetExecutionStatus", idc_setexecutionstatus),
      ("GetCodeLevel", c_void_p),
      ("SetCodeLevel", c_void_p),
      ("GetEngineOptions", c_void_p),
      ("AddEngineOptions", c_void_p),
      ("RemoveEngineOptions", c_void_p),
      ("SetEngineOptions", idc_setengineoptions),
      ("GetSystemErrorControl", c_void_p),
      ("SetSystemErrorControl", c_void_p),
      ("GetTextMacro", c_void_p),
      ("SetTextMacro", c_void_p),
      ("GetRadix", c_void_p),
      ("SetRadix", c_void_p),
      ("Evaluate", idc_evaluate),
      ("CoerceValue", c_void_p),
      ("CoerceValues", c_void_p),
      ("Execute", idc_execute),
      ("ExecuteCommandFile", c_void_p),
      ("GetNumberBreakpoints", c_void_p),
      ("GetBreakpointByIndex", c_void_p),
      ("GetBreakpointById", c_void_p),
      ("GetBreakpointParameters", c_void_p),
      ("AddBreakpoint", c_void_p),
      ("RemoveBreakpoint", c_void_p),
      ("AddExtension", c_void_p),
      ("RemoveExtension", c_void_p),
      ("GetExtensionByPath", c_void_p),
      ("CallExtension", c_void_p),
      ("GetExtensionFunction", c_void_p),
      ("GetWindbgExtensionApis32", c_void_p),
      ("GetWindbgExtensionApis64", c_void_p),
      ("GetNumberEventFilters", idc_getnumbereventfilters),
      ("GetEventFilterText", c_void_p),
      ("GetEventFilterCommand", c_void_p),
      ("SetEventFilterCommand", c_void_p),
      ("GetSpecificFilterParameters", c_void_p),
      ("SetSpecificFilterParameters", c_void_p),
      ("GetSpecificFilterArgument", c_void_p),
      ("SetSpecificFilterArgument", c_void_p),
      ("GetExceptionFilterParameters", c_void_p),
      ("SetExceptionFilterParameters", c_void_p),
      ("GetExceptionFilterSecondCommand", c_void_p),
      ("SetExceptionFilterSecondCommand", idc_setexceptionfiltersecondcommand),
      ("WaitForEvent", idc_waitforevent),
      ("GetLastEventInformation", c_void_p),
      ("GetCurrentTimeDate", c_void_p),
      ("GetCurrentSystemUpTime", c_void_p),
      ("GetDumpFormatFlags", c_void_p),
      ("GetNumberTextReplacements", c_void_p),
      ("GetTextReplacement", c_void_p),
      ("SetTextReplacement", c_void_p),
      ("RemoveTextReplacements", c_void_p),
      ("OutputTextReplacements", c_void_p),
      ("GetAssemblyOptions", c_void_p),
      ("AddAssemblyOptions", c_void_p),
      ("RemoveAssemblyOptions", c_void_p),
      ("SetAssemblyOptions", c_void_p),
      ("GetExpressionSyntax", c_void_p),
      ("SetExpressionSyntax", idc_setexpressionsyntax),
      ("SetExpressionSyntaxByName", c_void_p),
      ("GetNumberExpressionSyntaxes", c_void_p),
      ("GetExpressionSyntaxNames", c_void_p),
      ("GetNumberEvents", c_void_p),
      ("GetEventIndexDescription", c_void_p),
      ("GetCurrentEventIndex", c_void_p),
      ("SetNextEventIndex", c_void_p),
      ("GetLogFileWide", c_void_p),
      ("OpenLogFileWide", c_void_p),
      ("InputWide", c_void_p),
      ("ReturnInputWide", c_void_p),
      ("OutputWide", c_void_p),
      ("OutputVaListWide", c_void_p),
      ("ControlledOutputWide", c_void_p),
      ("ControlledOutputVaListWide", c_void_p),
      ("OutputPromptWide", c_void_p),
      ("OutputPromptVaListWide", c_void_p),
      ("GetPromptTextWide", c_void_p),
      ("AssembleWide", c_void_p),
      ("DisassembleWide", c_void_p),
      ("GetProcessrTypeNamesWide", c_void_p),
      ("GetTextMacroWide", c_void_p),
      ("SetTextMacroWide", c_void_p),
      ("EvaluateWide", c_void_p),
      ("ExecuteWide", c_void_p),
      ("ExecuteCommandFileWide", c_void_p),
      ("GetBreakpointByIndex2", c_void_p),
      ("GetBreakpointById2", c_void_p),
      ("AddBreakpoint2", idc_addbreakpoint2),
      ("RemoveBreakpoint2", c_void_p),
      ("AddExtensionWide", c_void_p),
      ("GetExtensionByPathWide", c_void_p),
      ("CallExtensionWide", c_void_p),
      ("GetExtensionFunctionWide", c_void_p),
      ("GetEventFilterTextWide", c_void_p),
      ("GetEventfilterCommandWide", c_void_p),
      ("SetEventFilterCommandWide", c_void_p),
      ("GetSpecificFilterArgumentWide", c_void_p),
      ("SetSpecificFilterArgumentWide", c_void_p),
      ("GetExceptionFilterSecondCommandWide", c_void_p),
      ("SetExceptionFilterSecondCommandWider", c_void_p),
      ("GetLastEventInformationWide", c_void_p),
      ("GetTextReplacementWide", c_void_p),
      ("SetTextReplacementWide", c_void_p),
      ("SetExpressionSyntaxByNameWide", c_void_p),
      ("GetExpressionSyntaxNamesWide", c_void_p),
      ("GetEventIndexDescriptionWide", c_void_p),
      ("GetLogFile2", c_void_p),
      ("OpenLogFile2", c_void_p),
      ("GetLogFile2Wide", c_void_p),
      ("OpenLogFile2Wide", c_void_p),
      ("GetSystemVersionValues", c_void_p),
      ("GetSystemVersionString", c_void_p),
      ("GetSystemVersionStringWide", c_void_p),
      ("GetContextStackTrace", c_void_p),
      ("OutputContextStackTrace", c_void_p),
      ("GetStoredEventInformation", c_void_p),
      ("GetManagedStatus", c_void_p),
      ("GetManagedStatusWide", c_void_p),
      ("ResetManagedStatus", c_void_p),
      ("GetStackTraceEx", idc_getstacktraceex),
      ("OutputStackTraceEx", c_void_p),
      ("GetContextStackTraceEx", c_void_p),
      ("OutputContextStackTraceEx", c_void_p),
      ("GetBreakpointByGuid", c_void_p),
      ("GetExecutionStatusEx", c_void_p),
      ("GetSynchronizationStatus", c_void_p),
      ("GetDebuggeeType2", c_void_p)
    ]

IDebugControl7._fields_ = [("lpVtbl", POINTER(IDebugControl7Vtbl))]

class DebugStatus(IntEnum):
  DEBUG_STATUS_NO_CHANGE =            0
  DEBUG_STATUS_GO =                   1
  DEBUG_STATUS_GO_HANDLED =           2
  DEBUG_STATUS_GO_NOT_HANDLED =       3
  DEBUG_STATUS_STEP_OVER =            4
  DEBUG_STATUS_STEP_INTO =            5
  DEBUG_STATUS_BREAK =                6
  DEBUG_STATUS_NO_DEBUGGEE =          7
  DEBUG_STATUS_STEP_BRANCH =          8
  DEBUG_STATUS_IGNORE_EVENT =         9
  DEBUG_STATUS_RESTART_REQUESTED =   10
  DEBUG_STATUS_REVERSE_GO =          11
  DEBUG_STATUS_REVERSE_STEP_BRANCH = 12
  DEBUG_STATUS_REVERSE_STEP_OVER =   13
  DEBUG_STATUS_REVERSE_STEP_INTO =   14
  DEBUG_STATUS_OUT_OF_SYNC =         15
  DEBUG_STATUS_WAIT_INPUT =          16
  DEBUG_STATUS_TIMEOUT =             17

class DebugSyntax(IntEnum):
  DEBUG_EXPR_MASM = 0
  DEBUG_EXPR_CPLUSPLUS = 1

class Control(object):
  def __init__(self, control):
    self.ptr = control
    self.control = control.contents
    self.vt = self.control.lpVtbl.contents
    # Keep a handy ulong for passing into C methods.
    self.ulong = c_ulong()

  def GetExecutionStatus(self, doprint=False):
    ret = self.vt.GetExecutionStatus(self.control, byref(self.ulong))
    aborter(ret, "GetExecutionStatus")
    status = DebugStatus(self.ulong.value)
    if doprint:
      print("Execution status: {}".format(status))
    return status

  def SetExecutionStatus(self, status):
    assert isinstance(status, DebugStatus)
    res = self.vt.SetExecutionStatus(self.control, status.value)
    aborter(res, "SetExecutionStatus")

  def WaitForEvent(self, timeout=100):
    # No flags are taken by WaitForEvent, hence 0
    ret = self.vt.WaitForEvent(self.control, 0, timeout)
    aborter(ret, "WaitforEvent", ignore=[S_FALSE])
    return ret

  def GetNumberEventFilters(self):
    specific_events = c_ulong()
    specific_exceptions = c_ulong()
    arbitrary_exceptions = c_ulong()
    res = self.vt.GetNumberEventFilters(self.control, byref(specific_events),
                                    byref(specific_exceptions),
                                    byref(arbitrary_exceptions))
    aborter(res, "GetNumberEventFilters")
    return (specific_events.value, specific_exceptions.value,
            arbitrary_exceptions.value)

  def SetExceptionFilterSecondCommand(self, index, command):
    buf = create_string_buffer(command.encode('ascii'))
    res = self.vt.SetExceptionFilterSecondCommand(self.control, index, buf)
    aborter(res, "SetExceptionFilterSecondCommand")
    return

  def AddBreakpoint2(self, offset=None, enabled=None):
    breakpoint = POINTER(DebugBreakpoint2)()
    res = self.vt.AddBreakpoint2(self.control, BreakpointTypes.DEBUG_BREAKPOINT_CODE, DEBUG_ANY_ID, byref(breakpoint))
    aborter(res, "Add breakpoint 2")
    bp = Breakpoint(breakpoint)

    if offset is not None:
      bp.SetOffset(offset)
    if enabled is not None and enabled:
      bp.SetFlags(BreakpointFlags.DEBUG_BREAKPOINT_ENABLED)

    return bp

  def RemoveBreakpoint(self, bp):
    res = self.vt.RemoveBreakpoint2(self.control, bp.breakpoint)
    aborter(res, "RemoveBreakpoint2")
    bp.die()

  def GetStackTraceEx(self):
    # XXX -- I can't find a way to query for how many stack frames there _are_
    # in  advance. Guess 128 for now.
    num_frames_buffer = 128

    frames = (DEBUG_STACK_FRAME_EX * num_frames_buffer)()
    numframes = c_ulong()

    # First three args are frame/stack/IP offsets -- leave them as zero to
    # default to the current instruction.
    res = self.vt.GetStackTraceEx(self.control, 0, 0, 0, frames, num_frames_buffer, byref(numframes))
    aborter(res, "GetStackTraceEx")
    return frames, numframes.value

  def Execute(self, command):
    # First zero is DEBUG_OUTCTL_*, which we leave as a default, second
    # zero is DEBUG_EXECUTE_* flags, of which we set none.
    res = self.vt.Execute(self.control, 0, command.encode('ascii'), 0)
    aborter(res, "Client execute")

  def SetExpressionSyntax(self, cpp=True):
    if cpp:
      syntax = DebugSyntax.DEBUG_EXPR_CPLUSPLUS
    else:
      syntax = DebugSyntax.DEBUG_EXPR_MASM

    res = self.vt.SetExpressionSyntax(self.control, syntax)
    aborter(res, "SetExpressionSyntax")

  def Evaluate(self, expr):
    ptr = DEBUG_VALUE()
    res = self.vt.Evaluate(self.control, expr.encode("ascii"), DebugValueType.DEBUG_VALUE_INVALID, byref(ptr), None)
    aborter(res, "Evaluate", ignore=[E_INTERNALEXCEPTION, E_FAIL])
    if res != 0:
      return None

    val_type = DebugValueType(ptr.Type)

    # Here's a map from debug value types to fields. Unclear what happens
    # with unsigned values, as DbgEng doesn't present any unsigned fields.

    extract_map = {
      DebugValueType.DEBUG_VALUE_INT8    : ("I8", "char"),
      DebugValueType.DEBUG_VALUE_INT16   : ("I16", "short"),
      DebugValueType.DEBUG_VALUE_INT32   : ("I32", "int"),
      DebugValueType.DEBUG_VALUE_INT64   : ("I64", "long"),
      DebugValueType.DEBUG_VALUE_FLOAT32 : ("F32", "float"),
      DebugValueType.DEBUG_VALUE_FLOAT64 : ("F64", "double")
    } # And everything else is invalid.

    if val_type not in extract_map:
      raise Exception("Unexpected debug value type {} when evalutaing".format(val_type))

    # Also produce a type name...

    return getattr(ptr.U, extract_map[val_type][0]), extract_map[val_type][1]

  def SetEngineOptions(self, opt):
    res = self.vt.SetEngineOptions(self.control, opt)
    aborter(res, "SetEngineOptions")
    return
