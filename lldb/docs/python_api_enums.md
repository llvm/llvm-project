---
orphan: true
---

% This is a sub page of the Python API docs and linked from the main API page.
% The page isn't in any toctree, so silence the sphinx warnings by marking it as orphan.

# Python API enumerators and constants

```{eval-rst}
.. py:currentmodule:: lldb
```

## Constants

### Generic register numbers

```{eval-rst}
.. py:data:: LLDB_REGNUM_GENERIC_PC

   Program counter.
```

```{eval-rst}
.. py:data:: LLDB_REGNUM_GENERIC_SP

   Stack pointer.
```

```{eval-rst}
.. py:data:: LLDB_REGNUM_GENERIC_FP

   Frame pointer.
```

```{eval-rst}
.. py:data:: LLDB_REGNUM_GENERIC_RA

   Return address.
```

```{eval-rst}
.. py:data:: LLDB_REGNUM_GENERIC_FLAGS

   Processor flags register.
```

```{eval-rst}
.. py:data:: LLDB_REGNUM_GENERIC_ARG1

   The register that would contain pointer size or less argument 1 (if any).
```

```{eval-rst}
.. py:data:: LLDB_REGNUM_GENERIC_ARG2

   The register that would contain pointer size or less argument 2 (if any).
```

```{eval-rst}
.. py:data:: LLDB_REGNUM_GENERIC_ARG3

   The register that would contain pointer size or less argument 3 (if any).
```

```{eval-rst}
.. py:data:: LLDB_REGNUM_GENERIC_ARG4

   The register that would contain pointer size or less argument 4 (if any).
```

```{eval-rst}
.. py:data:: LLDB_REGNUM_GENERIC_ARG5

   The register that would contain pointer size or less argument 5 (if any).
```

```{eval-rst}
.. py:data:: LLDB_REGNUM_GENERIC_ARG6

   The register that would contain pointer size or less argument 6 (if any).
```

```{eval-rst}
.. py:data:: LLDB_REGNUM_GENERIC_ARG7

   The register that would contain pointer size or less argument 7 (if any).
```

```{eval-rst}
.. py:data:: LLDB_REGNUM_GENERIC_ARG8

   The register that would contain pointer size or less argument 8 (if any).

```

### Invalid value definitions

```{eval-rst}
.. py:data:: LLDB_INVALID_BREAK_ID
```

```{eval-rst}
.. py:data:: LLDB_INVALID_WATCH_ID
```

```{eval-rst}
.. py:data:: LLDB_INVALID_ADDRESS
```

```{eval-rst}
.. py:data:: LLDB_INVALID_INDEX32
```

```{eval-rst}
.. py:data:: LLDB_INVALID_IVAR_OFFSET
```

```{eval-rst}
.. py:data:: LLDB_INVALID_IMAGE_TOKEN
```

```{eval-rst}
.. py:data:: LLDB_INVALID_MODULE_VERSION
```

```{eval-rst}
.. py:data:: LLDB_INVALID_REGNUM
```

```{eval-rst}
.. py:data:: LLDB_INVALID_UID
```

```{eval-rst}
.. py:data:: LLDB_INVALID_PROCESS_ID
```

```{eval-rst}
.. py:data:: LLDB_INVALID_THREAD_ID
```

```{eval-rst}
.. py:data:: LLDB_INVALID_FRAME_ID
```

```{eval-rst}
.. py:data:: LLDB_INVALID_SIGNAL_NUMBER
```

```{eval-rst}
.. py:data:: LLDB_INVALID_OFFSET
```

```{eval-rst}
.. py:data:: LLDB_INVALID_LINE_NUMBER
```

```{eval-rst}
.. py:data:: LLDB_INVALID_QUEUE_ID
```

### CPU types

```{eval-rst}
.. py:data:: LLDB_ARCH_DEFAULT
```

```{eval-rst}
.. py:data:: LLDB_ARCH_DEFAULT_32BIT
```

```{eval-rst}
.. py:data:: LLDB_ARCH_DEFAULT_64BIT
```

```{eval-rst}
.. py:data:: LLDB_INVALID_CPUTYPE

```

### Option set definitions

```{eval-rst}
.. py:data:: LLDB_MAX_NUM_OPTION_SETS
```

```{eval-rst}
.. py:data:: LLDB_OPT_SET_ALL
```

```{eval-rst}
.. py:data:: LLDB_OPT_SET_1
```

```{eval-rst}
.. py:data:: LLDB_OPT_SET_2
```

```{eval-rst}
.. py:data:: LLDB_OPT_SET_3
```

```{eval-rst}
.. py:data:: LLDB_OPT_SET_4
```

```{eval-rst}
.. py:data:: LLDB_OPT_SET_5
```

```{eval-rst}
.. py:data:: LLDB_OPT_SET_6
```

```{eval-rst}
.. py:data:: LLDB_OPT_SET_7
```

```{eval-rst}
.. py:data:: LLDB_OPT_SET_8
```

```{eval-rst}
.. py:data:: LLDB_OPT_SET_9
```

```{eval-rst}
.. py:data:: LLDB_OPT_SET_10
```

```{eval-rst}
.. py:data:: LLDB_OPT_SET_11
```

### Miscellaneous constants

```{eval-rst}
.. py:data:: LLDB_GENERIC_ERROR
```

```{eval-rst}
.. py:data:: LLDB_DEFAULT_BREAK_SIZE
```

```{eval-rst}
.. py:data:: LLDB_WATCH_TYPE_READ
```

```{eval-rst}
.. py:data:: LLDB_WATCH_TYPE_WRITE

```

## Enumerators

(state)=

### State

```{eval-rst}
.. py:data:: eStateInvalid
```

```{eval-rst}
.. py:data:: eStateUnloaded

   Process is object is valid, but not currently loaded.
```

```{eval-rst}
.. py:data:: eStateConnected

   Process is connected to remote debug services, but not
   launched or attached to anything yet.
```

```{eval-rst}
.. py:data:: eStateAttaching

   Process is in the process of launching.
```

```{eval-rst}
.. py:data:: eStateLaunching

   Process is in the process of launching.
```

```{eval-rst}
.. py:data:: eStateStopped

   Process or thread is stopped and can be examined.
```

```{eval-rst}
.. py:data:: eStateRunning

   Process or thread is running and can't be examined.
```

```{eval-rst}
.. py:data:: eStateStepping

   Process or thread is in the process of stepping and can
   not be examined.
```

```{eval-rst}
.. py:data:: eStateCrashed

   Process or thread has crashed and can be examined.
```

```{eval-rst}
.. py:data:: eStateDetached

   Process has been detached and can't be examined.
```

```{eval-rst}
.. py:data:: eStateExited

   Process has exited and can't be examined.
```

```{eval-rst}
.. py:data:: eStateSuspended

   Process or thread is in a suspended state as far
   as the debugger is concerned while other processes
   or threads get the chance to run.

```

(launchflag)=

### LaunchFlag

```{eval-rst}
.. py:data:: eLaunchFlagNone
```

```{eval-rst}
.. py:data:: eLaunchFlagExec

   Exec when launching and turn the calling process into a new process.
```

```{eval-rst}
.. py:data:: eLaunchFlagDebug

   Stop as soon as the process launches to allow the process to be debugged.
```

```{eval-rst}
.. py:data:: eLaunchFlagStopAtEntry

   Stop at the program entry point instead of auto-continuing when launching or attaching at entry point.
```

```{eval-rst}
.. py:data:: eLaunchFlagDisableASLR

   Disable Address Space Layout Randomization.
```

```{eval-rst}
.. py:data:: eLaunchFlagDisableSTDIO

   Disable stdio for inferior process (e.g. for a GUI app).
```

```{eval-rst}
.. py:data:: eLaunchFlagLaunchInTTY

   Launch the process in a new TTY if supported by the host.
```

```{eval-rst}
.. py:data:: eLaunchFlagLaunchInShell

   Launch the process inside a shell to get shell expansion.
```

```{eval-rst}
.. py:data:: eLaunchFlagLaunchInSeparateProcessGroup

   Launch the process in a separate process group if you are going to hand the process off (e.g. to debugserver)
```

```{eval-rst}
.. py:data:: eLaunchFlagDontSetExitStatus

   set this flag so lldb & the handee don't race to set its exit status.
```

```{eval-rst}
.. py:data:: eLaunchFlagDetachOnError

   If set, then the client stub should detach rather than killing  the debugee
   if it loses connection with lldb.
```

```{eval-rst}
.. py:data:: eLaunchFlagShellExpandArguments

   Perform shell-style argument expansion
```

```{eval-rst}
.. py:data:: eLaunchFlagCloseTTYOnExit

   Close the open TTY on exit
```

```{eval-rst}
.. py:data:: eLaunchFlagInheritTCCFromParent

   Don't make the inferior responsible for its own TCC
   permissions but instead inherit them from its parent.

```

(runmode)=

### RunMode

```{eval-rst}
.. py:data:: eOnlyThisThread
```

```{eval-rst}
.. py:data:: eAllThreads
```

```{eval-rst}
.. py:data:: eOnlyDuringStepping

```

(byteorder)=

### ByteOrder

```{eval-rst}
.. py:data:: eByteOrderInvalid
```

```{eval-rst}
.. py:data:: eByteOrderBig
```

```{eval-rst}
.. py:data:: eByteOrderPDP
```

```{eval-rst}
.. py:data:: eByteOrderLittle

```

(encoding)=

### Encoding

```{eval-rst}
.. py:data:: eEncodingInvalid
```

```{eval-rst}
.. py:data:: eEncodingUint
```

```{eval-rst}
.. py:data:: eEncodingSint
```

```{eval-rst}
.. py:data:: eEncodingIEEE754
```

```{eval-rst}
.. py:data:: eEncodingVector

```

(format)=

### Format

```{eval-rst}
.. py:data:: eFormatDefault
```

```{eval-rst}
.. py:data:: eFormatInvalid
```

```{eval-rst}
.. py:data:: eFormatBoolean
```

```{eval-rst}
.. py:data:: eFormatBinary
```

```{eval-rst}
.. py:data:: eFormatBytes
```

```{eval-rst}
.. py:data:: eFormatBytesWithASCII
```

```{eval-rst}
.. py:data:: eFormatChar
```

```{eval-rst}
.. py:data:: eFormatCharPrintable
```

```{eval-rst}
.. py:data:: eFormatComplex
```

```{eval-rst}
.. py:data:: eFormatComplexFloat
```

```{eval-rst}
.. py:data:: eFormatCString
```

```{eval-rst}
.. py:data:: eFormatDecimal
```

```{eval-rst}
.. py:data:: eFormatEnum
```

```{eval-rst}
.. py:data:: eFormatHex
```

```{eval-rst}
.. py:data:: eFormatHexUppercase
```

```{eval-rst}
.. py:data:: eFormatFloat
```

```{eval-rst}
.. py:data:: eFormatOctal
```

```{eval-rst}
.. py:data:: eFormatOSType
```

```{eval-rst}
.. py:data:: eFormatUnicode16
```

```{eval-rst}
.. py:data:: eFormatUnicode32
```

```{eval-rst}
.. py:data:: eFormatUnsigned
```

```{eval-rst}
.. py:data:: eFormatPointer
```

```{eval-rst}
.. py:data:: eFormatVectorOfChar
```

```{eval-rst}
.. py:data:: eFormatVectorOfSInt8
```

```{eval-rst}
.. py:data:: eFormatVectorOfUInt8
```

```{eval-rst}
.. py:data:: eFormatVectorOfSInt16
```

```{eval-rst}
.. py:data:: eFormatVectorOfUInt16
```

```{eval-rst}
.. py:data:: eFormatVectorOfSInt32
```

```{eval-rst}
.. py:data:: eFormatVectorOfUInt32
```

```{eval-rst}
.. py:data:: eFormatVectorOfSInt64
```

```{eval-rst}
.. py:data:: eFormatVectorOfUInt64
```

```{eval-rst}
.. py:data:: eFormatVectorOfFloat16
```

```{eval-rst}
.. py:data:: eFormatVectorOfFloat32
```

```{eval-rst}
.. py:data:: eFormatVectorOfFloat64
```

```{eval-rst}
.. py:data:: eFormatVectorOfUInt128
```

```{eval-rst}
.. py:data:: eFormatComplexInteger
```

```{eval-rst}
.. py:data:: eFormatCharArray
```

```{eval-rst}
.. py:data:: eFormatAddressInfo
```

```{eval-rst}
.. py:data:: eFormatHexFloat
```

```{eval-rst}
.. py:data:: eFormatInstruction
```

```{eval-rst}
.. py:data:: eFormatVoid
```

```{eval-rst}
.. py:data:: eFormatUnicode8
```

```{eval-rst}
.. py:data:: eFormatFloat128

```

(descriptionlevel)=

### DescriptionLevel

```{eval-rst}
.. py:data:: eDescriptionLevelBrief
```

```{eval-rst}
.. py:data:: eDescriptionLevelFull
```

```{eval-rst}
.. py:data:: eDescriptionLevelVerbose
```

```{eval-rst}
.. py:data:: eDescriptionLevelInitial

```

(scriptlanguage)=

### ScriptLanguage

```{eval-rst}
.. py:data:: eScriptLanguageNone
```

```{eval-rst}
.. py:data:: eScriptLanguagePython
```

```{eval-rst}
.. py:data:: eScriptLanguageLua
```

```{eval-rst}
.. py:data:: eScriptLanguageUnknown
```

```{eval-rst}
.. py:data:: eScriptLanguageDefault

```

(registerkind)=

### RegisterKind

```{eval-rst}
.. py:data:: eRegisterKindEHFrame
```

```{eval-rst}
.. py:data:: eRegisterKindDWARF
```

```{eval-rst}
.. py:data:: eRegisterKindGeneric
```

```{eval-rst}
.. py:data:: eRegisterKindProcessPlugin
```

```{eval-rst}
.. py:data:: eRegisterKindLLDB

```

(stopreason)=

### StopReason

```{eval-rst}
.. py:data:: eStopReasonInvalid
```

```{eval-rst}
.. py:data:: eStopReasonNone
```

```{eval-rst}
.. py:data:: eStopReasonTrace
```

```{eval-rst}
.. py:data:: eStopReasonBreakpoint
```

```{eval-rst}
.. py:data:: eStopReasonWatchpoint
```

```{eval-rst}
.. py:data:: eStopReasonSignal
```

```{eval-rst}
.. py:data:: eStopReasonException
```

```{eval-rst}
.. py:data:: eStopReasonExec
```

```{eval-rst}
.. py:data:: eStopReasonFork
```

```{eval-rst}
.. py:data:: eStopReasonVFork
```

```{eval-rst}
.. py:data:: eStopReasonVForkDone
```

```{eval-rst}
.. py:data:: eStopReasonPlanComplete
```

```{eval-rst}
.. py:data:: eStopReasonThreadExiting
```

```{eval-rst}
.. py:data:: eStopReasonInstrumentation

```

(returnstatus)=

### ReturnStatus

```{eval-rst}
.. py:data:: eReturnStatusInvalid
```

```{eval-rst}
.. py:data:: eReturnStatusSuccessFinishNoResult
```

```{eval-rst}
.. py:data:: eReturnStatusSuccessFinishResult
```

```{eval-rst}
.. py:data:: eReturnStatusSuccessContinuingNoResult
```

```{eval-rst}
.. py:data:: eReturnStatusSuccessContinuingResult
```

```{eval-rst}
.. py:data:: eReturnStatusStarted
```

```{eval-rst}
.. py:data:: eReturnStatusFailed
```

```{eval-rst}
.. py:data:: eReturnStatusQuit

```

(expression)=

### Expression

The results of expression evaluation.

```{eval-rst}
.. py:data:: eExpressionCompleted
```

```{eval-rst}
.. py:data:: eExpressionSetupError
```

```{eval-rst}
.. py:data:: eExpressionParseError
```

```{eval-rst}
.. py:data:: eExpressionDiscarded
```

```{eval-rst}
.. py:data:: eExpressionInterrupted
```

```{eval-rst}
.. py:data:: eExpressionHitBreakpoint
```

```{eval-rst}
.. py:data:: eExpressionTimedOut
```

```{eval-rst}
.. py:data:: eExpressionResultUnavailable
```

```{eval-rst}
.. py:data:: eExpressionStoppedForDebug
```

```{eval-rst}
.. py:data:: eExpressionThreadVanished

```

(searchdepth)=

### SearchDepth

```{eval-rst}
.. py:data:: eSearchDepthInvalid
```

```{eval-rst}
.. py:data:: eSearchDepthTarget
```

```{eval-rst}
.. py:data:: eSearchDepthModule
```

```{eval-rst}
.. py:data:: eSearchDepthCompUnit
```

```{eval-rst}
.. py:data:: eSearchDepthFunction
```

```{eval-rst}
.. py:data:: eSearchDepthBlock
```

```{eval-rst}
.. py:data:: eSearchDepthAddress

```

(connectionstatus)=

### ConnectionStatus

```{eval-rst}
.. py:data:: eConnectionStatusSuccess

   Success.
```

```{eval-rst}
.. py:data:: eConnectionStatusEndOfFile

   End-of-file encountered.
```

```{eval-rst}
.. py:data:: eConnectionStatusError

   Error encountered.
```

```{eval-rst}
.. py:data:: eConnectionStatusTimedOut

   Request timed out.
```

```{eval-rst}
.. py:data:: eConnectionStatusNoConnection

   No connection.
```

```{eval-rst}
.. py:data:: eConnectionStatusLostConnection

   Lost connection while connected to a  valid connection.
```

```{eval-rst}
.. py:data:: eConnectionStatusInterrupted

   Interrupted read.

```

(errortype)=

### ErrorType

```{eval-rst}
.. py:data:: eErrorTypeInvalid
```

```{eval-rst}
.. py:data:: eErrorTypeGeneric

   Generic errors that can be any value.
```

```{eval-rst}
.. py:data:: eErrorTypeMachKernel

   Mach kernel error codes.
```

```{eval-rst}
.. py:data:: eErrorTypePOSIX

   POSIX error codes.
```

```{eval-rst}
.. py:data:: eErrorTypeExpression

   These are from the ExpressionResults enum.
```

```{eval-rst}
.. py:data:: eErrorTypeWin32

   Standard Win32 error codes.

```

(valuetype)=

### ValueType

```{eval-rst}
.. py:data:: eValueTypeInvalid
```

```{eval-rst}
.. py:data:: eValueTypeVariableGlobal

   Global variable.
```

```{eval-rst}
.. py:data:: eValueTypeVariableStatic

   Static variable.
```

```{eval-rst}
.. py:data:: eValueTypeVariableArgument

   Function argument variable.
```

```{eval-rst}
.. py:data:: eValueTypeVariableLocal

   Function local variable.
```

```{eval-rst}
.. py:data:: eValueTypeRegister

   Stack frame register.
```

```{eval-rst}
.. py:data:: eValueTypeRegisterSet

   A collection of stack frame register values.
```

```{eval-rst}
.. py:data:: eValueTypeConstResult

   Constant result variables.
```

```{eval-rst}
.. py:data:: eValueTypeVariableThreadLocal

   Thread local storage variable.

```

(inputreadergranularity)=

### InputReaderGranularity

Token size/granularities for Input Readers.

```{eval-rst}
.. py:data:: eInputReaderGranularityInvalid
```

```{eval-rst}
.. py:data:: eInputReaderGranularityByte
```

```{eval-rst}
.. py:data:: eInputReaderGranularityWord
```

```{eval-rst}
.. py:data:: eInputReaderGranularityLine
```

```{eval-rst}
.. py:data:: eInputReaderGranularityAll

```

(symbolcontextitem)=

### SymbolContextItem

These mask bits allow a common interface for queries that can
limit the amount of information that gets parsed to only the
information that is requested. These bits also can indicate what
actually did get resolved during query function calls.

Each definition corresponds to one of the member variables
in this class, and requests that that item be resolved, or
indicates that the member did get resolved.

```{eval-rst}
.. py:data:: eSymbolContextTarget

   Set when target is requested from a query, or was located
   in query results.
```

```{eval-rst}
.. py:data:: eSymbolContextModule

   Set when module is requested from a query, or was located
   in query results.
```

```{eval-rst}
.. py:data:: eSymbolContextCompUnit

   Set when compilation unit is requested from a query, or was
   located in query results.
```

```{eval-rst}
.. py:data:: eSymbolContextFunction

   Set when function is requested from a query, or was located
   in query results.
```

```{eval-rst}
.. py:data:: eSymbolContextBlock

   Set when the deepest block is requested from a query, or
   was located in query results.
```

```{eval-rst}
.. py:data:: eSymbolContextLineEntry

   Set when line entry is requested from a query, or was
   located in query results.
```

```{eval-rst}
.. py:data:: eSymbolContextSymbol

   Set when symbol is requested from a query, or was located
   in query results
```

```{eval-rst}
.. py:data:: eSymbolContextEverything

   Indicates to try and lookup everything up during a routine
   symbol context query.
```

```{eval-rst}
.. py:data:: eSymbolContextVariable

   Set when global or static variable is requested from a
   query, or was located in query results.
   eSymbolContextVariable is potentially expensive to lookup so
   it isn't included in eSymbolContextEverything which stops it
   from being used during frame PC lookups and many other
   potential address to symbol context lookups.

```

(permissions)=

### Permissions

```{eval-rst}
.. py:data:: ePermissionsWritable
```

```{eval-rst}
.. py:data:: ePermissionsReadable
```

```{eval-rst}
.. py:data:: ePermissionsExecutable

```

(inputreader)=

### InputReader

```{eval-rst}
.. py:data:: eInputReaderActivate

   Reader is newly pushed onto the reader stack.
```

```{eval-rst}
.. py:data:: eInputReaderAsynchronousOutputWritten

   An async output event occurred; the reader may want to do something.
```

```{eval-rst}
.. py:data:: eInputReaderReactivate

   Reader is on top of the stack again after another  reader was popped off.
```

```{eval-rst}
.. py:data:: eInputReaderDeactivate

   Another reader was pushed on the stack.
```

```{eval-rst}
.. py:data:: eInputReaderGotToken

   Reader got one of its tokens (granularity).
```

```{eval-rst}
.. py:data:: eInputReaderInterrupt

   Reader received an interrupt signal (probably from  a control-c).
```

```{eval-rst}
.. py:data:: eInputReaderEndOfFile

   Reader received an EOF char (probably from a control-d).
```

```{eval-rst}
.. py:data:: eInputReaderDone

   Reader was just popped off the stack and is done.

```

(breakpointeventtype)=

### BreakpointEventType

```{eval-rst}
.. py:data:: eBreakpointEventTypeInvalidType
```

```{eval-rst}
.. py:data:: eBreakpointEventTypeAdded
```

```{eval-rst}
.. py:data:: eBreakpointEventTypeRemoved
```

```{eval-rst}
.. py:data:: eBreakpointEventTypeLocationsAdded
```

```{eval-rst}
.. py:data:: eBreakpointEventTypeLocationsRemoved
```

```{eval-rst}
.. py:data:: eBreakpointEventTypeLocationsResolved
```

```{eval-rst}
.. py:data:: eBreakpointEventTypeEnabled
```

```{eval-rst}
.. py:data:: eBreakpointEventTypeDisabled
```

```{eval-rst}
.. py:data:: eBreakpointEventTypeCommandChanged
```

```{eval-rst}
.. py:data:: eBreakpointEventTypeConditionChanged
```

```{eval-rst}
.. py:data:: eBreakpointEventTypeIgnoreChanged
```

```{eval-rst}
.. py:data:: eBreakpointEventTypeThreadChanged
```

```{eval-rst}
.. py:data:: eBreakpointEventTypeAutoContinueChanged

```

(watchpointeventtype)=

### WatchpointEventType

```{eval-rst}
.. py:data:: eWatchpointEventTypeInvalidType
```

```{eval-rst}
.. py:data:: eWatchpointEventTypeAdded
```

```{eval-rst}
.. py:data:: eWatchpointEventTypeRemoved
```

```{eval-rst}
.. py:data:: eWatchpointEventTypeEnabled
```

```{eval-rst}
.. py:data:: eWatchpointEventTypeDisabled
```

```{eval-rst}
.. py:data:: eWatchpointEventTypeCommandChanged
```

```{eval-rst}
.. py:data:: eWatchpointEventTypeConditionChanged
```

```{eval-rst}
.. py:data:: eWatchpointEventTypeIgnoreChanged
```

```{eval-rst}
.. py:data:: eWatchpointEventTypeThreadChanged
```

```{eval-rst}
.. py:data:: eWatchpointEventTypeTypeChanged

```

(languagetype)=

### LanguageType

```{eval-rst}
.. py:data:: eLanguageTypeUnknown
```

```{eval-rst}
.. py:data:: eLanguageTypeC89
```

```{eval-rst}
.. py:data:: eLanguageTypeC
```

```{eval-rst}
.. py:data:: eLanguageTypeAda83
```

```{eval-rst}
.. py:data:: eLanguageTypeC_plus_plus
```

```{eval-rst}
.. py:data:: eLanguageTypeCobol74
```

```{eval-rst}
.. py:data:: eLanguageTypeCobol85
```

```{eval-rst}
.. py:data:: eLanguageTypeFortran77
```

```{eval-rst}
.. py:data:: eLanguageTypeFortran90
```

```{eval-rst}
.. py:data:: eLanguageTypePascal83
```

```{eval-rst}
.. py:data:: eLanguageTypeModula2
```

```{eval-rst}
.. py:data:: eLanguageTypeJava
```

```{eval-rst}
.. py:data:: eLanguageTypeC99
```

```{eval-rst}
.. py:data:: eLanguageTypeAda95
```

```{eval-rst}
.. py:data:: eLanguageTypeFortran95
```

```{eval-rst}
.. py:data:: eLanguageTypePLI
```

```{eval-rst}
.. py:data:: eLanguageTypeObjC
```

```{eval-rst}
.. py:data:: eLanguageTypeObjC_plus_plus
```

```{eval-rst}
.. py:data:: eLanguageTypeUPC
```

```{eval-rst}
.. py:data:: eLanguageTypeD
```

```{eval-rst}
.. py:data:: eLanguageTypePython
```

```{eval-rst}
.. py:data:: eLanguageTypeOpenCL
```

```{eval-rst}
.. py:data:: eLanguageTypeGo
```

```{eval-rst}
.. py:data:: eLanguageTypeModula3
```

```{eval-rst}
.. py:data:: eLanguageTypeHaskell
```

```{eval-rst}
.. py:data:: eLanguageTypeC_plus_plus_03
```

```{eval-rst}
.. py:data:: eLanguageTypeC_plus_plus_11
```

```{eval-rst}
.. py:data:: eLanguageTypeOCaml
```

```{eval-rst}
.. py:data:: eLanguageTypeRust
```

```{eval-rst}
.. py:data:: eLanguageTypeC11
```

```{eval-rst}
.. py:data:: eLanguageTypeSwift
```

```{eval-rst}
.. py:data:: eLanguageTypeJulia
```

```{eval-rst}
.. py:data:: eLanguageTypeDylan
```

```{eval-rst}
.. py:data:: eLanguageTypeC_plus_plus_14
```

```{eval-rst}
.. py:data:: eLanguageTypeFortran03
```

```{eval-rst}
.. py:data:: eLanguageTypeFortran08
```

```{eval-rst}
.. py:data:: eLanguageTypeMipsAssembler
```

```{eval-rst}
.. py:data:: eLanguageTypeMojo
```

```{eval-rst}
.. py:data:: eLanguageTypeExtRenderScript
```

```{eval-rst}
.. py:data:: eNumLanguageTypes

```

(instrumentationruntimetype)=

### InstrumentationRuntimeType

```{eval-rst}
.. py:data:: eInstrumentationRuntimeTypeAddressSanitizer
```

```{eval-rst}
.. py:data:: eInstrumentationRuntimeTypeThreadSanitizer
```

```{eval-rst}
.. py:data:: eInstrumentationRuntimeTypeUndefinedBehaviorSanitizer
```

```{eval-rst}
.. py:data:: eInstrumentationRuntimeTypeMainThreadChecker
```

```{eval-rst}
.. py:data:: eInstrumentationRuntimeTypeSwiftRuntimeReporting
```

```{eval-rst}
.. py:data:: eNumInstrumentationRuntimeTypes

```

(dynamicvaluetype)=

### DynamicValueType

```{eval-rst}
.. py:data:: eNoDynamicValues
```

```{eval-rst}
.. py:data:: eDynamicCanRunTarget
```

```{eval-rst}
.. py:data:: eDynamicDontRunTarget

```

(stopshowcolumn)=

### StopShowColumn

```{eval-rst}
.. py:data:: eStopShowColumnAnsiOrCaret
```

```{eval-rst}
.. py:data:: eStopShowColumnAnsi
```

```{eval-rst}
.. py:data:: eStopShowColumnCaret
```

```{eval-rst}
.. py:data:: eStopShowColumnNone

```

(accesstype)=

### AccessType

```{eval-rst}
.. py:data:: eAccessNone
```

```{eval-rst}
.. py:data:: eAccessPublic
```

```{eval-rst}
.. py:data:: eAccessPrivate
```

```{eval-rst}
.. py:data:: eAccessProtected
```

```{eval-rst}
.. py:data:: eAccessPackage

```

(commandargumenttype)=

### CommandArgumentType

```{eval-rst}
.. py:data:: eArgTypeAddress
```

```{eval-rst}
.. py:data:: eArgTypeAddressOrExpression
```

```{eval-rst}
.. py:data:: eArgTypeAliasName
```

```{eval-rst}
.. py:data:: eArgTypeAliasOptions
```

```{eval-rst}
.. py:data:: eArgTypeArchitecture
```

```{eval-rst}
.. py:data:: eArgTypeBoolean
```

```{eval-rst}
.. py:data:: eArgTypeBreakpointID
```

```{eval-rst}
.. py:data:: eArgTypeBreakpointIDRange
```

```{eval-rst}
.. py:data:: eArgTypeBreakpointName
```

```{eval-rst}
.. py:data:: eArgTypeByteSize
```

```{eval-rst}
.. py:data:: eArgTypeClassName
```

```{eval-rst}
.. py:data:: eArgTypeCommandName
```

```{eval-rst}
.. py:data:: eArgTypeCount
```

```{eval-rst}
.. py:data:: eArgTypeDescriptionVerbosity
```

```{eval-rst}
.. py:data:: eArgTypeDirectoryName
```

```{eval-rst}
.. py:data:: eArgTypeDisassemblyFlavor
```

```{eval-rst}
.. py:data:: eArgTypeEndAddress
```

```{eval-rst}
.. py:data:: eArgTypeExpression
```

```{eval-rst}
.. py:data:: eArgTypeExpressionPath
```

```{eval-rst}
.. py:data:: eArgTypeExprFormat
```

```{eval-rst}
.. py:data:: eArgTypeFileLineColumn
```

```{eval-rst}
.. py:data:: eArgTypeFilename
```

```{eval-rst}
.. py:data:: eArgTypeFormat
```

```{eval-rst}
.. py:data:: eArgTypeFrameIndex
```

```{eval-rst}
.. py:data:: eArgTypeFullName
```

```{eval-rst}
.. py:data:: eArgTypeFunctionName
```

```{eval-rst}
.. py:data:: eArgTypeFunctionOrSymbol
```

```{eval-rst}
.. py:data:: eArgTypeGDBFormat
```

```{eval-rst}
.. py:data:: eArgTypeHelpText
```

```{eval-rst}
.. py:data:: eArgTypeIndex
```

```{eval-rst}
.. py:data:: eArgTypeLanguage
```

```{eval-rst}
.. py:data:: eArgTypeLineNum
```

```{eval-rst}
.. py:data:: eArgTypeLogCategory
```

```{eval-rst}
.. py:data:: eArgTypeLogChannel
```

```{eval-rst}
.. py:data:: eArgTypeMethod
```

```{eval-rst}
.. py:data:: eArgTypeName
```

```{eval-rst}
.. py:data:: eArgTypeNewPathPrefix
```

```{eval-rst}
.. py:data:: eArgTypeNumLines
```

```{eval-rst}
.. py:data:: eArgTypeNumberPerLine
```

```{eval-rst}
.. py:data:: eArgTypeOffset
```

```{eval-rst}
.. py:data:: eArgTypeOldPathPrefix
```

```{eval-rst}
.. py:data:: eArgTypeOneLiner
```

```{eval-rst}
.. py:data:: eArgTypePath
```

```{eval-rst}
.. py:data:: eArgTypePermissionsNumber
```

```{eval-rst}
.. py:data:: eArgTypePermissionsString
```

```{eval-rst}
.. py:data:: eArgTypePid
```

```{eval-rst}
.. py:data:: eArgTypePlugin
```

```{eval-rst}
.. py:data:: eArgTypeProcessName
```

```{eval-rst}
.. py:data:: eArgTypePythonClass
```

```{eval-rst}
.. py:data:: eArgTypePythonFunction
```

```{eval-rst}
.. py:data:: eArgTypePythonScript
```

```{eval-rst}
.. py:data:: eArgTypeQueueName
```

```{eval-rst}
.. py:data:: eArgTypeRegisterName
```

```{eval-rst}
.. py:data:: eArgTypeRegularExpression
```

```{eval-rst}
.. py:data:: eArgTypeRunArgs
```

```{eval-rst}
.. py:data:: eArgTypeRunMode
```

```{eval-rst}
.. py:data:: eArgTypeScriptedCommandSynchronicity
```

```{eval-rst}
.. py:data:: eArgTypeScriptLang
```

```{eval-rst}
.. py:data:: eArgTypeSearchWord
```

```{eval-rst}
.. py:data:: eArgTypeSelector
```

```{eval-rst}
.. py:data:: eArgTypeSettingIndex
```

```{eval-rst}
.. py:data:: eArgTypeSettingKey
```

```{eval-rst}
.. py:data:: eArgTypeSettingPrefix
```

```{eval-rst}
.. py:data:: eArgTypeSettingVariableName
```

```{eval-rst}
.. py:data:: eArgTypeShlibName
```

```{eval-rst}
.. py:data:: eArgTypeSourceFile
```

```{eval-rst}
.. py:data:: eArgTypeSortOrder
```

```{eval-rst}
.. py:data:: eArgTypeStartAddress
```

```{eval-rst}
.. py:data:: eArgTypeSummaryString
```

```{eval-rst}
.. py:data:: eArgTypeSymbol
```

```{eval-rst}
.. py:data:: eArgTypeThreadID
```

```{eval-rst}
.. py:data:: eArgTypeThreadIndex
```

```{eval-rst}
.. py:data:: eArgTypeThreadName
```

```{eval-rst}
.. py:data:: eArgTypeTypeName
```

```{eval-rst}
.. py:data:: eArgTypeUnsignedInteger
```

```{eval-rst}
.. py:data:: eArgTypeUnixSignal
```

```{eval-rst}
.. py:data:: eArgTypeVarName
```

```{eval-rst}
.. py:data:: eArgTypeValue
```

```{eval-rst}
.. py:data:: eArgTypeWidth
```

```{eval-rst}
.. py:data:: eArgTypeNone
```

```{eval-rst}
.. py:data:: eArgTypePlatform
```

```{eval-rst}
.. py:data:: eArgTypeWatchpointID
```

```{eval-rst}
.. py:data:: eArgTypeWatchpointIDRange
```

```{eval-rst}
.. py:data:: eArgTypeWatchType
```

```{eval-rst}
.. py:data:: eArgRawInput
```

```{eval-rst}
.. py:data:: eArgTypeCommand
```

```{eval-rst}
.. py:data:: eArgTypeColumnNum
```

```{eval-rst}
.. py:data:: eArgTypeModuleUUID
```

```{eval-rst}
.. py:data:: eArgTypeLastArg
```

```{eval-rst}
.. py:data:: eArgTypeCompletionType
```

(symboltype)=

### SymbolType

```{eval-rst}
.. py:data:: eSymbolTypeAny
```

```{eval-rst}
.. py:data:: eSymbolTypeInvalid
```

```{eval-rst}
.. py:data:: eSymbolTypeAbsolute
```

```{eval-rst}
.. py:data:: eSymbolTypeCode
```

```{eval-rst}
.. py:data:: eSymbolTypeResolver
```

```{eval-rst}
.. py:data:: eSymbolTypeData
```

```{eval-rst}
.. py:data:: eSymbolTypeTrampoline
```

```{eval-rst}
.. py:data:: eSymbolTypeRuntime
```

```{eval-rst}
.. py:data:: eSymbolTypeException
```

```{eval-rst}
.. py:data:: eSymbolTypeSourceFile
```

```{eval-rst}
.. py:data:: eSymbolTypeHeaderFile
```

```{eval-rst}
.. py:data:: eSymbolTypeObjectFile
```

```{eval-rst}
.. py:data:: eSymbolTypeCommonBlock
```

```{eval-rst}
.. py:data:: eSymbolTypeBlock
```

```{eval-rst}
.. py:data:: eSymbolTypeLocal
```

```{eval-rst}
.. py:data:: eSymbolTypeParam
```

```{eval-rst}
.. py:data:: eSymbolTypeVariable
```

```{eval-rst}
.. py:data:: eSymbolTypeVariableType
```

```{eval-rst}
.. py:data:: eSymbolTypeLineEntry
```

```{eval-rst}
.. py:data:: eSymbolTypeLineHeader
```

```{eval-rst}
.. py:data:: eSymbolTypeScopeBegin
```

```{eval-rst}
.. py:data:: eSymbolTypeScopeEnd
```

```{eval-rst}
.. py:data:: eSymbolTypeAdditional
```

```{eval-rst}
.. py:data:: eSymbolTypeCompiler
```

```{eval-rst}
.. py:data:: eSymbolTypeInstrumentation
```

```{eval-rst}
.. py:data:: eSymbolTypeUndefined
```

```{eval-rst}
.. py:data:: eSymbolTypeObjCClass
```

```{eval-rst}
.. py:data:: eSymbolTypeObjCMetaClass
```

```{eval-rst}
.. py:data:: eSymbolTypeObjCIVar
```

```{eval-rst}
.. py:data:: eSymbolTypeReExported

```

(sectiontype)=

### SectionType

```{eval-rst}
.. py:data:: eSectionTypeInvalid
```

```{eval-rst}
.. py:data:: eSectionTypeCode
```

```{eval-rst}
.. py:data:: eSectionTypeContainer
```

```{eval-rst}
.. py:data:: eSectionTypeData
```

```{eval-rst}
.. py:data:: eSectionTypeDataCString
```

```{eval-rst}
.. py:data:: eSectionTypeDataCStringPointers
```

```{eval-rst}
.. py:data:: eSectionTypeDataSymbolAddress
```

```{eval-rst}
.. py:data:: eSectionTypeData4
```

```{eval-rst}
.. py:data:: eSectionTypeData8
```

```{eval-rst}
.. py:data:: eSectionTypeData16
```

```{eval-rst}
.. py:data:: eSectionTypeDataPointers
```

```{eval-rst}
.. py:data:: eSectionTypeDebug
```

```{eval-rst}
.. py:data:: eSectionTypeZeroFill
```

```{eval-rst}
.. py:data:: eSectionTypeDataObjCMessageRefs
```

```{eval-rst}
.. py:data:: eSectionTypeDataObjCCFStrings
```

```{eval-rst}
.. py:data:: eSectionTypeDWARFDebugAbbrev
```

```{eval-rst}
.. py:data:: eSectionTypeDWARFDebugAddr
```

```{eval-rst}
.. py:data:: eSectionTypeDWARFDebugAranges
```

```{eval-rst}
.. py:data:: eSectionTypeDWARFDebugCuIndex
```

```{eval-rst}
.. py:data:: eSectionTypeDWARFDebugFrame
```

```{eval-rst}
.. py:data:: eSectionTypeDWARFDebugInfo
```

```{eval-rst}
.. py:data:: eSectionTypeDWARFDebugLine
```

```{eval-rst}
.. py:data:: eSectionTypeDWARFDebugLoc
```

```{eval-rst}
.. py:data:: eSectionTypeDWARFDebugMacInfo
```

```{eval-rst}
.. py:data:: eSectionTypeDWARFDebugMacro
```

```{eval-rst}
.. py:data:: eSectionTypeDWARFDebugPubNames
```

```{eval-rst}
.. py:data:: eSectionTypeDWARFDebugPubTypes
```

```{eval-rst}
.. py:data:: eSectionTypeDWARFDebugRanges
```

```{eval-rst}
.. py:data:: eSectionTypeDWARFDebugStr
```

```{eval-rst}
.. py:data:: eSectionTypeDWARFDebugStrOffsets
```

```{eval-rst}
.. py:data:: eSectionTypeDWARFAppleNames
```

```{eval-rst}
.. py:data:: eSectionTypeDWARFAppleTypes
```

```{eval-rst}
.. py:data:: eSectionTypeDWARFAppleNamespaces
```

```{eval-rst}
.. py:data:: eSectionTypeDWARFAppleObjC
```

```{eval-rst}
.. py:data:: eSectionTypeELFSymbolTable
```

```{eval-rst}
.. py:data:: eSectionTypeELFDynamicSymbols
```

```{eval-rst}
.. py:data:: eSectionTypeELFRelocationEntries
```

```{eval-rst}
.. py:data:: eSectionTypeELFDynamicLinkInfo
```

```{eval-rst}
.. py:data:: eSectionTypeEHFrame
```

```{eval-rst}
.. py:data:: eSectionTypeARMexidx
```

```{eval-rst}
.. py:data:: eSectionTypeARMextab
```

```{eval-rst}
.. py:data:: eSectionTypeCompactUnwind
```

```{eval-rst}
.. py:data:: eSectionTypeGoSymtab
```

```{eval-rst}
.. py:data:: eSectionTypeAbsoluteAddress
```

```{eval-rst}
.. py:data:: eSectionTypeDWARFGNUDebugAltLink
```

```{eval-rst}
.. py:data:: eSectionTypeDWARFDebugTypes
```

```{eval-rst}
.. py:data:: eSectionTypeDWARFDebugNames
```

```{eval-rst}
.. py:data:: eSectionTypeOther
```

```{eval-rst}
.. py:data:: eSectionTypeDWARFDebugLineStr
```

```{eval-rst}
.. py:data:: eSectionTypeDWARFDebugRngLists
```

```{eval-rst}
.. py:data:: eSectionTypeDWARFDebugLocLists
```

```{eval-rst}
.. py:data:: eSectionTypeDWARFDebugAbbrevDwo
```

```{eval-rst}
.. py:data:: eSectionTypeDWARFDebugInfoDwo
```

```{eval-rst}
.. py:data:: eSectionTypeDWARFDebugStrDwo
```

```{eval-rst}
.. py:data:: eSectionTypeDWARFDebugStrOffsetsDwo
```

```{eval-rst}
.. py:data:: eSectionTypeDWARFDebugTypesDwo
```

```{eval-rst}
.. py:data:: eSectionTypeDWARFDebugRngListsDwo
```

```{eval-rst}
.. py:data:: eSectionTypeDWARFDebugLocDwo
```

```{eval-rst}
.. py:data:: eSectionTypeDWARFDebugLocListsDwo
```

```{eval-rst}
.. py:data:: eSectionTypeDWARFDebugTuIndex

```

(emulatorinstructionoption)=

### EmulatorInstructionOption

```{eval-rst}
.. py:data:: eEmulateInstructionOptionNone
```

```{eval-rst}
.. py:data:: eEmulateInstructionOptionAutoAdvancePC
```

```{eval-rst}
.. py:data:: eEmulateInstructionOptionIgnoreConditions

```

(functionnametype)=

### FunctionNameType

```{eval-rst}
.. py:data:: eFunctionNameTypeNone
```

```{eval-rst}
.. py:data:: eFunctionNameTypeAuto
```

```{eval-rst}
.. py:data:: eFunctionNameTypeFull
```

```{eval-rst}
.. py:data:: eFunctionNameTypeBase
```

```{eval-rst}
.. py:data:: eFunctionNameTypeMethod
```

```{eval-rst}
.. py:data:: eFunctionNameTypeSelector
```

```{eval-rst}
.. py:data:: eFunctionNameTypeAny

```

(basictype)=

### BasicType

```{eval-rst}
.. py:data:: eBasicTypeInvalid
```

```{eval-rst}
.. py:data:: eBasicTypeVoid
```

```{eval-rst}
.. py:data:: eBasicTypeChar
```

```{eval-rst}
.. py:data:: eBasicTypeSignedChar
```

```{eval-rst}
.. py:data:: eBasicTypeUnsignedChar
```

```{eval-rst}
.. py:data:: eBasicTypeWChar
```

```{eval-rst}
.. py:data:: eBasicTypeSignedWChar
```

```{eval-rst}
.. py:data:: eBasicTypeUnsignedWChar
```

```{eval-rst}
.. py:data:: eBasicTypeChar16
```

```{eval-rst}
.. py:data:: eBasicTypeChar32
```

```{eval-rst}
.. py:data:: eBasicTypeChar8
```

```{eval-rst}
.. py:data:: eBasicTypeShort
```

```{eval-rst}
.. py:data:: eBasicTypeUnsignedShort
```

```{eval-rst}
.. py:data:: eBasicTypeInt
```

```{eval-rst}
.. py:data:: eBasicTypeUnsignedInt
```

```{eval-rst}
.. py:data:: eBasicTypeLong
```

```{eval-rst}
.. py:data:: eBasicTypeUnsignedLong
```

```{eval-rst}
.. py:data:: eBasicTypeLongLong
```

```{eval-rst}
.. py:data:: eBasicTypeUnsignedLongLong
```

```{eval-rst}
.. py:data:: eBasicTypeInt128
```

```{eval-rst}
.. py:data:: eBasicTypeUnsignedInt128
```

```{eval-rst}
.. py:data:: eBasicTypeBool
```

```{eval-rst}
.. py:data:: eBasicTypeHalf
```

```{eval-rst}
.. py:data:: eBasicTypeFloat
```

```{eval-rst}
.. py:data:: eBasicTypeDouble
```

```{eval-rst}
.. py:data:: eBasicTypeLongDouble
```

```{eval-rst}
.. py:data:: eBasicTypeFloatComplex
```

```{eval-rst}
.. py:data:: eBasicTypeDoubleComplex
```

```{eval-rst}
.. py:data:: eBasicTypeLongDoubleComplex
```

```{eval-rst}
.. py:data:: eBasicTypeObjCID
```

```{eval-rst}
.. py:data:: eBasicTypeObjCClass
```

```{eval-rst}
.. py:data:: eBasicTypeObjCSel
```

```{eval-rst}
.. py:data:: eBasicTypeNullPtr
```

```{eval-rst}
.. py:data:: eBasicTypeOther
```

```{eval-rst}
.. py:data:: eBasicTypeFloat128

```

(tracetype)=

### TraceType

```{eval-rst}
.. py:data:: eTraceTypeNone
```

```{eval-rst}
.. py:data:: eTraceTypeProcessorTrace

```

(structureddatatype)=

### StructuredDataType

```{eval-rst}
.. py:data:: eStructuredDataTypeInvalid
```

```{eval-rst}
.. py:data:: eStructuredDataTypeNull
```

```{eval-rst}
.. py:data:: eStructuredDataTypeGeneric
```

```{eval-rst}
.. py:data:: eStructuredDataTypeArray
```

```{eval-rst}
.. py:data:: eStructuredDataTypeInteger
```

```{eval-rst}
.. py:data:: eStructuredDataTypeFloat
```

```{eval-rst}
.. py:data:: eStructuredDataTypeBoolean
```

```{eval-rst}
.. py:data:: eStructuredDataTypeString
```

```{eval-rst}
.. py:data:: eStructuredDataTypeDictionary

```

(typeclass)=

### TypeClass

```{eval-rst}
.. py:data:: eTypeClassInvalid
```

```{eval-rst}
.. py:data:: eTypeClassArray
```

```{eval-rst}
.. py:data:: eTypeClassBlockPointer
```

```{eval-rst}
.. py:data:: eTypeClassBuiltin
```

```{eval-rst}
.. py:data:: eTypeClassClass
```

```{eval-rst}
.. py:data:: eTypeClassFloat
```

```{eval-rst}
.. py:data:: eTypeClassComplexInteger
```

```{eval-rst}
.. py:data:: eTypeClassComplexFloat
```

```{eval-rst}
.. py:data:: eTypeClassFunction
```

```{eval-rst}
.. py:data:: eTypeClassMemberPointer
```

```{eval-rst}
.. py:data:: eTypeClassObjCObject
```

```{eval-rst}
.. py:data:: eTypeClassObjCInterface
```

```{eval-rst}
.. py:data:: eTypeClassObjCObjectPointer
```

```{eval-rst}
.. py:data:: eTypeClassPointer
```

```{eval-rst}
.. py:data:: eTypeClassReference
```

```{eval-rst}
.. py:data:: eTypeClassStruct
```

```{eval-rst}
.. py:data:: eTypeClassTypedef
```

```{eval-rst}
.. py:data:: eTypeClassUnion
```

```{eval-rst}
.. py:data:: eTypeClassVector
```

```{eval-rst}
.. py:data:: eTypeClassOther
```

```{eval-rst}
.. py:data:: eTypeClassAny

```

(templateargument)=

### TemplateArgument

```{eval-rst}
.. py:data:: eTemplateArgumentKindNull
```

```{eval-rst}
.. py:data:: eTemplateArgumentKindType
```

```{eval-rst}
.. py:data:: eTemplateArgumentKindDeclaration
```

```{eval-rst}
.. py:data:: eTemplateArgumentKindIntegral
```

```{eval-rst}
.. py:data:: eTemplateArgumentKindTemplate
```

```{eval-rst}
.. py:data:: eTemplateArgumentKindTemplateExpansion
```

```{eval-rst}
.. py:data:: eTemplateArgumentKindExpression
```

```{eval-rst}
.. py:data:: eTemplateArgumentKindPack
```

```{eval-rst}
.. py:data:: eTemplateArgumentKindNullPtr
```

```{eval-rst}
.. py:data:: eTemplateArgumentKindUncommonValue

```

(typeoption)=

### TypeOption

Options that can be set for a formatter to alter its behavior. Not
all of these are applicable to all formatter types.

```{eval-rst}
.. py:data:: eTypeOptionNone
```

```{eval-rst}
.. py:data:: eTypeOptionCascade
```

```{eval-rst}
.. py:data:: eTypeOptionSkipPointers
```

```{eval-rst}
.. py:data:: eTypeOptionSkipReferences
```

```{eval-rst}
.. py:data:: eTypeOptionHideChildren
```

```{eval-rst}
.. py:data:: eTypeOptionHideValue
```

```{eval-rst}
.. py:data:: eTypeOptionShowOneLiner
```

```{eval-rst}
.. py:data:: eTypeOptionHideNames
```

```{eval-rst}
.. py:data:: eTypeOptionNonCacheable
```

```{eval-rst}
.. py:data:: eTypeOptionHideEmptyAggregates
```

```{eval-rst}
.. py:data:: eTypeOptionFrontEndWantsDereference


```

(framecompare)=

### FrameCompare

This is the return value for frame comparisons. If you are comparing frame
A to frame B the following cases arise:

> 1. When frame A pushes frame B (or a frame that ends up pushing
>    B) A is Older than B.
> 2. When frame A pushed frame B (or if frameA is on the stack
>    but B is not) A is Younger than B.
> 3. When frame A and frame B have the same StackID, they are
>    Equal.
> 4. When frame A and frame B have the same immediate parent
>    frame, but are not equal, the comparison yields SameParent.
> 5. If the two frames are on different threads or processes the
>    comparison is Invalid.
> 6. If for some reason we can't figure out what went on, we
>    return Unknown.

```{eval-rst}
.. py:data:: eFrameCompareInvalid
```

```{eval-rst}
.. py:data:: eFrameCompareUnknown
```

```{eval-rst}
.. py:data:: eFrameCompareEqual
```

```{eval-rst}
.. py:data:: eFrameCompareSameParent
```

```{eval-rst}
.. py:data:: eFrameCompareYounger
```

```{eval-rst}
.. py:data:: eFrameCompareOlder

```

(filepermissions)=

### FilePermissions

```{eval-rst}
.. py:data:: eFilePermissionsUserRead
```

```{eval-rst}
.. py:data:: eFilePermissionsUserWrite
```

```{eval-rst}
.. py:data:: eFilePermissionsUserExecute
```

```{eval-rst}
.. py:data:: eFilePermissionsGroupRead
```

```{eval-rst}
.. py:data:: eFilePermissionsGroupWrite
```

```{eval-rst}
.. py:data:: eFilePermissionsGroupExecute
```

```{eval-rst}
.. py:data:: eFilePermissionsWorldRead
```

```{eval-rst}
.. py:data:: eFilePermissionsWorldWrite
```

```{eval-rst}
.. py:data:: eFilePermissionsWorldExecute
```

```{eval-rst}
.. py:data:: eFilePermissionsUserRW
```

```{eval-rst}
.. py:data:: eFileFilePermissionsUserRX
```

```{eval-rst}
.. py:data:: eFilePermissionsUserRWX
```

```{eval-rst}
.. py:data:: eFilePermissionsGroupRW
```

```{eval-rst}
.. py:data:: eFilePermissionsGroupRX
```

```{eval-rst}
.. py:data:: eFilePermissionsGroupRWX
```

```{eval-rst}
.. py:data:: eFilePermissionsWorldRW
```

```{eval-rst}
.. py:data:: eFilePermissionsWorldRX
```

```{eval-rst}
.. py:data:: eFilePermissionsWorldRWX
```

```{eval-rst}
.. py:data:: eFilePermissionsEveryoneR
```

```{eval-rst}
.. py:data:: eFilePermissionsEveryoneW
```

```{eval-rst}
.. py:data:: eFilePermissionsEveryoneX
```

```{eval-rst}
.. py:data:: eFilePermissionsEveryoneRW
```

```{eval-rst}
.. py:data:: eFilePermissionsEveryoneRX
```

```{eval-rst}
.. py:data:: eFilePermissionsEveryoneRWX
```

```{eval-rst}
.. py:data:: eFilePermissionsFileDefault = eFilePermissionsUserRW,
```

```{eval-rst}
.. py:data:: eFilePermissionsDirectoryDefault

```

(queueitem)=

### QueueItem

```{eval-rst}
.. py:data:: eQueueItemKindUnknown
```

```{eval-rst}
.. py:data:: eQueueItemKindFunction
```

```{eval-rst}
.. py:data:: eQueueItemKindBlock

```

(queuekind)=

### QueueKind

libdispatch aka Grand Central Dispatch (GCD) queues can be either
serial (executing on one thread) or concurrent (executing on
multiple threads).

```{eval-rst}
.. py:data:: eQueueKindUnknown
```

```{eval-rst}
.. py:data:: eQueueKindSerial
```

```{eval-rst}
.. py:data:: eQueueKindConcurrent

```

(expressionevaluationphase)=

### ExpressionEvaluationPhase

These are the cancellable stages of expression evaluation, passed
to the expression evaluation callback, so that you can interrupt
expression evaluation at the various points in its lifecycle.

```{eval-rst}
.. py:data:: eExpressionEvaluationParse
```

```{eval-rst}
.. py:data:: eExpressionEvaluationIRGen
```

```{eval-rst}
.. py:data:: eExpressionEvaluationExecution
```

```{eval-rst}
.. py:data:: eExpressionEvaluationComplete

```

(watchpointkind)=

### WatchpointKind

Indicates what types of events cause the watchpoint to fire. Used by Native
-Protocol-related classes.

```{eval-rst}
.. py:data:: eWatchpointKindWrite
```

```{eval-rst}
.. py:data:: eWatchpointKindRead

```

(gdbsignal)=

### GdbSignal

```{eval-rst}
.. py:data:: eGdbSignalBadAccess
```

```{eval-rst}
.. py:data:: eGdbSignalBadInstruction
```

```{eval-rst}
.. py:data:: eGdbSignalArithmetic
```

```{eval-rst}
.. py:data:: eGdbSignalEmulation
```

```{eval-rst}
.. py:data:: eGdbSignalSoftware
```

```{eval-rst}
.. py:data:: eGdbSignalBreakpoint
```

(pathtype)=

### PathType

Used with {any}`SBHostOS.GetLLDBPath` to find files that are
related to LLDB on the current host machine. Most files are
relative to LLDB or are in known locations.

```{eval-rst}
.. py:data:: ePathTypeLLDBShlibDir

   The directory where the lldb.so (unix) or LLDB mach-o file in
   LLDB.framework (MacOSX) exists.
```

```{eval-rst}
.. py:data:: ePathTypeSupportExecutableDir

   Find LLDB support executable directory (debugserver, etc).
```

```{eval-rst}
.. py:data:: ePathTypeHeaderDir

   Find LLDB header file directory.
```

```{eval-rst}
.. py:data:: ePathTypePythonDir

   Find Python modules (PYTHONPATH) directory.
```

```{eval-rst}
.. py:data:: ePathTypeLLDBSystemPlugins

   System plug-ins directory
```

```{eval-rst}
.. py:data:: ePathTypeLLDBUserPlugins

   User plug-ins directory
```

```{eval-rst}
.. py:data:: ePathTypeLLDBTempSystemDir

   The LLDB temp directory for this system that will be cleaned up on exit.
```

```{eval-rst}
.. py:data:: ePathTypeGlobalLLDBTempSystemDir

   The LLDB temp directory for this system, NOT cleaned up on a process
   exit.
```

```{eval-rst}
.. py:data:: ePathTypeClangDir

   Find path to Clang builtin headers.

```

(memberfunctionkind)=

### MemberFunctionKind

```{eval-rst}
.. py:data:: eMemberFunctionKindUnknown
```

```{eval-rst}
.. py:data:: eMemberFunctionKindConstructor

   A function used to create instances.
```

```{eval-rst}
.. py:data:: eMemberFunctionKindDestructor

   A function used to tear down existing instances.
```

```{eval-rst}
.. py:data:: eMemberFunctionKindInstanceMethod

   A function that applies to a specific instance.
```

```{eval-rst}
.. py:data:: eMemberFunctionKindStaticMethod

   A function that applies to a type rather than any instance,

```

(typeflags)=

### TypeFlags

```{eval-rst}
.. py:data:: eTypeHasChildren
```

```{eval-rst}
.. py:data:: eTypeIsArray
```

```{eval-rst}
.. py:data:: eTypeIsBuiltIn
```

```{eval-rst}
.. py:data:: eTypeIsCPlusPlus
```

```{eval-rst}
.. py:data:: eTypeIsFuncPrototype
```

```{eval-rst}
.. py:data:: eTypeIsObjC
```

```{eval-rst}
.. py:data:: eTypeIsReference
```

```{eval-rst}
.. py:data:: eTypeIsTemplate
```

```{eval-rst}
.. py:data:: eTypeIsVector
```

```{eval-rst}
.. py:data:: eTypeIsInteger
```

```{eval-rst}
.. py:data:: eTypeIsComplex
```

```{eval-rst}
.. py:data:: eTypeInstanceIsPointer

```

(commandflags)=

### CommandFlags

```{eval-rst}
.. py:data:: eCommandRequiresTarget
```

```{eval-rst}
.. py:data:: eCommandRequiresProcess
```

```{eval-rst}
.. py:data:: eCommandRequiresThread
```

```{eval-rst}
.. py:data:: eCommandRequiresFrame
```

```{eval-rst}
.. py:data:: eCommandRequiresRegContext
```

```{eval-rst}
.. py:data:: eCommandTryTargetAPILock
```

```{eval-rst}
.. py:data:: eCommandProcessMustBeLaunched
```

```{eval-rst}
.. py:data:: eCommandProcessMustBePaused
```

```{eval-rst}
.. py:data:: eCommandProcessMustBeTraced

```

(typesummary)=

### TypeSummary

Whether a summary should cap how much data it returns to users or not.

```{eval-rst}
.. py:data:: eTypeSummaryCapped
```

```{eval-rst}
.. py:data:: eTypeSummaryUncapped

```

(commandinterpreterresult)=

### CommandInterpreterResult

The result from a command interpreter run.

```{eval-rst}
.. py:data:: eCommandInterpreterResultSuccess

   Command interpreter finished successfully.
```

```{eval-rst}
.. py:data:: eCommandInterpreterResultInferiorCrash

   Stopped because the corresponding option was set and the inferior
   crashed.
```

```{eval-rst}
.. py:data:: eCommandInterpreterResultCommandError

   Stopped because the corresponding option was set and a command returned
   an error.
```

```{eval-rst}
.. py:data:: eCommandInterpreterResultQuitRequested

   Stopped because quit was requested.

```

(watchpointvaluekind)=

### WatchPointValueKind

The type of value that the watchpoint was created to monitor.

```{eval-rst}
.. py:data:: eWatchPointValueKindInvalid

   Invalid kind.
```

```{eval-rst}
.. py:data:: eWatchPointValueKindVariable

   Watchpoint was created watching a variable
```

```{eval-rst}
.. py:data:: eWatchPointValueKindExpression

   Watchpoint was created watching the result of an expression that was
   evaluated at creation time.
```
