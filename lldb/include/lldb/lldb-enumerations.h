//===-- lldb-enumerations.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_LLDB_ENUMERATIONS_H
#define LLDB_LLDB_ENUMERATIONS_H

#include <cstdint>
#include <type_traits>

#ifndef SWIG
// Macro to enable bitmask operations on an enum.  Without this, Enum | Enum
// gets promoted to an int, so you have to say Enum a = Enum(eFoo | eBar).  If
// you mark Enum with LLDB_MARK_AS_BITMASK_ENUM(Enum), however, you can simply
// write Enum a = eFoo | eBar.
// Unfortunately, swig<3.0 doesn't recognise the constexpr keyword, so remove
// this entire block, as it is not necessary for swig processing.
#define LLDB_MARK_AS_BITMASK_ENUM(Enum)                                        \
  constexpr Enum operator|(Enum a, Enum b) {                                   \
    return static_cast<Enum>(                                                  \
        static_cast<std::underlying_type<Enum>::type>(a) |                     \
        static_cast<std::underlying_type<Enum>::type>(b));                     \
  }                                                                            \
  constexpr Enum operator&(Enum a, Enum b) {                                   \
    return static_cast<Enum>(                                                  \
        static_cast<std::underlying_type<Enum>::type>(a) &                     \
        static_cast<std::underlying_type<Enum>::type>(b));                     \
  }                                                                            \
  constexpr Enum operator~(Enum a) {                                           \
    return static_cast<Enum>(                                                  \
        ~static_cast<std::underlying_type<Enum>::type>(a));                    \
  }                                                                            \
  inline Enum &operator|=(Enum &a, Enum b) {                                   \
    a = a | b;                                                                 \
    return a;                                                                  \
  }                                                                            \
  inline Enum &operator&=(Enum &a, Enum b) {                                   \
    a = a & b;                                                                 \
    return a;                                                                  \
  }
#else
#define LLDB_MARK_AS_BITMASK_ENUM(Enum)
#endif

#ifndef SWIG
// With MSVC, the default type of an enum is always signed, even if one of the
// enumerator values is too large to fit into a signed integer but would
// otherwise fit into an unsigned integer.  As a result of this, all of LLDB's
// flag-style enumerations that specify something like eValueFoo = 1u << 31
// result in negative values.  This usually just results in a benign warning,
// but in a few places we actually do comparisons on the enum values, which
// would cause a real bug.  Furthermore, there's no way to silence only this
// warning, as it's part of -Wmicrosoft which also catches a whole slew of
// other useful issues.
//
// To make matters worse, early versions of SWIG don't recognize the syntax of
// specifying the underlying type of an enum (and Python doesn't care anyway)
// so we need a way to specify the underlying type when the enum is being used
// from C++ code, but just use a regular enum when swig is pre-processing.
#define FLAGS_ENUM(Name) enum Name : unsigned
#define FLAGS_ANONYMOUS_ENUM() enum : unsigned
#else
#define FLAGS_ENUM(Name) enum Name
#define FLAGS_ANONYMOUS_ENUM() enum
#endif

namespace lldb {

/// Process and Thread States.
enum StateType {
  eStateInvalid = 0,
  /// Process is object is valid, but not currently loaded
  eStateUnloaded,
  /// Process is connected to remote debug services, but not
  /// launched or attached to anything yet
  eStateConnected,
  /// Process is currently trying to attach
  eStateAttaching,
  /// Process is in the process of launching
  eStateLaunching,
  // The state changes eStateAttaching and eStateLaunching are both sent while
  // the private state thread is either not yet started or paused. For that
  // reason, they should only be signaled as public state changes, and not
  // private state changes.
  /// Process or thread is stopped and can be examined.
  eStateStopped,
  /// Process or thread is running and can't be examined.
  eStateRunning,
  /// Process or thread is in the process of stepping and can
  /// not be examined.
  eStateStepping,
  /// Process or thread has crashed and can be examined.
  eStateCrashed,
  /// Process has been detached and can't be examined.
  eStateDetached,
  /// Process has exited and can't be examined.
  eStateExited,
  /// Process or thread is in a suspended state as far
  /// as the debugger is concerned while other processes
  /// or threads get the chance to run.
  eStateSuspended,
  kLastStateType = eStateSuspended
};

/// Launch Flags.
FLAGS_ENUM(LaunchFlags){
    eLaunchFlagNone = 0u,
    /// Exec when launching and turn the calling process into a new process.
    eLaunchFlagExec = (1u << 0),
    /// Stop as soon as the process launches to allow the process to be
    /// debugged.
    eLaunchFlagDebug = (1u << 1),
    /// Stop at the program entry point instead of auto-continuing when
    /// launching or attaching at entry point.
    eLaunchFlagStopAtEntry = (1u << 2),
    /// Disable Address Space Layout Randomization.
    eLaunchFlagDisableASLR = (1u << 3),
    /// Disable stdio for inferior process (e.g. for a GUI app).
    eLaunchFlagDisableSTDIO = (1u << 4),
    /// Launch the process in a new TTY if supported by the host.
    eLaunchFlagLaunchInTTY = (1u << 5),
    /// Launch the process inside a shell to get shell expansion.
    eLaunchFlagLaunchInShell = (1u << 6),
    /// Launch the process in a separate process group. If you are going to
    /// hand the process off (e.g. to debugserver).
    eLaunchFlagLaunchInSeparateProcessGroup = (1u << 7),
    /// Set this flag so lldb & the handee don't race to set its exit status.
    eLaunchFlagDontSetExitStatus = (1u << 8),
    /// If set, then the client stub should detach rather than killing the
    /// debugee if it loses connection with lldb.
    eLaunchFlagDetachOnError = (1u << 9),
    /// Perform shell-style argument expansion.
    eLaunchFlagShellExpandArguments = (1u << 10),
    /// Close the open TTY on exit.
    eLaunchFlagCloseTTYOnExit = (1u << 11),
    /// Don't make the inferior responsible for its own TCC permissions but
    /// instead inherit them from its parent.
    eLaunchFlagInheritTCCFromParent = (1u << 12),
    /// Launch process with memory tagging explicitly enabled.
    eLaunchFlagMemoryTagging = (1u << 13),
    /// Use anonymous pipes for stdio instead of a ConPTY on Windows. Useful
    /// when terminal emulation is not needed (e.g. lldb-dap internalConsole
    /// mode).
    eLaunchFlagUsePipes = (1u << 14),
};

/// Thread Run Modes.
enum RunMode { eOnlyThisThread, eAllThreads, eOnlyDuringStepping };

/// Execution directions
enum RunDirection { eRunForward, eRunReverse };

/// Byte ordering definitions.
enum ByteOrder {
  eByteOrderInvalid = 0,
  eByteOrderBig = 1,
  eByteOrderPDP = 2,
  eByteOrderLittle = 4
};

/// Register encoding definitions.
enum Encoding {
  eEncodingInvalid = 0,
  /// unsigned integer
  eEncodingUint,
  /// signed integer
  eEncodingSint,
  /// float
  eEncodingIEEE754,
  /// vector registers
  eEncodingVector
};

/// Display format definitions.
enum Format {
  eFormatDefault = 0,
  eFormatInvalid = 0,
  eFormatBoolean,
  eFormatBinary,
  eFormatBytes,
  eFormatBytesWithASCII,
  eFormatChar,
  /// Only printable characters, '.' if not printable
  eFormatCharPrintable,
  /// Floating point complex type
  eFormatComplex,
  eFormatComplexFloat = eFormatComplex,
  /// NULL terminated C strings
  eFormatCString,
  eFormatDecimal,
  eFormatEnum,
  eFormatHex,
  eFormatHexUppercase,
  eFormatFloat,
  eFormatOctal,
  /// OS character codes encoded into an integer 'PICT' 'text'
  /// etc...
  eFormatOSType,
  eFormatUnicode16,
  eFormatUnicode32,
  eFormatUnsigned,
  eFormatPointer,
  eFormatVectorOfChar,
  eFormatVectorOfSInt8,
  eFormatVectorOfUInt8,
  eFormatVectorOfSInt16,
  eFormatVectorOfUInt16,
  eFormatVectorOfSInt32,
  eFormatVectorOfUInt32,
  eFormatVectorOfSInt64,
  eFormatVectorOfUInt64,
  eFormatVectorOfFloat16,
  eFormatVectorOfFloat32,
  eFormatVectorOfFloat64,
  eFormatVectorOfUInt128,
  /// Integer complex type
  eFormatComplexInteger,
  /// Print characters with no single quotes, used for
  /// character arrays that can contain non printable
  /// characters
  eFormatCharArray,
  /// Describe what an address points to (func + offset
  /// with file/line, symbol + offset, data, etc)
  eFormatAddressInfo,
  /// ISO C99 hex float string
  eFormatHexFloat,
  /// Disassemble an opcode
  eFormatInstruction,
  /// Do not print this
  eFormatVoid,
  eFormatUnicode8,
  /// Disambiguate between 128-bit `long double` (which uses
  /// `eFormatFloat`) and `__float128` (which uses
  /// `eFormatFloat128`). If the value being formatted is not
  /// 128 bits, then this is identical to `eFormatFloat`.
  eFormatFloat128,
  kNumFormats
};

/// Description levels for "void GetDescription(Stream *, DescriptionLevel)"
/// calls.
enum DescriptionLevel {
  eDescriptionLevelBrief = 0,
  eDescriptionLevelFull,
  eDescriptionLevelVerbose,
  eDescriptionLevelInitial,
  kNumDescriptionLevels
};

/// Script interpreter types.
enum ScriptLanguage {
  eScriptLanguageNone = 0,
  eScriptLanguagePython,
  eScriptLanguageLua,
  eScriptLanguageUnknown,
  eScriptLanguageDefault = eScriptLanguagePython
};

/// Register numbering types.
// See RegisterContext::ConvertRegisterKindToRegisterNumber to convert any of
// these to the lldb internal register numbering scheme (eRegisterKindLLDB).
enum RegisterKind {
  /// the register numbers seen in eh_frame
  eRegisterKindEHFrame = 0,
  /// the register numbers seen DWARF
  eRegisterKindDWARF,
  /// insn ptr reg, stack ptr reg, etc not specific to
  /// any particular target
  eRegisterKindGeneric,
  /// num used by the process plugin - e.g. by the
  /// remote gdb-protocol stub program
  eRegisterKindProcessPlugin,
  /// lldb's internal register numbers
  eRegisterKindLLDB,
  kNumRegisterKinds
};

/// Thread stop reasons.
enum StopReason {
  eStopReasonInvalid = 0,
  eStopReasonNone,
  eStopReasonTrace,
  eStopReasonBreakpoint,
  eStopReasonWatchpoint,
  eStopReasonSignal,
  eStopReasonException,
  /// Program was re-exec'ed
  eStopReasonExec,
  eStopReasonPlanComplete,
  eStopReasonThreadExiting,
  eStopReasonInstrumentation,
  eStopReasonProcessorTrace,
  eStopReasonFork,
  eStopReasonVFork,
  eStopReasonVForkDone,
  /// Thread requested interrupt
  eStopReasonInterrupt,
  // Indicates that execution stopped because the debugger backend relies
  // on recorded data and we reached the end of that data.
  eStopReasonHistoryBoundary,
};

/// Command Return Status Types.
enum ReturnStatus {
  eReturnStatusInvalid,
  eReturnStatusSuccessFinishNoResult,
  eReturnStatusSuccessFinishResult,
  eReturnStatusSuccessContinuingNoResult,
  eReturnStatusSuccessContinuingResult,
  eReturnStatusStarted,
  eReturnStatusFailed,
  eReturnStatusQuit
};

/// The results of expression evaluation.
enum ExpressionResults {
  eExpressionCompleted = 0,
  eExpressionSetupError,
  eExpressionParseError,
  eExpressionDiscarded,
  eExpressionInterrupted,
  eExpressionHitBreakpoint,
  eExpressionTimedOut,
  eExpressionResultUnavailable,
  eExpressionStoppedForDebug,
  eExpressionThreadVanished
};

enum SearchDepth {
  eSearchDepthInvalid = 0,
  eSearchDepthTarget,
  eSearchDepthModule,
  eSearchDepthCompUnit,
  eSearchDepthFunction,
  eSearchDepthBlock,
  eSearchDepthAddress,
  kLastSearchDepthKind = eSearchDepthAddress
};

/// Connection Status Types.
enum ConnectionStatus {
  /// Success
  eConnectionStatusSuccess,
  /// End-of-file encountered
  eConnectionStatusEndOfFile,
  /// Check GetError() for details
  eConnectionStatusError,
  /// Request timed out
  eConnectionStatusTimedOut,
  /// No connection
  eConnectionStatusNoConnection,
  /// Lost connection while connected to a
  /// valid connection
  eConnectionStatusLostConnection,
  /// Interrupted read
  eConnectionStatusInterrupted
};

enum ErrorType {
  eErrorTypeInvalid,
  /// Generic errors that can be any value.
  eErrorTypeGeneric,
  /// Mach kernel error codes.
  eErrorTypeMachKernel,
  /// POSIX error codes.
  eErrorTypePOSIX,
  /// These are from the ExpressionResults enum.
  eErrorTypeExpression,
  /// Standard Win32 error codes.
  eErrorTypeWin32
};

enum ValueType : uint32_t {
  eValueTypeInvalid = 0,
  /// globals variable
  eValueTypeVariableGlobal = 1,
  /// static variable
  eValueTypeVariableStatic = 2,
  /// function argument variables
  eValueTypeVariableArgument = 3,
  /// function local variables
  eValueTypeVariableLocal = 4,
  /// stack frame register value
  eValueTypeRegister = 5,
  /// A collection of stack frame register values
  eValueTypeRegisterSet = 6,
  /// constant result variables
  eValueTypeConstResult = 7,
  /// thread local storage variable
  eValueTypeVariableThreadLocal = 8,
  /// virtual function table
  eValueTypeVTable = 9,
  /// function pointer in virtual function table
  eValueTypeVTableEntry = 10,
};

/// A mask that we can use to check if the value type is synthetic or not.
// NOTE: This limits the number of value types to 31, but that's 3x more than
// what we currently have now. See lldb/Utility/ValueType.h for helpers for
// working with synthetic value types.
static constexpr unsigned ValueTypeSyntheticMask = 0x20;

/// Token size/granularities for Input Readers.

enum InputReaderGranularity {
  eInputReaderGranularityInvalid = 0,
  eInputReaderGranularityByte,
  eInputReaderGranularityWord,
  eInputReaderGranularityLine,
  eInputReaderGranularityAll
};

/// These mask bits allow a common interface for queries that can
/// limit the amount of information that gets parsed to only the
/// information that is requested. These bits also can indicate what
/// actually did get resolved during query function calls.
///
/// Each definition corresponds to a one of the member variables
/// in this class, and requests that that item be resolved, or
/// indicates that the member did get resolved.
FLAGS_ENUM(SymbolContextItem){
    /// Set when \a target is requested from a query, or was located
    /// in query results
    eSymbolContextTarget = (1u << 0),
    /// Set when \a module is requested from a query, or was located
    /// in query results
    eSymbolContextModule = (1u << 1),
    /// Set when \a comp_unit is requested from a query, or was
    /// located in query results
    eSymbolContextCompUnit = (1u << 2),
    /// Set when \a function is requested from a query, or was located
    /// in query results
    eSymbolContextFunction = (1u << 3),
    /// Set when the deepest \a block is requested from a query, or
    /// was located in query results
    eSymbolContextBlock = (1u << 4),
    /// Set when \a line_entry is requested from a query, or was
    /// located in query results
    eSymbolContextLineEntry = (1u << 5),
    /// Set when \a symbol is requested from a query, or was located
    /// in query results
    eSymbolContextSymbol = (1u << 6),
    /// Indicates to try and lookup everything up during a routine
    /// symbol context query.
    eSymbolContextEverything = ((eSymbolContextSymbol << 1) - 1u),
    /// Set when \a global or static variable is requested from a
    /// query, or was located in query results.
    /// eSymbolContextVariable is potentially expensive to lookup so
    /// it isn't included in eSymbolContextEverything which stops it
    /// from being used during frame PC lookups and many other
    /// potential address to symbol context lookups.
    eSymbolContextVariable = (1u << 7),

    // Keep this last and up-to-date for what the last enum value is.
    eSymbolContextLastItem = eSymbolContextVariable,
};
LLDB_MARK_AS_BITMASK_ENUM(SymbolContextItem)

FLAGS_ENUM(Permissions){ePermissionsWritable = (1u << 0),
                        ePermissionsReadable = (1u << 1),
                        ePermissionsExecutable = (1u << 2)};
LLDB_MARK_AS_BITMASK_ENUM(Permissions)

enum InputReaderAction {
  /// reader is newly pushed onto the reader stack
  eInputReaderActivate,
  /// an async output event occurred;
  /// the reader may want to do
  /// something
  eInputReaderAsynchronousOutputWritten,
  /// reader is on top of the stack again after another
  /// reader was popped off
  eInputReaderReactivate,
  /// another reader was pushed on the stack
  eInputReaderDeactivate,
  /// reader got one of its tokens (granularity)
  eInputReaderGotToken,
  /// reader received an interrupt signal (probably from
  /// a control-c)
  eInputReaderInterrupt,
  /// reader received an EOF char (probably from a
  /// control-d)
  eInputReaderEndOfFile,
  /// reader was just popped off the stack and is done
  eInputReaderDone
};

FLAGS_ENUM(BreakpointEventType){
    eBreakpointEventTypeInvalidType = (1u << 0),
    eBreakpointEventTypeAdded = (1u << 1),
    eBreakpointEventTypeRemoved = (1u << 2),
    /// Locations added doesn't
    /// get sent when the
    /// breakpoint is created
    eBreakpointEventTypeLocationsAdded = (1u << 3),
    eBreakpointEventTypeLocationsRemoved = (1u << 4),
    eBreakpointEventTypeLocationsResolved = (1u << 5),
    eBreakpointEventTypeEnabled = (1u << 6),
    eBreakpointEventTypeDisabled = (1u << 7),
    eBreakpointEventTypeCommandChanged = (1u << 8),
    eBreakpointEventTypeConditionChanged = (1u << 9),
    eBreakpointEventTypeIgnoreChanged = (1u << 10),
    eBreakpointEventTypeThreadChanged = (1u << 11),
    eBreakpointEventTypeAutoContinueChanged = (1u << 12)};

FLAGS_ENUM(WatchpointEventType){
    eWatchpointEventTypeInvalidType = (1u << 0),
    eWatchpointEventTypeAdded = (1u << 1),
    eWatchpointEventTypeRemoved = (1u << 2),
    eWatchpointEventTypeEnabled = (1u << 6),
    eWatchpointEventTypeDisabled = (1u << 7),
    eWatchpointEventTypeCommandChanged = (1u << 8),
    eWatchpointEventTypeConditionChanged = (1u << 9),
    eWatchpointEventTypeIgnoreChanged = (1u << 10),
    eWatchpointEventTypeThreadChanged = (1u << 11),
    eWatchpointEventTypeTypeChanged = (1u << 12)};

enum WatchpointWriteType {
  /// Don't stop when the watched memory region is written to.
  eWatchpointWriteTypeDisabled,
  /// Stop on any write access to the memory region, even if
  /// the value doesn't change.  On some architectures, a write
  /// near the memory region may be falsely reported as a match,
  /// and notify this spurious stop as a watchpoint trap.
  eWatchpointWriteTypeAlways,
  /// Stop on a write to the memory region that changes its value.
  /// This is most likely the behavior a user expects, and is the
  /// behavior in gdb.  lldb can silently ignore writes near the
  /// watched memory region that are reported as accesses to lldb.
  eWatchpointWriteTypeOnModify
};

/// Programming language type.
///
/// These enumerations use the same language enumerations as the DWARF
/// specification for ease of use and consistency.
/// The enum -> string code is in Language.cpp, don't change this
/// table without updating that code as well.
///
/// This datatype is used in SBExpressionOptions::SetLanguage() which
/// makes this type API. Do not change its underlying storage type!
enum LanguageType {
  /// Unknown or invalid language value.
  eLanguageTypeUnknown = 0x0000,
  /// ISO C:1989.
  eLanguageTypeC89 = 0x0001,
  /// Non-standardized C, such as K&R.
  eLanguageTypeC = 0x0002,
  /// ISO Ada:1983.
  eLanguageTypeAda83 = 0x0003,
  /// ISO C++:1998.
  eLanguageTypeC_plus_plus = 0x0004,
  /// ISO Cobol:1974.
  eLanguageTypeCobol74 = 0x0005,
  /// ISO Cobol:1985.
  eLanguageTypeCobol85 = 0x0006,
  /// ISO Fortran 77.
  eLanguageTypeFortran77 = 0x0007,
  /// ISO Fortran 90.
  eLanguageTypeFortran90 = 0x0008,
  /// ISO Pascal:1983.
  eLanguageTypePascal83 = 0x0009,
  /// ISO Modula-2:1996.
  eLanguageTypeModula2 = 0x000a,
  /// Java.
  eLanguageTypeJava = 0x000b,
  /// ISO C:1999.
  eLanguageTypeC99 = 0x000c,
  /// ISO Ada:1995.
  eLanguageTypeAda95 = 0x000d,
  /// ISO Fortran 95.
  eLanguageTypeFortran95 = 0x000e,
  /// ANSI PL/I:1976.
  eLanguageTypePLI = 0x000f,
  /// Objective-C.
  eLanguageTypeObjC = 0x0010,
  /// Objective-C++.
  eLanguageTypeObjC_plus_plus = 0x0011,
  /// Unified Parallel C.
  eLanguageTypeUPC = 0x0012,
  /// D.
  eLanguageTypeD = 0x0013,
  /// Python.
  eLanguageTypePython = 0x0014,
  // NOTE: The below are DWARF5 constants, subject to change upon
  // completion of the DWARF5 specification
  /// OpenCL.
  eLanguageTypeOpenCL = 0x0015,
  /// Go.
  eLanguageTypeGo = 0x0016,
  /// Modula 3.
  eLanguageTypeModula3 = 0x0017,
  /// Haskell.
  eLanguageTypeHaskell = 0x0018,
  /// ISO C++:2003.
  eLanguageTypeC_plus_plus_03 = 0x0019,
  /// ISO C++:2011.
  eLanguageTypeC_plus_plus_11 = 0x001a,
  /// OCaml.
  eLanguageTypeOCaml = 0x001b,
  /// Rust.
  eLanguageTypeRust = 0x001c,
  /// ISO C:2011.
  eLanguageTypeC11 = 0x001d,
  /// Swift.
  eLanguageTypeSwift = 0x001e,
  /// Julia.
  eLanguageTypeJulia = 0x001f,
  /// Dylan.
  eLanguageTypeDylan = 0x0020,
  /// ISO C++:2014.
  eLanguageTypeC_plus_plus_14 = 0x0021,
  /// ISO Fortran 2003.
  eLanguageTypeFortran03 = 0x0022,
  /// ISO Fortran 2008.
  eLanguageTypeFortran08 = 0x0023,
  eLanguageTypeRenderScript = 0x0024,
  eLanguageTypeBLISS = 0x0025,
  eLanguageTypeKotlin = 0x0026,
  eLanguageTypeZig = 0x0027,
  eLanguageTypeCrystal = 0x0028,
  /// ISO C++:2017.
  eLanguageTypeC_plus_plus_17 = 0x002a,
  /// ISO C++:2020.
  eLanguageTypeC_plus_plus_20 = 0x002b,
  eLanguageTypeC17 = 0x002c,
  eLanguageTypeFortran18 = 0x002d,
  eLanguageTypeAda2005 = 0x002e,
  eLanguageTypeAda2012 = 0x002f,
  eLanguageTypeHIP = 0x0030,
  eLanguageTypeAssembly = 0x0031,
  eLanguageTypeC_sharp = 0x0032,
  eLanguageTypeMojo = 0x0033,
  eLanguageTypeLastStandardLanguage = eLanguageTypeMojo,

  // Vendor Extensions
  // Note: Language::GetNameForLanguageType
  // assumes these can be used as indexes into array language_names, and
  // Language::SetLanguageFromCString and Language::AsCString assume these can
  // be used as indexes into array g_languages.
  /// Mips_Assembler.
  eLanguageTypeMipsAssembler,
  eNumLanguageTypes
};

enum InstrumentationRuntimeType {
  eInstrumentationRuntimeTypeAddressSanitizer = 0x0000,
  eInstrumentationRuntimeTypeThreadSanitizer = 0x0001,
  eInstrumentationRuntimeTypeUndefinedBehaviorSanitizer = 0x0002,
  eInstrumentationRuntimeTypeMainThreadChecker = 0x0003,
  eInstrumentationRuntimeTypeSwiftRuntimeReporting = 0x0004,
  eInstrumentationRuntimeTypeLibsanitizersAsan = 0x0005,
  eInstrumentationRuntimeTypeBoundsSafety = 0x0006,
  eNumInstrumentationRuntimeTypes
};

enum PluginDomainKind {
  ePluginDomainKindGlobal = 0x1,
  ePluginDomainKindDebugger = 0x2,
  ePluginDomainKindTarget = 0x4,
};

enum DynamicValueType {
  eNoDynamicValues = 0,
  eDynamicCanRunTarget = 1,
  eDynamicDontRunTarget = 2
};

enum StopShowColumn {
  eStopShowColumnAnsiOrCaret = 0,
  eStopShowColumnAnsi = 1,
  eStopShowColumnCaret = 2,
  eStopShowColumnNone = 3
};

enum AccessType {
  eAccessNone,
  eAccessPublic,
  eAccessPrivate,
  eAccessProtected,
  eAccessPackage
};

enum CommandArgumentType {
  eArgTypeAddress = 0,
  eArgTypeAddressOrExpression,
  eArgTypeAliasName,
  eArgTypeAliasOptions,
  eArgTypeArchitecture,
  eArgTypeBoolean,
  eArgTypeBreakpointID,
  eArgTypeBreakpointIDRange,
  eArgTypeBreakpointName,
  eArgTypeByteSize,
  eArgTypeClassName,
  eArgTypeCommandName,
  eArgTypeCount,
  eArgTypeDescriptionVerbosity,
  eArgTypeDirectoryName,
  eArgTypeDisassemblyFlavor,
  eArgTypeEndAddress,
  eArgTypeExpression,
  eArgTypeExpressionPath,
  eArgTypeExprFormat,
  eArgTypeFileLineColumn,
  eArgTypeFilename,
  eArgTypeFormat,
  eArgTypeFrameIndex,
  eArgTypeFrameProviderIDRange,
  eArgTypeFullName,
  eArgTypeFunctionName,
  eArgTypeFunctionOrSymbol,
  eArgTypeGDBFormat,
  eArgTypeHelpText,
  eArgTypeIndex,
  eArgTypeLanguage,
  eArgTypeLineNum,
  eArgTypeLogCategory,
  eArgTypeLogChannel,
  eArgTypeMethod,
  eArgTypeName,
  eArgTypeNewPathPrefix,
  eArgTypeNumLines,
  eArgTypeNumberPerLine,
  eArgTypeOffset,
  eArgTypeOldPathPrefix,
  eArgTypeOneLiner,
  eArgTypePath,
  eArgTypePermissionsNumber,
  eArgTypePermissionsString,
  eArgTypePid,
  eArgTypePlugin,
  eArgTypeProcessName,
  eArgTypePythonClass,
  eArgTypePythonFunction,
  eArgTypePythonScript,
  eArgTypeQueueName,
  eArgTypeRegisterName,
  eArgTypeRegularExpression,
  eArgTypeRunArgs,
  eArgTypeRunMode,
  eArgTypeScriptedCommandSynchronicity,
  eArgTypeScriptLang,
  eArgTypeSearchWord,
  eArgTypeSelector,
  eArgTypeSettingIndex,
  eArgTypeSettingKey,
  eArgTypeSettingPrefix,
  eArgTypeSettingVariableName,
  eArgTypeShlibName,
  eArgTypeSourceFile,
  eArgTypeSortOrder,
  eArgTypeStartAddress,
  eArgTypeSummaryString,
  eArgTypeSymbol,
  eArgTypeThreadID,
  eArgTypeThreadIndex,
  eArgTypeThreadName,
  eArgTypeTypeName,
  eArgTypeUnsignedInteger,
  eArgTypeUnixSignal,
  eArgTypeVarName,
  eArgTypeValue,
  eArgTypeWidth,
  eArgTypeNone,
  eArgTypePlatform,
  eArgTypeWatchpointID,
  eArgTypeWatchpointIDRange,
  eArgTypeWatchType,
  eArgRawInput,
  eArgTypeCommand,
  eArgTypeColumnNum,
  eArgTypeModuleUUID,
  eArgTypeSaveCoreStyle,
  eArgTypeLogHandler,
  eArgTypeSEDStylePair,
  eArgTypeRecognizerID,
  eArgTypeConnectURL,
  eArgTypeTargetID,
  eArgTypeStopHookID,
  eArgTypeCompletionType,
  eArgTypeRemotePath,
  eArgTypeRemoteFilename,
  eArgTypeModule,
  eArgTypeCPUName,
  eArgTypeCPUFeatures,
  eArgTypeManagedPlugin,
  eArgTypeProtocol,
  eArgTypeExceptionStage,
  eArgTypeNameMatchStyle,
  eArgTypePluginDomain,
  eArgTypeLastArg // Always keep this entry as the last entry in this
                  // enumeration!!
};

/// Symbol types.
// Symbol holds the SymbolType in a 6-bit field (m_type), so if you get over 63
// entries you will have to resize that field.
enum SymbolType {
  eSymbolTypeAny = 0,
  eSymbolTypeInvalid = 0,
  eSymbolTypeAbsolute,
  eSymbolTypeCode,
  eSymbolTypeResolver,
  eSymbolTypeData,
  eSymbolTypeTrampoline,
  eSymbolTypeRuntime,
  eSymbolTypeException,
  eSymbolTypeSourceFile,
  eSymbolTypeHeaderFile,
  eSymbolTypeObjectFile,
  eSymbolTypeCommonBlock,
  eSymbolTypeBlock,
  eSymbolTypeLocal,
  eSymbolTypeParam,
  eSymbolTypeVariable,
  eSymbolTypeVariableType,
  eSymbolTypeLineEntry,
  eSymbolTypeLineHeader,
  eSymbolTypeScopeBegin,
  eSymbolTypeScopeEnd,
  /// When symbols take more than one entry, the extra
  /// entries get this type
  eSymbolTypeAdditional,
  eSymbolTypeCompiler,
  eSymbolTypeInstrumentation,
  eSymbolTypeUndefined,
  eSymbolTypeObjCClass,
  eSymbolTypeObjCMetaClass,
  eSymbolTypeObjCIVar,
  eSymbolTypeReExported
};

enum SectionType {
  eSectionTypeInvalid,
  eSectionTypeCode,
  /// The section contains child sections
  eSectionTypeContainer,
  eSectionTypeData,
  /// Inlined C string data
  eSectionTypeDataCString,
  /// Pointers to C string data
  eSectionTypeDataCStringPointers,
  /// Address of a symbol in the symbol table
  eSectionTypeDataSymbolAddress,
  eSectionTypeData4,
  eSectionTypeData8,
  eSectionTypeData16,
  eSectionTypeDataPointers,
  eSectionTypeDebug,
  eSectionTypeZeroFill,
  /// Pointer to function pointer + selector
  eSectionTypeDataObjCMessageRefs,
  /// Objective-C const CFString/NSString
  /// objects
  eSectionTypeDataObjCCFStrings,
  eSectionTypeDWARFDebugAbbrev,
  eSectionTypeDWARFDebugAddr,
  eSectionTypeDWARFDebugAranges,
  eSectionTypeDWARFDebugCuIndex,
  eSectionTypeDWARFDebugFrame,
  eSectionTypeDWARFDebugInfo,
  eSectionTypeDWARFDebugLine,
  eSectionTypeDWARFDebugLoc,
  eSectionTypeDWARFDebugMacInfo,
  eSectionTypeDWARFDebugMacro,
  eSectionTypeDWARFDebugPubNames,
  eSectionTypeDWARFDebugPubTypes,
  eSectionTypeDWARFDebugRanges,
  eSectionTypeDWARFDebugStr,
  eSectionTypeDWARFDebugStrOffsets,
  eSectionTypeDWARFAppleNames,
  eSectionTypeDWARFAppleTypes,
  eSectionTypeDWARFAppleNamespaces,
  eSectionTypeDWARFAppleObjC,
  /// Elf SHT_SYMTAB section
  eSectionTypeELFSymbolTable,
  /// Elf SHT_DYNSYM section
  eSectionTypeELFDynamicSymbols,
  /// Elf SHT_REL or SHT_REL section
  eSectionTypeELFRelocationEntries,
  /// Elf SHT_DYNAMIC section
  eSectionTypeELFDynamicLinkInfo,
  eSectionTypeEHFrame,
  eSectionTypeARMexidx,
  eSectionTypeARMextab,
  /// compact unwind section in Mach-O,
  /// __TEXT,__unwind_info
  eSectionTypeCompactUnwind,
  eSectionTypeGoSymtab,
  /// Dummy section for symbols with absolute
  /// address
  eSectionTypeAbsoluteAddress,
  eSectionTypeDWARFGNUDebugAltLink,
  /// DWARF .debug_types section
  eSectionTypeDWARFDebugTypes,
  /// DWARF v5 .debug_names
  eSectionTypeDWARFDebugNames,
  eSectionTypeOther,
  /// DWARF v5 .debug_line_str
  eSectionTypeDWARFDebugLineStr,
  /// DWARF v5 .debug_rnglists
  eSectionTypeDWARFDebugRngLists,
  /// DWARF v5 .debug_loclists
  eSectionTypeDWARFDebugLocLists,
  eSectionTypeDWARFDebugAbbrevDwo,
  eSectionTypeDWARFDebugInfoDwo,
  eSectionTypeDWARFDebugStrDwo,
  eSectionTypeDWARFDebugStrOffsetsDwo,
  eSectionTypeDWARFDebugTypesDwo,
  eSectionTypeDWARFDebugRngListsDwo,
  eSectionTypeDWARFDebugLocDwo,
  eSectionTypeDWARFDebugLocListsDwo,
  eSectionTypeDWARFDebugTuIndex,
  eSectionTypeCTF,
  eSectionTypeLLDBTypeSummaries,
  eSectionTypeLLDBFormatters,
  eSectionTypeSwiftModules,
  eSectionTypeWasmName,
};

FLAGS_ENUM(EmulateInstructionOptions){
    eEmulateInstructionOptionNone = (0u),
    eEmulateInstructionOptionAutoAdvancePC = (1u << 0),
    eEmulateInstructionOptionIgnoreConditions = (1u << 1)};

FLAGS_ENUM(FunctionNameType){
    eFunctionNameTypeNone = 0u,
    /// Automatically figure out which FunctionNameType bits to set based on
    /// the function name.
    eFunctionNameTypeAuto = (1u << 1),
    /// The function name. For C this is the same as just the name of the
    /// function. For C++ this is the mangled or demangled version of the
    /// mangled name. For ObjC this is the full function signature with the +
    /// or - and the square brackets and the class and selector.
    eFunctionNameTypeFull = (1u << 2),
    /// The function name only, no namespaces or arguments and no class
    /// methods or selectors will be searched.
    eFunctionNameTypeBase = (1u << 3),
    /// Find function by method name (C++) with no namespace or arguments.
    eFunctionNameTypeMethod = (1u << 4),
    /// Find function by selector name (ObjC) names.
    eFunctionNameTypeSelector = (1u << 5),
    /// DEPRECATED: use eFunctionNameTypeAuto.
    eFunctionNameTypeAny = eFunctionNameTypeAuto
};
LLDB_MARK_AS_BITMASK_ENUM(FunctionNameType)

/// Basic types enumeration for the public API SBType::GetBasicType().
enum BasicType {
  eBasicTypeInvalid = 0,
  eBasicTypeVoid = 1,
  eBasicTypeChar,
  eBasicTypeSignedChar,
  eBasicTypeUnsignedChar,
  eBasicTypeWChar,
  eBasicTypeSignedWChar,
  eBasicTypeUnsignedWChar,
  eBasicTypeChar16,
  eBasicTypeChar32,
  eBasicTypeChar8,
  eBasicTypeShort,
  eBasicTypeUnsignedShort,
  eBasicTypeInt,
  eBasicTypeUnsignedInt,
  eBasicTypeLong,
  eBasicTypeUnsignedLong,
  eBasicTypeLongLong,
  eBasicTypeUnsignedLongLong,
  eBasicTypeInt128,
  eBasicTypeUnsignedInt128,
  eBasicTypeBool,
  eBasicTypeHalf,
  eBasicTypeFloat,
  eBasicTypeDouble,
  eBasicTypeLongDouble,
  eBasicTypeFloatComplex,
  eBasicTypeDoubleComplex,
  eBasicTypeLongDoubleComplex,
  eBasicTypeObjCID,
  eBasicTypeObjCClass,
  eBasicTypeObjCSel,
  eBasicTypeNullPtr,
  eBasicTypeOther,
  eBasicTypeFloat128
};

/// Deprecated
enum TraceType {
  eTraceTypeNone = 0,

  /// Intel Processor Trace
  eTraceTypeProcessorTrace
};

enum StructuredDataType {
  eStructuredDataTypeInvalid = -1,
  eStructuredDataTypeNull = 0,
  eStructuredDataTypeGeneric,
  eStructuredDataTypeArray,
  eStructuredDataTypeInteger,
  eStructuredDataTypeFloat,
  eStructuredDataTypeBoolean,
  eStructuredDataTypeString,
  eStructuredDataTypeDictionary,
  eStructuredDataTypeSignedInteger,
  eStructuredDataTypeUnsignedInteger = eStructuredDataTypeInteger,
};

FLAGS_ENUM(TypeClass){
    eTypeClassInvalid = (0u), eTypeClassArray = (1u << 0),
    eTypeClassBlockPointer = (1u << 1), eTypeClassBuiltin = (1u << 2),
    eTypeClassClass = (1u << 3), eTypeClassComplexFloat = (1u << 4),
    eTypeClassComplexInteger = (1u << 5), eTypeClassEnumeration = (1u << 6),
    eTypeClassFunction = (1u << 7), eTypeClassMemberPointer = (1u << 8),
    eTypeClassObjCObject = (1u << 9), eTypeClassObjCInterface = (1u << 10),
    eTypeClassObjCObjectPointer = (1u << 11), eTypeClassPointer = (1u << 12),
    eTypeClassReference = (1u << 13), eTypeClassStruct = (1u << 14),
    eTypeClassTypedef = (1u << 15), eTypeClassUnion = (1u << 16),
    eTypeClassVector = (1u << 17),
    // Define the last type class as the MSBit of a 32 bit value
    eTypeClassOther = (1u << 31),
    // Define a mask that can be used for any type when finding types
    eTypeClassAny = (0xffffffffu)};
LLDB_MARK_AS_BITMASK_ENUM(TypeClass)

enum TemplateArgumentKind {
  eTemplateArgumentKindNull = 0,
  eTemplateArgumentKindType,
  eTemplateArgumentKindDeclaration,
  eTemplateArgumentKindIntegral,
  eTemplateArgumentKindTemplate,
  eTemplateArgumentKindTemplateExpansion,
  eTemplateArgumentKindExpression,
  eTemplateArgumentKindPack,
  eTemplateArgumentKindNullPtr,
  eTemplateArgumentKindStructuralValue,
};

/// Type of match to be performed when looking for a formatter for a data type.
/// Used by classes like SBTypeNameSpecifier or lldb_private::TypeMatcher.
enum FormatterMatchType {
  eFormatterMatchExact,
  eFormatterMatchRegex,
  eFormatterMatchCallback,

  eLastFormatterMatchType = eFormatterMatchCallback,
};

/// Options that can be set for a formatter to alter its behavior. Not
/// all of these are applicable to all formatter types.
FLAGS_ENUM(TypeOptions){eTypeOptionNone = (0u),
                        eTypeOptionCascade = (1u << 0),
                        eTypeOptionSkipPointers = (1u << 1),
                        eTypeOptionSkipReferences = (1u << 2),
                        eTypeOptionHideChildren = (1u << 3),
                        eTypeOptionHideValue = (1u << 4),
                        eTypeOptionShowOneLiner = (1u << 5),
                        eTypeOptionHideNames = (1u << 6),
                        eTypeOptionNonCacheable = (1u << 7),
                        eTypeOptionHideEmptyAggregates = (1u << 8),
                        eTypeOptionFrontEndWantsDereference = (1u << 9),
                        eTypeOptionCustomSubscripting = (1u << 10)};

/// This is the return value for frame comparisons.  If you are comparing frame
/// A to frame B the following cases arise:
///
///    1) When frame A pushes frame B (or a frame that ends up pushing
///       B) A is Older than B.
///
///    2) When frame A pushed frame B (or if frameA is on the stack
///       but B is not) A is Younger than B.
///
///    3) When frame A and frame B have the same StackID, they are
///       Equal.
///
///    4) When frame A and frame B have the same immediate parent
///       frame, but are not equal, the comparison yields SameParent.
///
///    5) If the two frames are on different threads or processes the
///       comparison is Invalid.
///
///    6) If for some reason we can't figure out what went on, we
///       return Unknown.
enum FrameComparison {
  eFrameCompareInvalid,
  eFrameCompareUnknown,
  eFrameCompareEqual,
  eFrameCompareSameParent,
  eFrameCompareYounger,
  eFrameCompareOlder
};

/// File Permissions.
///
/// Designed to mimic the unix file permission bits so they can be used with
/// functions that set 'mode_t' to certain values for permissions.
FLAGS_ENUM(FilePermissions){
    eFilePermissionsUserRead = (1u << 8),
    eFilePermissionsUserWrite = (1u << 7),
    eFilePermissionsUserExecute = (1u << 6),
    eFilePermissionsGroupRead = (1u << 5),
    eFilePermissionsGroupWrite = (1u << 4),
    eFilePermissionsGroupExecute = (1u << 3),
    eFilePermissionsWorldRead = (1u << 2),
    eFilePermissionsWorldWrite = (1u << 1),
    eFilePermissionsWorldExecute = (1u << 0),

    eFilePermissionsUserRW = (eFilePermissionsUserRead |
                              eFilePermissionsUserWrite | 0),
    eFileFilePermissionsUserRX = (eFilePermissionsUserRead | 0 |
                                  eFilePermissionsUserExecute),
    eFilePermissionsUserRWX = (eFilePermissionsUserRead |
                               eFilePermissionsUserWrite |
                               eFilePermissionsUserExecute),

    eFilePermissionsGroupRW = (eFilePermissionsGroupRead |
                               eFilePermissionsGroupWrite | 0),
    eFilePermissionsGroupRX = (eFilePermissionsGroupRead | 0 |
                               eFilePermissionsGroupExecute),
    eFilePermissionsGroupRWX = (eFilePermissionsGroupRead |
                                eFilePermissionsGroupWrite |
                                eFilePermissionsGroupExecute),

    eFilePermissionsWorldRW = (eFilePermissionsWorldRead |
                               eFilePermissionsWorldWrite | 0),
    eFilePermissionsWorldRX = (eFilePermissionsWorldRead | 0 |
                               eFilePermissionsWorldExecute),
    eFilePermissionsWorldRWX = (eFilePermissionsWorldRead |
                                eFilePermissionsWorldWrite |
                                eFilePermissionsWorldExecute),

    eFilePermissionsEveryoneR = (eFilePermissionsUserRead |
                                 eFilePermissionsGroupRead |
                                 eFilePermissionsWorldRead),
    eFilePermissionsEveryoneW = (eFilePermissionsUserWrite |
                                 eFilePermissionsGroupWrite |
                                 eFilePermissionsWorldWrite),
    eFilePermissionsEveryoneX = (eFilePermissionsUserExecute |
                                 eFilePermissionsGroupExecute |
                                 eFilePermissionsWorldExecute),

    eFilePermissionsEveryoneRW = (eFilePermissionsEveryoneR |
                                  eFilePermissionsEveryoneW | 0),
    eFilePermissionsEveryoneRX = (eFilePermissionsEveryoneR | 0 |
                                  eFilePermissionsEveryoneX),
    eFilePermissionsEveryoneRWX = (eFilePermissionsEveryoneR |
                                   eFilePermissionsEveryoneW |
                                   eFilePermissionsEveryoneX),
    eFilePermissionsFileDefault = eFilePermissionsUserRW,
    eFilePermissionsDirectoryDefault = eFilePermissionsUserRWX,
};

/// Queue work item types.
///
/// The different types of work that can be enqueued on a libdispatch aka Grand
/// Central Dispatch (GCD) queue.
enum QueueItemKind {
  eQueueItemKindUnknown = 0,
  eQueueItemKindFunction,
  eQueueItemKindBlock
};

/// Queue type.
///
/// libdispatch aka Grand Central Dispatch (GCD) queues can be either
/// serial (executing on one thread) or concurrent (executing on
/// multiple threads).
enum QueueKind {
  eQueueKindUnknown = 0,
  eQueueKindSerial,
  eQueueKindConcurrent
};

/// Expression Evaluation Stages.
///
/// These are the cancellable stages of expression evaluation, passed
/// to the expression evaluation callback, so that you can interrupt
/// expression evaluation at the various points in its lifecycle.
enum ExpressionEvaluationPhase {
  eExpressionEvaluationParse = 0,
  eExpressionEvaluationIRGen,
  eExpressionEvaluationExecution,
  eExpressionEvaluationComplete
};

/// Architecture-agnostic categorization of instructions for traversing the
/// control flow of a trace.
///
/// A single instruction can match one or more of these categories.
enum InstructionControlFlowKind {
  /// The instruction could not be classified.
  eInstructionControlFlowKindUnknown = 0,
  /// The instruction is something not listed below, i.e. it's a sequential
  /// instruction that doesn't affect the control flow of the program.
  eInstructionControlFlowKindOther,
  /// The instruction is a near (function) call.
  eInstructionControlFlowKindCall,
  /// The instruction is a near (function) return.
  eInstructionControlFlowKindReturn,
  /// The instruction is a near unconditional jump.
  eInstructionControlFlowKindJump,
  /// The instruction is a near conditional jump.
  eInstructionControlFlowKindCondJump,
  /// The instruction is a call-like far transfer.
  /// E.g. SYSCALL, SYSENTER, or FAR CALL.
  eInstructionControlFlowKindFarCall,
  /// The instruction is a return-like far transfer.
  /// E.g. SYSRET, SYSEXIT, IRET, or FAR RET.
  eInstructionControlFlowKindFarReturn,
  /// The instruction is a jump-like far transfer.
  /// E.g. FAR JMP.
  eInstructionControlFlowKindFarJump
};

/// Watchpoint Kind.
///
/// Indicates what types of events cause the watchpoint to fire. Used by Native
/// *Protocol-related classes.
FLAGS_ENUM(WatchpointKind){eWatchpointKindWrite = (1u << 0),
                           eWatchpointKindRead = (1u << 1)};

enum GdbSignal {
  eGdbSignalBadAccess = 0x91,
  eGdbSignalBadInstruction = 0x92,
  eGdbSignalArithmetic = 0x93,
  eGdbSignalEmulation = 0x94,
  eGdbSignalSoftware = 0x95,
  eGdbSignalBreakpoint = 0x96
};

/// Used with SBHostOS::GetLLDBPath (lldb::PathType) to find files that are
/// related to LLDB on the current host machine. Most files are
/// relative to LLDB or are in known locations.
enum PathType {
  /// The directory where the lldb.so (unix) or LLDB
  /// mach-o file in LLDB.framework (MacOSX) exists
  ePathTypeLLDBShlibDir,
  /// Find LLDB support executable directory
  /// (debugserver, etc)
  ePathTypeSupportExecutableDir,
  /// Find LLDB header file directory
  ePathTypeHeaderDir,
  /// Find Python modules (PYTHONPATH) directory
  ePathTypePythonDir,
  /// System plug-ins directory
  ePathTypeLLDBSystemPlugins,
  /// User plug-ins directory
  ePathTypeLLDBUserPlugins,
  /// The LLDB temp directory for this system that
  /// will be cleaned up on exit
  ePathTypeLLDBTempSystemDir,
  /// The LLDB temp directory for this
  /// system, NOT cleaned up on a process
  /// exit.
  ePathTypeGlobalLLDBTempSystemDir,
  /// Find path to Clang builtin headers
  ePathTypeClangDir
};

/// Kind of member function.
///
/// Used by the type system.
enum MemberFunctionKind {
  /// Not sure what the type of this is
  eMemberFunctionKindUnknown = 0,
  /// A function used to create instances
  eMemberFunctionKindConstructor,
  /// A function used to tear down existing
  /// instances
  eMemberFunctionKindDestructor,
  /// A function that applies to a specific
  /// instance
  eMemberFunctionKindInstanceMethod,
  /// A function that applies to a type rather
  /// than any instance
  eMemberFunctionKindStaticMethod
};

/// String matching algorithm used by SBTarget.
enum MatchType {
  eMatchTypeNormal,
  eMatchTypeRegex,
  eMatchTypeStartsWith,
  eMatchTypeRegexInsensitive
};

/// Bitmask that describes details about a type.
FLAGS_ENUM(TypeFlags){
    eTypeHasChildren = (1u << 0),       eTypeHasValue = (1u << 1),
    eTypeIsArray = (1u << 2),           eTypeIsBlock = (1u << 3),
    eTypeIsBuiltIn = (1u << 4),         eTypeIsClass = (1u << 5),
    eTypeIsCPlusPlus = (1u << 6),       eTypeIsEnumeration = (1u << 7),
    eTypeIsFuncPrototype = (1u << 8),   eTypeIsMember = (1u << 9),
    eTypeIsObjC = (1u << 10),           eTypeIsPointer = (1u << 11),
    eTypeIsReference = (1u << 12),      eTypeIsStructUnion = (1u << 13),
    eTypeIsTemplate = (1u << 14),       eTypeIsTypedef = (1u << 15),
    eTypeIsVector = (1u << 16),         eTypeIsScalar = (1u << 17),
    eTypeIsInteger = (1u << 18),        eTypeIsFloat = (1u << 19),
    eTypeIsComplex = (1u << 20),        eTypeIsSigned = (1u << 21),
    eTypeInstanceIsPointer = (1u << 22)};

FLAGS_ENUM(CommandFlags){
    /// eCommandRequiresTarget
    ///
    /// Ensures a valid target is contained in m_exe_ctx prior to executing the
    /// command. If a target doesn't exist or is invalid, the command will fail
    /// and CommandObject::GetInvalidTargetDescription() will be returned as the
    /// error. CommandObject subclasses can override the virtual function for
    /// GetInvalidTargetDescription() to provide custom strings when needed.
    eCommandRequiresTarget = (1u << 0),
    /// eCommandRequiresProcess
    ///
    /// Ensures a valid process is contained in m_exe_ctx prior to executing the
    /// command. If a process doesn't exist or is invalid, the command will fail
    /// and CommandObject::GetInvalidProcessDescription() will be returned as
    /// the error. CommandObject subclasses can override the virtual function
    /// for GetInvalidProcessDescription() to provide custom strings when
    /// needed.
    eCommandRequiresProcess = (1u << 1),
    /// eCommandRequiresThread
    ///
    /// Ensures a valid thread is contained in m_exe_ctx prior to executing the
    /// command. If a thread doesn't exist or is invalid, the command will fail
    /// and CommandObject::GetInvalidThreadDescription() will be returned as the
    /// error. CommandObject subclasses can override the virtual function for
    /// GetInvalidThreadDescription() to provide custom strings when needed.
    eCommandRequiresThread = (1u << 2),
    /// eCommandRequiresFrame
    ///
    /// Ensures a valid frame is contained in m_exe_ctx prior to executing the
    /// command. If a frame doesn't exist or is invalid, the command will fail
    /// and CommandObject::GetInvalidFrameDescription() will be returned as the
    /// error. CommandObject subclasses can override the virtual function for
    /// GetInvalidFrameDescription() to provide custom strings when needed.
    eCommandRequiresFrame = (1u << 3),
    /// eCommandRequiresRegContext
    ///
    /// Ensures a valid register context (from the selected frame if there is a
    /// frame in m_exe_ctx, or from the selected thread from m_exe_ctx) is
    /// available from m_exe_ctx prior to executing the command. If a target
    /// doesn't exist or is invalid, the command will fail and
    /// CommandObject::GetInvalidRegContextDescription() will be returned as the
    /// error. CommandObject subclasses can override the virtual function for
    /// GetInvalidRegContextDescription() to provide custom strings when needed.
    eCommandRequiresRegContext = (1u << 4),
    /// eCommandTryTargetAPILock
    ///
    /// Attempts to acquire the target lock if a target is selected in the
    /// command interpreter. If the command object fails to acquire the API
    /// lock, the command will fail with an appropriate error message.
    eCommandTryTargetAPILock = (1u << 5),
    /// eCommandProcessMustBeLaunched
    ///
    /// Verifies that there is a launched process in m_exe_ctx, if there isn't,
    /// the command will fail with an appropriate error message.
    eCommandProcessMustBeLaunched = (1u << 6),
    /// eCommandProcessMustBePaused
    ///
    /// Verifies that there is a paused process in m_exe_ctx, if there isn't,
    /// the command will fail with an appropriate error message.
    eCommandProcessMustBePaused = (1u << 7),
    /// eCommandProcessMustBeTraced
    ///
    /// Verifies that the process is being traced by a Trace plug-in, if it
    /// isn't the command will fail with an appropriate error message.
    eCommandProcessMustBeTraced = (1u << 8),
    /// eCommandAllowsDummyTarget
    ///
    /// Indicates that the command can legitimately operate on the dummy target
    /// (e.g. `breakpoint set` priming future targets). Without this flag,
    /// CommandObject::GetTarget filters the dummy target out and returns null
    /// when no real target is selected.
    eCommandAllowsDummyTarget = (1u << 9)};

/// Whether a summary should cap how much data it returns to users or not.
enum TypeSummaryCapping {
  eTypeSummaryCapped = true,
  eTypeSummaryUncapped = false
};

/// The result from a command interpreter run.
enum CommandInterpreterResult {
  /// Command interpreter finished successfully.
  eCommandInterpreterResultSuccess,
  /// Stopped because the corresponding option was set and the inferior
  /// crashed.
  eCommandInterpreterResultInferiorCrash,
  /// Stopped because the corresponding option was set and a command returned
  /// an error.
  eCommandInterpreterResultCommandError,
  /// Stopped because quit was requested.
  eCommandInterpreterResultQuitRequested,
};

// Style of core file to create when calling SaveCore.
enum SaveCoreStyle {
  eSaveCoreUnspecified = 0,
  eSaveCoreFull = 1,
  eSaveCoreDirtyOnly = 2,
  eSaveCoreStackOnly = 3,
  eSaveCoreCustomOnly = 4,
};

/// Events that might happen during a trace session.
enum TraceEvent {
  /// Tracing was disabled for some time due to a software trigger.
  eTraceEventDisabledSW,
  /// Tracing was disable for some time due to a hardware trigger.
  eTraceEventDisabledHW,
  /// Event due to CPU change for a thread. This event is also fired when
  /// suddenly it's not possible to identify the cpu of a given thread.
  eTraceEventCPUChanged,
  /// Event due to a CPU HW clock tick.
  eTraceEventHWClockTick,
  /// The underlying tracing technology emitted a synchronization event used by
  /// trace processors.
  eTraceEventSyncPoint,
};

// Enum used to identify which kind of item a \a TraceCursor is pointing at
enum TraceItemKind {
  eTraceItemKindError = 0,
  eTraceItemKindEvent,
  eTraceItemKindInstruction,
};

/// Enum to indicate the reference point when invoking
/// \a TraceCursor::Seek().
/// The following values are inspired by \a std::istream::seekg.
enum TraceCursorSeekType {
  /// The beginning of the trace, i.e the oldest item.
  eTraceCursorSeekTypeBeginning = 0,
  /// The current position in the trace.
  eTraceCursorSeekTypeCurrent,
  /// The end of the trace, i.e the most recent item.
  eTraceCursorSeekTypeEnd
};

/// Enum to control the verbosity level of `dwim-print` execution.
enum DWIMPrintVerbosity {
  /// Run `dwim-print` with no verbosity.
  eDWIMPrintVerbosityNone,
  /// Print a message when `dwim-print` uses `expression` evaluation.
  eDWIMPrintVerbosityExpression,
  /// Always print a message indicating how `dwim-print` is evaluating its
  /// expression.
  eDWIMPrintVerbosityFull,
};

enum WatchpointValueKind {
  eWatchPointValueKindInvalid = 0,
  /// Watchpoint was created watching a variable
  eWatchPointValueKindVariable = 1,
  /// Watchpoint was created watching the result of an expression that was
  /// evaluated at creation time.
  eWatchPointValueKindExpression = 2,
};

enum CompletionType {
  eNoCompletion = 0ul,
  eSourceFileCompletion = (1ul << 0),
  eDiskFileCompletion = (1ul << 1),
  eDiskDirectoryCompletion = (1ul << 2),
  eSymbolCompletion = (1ul << 3),
  eModuleCompletion = (1ul << 4),
  eSettingsNameCompletion = (1ul << 5),
  ePlatformPluginCompletion = (1ul << 6),
  eArchitectureCompletion = (1ul << 7),
  eVariablePathCompletion = (1ul << 8),
  eRegisterCompletion = (1ul << 9),
  eBreakpointCompletion = (1ul << 10),
  eProcessPluginCompletion = (1ul << 11),
  eDisassemblyFlavorCompletion = (1ul << 12),
  eTypeLanguageCompletion = (1ul << 13),
  eFrameIndexCompletion = (1ul << 14),
  eModuleUUIDCompletion = (1ul << 15),
  eStopHookIDCompletion = (1ul << 16),
  eThreadIndexCompletion = (1ul << 17),
  eWatchpointIDCompletion = (1ul << 18),
  eBreakpointNameCompletion = (1ul << 19),
  eProcessIDCompletion = (1ul << 20),
  eProcessNameCompletion = (1ul << 21),
  eRemoteDiskFileCompletion = (1ul << 22),
  eRemoteDiskDirectoryCompletion = (1ul << 23),
  eTypeCategoryNameCompletion = (1ul << 24),
  eCustomCompletion = (1ul << 25),
  eThreadIDCompletion = (1ul << 26),
  eManagedPluginCompletion = (1ul << 27),
  // This last enum element is just for input validation.
  // Add new completions before this element,
  // and then increment eTerminatorCompletion's shift value
  eTerminatorCompletion = (1ul << 28)
};

/// Specifies if children need to be re-computed
/// after a call to \ref SyntheticChildrenFrontEnd::Update.
enum ChildCacheState {
  /// Children need to be recomputed dynamically.
  eRefetch = 0,

  /// Children did not change and don't need to be recomputed;
  /// re-use what we computed the last time we called Update.
  eReuse = 1,
};

enum SymbolDownload {
  eSymbolDownloadOff = 0,
  eSymbolDownloadBackground = 1,
  eSymbolDownloadForeground = 2,
};

enum SymbolSharedCacheUse {
  eSymbolSharedCacheUseHostLLDBMemory = 1,
  eSymbolSharedCacheUseHostSharedCache = 2,
  eSymbolSharedCacheUseHostAndInferiorSharedCache = 3,
  eSymbolSharedCacheUseInferiorSharedCacheOnly = 4,
};

/// Used in the SBProcess AddressMask/FixAddress methods.
enum AddressMaskType {
  eAddressMaskTypeCode = 0,
  eAddressMaskTypeData,
  eAddressMaskTypeAny,
  eAddressMaskTypeAll = eAddressMaskTypeAny
};

/// Used in the SBProcess AddressMask/FixAddress methods.
enum AddressMaskRange {
  eAddressMaskRangeLow = 0,
  eAddressMaskRangeHigh,
  eAddressMaskRangeAny,
  eAddressMaskRangeAll = eAddressMaskRangeAny,
};

/// Used by the debugger to indicate which events are being broadcasted.
enum DebuggerBroadcastBit {
  eBroadcastBitProgress = (1 << 0),
  eBroadcastBitWarning = (1 << 1),
  eBroadcastBitError = (1 << 2),
  eBroadcastSymbolChange = (1 << 3),
  /// Deprecated
  eBroadcastBitProgressCategory = (1 << 4),
  eBroadcastBitExternalProgress = (1 << 5),
  /// Deprecated
  eBroadcastBitExternalProgressCategory = (1 << 6),
};

/// Used for expressing severity in logs and diagnostics.
enum Severity {
  eSeverityError,
  eSeverityWarning,
  eSeverityInfo, // Equivalent to Remark used in clang.
};

/// Callback return value, indicating whether it handled printing the
/// CommandReturnObject or deferred doing so to the CommandInterpreter.
enum CommandReturnObjectCallbackResult {
  /// The callback deferred printing the command return object.
  eCommandReturnObjectPrintCallbackSkipped = 0,
  /// The callback handled printing the command return object.
  eCommandReturnObjectPrintCallbackHandled = 1,
};

/// Used to determine when to show disassembly.
enum StopDisassemblyType {
  eStopDisassemblyTypeNever = 0,
  eStopDisassemblyTypeNoDebugInfo,
  eStopDisassemblyTypeNoSource,
  eStopDisassemblyTypeAlways
};

enum ExceptionStage {
  eExceptionStageCreate = (1 << 0),
  eExceptionStageThrow = (1 << 1),
  eExceptionStageReThrow = (1 << 2),
  eExceptionStageCatch = (1 << 3)
};

enum NameMatchStyle {
  eNameMatchStyleAuto = eFunctionNameTypeAuto,
  eNameMatchStyleFull = eFunctionNameTypeFull,
  eNameMatchStyleBase = eFunctionNameTypeBase,
  eNameMatchStyleMethod = eFunctionNameTypeMethod,
  eNameMatchStyleSelector = eFunctionNameTypeSelector,
  eNameMatchStyleRegex = eFunctionNameTypeSelector << 1
};

/// Data Inspection Language (DIL) evaluation modes.
/// DIL will only attempt evaluating expressions that contain tokens
/// allowed by a selected mode.
enum DILMode {
  /// Allowed: identifiers, operators: '.'.
  eDILModeSimple,
  /// Allowed: identifiers, integers, operators: '.', '->', '*', '&', '[]'.
  eDILModeLegacy,
  /// Allowed: everything supported by DIL.
  /// \see lldb/docs/dil-expr-lang.ebnf
  eDILModeFull
};

/// When the Process plugin can retrieve information
/// about all binaries loaded in the target process,
/// or given a list of binary load addresses, this
/// enum specifies how much information needed from
/// the Process plugin; there may be performance reasons
/// to limit the amount of information returned.
enum BinaryInformationLevel {
  eBinaryInformationLevelAddrOnly,
  eBinaryInformationLevelAddrName,
  eBinaryInformationLevelAddrNameUUID,
  eBinaryInformationLevelFull
};

} // namespace lldb

#endif // LLDB_LLDB_ENUMERATIONS_H
