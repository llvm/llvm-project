//===-- lldb-enumerations.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_lldb_enumerations_h_
#define LLDB_lldb_enumerations_h_

namespace lldb {

    //----------------------------------------------------------------------
    // Process and Thread States
    //----------------------------------------------------------------------
    typedef enum StateType
    {
        eStateInvalid = 0,
        eStateUnloaded,     ///< Process is object is valid, but not currently loaded
        eStateConnected,    ///< Process is connected to remote debug services, but not launched or attached to anything yet
        eStateAttaching,    ///< Process is currently trying to attach
        eStateLaunching,    ///< Process is in the process of launching
        eStateStopped,      ///< Process or thread is stopped and can be examined.
        eStateRunning,      ///< Process or thread is running and can't be examined.
        eStateStepping,     ///< Process or thread is in the process of stepping and can not be examined.
        eStateCrashed,      ///< Process or thread has crashed and can be examined.
        eStateDetached,     ///< Process has been detached and can't be examined.
        eStateExited,       ///< Process has exited and can't be examined.
        eStateSuspended     ///< Process or thread is in a suspended state as far
                            ///< as the debugger is concerned while other processes
                            ///< or threads get the chance to run.
    } StateType;

    //----------------------------------------------------------------------
    // Launch Flags
    //----------------------------------------------------------------------
    typedef enum LaunchFlags
    {
        eLaunchFlagNone         = 0u,
        eLaunchFlagExec         = (1u << 0),  ///< Exec when launching and turn the calling process into a new process
        eLaunchFlagDebug        = (1u << 1),  ///< Stop as soon as the process launches to allow the process to be debugged
        eLaunchFlagStopAtEntry  = (1u << 2),  ///< Stop at the program entry point instead of auto-continuing when launching or attaching at entry point
        eLaunchFlagDisableASLR  = (1u << 3),  ///< Disable Address Space Layout Randomization
        eLaunchFlagDisableSTDIO = (1u << 4),  ///< Disable stdio for inferior process (e.g. for a GUI app)
        eLaunchFlagLaunchInTTY  = (1u << 5),  ///< Launch the process in a new TTY if supported by the host 
        eLaunchFlagLaunchInShell= (1u << 6),   ///< Launch the process inside a shell to get shell expansion
        eLaunchFlagLaunchInSeparateProcessGroup = (1u << 7) ///< Launch the process in a separate process group
    } LaunchFlags;
        
    //----------------------------------------------------------------------
    // Thread Run Modes
    //----------------------------------------------------------------------
    typedef enum RunMode {
        eOnlyThisThread,
        eAllThreads,
        eOnlyDuringStepping
    } RunMode;

    //----------------------------------------------------------------------
    // Byte ordering definitions
    //----------------------------------------------------------------------
    typedef enum ByteOrder
    {
        eByteOrderInvalid   = 0,
        eByteOrderBig       = 1,
        eByteOrderPDP       = 2,
        eByteOrderLittle    = 4
    } ByteOrder;

    //----------------------------------------------------------------------
    // Register encoding definitions
    //----------------------------------------------------------------------
    typedef enum Encoding
    {
        eEncodingInvalid = 0,
        eEncodingUint,               // unsigned integer
        eEncodingSint,               // signed integer
        eEncodingIEEE754,            // float
        eEncodingVector              // vector registers
    } Encoding;

    //----------------------------------------------------------------------
    // Display format definitions
    //----------------------------------------------------------------------
    typedef enum Format
    {
        eFormatDefault = 0,
        eFormatInvalid = 0,
        eFormatBoolean,
        eFormatBinary,
        eFormatBytes,
        eFormatBytesWithASCII,
        eFormatChar,
        eFormatCharPrintable,   // Only printable characters, space if not printable
        eFormatComplex,         // Floating point complex type
        eFormatComplexFloat = eFormatComplex,
        eFormatCString,         // NULL terminated C strings
        eFormatDecimal,
        eFormatEnum,
        eFormatHex,
        eFormatHexUppercase,
        eFormatFloat,
        eFormatOctal,
        eFormatOSType,          // OS character codes encoded into an integer 'PICT' 'text' etc...
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
        eFormatVectorOfFloat32,
        eFormatVectorOfFloat64,
        eFormatVectorOfUInt128,
        eFormatComplexInteger,      // Integer complex type
        eFormatCharArray,           // Print characters with no single quotes, used for character arrays that can contain non printable characters
        eFormatAddressInfo,         // Describe what an address points to (func + offset with file/line, symbol + offset, data, etc)
        eFormatHexFloat,            // ISO C99 hex float string
        eFormatInstruction,         // Disassemble an opcode
        eFormatVoid,                // Do not print this
        kNumFormats
    } Format;

    //----------------------------------------------------------------------
    // Description levels for "void GetDescription(Stream *, DescriptionLevel)" calls
    //----------------------------------------------------------------------
    typedef enum DescriptionLevel
    {
        eDescriptionLevelBrief = 0,
        eDescriptionLevelFull,
        eDescriptionLevelVerbose,
        eDescriptionLevelInitial,
        kNumDescriptionLevels
    } DescriptionLevel;

    //----------------------------------------------------------------------
    // Script interpreter types
    //----------------------------------------------------------------------
    typedef enum ScriptLanguage
    {
        eScriptLanguageNone,
        eScriptLanguagePython,
        eScriptLanguageDefault = eScriptLanguagePython
    } ScriptLanguage;

    //----------------------------------------------------------------------
    // Register numbering types
    //----------------------------------------------------------------------
    typedef enum RegisterKind
    {
        eRegisterKindGCC = 0,    // the register numbers seen in eh_frame
        eRegisterKindDWARF,      // the register numbers seen DWARF
        eRegisterKindGeneric,    // insn ptr reg, stack ptr reg, etc not specific to any particular target
        eRegisterKindGDB,        // the register numbers gdb uses (matches stabs numbers?)
        eRegisterKindLLDB,       // lldb's internal register numbers
        kNumRegisterKinds
    } RegisterKind;

    //----------------------------------------------------------------------
    // Thread stop reasons
    //----------------------------------------------------------------------
    typedef enum StopReason
    {
        eStopReasonInvalid = 0,
        eStopReasonNone,
        eStopReasonTrace,
        eStopReasonBreakpoint,
        eStopReasonWatchpoint,
        eStopReasonSignal,
        eStopReasonException,
        eStopReasonPlanComplete
    } StopReason;

    //----------------------------------------------------------------------
    // Command Return Status Types
    //----------------------------------------------------------------------
    typedef enum ReturnStatus
    {
        eReturnStatusInvalid,
        eReturnStatusSuccessFinishNoResult,
        eReturnStatusSuccessFinishResult,
        eReturnStatusSuccessContinuingNoResult,
        eReturnStatusSuccessContinuingResult,
        eReturnStatusStarted,
        eReturnStatusFailed,
        eReturnStatusQuit
    } ReturnStatus;


    //----------------------------------------------------------------------
    // Connection Status Types
    //----------------------------------------------------------------------
    typedef enum ConnectionStatus
    {
        eConnectionStatusSuccess,         // Success
        eConnectionStatusEndOfFile,       // End-of-file encountered
        eConnectionStatusError,           // Check GetError() for details
        eConnectionStatusTimedOut,        // Request timed out
        eConnectionStatusNoConnection,    // No connection
        eConnectionStatusLostConnection   // Lost connection while connected to a valid connection
    } ConnectionStatus;

    typedef enum ErrorType
    {
        eErrorTypeInvalid,
        eErrorTypeGeneric,      ///< Generic errors that can be any value.
        eErrorTypeMachKernel,   ///< Mach kernel error codes.
        eErrorTypePOSIX         ///< POSIX error codes.
    } ErrorType;


    typedef enum ValueType
    {
        eValueTypeInvalid           = 0,
        eValueTypeVariableGlobal    = 1,    // globals variable
        eValueTypeVariableStatic    = 2,    // static variable
        eValueTypeVariableArgument  = 3,    // function argument variables
        eValueTypeVariableLocal     = 4,    // function local variables
        eValueTypeRegister          = 5,    // stack frame register value
        eValueTypeRegisterSet       = 6,    // A collection of stack frame register values
        eValueTypeConstResult       = 7     // constant result variables
    } ValueType;

    //----------------------------------------------------------------------
    // Token size/granularities for Input Readers
    //----------------------------------------------------------------------

    typedef enum InputReaderGranularity
    {
        eInputReaderGranularityInvalid = 0,
        eInputReaderGranularityByte,
        eInputReaderGranularityWord,
        eInputReaderGranularityLine,
        eInputReaderGranularityAll
    } InputReaderGranularity;

    //------------------------------------------------------------------
    /// These mask bits allow a common interface for queries that can
    /// limit the amount of information that gets parsed to only the
    /// information that is requested. These bits also can indicate what
    /// actually did get resolved during query function calls.
    ///
    /// Each definition corresponds to a one of the member variables
    /// in this class, and requests that that item be resolved, or
    /// indicates that the member did get resolved.
    //------------------------------------------------------------------
    typedef enum SymbolContextItem
    {
        eSymbolContextTarget     = (1u << 0), ///< Set when \a target is requested from a query, or was located in query results
        eSymbolContextModule     = (1u << 1), ///< Set when \a module is requested from a query, or was located in query results
        eSymbolContextCompUnit   = (1u << 2), ///< Set when \a comp_unit is requested from a query, or was located in query results
        eSymbolContextFunction   = (1u << 3), ///< Set when \a function is requested from a query, or was located in query results
        eSymbolContextBlock      = (1u << 4), ///< Set when the deepest \a block is requested from a query, or was located in query results
        eSymbolContextLineEntry  = (1u << 5), ///< Set when \a line_entry is requested from a query, or was located in query results
        eSymbolContextSymbol     = (1u << 6), ///< Set when \a symbol is requested from a query, or was located in query results
        eSymbolContextEverything = ((eSymbolContextSymbol << 1) - 1u)  ///< Indicates to try and lookup everything up during a query.
    } SymbolContextItem;

    typedef enum Permissions
    {
        ePermissionsWritable    = (1u << 0),
        ePermissionsReadable    = (1u << 1),
        ePermissionsExecutable  = (1u << 2)
    } Permissions;

    typedef enum InputReaderAction
    {
        eInputReaderActivate,   // reader is newly pushed onto the reader stack 
        eInputReaderAsynchronousOutputWritten, // an async output event occurred; the reader may want to do something
        eInputReaderReactivate, // reader is on top of the stack again after another reader was popped off 
        eInputReaderDeactivate, // another reader was pushed on the stack 
        eInputReaderGotToken,   // reader got one of its tokens (granularity)
        eInputReaderInterrupt,  // reader received an interrupt signal (probably from a control-c)
        eInputReaderEndOfFile,  // reader received an EOF char (probably from a control-d)
        eInputReaderDone        // reader was just popped off the stack and is done
    } InputReaderAction;

    typedef enum BreakpointEventType
    {
        eBreakpointEventTypeInvalidType         = (1u << 0),
        eBreakpointEventTypeAdded               = (1u << 1),
        eBreakpointEventTypeRemoved             = (1u << 2),
        eBreakpointEventTypeLocationsAdded      = (1u << 3),  // Locations added doesn't get sent when the breakpoint is created
        eBreakpointEventTypeLocationsRemoved    = (1u << 4),
        eBreakpointEventTypeLocationsResolved   = (1u << 5),
        eBreakpointEventTypeEnabled             = (1u << 6),
        eBreakpointEventTypeDisabled            = (1u << 7),
        eBreakpointEventTypeCommandChanged      = (1u << 8),
        eBreakpointEventTypeConditionChanged    = (1u << 9),
        eBreakpointEventTypeIgnoreChanged       = (1u << 10),
        eBreakpointEventTypeThreadChanged       = (1u << 11)
    } BreakpointEventType;


    //----------------------------------------------------------------------
    /// Programming language type.
    ///
    /// These enumerations use the same language enumerations as the DWARF
    /// specification for ease of use and consistency.
    /// The enum -> string code is in LanguageRuntime.cpp, don't change this
    /// table without updating that code as well.
    //----------------------------------------------------------------------
    typedef enum LanguageType
    {
        eLanguageTypeUnknown         = 0x0000,   ///< Unknown or invalid language value.
        eLanguageTypeC89             = 0x0001,   ///< ISO C:1989.
        eLanguageTypeC               = 0x0002,   ///< Non-standardized C, such as K&R.
        eLanguageTypeAda83           = 0x0003,   ///< ISO Ada:1983.
        eLanguageTypeC_plus_plus     = 0x0004,   ///< ISO C++:1998.
        eLanguageTypeCobol74         = 0x0005,   ///< ISO Cobol:1974.
        eLanguageTypeCobol85         = 0x0006,   ///< ISO Cobol:1985.
        eLanguageTypeFortran77       = 0x0007,   ///< ISO Fortran 77.
        eLanguageTypeFortran90       = 0x0008,   ///< ISO Fortran 90.
        eLanguageTypePascal83        = 0x0009,   ///< ISO Pascal:1983.
        eLanguageTypeModula2         = 0x000a,   ///< ISO Modula-2:1996.
        eLanguageTypeJava            = 0x000b,   ///< Java.
        eLanguageTypeC99             = 0x000c,   ///< ISO C:1999.
        eLanguageTypeAda95           = 0x000d,   ///< ISO Ada:1995.
        eLanguageTypeFortran95       = 0x000e,   ///< ISO Fortran 95.
        eLanguageTypePLI             = 0x000f,   ///< ANSI PL/I:1976.
        eLanguageTypeObjC            = 0x0010,   ///< Objective-C.
        eLanguageTypeObjC_plus_plus  = 0x0011,   ///< Objective-C++.
        eLanguageTypeUPC             = 0x0012,   ///< Unified Parallel C.
        eLanguageTypeD               = 0x0013,   ///< D.
        eLanguageTypePython          = 0x0014,   ///< Python.
        eNumLanguageTypes
    } LanguageType;

    typedef enum DynamicValueType
    {
        eNoDynamicValues = 0,
        eDynamicCanRunTarget    = 1,
        eDynamicDontRunTarget   = 2
    } DynamicValueType;
    
    typedef enum AccessType
    {
        eAccessNone,
        eAccessPublic,
        eAccessPrivate,
        eAccessProtected,
        eAccessPackage
    } AccessType;

    typedef enum CommandArgumentType
    {
        eArgTypeAddress = 0,
        eArgTypeAliasName,
        eArgTypeAliasOptions,
        eArgTypeArchitecture,
        eArgTypeBoolean,
        eArgTypeBreakpointID,
        eArgTypeBreakpointIDRange,
        eArgTypeByteSize,
        eArgTypeClassName,
        eArgTypeCommandName,
        eArgTypeCount,
        eArgTypeDirectoryName,
        eArgTypeEndAddress,
        eArgTypeExpression,
        eArgTypeExpressionPath,
        eArgTypeExprFormat,
        eArgTypeFilename,
        eArgTypeFormat,
        eArgTypeFrameIndex,
        eArgTypeFullName,
        eArgTypeFunctionName,
        eArgTypeFunctionOrSymbol,
        eArgTypeGDBFormat,
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
        eArgTypeLastArg  // Always keep this entry as the last entry in this enumeration!!
    } CommandArgumentType;

    //----------------------------------------------------------------------
    // Symbol types
    //----------------------------------------------------------------------
    typedef enum SymbolType
    {
        eSymbolTypeAny = 0,
        eSymbolTypeInvalid = 0,
        eSymbolTypeAbsolute,
        eSymbolTypeCode,
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
        eSymbolTypeAdditional, // When symbols take more than one entry, the extra entries get this type
        eSymbolTypeCompiler,
        eSymbolTypeInstrumentation,
        eSymbolTypeUndefined,
        eSymbolTypeObjCClass,
        eSymbolTypeObjCMetaClass,
        eSymbolTypeObjCIVar
    } SymbolType;
    
    typedef enum SectionType
    {
        eSectionTypeInvalid,
        eSectionTypeCode,
        eSectionTypeContainer,              // The section contains child sections
        eSectionTypeData,
        eSectionTypeDataCString,            // Inlined C string data
        eSectionTypeDataCStringPointers,    // Pointers to C string data
        eSectionTypeDataSymbolAddress,      // Address of a symbol in the symbol table
        eSectionTypeData4,
        eSectionTypeData8,
        eSectionTypeData16,
        eSectionTypeDataPointers,
        eSectionTypeDebug,
        eSectionTypeZeroFill,
        eSectionTypeDataObjCMessageRefs,    // Pointer to function pointer + selector
        eSectionTypeDataObjCCFStrings,      // Objective C const CFString/NSString objects
        eSectionTypeDWARFDebugAbbrev,
        eSectionTypeDWARFDebugAranges,
        eSectionTypeDWARFDebugFrame,
        eSectionTypeDWARFDebugInfo,
        eSectionTypeDWARFDebugLine,
        eSectionTypeDWARFDebugLoc,
        eSectionTypeDWARFDebugMacInfo,
        eSectionTypeDWARFDebugPubNames,
        eSectionTypeDWARFDebugPubTypes,
        eSectionTypeDWARFDebugRanges,
        eSectionTypeDWARFDebugStr,
        eSectionTypeDWARFAppleNames,
        eSectionTypeDWARFAppleTypes,
        eSectionTypeDWARFAppleNamespaces,
        eSectionTypeDWARFAppleObjC,
        eSectionTypeEHFrame,
        eSectionTypeOther
        
    } SectionType;

    typedef enum EmulateInstructionOptions
    {
        eEmulateInstructionOptionNone               = (0u),
        eEmulateInstructionOptionAutoAdvancePC      = (1u << 0),
        eEmulateInstructionOptionIgnoreConditions   = (1u << 1)
    } EmulateInstructionOptions;

    typedef enum FunctionNameType 
    {
        eFunctionNameTypeNone       = 0u,
        eFunctionNameTypeAuto       = (1u << 1),    // Automatically figure out which FunctionNameType
                                                    // bits to set based on the function name.
        eFunctionNameTypeFull       = (1u << 2),    // The function name.
                                                    // For C this is the same as just the name of the function
                                                    // For C++ this is the mangled or demangled version of the mangled name.
                                                    // For ObjC this is the full function signature with the + or
                                                    // - and the square brackets and the class and selector
        eFunctionNameTypeBase       = (1u << 3),    // The function name only, no namespaces or arguments and no class 
                                                    // methods or selectors will be searched.
        eFunctionNameTypeMethod     = (1u << 4),    // Find function by method name (C++) with no namespace or arguments
        eFunctionNameTypeSelector   = (1u << 5),    // Find function by selector name (ObjC) names
        eFunctionNameTypeAny        = (eFunctionNameTypeFull     |
                                       eFunctionNameTypeBase     |
                                       eFunctionNameTypeMethod   |
                                       eFunctionNameTypeSelector )
    } FunctionNameType;
    
    
    //----------------------------------------------------------------------
    // Basic types enumeration for the public API SBType::GetBasicType()
    //----------------------------------------------------------------------
    typedef enum BasicType
    {
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
        eBasicTypeOther
    } BasicType;

    typedef enum TypeClass
    {
        eTypeClassInvalid           = (0u),
        eTypeClassArray             = (1u << 0),
        eTypeClassBlockPointer      = (1u << 1),
        eTypeClassBuiltin           = (1u << 2),
        eTypeClassClass             = (1u << 3),
        eTypeClassComplexFloat      = (1u << 4),
        eTypeClassComplexInteger    = (1u << 5),
        eTypeClassEnumeration       = (1u << 6),
        eTypeClassFunction          = (1u << 7),
        eTypeClassMemberPointer     = (1u << 8),
        eTypeClassObjCObject        = (1u << 9),
        eTypeClassObjCInterface     = (1u << 10),
        eTypeClassObjCObjectPointer = (1u << 11),
        eTypeClassPointer           = (1u << 12),
        eTypeClassReference         = (1u << 13),
        eTypeClassStruct            = (1u << 14),
        eTypeClassTypedef           = (1u << 15),
        eTypeClassUnion             = (1u << 16),
        eTypeClassVector            = (1u << 17),
        // Define the last type class as the MSBit of a 32 bit value
        eTypeClassOther             = (1u << 31),
        // Define a mask that can be used for any type when finding types
        eTypeClassAny               = (0xffffffffu)
    }TypeClass;

    typedef enum TemplateArgumentKind
    {
        eTemplateArgumentKindNull = 0,
        eTemplateArgumentKindType,
        eTemplateArgumentKindDeclaration,
        eTemplateArgumentKindIntegral,
        eTemplateArgumentKindTemplate,
        eTemplateArgumentKindTemplateExpansion,
        eTemplateArgumentKindExpression,
        eTemplateArgumentKindPack

    } TemplateArgumentKind;

    //----------------------------------------------------------------------
    // Options that can be set for a formatter to alter its behavior
    // Not all of these are applicable to all formatter types
    //----------------------------------------------------------------------
    typedef enum TypeOptions
    {
        eTypeOptionNone            = (0u),
        eTypeOptionCascade         = (1u << 0),
        eTypeOptionSkipPointers    = (1u << 1),
        eTypeOptionSkipReferences  = (1u << 2),
        eTypeOptionHideChildren    = (1u << 3),
        eTypeOptionHideValue       = (1u << 4),
        eTypeOptionShowOneLiner    = (1u << 5),
        eTypeOptionHideNames       = (1u << 6)
    } TypeOptions;

   //----------------------------------------------------------------------
   // This is the return value for frame comparisons.  When frame A pushes
   // frame B onto the stack, frame A is OLDER than frame B.
   //----------------------------------------------------------------------
   typedef enum FrameComparison
   {
       eFrameCompareInvalid,
       eFrameCompareUnknown,
       eFrameCompareEqual,
       eFrameCompareYounger,
       eFrameCompareOlder
   } FrameComparison;
   
    //----------------------------------------------------------------------
    // Address Class
    //
    // A way of classifying an address used for disassembling and setting 
    // breakpoints. Many object files can track exactly what parts of their
    // object files are code, data and other information. This is of course
    // above and beyond just looking at the section types. For example, code
    // might contain PC relative data and the object file might be able to
    // tell us that an address in code is data.
    //----------------------------------------------------------------------
    typedef enum AddressClass
    {
        eAddressClassInvalid,
        eAddressClassUnknown,
        eAddressClassCode,
        eAddressClassCodeAlternateISA,
        eAddressClassData,
        eAddressClassDebug,
        eAddressClassRuntime
    } AddressClass;

} // namespace lldb


#endif  // LLDB_lldb_enumerations_h_
