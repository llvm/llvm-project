//===-- lldb-private-enumerations.h -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_lldb_private_enumerations_h_
#define LLDB_lldb_private_enumerations_h_

namespace lldb_private {

//----------------------------------------------------------------------
// Thread Step Types
//----------------------------------------------------------------------
typedef enum StepType {
  eStepTypeNone,
  eStepTypeTrace,     ///< Single step one instruction.
  eStepTypeTraceOver, ///< Single step one instruction, stepping over.
  eStepTypeInto,      ///< Single step into a specified context.
  eStepTypeOver,      ///< Single step over a specified context.
  eStepTypeOut,       ///< Single step out a specified context.
  eStepTypeScripted   ///< A step type implemented by the script interpreter.
} StepType;

//----------------------------------------------------------------------
// Address Types
//----------------------------------------------------------------------
typedef enum AddressType {
  eAddressTypeInvalid = 0,
  eAddressTypeFile, ///< Address is an address as found in an object or symbol
                    ///file
  eAddressTypeLoad, ///< Address is an address as in the current target inferior
                    ///process
  eAddressTypeHost  ///< Address is an address in the process that is running
                    ///this code
} AddressType;

//----------------------------------------------------------------------
// Votes - Need a tri-state, yes, no, no opinion...
//----------------------------------------------------------------------
typedef enum Vote { eVoteNo = -1, eVoteNoOpinion = 0, eVoteYes = 1 } Vote;

typedef enum ArchitectureType {
  eArchTypeInvalid,
  eArchTypeMachO,
  eArchTypeELF,
  eArchTypeCOFF,
  kNumArchTypes
} ArchitectureType;

//----------------------------------------------------------------------
/// Settable state variable types.
///
//----------------------------------------------------------------------

// typedef enum SettableVariableType
//{
//    eSetVarTypeInt,
//    eSetVarTypeBoolean,
//    eSetVarTypeString,
//    eSetVarTypeArray,
//    eSetVarTypeDictionary,
//    eSetVarTypeEnum,
//    eSetVarTypeNone
//} SettableVariableType;

typedef enum VarSetOperationType {
  eVarSetOperationReplace,
  eVarSetOperationInsertBefore,
  eVarSetOperationInsertAfter,
  eVarSetOperationRemove,
  eVarSetOperationAppend,
  eVarSetOperationClear,
  eVarSetOperationAssign,
  eVarSetOperationInvalid
} VarSetOperationType;

typedef enum ArgumentRepetitionType {
  eArgRepeatPlain,        // Exactly one occurrence
  eArgRepeatOptional,     // At most one occurrence, but it's optional
  eArgRepeatPlus,         // One or more occurrences
  eArgRepeatStar,         // Zero or more occurrences
  eArgRepeatRange,        // Repetition of same argument, from 1 to n
  eArgRepeatPairPlain,    // A pair of arguments that must always go together
                          // ([arg-type arg-value]), occurs exactly once
  eArgRepeatPairOptional, // A pair that occurs at most once (optional)
  eArgRepeatPairPlus,     // One or more occurrences of a pair
  eArgRepeatPairStar,     // Zero or more occurrences of a pair
  eArgRepeatPairRange,    // A pair that repeats from 1 to n
  eArgRepeatPairRangeOptional // A pair that repeats from 1 to n, but is
                              // optional
} ArgumentRepetitionType;

typedef enum SortOrder {
  eSortOrderNone,
  eSortOrderByAddress,
  eSortOrderByName
} SortOrder;

//----------------------------------------------------------------------
// LazyBool is for boolean values that need to be calculated lazily.
// Values start off set to eLazyBoolCalculate, and then they can be
// calculated once and set to eLazyBoolNo or eLazyBoolYes.
//----------------------------------------------------------------------
typedef enum LazyBool {
  eLazyBoolCalculate = -1,
  eLazyBoolNo = 0,
  eLazyBoolYes = 1
} LazyBool;

//------------------------------------------------------------------
/// Name matching
//------------------------------------------------------------------
typedef enum NameMatchType {
  eNameMatchIgnore,
  eNameMatchEquals,
  eNameMatchContains,
  eNameMatchStartsWith,
  eNameMatchEndsWith,
  eNameMatchRegularExpression

} NameMatchType;

//------------------------------------------------------------------
/// Instruction types
//------------------------------------------------------------------
typedef enum InstructionType {
  eInstructionTypeAny, // Support for any instructions at all (at least one)
  eInstructionTypePrologueEpilogue, // All prologue and epilogue instructions
                                    // that push and pop register values and
                                    // modify sp/fp
  eInstructionTypePCModifying,      // Any instruction that modifies the program
                                    // counter/instruction pointer
  eInstructionTypeAll               // All instructions of any kind

} InstructionType;

//------------------------------------------------------------------
/// Format category entry types
//------------------------------------------------------------------
typedef enum FormatCategoryItem {
  eFormatCategoryItemSummary = 0x0001,
  eFormatCategoryItemRegexSummary = 0x0002,
  eFormatCategoryItemFilter = 0x0004,
  eFormatCategoryItemRegexFilter = 0x0008,
  eFormatCategoryItemSynth = 0x0010,
  eFormatCategoryItemRegexSynth = 0x0020,
  eFormatCategoryItemValue = 0x0040,
  eFormatCategoryItemRegexValue = 0x0080,
  eFormatCategoryItemValidator = 0x0100,
  eFormatCategoryItemRegexValidator = 0x0200
} FormatCategoryItem;

//------------------------------------------------------------------
/// Expression execution policies
//------------------------------------------------------------------
typedef enum {
  eExecutionPolicyOnlyWhenNeeded,
  eExecutionPolicyNever,
  eExecutionPolicyAlways,
  eExecutionPolicyTopLevel // used for top-level code
} ExecutionPolicy;

//----------------------------------------------------------------------
// Ways that the FormatManager picks a particular format for a type
//----------------------------------------------------------------------
typedef enum FormatterChoiceCriterion {
  eFormatterChoiceCriterionDirectChoice = 0x00000000,
  eFormatterChoiceCriterionStrippedPointerReference = 0x00000001,
  eFormatterChoiceCriterionNavigatedTypedefs = 0x00000002,
  eFormatterChoiceCriterionRegularExpressionSummary = 0x00000004,
  eFormatterChoiceCriterionRegularExpressionFilter = 0x00000004,
  eFormatterChoiceCriterionLanguagePlugin = 0x00000008,
  eFormatterChoiceCriterionStrippedBitField = 0x00000010,
  eFormatterChoiceCriterionWentToStaticValue = 0x00000020
} FormatterChoiceCriterion;

//----------------------------------------------------------------------
// Synchronicity behavior of scripted commands
//----------------------------------------------------------------------
typedef enum ScriptedCommandSynchronicity {
  eScriptedCommandSynchronicitySynchronous,
  eScriptedCommandSynchronicityAsynchronous,
  eScriptedCommandSynchronicityCurrentValue // use whatever the current
                                            // synchronicity is
} ScriptedCommandSynchronicity;

//----------------------------------------------------------------------
// Verbosity mode of "po" output
//----------------------------------------------------------------------
typedef enum LanguageRuntimeDescriptionDisplayVerbosity {
  eLanguageRuntimeDescriptionDisplayVerbosityCompact, // only print the
                                                      // description string, if
                                                      // any
  eLanguageRuntimeDescriptionDisplayVerbosityFull,    // print the full-blown
                                                      // output
} LanguageRuntimeDescriptionDisplayVerbosity;

//----------------------------------------------------------------------
// Loading modules from memory
//----------------------------------------------------------------------
typedef enum MemoryModuleLoadLevel {
  eMemoryModuleLoadLevelMinimal,  // Load sections only
  eMemoryModuleLoadLevelPartial,  // Load function bounds but no symbols
  eMemoryModuleLoadLevelComplete, // Load sections and all symbols
} MemoryModuleLoadLevel;

//----------------------------------------------------------------------
// Result enums for when reading multiple lines from IOHandlers
//----------------------------------------------------------------------
enum class LineStatus {
  Success, // The line that was just edited if good and should be added to the
           // lines
  Error, // There is an error with the current line and it needs to be re-edited
         // before it can be accepted
  Done   // Lines are complete
};

//----------------------------------------------------------------------
// Exit Type for inferior processes
//----------------------------------------------------------------------
typedef enum ExitType {
  eExitTypeInvalid,
  eExitTypeExit,   // The exit status represents the return code from normal
                   // program exit (i.e. WIFEXITED() was true)
  eExitTypeSignal, // The exit status represents the signal number that caused
                   // the program to exit (i.e. WIFSIGNALED() was true)
  eExitTypeStop,   // The exit status represents the stop signal that caused the
                   // program to exit (i.e. WIFSTOPPED() was true)
} ExitType;

//----------------------------------------------------------------------
// Boolean result of running a Type Validator
//----------------------------------------------------------------------
enum class TypeValidatorResult : bool { Success = true, Failure = false };

//----------------------------------------------------------------------
// Enumerations that can be used to specify scopes types when looking up
// types.
//----------------------------------------------------------------------
enum class CompilerContextKind {
  Invalid = 0,
  TranslationUnit,
  Module,
  Namespace,
  Class,
  Structure,
  Union,
  Function,
  Variable,
  Enumeration,
  Typedef
};

} // namespace lldb_private

#endif // LLDB_lldb_private_enumerations_h_
