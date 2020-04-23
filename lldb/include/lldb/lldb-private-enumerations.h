//===-- lldb-private-enumerations.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_LLDB_PRIVATE_ENUMERATIONS_H
#define LLDB_LLDB_PRIVATE_ENUMERATIONS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatProviders.h"
#include "llvm/Support/raw_ostream.h"

namespace lldb_private {

// Thread Step Types
enum StepType {
  eStepTypeNone,
  eStepTypeTrace,     ///< Single step one instruction.
  eStepTypeTraceOver, ///< Single step one instruction, stepping over.
  eStepTypeInto,      ///< Single step into a specified context.
  eStepTypeOver,      ///< Single step over a specified context.
  eStepTypeOut,       ///< Single step out a specified context.
  eStepTypeScripted   ///< A step type implemented by the script interpreter.
};

// Address Types
enum AddressType {
  eAddressTypeInvalid = 0,
  eAddressTypeFile, ///< Address is an address as found in an object or symbol
                    /// file
  eAddressTypeLoad, ///< Address is an address as in the current target inferior
                    /// process
  eAddressTypeHost  ///< Address is an address in the process that is running
                    /// this code
};

// Address Class
//
// A way of classifying an address used for disassembling and setting
// breakpoints. Many object files can track exactly what parts of their object
// files are code, data and other information. This is of course above and
// beyond just looking at the section types. For example, code might contain PC
// relative data and the object file might be able to tell us that an address
// in code is data.
enum class AddressClass {
  eInvalid,
  eUnknown,
  eCode,
  eCodeAlternateISA,
  eData,
  eDebug,
  eRuntime
};

// Votes - Need a tri-state, yes, no, no opinion...
enum Vote { eVoteNo = -1, eVoteNoOpinion = 0, eVoteYes = 1 };

enum ArchitectureType {
  eArchTypeInvalid,
  eArchTypeMachO,
  eArchTypeELF,
  eArchTypeCOFF,
  kNumArchTypes
};

/// Settable state variable types.
///

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

enum VarSetOperationType {
  eVarSetOperationReplace,
  eVarSetOperationInsertBefore,
  eVarSetOperationInsertAfter,
  eVarSetOperationRemove,
  eVarSetOperationAppend,
  eVarSetOperationClear,
  eVarSetOperationAssign,
  eVarSetOperationInvalid
};

enum ArgumentRepetitionType {
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
};

enum SortOrder { eSortOrderNone, eSortOrderByAddress, eSortOrderByName };

// LazyBool is for boolean values that need to be calculated lazily. Values
// start off set to eLazyBoolCalculate, and then they can be calculated once
// and set to eLazyBoolNo or eLazyBoolYes.
enum LazyBool { eLazyBoolCalculate = -1, eLazyBoolNo = 0, eLazyBoolYes = 1 };

/// Instruction types
enum InstructionType {
  eInstructionTypeAny, // Support for any instructions at all (at least one)
  eInstructionTypePrologueEpilogue, // All prologue and epilogue instructions
                                    // that push and pop register values and
                                    // modify sp/fp
  eInstructionTypePCModifying,      // Any instruction that modifies the program
                                    // counter/instruction pointer
  eInstructionTypeAll               // All instructions of any kind

};

/// Format category entry types
enum FormatCategoryItem {
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
};

/// Expression execution policies
enum ExecutionPolicy {
  eExecutionPolicyOnlyWhenNeeded,
  eExecutionPolicyNever,
  eExecutionPolicyAlways,
  eExecutionPolicyTopLevel // used for top-level code
};

// Synchronicity behavior of scripted commands
enum ScriptedCommandSynchronicity {
  eScriptedCommandSynchronicitySynchronous,
  eScriptedCommandSynchronicityAsynchronous,
  eScriptedCommandSynchronicityCurrentValue // use whatever the current
                                            // synchronicity is
};

// Verbosity mode of "po" output
enum LanguageRuntimeDescriptionDisplayVerbosity {
  eLanguageRuntimeDescriptionDisplayVerbosityCompact, // only print the
                                                      // description string, if
                                                      // any
  eLanguageRuntimeDescriptionDisplayVerbosityFull,    // print the full-blown
                                                      // output
};

// Loading modules from memory
enum MemoryModuleLoadLevel {
  eMemoryModuleLoadLevelMinimal,  // Load sections only
  eMemoryModuleLoadLevelPartial,  // Load function bounds but no symbols
  eMemoryModuleLoadLevelComplete, // Load sections and all symbols
};

// Result enums for when reading multiple lines from IOHandlers
enum class LineStatus {
  Success, // The line that was just edited if good and should be added to the
           // lines
  Status,  // There is an error with the current line and it needs to be
           // re-edited
           // before it can be accepted
  Done     // Lines are complete
};

// Boolean result of running a Type Validator
enum class TypeValidatorResult : bool { Success = true, Failure = false };

// Enumerations that can be used to specify scopes types when looking up types.
enum class CompilerContextKind : uint16_t {
  Invalid = 0,
  TranslationUnit = 1,
  Module = 1 << 1,
  Namespace = 1 << 2,
  Class = 1 << 3,
  Struct = 1 << 4,
  Union = 1 << 5,
  Function = 1 << 6,
  Variable = 1 << 7,
  Enum = 1 << 8,
  Typedef = 1 << 9,

  Any = 1 << 15,
  /// Match 0..n nested modules.
  AnyModule = Any | Module,
  /// Match any type.
  AnyType = Any | Class | Struct | Union | Enum | Typedef
};

// Enumerations that can be used to specify the kind of metric we're looking at
// when collecting stats.
enum StatisticKind {
  ExpressionSuccessful = 0,
  ExpressionFailure = 1,
  FrameVarSuccess = 2,
  FrameVarFailure = 3,
  StatisticMax = 4
};


inline std::string GetStatDescription(lldb_private::StatisticKind K) {
   switch (K) {
   case StatisticKind::ExpressionSuccessful:
     return "Number of expr evaluation successes";
   case StatisticKind::ExpressionFailure:
     return "Number of expr evaluation failures";
   case StatisticKind::FrameVarSuccess:
     return "Number of frame var successes";
   case StatisticKind::FrameVarFailure:
     return "Number of frame var failures";
   case StatisticKind::StatisticMax:
     return "";
   }
   llvm_unreachable("Statistic not registered!");
}

} // namespace lldb_private

namespace llvm {
template <> struct format_provider<lldb_private::Vote> {
  static void format(const lldb_private::Vote &V, llvm::raw_ostream &Stream,
                     StringRef Style) {
    switch (V) {
    case lldb_private::eVoteNo:
      Stream << "no";
      return;
    case lldb_private::eVoteNoOpinion:
      Stream << "no opinion";
      return;
    case lldb_private::eVoteYes:
      Stream << "yes";
      return;
    }
    Stream << "invalid";
  }
};
}

#endif // LLDB_LLDB_PRIVATE_ENUMERATIONS_H
