//===- ArgumentsAdjusters.h - Command line arguments adjuster ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares type ArgumentsAdjuster and functions to create several
// useful argument adjusters.
// ArgumentsAdjusters modify command line arguments obtained from a compilation
// database before they are used to run a frontend action.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_ARGUMENTSADJUSTERS_H
#define LLVM_CLANG_TOOLING_ARGUMENTSADJUSTERS_H

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include <functional>
#include <string>
#include <vector>

namespace clang {
namespace tooling {

/// \brief A sequence of command line arguments.
using CommandLineArguments = std::vector<std::string>;

/// \brief A prototype of a command line adjuster.
///
/// Command line argument adjuster is responsible for command line arguments
/// modification before the arguments are used to run a frontend action.
using ArgumentsAdjuster = std::function<CommandLineArguments(
    const CommandLineArguments &, StringRef Filename)>;

/// \brief Gets an argument adjuster that converts input command line arguments
/// to the "syntax check only" variant.
ArgumentsAdjuster getClangSyntaxOnlyAdjuster();

/// \brief Gets an argument adjuster which removes output-related command line
/// arguments.
ArgumentsAdjuster getClangStripOutputAdjuster();

/// \brief Gets an argument adjuster which removes dependency-file
/// related command line arguments.
ArgumentsAdjuster getClangStripDependencyFileAdjuster();

enum class ArgumentInsertPosition { BEGIN, END };

/// \brief Gets an argument adjuster which inserts \p Extra arguments in the
/// specified position.
ArgumentsAdjuster getInsertArgumentAdjuster(const CommandLineArguments &Extra,
                                            ArgumentInsertPosition Pos);

/// \brief Gets an argument adjuster which inserts an \p Extra argument in the
/// specified position.
ArgumentsAdjuster getInsertArgumentAdjuster(
    const char *Extra,
    ArgumentInsertPosition Pos = ArgumentInsertPosition::END);

/// \brief Gets an argument adjuster which adjusts the arguments in sequence
/// with the \p First adjuster and then with the \p Second one.
ArgumentsAdjuster combineAdjusters(ArgumentsAdjuster First,
                                   ArgumentsAdjuster Second);

} // namespace tooling
} // namespace clang

#endif // LLVM_CLANG_TOOLING_ARGUMENTSADJUSTERS_H
