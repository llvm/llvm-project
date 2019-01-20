//===--- Modularize.h - Common definitions for Modularize -*- C++ -*-----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------------===//
///
/// \file
/// \brief Common definitions for Modularize.
///
//===--------------------------------------------------------------------===//

#ifndef MODULARIZE_H
#define MODULARIZE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <string>
#include <vector>

// Save the program name for error messages.
extern const char *Argv0;
// Save the command line for comments.
extern std::string CommandLine;

// Dependency types.
typedef llvm::SmallVector<std::string, 4> DependentsVector;
typedef llvm::StringMap<DependentsVector> DependencyMap;

// Global function declarations.

/// Create the module map file.
/// \param ModuleMapPath The path to the module map file to be generated.
/// \param HeaderFileNames The list of header files, absolute native paths.
/// \param ProblemFileNames The list of problem header files.
/// \param Dependencies Map of headers that depend on other headers.
/// \param HeaderPrefix Tells the code where the headers are, if they
///   aren's in the current directory, allowing the generator to strip
///   the leading, non-relative beginning of the header paths.
/// \brief RootModuleName If not empty, specifies that a root module
///   should be created with this name.
/// \returns True if successful.
bool createModuleMap(llvm::StringRef ModuleMapPath,
                     llvm::ArrayRef<std::string> HeaderFileNames,
                     llvm::ArrayRef<std::string> ProblemFileNames,
                     DependencyMap &Dependencies, llvm::StringRef HeaderPrefix,
                     llvm::StringRef RootModuleName);

#endif // MODULARIZE_H
