//===--- CompilationDatabasePluginRegistry.h - ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_COMPILATIONDATABASEPLUGINREGISTRY_H
#define LLVM_CLANG_TOOLING_COMPILATIONDATABASEPLUGINREGISTRY_H

#include "clang/Tooling/CompilationDatabase.h"
#include "llvm/Support/Registry.h"

namespace clang {
namespace tooling {

class CompilationDatabasePlugin;

typedef llvm::Registry<CompilationDatabasePlugin>
    CompilationDatabasePluginRegistry;

} // end namespace tooling
} // end namespace clang

#endif
