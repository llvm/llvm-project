//===-- LoopConvert/LoopConvert.h - C++11 for-loop migration ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file provides the definition of the UseNullptrTransform
/// class which is the main interface to the use-nullptr transform
/// that tries to make use of nullptr where possible.
///
//===----------------------------------------------------------------------===//
#ifndef LLVM_TOOLS_CLANG_TOOLS_EXTRA_CPP11_MIGRATE_USE_NULLPTR_H
#define LLVM_TOOLS_CLANG_TOOLS_EXTRA_CPP11_MIGRATE_USE_NULLPTR_H

#include "Core/Transform.h"
#include "llvm/Support/Compiler.h" // For LLVM_OVERRIDE

/// \brief Subclass of Transform that transforms null pointer constants into
/// C++11's nullptr keyword where possible.
class UseNullptrTransform : public Transform {
public:
  UseNullptrTransform() : Transform("UseNullptr") {}

  /// \see Transform::run().
  virtual int apply(const FileContentsByPath &InputStates,
                    RiskLevel MaxRiskLEvel,
                    const clang::tooling::CompilationDatabase &Database,
                    const std::vector<std::string> &SourcePaths,
                    FileContentsByPath &ResultStates) LLVM_OVERRIDE;
};

#endif // LLVM_TOOLS_CLANG_TOOLS_EXTRA_CPP11_MIGRATE_USE_NULLPTR_H
