//=--- CommonBugCategories.h - Provides common issue categories -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_STATIC_ANALYZER_CHECKER_CATEGORIES_H
#define LLVM_CLANG_STATIC_ANALYZER_CHECKER_CATEGORIES_H

// Common strings used for the "category" of many static analyzer issues.
namespace clang {
  namespace ento {
    namespace categories {
      extern const char *CoreFoundationObjectiveC;
      extern const char *MemoryCoreFoundationObjectiveC;
      extern const char *UnixAPI;
    }
  }
}
#endif

