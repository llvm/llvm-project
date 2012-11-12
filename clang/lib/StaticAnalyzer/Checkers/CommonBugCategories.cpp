//=--- CommonBugCategories.cpp - Provides common issue categories -*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Common strings used for the "category" of many static analyzer issues.
namespace clang { namespace ento { namespace categories {

const char *CoreFoundationObjectiveC = "Core Foundation/Objective-C";
const char *MemoryCoreFoundationObjectiveC =
  "Memory (Core Foundation/Objective-C)";
const char *UnixAPI = "Unix API";
}}}

