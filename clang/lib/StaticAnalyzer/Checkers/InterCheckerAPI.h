//==--- InterCheckerAPI.h ---------------------------------------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This file allows introduction of checker dependencies. It contains APIs for
// inter-checker communications.
//===----------------------------------------------------------------------===//

#ifndef INTERCHECKERAPI_H_
#define INTERCHECKERAPI_H_
namespace clang {
namespace ento {

/// Register the checker which evaluates CString API calls.
void registerCStringCheckerBasic(CheckerManager &Mgr);

}}
#endif /* INTERCHECKERAPI_H_ */
