//===-- swift-lldb-forward.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_swift_lldb_forward_h_
#define LLDB_swift_lldb_forward_h_

#include "lldb-forward.h"

#if defined(__cplusplus)

#include "lldb/Utility/SharingPtr.h"

// lldb forward declarations
namespace lldb_private {
class SwiftASTContext;
class SwiftLanguageRuntime;
class SwiftREPL;
} // namespace lldb_private

// lldb forward declarations
namespace lldb {
typedef std::shared_ptr<lldb_private::SwiftASTContext> SwiftASTContextSP;
} // namespace lldb

// llvm forward declarations
namespace llvm {

} // namespace llvm

#endif // #if defined(__cplusplus)
#endif // LLDB_swift_lldb_forward_h_
