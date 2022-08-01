//===--- ScanAndUpdateArgs.h - Util for CC1 Dependency Scanning -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_DRIVER_SCANANDUPDATEARGS_H
#define LLVM_CLANG_DRIVER_SCANANDUPDATEARGS_H

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
namespace cas {
class CASID;
}
} // namespace llvm

namespace clang {

class CASOptions;
class CompilerInvocation;
class DiagnosticConsumer;

namespace tooling {
namespace dependencies {
class DependencyScanningTool;
}
} // namespace tooling

namespace cc1depscand {
struct DepscanPrefixMapping {
  Optional<StringRef> NewSDKPath;
  Optional<StringRef> NewToolchainPath;
  SmallVector<StringRef> PrefixMap;
};
} // namespace cc1depscand

Expected<llvm::cas::CASID> scanAndUpdateCC1InlineWithTool(
    tooling::dependencies::DependencyScanningTool &Tool,
    DiagnosticConsumer &DiagsConsumer, raw_ostream *VerboseOS, const char *Exec,
    CompilerInvocation &Invocation, StringRef WorkingDirectory,
    const cc1depscand::DepscanPrefixMapping &PrefixMapping);

} // end namespace clang

#endif
