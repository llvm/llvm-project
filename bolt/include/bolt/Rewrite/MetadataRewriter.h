//===- bolt/Rewrite/MetadataRewriter.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Interface for reading and updating metadata in a file.
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_REWRITE_METADATA_REWRITER_H
#define BOLT_REWRITE_METADATA_REWRITER_H

#include "bolt/Core/BinaryContext.h"
#include "llvm/Support/Error.h"

namespace llvm {
namespace bolt {

/// Base class for handling file sections with metadata. In this context,
/// metadata encompasses a wide range of data that references code and other
/// data. Such metadata may or may not have an impact on program execution.
/// Examples include: debug information, unwind information, exception handling
/// tables, etc.
//
/// The metadata can occupy a section (e.g. .note.stapsdt), span a number of
/// sections (e.g.,  DWARF debug info), or exist as subsection of another
/// section in the binary (e.g., static-key jump tables embedded in .rodata
/// section in the Linux Kernel).
class MetadataRewriter {
  /// The name of the data type handled by an instance of this class.
  StringRef Name;

protected:
  /// Provides access to the binary context.
  BinaryContext &BC;

  MetadataRewriter(StringRef Name, BinaryContext &BC) : Name(Name), BC(BC) {}

public:
  virtual ~MetadataRewriter() = default;

  /// Return name for the rewriter.
  StringRef getName() const { return Name; }

  /// Run initialization after the binary is read and sections are identified,
  /// but before functions are discovered.
  virtual Error sectionInitializer() { return Error::success(); }

  /// Interface for modifying/annotating functions in the binary based on the
  /// contents of the section. Functions are in pre-cfg state.
  virtual Error preCFGInitializer() { return Error::success(); }

  /// Run the rewriter once the functions are in CFG state.
  virtual Error postCFGInitializer() { return Error::success(); }

  /// Run the pass before the binary is emitted.
  virtual Error preEmitFinalizer() { return Error::success(); }

  /// Finalize section contents based on the new context after the new code is
  /// emitted.
  virtual Error postEmitFinalizer() { return Error::success(); }
};

} // namespace bolt
} // namespace llvm

#endif // BOLT_REWRITE_METADATA_REWRITER_H
