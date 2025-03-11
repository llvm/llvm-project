//===-- Demangle.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_CORE_DEMANGLE_H
#define LLDB_CORE_DEMANGLE_H

#include "llvm/Demangle/ItaniumDemangle.h"
#include "llvm/Demangle/Utility.h"

#include <cstddef>
#include <utility>

namespace lldb_private {

struct TrackingOutputBuffer;

// Stores information about parts of a demangled function name.
struct FunctionNameInfo {
  /// A [start, end) pair for the function basename.
  /// The basename is the name without scope qualifiers
  /// and without template parameters. E.g.,
  /// \code{.cpp}
  ///    void foo::bar<int>::someFunc<float>(int) const &&
  ///                        ^       ^
  ///                      start    end
  /// \endcode
  std::pair<size_t, size_t> BasenameLocs;

  /// A [start, end) pair for the function scope qualifiers.
  /// E.g., for
  /// \code{.cpp}
  ///    void foo::bar<int>::qux<float>(int) const &&
  ///         ^              ^
  ///       start           end
  /// \endcode
  std::pair<size_t, size_t> ScopeLocs;

  /// Indicates the [start, end) of the function argument lits.
  /// E.g.,
  /// \code{.cpp}
  ///    int (*getFunc<float>(float, double))(int, int)
  ///                        ^              ^
  ///                      start           end
  /// \endcode
  std::pair<size_t, size_t> ArgumentLocs;

  bool startedPrintingArguments() const;
  bool shouldTrack(TrackingOutputBuffer &OB) const;
  bool canFinalize(TrackingOutputBuffer &OB) const;
  void updateBasenameEnd(TrackingOutputBuffer &OB);
  void updateScopeStart(TrackingOutputBuffer &OB);
  void updateScopeEnd(TrackingOutputBuffer &OB);
  void finalizeArgumentEnd(TrackingOutputBuffer &OB);
  void finalizeStart(TrackingOutputBuffer &OB);
  void finalizeEnd(TrackingOutputBuffer &OB);
  bool hasBasename() const;
};

struct TrackingOutputBuffer : public llvm::itanium_demangle::OutputBuffer {
  using OutputBuffer::OutputBuffer;

  FunctionNameInfo FunctionInfo;
  unsigned FunctionPrintingDepth = 0;

  [[nodiscard]] llvm::itanium_demangle::ScopedOverride<unsigned>
  enterFunctionTypePrinting();

  bool isPrintingTopLevelFunctionType() const;

  void printLeft(const llvm::itanium_demangle::Node &N) override;
  void printRight(const llvm::itanium_demangle::Node &N) override;

private:
  void printLeftImpl(const llvm::itanium_demangle::FunctionType &N);
  void printRightImpl(const llvm::itanium_demangle::FunctionType &N);

  void printLeftImpl(const llvm::itanium_demangle::FunctionEncoding &N);
  void printRightImpl(const llvm::itanium_demangle::FunctionEncoding &N);

  void printLeftImpl(const llvm::itanium_demangle::NestedName &N);
  void printLeftImpl(const llvm::itanium_demangle::NameWithTemplateArgs &N);
};
} // namespace lldb_private

#endif // LLDB_CORE_DEMANGLE_H
