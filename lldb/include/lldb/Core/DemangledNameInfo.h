//===-- DemangledNameInfo.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_CORE_DEMANGLEDNAMEINFO_H
#define LLDB_CORE_DEMANGLEDNAMEINFO_H

#include "llvm/Demangle/ItaniumDemangle.h"
#include "llvm/Demangle/Utility.h"

#include <cstddef>
#include <utility>

namespace lldb_private {

/// Stores information about where certain portions of a demangled
/// function name begin and end.
struct DemangledNameInfo {
  /// A [start, end) pair for the function basename.
  /// The basename is the name without scope qualifiers
  /// and without template parameters. E.g.,
  /// \code{.cpp}
  ///    void foo::bar<int>::someFunc<float>(int) const &&
  ///                        ^       ^
  ///                      start    end
  /// \endcode
  std::pair<size_t, size_t> BasenameRange;

  /// A [start, end) pair for the function scope qualifiers.
  /// E.g., for
  /// \code{.cpp}
  ///    void foo::bar<int>::qux<float>(int) const &&
  ///         ^              ^
  ///       start           end
  /// \endcode
  std::pair<size_t, size_t> ScopeRange;

  /// Indicates the [start, end) of the function argument lits.
  /// E.g.,
  /// \code{.cpp}
  ///    int (*getFunc<float>(float, double))(int, int)
  ///                        ^              ^
  ///                      start           end
  /// \endcode
  std::pair<size_t, size_t> ArgumentsRange;

  /// Indicates the [start, end) of the function qualifiers
  /// (e.g., CV-qualifiers, reference qualifiers, requires clauses).
  ///
  /// E.g.,
  /// \code{.cpp}
  ///    void foo::bar<int>::qux<float>(int) const &&
  ///                                       ^        ^
  ///                                     start     end
  /// \endcode
  std::pair<size_t, size_t> QualifiersRange;

  /// Returns \c true if this object holds a valid basename range.
  bool hasBasename() const {
    return BasenameRange.second > BasenameRange.first &&
           BasenameRange.second > 0;
  }
};

/// An OutputBuffer which keeps a record of where certain parts of a
/// demangled name begin/end (e.g., basename, scope, argument list, etc.).
/// The tracking occurs during printing of the Itanium demangle tree.
///
/// Usage:
/// \code{.cpp}
///
/// Node *N = mangling_parser.parseType();
///
/// TrackingOutputBuffer buffer;
/// N->printLeft(OB);
///
/// assert (buffer.NameInfo.hasBasename());
///
/// \endcode
struct TrackingOutputBuffer : public llvm::itanium_demangle::OutputBuffer {
  using OutputBuffer::OutputBuffer;

  /// Holds information about the demangled name that is
  /// being printed into this buffer.
  DemangledNameInfo NameInfo;

  void printLeft(const llvm::itanium_demangle::Node &N) override;
  void printRight(const llvm::itanium_demangle::Node &N) override;

private:
  void printLeftImpl(const llvm::itanium_demangle::FunctionType &N);
  void printRightImpl(const llvm::itanium_demangle::FunctionType &N);

  void printLeftImpl(const llvm::itanium_demangle::FunctionEncoding &N);
  void printRightImpl(const llvm::itanium_demangle::FunctionEncoding &N);

  void printLeftImpl(const llvm::itanium_demangle::NestedName &N);
  void printLeftImpl(const llvm::itanium_demangle::NameWithTemplateArgs &N);

  /// Called whenever we start printing a function type in the Itanium
  /// mangling scheme. Examples include \ref FunctionEncoding, \ref
  /// FunctionType, etc.
  ///
  /// \returns A ScopedOverride which will update the nesting depth of
  /// currently printed function types on destruction.
  [[nodiscard]] llvm::itanium_demangle::ScopedOverride<unsigned>
  enterFunctionTypePrinting();

  /// Returns \c true if we're not printing any nested function types,
  /// just a \ref FunctionEncoding in the Itanium mangling scheme.
  bool isPrintingTopLevelFunctionType() const;

  /// If this object \ref shouldTrack, then update the end of
  /// the basename range to the current \c OB position.
  void updateBasenameEnd();

  /// If this object \ref shouldTrack, then update the beginning
  /// of the scope range to the current \c OB position.
  void updateScopeStart();

  /// If this object \ref shouldTrack, then update the end of
  /// the scope range to the current \c OB position.
  void updateScopeEnd();

  /// Returns \c true if the members of this object can be
  /// updated. E.g., when we're printing nested template
  /// arguments, we don't need to be tracking basename
  /// locations.
  bool shouldTrack() const;

  /// Helpers called to track beginning and end of the function
  /// arguments.
  void finalizeArgumentEnd();
  void finalizeStart();
  void finalizeEnd();
  void finalizeQualifiersStart();
  void finalizeQualifiersEnd();

  /// Helper used in the finalize APIs.
  bool canFinalize() const;

  /// Incremented each time we start printing a function type node
  /// in the Itanium mangling scheme (e.g., \ref FunctionEncoding
  /// or \ref FunctionType).
  unsigned FunctionPrintingDepth = 0;
};
} // namespace lldb_private

#endif // LLDB_CORE_DEMANGLEDNAMEINFO_H
