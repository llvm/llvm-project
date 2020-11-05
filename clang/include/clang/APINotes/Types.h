//===-- Types.h - API Notes Data Types --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_APINOTES_TYPES_H
#define LLVM_CLANG_APINOTES_TYPES_H

namespace clang {
namespace api_notes {
enum class RetainCountConventionKind {
  None,
  CFReturnsRetained,
  CFReturnsNotRetained,
  NSReturnsRetained,
  NSReturnsNotRetained,
};

/// The payload for an enum_extensibility attribute. This is a tri-state rather
/// than just a boolean because the presence of the attribute indicates
/// auditing.
enum class EnumExtensibilityKind {
  None,
  Open,
  Closed,
};

/// The kind of a swift_wrapper/swift_newtype.
enum class SwiftNewTypeKind {
  None,
  Struct,
  Enum,
};
} // namespace api_notes
} // namespace clang

#endif
