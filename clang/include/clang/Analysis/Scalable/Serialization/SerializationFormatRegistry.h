//===- SerializationFormatRegistry.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Registry for SerializationFormats, and some helper functions.
//
// To register some custom serialization format, you will need to add some
// declarations and defintions.
//
// Insert this code to the header file:
//
//   namespace llvm {
//   extern template class CLANG_TEMPLATE_ABI
//     Registry<clang::ssaf::MyFormat::FormatInfo>;
//   } // namespace llvm
//
// Insert this declaration to the MyFormat class:
//
//   using FormatInfo = FormatInfoEntry<SerializerFn, DeserializerFn>;
//
// Insert this code to the cpp file:
//
//   LLVM_INSTANTIATE_REGISTRY(llvm::Registry<MyFormat::FormatInfo>)
//
//   static SerializationFormatRegistry::Add<MyFormat>
//     RegisterFormat("MyFormat", "My awesome serialization format");
//
// Then implement the formatter for the specific analysis and register the
// format info for it:
//
//   namespace {
//   using FormatInfo = MyFormat::FormatInfo;
//   struct MyAnalysisFormatInfo final : FormatInfo {
//     MyAnalysisFormatInfo() : FormatInfo{
//               SummaryName("MyAnalysis"),
//               serializeMyAnalysis,
//               deserializeMyAnalysis,
//           } {}
//   };
//   } // namespace
//
//   static llvm::Registry<FormatInfo>::Add<MyAnalysisFormatInfo>
//       RegisterFormatInfo(
//         "MyAnalysisFormatInfo",
//         "The MyFormat format info implementation for MyAnalysis"
//       );
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_ANALYSIS_SCALABLE_SERIALIZATION_SERIALIZATION_FORMAT_REGISTRY_H
#define CLANG_ANALYSIS_SCALABLE_SERIALIZATION_SERIALIZATION_FORMAT_REGISTRY_H

#include "clang/Analysis/Scalable/Serialization/SerializationFormat.h"
#include "clang/Support/Compiler.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Registry.h"

namespace clang::ssaf {

/// Check if a SerializationFormat was registered with a given name.
bool isFormatRegistered(llvm::StringRef FormatName);

/// Try to instantiate a SerializationFormat with a given name.
/// This might return null if the construction of the desired
/// SerializationFormat failed.
/// It's a fatal error if there is no format registered with the name.
std::unique_ptr<SerializationFormat> makeFormat(llvm::StringRef FormatName);

// Registry for adding new SerializationFormat implementations.
using SerializationFormatRegistry = llvm::Registry<SerializationFormat>;

} // namespace clang::ssaf

namespace llvm {
extern template class CLANG_TEMPLATE_ABI
    Registry<clang::ssaf::SerializationFormat>;
} // namespace llvm

#endif // CLANG_ANALYSIS_SCALABLE_SERIALIZATION_SERIALIZATION_FORMAT_REGISTRY_H
