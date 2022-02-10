//===- SymbolGraph/Serialization.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Defines the SymbolGraph serializer and parser.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SYMBOLGRAPH_SERIALIZATION_H
#define LLVM_CLANG_SYMBOLGRAPH_SERIALIZATION_H

#include "clang/SymbolGraph/API.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/VersionTuple.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace symbolgraph {

using namespace llvm::json;

struct SerializerOption {
  bool Compact;
};

class Serializer {
public:
  Serializer(const API &API, SerializerOption Options = {})
      : API(API), Options(Options) {}

  Object serialize();
  void serialize(raw_ostream &os);

private:
  Object serializeMetadata() const;
  Object serializeModule() const;
  Optional<Object> serializeAPIRecord(const APIRecord &Record) const;
  void serializeGlobalRecord(const GlobalRecord &Record);

  bool shouldSkip(const APIRecord &Record) const;

  const API &API;
  SerializerOption Options;
  Array Symbols;
  Array Relationships;

  static const VersionTuple FormatVersion;
};

} // namespace symbolgraph
} // namespace clang

#endif // LLVM_CLANG_SYMBOLGRAPH_SERIALIZATION_H
