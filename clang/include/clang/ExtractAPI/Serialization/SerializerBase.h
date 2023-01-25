//===- ExtractAPI/Serialization/SerializerBase.h ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the ExtractAPI APISerializer interface.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_EXTRACTAPI_SERIALIZATION_SERIALIZERBASE_H
#define LLVM_CLANG_EXTRACTAPI_SERIALIZATION_SERIALIZERBASE_H

#include "clang/ExtractAPI/API.h"
#include "clang/ExtractAPI/APIIgnoresList.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace extractapi {

/// Common options to customize the serializer output.
struct APISerializerOption {
  /// Do not include unnecessary whitespaces to save space.
  bool Compact;
};

/// The base interface of serializers for API information.
class APISerializer {
public:
  /// Serialize the API information to \p os.
  virtual void serialize(raw_ostream &os) = 0;

protected:
  const APISet &API;

  /// The list of symbols to ignore.
  ///
  /// Note: This should be consulted before emitting a symbol.
  const APIIgnoresList &IgnoresList;

  APISerializerOption Options;

public:
  APISerializer() = delete;
  APISerializer(const APISerializer &) = delete;
  APISerializer(APISerializer &&) = delete;
  APISerializer &operator=(const APISerializer &) = delete;
  APISerializer &operator=(APISerializer &&) = delete;

protected:
  APISerializer(const APISet &API, const APIIgnoresList &IgnoresList,
                APISerializerOption Options = {})
      : API(API), IgnoresList(IgnoresList), Options(Options) {}

  virtual ~APISerializer() = default;
};

} // namespace extractapi
} // namespace clang

#endif // LLVM_CLANG_EXTRACTAPI_SERIALIZATION_SERIALIZERBASE_H
