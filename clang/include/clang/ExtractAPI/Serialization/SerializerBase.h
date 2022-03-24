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

  /// The product name of API.
  ///
  /// Note: This should be used for populating metadata about the API.
  StringRef ProductName;

  APISerializerOption Options;

public:
  APISerializer() = delete;
  APISerializer(const APISerializer &) = delete;
  APISerializer(APISerializer &&) = delete;
  APISerializer &operator=(const APISerializer &) = delete;
  APISerializer &operator=(APISerializer &&) = delete;

protected:
  APISerializer(const APISet &API, StringRef ProductName,
                APISerializerOption Options = {})
      : API(API), ProductName(ProductName), Options(Options) {}

  virtual ~APISerializer() = default;
};

} // namespace extractapi
} // namespace clang

#endif // LLVM_CLANG_EXTRACTAPI_SERIALIZATION_SERIALIZERBASE_H
