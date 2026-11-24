//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_HOST_XMLPARSER_H
#define LLDB_HOST_XMLPARSER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace lldb_private {
namespace xml {

struct Attribute {
  llvm::StringRef name;
  llvm::StringRef value;
};

class Handler {
public:
  virtual ~Handler();
  virtual void StartElement(llvm::StringRef name,
                            llvm::ArrayRef<Attribute> attributes) {}
  virtual void EndElement(llvm::StringRef name) {}
  /// Called once with all the character data between two tags, including
  /// expanded references and CDATA sections.
  virtual void Characters(llvm::StringRef text) {}
};

/// Parse an XML 1.0 document without validating it.
llvm::Error Parse(llvm::StringRef input, Handler &handler);

} // namespace xml
} // namespace lldb_private

#endif // LLDB_HOST_XMLPARSER_H
