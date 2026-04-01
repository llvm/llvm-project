//===- GenInfo.h - Generator info -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TABLEGEN_GENINFO_H_
#define AIIR_TABLEGEN_GENINFO_H_

#include "aiir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include <functional>
#include <utility>

namespace llvm {
class RecordKeeper;
} // namespace llvm

namespace aiir {

/// Generator function to invoke.
using GenFunction =
    std::function<bool(const llvm::RecordKeeper &records, raw_ostream &os)>;

/// Structure to group information about a generator (argument to invoke via
/// aiir-tblgen, description, and generator function).
class GenInfo {
public:
  /// GenInfo constructor should not be invoked directly, instead use
  /// GenRegistration or registerGen.
  GenInfo(StringRef arg, StringRef description, GenFunction generator)
      : arg(arg), description(description), generator(std::move(generator)) {}

  /// Invokes the generator and returns whether the generator failed.
  bool invoke(const llvm::RecordKeeper &records, raw_ostream &os) const {
    assert(generator && "Cannot call generator with null generator");
    return generator(records, os);
  }

  /// Returns the command line option that may be passed to 'aiir-tblgen' to
  /// invoke this generator.
  StringRef getGenArgument() const { return arg; }

  /// Returns a description for the generator.
  StringRef getGenDescription() const { return description; }

private:
  // The argument with which to invoke the generator via aiir-tblgen.
  StringRef arg;

  // Description of the generator.
  StringRef description;

  // Generator function.
  GenFunction generator;
};

/// GenRegistration provides a global initializer that registers a generator
/// function.
///
/// Usage:
///
///   // At namespace scope.
///   static GenRegistration Print("print", "Print records", [](...){...});
struct GenRegistration {
  GenRegistration(StringRef arg, StringRef description,
                  const GenFunction &function);
};

} // namespace aiir

#endif // AIIR_TABLEGEN_GENINFO_H_
