//===-- lldb-private-types.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_LLDB_PRIVATE_TYPES_H
#define LLDB_LLDB_PRIVATE_TYPES_H

#include "lldb/lldb-types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include <type_traits>

namespace llvm {
namespace sys {
class DynamicLibrary;
}
}

namespace lldb_private {
class Platform;
class ExecutionContext;

typedef llvm::SmallString<256> PathSmallString;

typedef llvm::sys::DynamicLibrary (*LoadPluginCallbackType)(
    const lldb::DebuggerSP &debugger_sp, const FileSpec &spec, Status &error);

/// A type-erased pair of llvm::dwarf::SourceLanguageName and version.
struct SourceLanguage {
  SourceLanguage() = default;
  explicit SourceLanguage(lldb::LanguageType language_type);

  SourceLanguage(uint16_t name, uint32_t version)
      : name(name), version(version) {}

  explicit SourceLanguage(
      std::optional<std::pair<uint16_t, uint32_t>> name_vers)
      : name(name_vers ? name_vers->first : 0),
        version(name_vers ? name_vers->second : 0) {}

  explicit operator bool() const { return name > 0; }

  lldb::LanguageType AsLanguageType() const;
  llvm::StringRef GetDescription() const;
  bool IsC() const;
  bool IsObjC() const;
  bool IsCPlusPlus() const;
  uint16_t name = 0;
  uint32_t version = 0;
};

struct OptionEnumValueElement {
  int64_t value;
  const char *string_value;
  const char *usage;
};

using OptionEnumValues = llvm::ArrayRef<OptionEnumValueElement>;

struct OptionValidator {
  virtual ~OptionValidator() = default;
  virtual bool IsValid(Platform &platform,
                       const ExecutionContext &target) const = 0;
  virtual const char *ShortConditionString() const = 0;
  virtual const char *LongConditionString() const = 0;
};

typedef struct type128 { uint64_t x[2]; } type128;
typedef struct type256 { uint64_t x[4]; } type256;

/// Functor that returns a ValueObjectSP for a variable given its name
/// and the StackFrame of interest. Used primarily in the Materializer
/// to refetch a ValueObject when the ExecutionContextScope changes.
using ValueObjectProviderTy =
    std::function<lldb::ValueObjectSP(ConstString, StackFrame *)>;

typedef void (*DebuggerDestroyCallback)(lldb::user_id_t debugger_id,
                                        void *baton);
typedef bool (*CommandOverrideCallbackWithResult)(
    void *baton, const char **argv, lldb_private::CommandReturnObject &result);
} // namespace lldb_private

#endif // LLDB_LLDB_PRIVATE_TYPES_H
