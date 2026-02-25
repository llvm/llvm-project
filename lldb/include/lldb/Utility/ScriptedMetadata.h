//===-- ScriptedMetadata.h ------------------------------------ -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_SCRIPTEDMETADATA_H
#define LLDB_UTILITY_SCRIPTEDMETADATA_H

#include "lldb/Utility/ProcessInfo.h"
#include "lldb/Utility/StreamString.h"
#include "lldb/Utility/StructuredData.h"
#include "llvm/ADT/Hashing.h"

namespace lldb_private {
class ScriptedMetadata {
public:
  ScriptedMetadata(llvm::StringRef class_name,
                   StructuredData::DictionarySP dict_sp)
      : m_class_name(class_name.data()), m_args_sp(dict_sp) {}

  ScriptedMetadata(const ProcessInfo &process_info) {
    lldb::ScriptedMetadataSP metadata_sp = process_info.GetScriptedMetadata();
    if (metadata_sp) {
      m_class_name = metadata_sp->GetClassName();
      m_args_sp = metadata_sp->GetArgsSP();
    }
  }

  ScriptedMetadata(const ScriptedMetadata &other)
      : m_class_name(other.m_class_name), m_args_sp(other.m_args_sp) {}

  explicit operator bool() const { return !m_class_name.empty(); }

  llvm::StringRef GetClassName() const { return m_class_name; }
  StructuredData::DictionarySP GetArgsSP() const { return m_args_sp; }

  /// Get a unique identifier for this metadata based on its contents.
  /// The ID is computed from the class name and arguments dictionary,
  /// not from the pointer address, so two metadata objects with the same
  /// contents will have the same ID.
  uint32_t GetID() const {
    if (m_class_name.empty())
      return 0;

    // Hash the class name.
    llvm::hash_code hash = llvm::hash_value(m_class_name);

    // Hash the arguments dictionary if present.
    if (m_args_sp) {
      StreamString ss;
      m_args_sp->GetDescription(ss);
      hash = llvm::hash_combine(hash, llvm::hash_value(ss.GetData()));
    }

    // Return the lower 32 bits of the hash.
    return static_cast<uint32_t>(hash);
  }

private:
  std::string m_class_name;
  StructuredData::DictionarySP m_args_sp;
};
} // namespace lldb_private

#endif // LLDB_UTILITY_SCRIPTEDMETADATA_H
