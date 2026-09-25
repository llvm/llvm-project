//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_INTERPRETER_SCRIPTEDINSTANCEREGISTRY_H
#define LLDB_INTERPRETER_SCRIPTEDINSTANCEREGISTRY_H

#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/StructuredData.h"
#include "lldb/Utility/UUID.h"
#include "lldb/lldb-defines.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-forward.h"
#include "lldb/lldb-types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace lldb_private {

struct ScriptedInstanceInfo {
  uint64_t id = 0;
  UUID uuid;
  lldb::addr_t object_address = LLDB_INVALID_ADDRESS;
  /// Plugin names are string literals, so this never dangles.
  llvm::StringRef plugin_name;
  std::string class_name;
  FileSpec source_path;
  StructuredData::DictionarySP args_sp;
};

struct ScriptedInstanceGroup {
  std::string class_name;
  lldb::ScriptedExtension extension = lldb::eScriptedExtensionInvalid;
  FileSpec source_path;
  std::vector<ScriptedInstanceInfo> instances;

  StructuredData::DictionarySP ToStructuredData() const;

  void Dump(Stream &s, const Debugger &debugger, bool use_color) const;
};

/// Entries are snapshots taken when the object is created, so listing them
/// never calls back into an interface that might be mid-destruction on
/// another thread.
class ScriptedInstanceRegistry {
public:
  uint64_t Add(ScriptedInstanceInfo info) {
    std::lock_guard<std::mutex> guard(m_mutex);
    const uint64_t id = m_next_id++;
    info.id = id;
    info.uuid = UUID::Generate();
    m_instances[id] = std::move(info);
    return id;
  }

  void Remove(uint64_t id) {
    std::lock_guard<std::mutex> guard(m_mutex);
    m_instances.erase(id);
  }

  std::vector<ScriptedInstanceInfo> GetInstances() const {
    std::vector<ScriptedInstanceInfo> instances = CopyInstances();
    llvm::sort(instances,
               [](const ScriptedInstanceInfo &lhs,
                  const ScriptedInstanceInfo &rhs) { return lhs.id < rhs.id; });
    return instances;
  }

  std::vector<ScriptedInstanceGroup> GetInstanceGroups(
      llvm::ArrayRef<lldb::ScriptedExtension> extensions = {}) const;

private:
  std::vector<ScriptedInstanceInfo> CopyInstances() const {
    std::lock_guard<std::mutex> guard(m_mutex);
    std::vector<ScriptedInstanceInfo> instances;
    instances.reserve(m_instances.size());
    for (const auto &entry : m_instances)
      instances.push_back(entry.second);
    return instances;
  }

  mutable std::mutex m_mutex;
  uint64_t m_next_id = 1;
  llvm::DenseMap<uint64_t, ScriptedInstanceInfo> m_instances;
};

} // namespace lldb_private

#endif // LLDB_INTERPRETER_SCRIPTEDINSTANCEREGISTRY_H
