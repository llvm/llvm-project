//===------------------SharedCluster.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_SHAREDCLUSTER_H
#define LLDB_UTILITY_SHAREDCLUSTER_H

#include "lldb/Utility/LLDBAssert.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <memory>
#include <mutex>

namespace lldb_private {

template <class T>
class ClusterManager : public std::enable_shared_from_this<ClusterManager<T>> {
public:
  static std::shared_ptr<ClusterManager> Create() {
    return std::shared_ptr<ClusterManager>(new ClusterManager());
  }

  ~ClusterManager() {
    for (T *obj : m_objects)
      delete obj;
  }

  void ManageObject(T *new_object) {
    std::lock_guard<std::mutex> guard(m_mutex);
    assert(!llvm::is_contained(m_objects, new_object) &&
           "ManageObject called twice for the same object?");
    m_objects.push_back(new_object);
  }

  std::shared_ptr<T> GetSharedPointer(T *desired_object) {
    std::lock_guard<std::mutex> guard(m_mutex);
    auto this_sp = this->shared_from_this();
    if (!llvm::is_contained(m_objects, desired_object)) {
      lldbassert(false && "object not found in shared cluster when expected");
      desired_object = nullptr;
    }
    return {std::move(this_sp), desired_object};
  }

private:
  ClusterManager() : m_objects(), m_mutex() {}

  llvm::SmallVector<T *, 16> m_objects;
  std::mutex m_mutex;
};

} // namespace lldb_private

#endif // LLDB_UTILITY_SHAREDCLUSTER_H
