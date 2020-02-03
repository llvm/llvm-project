//===------------------SharedCluster.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef utility_SharedCluster_h_
#define utility_SharedCluster_h_

#include "lldb/Utility/LLDBAssert.h"
#include "lldb/Utility/SharingPtr.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <mutex>

namespace lldb_private {

namespace imp {
template <typename T>
class shared_ptr_refcount : public lldb_private::imp::shared_count {
public:
  template <class Y>
  shared_ptr_refcount(Y *in) : shared_count(0), manager(in) {}

  shared_ptr_refcount() : shared_count(0) {}

  ~shared_ptr_refcount() override {}

  void on_zero_shared() override { manager->DecrementRefCount(); }

private:
  T *manager;
};

} // namespace imp

template <class T> class ClusterManager {
public:
  ClusterManager() : m_objects(), m_external_ref(0), m_mutex() {}

  ~ClusterManager() {
    for (T *obj : m_objects)
      delete obj;

    // Decrement refcount should have been called on this ClusterManager, and
    // it should have locked the mutex, now we will unlock it before we destroy
    // it...
    m_mutex.unlock();
  }

  void ManageObject(T *new_object) {
    std::lock_guard<std::mutex> guard(m_mutex);
    assert(!llvm::is_contained(m_objects, new_object) &&
           "ManageObject called twice for the same object?");
    m_objects.push_back(new_object);
  }

  typename lldb_private::SharingPtr<T> GetSharedPointer(T *desired_object) {
    {
      std::lock_guard<std::mutex> guard(m_mutex);
      m_external_ref++;
      if (!llvm::is_contained(m_objects, desired_object)) {
        lldbassert(false && "object not found in shared cluster when expected");
        desired_object = nullptr;
      }
    }
    return typename lldb_private::SharingPtr<T>(
        desired_object, new imp::shared_ptr_refcount<ClusterManager>(this));
  }

private:
  void DecrementRefCount() {
    m_mutex.lock();
    m_external_ref--;
    if (m_external_ref == 0)
      delete this;
    else
      m_mutex.unlock();
  }

  friend class imp::shared_ptr_refcount<ClusterManager>;

  llvm::SmallVector<T *, 16> m_objects;
  int m_external_ref;
  std::mutex m_mutex;
};

} // namespace lldb_private

#endif // utility_SharedCluster_h_
