//===-- ThreadSafeDenseSet.h ------------------------------------------*- C++
//-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ThreadSafeDenseSet_h_
#define liblldb_ThreadSafeDenseSet_h_

// C Includes
// C++ Includes

// Other libraries and framework includes
#include "llvm/ADT/DenseSet.h"

// Project includes
#include "lldb/Host/Mutex.h"

namespace lldb_private {

template <typename _ElementType> class ThreadSafeDenseSet {
public:
  typedef llvm::DenseSet<_ElementType> LLVMSetType;

  ThreadSafeDenseSet(unsigned set_initial_capacity = 0,
                     Mutex::Type mutex_type = Mutex::eMutexTypeNormal)
      : m_set(set_initial_capacity), m_mutex(mutex_type) {}

  void Insert(_ElementType e) {
    Mutex::Locker locker(m_mutex);
    m_set.insert(e);
  }

  void Erase(_ElementType e) {
    Mutex::Locker locker(m_mutex);
    m_set.erase(e);
  }

  bool Lookup(_ElementType e) {
    Mutex::Locker locker(m_mutex);
    return (m_set.count(e) > 0);
  }

  void Clear() {
    Mutex::Locker locker(m_mutex);
    m_set.clear();
  }

protected:
  LLVMSetType m_set;
  Mutex m_mutex;
};

} // namespace lldb_private

#endif // liblldb_ThreadSafeDenseSet_h_
