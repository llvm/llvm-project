//===-- ThreadSafeDenseMap.h ------------------------------------------*- C++
//-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ThreadSafeDenseMap_h_
#define liblldb_ThreadSafeDenseMap_h_

// C Includes
// C++ Includes

// Other libraries and framework includes
#include "llvm/ADT/DenseMap.h"

// Project includes
#include "lldb/Host/Mutex.h"

namespace lldb_private {

template <typename _KeyType, typename _ValueType> class ThreadSafeDenseMap {
public:
  typedef llvm::DenseMap<_KeyType, _ValueType> LLVMMapType;

  ThreadSafeDenseMap(unsigned map_initial_capacity = 0,
                     Mutex::Type mutex_type = Mutex::eMutexTypeNormal)
      : m_map(map_initial_capacity), m_mutex(mutex_type) {}

  void Insert(_KeyType k, _ValueType v) {
    Mutex::Locker locker(m_mutex);
    m_map.insert(std::make_pair(k, v));
  }

  void Erase(_KeyType k) {
    Mutex::Locker locker(m_mutex);
    m_map.erase(k);
  }

  _ValueType Lookup(_KeyType k) {
    Mutex::Locker locker(m_mutex);
    return m_map.lookup(k);
  }

  bool Lookup(_KeyType k, _ValueType &v) {
    Mutex::Locker locker(m_mutex);
    auto iter = m_map.find(k), end = m_map.end();
    if (iter == end)
      return false;
    v = iter->second;
    return true;
  }

  void Clear() {
    Mutex::Locker locker(m_mutex);
    m_map.clear();
  }

protected:
  LLVMMapType m_map;
  Mutex m_mutex;
};

} // namespace lldb_private

#endif // liblldb_ThreadSafeSTLMap_h_
