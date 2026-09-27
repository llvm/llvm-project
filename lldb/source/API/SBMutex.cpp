//===-- SBMutex.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBMutex.h"
#include "lldb/Target/TargetAPIMutex.h"
#include "lldb/Utility/Instrumentation.h"
#include "lldb/lldb-forward.h"
#include <memory>
#include <mutex>
#include <variant>

using namespace lldb;
using namespace lldb_private;

/// Holds either a standalone std::recursive_mutex (default-constructed
/// SBMutex, no Target to resolve through) or a TargetAPIMutex (constructed
/// from a Target). Kept out of SBMutex.h since std::variant is a C++17
/// feature and the public SB headers must stay usable from a C++11 client.
class SBMutex::MutexVariant {
public:
  MutexVariant() : m_variant(std::in_place_type<std::recursive_mutex>) {}
  explicit MutexVariant(lldb::TargetSP target_sp)
      : m_variant(std::in_place_type<TargetAPIMutex>, std::move(target_sp)) {}

  void lock() {
    std::visit([](auto &mutex) { mutex.lock(); }, m_variant);
  }
  void unlock() {
    std::visit([](auto &mutex) { mutex.unlock(); }, m_variant);
  }
  bool try_lock() {
    return std::visit([](auto &mutex) { return mutex.try_lock(); }, m_variant);
  }

private:
  std::variant<std::recursive_mutex, TargetAPIMutex> m_variant;
};

SBMutex::SBMutex() : m_opaque_sp(std::make_shared<MutexVariant>()) {
  LLDB_INSTRUMENT_VA(this);
}

SBMutex::SBMutex(const SBMutex &rhs) : m_opaque_sp(rhs.m_opaque_sp) {
  LLDB_INSTRUMENT_VA(this);
}

const SBMutex &SBMutex::operator=(const SBMutex &rhs) {
  LLDB_INSTRUMENT_VA(this);

  m_opaque_sp = rhs.m_opaque_sp;
  return *this;
}

SBMutex::SBMutex(lldb::TargetSP target_sp)
    : m_opaque_sp(std::make_shared<MutexVariant>(target_sp)) {
  LLDB_INSTRUMENT_VA(this, target_sp);
}

SBMutex::~SBMutex() { LLDB_INSTRUMENT_VA(this); }

bool SBMutex::IsValid() const {
  LLDB_INSTRUMENT_VA(this);

  return static_cast<bool>(m_opaque_sp);
}

void SBMutex::lock() const {
  LLDB_INSTRUMENT_VA(this);

  if (m_opaque_sp)
    m_opaque_sp->lock();
}

void SBMutex::unlock() const {
  LLDB_INSTRUMENT_VA(this);

  if (m_opaque_sp)
    m_opaque_sp->unlock();
}

bool SBMutex::try_lock() const {
  LLDB_INSTRUMENT_VA(this);

  if (!m_opaque_sp)
    return false;
  return m_opaque_sp->try_lock();
}
