//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_PROCESSADDRESS_H
#define LLDB_UTILITY_PROCESSADDRESS_H

#include "lldb/lldb-defines.h"
#include "lldb/lldb-types.h"
#include <optional>

namespace lldb_private {

/// An address in a process, qualified by an address space.
class ProcessAddress {
  lldb::addr_t m_value;
  lldb::addr_space_t m_addr_space = LLDB_DEFAULT_ADDRESS_SPACE_ID;
  /// If this has a value, then this is a thread specific address.
  std::optional<lldb::tid_t> m_tid;

public:
  /// Implicit so existing lldb::addr_t call sites keep working.
  ProcessAddress(lldb::addr_t load_addr) : m_value(load_addr) {}

  ProcessAddress(lldb::addr_t addr, lldb::addr_space_t addr_space,
                 std::optional<lldb::tid_t> tid = std::nullopt)
      : m_value(addr), m_addr_space(addr_space), m_tid(tid) {}

  bool IsInDefaultAddressSpace() const {
    return m_addr_space == LLDB_DEFAULT_ADDRESS_SPACE_ID;
  }

  lldb::addr_t GetValue() const { return m_value; }

  lldb::addr_space_t GetAddressSpace() const { return m_addr_space; }

  std::optional<lldb::tid_t> GetThreadID() const { return m_tid; }
};

} // namespace lldb_private

#endif // LLDB_UTILITY_PROCESSADDRESS_H
