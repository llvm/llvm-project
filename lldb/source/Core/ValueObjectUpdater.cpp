//===-- ValueObjectUpdater.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/ValueObjectUpdater.h"

using namespace lldb_private;

ValueObjectUpdater::ValueObjectUpdater(lldb::ValueObjectSP in_valobj_sp) {
  // If the user passes in a value object that is dynamic or synthetic, then
  // water it down to the static type.
  m_root_valobj_sp = in_valobj_sp->GetQualifiedRepresentationIfAvailable(
      lldb::eNoDynamicValues, false);
}

std::optional<lldb::ValueObjectSP> ValueObjectUpdater::GetSP() {
  lldb::ProcessSP process_sp = GetProcessSP();
  if (!process_sp)
    return {};

  const uint32_t current_stop_id = process_sp->GetLastNaturalStopID();
  if (current_stop_id == m_stop_id)
    return m_user_valobj_sp;

  m_stop_id = current_stop_id;

  if (!m_root_valobj_sp) {
    if (m_user_valobj_sp)
      m_user_valobj_sp.value().reset();
    return m_root_valobj_sp;
  }

  m_user_valobj_sp = m_root_valobj_sp;

  std::optional<lldb::ValueObjectSP> dynamic_sp =
      m_user_valobj_sp.value()->GetDynamicValue(lldb::eDynamicDontRunTarget);
  if (dynamic_sp)
    m_user_valobj_sp = dynamic_sp;

  std::optional<lldb::ValueObjectSP> synthetic_sp =
      m_user_valobj_sp.value()->GetSyntheticValue();
  if (synthetic_sp)
    m_user_valobj_sp = synthetic_sp;

  return m_user_valobj_sp;
}

lldb::ProcessSP ValueObjectUpdater::GetProcessSP() const {
  if (m_root_valobj_sp)
    return m_root_valobj_sp.value()->GetProcessSP();
  return lldb::ProcessSP();
}
