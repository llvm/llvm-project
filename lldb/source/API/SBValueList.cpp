//===-- SBValueList.cpp -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBValueList.h"
#include "lldb/API/SBStream.h"
#include "lldb/API/SBValue.h"
#include "lldb/Core/ValueObjectList.h"
#include "lldb/Utility/Log.h"

#include <vector>

using namespace lldb;
using namespace lldb_private;

class ValueListImpl {
public:
  ValueListImpl() : m_values() {}

  ValueListImpl(const ValueListImpl &rhs) : m_values(rhs.m_values) {}

  ValueListImpl &operator=(const ValueListImpl &rhs) {
    if (this == &rhs)
      return *this;
    m_values = rhs.m_values;
    return *this;
  }

  uint32_t GetSize() { return m_values.size(); }

  void Append(const lldb::SBValue &sb_value) { m_values.push_back(sb_value); }

  void Append(const ValueListImpl &list) {
    for (auto val : list.m_values)
      Append(val);
  }

  lldb::SBValue GetValueAtIndex(uint32_t index) {
    if (index >= GetSize())
      return lldb::SBValue();
    return m_values[index];
  }

  lldb::SBValue FindValueByUID(lldb::user_id_t uid) {
    for (auto val : m_values) {
      if (val.IsValid() && val.GetID() == uid)
        return val;
    }
    return lldb::SBValue();
  }

  lldb::SBValue GetFirstValueByName(const char *name) const {
    if (name) {
      for (auto val : m_values) {
        if (val.IsValid() && val.GetName() && strcmp(name, val.GetName()) == 0)
          return val;
      }
    }
    return lldb::SBValue();
  }

private:
  std::vector<lldb::SBValue> m_values;
};

SBValueList::SBValueList() : m_opaque_up() {}

SBValueList::SBValueList(const SBValueList &rhs) : m_opaque_up() {
  Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_API));

  if (rhs.IsValid())
    m_opaque_up.reset(new ValueListImpl(*rhs));

  if (log) {
    log->Printf(
        "SBValueList::SBValueList (rhs.ap=%p) => this.ap = %p",
        static_cast<void *>(rhs.IsValid() ? rhs.m_opaque_up.get() : NULL),
        static_cast<void *>(m_opaque_up.get()));
  }
}

SBValueList::SBValueList(const ValueListImpl *lldb_object_ptr) : m_opaque_up() {
  Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_API));

  if (lldb_object_ptr)
    m_opaque_up.reset(new ValueListImpl(*lldb_object_ptr));

  if (log) {
    log->Printf("SBValueList::SBValueList (lldb_object_ptr=%p) => this.ap = %p",
                static_cast<const void *>(lldb_object_ptr),
                static_cast<void *>(m_opaque_up.get()));
  }
}

SBValueList::~SBValueList() {}

bool SBValueList::IsValid() const { return (m_opaque_up != NULL); }

void SBValueList::Clear() { m_opaque_up.reset(); }

const SBValueList &SBValueList::operator=(const SBValueList &rhs) {
  if (this != &rhs) {
    if (rhs.IsValid())
      m_opaque_up.reset(new ValueListImpl(*rhs));
    else
      m_opaque_up.reset();
  }
  return *this;
}

ValueListImpl *SBValueList::operator->() { return m_opaque_up.get(); }

ValueListImpl &SBValueList::operator*() { return *m_opaque_up; }

const ValueListImpl *SBValueList::operator->() const {
  return m_opaque_up.get();
}

const ValueListImpl &SBValueList::operator*() const { return *m_opaque_up; }

void SBValueList::Append(const SBValue &val_obj) {
  CreateIfNeeded();
  m_opaque_up->Append(val_obj);
}

void SBValueList::Append(lldb::ValueObjectSP &val_obj_sp) {
  if (val_obj_sp) {
    CreateIfNeeded();
    m_opaque_up->Append(SBValue(val_obj_sp));
  }
}

void SBValueList::Append(const lldb::SBValueList &value_list) {
  if (value_list.IsValid()) {
    CreateIfNeeded();
    m_opaque_up->Append(*value_list);
  }
}

SBValue SBValueList::GetValueAtIndex(uint32_t idx) const {
  Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_API));

  // if (log)
  //    log->Printf ("SBValueList::GetValueAtIndex (uint32_t idx) idx = %d",
  //    idx);

  SBValue sb_value;
  if (m_opaque_up)
    sb_value = m_opaque_up->GetValueAtIndex(idx);

  if (log) {
    SBStream sstr;
    sb_value.GetDescription(sstr);
    log->Printf("SBValueList::GetValueAtIndex (this.ap=%p, idx=%d) => SBValue "
                "(this.sp = %p, '%s')",
                static_cast<void *>(m_opaque_up.get()), idx,
                static_cast<void *>(sb_value.GetSP().get()), sstr.GetData());
  }

  return sb_value;
}

uint32_t SBValueList::GetSize() const {
  Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_API));

  // if (log)
  //    log->Printf ("SBValueList::GetSize ()");

  uint32_t size = 0;
  if (m_opaque_up)
    size = m_opaque_up->GetSize();

  if (log)
    log->Printf("SBValueList::GetSize (this.ap=%p) => %d",
                static_cast<void *>(m_opaque_up.get()), size);

  return size;
}

void SBValueList::CreateIfNeeded() {
  if (m_opaque_up == NULL)
    m_opaque_up.reset(new ValueListImpl());
}

SBValue SBValueList::FindValueObjectByUID(lldb::user_id_t uid) {
  SBValue sb_value;
  if (m_opaque_up)
    sb_value = m_opaque_up->FindValueByUID(uid);
  return sb_value;
}

SBValue SBValueList::GetFirstValueByName(const char *name) const {
  SBValue sb_value;
  if (m_opaque_up)
    sb_value = m_opaque_up->GetFirstValueByName(name);
  return sb_value;
}

void *SBValueList::opaque_ptr() { return m_opaque_up.get(); }

ValueListImpl &SBValueList::ref() {
  CreateIfNeeded();
  return *m_opaque_up;
}
