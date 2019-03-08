//===-- SBAddress.cpp -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBAddress.h"
#include "SBReproducerPrivate.h"
#include "Utils.h"
#include "lldb/API/SBProcess.h"
#include "lldb/API/SBSection.h"
#include "lldb/API/SBStream.h"
#include "lldb/Core/Address.h"
#include "lldb/Core/Module.h"
#include "lldb/Symbol/LineEntry.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/StreamString.h"

using namespace lldb;
using namespace lldb_private;

SBAddress::SBAddress() : m_opaque_up(new Address()) {
  LLDB_RECORD_CONSTRUCTOR_NO_ARGS(SBAddress);
}

SBAddress::SBAddress(const Address *lldb_object_ptr)
    : m_opaque_up(new Address()) {
  if (lldb_object_ptr)
    m_opaque_up = llvm::make_unique<Address>(*lldb_object_ptr);
}

SBAddress::SBAddress(const SBAddress &rhs) : m_opaque_up(new Address()) {
  LLDB_RECORD_CONSTRUCTOR(SBAddress, (const lldb::SBAddress &), rhs);

  m_opaque_up = clone(rhs.m_opaque_up);
}

SBAddress::SBAddress(lldb::SBSection section, lldb::addr_t offset)
    : m_opaque_up(new Address(section.GetSP(), offset)) {
  LLDB_RECORD_CONSTRUCTOR(SBAddress, (lldb::SBSection, lldb::addr_t), section,
                          offset);
}

// Create an address by resolving a load address using the supplied target
SBAddress::SBAddress(lldb::addr_t load_addr, lldb::SBTarget &target)
    : m_opaque_up(new Address()) {
  LLDB_RECORD_CONSTRUCTOR(SBAddress, (lldb::addr_t, lldb::SBTarget &),
                          load_addr, target);

  SetLoadAddress(load_addr, target);
}

SBAddress::~SBAddress() {}

const SBAddress &SBAddress::operator=(const SBAddress &rhs) {
  LLDB_RECORD_METHOD(const lldb::SBAddress &,
                     SBAddress, operator=,(const lldb::SBAddress &), rhs);

  if (this != &rhs)
    m_opaque_up = clone(rhs.m_opaque_up);
  return *this;
}

bool lldb::operator==(const SBAddress &lhs, const SBAddress &rhs) {
  if (lhs.IsValid() && rhs.IsValid())
    return lhs.ref() == rhs.ref();
  return false;
}

bool SBAddress::IsValid() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBAddress, IsValid);

  return m_opaque_up != NULL && m_opaque_up->IsValid();
}

void SBAddress::Clear() {
  LLDB_RECORD_METHOD_NO_ARGS(void, SBAddress, Clear);

  m_opaque_up.reset(new Address());
}

void SBAddress::SetAddress(lldb::SBSection section, lldb::addr_t offset) {
  LLDB_RECORD_METHOD(void, SBAddress, SetAddress,
                     (lldb::SBSection, lldb::addr_t), section, offset);

  Address &addr = ref();
  addr.SetSection(section.GetSP());
  addr.SetOffset(offset);
}

void SBAddress::SetAddress(const Address *lldb_object_ptr) {
  if (lldb_object_ptr)
    ref() = *lldb_object_ptr;
  else
    m_opaque_up.reset(new Address());
}

lldb::addr_t SBAddress::GetFileAddress() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(lldb::addr_t, SBAddress, GetFileAddress);

  if (m_opaque_up->IsValid())
    return m_opaque_up->GetFileAddress();
  else
    return LLDB_INVALID_ADDRESS;
}

lldb::addr_t SBAddress::GetLoadAddress(const SBTarget &target) const {
  LLDB_RECORD_METHOD_CONST(lldb::addr_t, SBAddress, GetLoadAddress,
                           (const lldb::SBTarget &), target);

  lldb::addr_t addr = LLDB_INVALID_ADDRESS;
  TargetSP target_sp(target.GetSP());
  if (target_sp) {
    if (m_opaque_up->IsValid()) {
      std::lock_guard<std::recursive_mutex> guard(target_sp->GetAPIMutex());
      addr = m_opaque_up->GetLoadAddress(target_sp.get());
    }
  }

  return addr;
}

void SBAddress::SetLoadAddress(lldb::addr_t load_addr, lldb::SBTarget &target) {
  LLDB_RECORD_METHOD(void, SBAddress, SetLoadAddress,
                     (lldb::addr_t, lldb::SBTarget &), load_addr, target);

  // Create the address object if we don't already have one
  ref();
  if (target.IsValid())
    *this = target.ResolveLoadAddress(load_addr);
  else
    m_opaque_up->Clear();

  // Check if we weren't were able to resolve a section offset address. If we
  // weren't it is ok, the load address might be a location on the stack or
  // heap, so we should just have an address with no section and a valid offset
  if (!m_opaque_up->IsValid())
    m_opaque_up->SetOffset(load_addr);
}

bool SBAddress::OffsetAddress(addr_t offset) {
  LLDB_RECORD_METHOD(bool, SBAddress, OffsetAddress, (lldb::addr_t), offset);

  if (m_opaque_up->IsValid()) {
    addr_t addr_offset = m_opaque_up->GetOffset();
    if (addr_offset != LLDB_INVALID_ADDRESS) {
      m_opaque_up->SetOffset(addr_offset + offset);
      return true;
    }
  }
  return false;
}

lldb::SBSection SBAddress::GetSection() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBSection, SBAddress, GetSection);

  lldb::SBSection sb_section;
  if (m_opaque_up->IsValid())
    sb_section.SetSP(m_opaque_up->GetSection());
  return LLDB_RECORD_RESULT(sb_section);
}

lldb::addr_t SBAddress::GetOffset() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::addr_t, SBAddress, GetOffset);

  if (m_opaque_up->IsValid())
    return m_opaque_up->GetOffset();
  return 0;
}

Address *SBAddress::operator->() { return m_opaque_up.get(); }

const Address *SBAddress::operator->() const { return m_opaque_up.get(); }

Address &SBAddress::ref() {
  if (m_opaque_up == NULL)
    m_opaque_up.reset(new Address());
  return *m_opaque_up;
}

const Address &SBAddress::ref() const {
  // This object should already have checked with "IsValid()" prior to calling
  // this function. In case you didn't we will assert and die to let you know.
  assert(m_opaque_up.get());
  return *m_opaque_up;
}

Address *SBAddress::get() { return m_opaque_up.get(); }

bool SBAddress::GetDescription(SBStream &description) {
  LLDB_RECORD_METHOD(bool, SBAddress, GetDescription, (lldb::SBStream &),
                     description);

  // Call "ref()" on the stream to make sure it creates a backing stream in
  // case there isn't one already...
  Stream &strm = description.ref();
  if (m_opaque_up->IsValid()) {
    m_opaque_up->Dump(&strm, NULL, Address::DumpStyleResolvedDescription,
                      Address::DumpStyleModuleWithFileAddress, 4);
    StreamString sstrm;
    //        m_opaque_up->Dump (&sstrm, NULL,
    //        Address::DumpStyleResolvedDescription, Address::DumpStyleInvalid,
    //        4);
    //        if (sstrm.GetData())
    //            strm.Printf (" (%s)", sstrm.GetData());
  } else
    strm.PutCString("No value");

  return true;
}

SBModule SBAddress::GetModule() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBModule, SBAddress, GetModule);

  SBModule sb_module;
  if (m_opaque_up->IsValid())
    sb_module.SetSP(m_opaque_up->GetModule());
  return LLDB_RECORD_RESULT(sb_module);
}

SBSymbolContext SBAddress::GetSymbolContext(uint32_t resolve_scope) {
  LLDB_RECORD_METHOD(lldb::SBSymbolContext, SBAddress, GetSymbolContext,
                     (uint32_t), resolve_scope);

  SBSymbolContext sb_sc;
  SymbolContextItem scope = static_cast<SymbolContextItem>(resolve_scope);
  if (m_opaque_up->IsValid())
    m_opaque_up->CalculateSymbolContext(&sb_sc.ref(), scope);
  return LLDB_RECORD_RESULT(sb_sc);
}

SBCompileUnit SBAddress::GetCompileUnit() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBCompileUnit, SBAddress, GetCompileUnit);

  SBCompileUnit sb_comp_unit;
  if (m_opaque_up->IsValid())
    sb_comp_unit.reset(m_opaque_up->CalculateSymbolContextCompileUnit());
  return LLDB_RECORD_RESULT(sb_comp_unit);
}

SBFunction SBAddress::GetFunction() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBFunction, SBAddress, GetFunction);

  SBFunction sb_function;
  if (m_opaque_up->IsValid())
    sb_function.reset(m_opaque_up->CalculateSymbolContextFunction());
  return LLDB_RECORD_RESULT(sb_function);
}

SBBlock SBAddress::GetBlock() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBBlock, SBAddress, GetBlock);

  SBBlock sb_block;
  if (m_opaque_up->IsValid())
    sb_block.SetPtr(m_opaque_up->CalculateSymbolContextBlock());
  return LLDB_RECORD_RESULT(sb_block);
}

SBSymbol SBAddress::GetSymbol() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBSymbol, SBAddress, GetSymbol);

  SBSymbol sb_symbol;
  if (m_opaque_up->IsValid())
    sb_symbol.reset(m_opaque_up->CalculateSymbolContextSymbol());
  return LLDB_RECORD_RESULT(sb_symbol);
}

SBLineEntry SBAddress::GetLineEntry() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBLineEntry, SBAddress, GetLineEntry);

  SBLineEntry sb_line_entry;
  if (m_opaque_up->IsValid()) {
    LineEntry line_entry;
    if (m_opaque_up->CalculateSymbolContextLineEntry(line_entry))
      sb_line_entry.SetLineEntry(line_entry);
  }
  return LLDB_RECORD_RESULT(sb_line_entry);
}
