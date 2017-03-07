//===-- AddressResolver.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/AddressResolver.h"

// Project includes

#include "lldb/Core/Address.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Core/SearchFilter.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Stream.h"
#include "lldb/Utility/StreamString.h"

using namespace lldb_private;

//----------------------------------------------------------------------
// AddressResolver:
//----------------------------------------------------------------------
AddressResolver::AddressResolver() {}

AddressResolver::~AddressResolver() {}

void AddressResolver::ResolveAddressInModules(SearchFilter &filter,
                                              ModuleList &modules) {
  filter.SearchInModuleList(*this, modules);
}

void AddressResolver::ResolveAddress(SearchFilter &filter) {
  filter.Search(*this);
}

std::vector<AddressRange> &AddressResolver::GetAddressRanges() {
  return m_address_ranges;
}

size_t AddressResolver::GetNumberOfAddresses() {
  return m_address_ranges.size();
}

AddressRange &AddressResolver::GetAddressRangeAtIndex(size_t idx) {
  return m_address_ranges[idx];
}
