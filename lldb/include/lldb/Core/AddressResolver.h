//===-- AddressResolver.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_AddressResolver_h_
#define liblldb_AddressResolver_h_

#include "lldb/Core/AddressRange.h"
#include "lldb/Core/SearchFilter.h"
#include "lldb/lldb-defines.h"

#include <stddef.h>
#include <vector>

namespace lldb_private {
class ModuleList;
}
namespace lldb_private {
class Stream;
}
namespace lldb_private {

//----------------------------------------------------------------------
/// @class AddressResolver AddressResolver.h "lldb/Core/AddressResolver.h"
/// This class works with SearchFilter to resolve function names and source
/// file locations to their concrete addresses.
//----------------------------------------------------------------------

//----------------------------------------------------------------------
/// General Outline:
/// The AddressResolver is a Searcher.  In that protocol, the SearchFilter
/// asks the question "At what depth of the symbol context descent do you want
/// your callback to get called?" of the filter.  The resolver answers this
/// question (in the GetDepth method) and provides the resolution callback.
//----------------------------------------------------------------------

class AddressResolver : public Searcher {
public:
  typedef enum { Exact, Regexp, Glob } MatchType;

  AddressResolver();

  ~AddressResolver() override;

  virtual void ResolveAddress(SearchFilter &filter);

  virtual void ResolveAddressInModules(SearchFilter &filter,
                                       ModuleList &modules);

  void GetDescription(Stream *s) override = 0;

  std::vector<AddressRange> &GetAddressRanges();

  size_t GetNumberOfAddresses();

  AddressRange &GetAddressRangeAtIndex(size_t idx);

protected:
  std::vector<AddressRange> m_address_ranges;

private:
  DISALLOW_COPY_AND_ASSIGN(AddressResolver);
};

} // namespace lldb_private

#endif // liblldb_AddressResolver_h_
