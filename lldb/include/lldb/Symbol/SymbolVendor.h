//===-- SymbolVendor.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_SymbolVendor_h_
#define liblldb_SymbolVendor_h_

#include <vector>

#include "lldb/Core/ModuleChild.h"
#include "lldb/Core/PluginInterface.h"
#include "lldb/Symbol/SourceModule.h"
#include "lldb/Symbol/TypeMap.h"
#include "lldb/lldb-private.h"
#include "llvm/ADT/DenseSet.h"

namespace lldb_private {

// The symbol vendor class is designed to abstract the process of searching for
// debug information for a given module. Platforms can subclass this class and
// provide extra ways to find debug information. Examples would be a subclass
// that would allow for locating a stand alone debug file, parsing debug maps,
// or runtime data in the object files. A symbol vendor can use multiple
// sources (SymbolFile objects) to provide the information and only parse as
// deep as needed in order to provide the information that is requested.
class SymbolVendor : public ModuleChild, public PluginInterface {
public:
  static SymbolVendor *FindPlugin(const lldb::ModuleSP &module_sp,
                                  Stream *feedback_strm);

  // Constructors and Destructors
  SymbolVendor(const lldb::ModuleSP &module_sp);

  ~SymbolVendor() override;

  void AddSymbolFileRepresentation(const lldb::ObjectFileSP &objfile_sp);

  SymbolFile *GetSymbolFile() { return m_sym_file_up.get(); }

  // PluginInterface protocol
  ConstString GetPluginName() override;

  uint32_t GetPluginVersion() override;

protected:
  std::unique_ptr<SymbolFile> m_sym_file_up; // A single symbol file. Subclasses
                                             // can add more of these if needed.

private:
  // For SymbolVendor only
  DISALLOW_COPY_AND_ASSIGN(SymbolVendor);
};

} // namespace lldb_private

#endif // liblldb_SymbolVendor_h_
