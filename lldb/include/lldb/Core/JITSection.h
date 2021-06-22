//===-- JITSection.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_CORE_JIT_SECTION_H
#define LLDB_CORE_JIT_SECTION_H

#include "Section.h"

namespace lldb_private {

/// A JIT-compiled section. This type of section may contain JIT specific data
/// at the end of its buffer when compared to a regular section.
class JITSection : public Section {
public:
  JITSection(const lldb::ModuleSP &module_sp, ObjectFile *obj_file,
             lldb::user_id_t sect_id, ConstString name,
             lldb::SectionType sect_type, lldb::addr_t file_vm_addr,
             lldb::addr_t vm_size, lldb::offset_t file_offset,
             lldb::offset_t file_size, uint32_t log2align, uint32_t flags,
             size_t non_jit_size, uint32_t target_byte_size = 1)
      : Section(module_sp, obj_file, sect_id, name, sect_type, file_vm_addr,
                vm_size, file_offset, file_size, log2align, flags,
                target_byte_size),
        m_non_jit_size(non_jit_size) {}

  JITSection(const lldb::SectionSP &parent_section_sp,
             const lldb::ModuleSP &module_sp, ObjectFile *obj_file,
             lldb::user_id_t sect_id, ConstString name,
             lldb::SectionType sect_type, lldb::addr_t file_vm_addr,
             lldb::addr_t vm_size, lldb::offset_t file_offset,
             lldb::offset_t file_size, uint32_t log2align, uint32_t flags,
             size_t non_jit_size, uint32_t target_byte_size = 1)
      : Section(parent_section_sp, module_sp, obj_file, sect_id, name,
                sect_type, file_vm_addr, vm_size, file_offset, file_size,
                log2align, flags, target_byte_size),
        m_non_jit_size(non_jit_size) {}

  // LLVM RTTI support
  static char ID;
  virtual bool isA(const void *ClassID) const {
    return ClassID == &ID || Section::isA(ClassID);
  }
  static bool classof(const Section *obj) { return obj->isA(&ID); }

  /// Returns the section size discounting the jit specific data.
  size_t getNonJitSize() const { return m_non_jit_size; }

private:
  size_t m_non_jit_size = 0;
};
} // namespace lldb_private

#endif // LLDB_CORE_JIT_SECTION_H
