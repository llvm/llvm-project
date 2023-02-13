//===-- dwarf.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_CORE_DWARF_H
#define LLDB_CORE_DWARF_H

#include "lldb/Utility/RangeMap.h"
#include <cstdint>

// Get the DWARF constant definitions from llvm
#include "llvm/BinaryFormat/Dwarf.h"

namespace lldb_private {
namespace dwarf {
  using namespace llvm::dwarf;
}
}

typedef uint32_t dw_uleb128_t;
typedef int32_t dw_sleb128_t;
typedef uint16_t dw_attr_t;
typedef uint16_t dw_form_t;
typedef llvm::dwarf::Tag dw_tag_t;
typedef uint64_t dw_addr_t; // Dwarf address define that must be big enough for
                            // any addresses in the compile units that get
                            // parsed

typedef uint32_t dw_offset_t; // Dwarf Debug Information Entry offset for any
                              // offset into the file

/* Constants */
#define DW_INVALID_OFFSET (~(dw_offset_t)0)
#define DW_INVALID_INDEX 0xFFFFFFFFul

// #define DW_ADDR_none 0x0

#define DW_EH_PE_MASK_ENCODING 0x0F

typedef lldb_private::RangeVector<dw_addr_t, dw_addr_t, 2> DWARFRangeList;

#endif // LLDB_CORE_DWARF_H
