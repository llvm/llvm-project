//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Text presentation of register metadata and its relationships.
///
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_DUMPREGISTERINFO_H
#define LLDB_REGISTER_DUMPREGISTERINFO_H

#include <stdint.h>
#include <utility>
#include <vector>

namespace lldb_private {

class Stream;
class RegisterDescription;
struct RegisterDescriptor;
class RegisterType;

/// Print names, size, relationships, presentation groups, and type details.
/// \p info must belong to \p description so relationship IDs can be resolved. The
/// terminal width controls wrapping; no register value or process state is
/// acquired by this metadata-only operation.
void DumpRegisterInfo(Stream &strm, const RegisterDescription &description,
                      const RegisterDescriptor &info, uint32_t terminal_width);

/// Format already resolved metadata without consulting a register description.
/// This is a test seam; production callers should use DumpRegisterInfo().
void DoDumpRegisterInfo(
    Stream &strm, const char *name, const char *alt_name, uint32_t byte_size,
    const std::vector<const char *> &invalidates,
    const std::vector<const char *> &read_from,
    const std::vector<std::pair<const char *, uint32_t>> &in_sets,
    const RegisterType *register_type, uint32_t terminal_width);

} // namespace lldb_private

#endif // LLDB_REGISTER_DUMPREGISTERINFO_H
