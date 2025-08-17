//===-- lib/runtime/descriptor-io.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FLANG_RT_RUNTIME_DESCRIPTOR_IO_H_
#define FLANG_RT_RUNTIME_DESCRIPTOR_IO_H_

#include "flang-rt/runtime/connection.h"

namespace Fortran::runtime {
class Descriptor;
} // namespace Fortran::runtime

namespace Fortran::runtime::io {
class IoStatementState;
struct NonTbpDefinedIoTable;
} // namespace Fortran::runtime::io

namespace Fortran::runtime::io::descr {

template <Direction DIR>
RT_API_ATTRS bool DescriptorIO(IoStatementState &, const Descriptor &,
    const NonTbpDefinedIoTable * = nullptr);

extern template RT_API_ATTRS bool DescriptorIO<Direction::Output>(
    IoStatementState &, const Descriptor &, const NonTbpDefinedIoTable *);
extern template RT_API_ATTRS bool DescriptorIO<Direction::Input>(
    IoStatementState &, const Descriptor &, const NonTbpDefinedIoTable *);

} // namespace Fortran::runtime::io::descr
#endif // FLANG_RT_RUNTIME_DESCRIPTOR_IO_H_
