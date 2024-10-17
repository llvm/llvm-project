//===-- include/flang-rt/io-api-funcs.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Defines API between compiled code and I/O runtime library.

#ifndef FLANGRT_IO_API_FUNCS_H_
#define FLANGRT_IO_API_FUNCS_H_

#include "flang/Common/uint128.h"
#include "flang/Runtime/entry-names.h"
#include "flang/Runtime/io-api.h"
#include "flang/Runtime/iostat.h"
#include "flang/Runtime/magic-numbers.h"
#include <cinttypes>
#include <cstddef>

namespace Fortran::runtime {
class Descriptor;
} // namespace Fortran::runtime

namespace Fortran::runtime::io {

struct NonTbpDefinedIoTable;
class NamelistGroup;
class IoStatementState;
using Cookie = IoStatementState *;
using ExternalUnit = int;
using AsynchronousId = int;

RT_API_ATTRS const char *InquiryKeywordHashDecode(
    char *buffer, std::size_t, InquiryKeywordHash);

} // namespace Fortran::runtime::io
#endif /* FLANGRT_IO_API_FUNCS_H_ */
