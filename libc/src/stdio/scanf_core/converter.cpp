//===-- Format specifier converter implmentation for scanf -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/scanf_core/converter.h"

#include "src/__support/ctype_utils.h"
#include "src/__support/macros/config.h"
#include "src/stdio/scanf_core/core_structs.h"
#include "src/stdio/scanf_core/reader.h"

#ifndef LIBC_COPT_SCANF_DISABLE_FLOAT
#include "src/stdio/scanf_core/float_converter.h"
#endif // LIBC_COPT_SCANF_DISABLE_FLOAT
#include "src/stdio/scanf_core/current_pos_converter.h"
#include "src/stdio/scanf_core/int_converter.h"
#include "src/stdio/scanf_core/ptr_converter.h"
#include "src/stdio/scanf_core/string_converter.h"

#include <stddef.h>

namespace LIBC_NAMESPACE_DECL {
namespace scanf_core {

} // namespace scanf_core
} // namespace LIBC_NAMESPACE_DECL
