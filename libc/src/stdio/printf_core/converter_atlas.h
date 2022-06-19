//===-- Map of converter headers in printf ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This file exists so that if the user wants to supply a custom atlas they can
// just replace the #include, additionally it keeps the ifdefs out of the
// converter header.

#ifndef LLVM_LIBC_SRC_STDIO_PRINTF_CORE_CONVERTER_ATLAS_H
#define LLVM_LIBC_SRC_STDIO_PRINTF_CORE_CONVERTER_ATLAS_H

// defines convert_string
#include "src/stdio/printf_core/string_converter.h"

// defines convert_char
#include "src/stdio/printf_core/char_converter.h"

// defines convert_int
#include "src/stdio/printf_core/int_converter.h"

// defines convert_oct
// defines convert_hex
#include "src/stdio/printf_core/hex_converter.h"

// TODO(michaelrj): add a flag to disable float point values here
// defines convert_float_decimal
// defines convert_float_dec_exp
// defines convert_float_hex_exp
// defines convert_float_mixed

// TODO(michaelrj): add a flag to disable writing an int here
// defines convert_write_int

// defines convert_pointer
#include "src/stdio/printf_core/ptr_converter.h"

#endif // LLVM_LIBC_SRC_STDIO_PRINTF_CORE_CONVERTER_ATLAS_H
