//===-- Printf Configuration Handler ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_PRINTF_CORE_PRINTF_CONFIG_H
#define LLVM_LIBC_SRC_STDIO_PRINTF_CORE_PRINTF_CONFIG_H

// The index array buffer is always initialized when printf is called. In cases
// where index mode is necessary but memory is limited, or when index mode
// performance is important and memory is available, this compile option
// provides a knob to adjust memory usage to an appropriate level. 128 is picked
// as the default size since that's big enough to handle even extreme cases and
// the runtime penalty for not having enough space is severe.
// When an index mode argument is requested, if its index is before the most
// recently read index, then the arg list must be restarted from the beginning,
// and all of the arguments before the new index must be requested with the
// correct types. The index array caches the types of the values in the arg
// list. For every number between the last index cached in the array and the
// requested index, the format string must be parsed again to find the
// type of that index. As an example, if the format string has 20 indexes, and
// the index array is 10, then when the 20th index is requested the first 10
// types can be found immediately, and then the format string must be parsed 10
// times to find the types of the next 10 arguments.
#ifndef LIBC_COPT_PRINTF_INDEX_ARR_LEN
#define LIBC_COPT_PRINTF_INDEX_ARR_LEN 128
#endif

// TODO(michaelrj): Provide a proper interface for these options.
// LIBC_COPT_FLOAT_TO_STR_USE_MEGA_LONG_DOUBLE_TABLE
// LIBC_COPT_FLOAT_TO_STR_USE_DYADIC_FLOAT
// LIBC_COPT_FLOAT_TO_STR_USE_DYADIC_FLOAT_LD
// LIBC_COPT_FLOAT_TO_STR_USE_INT_CALC
// LIBC_COPT_FLOAT_TO_STR_NO_TABLE
// LIBC_COPT_PRINTF_HEX_LONG_DOUBLE

// TODO(michaelrj): Move the other printf configuration options into this file.

#endif // LLVM_LIBC_SRC_STDIO_PRINTF_CORE_PRINTF_CONFIG_H
