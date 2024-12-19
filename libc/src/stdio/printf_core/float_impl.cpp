//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file instantiates the functionality needed for supporting floating
/// point arguments in modular printf builds. Non-modular printf builds
/// implicitly instantiate these functions.
///
//===----------------------------------------------------------------------===//

#ifdef LIBC_COPT_PRINTF_MODULAR
#include "src/__support/arg_list.h"

#define LIBC_PRINTF_DEFINE_MODULAR
#include "src/stdio/printf_core/float_dec_converter.h"
#include "src/stdio/printf_core/float_hex_converter.h"
#include "src/stdio/printf_core/parser.h"

namespace LIBC_NAMESPACE_DECL {
namespace printf_core {
template class Parser<internal::ArgList>;
template class Parser<internal::DummyArgList<false>>;
template class Parser<internal::DummyArgList<true>>;
template class Parser<internal::StructArgList<false>>;
template class Parser<internal::StructArgList<true>>;

#define INSTANTIATE_CONVERT_FN(NAME)                                           \
  template int NAME<WriteMode::FILL_BUFF_AND_DROP_OVERFLOW>(                   \
      Writer<WriteMode::FILL_BUFF_AND_DROP_OVERFLOW> * writer,                 \
      const FormatSection &to_conv);                                           \
  template int NAME<WriteMode::FLUSH_TO_STREAM>(                               \
      Writer<WriteMode::FLUSH_TO_STREAM> * writer,                             \
      const FormatSection &to_conv);                                           \
  template int NAME<WriteMode::RESIZE_AND_FILL_BUFF>(                          \
      Writer<WriteMode::RESIZE_AND_FILL_BUFF> * writer,                        \
      const FormatSection &to_conv);                                           \
  template int NAME<WriteMode::RUNTIME_DISPATCH>(                              \
      Writer<WriteMode::RUNTIME_DISPATCH> * writer,                            \
      const FormatSection &to_conv)

INSTANTIATE_CONVERT_FN(convert_float_decimal);
INSTANTIATE_CONVERT_FN(convert_float_dec_exp);
INSTANTIATE_CONVERT_FN(convert_float_dec_auto);
INSTANTIATE_CONVERT_FN(convert_float_hex_exp);

} // namespace printf_core
} // namespace LIBC_NAMESPACE_DECL

// Bring this file into the link if __printf_float is referenced.
extern "C" void __printf_float() {}
#endif
