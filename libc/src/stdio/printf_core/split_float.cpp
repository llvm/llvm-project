#include "src/__support/arg_list.h"

#define LIBC_PRINTF_DEFINE_SPLIT
#include "src/stdio/printf_core/float_dec_converter.h"
#include "src/stdio/printf_core/float_hex_converter.h"
#include "src/stdio/printf_core/parser.h"

namespace LIBC_NAMESPACE_DECL {
namespace printf_core {
// Explicitly instantiate templates containing functions only defined in this
// file.
template class Parser<internal::ArgList>;
template class Parser<internal::DummyArgList<false>>;
template class Parser<internal::DummyArgList<true>>;
template class Parser<internal::StructArgList<false>>;
template class Parser<internal::StructArgList<true>>;
} // namespace printf_core
} // namespace LIBC_NAMESPACE_DECL

// Bring this file into the link if __printf_float is referenced.
extern "C" void __printf_float() {}
