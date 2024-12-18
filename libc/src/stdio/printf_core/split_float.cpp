#define LIBC_PRINTF_DEFINE_SPLIT
#include "src/stdio/printf_core/float_dec_converter.h"
#include "src/stdio/printf_core/float_hex_converter.h"
#include "src/stdio/printf_core/parser.h"

// Bring this file into the link if __printf_float is referenced.
void __printf_float() {}
