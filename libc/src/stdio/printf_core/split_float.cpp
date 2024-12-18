#include "src/stdio/printf_core/float_dec_converter.h"
#include "src/stdio/printf_core/float_hex_converter.h"

// Bring this file into the link if __printf_float is referenced.
void __printf_float() {
  // Bring the printf floating point implementation into the link.
  __libc_printf_float_dec_converter();
  __libc_printf_float_hex_converter();
}
