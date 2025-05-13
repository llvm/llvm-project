//===-- StdSpecialSymbolMap.inc -----------------------------------*- C -*-===//
//
// This is a hand-curated list for C symbols that cannot be parsed/extracted
// via the include-mapping tool (gen_std.py).
//
//===----------------------------------------------------------------------===//

SYMBOL(size_t, None, <stddef.h>)
SYMBOL(size_t, None, <stdio.h>)
SYMBOL(size_t, None, <stdlib.h>)
SYMBOL(size_t, None, <string.h>)
SYMBOL(size_t, None, <time.h>)
SYMBOL(size_t, None, <uchar.h>)
SYMBOL(size_t, None, <wchar.h>)

// These macros trigger ambiguity when parsing cppreference index.
// Taken from https://en.cppreference.com/w/c/types/integer.
SYMBOL(INT8_C, None, <stdint.h>)
SYMBOL(INT16_C, None, <stdint.h>)
SYMBOL(INT32_C, None, <stdint.h>)
SYMBOL(INT64_C, None, <stdint.h>)
SYMBOL(INTMAX_C, None, <stdint.h>)

SYMBOL(UINT8_C, None, <stdint.h>)
SYMBOL(UINT16_C, None, <stdint.h>)
SYMBOL(UINT32_C, None, <stdint.h>)
SYMBOL(UINT64_C, None, <stdint.h>)
SYMBOL(UINTMAX_C, None, <stdint.h>)
