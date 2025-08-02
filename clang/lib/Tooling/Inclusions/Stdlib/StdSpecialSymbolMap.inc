//===-- StdSpecialSymbolMap.inc ---------------------------------*- C++ -*-===//
//
// This is a hand-curated list for C++ symbols that cannot be parsed/extracted
// via the include-mapping tool (gen_std.py).
//
//===----------------------------------------------------------------------===//

// Symbols that can be provided by any of the headers, ordered by the header
// preference.

// Add headers for generic integer-type abs.
// Ignore other variants (std::complex, std::valarray, std::intmax_t)
SYMBOL(abs, std::, <cstdlib>)
SYMBOL(abs, std::, <cmath>)
SYMBOL(abs, None, <cstdlib>)
SYMBOL(abs, None, <stdlib.h>)
SYMBOL(abs, None, <cmath>)
SYMBOL(abs, None, <math.h>)

// std::get has a few variants for different types (tuple, array, pair etc)
// which is tricky to disambiguate without type information.
// Don't set any header for it, as it comes with the type header.
SYMBOL(get, std::, /*no headers*/)
// Similarly make_error_{code,condition} also have different overloads (errc,
// io_errc, future_errc) and each of them are provided by relevant headers
// providing the type.
SYMBOL(make_error_code, std::, /*no headers*/)
SYMBOL(make_error_condition, std::, /*no headers*/)
// Similar to std::get, has variants for multiple containers
// (vector, deque, list, etc.)
SYMBOL(erase, std::, /*no headers*/)
SYMBOL(erase_if, std::, /*no headers*/)

// Symbols missing from the generated symbol map as reported by users.
// Remove when the generator starts producing them.
SYMBOL(abort, std::, <cstdlib>)

// These are C symbols that are not under std namespace.
SYMBOL(localtime_r, None, <ctime>)
SYMBOL(localtime_r, None, <time.h>)
SYMBOL(localtime_s, None, <ctime>)
SYMBOL(localtime_s, None, <time.h>)
SYMBOL(gmtime_r, None, <ctime>)
SYMBOL(gmtime_r, None, <time.h>)
SYMBOL(gmtime_s, None, <ctime>)
SYMBOL(gmtime_s, None, <time.h>)

// The std::placeholder symbols (_1, ..., _N) are listed in the cppreference
// placeholder.html, but the index only contains a single entry with "_1, _2, ..., _N"
// text, which are not handled by the script.
// N is an implementation-defined number (10 for libc++; 29 for libstdc++).
SYMBOL(_1, std::placeholders::, <functional>)
SYMBOL(_2, std::placeholders::, <functional>)
SYMBOL(_3, std::placeholders::, <functional>)
SYMBOL(_4, std::placeholders::, <functional>)
SYMBOL(_5, std::placeholders::, <functional>)
SYMBOL(_6, std::placeholders::, <functional>)
SYMBOL(_7, std::placeholders::, <functional>)
SYMBOL(_8, std::placeholders::, <functional>)
SYMBOL(_9, std::placeholders::, <functional>)
SYMBOL(_10, std::placeholders::, <functional>)
SYMBOL(_11, std::placeholders::, <functional>)
SYMBOL(_12, std::placeholders::, <functional>)
SYMBOL(_13, std::placeholders::, <functional>)
SYMBOL(_14, std::placeholders::, <functional>)
SYMBOL(_15, std::placeholders::, <functional>)
SYMBOL(_16, std::placeholders::, <functional>)
SYMBOL(_17, std::placeholders::, <functional>)
SYMBOL(_18, std::placeholders::, <functional>)
SYMBOL(_19, std::placeholders::, <functional>)
SYMBOL(_20, std::placeholders::, <functional>)
SYMBOL(_21, std::placeholders::, <functional>)
SYMBOL(_22, std::placeholders::, <functional>)
SYMBOL(_23, std::placeholders::, <functional>)
SYMBOL(_24, std::placeholders::, <functional>)
SYMBOL(_25, std::placeholders::, <functional>)
SYMBOL(_26, std::placeholders::, <functional>)
SYMBOL(_27, std::placeholders::, <functional>)
SYMBOL(_28, std::placeholders::, <functional>)
SYMBOL(_29, std::placeholders::, <functional>)
