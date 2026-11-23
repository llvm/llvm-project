// RUN: %clang_cc1 -triple i686-pc-linux-gnu -fexperimental-new-constant-interpreter -verify=expected,both %s
// RUN: %clang_cc1 -triple i686-pc-linux-gnu                                         -verify=ref,both      %s


// both-no-diagnostics

/// FIXME: Allocating the array below causes OOM failures with the bytecode interpreter.
#if 0
char melchizedek[2200000000];
typedef decltype(melchizedek[1] - melchizedek[0]) ptrdiff_t;
constexpr ptrdiff_t d1 = &melchizedek[0x7fffffff] - &melchizedek[0];
constexpr ptrdiff_t d2 = &melchizedek[0x80000000u] - &melchizedek[0]; // both-error {{constant expression}} \
                                                                      // both-note {{ 2147483648 }}
constexpr ptrdiff_t d3 = &melchizedek[0] - &melchizedek[0x80000000u];
constexpr ptrdiff_t d4 = &melchizedek[0] - &melchizedek[0x80000001u]; // both-error {{constant expression}} \
                                                                      // both-note {{ -2147483649 }}
#endif
