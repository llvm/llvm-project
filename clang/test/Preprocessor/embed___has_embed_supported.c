// RUN: %clang_cc1 %s -E -CC -verify

#if !__has_embed(__FILE__)
#error 1
#elif !__has_embed(__FILE__)
#error 2
#elif !__has_embed(__FILE__ suffix(x))
#error 3
#elif !__has_embed(__FILE__ suffix(x) limit(1))
#error 4
#elif !__has_embed(__FILE__ suffix(x) limit(1) prefix(1))
#error 5
#elif !__has_embed(__FILE__ suffix(x) limit(2) prefix(1) clang::offset(1))
#error 6
#elif !__has_embed(__FILE__ suffix(x) limit(0) prefix(1))
#error 7
#elif __has_embed(__FILE__ suffix(x) limit(1) prefix(1) clang::offset(1)) != 2
#error 8
#elif __has_embed(__FILE__ suffix(x) limit(0)) != 2
#error 9
#elif __has_embed(__FILE__ suffix(x) limit(0) if_empty(:3)) != 2
#error 10
#endif
// expected-no-diagnostics
