// RUN: %clang_cc1 -std=c23 %s -E -verify

#if __has_embed(__FILE__) != __STDC_EMBED_FOUND__
#error 1
#elif __has_embed(__FILE__) != __STDC_EMBED_FOUND__
#error 2
#elif __has_embed(__FILE__ suffix(x)) != __STDC_EMBED_FOUND__
#error 3
#elif __has_embed(__FILE__ suffix(x) limit(1)) != __STDC_EMBED_FOUND__
#error 4
#elif __has_embed(__FILE__ suffix(x) limit(1) prefix(1)) != __STDC_EMBED_FOUND__
#error 5
#elif __has_embed(__FILE__ suffix(x) limit(2) prefix(1) clang::offset(1)) != __STDC_EMBED_FOUND__
#error 6
#elif __has_embed(__FILE__ suffix(x) limit(0) prefix(1)) != __STDC_EMBED_EMPTY__
#error 7
#elif __has_embed(__FILE__ suffix(x) limit(1) prefix(1) clang::offset(1)) != __STDC_EMBED_FOUND__
#error 8
#elif __has_embed(__FILE__ suffix(x) limit(0)) != __STDC_EMBED_EMPTY__
#error 9
#elif __has_embed(__FILE__ suffix(x) limit(0) if_empty(:3)) != __STDC_EMBED_EMPTY__
#error 10
#endif
// expected-no-diagnostics
