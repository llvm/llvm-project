// RUN: %clang_cc1 -std=c++2c -fexceptions -fcxx-exceptions -pedantic-errors -verify-directives -Wno-c23-extensions -verify %s

// C doesn't have an analogue to CWG3013: There is a corresponding C test at
// clang/test/Preprocessor/embed_cxx_compat.c testing an opt-in CXX-compat
// diagnostic.

namespace cwg3013 { // cwg3013: 24

#define limit limit // #cwg3013-limit
const int a[] = {
#embed __FILE__ limit(2)
// expected-error@-1 {{cannot use 'limit' as an '#embed' parameter if also defined as a macro}}
//   expected-note@#cwg3013-limit {{macro 'limit' defined here}}
};

#define prefix prefix // #cwg3013-prefix
const int b[] = {
#embed __FILE__ prefix(0,)
// expected-error@-1 {{cannot use 'prefix' as an '#embed' parameter if also defined as a macro}}
//   expected-note@#cwg3013-prefix {{macro 'prefix' defined here}}
};

#define suffix suffix // #cwg3013-suffix
const int c[] = {
#embed __FILE__ suffix(,0)
// expected-error@-1 {{cannot use 'suffix' as an '#embed' parameter if also defined as a macro}}
//   expected-note@#cwg3013-suffix {{macro 'suffix' defined here}}
};

#define if_empty if_empty // #cwg3013-if_empty
const int d[] = {
#embed __FILE__ if_empty(0)
// expected-error@-1 {{cannot use 'if_empty' as an '#embed' parameter if also defined as a macro}}
//   expected-note@#cwg3013-if_empty {{macro 'if_empty' defined here}}
};

#if __has_embed(__FILE__ limit(1) suffix(0))
// expected-error@-1 {{cannot use 'limit' as a '__has_embed' parameter if also defined as a macro}}
//   expected-note@#cwg3013-limit {{macro 'limit' defined here}}
// expected-error@-3 {{cannot use 'suffix' as a '__has_embed' parameter if also defined as a macro}}
//   expected-note@#cwg3013-suffix {{macro 'suffix' defined here}}
int e;
#endif

} // namespace cwg3013