// RUN: %clang_cc1 %s -Eonly -verify

// This should not be rejected.
#ifdef defined
#endif



// PR3764

// This should not produce a redefinition warning.
#define FUNC_LIKE(a) (a)
#define FUNC_LIKE(a)(a)

// This either.
#define FUNC_LIKE2(a)\
(a)
#define FUNC_LIKE2(a) (a)

// This should.
#define FUNC_LIKE3(a) ( a)  // expected-note {{previous definition is here}}
#define FUNC_LIKE3(a) (a) // expected-warning {{'FUNC_LIKE3' macro redefined}}

