// RUN: %clang_cc1 %s -Eonly -verify

// This should not be rejected.
#ifdef defined
#elifdef defined
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

// RUN: %clang_cc1 -fms-extensions -DMS_EXT %s -Eonly -verify
#ifndef MS_EXT
// This should under C99.
#define FUNC_LIKE4(a,b) (a+b)  // expected-note {{previous definition is here}}
#define FUNC_LIKE4(x,y) (x+y) // expected-warning {{'FUNC_LIKE4' macro redefined}}
#else
// This shouldn't under MS extensions.
#define FUNC_LIKE4(a,b) (a+b)
#define FUNC_LIKE4(x,y) (x+y)

// This should.
#define FUNC_LIKE5(a,b) (a+b) // expected-note {{previous definition is here}}
#define FUNC_LIKE5(x,y) (y+x) // expected-warning {{'FUNC_LIKE5' macro redefined}}
#endif
