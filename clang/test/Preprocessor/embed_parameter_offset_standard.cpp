// RUN: %clang_cc1 -std=c++2d %s -fsyntax-only --embed-dir=%S/Inputs -verify=all -pedantic-errors
// RUN: %clang_cc1 -std=c++2d %s -fsyntax-only --embed-dir=%S/Inputs -verify=all,compat -Wpre-c++2d-compat
// RUN: %clang_cc1 -std=c++2c %s -fsyntax-only --embed-dir=%S/Inputs -verify=all,ext -pedantic
// RUN: %clang_cc1 -x c -std=c23 %s -fsyntax-only --embed-dir=%S/Inputs -verify=all,cext

// The standard 'offset' spelling is a C++2d feature (P3540R3); it is an
// extension in earlier C++ language modes and in C.

constexpr unsigned char skipped[] = {
#embed <jk.txt> offset(1)
/* compat-warning@-1 {{the 'offset' #embed parameter is incompatible with C++ standards before C++2d}}
   ext-warning@-1 {{the 'offset' #embed parameter is a C++2d extension}}
   cext-warning@-1 {{the 'offset' #embed parameter is a Clang extension}} */
};
static_assert(sizeof(skipped) == 1 && skipped[0] == 'k', "offset skips elements");

// offset applies to the original resource size, before limit.
constexpr unsigned char window[] = {
#embed <jk.txt> limit(1) offset(1)
/* compat-warning@-1 {{the 'offset' #embed parameter is incompatible with C++ standards before C++2d}}
   ext-warning@-1 {{the 'offset' #embed parameter is a C++2d extension}}
   cext-warning@-1 {{the 'offset' #embed parameter is a Clang extension}} */
};
static_assert(sizeof(window) == 1 && window[0] == 'k', "offset applies before limit");

// offset(0) has no effect.
constexpr unsigned char whole[] = {
#embed <jk.txt> offset(0)
/* compat-warning@-1 {{the 'offset' #embed parameter is incompatible with C++ standards before C++2d}}
   ext-warning@-1 {{the 'offset' #embed parameter is a C++2d extension}}
   cext-warning@-1 {{the 'offset' #embed parameter is a Clang extension}} */
};
static_assert(sizeof(whole) == 2 && whole[0] == 'j' && whole[1] == 'k', "offset(0) is a no-op");

// An offset past the end of the resource makes it empty, so if_empty applies.
constexpr unsigned char fallback[] = {
#embed <jk.txt> offset(3) if_empty('x')
/* compat-warning@-1 {{the 'offset' #embed parameter is incompatible with C++ standards before C++2d}}
   ext-warning@-1 {{the 'offset' #embed parameter is a C++2d extension}}
   cext-warning@-1 {{the 'offset' #embed parameter is a Clang extension}} */
};
static_assert(sizeof(fallback) == 1 && fallback[0] == 'x', "past-the-end offset is empty");

/* compat-warning@+3 {{the 'offset' #embed parameter is incompatible with C++ standards before C++2d}}
   ext-warning@+3 {{the 'offset' #embed parameter is a C++2d extension}}
   cext-warning@+3 {{the 'offset' #embed parameter is a Clang extension}} */
#if __has_embed(<jk.txt> offset(3)) != __STDC_EMBED_EMPTY__
#error "__has_embed should report an empty resource"
#endif

// 'offset' and 'clang::offset' are the same parameter.
const unsigned char dup[] = {
#embed <jk.txt> offset(1) clang::offset(1)
/* all-error@-1 {{cannot specify parameter 'clang::offset' twice in the same '#embed' directive}}
   compat-warning@-1 {{the 'offset' #embed parameter is incompatible with C++ standards before C++2d}}
   ext-warning@-1 {{the 'offset' #embed parameter is a C++2d extension}}
   cext-warning@-1 {{the 'offset' #embed parameter is a Clang extension}} */
};

// Cannot have a negative offset.
const unsigned char negative[] = {
#embed <jk.txt> offset(-1)
/* all-error@-1 {{invalid value '-1'; must be positive}}
   compat-warning@-1 {{the 'offset' #embed parameter is incompatible with C++ standards before C++2d}}
   ext-warning@-1 {{the 'offset' #embed parameter is a C++2d extension}}
   cext-warning@-1 {{the 'offset' #embed parameter is a Clang extension}} */
'j'};
