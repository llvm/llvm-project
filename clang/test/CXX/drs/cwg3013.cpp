// RUN: %clang_cc1 --embed-dir=%S/inputs -Wno-c23-extensions -fsyntax-only -verify=cxx,common -x c++ %s
// RUN: %clang_cc1 --embed-dir=%S/inputs -fsyntax-only -verify=c -std=c23 -x c %s
// RUN: %clang_cc1 --embed-dir=%S/inputs -fsyntax-only -verify=c-compat,common -std=c23 -Wc++-compat -x c %s
//
// Test -Wembed-parameter-is-macro:
// RUN: %clang_cc1 --embed-dir=%S/inputs -fsyntax-only -verify=c-compat,common -std=c23 -Wembed-parameter-is-macro -x c %s
// RUN: %clang_cc1 --embed-dir=%S/inputs -fsyntax-only -verify=c -std=c23 -Wc++-compat -Wno-embed-parameter-is-macro -x c %s

// CWG3013: if one of the pp-tokens of a #embed directive (or a
// has-embed-expression) is the identifier limit, prefix, suffix, or if_empty
// and that identifier is defined as a macro, the program is ill-formed.
// (C++ [cpp.pre]/p4, [cpp.cond]/p9)
//
// However, C doesn't have this restriction, so we should only issue a warning
// for C if -Wc++-compat/-Wembed-parameter-is-macro is enabled.

// c-no-diagnostics

#define limit limit
// common-note@-1 2 {{macro 'limit' defined here}}
const int a[] = {
#embed <media/art.txt> limit(2)
// cxx-error@-1 {{cannot use 'limit' as an '#embed' parameter if also defined as a macro}}
// c-compat-warning@-2 {{'limit' is defined as a macro and gets expanded when used as an '#embed' parameter in C; this is ill-formed in C++}}
};

#define prefix prefix
// common-note@-1 {{macro 'prefix' defined here}}
const int b[] = {
#embed <media/art.txt> prefix(0,)
// cxx-error@-1 {{cannot use 'prefix' as an '#embed' parameter if also defined as a macro}}
// c-compat-warning@-2 {{'prefix' is defined as a macro and gets expanded when used as an '#embed' parameter in C; this is ill-formed in C++}}
};

#define suffix suffix
// common-note@-1 2 {{macro 'suffix' defined here}}
const int c[] = {
#embed <media/art.txt> suffix(,0)
// cxx-error@-1 {{cannot use 'suffix' as an '#embed' parameter if also defined as a macro}}
// c-compat-warning@-2 {{'suffix' is defined as a macro and gets expanded when used as an '#embed' parameter in C; this is ill-formed in C++}}
};

#define if_empty if_empty
// common-note@-1 {{macro 'if_empty' defined here}}
const int d[] = {
#embed <media/empty> if_empty(0)
// cxx-error@-1 {{cannot use 'if_empty' as an '#embed' parameter if also defined as a macro}}
// c-compat-warning@-2 {{'if_empty' is defined as a macro and gets expanded when used as an '#embed' parameter in C; this is ill-formed in C++}}
};

// The prohibition also covers the __has_embed argument.
#if __has_embed(<media/art.txt> limit(1) suffix(0))
// cxx-error@-1 {{cannot use 'limit' as a '__has_embed' parameter if also defined as a macro}}
// cxx-error@-2 {{cannot use 'suffix' as a '__has_embed' parameter if also defined as a macro}}
// c-compat-warning@-3 {{'limit' is defined as a macro and gets expanded when used as a '__has_embed' parameter in C; this is ill-formed in C++}}
// c-compat-warning@-4 {{'suffix' is defined as a macro and gets expanded when used as a '__has_embed' parameter in C; this is ill-formed in C++}}
int e;
#endif
