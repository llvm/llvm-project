// RUN: %clang_cc1 -std=c23 --embed-dir=%S/Inputs -fsyntax-only -verify=silent %s
// RUN: %clang_cc1 -std=c23 --embed-dir=%S/Inputs -fsyntax-only -verify=compat -Wc++-compat %s
//
// Test -Wembed-parameter-is-macro:
// RUN: %clang_cc1 -std=c23 --embed-dir=%S/Inputs -fsyntax-only -verify=compat -Wembed-parameter-is-macro %s
// RUN: %clang_cc1 -std=c23 --embed-dir=%S/Inputs -fsyntax-only -verify=silent -Wc++-compat -Wno-embed-parameter-is-macro %s

// C++ has CWG3013: if one of the pp-tokens of a #embed directive (or a
// has-embed-expression) is the identifier limit, prefix, suffix, or if_empty
// and that identifier is defined as a macro, the program is ill-formed.
// (C++ [cpp.pre]/p4, [cpp.cond]/p9)
//
// ... But C doesn't seem to have this restriction, so we allow macro-expansion
// and only warn if -Wc++-compat or -Wembed-parameter-is-macro is enabled. C++
// conformance with CWG3013 is tested in clang/test/CXX/drs/cwg3013.cpp.

// silent-no-diagnostics

#define limit limit
// compat-note@-1 2 {{macro 'limit' defined here}}
const int a[] = {
#embed __FILE__ limit(2)
// compat-warning@-1 {{'limit' is defined as a macro and gets expanded when used as an '#embed' parameter in C; this is ill-formed in C++}}
};

#define prefix prefix
// compat-note@-1 {{macro 'prefix' defined here}}
const int b[] = {
#embed __FILE__ prefix(0,)
// compat-warning@-1 {{'prefix' is defined as a macro and gets expanded when used as an '#embed' parameter in C; this is ill-formed in C++}}
};

#define suffix suffix
// compat-note@-1 2 {{macro 'suffix' defined here}}
const int c[] = {
#embed __FILE__ suffix(,0)
// compat-warning@-1 {{'suffix' is defined as a macro and gets expanded when used as an '#embed' parameter in C; this is ill-formed in C++}}
};

#define if_empty if_empty
// compat-note@-1 {{macro 'if_empty' defined here}}
const int d[] = {
#embed __FILE__ if_empty(0)
// compat-warning@-1 {{'if_empty' is defined as a macro and gets expanded when used as an '#embed' parameter in C; this is ill-formed in C++}}
};

#if __has_embed(__FILE__ limit(1) suffix(0))
// compat-warning@-1 {{'limit' is defined as a macro and gets expanded when used as a '__has_embed' parameter in C; this is ill-formed in C++}}
// compat-warning@-2 {{'suffix' is defined as a macro and gets expanded when used as a '__has_embed' parameter in C; this is ill-formed in C++}}
int e;
#endif
