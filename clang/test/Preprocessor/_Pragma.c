// RUN: %clang_cc1 %s -verify -Wall

_Pragma ("GCC system_header")  // expected-warning {{system_header ignored in main file}}

// rdar://6880630
_Pragma("#define macro")    // expected-warning {{unknown pragma ignored}}

#ifdef macro
#error #define invalid
#endif
