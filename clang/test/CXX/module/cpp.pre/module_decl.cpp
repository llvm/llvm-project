// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/line.cpp -verify -o %t/line.pcm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/gnu_line_marker.cpp -verify -o %t/gnu_line_marker.pcm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/include.cpp -verify -o %t/include.pcm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/ident.cpp -verify -o %t/ident.pcm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/pragma_comment.cpp -verify -o %t/pragma_comment.pcm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/pragma_mark.cpp -verify -o %t/pragma_mark.pcm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/pragma_detect_mismatch.cpp -verify -o %t/pragma_detect_mismatch.pcm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/pragma_clang_debug.cpp -verify -o %t/pragma_clang_debug.pcm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/pragma_message.cpp -verify -o %t/pragma_message.pcm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/pragma_gcc_warn.cpp -verify -o %t/pragma_gcc_warn.pcm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/pragma_gcc_error.cpp -verify -o %t/pragma_gcc_error.pcm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/pragma_diag_push_pop.cpp -verify -o %t/pragma_diag_push_pop.pcm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/pragma_diag_ignore.cpp -verify -o %t/pragma_diag_ignore.pcm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/pragma_opencl_ext.cpp -verify -o %t/pragma_opencl_ext.pcm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/pragma_push_pop.cpp -verify -o %t/pragma_push_pop.pcm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/pragma_exec_charset.cpp -verify -o %t/pragma_exec_charset.pcm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/pragma_clang_assume_nonnull.cpp -verify -o %t/pragma_clang_assume_nonnull.pcm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/marco_expand.cpp -DMACRO="" -verify -o %t/marco_expand.pcm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/define.cpp -verify -o %t/define.pcm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/undef.cpp -verify -o %t/undef.pcm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/defined.cpp -verify -o %t/defined.pcm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/has_embed.cpp -verify -o %t/has_embed.pcm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/has_include.cpp -verify -o %t/has_include.pcm

//--- header.h
#ifndef HEADER_H
#define HEADER_H

#endif // HEADER_H

//--- line.cpp
// expected-no-diagnostics
#line 3
export module M;

//--- gnu_line_marker.cpp
// expected-no-diagnostics
# 1 __FILE__ 1 3
export module M;

//--- include.cpp
#include "header.h" // expected-note {{add 'module;' to the start of the file to introduce a global module fragment}}
export module M; // expected-error {{module declaration must occur at the start of the translation unit}}

//--- ident.cpp
// expected-no-diagnostics
#ident "$Header:$"
export module M;

//--- pragma_comment.cpp
// expected-no-diagnostics
#pragma comment(lib, "msvcrt.lib")
export module M;

//--- pragma_mark.cpp
// expected-no-diagnostics
#pragma mark LLVM's world
export module M;

//--- pragma_detect_mismatch.cpp
// expected-no-diagnostics
#pragma detect_mismatch("test", "1")
export module M;

//--- pragma_clang_debug.cpp
// expected-no-diagnostics
#pragma clang __debug dump Test
export module M;

//--- pragma_message.cpp
#pragma message "test" // expected-warning {{test}}
export module M;

//--- pragma_gcc_warn.cpp
#pragma GCC warning "Foo" // expected-warning {{Foo}}
export module M;

//--- pragma_gcc_error.cpp
#pragma GCC error "Foo" // expected-error {{Foo}}
export module M;

//--- pragma_diag_push_pop.cpp
// expected-no-diagnostics
#pragma gcc diagnostic push
#pragma gcc diagnostic pop
export module M;

//--- pragma_diag_ignore.cpp
// expected-no-diagnostics
#pragma GCC diagnostic ignored "-Wframe-larger-than"
export module M;

//--- pragma_opencl_ext.cpp
// expected-no-diagnostics
#pragma OPENCL EXTENSION __cl_clang_variadic_functions : enable
export module M;

//--- pragma_push_pop.cpp
// expected-no-diagnostics
#pragma warning(push)
#pragma warning(pop)
export module M;

//--- pragma_exec_charset.cpp
// expected-no-diagnostics
#pragma execution_character_set(push, "UTF-8")
#pragma execution_character_set(pop)
export module M;

//--- pragma_clang_assume_nonnull.cpp
// expected-no-diagnostics
#pragma clang assume_nonnull begin
#pragma clang assume_nonnull end
export module M;

//--- marco_expand.cpp
MACRO // expected-note {{add 'module;' to the start of the file to introduce a global module fragment}}
export module M; // expected-error {{module declaration must occur at the start of the translation unit}}

//--- define.cpp
// This is a comment
#define I32 int // expected-note {{add 'module;' to the start of the file to introduce a global module fragment}}
export module M; // expected-error {{module declaration must occur at the start of the translation unit}}
export I32 i32;

//--- undef.cpp
#undef FOO // expected-note {{add 'module;' to the start of the file to introduce a global module fragment}}
export module M; // expected-error {{module declaration must occur at the start of the translation unit}}

//--- defined.cpp
#if defined(FOO) // expected-note {{add 'module;' to the start of the file to introduce a global module fragment}}
#endif
export module M; // expected-error {{module declaration must occur at the start of the translation unit}}

//--- has_embed.cpp
#if __has_embed(__FILE__ ext::token(0xB055)) // expected-note {{add 'module;' to the start of the file to introduce a global module fragment}}
#endif
export module M; // expected-error {{module declaration must occur at the start of the translation unit}}

//--- has_include.cpp
#if __has_include(<stdio.h>) || __has_include_next(<stdlib.h>) // expected-note {{add 'module;' to the start of the file to introduce a global module fragment}} \
                                                               // expected-warning {{#include_next in primary source file; will search from start of include path}}
#endif
export module M; // expected-error {{module declaration must occur at the start of the translation unit}}
