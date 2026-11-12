// RUN: %clang_cc1 -E -Werror -verify %s 2>&1 | FileCheck %s

#pragma message "\\test" // expected-warning {{\test}}
// CHECK: #pragma message("\134test")

#pragma message("\\test") // expected-warning {{\test}}
// CHECK: #pragma message("\134test")

#pragma GCC warning "\"" "te" "st" "\"" // expected-warning {{"test"}}
// CHECK: #pragma GCC warning "\042test\042"

#pragma GCC warning("\"" "te" "st" "\"") // expected-warning {{"test"}}
// CHECK: #pragma GCC warning "\042test\042"

#pragma GCC error "" "[	]" "" // expected-error {{[	]}}
// CHECK: #pragma GCC error "[\011]"

#pragma GCC error("" "[	]" "") // expected-error {{[	]}}
// CHECK: #pragma GCC error "[\011]"
