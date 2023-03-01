// RUN: %clang -c %s -o /dev/null 2>&1 | FileCheck %s

// Check that preprocessor directives are recognised as such, but lines starting
// with a # that aren't directives are instead treated as comments.

#define MACRO .warning "This is a macro"
        MACRO

// CHECK: directives_asm.S:7:9: warning: This is a macro

#not a preprocessing directive

// CHECK-NOT: error: invalid preprocessing directive

# 100

        .warning "line number should not change"

// CHECK: directives_asm.S:17:9: warning: line number should not change

#line 100

        .warning "line number should change"

// CHECK: directives_asm.S:101:9: warning: line number should change
