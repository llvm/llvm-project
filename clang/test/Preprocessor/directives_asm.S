// REQUIRES: x86-registered-target
// RUN: %clang --target=i386-pc-linux -c %s -o /dev/null 2>&1 | FileCheck %s

// Check that preprocessor directives are recognised as such, but lines starting
// with a # that aren't directives are instead treated as comments.

#define MACRO .warning "This is a macro"
        MACRO

// CHECK: directives_asm.S:8:9: warning: This is a macro

#not a preprocessing directive

// CHECK-NOT: error: invalid preprocessing directive

# 100

        .warning "line number should not change"

// CHECK: directives_asm.S:18:9: warning: line number should not change

#line 100

        .warning "line number should change"

// CHECK: directives_asm.S:101:9: warning: line number should change
