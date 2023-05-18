// RUN: %clang_cc1 -triple thumbv7-apple-ios10.0.0 -emit-llvm -fvisibility=hidden -w %s -o - | FileCheck %s

// CHECK: @"OBJC_CLASS_$_a" = hidden global %struct._class_t

__attribute__((availability(ios, introduced = 11.0))) @interface a @end
    @implementation a @end
