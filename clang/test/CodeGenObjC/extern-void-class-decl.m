// RUN: %clang_cc1 -triple x86_64-apple-macosx10.14.0 %s -emit-llvm -o - | FileCheck %s

// rdar://45077269

extern void OBJC_CLASS_$_f;
Class c = (Class)&OBJC_CLASS_$_f;

@implementation f @end

// Check that we override the initializer for c, and that OBJC_CLASS_$_f gets
// the right definition.

// CHECK: @c ={{.*}} global ptr @"OBJC_CLASS_$_f"
// CHECK: @"OBJC_CLASS_$_f" ={{.*}} global %struct._class_t
