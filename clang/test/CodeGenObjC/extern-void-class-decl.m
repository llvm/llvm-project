// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-apple-macosx10.14.0 %s -emit-llvm -o - | FileCheck %s

// rdar://45077269

extern void OBJC_CLASS_$_f;
Class c = (Class)&OBJC_CLASS_$_f;

@implementation f @end

// Check that we override the initializer for c, and that OBJC_CLASS_$_f gets
// the right definition.

// CHECK: @c ={{.*}} global i8* bitcast (%struct._class_t* @"OBJC_CLASS_$_f" to i8*)
// CHECK: @"OBJC_CLASS_$_f" ={{.*}} global %struct._class_t
