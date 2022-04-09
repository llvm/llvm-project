// RUN: %clang_cc1 -no-opaque-pointers -triple i686-apple-darwin -emit-llvm %s -o - | FileCheck %s
// rdar://7589850

// CHECK: @.str = private unnamed_addr constant [9 x i16] [i16 103, i16 111, i16 111, i16 100, i16 0, i16 98, i16 121, i16 101, i16 0], section "__TEXT,__ustring", align 2
// CHECK: @_unnamed_cfstring_ = private global %struct.__NSConstantString_tag { i32* getelementptr inbounds ([0 x i32], [0 x i32]* @__CFConstantStringClassReference, i32 0, i32 0), i32 2000, i8* bitcast ([9 x i16]* @.str to i8*), i32 8 }, section "__DATA,__cfstring"
// CHECK: @P ={{.*}} global i8* bitcast (%struct.__NSConstantString_tag* @_unnamed_cfstring_ to i8*), align 4
void *P = @"good\0bye";
