// REQUIRES: x86-registered-target

// RUN: %clang_cc1 -triple x86_64-macho -emit-llvm -w -x objective-c %s -o - | FileCheck %s

// CHECK: private unnamed_addr constant [6 x i16] [i16 105, i16 80, i16 111, i16 100, i16 8482, i16 0], section "__TEXT,__ustring", align 2
void *P = @"iPod™";
