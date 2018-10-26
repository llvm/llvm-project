// RUN: %clang_cc1 -std=c++2a -fms-extensions -emit-llvm %s -o - -triple=i386-pc-win32 -fms-compatibility-version=19.00 | FileCheck %s

struct A {};

// CHECK-DAG: define {{.*}} @"??__M@YAXUA@@0@Z"
void operator<=>(A, A) {}
