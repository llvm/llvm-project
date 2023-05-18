// RUN: %clang_cc1 %s -emit-pch -o %t1.pch -DMACRO1=1
// RUN: %clang_cc1 -fsyntax-only %s -include-pch %t1.pch -DMACRO2=1 2>&1 | FileCheck %s

#ifndef HEADER
#define HEADER
#else
#define MACRO1 2
// CHECK: macro-cmdline.c{{.*}}'MACRO1' macro redefined
// CHECK: <command line>{{.*}}previous definition is here
#define MACRO2 2
// CHECK: macro-cmdline.c{{.*}}'MACRO2' macro redefined
// CHECK: <command line>{{.*}}previous definition is here
#endif
