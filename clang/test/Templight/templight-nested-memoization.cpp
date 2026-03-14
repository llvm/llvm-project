// RUN: %clang_cc1 -templight-dump %s 2>&1 | FileCheck %s

template <int N>
struct fib
{
  static const int value = fib<N-1>::value + fib<N-2>::value;
};

template <>
struct fib<0>
{
  static const int value = 1;
};

template <>
struct fib<1>
{
  static const int value = 1;
};

// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'fib<4>'$}}
// CHECK: {{^kind:[ ]+TemplateInstantiation$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-nested-memoization.cpp:4:8'}}
// CHECK: {{^poi:[ ]+'.*templight-nested-memoization.cpp:173:8'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'fib<4>'$}}
// CHECK: {{^kind:[ ]+TemplateInstantiation$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-nested-memoization.cpp:4:8'}}
// CHECK: {{^poi:[ ]+'.*templight-nested-memoization.cpp:173:8'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'fib<4>'$}}
// CHECK: {{^kind:[ ]+TemplateInstantiation$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-nested-memoization.cpp:4:8'}}
// CHECK: {{^poi:[ ]+'.*templight-nested-memoization.cpp:173:8'$}}
//
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'fib<3>'$}}
// CHECK: {{^kind:[ ]+TemplateInstantiation$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-nested-memoization.cpp:4:8'}}
// CHECK: {{^poi:[ ]+'.*templight-nested-memoization.cpp:6:28'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'fib<3>'$}}
// CHECK: {{^kind:[ ]+TemplateInstantiation$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-nested-memoization.cpp:4:8'}}
// CHECK: {{^poi:[ ]+'.*templight-nested-memoization.cpp:6:28'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'fib<3>'$}}
// CHECK: {{^kind:[ ]+TemplateInstantiation$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-nested-memoization.cpp:4:8'}}
// CHECK: {{^poi:[ ]+'.*templight-nested-memoization.cpp:6:28'$}}
//
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'fib<2>'$}}
// CHECK: {{^kind:[ ]+TemplateInstantiation$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-nested-memoization.cpp:4:8'}}
// CHECK: {{^poi:[ ]+'.*templight-nested-memoization.cpp:6:28'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'fib<2>'$}}
// CHECK: {{^kind:[ ]+TemplateInstantiation$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-nested-memoization.cpp:4:8'}}
// CHECK: {{^poi:[ ]+'.*templight-nested-memoization.cpp:6:28'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'fib<2>'$}}
// CHECK: {{^kind:[ ]+TemplateInstantiation$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-nested-memoization.cpp:4:8'}}
// CHECK: {{^poi:[ ]+'.*templight-nested-memoization.cpp:6:28'$}}

// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'fib<1>'$}}
// CHECK: {{^kind:[ ]+Memoization$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-nested-memoization.cpp:16:8'}}
// CHECK: {{^poi:[ ]+'.*templight-nested-memoization.cpp:6:28'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'fib<1>'$}}
// CHECK: {{^kind:[ ]+Memoization$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-nested-memoization.cpp:16:8'}}
// CHECK: {{^poi:[ ]+'.*templight-nested-memoization.cpp:6:28'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'fib<0>'$}}
// CHECK: {{^kind:[ ]+Memoization$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-nested-memoization.cpp:10:8'}}
// CHECK: {{^poi:[ ]+'.*templight-nested-memoization.cpp:6:46'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'fib<0>'$}}
// CHECK: {{^kind:[ ]+Memoization$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-nested-memoization.cpp:10:8'}}
// CHECK: {{^poi:[ ]+'.*templight-nested-memoization.cpp:6:46'$}}
//
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'fib<2>'$}}
// CHECK: {{^kind:[ ]+TemplateInstantiation$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-nested-memoization.cpp:4:8'}}
// CHECK: {{^poi:[ ]+'.*templight-nested-memoization.cpp:6:28'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'fib<2>'$}}
// CHECK: {{^kind:[ ]+Memoization$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-nested-memoization.cpp:4:8'}}
// CHECK: {{^poi:[ ]+'.*templight-nested-memoization.cpp:6:28'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'fib<2>'$}}
// CHECK: {{^kind:[ ]+Memoization$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-nested-memoization.cpp:4:8'}}
// CHECK: {{^poi:[ ]+'.*templight-nested-memoization.cpp:6:28'$}}
//
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'fib<1>'$}}
// CHECK: {{^kind:[ ]+Memoization$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-nested-memoization.cpp:16:8'}}
// CHECK: {{^poi:[ ]+'.*templight-nested-memoization.cpp:6:46'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'fib<1>'$}}
// CHECK: {{^kind:[ ]+Memoization$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-nested-memoization.cpp:16:8'}}
// CHECK: {{^poi:[ ]+'.*templight-nested-memoization.cpp:6:46'$}}
//
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'fib<3>'$}}
// CHECK: {{^kind:[ ]+TemplateInstantiation$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-nested-memoization.cpp:4:8'}}
// CHECK: {{^poi:[ ]+'.*templight-nested-memoization.cpp:6:28'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'fib<3>'$}}
// CHECK: {{^kind:[ ]+Memoization$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-nested-memoization.cpp:4:8'}}
// CHECK: {{^poi:[ ]+'.*templight-nested-memoization.cpp:6:28'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'fib<3>'$}}
// CHECK: {{^kind:[ ]+Memoization$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-nested-memoization.cpp:4:8'}}
// CHECK: {{^poi:[ ]+'.*templight-nested-memoization.cpp:6:28'$}}
//
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'fib<2>'$}}
// CHECK: {{^kind:[ ]+Memoization$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-nested-memoization.cpp:4:8'}}
// CHECK: {{^poi:[ ]+'.*templight-nested-memoization.cpp:6:46'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'fib<2>'$}}
// CHECK: {{^kind:[ ]+Memoization$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-nested-memoization.cpp:4:8'}}
// CHECK: {{^poi:[ ]+'.*templight-nested-memoization.cpp:6:46'$}}
// CHECK-LABEL: {{^---$}}
//
// CHECK: {{^name:[ ]+'fib<4>'$}}
// CHECK: {{^kind:[ ]+TemplateInstantiation$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-nested-memoization.cpp:4:8'}}
// CHECK: {{^poi:[ ]+'.*templight-nested-memoization.cpp:173:8'$}}
fib<4> x;

