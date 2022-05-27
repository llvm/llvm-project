// RUN: rm -rf %t.dir
// RUN: rm -rf %t.cdb
// RUN: mkdir -p %t.dir
// RUN: cp %s %t.dir/has_include_if_elif2.cpp
// RUN: cp %s %t.dir/has_include_if_elif2_clangcl.cpp
// RUN: mkdir %t.dir/Inputs
// RUN: cp %S/Inputs/header.h %t.dir/Inputs/header.h
// RUN: cp %S/Inputs/header.h %t.dir/Inputs/header2.h
// RUN: cp %S/Inputs/header.h %t.dir/Inputs/header3.h
// RUN: cp %S/Inputs/header.h %t.dir/Inputs/header4.h
// RUN: sed -e "s|DIR|%/t.dir|g" %S/Inputs/has_include_if_elif.json > %t.cdb
//
// RUN: clang-scan-deps -compilation-database %t.cdb -j 1 -mode preprocess-dependency-directives | \
// RUN:   FileCheck %s
// RUN: clang-scan-deps -compilation-database %t.cdb -j 1 -mode preprocess | \
// RUN:   FileCheck %s

#if __has_include("header.h")
#endif

#if 0
#elif __has_include("header2.h")
#endif

#define H3 __has_include("header3.h")
#if H3
#endif

#define H4 __has_include("header4.h")

#if 0
#elif H4
#endif

// CHECK: has_include_if_elif2.cpp
// CHECK-NEXT: Inputs{{/|\\}}header.h
// CHECK-NEXT: Inputs{{/|\\}}header2.h
// CHECK-NEXT: Inputs{{/|\\}}header3.h
// CHECK-NEXT: Inputs{{/|\\}}header4.h

// CHECK: has_include_if_elif2_clangcl.cpp
// CHECK-NEXT: Inputs{{/|\\}}header.h
// CHECK-NEXT: Inputs{{/|\\}}header2.h
// CHECK-NEXT: Inputs{{/|\\}}header3.h
// CHECK-NEXT: Inputs{{/|\\}}header4.h
