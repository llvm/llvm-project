// expected-no-diagnostics
#ifndef HEADER
#define HEADER

#include "target_map_codegen_18.inc"
// For convenience, put the input file in a temp dir, so we don't have to use
// its full name in every FileCheck command.
// RUN: cp %S/target_map_codegen_18.inc %t.inc

// RUN: %clang_cc1 -no-opaque-pointers -DUSE -DCK19 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap --check-prefixes=CK19,CK19-32,CK19-USE %t.inc
// RUN: %clang_cc1 -no-opaque-pointers -DUSE -DCK19 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -DUSE -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap --check-prefixes=CK19,CK19-32,CK19-USE %t.inc

// RUN: %clang_cc1 -no-opaque-pointers -DUSE -DCK19 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap --check-prefixes=CK19,CK19-64,CK19-USE %t.inc
// RUN: %clang_cc1 -no-opaque-pointers -DUSE -DCK19 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -DUSE -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap --check-prefixes=CK19,CK19-64,CK19-USE %t.inc
// RUN: %clang_cc1 -no-opaque-pointers -DUSE -DCK19 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap --check-prefixes=CK19,CK19-32,CK19-USE %t.inc
// RUN: %clang_cc1 -no-opaque-pointers -DUSE -DCK19 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -DUSE -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap --check-prefixes=CK19,CK19-32,CK19-USE %t.inc

#endif
