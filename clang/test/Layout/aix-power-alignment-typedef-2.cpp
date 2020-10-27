// RUN: %clang_cc1 -triple powerpc-ibm-aix-xcoff -S -emit-llvm -x c++ < %s | \
// RUN:   FileCheck %s

// RUN: %clang_cc1 -triple powerpc64-ibm-aix-xcoff -S -emit-llvm -x c++ < %s | \
// RUN:   FileCheck %s

struct C {
  double x;
};

typedef struct C __attribute__((__aligned__(2))) CC;

CC cc;

// CHECK: @cc = global %struct.C zeroinitializer, align 2
