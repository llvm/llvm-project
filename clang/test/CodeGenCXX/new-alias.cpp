// RUN: %clang_cc1 -emit-llvm -triple x86_64-linux-gnu -std=c++11 -o - %s | FileCheck %s

using size_t = decltype(sizeof(0));

extern "C" char *something(long long x) {
}

// CHECK: @_Znwm ={{.*}} alias ptr (i64), ptr @something
void *operator new(size_t) __attribute__((alias("something")));

// PR16715: don't assert here.
// CHECK: call noalias noundef nonnull ptr @_Znwm(i64 noundef 4) #3{{$}}
int *pr16715 = new int;
