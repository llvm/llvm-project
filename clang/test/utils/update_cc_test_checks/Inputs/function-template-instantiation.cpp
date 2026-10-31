// RUN: %clang_cc1 -triple=x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

template <typename T> T twice(T x) { return x + x; }
template int twice<int>(int);
template long twice<long>(long);

template <typename T> T unused(T x) { return x; }
