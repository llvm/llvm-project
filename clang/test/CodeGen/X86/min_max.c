// RUN: %clang_cc1 %s -O2 -triple=x86_64-apple-darwin -emit-llvm -o - | FileCheck %s

short vecreduce_smax_v2i16(int n, short* v)
{
  // CHECK: @llvm.smax
  short p = 0;
  for (int i = 0; i < n; ++i)
    p = p < v[i] ? v[i] : p;
  return p;
}

short vecreduce_smin_v2i16(int n, short* v)
{
  // CHECK: @llvm.smin
  short p = 0;
  for (int i = 0; i < n; ++i)
    p = p > v[i] ? v[i] : p;
  return p;
}