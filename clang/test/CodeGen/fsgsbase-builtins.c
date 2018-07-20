// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +fsgsbase -emit-llvm -o - | FileCheck %s


#include <immintrin.h>

unsigned int test_readfsbase_u32()
{
  // CHECK: @llvm.x86.rdfsbase.32
  return _readfsbase_u32();
}

unsigned long long test_readfsbase_u64()
{
  // CHECK: @llvm.x86.rdfsbase.64
  return _readfsbase_u64();
}

unsigned int test_readgsbase_u32()
{
  // CHECK: @llvm.x86.rdgsbase.32
  return _readgsbase_u32();
}

unsigned long long test_readgsbase_u64()
{
  // CHECK: @llvm.x86.rdgsbase.64
  return _readgsbase_u64();
}

void test_writefsbase_u32(unsigned int __X)
{
  // CHECK: @llvm.x86.wrfsbase.32
  _writefsbase_u32(__X);
}

void test_writefsbase_u64(unsigned long long __X)
{
  // CHECK: @llvm.x86.wrfsbase.64
  _writefsbase_u64(__X);
}

void test_writegsbase_u32(unsigned int __X)
{
  // CHECK: @llvm.x86.wrgsbase.32
  _writegsbase_u32(__X);
}

void test_writegsbase_u64(unsigned long long __X)
{
  // CHECK: @llvm.x86.wrgsbase.64
  _writegsbase_u64(__X);
}
