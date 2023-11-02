// RUN: %clang_cc1 -O2 -triple wasm32-unknown-unknown-wasm -emit-llvm -o - %s | FileCheck %s

// From issue 69894. Reftypes need to be marked as not valid as vector elements.
__externref_t foo(void);
void bar(__externref_t);

void
test(int flag, __externref_t ref1, __externref_t ref2)
{
  if (flag) {
    ref1 = foo();
    ref2 = foo();
  }
  bar(ref1);
  bar(ref2);
}