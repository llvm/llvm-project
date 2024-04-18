// RUN: %clang --target=x86_64-pc-linux -S -fno-discard-value-names -emit-llvm -o - %s | FileCheck %s

#include <new>

// CHECK: call void @llvm.memset.p0.i64(ptr align 1 %x, i8 0, i64 8, i1 false)
// CHECK: call void @llvm.memset.p0.i64(ptr align 16 %x, i8 0, i64 32, i1 false)
template <typename TYPE>
void f()
{
    typedef TYPE TArray[8];

    TArray x;
    new(&x) TArray();
}

int main()
{
    f<char>();
    f<int>();
}
