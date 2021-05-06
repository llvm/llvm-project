// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp %s -emit-llvm -o - | FileCheck %s

void __test_offloading_42_abcdef_bar_l123();
void use(int);

void foo(int a)
{
    #pragma omp target
        use(a);

    __test_offloading_42_abcdef_bar_l123();
    int somevar_abc123_;
}
