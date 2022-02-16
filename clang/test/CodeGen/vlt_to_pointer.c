// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s

int c[1][3*2];
// CHECK: @{{.+}} ={{.*}}global [1 x [6 x {{i[0-9]+}}]] zeroinitializer

// CHECK-LABEL: @f
int f(int * const m, int (**v)[*m * 2])
{
    return &(c[0][*m]) == &((*v)[0][*m]);
    // CHECK: icmp
    // CHECK: ret i{{[0-9]+}}
}

// CHECK-LABEL: @test
int test(int n, int (*(*fn)(void))[n]) {
  return (*fn())[0];
}

// CHECK-LABEL: @main
int main(void)
{
    int m = 3;
    int (*d)[3*2] = c;
    int (*fn[m])(void);
    return f(&m, &d) + test(m, &fn);

    // CHECK: call {{.+}} @f(
    // CHECK: ret i{{[0-9]+}}
}

