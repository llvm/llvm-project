// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s

// CHECK-LABEL: @main

struct dyn_array { 
    int size;
    int data[];
};

int foo(dyn_array **&d) {
  return (*d)->data[1];
}

int main()
{
    dyn_array **d;
    return foo(d);

    // CHECK: call {{.+}} @{{.+}}foo{{.+}}(
    // CHECK: ret i{{[0-9]+}}
}

