// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple aarch64-none-linux-android21 -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

__attribute__((nothrow))
int s0(int a, int b) {
  int x = a + b;
  return x;
}

__attribute__((noinline))
int s1(int a, int b) {
  return s0(a,b);
}

int s2(int a, int b) {
  return s1(a, b);
}

// CIR: #fn_attr = #cir<extra({inline = #cir.inline<no>, nothrow = #cir.nothrow, optnone = #cir.optnone})>
// CIR: #fn_attr1 = #cir<extra({nothrow = #cir.nothrow})>

// CIR: cir.func @_Z2s0ii(%{{.*}}, %{{.*}}) -> {{.*}} extra(#fn_attr)
// CIR: cir.func @_Z2s1ii(%{{.*}}, %{{.*}}) -> {{.*}} extra(#fn_attr)
// CIR: cir.call @_Z2s0ii(%{{.*}}, %{{.*}}) : ({{.*}}, {{.*}}) -> {{.*}} extra(#fn_attr1)
// CIR: cir.func @_Z2s2ii(%{{.*}}, %{{.*}}) -> {{.*}} extra(#fn_attr)
// CHECK-NOT: cir.call @_Z2s1ii(%{{.*}}, %{{.*}}) : ({{.*}}, {{.*}}) -> {{.*}} extra(#fn_attr{{.*}})

// LLVM: define dso_local i32 @_Z2s0ii(i32 %0, i32 %1) #[[#ATTR1:]]
// LLVM: define dso_local i32 @_Z2s1ii(i32 %0, i32 %1) #[[#ATTR1:]]
// LLVM: define dso_local i32 @_Z2s2ii(i32 %0, i32 %1) #[[#ATTR1:]]

// LLVM: attributes #[[#ATTR1]] = {{.*}} noinline nounwind optnone
