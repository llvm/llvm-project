// RUN: %clang_cc1 -O2 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -O2 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM


inline int s0(int a, int b) {
  int x = a + b;
  return x;
}

__attribute__((noinline))
int s1(int a, int b) {
  return s0(a,b);
}

__attribute__((always_inline))
int s2(int a, int b) {
  return s0(a,b);
}

int s3(int a, int b) {
  int x = a + b;
  return x;
}


// CIR:   cir.func linkonce_odr @_Z2s0ii(%arg0:{{.*}}, %arg1:{{.*}} -> {{.*}} extra( {inline = #cir.inline<hint>, nothrow = #cir.nothrow} )
// CIR:   cir.func @_Z2s1ii(%arg0:{{.*}}, %arg1:{{.*}} -> {{.*}} extra( {inline = #cir.inline<no>, nothrow = #cir.nothrow} )
// CIR:   cir.func @_Z2s2ii(%arg0:{{.*}}, %arg1:{{.*}} -> {{.*}} extra( {inline = #cir.inline<always>, nothrow = #cir.nothrow} )
// CIR:   cir.func @_Z2s3ii(%arg0:{{.*}}, %arg1:{{.*}} -> {{.*}} {

// LLVM: define i32 @_Z2s1ii(i32 %0, i32 %1) {{.*}} #[[#ATTR1:]]
// LLVM: define i32 @_Z2s2ii(i32 %0, i32 %1) {{.*}} #[[#ATTR2:]]
// LLVM: attributes #[[#ATTR1]] = {{.*}} noinline
// LLVM: attributes #[[#ATTR2]] = {{.*}} alwaysinline
