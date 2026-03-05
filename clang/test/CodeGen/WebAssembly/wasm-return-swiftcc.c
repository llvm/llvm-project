// RUN: %clang_cc1 -triple wasm32-unknown-unknown %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple wasm64-unknown-unknown %s -emit-llvm -target-abi experimental-mv -o - | FileCheck %s -check-prefix=EXPERIMENTAL-MV

typedef struct {
  int aa;
  int bb;
} s1;

// Multiple-element structs should be returned through sret.
// CHECK: define swiftcc void @return_s1(ptr dead_on_unwind noalias writable sret(%struct.s1) align 4 %agg.result)
// EXPERIMENTAL-MV: define swiftcc i64 @return_s1()
__attribute__((swiftcall))
s1 return_s1(void) {
  s1 foo;
  return foo;
}

typedef struct {
  int cc;
} s2;

// Single-element structs should be returned directly.
// CHECK: define swiftcc i32 @return_s2()
// EXPERIMENTAL-MV: define swiftcc i32 @return_s2()
__attribute__((swiftcall))
s2 return_s2(void) {
  s2 foo;
  return foo;
}

typedef struct {
    char c1[4];
} s3;

// CHECK: define swiftcc i32 @return_s3()
// EXPERIMENTAL-MV: define swiftcc i32 @return_s3()
__attribute__((swiftcall))
s3 return_s3(void) {
  s3 foo;
  return foo;
}

typedef struct {
    int bf1 : 4;
    int bf2 : 3;
    int bf3 : 8;
} s4;

// CHECK: define swiftcc i16 @return_s4()
// EXPERIMENTAL-MV: define swiftcc i16 @return_s4()
__attribute__((swiftcall))
s4 return_s4(void) {
  s4 foo;
  return foo;
}

// Single-element structs fitting in a i64 should be returned directly.
typedef struct {
    long long v;
} s5;

// CHECK: define swiftcc i64 @return_s5()
// EXPERIMENTAL-MV: define swiftcc i64 @return_s5()
__attribute__((swiftcall))
s5 return_s5(void) {
  s5 foo;
  return foo;
}

// Multiple-element structs not fitting in a i64
typedef struct {
    long long v1;
    long long v2;
} s6;

// CHECK: define swiftcc void @return_s6(ptr dead_on_unwind noalias writable sret(%struct.s6) align 8 %agg.result)
// EXPERIMENTAL-MV: define swiftcc { i64, i64 } @return_s6()
__attribute__((swiftcall))
s6 return_s6(void) {
  s6 foo;
  return foo;
}
