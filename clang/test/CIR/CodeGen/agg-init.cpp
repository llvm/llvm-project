// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -Wno-unused-value -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// CHECK !ty_22struct2EZero22 = !cir.struct<"struct.Zero", i8>
// CHECK !ty_22struct2Eyep_22 = !cir.struct<"struct.yep_", i32, i32>

struct Zero {
  void yolo(); 
};

void f() {
  Zero z0 = Zero();
  // {} no element init.
  Zero z1 = Zero{};
}

// CHECK: cir.func @_Z1fv() {
// CHECK:     %0 = cir.alloca !ty_22struct2EZero22, cir.ptr <!ty_22struct2EZero22>, ["z0", init]
// CHECK:     %1 = cir.alloca !ty_22struct2EZero22, cir.ptr <!ty_22struct2EZero22>, ["z1"]
// CHECK:     cir.call @_ZN4ZeroC1Ev(%0) : (!cir.ptr<!ty_22struct2EZero22>) -> ()
// CHECK:     cir.return

typedef enum xxy_ {
  xxy_Low = 0,
  xxy_High = 0x3f800000,
  xxy_EnumSize = 0x7fffffff
} xxy;

typedef struct yep_ {
  unsigned int Status;
  xxy HC;
} yop;

void use() { yop{}; }

// CHECK: cir.func @_Z3usev() {
// CHECK:   %0 = cir.alloca !ty_22struct2Eyep_22, cir.ptr <!ty_22struct2Eyep_22>, ["agg.tmp0"] {alignment = 4 : i64}
// CHECK:   %1 = "cir.struct_element_addr"(%0) <{member_name = "Status"}> : (!cir.ptr<!ty_22struct2Eyep_22>) -> !cir.ptr<i32>
// CHECK:   %2 = cir.const(0 : i32) : i32
// CHECK:   cir.store %2, %1 : i32, cir.ptr <i32>
// CHECK:   %3 = "cir.struct_element_addr"(%0) <{member_name = "HC"}> : (!cir.ptr<!ty_22struct2Eyep_22>) -> !cir.ptr<i32>
// CHECK:   %4 = cir.const(0 : i32) : i32
// CHECK:   cir.store %4, %3 : i32, cir.ptr <i32>
// CHECK:   cir.return
// CHECK: }
