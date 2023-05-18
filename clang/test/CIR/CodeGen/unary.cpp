// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -Wno-unused-value -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

unsigned up0() {
  unsigned a = 1;
  return +a;
}

// CHECK: cir.func @_Z3up0v() -> i32 {
// CHECK: %[[#RET:]] = cir.alloca i32, cir.ptr <i32>, ["__retval"]
// CHECK: %[[#A:]] = cir.alloca i32, cir.ptr <i32>, ["a", init]
// CHECK: %[[#INPUT:]] = cir.load %[[#A]]
// CHECK: %[[#OUTPUT:]] = cir.unary(plus, %[[#INPUT]])
// CHECK: cir.store %[[#OUTPUT]], %[[#RET]]

unsigned um0() {
  unsigned a = 1;
  return -a;
}

// CHECK: cir.func @_Z3um0v() -> i32 {
// CHECK: %[[#RET:]] = cir.alloca i32, cir.ptr <i32>, ["__retval"]
// CHECK: %[[#A:]] = cir.alloca i32, cir.ptr <i32>, ["a", init]
// CHECK: %[[#INPUT:]] = cir.load %[[#A]]
// CHECK: %[[#OUTPUT:]] = cir.unary(minus, %[[#INPUT]])
// CHECK: cir.store %[[#OUTPUT]], %[[#RET]]

unsigned un0() {
  unsigned a = 1;
  return ~a; // a ^ -1 , not
}

// CHECK: cir.func @_Z3un0v() -> i32 {
// CHECK: %[[#RET:]] = cir.alloca i32, cir.ptr <i32>, ["__retval"]
// CHECK: %[[#A:]] = cir.alloca i32, cir.ptr <i32>, ["a", init]
// CHECK: %[[#INPUT:]] = cir.load %[[#A]]
// CHECK: %[[#OUTPUT:]] = cir.unary(not, %[[#INPUT]])
// CHECK: cir.store %[[#OUTPUT]], %[[#RET]]

unsigned inc0() {
  unsigned a = 1;
  ++a;
  return a;
}

// CHECK: cir.func @_Z4inc0v() -> i32 {
// CHECK: %[[#RET:]] = cir.alloca i32, cir.ptr <i32>, ["__retval"]
// CHECK: %[[#A:]] = cir.alloca i32, cir.ptr <i32>, ["a", init]
// CHECK: %[[#ATMP:]] = cir.const(1 : i32) : i32
// CHECK: cir.store %[[#ATMP]], %[[#A]] : i32
// CHECK: %[[#INPUT:]] = cir.load %[[#A]]
// CHECK: %[[#INCREMENTED:]] = cir.unary(inc, %[[#INPUT]])
// CHECK: cir.store %[[#INCREMENTED]], %[[#A]]
// CHECK: %[[#A_TO_OUTPUT:]] = cir.load %[[#A]]
// CHECK: cir.store %[[#A_TO_OUTPUT]], %[[#RET]]
// CHECK: %[[#OUTPUT:]] = cir.load %[[#RET]]
// CHECK: cir.return %[[#OUTPUT]] : i32

unsigned dec0() {
  unsigned a = 1;
  --a;
  return a;
}

// CHECK: cir.func @_Z4dec0v() -> i32 {
// CHECK: %[[#RET:]] = cir.alloca i32, cir.ptr <i32>, ["__retval"]
// CHECK: %[[#A:]] = cir.alloca i32, cir.ptr <i32>, ["a", init]
// CHECK: %[[#ATMP:]] = cir.const(1 : i32) : i32
// CHECK: cir.store %[[#ATMP]], %[[#A]] : i32
// CHECK: %[[#INPUT:]] = cir.load %[[#A]]
// CHECK: %[[#INCREMENTED:]] = cir.unary(dec, %[[#INPUT]])
// CHECK: cir.store %[[#INCREMENTED]], %[[#A]]
// CHECK: %[[#A_TO_OUTPUT:]] = cir.load %[[#A]]
// CHECK: cir.store %[[#A_TO_OUTPUT]], %[[#RET]]
// CHECK: %[[#OUTPUT:]] = cir.load %[[#RET]]
// CHECK: cir.return %[[#OUTPUT]] : i32

unsigned inc1() {
  unsigned a = 1;
  a++;
  return a;
}

// CHECK: cir.func @_Z4inc1v() -> i32 {
// CHECK: %[[#RET:]] = cir.alloca i32, cir.ptr <i32>, ["__retval"]
// CHECK: %[[#A:]] = cir.alloca i32, cir.ptr <i32>, ["a", init]
// CHECK: %[[#ATMP:]] = cir.const(1 : i32) : i32
// CHECK: cir.store %[[#ATMP]], %[[#A]] : i32
// CHECK: %[[#INPUT:]] = cir.load %[[#A]]
// CHECK: %[[#INCREMENTED:]] = cir.unary(inc, %[[#INPUT]])
// CHECK: cir.store %[[#INCREMENTED]], %[[#A]]
// CHECK: %[[#A_TO_OUTPUT:]] = cir.load %[[#A]]
// CHECK: cir.store %[[#A_TO_OUTPUT]], %[[#RET]]
// CHECK: %[[#OUTPUT:]] = cir.load %[[#RET]]
// CHECK: cir.return %[[#OUTPUT]] : i32

unsigned dec1() {
  unsigned a = 1;
  a--;
  return a;
}

// CHECK: cir.func @_Z4dec1v() -> i32 {
// CHECK: %[[#RET:]] = cir.alloca i32, cir.ptr <i32>, ["__retval"]
// CHECK: %[[#A:]] = cir.alloca i32, cir.ptr <i32>, ["a", init]
// CHECK: %[[#ATMP:]] = cir.const(1 : i32) : i32
// CHECK: cir.store %[[#ATMP]], %[[#A]] : i32
// CHECK: %[[#INPUT:]] = cir.load %[[#A]]
// CHECK: %[[#INCREMENTED:]] = cir.unary(dec, %[[#INPUT]])
// CHECK: cir.store %[[#INCREMENTED]], %[[#A]]
// CHECK: %[[#A_TO_OUTPUT:]] = cir.load %[[#A]]
// CHECK: cir.store %[[#A_TO_OUTPUT]], %[[#RET]]
// CHECK: %[[#OUTPUT:]] = cir.load %[[#RET]]
// CHECK: cir.return %[[#OUTPUT]] : i32

// Ensure the increment is performed after the assignment to b.
unsigned inc2() {
  unsigned a = 1;
  unsigned b = a++;
  return b;
}

// CHECK: cir.func @_Z4inc2v() -> i32 {
// CHECK: %[[#RET:]] = cir.alloca i32, cir.ptr <i32>, ["__retval"]
// CHECK: %[[#A:]] = cir.alloca i32, cir.ptr <i32>, ["a", init]
// CHECK: %[[#B:]] = cir.alloca i32, cir.ptr <i32>, ["b", init]
// CHECK: %[[#ATMP:]] = cir.const(1 : i32) : i32
// CHECK: cir.store %[[#ATMP]], %[[#A]] : i32
// CHECK: %[[#ATOB:]] = cir.load %[[#A]]
// CHECK: %[[#INCREMENTED:]] = cir.unary(inc, %[[#ATOB]])
// CHECK: cir.store %[[#INCREMENTED]], %[[#A]]
// CHECK: cir.store %[[#ATOB]], %[[#B]]
// CHECK: %[[#B_TO_OUTPUT:]] = cir.load %[[#B]]
// CHECK: cir.store %[[#B_TO_OUTPUT]], %[[#RET]]
// CHECK: %[[#OUTPUT:]] = cir.load %[[#RET]]
// CHECK: cir.return %[[#OUTPUT]] : i32

int *inc_p(int *i) {
  --i;
  ++i;
  return i;
}

// CHECK: cir.func @_Z5inc_pPi(%arg0: !cir.ptr<i32>

// CHECK:   %[[#i_addr:]] = cir.alloca !cir.ptr<i32>, cir.ptr <!cir.ptr<i32>>, ["i", init] {alignment = 8 : i64}
// CHECK:   %[[#i_dec:]] = cir.load %[[#i_addr]] : cir.ptr <!cir.ptr<i32>>, !cir.ptr<i32>
// CHECK:   %[[#dec_const:]] = cir.const(-1 : i32) : i32
// CHECK:   = cir.ptr_stride(%[[#i_dec]] : !cir.ptr<i32>, %[[#dec_const]] : i32), !cir.ptr<i32>

// CHECK:   %[[#i_inc:]] = cir.load %0 : cir.ptr <!cir.ptr<i32>>, !cir.ptr<i32>
// CHECK:   %[[#inc_const:]] = cir.const(1 : i32) : i32
// CHECK:   = cir.ptr_stride(%[[#i_inc]] : !cir.ptr<i32>, %[[#inc_const]] : i32), !cir.ptr<i32>