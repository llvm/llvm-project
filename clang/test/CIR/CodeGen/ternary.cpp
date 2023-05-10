// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

int x(int y) {
  return y > 0 ? 3 : 5;
}

// CHECK: cir.func @_Z1xi
// CHECK:     %0 = cir.alloca i32, cir.ptr <i32>, ["y", init] {alignment = 4 : i64}
// CHECK:     %1 = cir.alloca i32, cir.ptr <i32>, ["__retval"] {alignment = 4 : i64}
// CHECK:     cir.store %arg0, %0 : i32, cir.ptr <i32>
// CHECK:     %2 = cir.load %0 : cir.ptr <i32>, i32
// CHECK:     %3 = cir.const(0 : i32) : i32
// CHECK:     %4 = cir.cmp(gt, %2, %3) : i32, !cir.bool
// CHECK:     %5 = cir.ternary(%4, true {
// CHECK:       %7 = cir.const(3 : i32) : i32
// CHECK:       cir.yield %7 : i32
// CHECK:     }, false {
// CHECK:       %7 = cir.const(5 : i32) : i32
// CHECK:       cir.yield %7 : i32
// CHECK:     }) : i32
// CHECK:     cir.store %5, %1 : i32, cir.ptr <i32>
// CHECK:     %6 = cir.load %1 : cir.ptr <i32>, i32
// CHECK:     cir.return %6 : i32
// CHECK:   }

typedef enum {
  API_A,
  API_EnumSize = 0x7fffffff
} APIType;

void oba(const char *);

void m(APIType api) {
  ((api == API_A) ? (static_cast<void>(0)) : oba("yo.cpp"));
}

// CHECK: cir.func @_Z1m7APIType
// CHECK:     %0 = cir.alloca i32, cir.ptr <i32>, ["api", init] {alignment = 4 : i64}
// CHECK:     cir.store %arg0, %0 : i32, cir.ptr <i32>
// CHECK:     %1 = cir.load %0 : cir.ptr <i32>, i32
// CHECK:     %2 = cir.const(0 : i32) : i32
// CHECK:     %3 = cir.cmp(eq, %1, %2) : i32, !cir.bool
// CHECK:     %4 = cir.ternary(%3, true {
// CHECK:       %5 = cir.const(0 : i32) : i32
// CHECK:       %6 = cir.const(0 : i8) : i8
// CHECK:       cir.yield %6 : i8
// CHECK:     }, false {
// CHECK:       %5 = cir.get_global @".str" : cir.ptr <!cir.array<i8 x 7>>
// CHECK:       %6 = cir.cast(array_to_ptrdecay, %5 : !cir.ptr<!cir.array<i8 x 7>>), !cir.ptr<i8>
// CHECK:       cir.call @_Z3obaPKc(%6) : (!cir.ptr<i8>) -> ()
// CHECK:       %7 = cir.const(0 : i8) : i8
// CHECK:       cir.yield %7 : i8
// CHECK:     }) : i8
// CHECK:     cir.return
// CHECK:   }