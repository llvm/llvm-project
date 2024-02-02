// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

int x(int y) {
  return y > 0 ? 3 : 5;
}

// CHECK: cir.func @_Z1xi
// CHECK:     %0 = cir.alloca !s32i, cir.ptr <!s32i>, ["y", init] {alignment = 4 : i64}
// CHECK:     %1 = cir.alloca !s32i, cir.ptr <!s32i>, ["__retval"] {alignment = 4 : i64}
// CHECK:     cir.store %arg0, %0 : !s32i, cir.ptr <!s32i>
// CHECK:     %2 = cir.load %0 : cir.ptr <!s32i>, !s32i
// CHECK:     %3 = cir.const(#cir.int<0> : !s32i) : !s32i
// CHECK:     %4 = cir.cmp(gt, %2, %3) : !s32i, !cir.bool
// CHECK:     %5 = cir.ternary(%4, true {
// CHECK:       %7 = cir.const(#cir.int<3> : !s32i) : !s32i
// CHECK:       cir.yield %7 : !s32i
// CHECK:     }, false {
// CHECK:       %7 = cir.const(#cir.int<5> : !s32i) : !s32i
// CHECK:       cir.yield %7 : !s32i
// CHECK:     }) : (!cir.bool) -> !s32i
// CHECK:     cir.store %5, %1 : !s32i, cir.ptr <!s32i>
// CHECK:     %6 = cir.load %1 : cir.ptr <!s32i>, !s32i
// CHECK:     cir.return %6 : !s32i
// CHECK:   }

typedef enum {
  API_A,
  API_EnumSize = 0x7fffffff
} APIType;

void oba(const char *);

void m(APIType api) {
  ((api == API_A) ? (static_cast<void>(0)) : oba("yo.cpp"));
}

// CHECK:  cir.func @_Z1m7APIType
// CHECK:    %0 = cir.alloca !u32i, cir.ptr <!u32i>, ["api", init] {alignment = 4 : i64}
// CHECK:    cir.store %arg0, %0 : !u32i, cir.ptr <!u32i>
// CHECK:    %1 = cir.load %0 : cir.ptr <!u32i>, !u32i
// CHECK:    %2 = cir.cast(integral, %1 : !u32i), !s32i
// CHECK:    %3 = cir.const(#cir.int<0> : !u32i) : !u32i
// CHECK:    %4 = cir.cast(integral, %3 : !u32i), !s32i
// CHECK:    %5 = cir.cmp(eq, %2, %4) : !s32i, !cir.bool
// CHECK:    cir.ternary(%5, true {
// CHECK:      %6 = cir.const(#cir.int<0> : !s32i) : !s32i
// CHECK:      cir.yield
// CHECK:    }, false {
// CHECK:      %6 = cir.get_global @".str" : cir.ptr <!cir.array<!s8i x 7>>
// CHECK:      %7 = cir.cast(array_to_ptrdecay, %6 : !cir.ptr<!cir.array<!s8i x 7>>), !cir.ptr<!s8i>
// CHECK:      cir.call @_Z3obaPKc(%7) : (!cir.ptr<!s8i>) -> ()
// CHECK:      cir.yield
// CHECK:    }) : (!cir.bool) -> ()
// CHECK:    cir.return
// CHECK:  }

int foo(int a, int b) {
  if (a < b ? 0 : a)
    return -1;
  return 0;
}

// CHECK:  cir.func @_Z3fooii
// CHECK:   [[A0:%.*]] = cir.load {{.*}} : cir.ptr <!s32i>, !s32i
// CHECK:   [[B0:%.*]] = cir.load {{.*}} : cir.ptr <!s32i>, !s32i
// CHECK:   [[CMP:%.*]] = cir.cmp(lt, [[A0]], [[B0]]) : !s32i, !cir.bool
// CHECK:   [[RES:%.*]] = cir.ternary([[CMP]], true {
// CHECK:     [[ZERO:%.*]] = cir.const(#cir.int<0> : !s32i) : !s32i
// CHECK:     cir.yield [[ZERO]] : !s32i
// CHECK:   }, false {
// CHECK:     [[A1:%.*]] = cir.load {{.*}} : cir.ptr <!s32i>, !s32i
// CHECK:     cir.yield [[A1]] : !s32i
// CHECK:   }) : (!cir.bool) -> !s32i
// CHECK:   [[RES_CAST:%.*]] = cir.cast(int_to_bool, [[RES]] : !s32i), !cir.bool
// CHECK:   cir.if [[RES_CAST]]
