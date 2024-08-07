// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-simplify %s -o %t1.cir 2>&1 | FileCheck -check-prefix=BEFORE %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-after=cir-simplify %s -o %t2.cir 2>&1 | FileCheck -check-prefix=AFTER %s


#define CHECK_PTR(ptr)  \
  do {                   \
    if (__builtin_expect((!!((ptr) == 0)), 0))\
      return -42; \
  } while(0)

int foo(int* ptr) {  
  CHECK_PTR(ptr);

  (*ptr)++;
  return 0;
}

// BEFORE:  cir.func {{.*@foo}}
// BEFORE:  [[X0:%.*]] = cir.load {{.*}} : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// BEFORE:  [[X1:%.*]] = cir.const #cir.ptr<null> : !cir.ptr<!s32i>
// BEFORE:  [[X2:%.*]] = cir.cmp(eq, [[X0]], [[X1]]) : !cir.ptr<!s32i>, !s32i
// BEFORE:  [[X3:%.*]] = cir.cast(int_to_bool, [[X2]] : !s32i), !cir.bool
// BEFORE:  [[X4:%.*]] = cir.unary(not, [[X3]]) : !cir.bool, !cir.bool
// BEFORE:  [[X5:%.*]] = cir.cast(bool_to_int, [[X4]] : !cir.bool), !s32i
// BEFORE:  [[X6:%.*]] = cir.cast(int_to_bool, [[X5]] : !s32i), !cir.bool
// BEFORE:  [[X7:%.*]] = cir.unary(not, [[X6]]) : !cir.bool, !cir.bool
// BEFORE:  [[X8:%.*]] = cir.cast(bool_to_int, [[X7]] : !cir.bool), !s32i
// BEFORE:  [[X9:%.*]] = cir.cast(integral, [[X8]] : !s32i), !s64i
// BEFORE:  [[X10:%.*]] = cir.const #cir.int<0> : !s32i
// BEFORE:  [[X11:%.*]] = cir.cast(integral, [[X10]] : !s32i), !s64i
// BEFORE:  [[X12:%.*]] = cir.cast(int_to_bool, [[X9]] : !s64i), !cir.bool
// BEFORE:  cir.if [[X12]] 

// AFTER:   [[X0:%.*]] = cir.load {{.*}} : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// AFTER:   [[X1:%.*]] = cir.const #cir.ptr<null> : !cir.ptr<!s32i>
// AFTER:   [[X2:%.*]] = cir.cmp(eq, [[X0]], [[X1]]) : !cir.ptr<!s32i>, !s32i
// AFTER:   [[X3:%.*]] = cir.cast(int_to_bool, [[X2]] : !s32i), !cir.bool
// AFTER:   cir.if [[X3]]