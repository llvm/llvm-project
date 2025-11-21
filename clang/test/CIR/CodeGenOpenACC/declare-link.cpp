// RUN: %clang_cc1 -fopenacc -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir %s -o - | FileCheck %s

struct HasSideEffects {
  HasSideEffects();
  ~HasSideEffects();
};

// TODO: OpenACC: Implement 'global', NS lowering.

struct Struct {
  static const HasSideEffects StaticMemHSE;
  static const HasSideEffects StaticMemHSEArr[5];
  static const int StaticMemInt;

  // TODO: OpenACC: Implement static-local lowering.

  void MemFunc1() {
    // CHECK: cir.func {{.*}}MemFunc1{{.*}}({{.*}}) {
    // CHECK-NEXT: cir.alloca{{.*}}["this"
    // CHECK-NEXT: cir.store
    // CHECK-NEXT: cir.load
    extern HasSideEffects LocalHSE;
    extern HasSideEffects LocalHSEArr[5];
    extern int LocalInt;
#pragma acc declare link(LocalHSE, LocalInt, LocalHSEArr[1:1])

    // CHECK-NEXT: %[[GET_LOCAL_HSE:.*]] = cir.get_global @LocalHSE : !cir.ptr<!rec_HasSideEffects>
    // CHECK-NEXT: %[[HSE_LINK:.*]] = acc.declare_link varPtr(%[[GET_LOCAL_HSE]] : !cir.ptr<!rec_HasSideEffects>) -> !cir.ptr<!rec_HasSideEffects> {name = "LocalHSE"}
    //
    // CHECK-NEXT: %[[GET_LOCAL_INT:.*]] = cir.get_global @LocalInt : !cir.ptr<!s32i>
    // CHECK-NEXT: %[[INT_LINK:.*]] = acc.declare_link varPtr(%[[GET_LOCAL_INT]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {name = "LocalInt"}
    //
    // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
    // CHECK-NEXT: %[[LB:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
    // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
    // CHECK-NEXT: %[[UB:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
    // CHECK-NEXT: %[[ZERO:.*]] = arith.constant 0 : i64
    // CHECK-NEXT: %[[ONE:.*]] = arith.constant 1 : i64
    // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[LB]] : si32) extent(%[[UB]] : si32) stride(%[[ONE]] : i64) startIdx(%[[ZERO]] : i64)
    // CHECK-NEXT: %[[GET_LOCAL_ARR:.*]] = cir.get_global @LocalHSEArr : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>
    // CHECK-NEXT: %[[ARR_LINK:.*]] = acc.declare_link varPtr(%[[GET_LOCAL_ARR]] : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!rec_HasSideEffects x 5>> {name = "LocalHSEArr[1:1]"}
    //
    // CHECK-NEXT: %[[ENTER:.*]] = acc.declare_enter dataOperands(%[[HSE_LINK]], %[[INT_LINK]], %[[ARR_LINK]] : !cir.ptr<!rec_HasSideEffects>, !cir.ptr<!s32i>, !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>)
    //
    // CHECK-NEXT: acc.declare_exit token(%[[ENTER]])
  }

  void MemFunc2();
};
void use() {
  Struct s;
  s.MemFunc1();
}

void Struct::MemFunc2() {
    // CHECK: cir.func {{.*}}MemFunc2{{.*}}({{.*}}) {
    // CHECK-NEXT: cir.alloca{{.*}}["this"
    // CHECK-NEXT: cir.store
    // CHECK-NEXT: cir.load
    extern HasSideEffects LocalHSE2;
    extern HasSideEffects LocalHSEArr2[5];
    extern int LocalInt2;

#pragma acc declare link(LocalHSE2, LocalInt2, LocalHSEArr2[1:1])
    // CHECK-NEXT: %[[GET_LOCAL_HSE:.*]] = cir.get_global @LocalHSE2 : !cir.ptr<!rec_HasSideEffects>
    // CHECK-NEXT: %[[HSE_LINK:.*]] = acc.declare_link varPtr(%[[GET_LOCAL_HSE]] : !cir.ptr<!rec_HasSideEffects>) -> !cir.ptr<!rec_HasSideEffects> {name = "LocalHSE2"}
    //
    // CHECK-NEXT: %[[GET_LOCAL_INT:.*]] = cir.get_global @LocalInt2 : !cir.ptr<!s32i>
    // CHECK-NEXT: %[[INT_LINK:.*]] = acc.declare_link varPtr(%[[GET_LOCAL_INT]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {name = "LocalInt2"}
    //
    // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
    // CHECK-NEXT: %[[LB:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
    // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
    // CHECK-NEXT: %[[UB:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
    // CHECK-NEXT: %[[ZERO:.*]] = arith.constant 0 : i64
    // CHECK-NEXT: %[[ONE:.*]] = arith.constant 1 : i64
    // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[LB]] : si32) extent(%[[UB]] : si32) stride(%[[ONE]] : i64) startIdx(%[[ZERO]] : i64)
    // CHECK-NEXT: %[[GET_LOCAL_ARR:.*]] = cir.get_global @LocalHSEArr2 : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>
    // CHECK-NEXT: %[[ARR_LINK:.*]] = acc.declare_link varPtr(%[[GET_LOCAL_ARR]] : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!rec_HasSideEffects x 5>> {name = "LocalHSEArr2[1:1]"}
    //
    // CHECK-NEXT: %[[ENTER:.*]] = acc.declare_enter dataOperands(%[[HSE_LINK]], %[[INT_LINK]], %[[ARR_LINK]] : !cir.ptr<!rec_HasSideEffects>, !cir.ptr<!s32i>, !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>)
    //
    // CHECK-NEXT: acc.declare_exit token(%[[ENTER]])
}

extern "C" void do_thing();

void NormalFunc() {
    // CHECK: cir.func {{.*}}NormalFunc{{.*}}()
    extern HasSideEffects LocalHSE3;
    extern HasSideEffects LocalHSEArr3[5];
    extern int LocalInt3;
    // CHECK-NEXT: cir.scope
    {
    extern HasSideEffects InnerHSE;
#pragma acc declare link(LocalHSE3, LocalInt3, LocalHSEArr3[1:1], InnerHSE)
    // CHECK-NEXT: %[[GET_LOCAL_HSE:.*]] = cir.get_global @LocalHSE3 : !cir.ptr<!rec_HasSideEffects>
    // CHECK-NEXT: %[[HSE_LINK:.*]] = acc.declare_link varPtr(%[[GET_LOCAL_HSE]] : !cir.ptr<!rec_HasSideEffects>) -> !cir.ptr<!rec_HasSideEffects> {name = "LocalHSE3"}
    //
    // CHECK-NEXT: %[[GET_LOCAL_INT:.*]] = cir.get_global @LocalInt3 : !cir.ptr<!s32i>
    // CHECK-NEXT: %[[INT_LINK:.*]] = acc.declare_link varPtr(%[[GET_LOCAL_INT]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {name = "LocalInt3"}
    //
    // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
    // CHECK-NEXT: %[[LB:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
    // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
    // CHECK-NEXT: %[[UB:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
    // CHECK-NEXT: %[[ZERO:.*]] = arith.constant 0 : i64
    // CHECK-NEXT: %[[ONE:.*]] = arith.constant 1 : i64
    // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[LB]] : si32) extent(%[[UB]] : si32) stride(%[[ONE]] : i64) startIdx(%[[ZERO]] : i64)
    // CHECK-NEXT: %[[GET_LOCAL_ARR:.*]] = cir.get_global @LocalHSEArr3 : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>
    // CHECK-NEXT: %[[ARR_LINK:.*]] = acc.declare_link varPtr(%[[GET_LOCAL_ARR]] : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!rec_HasSideEffects x 5>> {name = "LocalHSEArr3[1:1]"}
    //
    // CHECK-NEXT: %[[GET_LOCAL_HSE:.*]] = cir.get_global @InnerHSE : !cir.ptr<!rec_HasSideEffects>
    // CHECK-NEXT: %[[INNERHSE_LINK:.*]] = acc.declare_link varPtr(%[[GET_LOCAL_HSE]] : !cir.ptr<!rec_HasSideEffects>) -> !cir.ptr<!rec_HasSideEffects> {name = "InnerHSE"}
    //
    // CHECK-NEXT: %[[ENTER:.*]] = acc.declare_enter dataOperands(%[[HSE_LINK]], %[[INT_LINK]], %[[ARR_LINK]], %[[INNERHSE_LINK]] : !cir.ptr<!rec_HasSideEffects>, !cir.ptr<!s32i>, !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>, !cir.ptr<!rec_HasSideEffects>)
    //
    // CHECK

    do_thing();
    // CHECK-NEXT: cir.call @do_thing

    // CHECK-NEXT: acc.declare_exit token(%[[ENTER]])
    }
    // CHECK-NEXT: }

    do_thing();
    // CHECK-NEXT: cir.call @do_thing
}

