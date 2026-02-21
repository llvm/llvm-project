// RUN: %clang_cc1 -fopenacc -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir %s -o - | FileCheck %s

struct HasSideEffects {
  HasSideEffects();
  ~HasSideEffects();
};

struct Struct {
  static const HasSideEffects StaticMemHSE;
  static const HasSideEffects StaticMemHSEArr[5];
  static const int StaticMemInt;

  void MemFunc1(HasSideEffects ArgHSE, int ArgInt, HasSideEffects *ArgHSEPtr) {
    // CHECK: cir.func {{.*}}MemFunc1{{.*}}(%{{.*}}: !cir.ptr<!rec_Struct>{{.*}}, %[[ARG_HSE:.*]]: !rec_HasSideEffects{{.*}}, %[[ARG_INT:.*]]: !s32i {{.*}}, %[[ARG_HSE_PTR:.*]]: !cir.ptr<!rec_HasSideEffects>{{.*}})
    // CHECK-NEXT: cir.alloca{{.*}}["this"
    // CHECK-NEXT: %[[ARG_HSE_ALLOCA:.*]] = cir.alloca !rec_HasSideEffects{{.*}}["ArgHSE"
    // CHECK-NEXT: %[[ARG_INT_ALLOCA:.*]] = cir.alloca !s32i{{.*}}["ArgInt
    // CHECK-NEXT: %[[ARG_HSE_PTR_ALLOCA:.*]] = cir.alloca !cir.ptr<!rec_HasSideEffects>{{.*}}["ArgHSEPtr"
    // CHECK-NEXT: %[[LOC_HSE_ALLOCA:.*]] = cir.alloca !rec_HasSideEffects{{.*}}["LocalHSE
    // CHECK-NEXT: %[[LOC_HSE_ARR_ALLOCA:.*]] = cir.alloca !cir.array<!rec_HasSideEffects x 5>{{.*}}["LocalHSEArr
    // CHECK-NEXT: %[[LOC_INT_ALLOCA:.*]] = cir.alloca !s32i{{.*}}["LocalInt
    // CHECK-NEXT: cir.store
    // CHECK-NEXT: cir.store
    // CHECK-NEXT: cir.store
    // CHECK-NEXT: cir.store
    // CHECK-NEXT: cir.load

    HasSideEffects LocalHSE;
    // CHECK-NEXT: cir.call{{.*}} : (!cir.ptr<!rec_HasSideEffects>) -> ()
    HasSideEffects LocalHSEArr[5];
    int LocalInt;

#pragma acc declare copy(always:ArgHSE, ArgInt, LocalHSE, LocalInt, ArgHSEPtr[1:1], LocalHSEArr[1:1])
    // CHECK: %[[ARG_HSE_COPYIN:.*]] = acc.copyin varPtr(%[[ARG_HSE_ALLOCA]] : !cir.ptr<!rec_HasSideEffects>) -> !cir.ptr<!rec_HasSideEffects> {dataClause = #acc<data_clause acc_copy>, modifiers = #acc<data_clause_modifier always>, name = "ArgHSE"}
    // CHECK-NEXT: %[[ARG_INT_COPYIN:.*]] = acc.copyin varPtr(%[[ARG_INT_ALLOCA]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copy>, modifiers = #acc<data_clause_modifier always>, name = "ArgInt"}
    // CHECK-NEXT: %[[LOC_HSE_COPYIN:.*]] = acc.copyin varPtr(%[[LOC_HSE_ALLOCA]] : !cir.ptr<!rec_HasSideEffects>) -> !cir.ptr<!rec_HasSideEffects> {dataClause = #acc<data_clause acc_copy>, modifiers = #acc<data_clause_modifier always>, name = "LocalHSE"}
    // CHECK-NEXT: %[[LOC_INT_COPYIN:.*]] = acc.copyin varPtr(%[[LOC_INT_ALLOCA]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copy>, modifiers = #acc<data_clause_modifier always>, name = "LocalInt"}
    // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
    // CHECK-NEXT: %[[LB:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
    // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
    // CHECK-NEXT: %[[UB:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
    // CHECK-NEXT: %[[IDX:.*]] = arith.constant 0 : i64
    // CHECK-NEXT: %[[STRIDE:.*]] = arith.constant 1 : i64
    // CHECK-NEXT: %[[BOUND1:.*]] = acc.bounds lowerbound(%[[LB]] : si32) extent(%[[UB]] : si32) stride(%[[STRIDE]] : i64) startIdx(%[[IDX]] : i64)
    // CHECK-NEXT: %[[ARG_HSE_PTR_COPYIN:.*]] = acc.copyin varPtr(%[[ARG_HSE_PTR_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_HasSideEffects>>) bounds(%[[BOUND1]]) -> !cir.ptr<!cir.ptr<!rec_HasSideEffects>> {dataClause = #acc<data_clause acc_copy>, modifiers = #acc<data_clause_modifier always>, name = "ArgHSEPtr[1:1]"}
    // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
    // CHECK-NEXT: %[[LB:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
    // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
    // CHECK-NEXT: %[[UB:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
    // CHECK-NEXT: %[[IDX:.*]] = arith.constant 0 : i64
    // CHECK-NEXT: %[[STRIDE:.*]] = arith.constant 1 : i64
    // CHECK-NEXT: %[[BOUND2:.*]] = acc.bounds lowerbound(%[[LB]] : si32) extent(%[[UB]] : si32) stride(%[[STRIDE]] : i64) startIdx(%[[IDX]] : i64)
    // CHECK-NEXT: %[[LOC_HSE_ARR_COPYIN:.*]] = acc.copyin varPtr(%[[LOC_HSE_ARR_ALLOCA]] : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>) bounds(%[[BOUND2]]) -> !cir.ptr<!cir.array<!rec_HasSideEffects x 5>> {dataClause = #acc<data_clause acc_copy>, modifiers = #acc<data_clause_modifier always>, name = "LocalHSEArr[1:1]"}
    // CHECK-NEXT: %[[ENTER:.*]] = acc.declare_enter dataOperands(%[[ARG_HSE_COPYIN]], %[[ARG_INT_COPYIN]], %[[LOC_HSE_COPYIN]], %[[LOC_INT_COPYIN]], %[[ARG_HSE_PTR_COPYIN]], %[[LOC_HSE_ARR_COPYIN]] : !cir.ptr<!rec_HasSideEffects>, !cir.ptr<!s32i>, !cir.ptr<!rec_HasSideEffects>, !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!rec_HasSideEffects>>, !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>)
    //
    // CHECK-NEXT: acc.declare_exit token(%[[ENTER]]) dataOperands(%[[ARG_HSE_COPYIN]], %[[ARG_INT_COPYIN]], %[[LOC_HSE_COPYIN]], %[[LOC_INT_COPYIN]], %[[ARG_HSE_PTR_COPYIN]], %[[LOC_HSE_ARR_COPYIN]] : !cir.ptr<!rec_HasSideEffects>, !cir.ptr<!s32i>, !cir.ptr<!rec_HasSideEffects>, !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!rec_HasSideEffects>>, !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>)
    // CHECK-NEXT: acc.copyout accPtr(%[[ARG_HSE_COPYIN]] : !cir.ptr<!rec_HasSideEffects>) to varPtr(%[[ARG_HSE_ALLOCA]] : !cir.ptr<!rec_HasSideEffects>) {dataClause = #acc<data_clause acc_copy>, modifiers = #acc<data_clause_modifier always>, name = "ArgHSE"}
    // CHECK-NEXT: acc.copyout accPtr(%[[ARG_INT_COPYIN]] : !cir.ptr<!s32i>) to varPtr(%[[ARG_INT_ALLOCA]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copy>, modifiers = #acc<data_clause_modifier always>, name = "ArgInt"}
    // CHECK-NEXT: acc.copyout accPtr(%[[LOC_HSE_COPYIN]] : !cir.ptr<!rec_HasSideEffects>) to varPtr(%[[LOC_HSE_ALLOCA]] : !cir.ptr<!rec_HasSideEffects>) {dataClause = #acc<data_clause acc_copy>, modifiers = #acc<data_clause_modifier always>, name = "LocalHSE"}
    // CHECK-NEXT: acc.copyout accPtr(%[[LOC_INT_COPYIN]] : !cir.ptr<!s32i>) to varPtr(%[[LOC_INT_ALLOCA]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copy>, modifiers = #acc<data_clause_modifier always>, name = "LocalInt"}
    // CHECK-NEXT: acc.copyout accPtr(%[[ARG_HSE_PTR_COPYIN]] : !cir.ptr<!cir.ptr<!rec_HasSideEffects>>) bounds(%[[BOUND1]]) to varPtr(%[[ARG_HSE_PTR_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_HasSideEffects>>) {dataClause = #acc<data_clause acc_copy>, modifiers = #acc<data_clause_modifier always>, name = "ArgHSEPtr[1:1]"}
    // CHECK-NEXT: acc.copyout accPtr(%[[LOC_HSE_ARR_COPYIN]] : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>) bounds(%[[BOUND2]]) to varPtr(%[[LOC_HSE_ARR_ALLOCA]] : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>) {dataClause = #acc<data_clause acc_copy>, modifiers = #acc<data_clause_modifier always>, name = "LocalHSEArr[1:1]"}
  }
  void MemFunc2(HasSideEffects ArgHSE, int ArgInt, HasSideEffects *ArgHSEPtr);
};

void use() {
  Struct s;
  s.MemFunc1(HasSideEffects{}, 0, nullptr);
}

void Struct::MemFunc2(HasSideEffects ArgHSE, int ArgInt, HasSideEffects *ArgHSEPtr) {
    // CHECK: cir.func {{.*}}MemFunc2{{.*}}(%{{.*}}: !cir.ptr<!rec_Struct>{{.*}}, %[[ARG_HSE:.*]]: !rec_HasSideEffects{{.*}}, %[[ARG_INT:.*]]: !s32i {{.*}}, %[[ARG_HSE_PTR:.*]]: !cir.ptr<!rec_HasSideEffects>{{.*}})
    // CHECK-NEXT: cir.alloca{{.*}}["this"
    // CHECK-NEXT: %[[ARG_HSE_ALLOCA:.*]] = cir.alloca !rec_HasSideEffects{{.*}}["ArgHSE"
    // CHECK-NEXT: %[[ARG_INT_ALLOCA:.*]] = cir.alloca !s32i{{.*}}["ArgInt
    // CHECK-NEXT: %[[ARG_HSE_PTR_ALLOCA:.*]] = cir.alloca !cir.ptr<!rec_HasSideEffects>{{.*}}["ArgHSEPtr"
    // CHECK-NEXT: %[[LOC_HSE_ALLOCA:.*]] = cir.alloca !rec_HasSideEffects{{.*}}["LocalHSE
    // CHECK-NEXT: %[[LOC_HSE_ARR_ALLOCA:.*]] = cir.alloca !cir.array<!rec_HasSideEffects x 5>{{.*}}["LocalHSEArr
    // CHECK-NEXT: %[[LOC_INT_ALLOCA:.*]] = cir.alloca !s32i{{.*}}["LocalInt
    // CHECK-NEXT: cir.store
    // CHECK-NEXT: cir.store
    // CHECK-NEXT: cir.store
    // CHECK-NEXT: cir.store
    // CHECK-NEXT: cir.load
    HasSideEffects LocalHSE;
    // CHECK-NEXT: cir.call{{.*}} : (!cir.ptr<!rec_HasSideEffects>) -> ()
    HasSideEffects LocalHSEArr[5];
    // CHECK: do {
    // CHECK: } while {
    // CHECK: }
    int LocalInt;
#pragma acc declare copy(alwaysin:ArgHSE, ArgInt, ArgHSEPtr[1:1])
    // CHECK: %[[ARG_HSE_COPYIN:.*]] = acc.copyin varPtr(%[[ARG_HSE_ALLOCA]] : !cir.ptr<!rec_HasSideEffects>) -> !cir.ptr<!rec_HasSideEffects> {dataClause = #acc<data_clause acc_copy>, modifiers = #acc<data_clause_modifier alwaysin>, name = "ArgHSE"}
    // CHECK-NEXT: %[[ARG_INT_COPYIN:.*]] = acc.copyin varPtr(%[[ARG_INT_ALLOCA]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copy>, modifiers = #acc<data_clause_modifier alwaysin>, name = "ArgInt"}
    // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
    // CHECK-NEXT: %[[LB:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
    // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
    // CHECK-NEXT: %[[UB:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
    // CHECK-NEXT: %[[IDX:.*]] = arith.constant 0 : i64
    // CHECK-NEXT: %[[STRIDE:.*]] = arith.constant 1 : i64
    // CHECK-NEXT: %[[BOUND1:.*]] = acc.bounds lowerbound(%[[LB]] : si32) extent(%[[UB]] : si32) stride(%[[STRIDE]] : i64) startIdx(%[[IDX]] : i64)
    // CHECK-NEXT: %[[ARG_HSE_PTR_COPYIN:.*]] = acc.copyin varPtr(%[[ARG_HSE_PTR_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_HasSideEffects>>) bounds(%[[BOUND1]]) -> !cir.ptr<!cir.ptr<!rec_HasSideEffects>> {dataClause = #acc<data_clause acc_copy>, modifiers = #acc<data_clause_modifier alwaysin>, name = "ArgHSEPtr[1:1]"}
    // CHECK-NEXT: %[[ENTER1:.*]] = acc.declare_enter dataOperands(%[[ARG_HSE_COPYIN]], %[[ARG_INT_COPYIN]], %[[ARG_HSE_PTR_COPYIN]] : !cir.ptr<!rec_HasSideEffects>, !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!rec_HasSideEffects>>)

#pragma acc declare copy(alwaysout:LocalHSE, LocalInt, LocalHSEArr[1:1])
    // CHECK-NEXT: %[[LOC_HSE_COPYIN:.*]] = acc.copyin varPtr(%[[LOC_HSE_ALLOCA]] : !cir.ptr<!rec_HasSideEffects>) -> !cir.ptr<!rec_HasSideEffects> {dataClause = #acc<data_clause acc_copy>, modifiers = #acc<data_clause_modifier alwaysout>, name = "LocalHSE"}
    // CHECK-NEXT: %[[LOC_INT_COPYIN:.*]] = acc.copyin varPtr(%[[LOC_INT_ALLOCA]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copy>, modifiers = #acc<data_clause_modifier alwaysout>, name = "LocalInt"}
    // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
    // CHECK-NEXT: %[[LB:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
    // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
    // CHECK-NEXT: %[[UB:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
    // CHECK-NEXT: %[[IDX:.*]] = arith.constant 0 : i64
    // CHECK-NEXT: %[[STRIDE:.*]] = arith.constant 1 : i64
    // CHECK-NEXT: %[[BOUND2:.*]] = acc.bounds lowerbound(%[[LB]] : si32) extent(%[[UB]] : si32) stride(%[[STRIDE]] : i64) startIdx(%[[IDX]] : i64)
    // CHECK-NEXT: %[[LOC_HSE_ARR_COPYIN:.*]] = acc.copyin varPtr(%[[LOC_HSE_ARR_ALLOCA]] : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>) bounds(%[[BOUND2]]) -> !cir.ptr<!cir.array<!rec_HasSideEffects x 5>> {dataClause = #acc<data_clause acc_copy>, modifiers = #acc<data_clause_modifier alwaysout>, name = "LocalHSEArr[1:1]"}
    // CHECK-NEXT: %[[ENTER2:.*]] = acc.declare_enter dataOperands(%[[LOC_HSE_COPYIN]], %[[LOC_INT_COPYIN]], %[[LOC_HSE_ARR_COPYIN]] : !cir.ptr<!rec_HasSideEffects>, !cir.ptr<!s32i>, !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>)

    // CHECK-NEXT: acc.declare_exit token(%[[ENTER2]]) dataOperands(%[[LOC_HSE_COPYIN]], %[[LOC_INT_COPYIN]], %[[LOC_HSE_ARR_COPYIN]] : !cir.ptr<!rec_HasSideEffects>, !cir.ptr<!s32i>, !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>)
    // CHECK-NEXT: acc.copyout accPtr(%[[LOC_HSE_COPYIN]] : !cir.ptr<!rec_HasSideEffects>) to varPtr(%[[LOC_HSE_ALLOCA]] : !cir.ptr<!rec_HasSideEffects>) {dataClause = #acc<data_clause acc_copy>, modifiers = #acc<data_clause_modifier alwaysout>, name = "LocalHSE"}
    // CHECK-NEXT: acc.copyout accPtr(%[[LOC_INT_COPYIN]] : !cir.ptr<!s32i>) to varPtr(%[[LOC_INT_ALLOCA]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copy>, modifiers = #acc<data_clause_modifier alwaysout>, name = "LocalInt"}
    // CHECK-NEXT: acc.copyout accPtr(%[[LOC_HSE_ARR_COPYIN]] : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>) bounds(%[[BOUND2]]) to varPtr(%[[LOC_HSE_ARR_ALLOCA]] : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>) {dataClause = #acc<data_clause acc_copy>, modifiers = #acc<data_clause_modifier alwaysout>, name = "LocalHSEArr[1:1]"}
    //
    // CHECK-NEXT: acc.declare_exit token(%[[ENTER1]]) dataOperands(%[[ARG_HSE_COPYIN]], %[[ARG_INT_COPYIN]], %[[ARG_HSE_PTR_COPYIN]] : !cir.ptr<!rec_HasSideEffects>, !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!rec_HasSideEffects>>)
    // CHECK-NEXT: acc.copyout accPtr(%[[ARG_HSE_COPYIN]] : !cir.ptr<!rec_HasSideEffects>) to varPtr(%[[ARG_HSE_ALLOCA]] : !cir.ptr<!rec_HasSideEffects>) {dataClause = #acc<data_clause acc_copy>, modifiers = #acc<data_clause_modifier alwaysin>, name = "ArgHSE"}
    // CHECK-NEXT: acc.copyout accPtr(%[[ARG_INT_COPYIN]] : !cir.ptr<!s32i>) to varPtr(%[[ARG_INT_ALLOCA]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copy>, modifiers = #acc<data_clause_modifier alwaysin>, name = "ArgInt"}
    // CHECK-NEXT: acc.copyout accPtr(%[[ARG_HSE_PTR_COPYIN]] : !cir.ptr<!cir.ptr<!rec_HasSideEffects>>) bounds(%[[BOUND1]]) to varPtr(%[[ARG_HSE_PTR_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_HasSideEffects>>) {dataClause = #acc<data_clause acc_copy>, modifiers = #acc<data_clause_modifier alwaysin>, name = "ArgHSEPtr[1:1]"}
}

extern "C" void do_thing();

extern "C" void NormalFunc(HasSideEffects ArgHSE, int ArgInt, HasSideEffects *ArgHSEPtr) {
    // CHECK: cir.func {{.*}}NormalFunc(%[[ARG_HSE:.*]]: !rec_HasSideEffects{{.*}}, %[[ARG_INT:.*]]: !s32i {{.*}}, %[[ARG_HSE_PTR:.*]]: !cir.ptr<!rec_HasSideEffects>{{.*}})
    // CHECK-NEXT: %[[ARG_HSE_ALLOCA:.*]] = cir.alloca !rec_HasSideEffects{{.*}}["ArgHSE"
    // CHECK-NEXT: %[[ARG_INT_ALLOCA:.*]] = cir.alloca !s32i{{.*}}["ArgInt
    // CHECK-NEXT: %[[ARG_HSE_PTR_ALLOCA:.*]] = cir.alloca !cir.ptr<!rec_HasSideEffects>{{.*}}["ArgHSEPtr"
    // CHECK-NEXT: %[[LOC_HSE_ALLOCA:.*]] = cir.alloca !rec_HasSideEffects{{.*}}["LocalHSE
    // CHECK-NEXT: %[[LOC_HSE_ARR_ALLOCA:.*]] = cir.alloca !cir.array<!rec_HasSideEffects x 5>{{.*}}["LocalHSEArr
    // CHECK-NEXT: %[[LOC_INT_ALLOCA:.*]] = cir.alloca !s32i{{.*}}["LocalInt
    // CHECK-NEXT: cir.store
    // CHECK-NEXT: cir.store
    // CHECK-NEXT: cir.store
    HasSideEffects LocalHSE;
    // CHECK-NEXT: cir.call{{.*}} : (!cir.ptr<!rec_HasSideEffects>) -> ()
    HasSideEffects LocalHSEArr[5];
    // CHECK: do {
    // CHECK: } while {
    // CHECK: }
    int LocalInt;
#pragma acc declare copy(capture:ArgHSE, ArgInt, ArgHSEPtr[1:1])
    // CHECK: %[[ARG_HSE_COPYIN:.*]] = acc.copyin varPtr(%[[ARG_HSE_ALLOCA]] : !cir.ptr<!rec_HasSideEffects>) -> !cir.ptr<!rec_HasSideEffects> {dataClause = #acc<data_clause acc_copy>, modifiers = #acc<data_clause_modifier capture>, name = "ArgHSE"}
    // CHECK-NEXT: %[[ARG_INT_COPYIN:.*]] = acc.copyin varPtr(%[[ARG_INT_ALLOCA]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copy>, modifiers = #acc<data_clause_modifier capture>, name = "ArgInt"}
    // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
    // CHECK-NEXT: %[[LB:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
    // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
    // CHECK-NEXT: %[[UB:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
    // CHECK-NEXT: %[[IDX:.*]] = arith.constant 0 : i64
    // CHECK-NEXT: %[[STRIDE:.*]] = arith.constant 1 : i64
    // CHECK-NEXT: %[[BOUND1:.*]] = acc.bounds lowerbound(%[[LB]] : si32) extent(%[[UB]] : si32) stride(%[[STRIDE]] : i64) startIdx(%[[IDX]] : i64)
    // CHECK-NEXT: %[[ARG_HSE_PTR_COPYIN:.*]] = acc.copyin varPtr(%[[ARG_HSE_PTR_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_HasSideEffects>>) bounds(%[[BOUND1]]) -> !cir.ptr<!cir.ptr<!rec_HasSideEffects>> {dataClause = #acc<data_clause acc_copy>, modifiers = #acc<data_clause_modifier capture>, name = "ArgHSEPtr[1:1]"}
    // CHECK-NEXT: %[[ENTER1:.*]] = acc.declare_enter dataOperands(%[[ARG_HSE_COPYIN]], %[[ARG_INT_COPYIN]], %[[ARG_HSE_PTR_COPYIN]] : !cir.ptr<!rec_HasSideEffects>, !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!rec_HasSideEffects>>)
    {
      // CHECK-NEXT: cir.scope {
#pragma acc declare copy(LocalHSE, LocalInt, LocalHSEArr[1:1])
    // CHECK-NEXT: %[[LOC_HSE_COPYIN:.*]] = acc.copyin varPtr(%[[LOC_HSE_ALLOCA]] : !cir.ptr<!rec_HasSideEffects>) -> !cir.ptr<!rec_HasSideEffects> {dataClause = #acc<data_clause acc_copy>, name = "LocalHSE"}
    // CHECK-NEXT: %[[LOC_INT_COPYIN:.*]] = acc.copyin varPtr(%[[LOC_INT_ALLOCA]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copy>, name = "LocalInt"}
    // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
    // CHECK-NEXT: %[[LB:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
    // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
    // CHECK-NEXT: %[[UB:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
    // CHECK-NEXT: %[[IDX:.*]] = arith.constant 0 : i64
    // CHECK-NEXT: %[[STRIDE:.*]] = arith.constant 1 : i64
    // CHECK-NEXT: %[[BOUND2:.*]] = acc.bounds lowerbound(%[[LB]] : si32) extent(%[[UB]] : si32) stride(%[[STRIDE]] : i64) startIdx(%[[IDX]] : i64)
    // CHECK-NEXT: %[[LOC_HSE_ARR_COPYIN:.*]] = acc.copyin varPtr(%[[LOC_HSE_ARR_ALLOCA]] : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>) bounds(%[[BOUND2]]) -> !cir.ptr<!cir.array<!rec_HasSideEffects x 5>> {dataClause = #acc<data_clause acc_copy>, name = "LocalHSEArr[1:1]"}
    // CHECK-NEXT: %[[ENTER2:.*]] = acc.declare_enter dataOperands(%[[LOC_HSE_COPYIN]], %[[LOC_INT_COPYIN]], %[[LOC_HSE_ARR_COPYIN]] : !cir.ptr<!rec_HasSideEffects>, !cir.ptr<!s32i>, !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>)

    do_thing();
    // CHECK-NEXT: cir.call @do_thing
    // CHECK-NEXT: acc.declare_exit token(%[[ENTER2]]) dataOperands(%[[LOC_HSE_COPYIN]], %[[LOC_INT_COPYIN]], %[[LOC_HSE_ARR_COPYIN]] : !cir.ptr<!rec_HasSideEffects>, !cir.ptr<!s32i>, !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>)
    // CHECK-NEXT: acc.copyout accPtr(%[[LOC_HSE_COPYIN]] : !cir.ptr<!rec_HasSideEffects>) to varPtr(%[[LOC_HSE_ALLOCA]] : !cir.ptr<!rec_HasSideEffects>) {dataClause = #acc<data_clause acc_copy>, name = "LocalHSE"}
    // CHECK-NEXT: acc.copyout accPtr(%[[LOC_INT_COPYIN]] : !cir.ptr<!s32i>) to varPtr(%[[LOC_INT_ALLOCA]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copy>, name = "LocalInt"}
    // CHECK-NEXT: acc.copyout accPtr(%[[LOC_HSE_ARR_COPYIN]] : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>) bounds(%[[BOUND2]]) to varPtr(%[[LOC_HSE_ARR_ALLOCA]] : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "LocalHSEArr[1:1]"}
    }
    // CHECK-NEXT: }

    // Make sure that cleanup gets put in the right scope.
    do_thing();
    // CHECK-NEXT: cir.call @do_thing
    // CHECK-NEXT: acc.declare_exit token(%[[ENTER1]]) dataOperands(%[[ARG_HSE_COPYIN]], %[[ARG_INT_COPYIN]], %[[ARG_HSE_PTR_COPYIN]] : !cir.ptr<!rec_HasSideEffects>, !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!rec_HasSideEffects>>)

    // CHECK-NEXT: acc.copyout accPtr(%[[ARG_HSE_COPYIN]] : !cir.ptr<!rec_HasSideEffects>) to varPtr(%[[ARG_HSE_ALLOCA]] : !cir.ptr<!rec_HasSideEffects>) {dataClause = #acc<data_clause acc_copy>, modifiers = #acc<data_clause_modifier capture>, name = "ArgHSE"}
    // CHECK-NEXT: acc.copyout accPtr(%[[ARG_INT_COPYIN]] : !cir.ptr<!s32i>) to varPtr(%[[ARG_INT_ALLOCA]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copy>, modifiers = #acc<data_clause_modifier capture>, name = "ArgInt"}
    // CHECK-NEXT: acc.copyout accPtr(%[[ARG_HSE_PTR_COPYIN]] : !cir.ptr<!cir.ptr<!rec_HasSideEffects>>) bounds(%[[BOUND1]]) to varPtr(%[[ARG_HSE_PTR_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_HasSideEffects>>) {dataClause = #acc<data_clause acc_copy>, modifiers = #acc<data_clause_modifier capture>, name = "ArgHSEPtr[1:1]"}
}

