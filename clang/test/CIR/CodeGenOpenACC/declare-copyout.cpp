// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenacc -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir %s -o - | FileCheck %s

struct HasSideEffects {
  HasSideEffects();
  ~HasSideEffects();
};

struct Struct {
  static const HasSideEffects StaticMemHSE;
  static const HasSideEffects StaticMemHSEArr[5];
  static const int StaticMemInt;

  void MemFunc1(HasSideEffects ArgHSE, int ArgInt, HasSideEffects *ArgHSEPtr) {
    // CHECK: cir.func {{.*}}MemFunc1{{.*}}(%{{.*}}: !cir.ptr<!rec_Struct>{{.*}}, %[[ARG_HSE:.*]]: !cir.ptr<!rec_HasSideEffects> {llvm.align = 1 : i64, llvm.byref = !rec_HasSideEffects{{.*}}, %[[ARG_INT:.*]]: !s32i {{.*}}, %[[ARG_HSE_PTR:.*]]: !cir.ptr<!rec_HasSideEffects>{{.*}})
    // CHECK-NEXT: cir.alloca{{.*}}"this"
    // CHECK-NEXT: %[[ARG_INT_ALLOCA:.*]] = cir.alloca "ArgInt" {{.*}} : !cir.ptr<!s32i>
    // CHECK-NEXT: %[[ARG_HSE_PTR_ALLOCA:.*]] = cir.alloca "ArgHSEPtr" {{.*}} : !cir.ptr<!cir.ptr<!rec_HasSideEffects>>
    // CHECK-NEXT: %[[LOC_HSE_ALLOCA:.*]] = cir.alloca "LocalHSE" {{.*}} : !cir.ptr<!rec_HasSideEffects>
    // CHECK-NEXT: %[[LOC_HSE_ARR_ALLOCA:.*]] = cir.alloca "LocalHSEArr" {{.*}} : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>
    // CHECK-NEXT: %[[LOC_INT_ALLOCA:.*]] = cir.alloca "LocalInt" {{.*}} : !cir.ptr<!s32i>
    // CHECK-NEXT: cir.store
    // CHECK-NEXT: cir.store
    // CHECK-NEXT: cir.store
    // CHECK-NEXT: cir.load

    HasSideEffects LocalHSE;
    // CHECK-NEXT: cir.call{{.*}} : (!cir.ptr<!rec_HasSideEffects>{{.*}}) -> ()
    HasSideEffects LocalHSEArr[5];
    int LocalInt;

#pragma acc declare copyout(always:ArgHSE, ArgInt, LocalHSE, LocalInt, ArgHSEPtr[1:1], LocalHSEArr[1:1])
    // CHECK: %[[ARG_HSE_CREATE:.*]] = acc.create varPtr(%[[ARG_HSE]] : !cir.ptr<!rec_HasSideEffects>) dataClause(acc_copyout) name("ArgHSE") <modifiers = [always]> -> !cir.ptr<!rec_HasSideEffects>
    // CHECK-NEXT: %[[ARG_INT_CREATE:.*]] = acc.create varPtr(%[[ARG_INT_ALLOCA]] : !cir.ptr<!s32i>) dataClause(acc_copyout) name("ArgInt") <modifiers = [always]> -> !cir.ptr<!s32i>
    // CHECK-NEXT: %[[LOC_HSE_CREATE:.*]] = acc.create varPtr(%[[LOC_HSE_ALLOCA]] : !cir.ptr<!rec_HasSideEffects>) dataClause(acc_copyout) name("LocalHSE") <modifiers = [always]> -> !cir.ptr<!rec_HasSideEffects>
    // CHECK-NEXT: %[[LOC_INT_CREATE:.*]] = acc.create varPtr(%[[LOC_INT_ALLOCA]] : !cir.ptr<!s32i>) dataClause(acc_copyout) name("LocalInt") <modifiers = [always]> -> !cir.ptr<!s32i>
    // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
    // CHECK-NEXT: %[[LB:.*]] = cir.builtin_int_cast %[[ONE]] : !s32i -> si32
    // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
    // CHECK-NEXT: %[[UB:.*]] = cir.builtin_int_cast %[[ONE]] : !s32i -> si32
    // CHECK-NEXT: %[[IDX:.*]] = arith.constant 0 : i64
    // CHECK-NEXT: %[[STRIDE:.*]] = arith.constant 1 : i64
    // CHECK-NEXT: %[[BOUND1:.*]] = acc.bounds lowerbound(%[[LB]] : si32) extent(%[[UB]] : si32) stride(%[[STRIDE]] : i64) startIdx(%[[IDX]] : i64)
    // CHECK-NEXT: %[[ARG_HSE_PTR_CREATE:.*]] = acc.create varPtr(%[[ARG_HSE_PTR_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_HasSideEffects>>) bounds(%[[BOUND1]]) dataClause(acc_copyout) name("ArgHSEPtr[1:1]") <modifiers = [always]> -> !cir.ptr<!cir.ptr<!rec_HasSideEffects>>
    // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
    // CHECK-NEXT: %[[LB:.*]] = cir.builtin_int_cast %[[ONE]] : !s32i -> si32
    // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
    // CHECK-NEXT: %[[UB:.*]] = cir.builtin_int_cast %[[ONE]] : !s32i -> si32
    // CHECK-NEXT: %[[IDX:.*]] = arith.constant 0 : i64
    // CHECK-NEXT: %[[STRIDE:.*]] = arith.constant 1 : i64
    // CHECK-NEXT: %[[BOUND2:.*]] = acc.bounds lowerbound(%[[LB]] : si32) extent(%[[UB]] : si32) stride(%[[STRIDE]] : i64) startIdx(%[[IDX]] : i64)
    // CHECK-NEXT: %[[LOC_HSE_ARR_CREATE:.*]] = acc.create varPtr(%[[LOC_HSE_ARR_ALLOCA]] : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>) bounds(%[[BOUND2]]) dataClause(acc_copyout) name("LocalHSEArr[1:1]") <modifiers = [always]> -> !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>
    // CHECK-NEXT: %[[ENTER:.*]] = acc.declare_enter dataOperands(%[[ARG_HSE_CREATE]], %[[ARG_INT_CREATE]], %[[LOC_HSE_CREATE]], %[[LOC_INT_CREATE]], %[[ARG_HSE_PTR_CREATE]], %[[LOC_HSE_ARR_CREATE]] : !cir.ptr<!rec_HasSideEffects>, !cir.ptr<!s32i>, !cir.ptr<!rec_HasSideEffects>, !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!rec_HasSideEffects>>, !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>)

    // CHECK-NEXT: cir.cleanup.scope {
    // CHECK-NEXT:   cir.yield
    // CHECK-NEXT: } cleanup normal {
    // CHECK-NEXT:   acc.declare_exit token(%[[ENTER]]) dataOperands(%[[ARG_HSE_CREATE]], %[[ARG_INT_CREATE]], %[[LOC_HSE_CREATE]], %[[LOC_INT_CREATE]], %[[ARG_HSE_PTR_CREATE]], %[[LOC_HSE_ARR_CREATE]] : !cir.ptr<!rec_HasSideEffects>, !cir.ptr<!s32i>, !cir.ptr<!rec_HasSideEffects>, !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!rec_HasSideEffects>>, !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>)
    // CHECK-NEXT: acc.copyout accPtr(%[[ARG_HSE_CREATE]] : !cir.ptr<!rec_HasSideEffects>) to varPtr(%[[ARG_HSE]] : !cir.ptr<!rec_HasSideEffects>) name("ArgHSE") <modifiers = [always]>
    // CHECK-NEXT: acc.copyout accPtr(%[[ARG_INT_CREATE]] : !cir.ptr<!s32i>) to varPtr(%[[ARG_INT_ALLOCA]] : !cir.ptr<!s32i>) name("ArgInt") <modifiers = [always]>
    // CHECK-NEXT: acc.copyout accPtr(%[[LOC_HSE_CREATE]] : !cir.ptr<!rec_HasSideEffects>) to varPtr(%[[LOC_HSE_ALLOCA]] : !cir.ptr<!rec_HasSideEffects>) name("LocalHSE") <modifiers = [always]>
    // CHECK-NEXT: acc.copyout accPtr(%[[LOC_INT_CREATE]] : !cir.ptr<!s32i>) to varPtr(%[[LOC_INT_ALLOCA]] : !cir.ptr<!s32i>) name("LocalInt") <modifiers = [always]>
    // CHECK-NEXT: acc.copyout accPtr(%[[ARG_HSE_PTR_CREATE]] : !cir.ptr<!cir.ptr<!rec_HasSideEffects>>) bounds(%[[BOUND1]]) to varPtr(%[[ARG_HSE_PTR_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_HasSideEffects>>) name("ArgHSEPtr[1:1]") <modifiers = [always]>
    // CHECK-NEXT: acc.copyout accPtr(%[[LOC_HSE_ARR_CREATE]] : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>) bounds(%[[BOUND2]]) to varPtr(%[[LOC_HSE_ARR_ALLOCA]] : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>) name("LocalHSEArr[1:1]") <modifiers = [always]>
    // CHECK-NEXT:   cir.yield
    // CHECK-NEXT: }
  }
  void MemFunc2(HasSideEffects ArgHSE, int ArgInt, HasSideEffects *ArgHSEPtr);
};

void use() {
  Struct s;
  s.MemFunc1(HasSideEffects{}, 0, nullptr);
}

void Struct::MemFunc2(HasSideEffects ArgHSE, int ArgInt, HasSideEffects *ArgHSEPtr) {
    // CHECK: cir.func {{.*}}MemFunc2{{.*}}(%{{.*}}: !cir.ptr<!rec_Struct>{{.*}}, %[[ARG_HSE:.*]]: !cir.ptr<!rec_HasSideEffects> {llvm.align = 1 : i64, llvm.byref = !rec_HasSideEffects{{.*}}, %[[ARG_INT:.*]]: !s32i {{.*}}, %[[ARG_HSE_PTR:.*]]: !cir.ptr<!rec_HasSideEffects>{{.*}})
    // CHECK-NEXT: cir.alloca{{.*}}"this"
    // CHECK-NEXT: %[[ARG_INT_ALLOCA:.*]] = cir.alloca "ArgInt" {{.*}} : !cir.ptr<!s32i>
    // CHECK-NEXT: %[[ARG_HSE_PTR_ALLOCA:.*]] = cir.alloca "ArgHSEPtr" {{.*}} : !cir.ptr<!cir.ptr<!rec_HasSideEffects>>
    // CHECK-NEXT: %[[LOC_HSE_ALLOCA:.*]] = cir.alloca "LocalHSE" {{.*}} : !cir.ptr<!rec_HasSideEffects>
    // CHECK-NEXT: %[[LOC_HSE_ARR_ALLOCA:.*]] = cir.alloca "LocalHSEArr" {{.*}} : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>
    // CHECK-NEXT: %[[LOC_INT_ALLOCA:.*]] = cir.alloca "LocalInt" {{.*}} : !cir.ptr<!s32i>
    // CHECK-NEXT: cir.store
    // CHECK-NEXT: cir.store
    // CHECK-NEXT: cir.store
    // CHECK-NEXT: cir.load
    HasSideEffects LocalHSE;
    // CHECK-NEXT: cir.call{{.*}} : (!cir.ptr<!rec_HasSideEffects>{{.*}}) -> ()
    HasSideEffects LocalHSEArr[5];
    // CHECK: do {
    // CHECK: } while {
    // CHECK: }
    int LocalInt;
#pragma acc declare copyout(alwaysout:ArgHSE, ArgInt, ArgHSEPtr[1:1])
    // CHECK: %[[ARG_HSE_CREATE:.*]] = acc.create varPtr(%[[ARG_HSE]] : !cir.ptr<!rec_HasSideEffects>) dataClause(acc_copyout) name("ArgHSE") <modifiers = [alwaysout]> -> !cir.ptr<!rec_HasSideEffects>
    // CHECK-NEXT: %[[ARG_INT_CREATE:.*]] = acc.create varPtr(%[[ARG_INT_ALLOCA]] : !cir.ptr<!s32i>) dataClause(acc_copyout) name("ArgInt") <modifiers = [alwaysout]> -> !cir.ptr<!s32i>
    // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
    // CHECK-NEXT: %[[LB:.*]] = cir.builtin_int_cast %[[ONE]] : !s32i -> si32
    // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
    // CHECK-NEXT: %[[UB:.*]] = cir.builtin_int_cast %[[ONE]] : !s32i -> si32
    // CHECK-NEXT: %[[IDX:.*]] = arith.constant 0 : i64
    // CHECK-NEXT: %[[STRIDE:.*]] = arith.constant 1 : i64
    // CHECK-NEXT: %[[BOUND1:.*]] = acc.bounds lowerbound(%[[LB]] : si32) extent(%[[UB]] : si32) stride(%[[STRIDE]] : i64) startIdx(%[[IDX]] : i64)
    // CHECK-NEXT: %[[ARG_HSE_PTR_CREATE:.*]] = acc.create varPtr(%[[ARG_HSE_PTR_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_HasSideEffects>>) bounds(%[[BOUND1]]) dataClause(acc_copyout) name("ArgHSEPtr[1:1]") <modifiers = [alwaysout]> -> !cir.ptr<!cir.ptr<!rec_HasSideEffects>>
    // CHECK-NEXT: %[[ENTER1:.*]] = acc.declare_enter dataOperands(%[[ARG_HSE_CREATE]], %[[ARG_INT_CREATE]], %[[ARG_HSE_PTR_CREATE]] : !cir.ptr<!rec_HasSideEffects>, !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!rec_HasSideEffects>>)

#pragma acc declare copyout(alwaysout:LocalHSE, LocalInt, LocalHSEArr[1:1])
    // CHECK-NEXT: cir.cleanup.scope {
    // CHECK-NEXT: %[[LOC_HSE_CREATE:.*]] = acc.create varPtr(%[[LOC_HSE_ALLOCA]] : !cir.ptr<!rec_HasSideEffects>) dataClause(acc_copyout) name("LocalHSE") <modifiers = [alwaysout]> -> !cir.ptr<!rec_HasSideEffects>
    // CHECK-NEXT: %[[LOC_INT_CREATE:.*]] = acc.create varPtr(%[[LOC_INT_ALLOCA]] : !cir.ptr<!s32i>) dataClause(acc_copyout) name("LocalInt") <modifiers = [alwaysout]> -> !cir.ptr<!s32i>
    // CHECK-NEXT:   %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
    // CHECK-NEXT:   %[[LB:.*]] = cir.builtin_int_cast %[[ONE]] : !s32i -> si32
    // CHECK-NEXT:   %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
    // CHECK-NEXT:   %[[UB:.*]] = cir.builtin_int_cast %[[ONE]] : !s32i -> si32
    // CHECK-NEXT:   %[[IDX:.*]] = arith.constant 0 : i64
    // CHECK-NEXT:   %[[STRIDE:.*]] = arith.constant 1 : i64
    // CHECK-NEXT:   %[[BOUND2:.*]] = acc.bounds lowerbound(%[[LB]] : si32) extent(%[[UB]] : si32) stride(%[[STRIDE]] : i64) startIdx(%[[IDX]] : i64)
    // CHECK-NEXT: %[[LOC_HSE_ARR_CREATE:.*]] = acc.create varPtr(%[[LOC_HSE_ARR_ALLOCA]] : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>) bounds(%[[BOUND2]]) dataClause(acc_copyout) name("LocalHSEArr[1:1]") <modifiers = [alwaysout]> -> !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>
    // CHECK-NEXT:   %[[ENTER2:.*]] = acc.declare_enter dataOperands(%[[LOC_HSE_CREATE]], %[[LOC_INT_CREATE]], %[[LOC_HSE_ARR_CREATE]] : !cir.ptr<!rec_HasSideEffects>, !cir.ptr<!s32i>, !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>)

    // CHECK-NEXT: cir.cleanup.scope {
    // CHECK-NEXT:   cir.yield
    // CHECK-NEXT: cleanup normal {
    // CHECK-NEXT:   acc.declare_exit token(%[[ENTER2]]) dataOperands(%[[LOC_HSE_CREATE]], %[[LOC_INT_CREATE]], %[[LOC_HSE_ARR_CREATE]] : !cir.ptr<!rec_HasSideEffects>, !cir.ptr<!s32i>, !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>)
    // CHECK-NEXT: acc.copyout accPtr(%[[LOC_HSE_CREATE]] : !cir.ptr<!rec_HasSideEffects>) to varPtr(%[[LOC_HSE_ALLOCA]] : !cir.ptr<!rec_HasSideEffects>) name("LocalHSE") <modifiers = [alwaysout]>
    // CHECK-NEXT: acc.copyout accPtr(%[[LOC_INT_CREATE]] : !cir.ptr<!s32i>) to varPtr(%[[LOC_INT_ALLOCA]] : !cir.ptr<!s32i>) name("LocalInt") <modifiers = [alwaysout]>
    // CHECK-NEXT: acc.copyout accPtr(%[[LOC_HSE_ARR_CREATE]] : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>) bounds(%[[BOUND2]]) to varPtr(%[[LOC_HSE_ARR_ALLOCA]] : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>) name("LocalHSEArr[1:1]") <modifiers = [alwaysout]>
    // CHECK-NEXT:   cir.yield
    // CHECK-NEXT: }
 
    // CHECK-NEXT: cir.yield

    // CHECK-NEXT: cleanup normal {
    // CHECK-NEXT:   acc.declare_exit token(%[[ENTER1]]) dataOperands(%[[ARG_HSE_CREATE]], %[[ARG_INT_CREATE]], %[[ARG_HSE_PTR_CREATE]] : !cir.ptr<!rec_HasSideEffects>, !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!rec_HasSideEffects>>)
    // CHECK-NEXT: acc.copyout accPtr(%[[ARG_HSE_CREATE]] : !cir.ptr<!rec_HasSideEffects>) to varPtr(%[[ARG_HSE]] : !cir.ptr<!rec_HasSideEffects>) name("ArgHSE") <modifiers = [alwaysout]>
    // CHECK-NEXT: acc.copyout accPtr(%[[ARG_INT_CREATE]] : !cir.ptr<!s32i>) to varPtr(%[[ARG_INT_ALLOCA]] : !cir.ptr<!s32i>) name("ArgInt") <modifiers = [alwaysout]>
    // CHECK-NEXT: acc.copyout accPtr(%[[ARG_HSE_PTR_CREATE]] : !cir.ptr<!cir.ptr<!rec_HasSideEffects>>) bounds(%[[BOUND1]]) to varPtr(%[[ARG_HSE_PTR_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_HasSideEffects>>) name("ArgHSEPtr[1:1]") <modifiers = [alwaysout]>
    // CHECK-NEXT:   cir.yield
    // CHECK-NEXT: }
}

extern "C" void do_thing();

extern "C" void NormalFunc(HasSideEffects ArgHSE, int ArgInt, HasSideEffects *ArgHSEPtr) {
    // CHECK: cir.func {{.*}}NormalFunc(%[[ARG_HSE:.*]]: !cir.ptr<!rec_HasSideEffects> {llvm.align = 1 : i64, llvm.byref = !rec_HasSideEffects{{.*}}, %[[ARG_INT:.*]]: !s32i {{.*}}, %[[ARG_HSE_PTR:.*]]: !cir.ptr<!rec_HasSideEffects>{{.*}})
    // CHECK-NEXT: %[[ARG_INT_ALLOCA:.*]] = cir.alloca "ArgInt" {{.*}} : !cir.ptr<!s32i>
    // CHECK-NEXT: %[[ARG_HSE_PTR_ALLOCA:.*]] = cir.alloca "ArgHSEPtr" {{.*}} : !cir.ptr<!cir.ptr<!rec_HasSideEffects>>
    // CHECK-NEXT: %[[LOC_HSE_ALLOCA:.*]] = cir.alloca "LocalHSE" {{.*}} : !cir.ptr<!rec_HasSideEffects>
    // CHECK-NEXT: %[[LOC_HSE_ARR_ALLOCA:.*]] = cir.alloca "LocalHSEArr" {{.*}} : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>
    // CHECK-NEXT: %[[LOC_INT_ALLOCA:.*]] = cir.alloca "LocalInt" {{.*}} : !cir.ptr<!s32i>
    // CHECK-NEXT: cir.store
    // CHECK-NEXT: cir.store
    HasSideEffects LocalHSE;
    // CHECK-NEXT: cir.call{{.*}} : (!cir.ptr<!rec_HasSideEffects>{{.*}}) -> ()
    HasSideEffects LocalHSEArr[5];
    // CHECK: do {
    // CHECK: } while {
    // CHECK: }
    int LocalInt;
#pragma acc declare copyout(always:ArgHSE, ArgInt, ArgHSEPtr[1:1])
    // CHECK: %[[ARG_HSE_CREATE:.*]] = acc.create varPtr(%[[ARG_HSE]] : !cir.ptr<!rec_HasSideEffects>) dataClause(acc_copyout) name("ArgHSE") <modifiers = [always]> -> !cir.ptr<!rec_HasSideEffects>
    // CHECK-NEXT: %[[ARG_INT_CREATE:.*]] = acc.create varPtr(%[[ARG_INT_ALLOCA]] : !cir.ptr<!s32i>) dataClause(acc_copyout) name("ArgInt") <modifiers = [always]> -> !cir.ptr<!s32i>
    // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
    // CHECK-NEXT: %[[LB:.*]] = cir.builtin_int_cast %[[ONE]] : !s32i -> si32
    // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
    // CHECK-NEXT: %[[UB:.*]] = cir.builtin_int_cast %[[ONE]] : !s32i -> si32
    // CHECK-NEXT: %[[IDX:.*]] = arith.constant 0 : i64
    // CHECK-NEXT: %[[STRIDE:.*]] = arith.constant 1 : i64
    // CHECK-NEXT: %[[BOUND1:.*]] = acc.bounds lowerbound(%[[LB]] : si32) extent(%[[UB]] : si32) stride(%[[STRIDE]] : i64) startIdx(%[[IDX]] : i64)
    // CHECK-NEXT: %[[ARG_HSE_PTR_CREATE:.*]] = acc.create varPtr(%[[ARG_HSE_PTR_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_HasSideEffects>>) bounds(%[[BOUND1]]) dataClause(acc_copyout) name("ArgHSEPtr[1:1]") <modifiers = [always]> -> !cir.ptr<!cir.ptr<!rec_HasSideEffects>>
    // CHECK-NEXT: %[[ENTER1:.*]] = acc.declare_enter dataOperands(%[[ARG_HSE_CREATE]], %[[ARG_INT_CREATE]], %[[ARG_HSE_PTR_CREATE]] : !cir.ptr<!rec_HasSideEffects>, !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!rec_HasSideEffects>>)
    {

    // CHECK-NEXT: cir.cleanup.scope {
    // CHECK-NEXT: cir.scope {
#pragma acc declare copyout(LocalHSE, LocalInt, LocalHSEArr[1:1])
    // CHECK-NEXT: %[[LOC_HSE_CREATE:.*]] = acc.create varPtr(%[[LOC_HSE_ALLOCA]] : !cir.ptr<!rec_HasSideEffects>) dataClause(acc_copyout) name("LocalHSE") -> !cir.ptr<!rec_HasSideEffects>
    // CHECK-NEXT: %[[LOC_INT_CREATE:.*]] = acc.create varPtr(%[[LOC_INT_ALLOCA]] : !cir.ptr<!s32i>) dataClause(acc_copyout) name("LocalInt") -> !cir.ptr<!s32i>
    // CHECK-NEXT:   %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
    // CHECK-NEXT:   %[[LB:.*]] = cir.builtin_int_cast %[[ONE]] : !s32i -> si32
    // CHECK-NEXT:   %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
    // CHECK-NEXT:   %[[UB:.*]] = cir.builtin_int_cast %[[ONE]] : !s32i -> si32
    // CHECK-NEXT:   %[[IDX:.*]] = arith.constant 0 : i64
    // CHECK-NEXT:   %[[STRIDE:.*]] = arith.constant 1 : i64
    // CHECK-NEXT:   %[[BOUND2:.*]] = acc.bounds lowerbound(%[[LB]] : si32) extent(%[[UB]] : si32) stride(%[[STRIDE]] : i64) startIdx(%[[IDX]] : i64)
    // CHECK-NEXT: %[[LOC_HSE_ARR_CREATE:.*]] = acc.create varPtr(%[[LOC_HSE_ARR_ALLOCA]] : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>) bounds(%[[BOUND2]]) dataClause(acc_copyout) name("LocalHSEArr[1:1]") -> !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>
    // CHECK-NEXT:   %[[ENTER2:.*]] = acc.declare_enter dataOperands(%[[LOC_HSE_CREATE]], %[[LOC_INT_CREATE]], %[[LOC_HSE_ARR_CREATE]] : !cir.ptr<!rec_HasSideEffects>, !cir.ptr<!s32i>, !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>)

    do_thing();
    // CHECK-NEXT: cir.cleanup.scope {
    // CHECK-NEXT:   cir.call @do_thing
    // CHECK-NEXT:   cir.yield
    // CHECK-NEXT: } cleanup normal {
    // CHECK-NEXT:   acc.declare_exit token(%[[ENTER2]]) dataOperands(%[[LOC_HSE_CREATE]], %[[LOC_INT_CREATE]], %[[LOC_HSE_ARR_CREATE]] : !cir.ptr<!rec_HasSideEffects>, !cir.ptr<!s32i>, !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>)
    // CHECK-NEXT: acc.copyout accPtr(%[[LOC_HSE_CREATE]] : !cir.ptr<!rec_HasSideEffects>) to varPtr(%[[LOC_HSE_ALLOCA]] : !cir.ptr<!rec_HasSideEffects>) name("LocalHSE")
    // CHECK-NEXT: acc.copyout accPtr(%[[LOC_INT_CREATE]] : !cir.ptr<!s32i>) to varPtr(%[[LOC_INT_ALLOCA]] : !cir.ptr<!s32i>) name("LocalInt")
    // CHECK-NEXT: acc.copyout accPtr(%[[LOC_HSE_ARR_CREATE]] : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>) bounds(%[[BOUND2]]) to varPtr(%[[LOC_HSE_ARR_ALLOCA]] : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>) name("LocalHSEArr[1:1]")
    // CHECK-NEXT:   cir.yield
    // CHECK-NEXT: }
    }
    // CHECK-NEXT: }

    // Make sure that cleanup gets put in the right scope.
    do_thing();
    // CHECK-NEXT: cir.call @do_thing
    // CHECK-NEXT: cir.yield

    // CHECK-NEXT: } cleanup normal {
    // CHECK-NEXT:   acc.declare_exit token(%[[ENTER1]]) dataOperands(%[[ARG_HSE_CREATE]], %[[ARG_INT_CREATE]], %[[ARG_HSE_PTR_CREATE]] : !cir.ptr<!rec_HasSideEffects>, !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!rec_HasSideEffects>>)

    // CHECK-NEXT: acc.copyout accPtr(%[[ARG_HSE_CREATE]] : !cir.ptr<!rec_HasSideEffects>) to varPtr(%[[ARG_HSE]] : !cir.ptr<!rec_HasSideEffects>) name("ArgHSE") <modifiers = [always]>
    // CHECK-NEXT: acc.copyout accPtr(%[[ARG_INT_CREATE]] : !cir.ptr<!s32i>) to varPtr(%[[ARG_INT_ALLOCA]] : !cir.ptr<!s32i>) name("ArgInt") <modifiers = [always]>
    // CHECK-NEXT: acc.copyout accPtr(%[[ARG_HSE_PTR_CREATE]] : !cir.ptr<!cir.ptr<!rec_HasSideEffects>>) bounds(%[[BOUND1]]) to varPtr(%[[ARG_HSE_PTR_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_HasSideEffects>>) name("ArgHSEPtr[1:1]") <modifiers = [always]>
}
