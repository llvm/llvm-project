// RUN: %clang_cc1 -fopenacc -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir %s -o - | FileCheck %s

struct HasSideEffects {
  HasSideEffects();
  ~HasSideEffects();
};

struct Struct {
  static const HasSideEffects StaticMemHSE;
  static const HasSideEffects StaticMemHSEArr[5];
  static const int StaticMemInt;

  void MemFunc1(HasSideEffects *ArgHSE, int *ArgInt) {
    // CHECK: cir.func {{.*}}MemFunc1{{.*}}(%{{.*}}: !cir.ptr<!rec_Struct>{{.*}}, %[[ARG_HSE:.*]]: !cir.ptr<!rec_HasSideEffects>{{.*}}, %[[ARG_INT:.*]]: !cir.ptr<!s32i> {{.*}})
    // CHECK-NEXT: cir.alloca{{.*}}["this"
    // CHECK-NEXT: %[[ARG_HSE_ALLOCA:.*]] = cir.alloca !cir.ptr<!rec_HasSideEffects>{{.*}}["ArgHSE
    // CHECK-NEXT: %[[ARG_INT_ALLOCA:.*]] = cir.alloca !cir.ptr<!s32i>{{.*}}["ArgInt
    // CHECK-NEXT: %[[LOC_HSE_ALLOCA:.*]] = cir.alloca !cir.ptr<!rec_HasSideEffects>{{.*}}["LocalHSE
    // CHECK-NEXT: %[[LOC_INT_ALLOCA:.*]] = cir.alloca !cir.ptr<!s32i>{{.*}}["LocalInt
    // CHECK-NEXT: cir.store
    // CHECK-NEXT: cir.store
    // CHECK-NEXT: cir.store
    // CHECK-NEXT: cir.load
    HasSideEffects *LocalHSE;
    int *LocalInt;
#pragma acc declare deviceptr(ArgHSE, ArgInt, LocalHSE, LocalInt)
    // CHECK-NEXT: %[[DEV_PTR_ARG_HSE:.*]] = acc.deviceptr varPtr(%[[ARG_HSE_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_HasSideEffects>>) -> !cir.ptr<!cir.ptr<!rec_HasSideEffects>> {name = "ArgHSE"}
    // CHECK-NEXT: %[[DEV_PTR_ARG_INT:.*]] = acc.deviceptr varPtr(%[[ARG_INT_ALLOCA]] : !cir.ptr<!cir.ptr<!s32i>>) -> !cir.ptr<!cir.ptr<!s32i>> {name = "ArgInt"}
    // CHECK-NEXT: %[[DEV_PTR_LOC_HSE:.*]] = acc.deviceptr varPtr(%[[LOC_HSE_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_HasSideEffects>>) -> !cir.ptr<!cir.ptr<!rec_HasSideEffects>> {name = "LocalHSE"}
    // CHECK-NEXT: %[[DEV_PTR_LOC_INT:.*]] = acc.deviceptr varPtr(%[[LOC_INT_ALLOCA]] : !cir.ptr<!cir.ptr<!s32i>>) -> !cir.ptr<!cir.ptr<!s32i>> {name = "LocalInt"}
    // CHECK-NEXT: %[[ENTER:.*]] = acc.declare_enter dataOperands(%[[DEV_PTR_ARG_HSE]], %[[DEV_PTR_ARG_INT]], %[[DEV_PTR_LOC_HSE]], %[[DEV_PTR_LOC_INT]] : !cir.ptr<!cir.ptr<!rec_HasSideEffects>>, !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!cir.ptr<!rec_HasSideEffects>>, !cir.ptr<!cir.ptr<!s32i>>)

    // CHECK-NEXT: acc.declare_exit token(%[[ENTER]])
  }
  void MemFunc2(HasSideEffects *ArgHSE, int *ArgInt);
};

void use() {
  Struct s;
  s.MemFunc1(nullptr, nullptr);
}

void Struct::MemFunc2(HasSideEffects *ArgHSE, int *ArgInt) {
    // CHECK: cir.func {{.*}}MemFunc2{{.*}}(%{{.*}}: !cir.ptr<!rec_Struct>{{.*}}, %[[ARG_HSE:.*]]: !cir.ptr<!rec_HasSideEffects>{{.*}}, %[[ARG_INT:.*]]: !cir.ptr<!s32i> {{.*}})
    // CHECK-NEXT: cir.alloca{{.*}}["this"
    // CHECK-NEXT: %[[ARG_HSE_ALLOCA:.*]] = cir.alloca !cir.ptr<!rec_HasSideEffects>{{.*}}["ArgHSE
    // CHECK-NEXT: %[[ARG_INT_ALLOCA:.*]] = cir.alloca !cir.ptr<!s32i>{{.*}}["ArgInt
    // CHECK-NEXT: %[[LOC_HSE_ALLOCA:.*]] = cir.alloca !cir.ptr<!rec_HasSideEffects>{{.*}}["LocalHSE
    // CHECK-NEXT: %[[LOC_INT_ALLOCA:.*]] = cir.alloca !cir.ptr<!s32i>{{.*}}["LocalInt
    // CHECK-NEXT: cir.store
    // CHECK-NEXT: cir.store
    // CHECK-NEXT: cir.store
    // CHECK-NEXT: cir.load
    HasSideEffects *LocalHSE;
    int *LocalInt;
#pragma acc declare deviceptr(ArgHSE, ArgInt)
    // CHECK-NEXT: %[[DEV_PTR_ARG_HSE:.*]] = acc.deviceptr varPtr(%[[ARG_HSE_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_HasSideEffects>>) -> !cir.ptr<!cir.ptr<!rec_HasSideEffects>> {name = "ArgHSE"}
    // CHECK-NEXT: %[[DEV_PTR_ARG_INT:.*]] = acc.deviceptr varPtr(%[[ARG_INT_ALLOCA]] : !cir.ptr<!cir.ptr<!s32i>>) -> !cir.ptr<!cir.ptr<!s32i>> {name = "ArgInt"}
    // CHECK-NEXT: %[[ENTER1:.*]] = acc.declare_enter dataOperands(%[[DEV_PTR_ARG_HSE]], %[[DEV_PTR_ARG_INT]] : !cir.ptr<!cir.ptr<!rec_HasSideEffects>>, !cir.ptr<!cir.ptr<!s32i>>)

#pragma acc declare deviceptr(LocalHSE, LocalInt)
    // CHECK-NEXT: %[[DEV_PTR_LOC_HSE:.*]] = acc.deviceptr varPtr(%[[LOC_HSE_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_HasSideEffects>>) -> !cir.ptr<!cir.ptr<!rec_HasSideEffects>> {name = "LocalHSE"}
    // CHECK-NEXT: %[[DEV_PTR_LOC_INT:.*]] = acc.deviceptr varPtr(%[[LOC_INT_ALLOCA]] : !cir.ptr<!cir.ptr<!s32i>>) -> !cir.ptr<!cir.ptr<!s32i>> {name = "LocalInt"}
    // CHECK-NEXT: %[[ENTER2:.*]] = acc.declare_enter dataOperands(%[[DEV_PTR_LOC_HSE]], %[[DEV_PTR_LOC_INT]] : !cir.ptr<!cir.ptr<!rec_HasSideEffects>>, !cir.ptr<!cir.ptr<!s32i>>)
    //
    // CHECK-NEXT: acc.declare_exit token(%[[ENTER2]])
    // CHECK-NEXT: acc.declare_exit token(%[[ENTER1]])
}

extern "C" void do_thing();

void NormalFunc(HasSideEffects *ArgHSE, int *ArgInt) {
    // CHECK: cir.func {{.*}}NormalFunc{{.*}}(%[[ARG_HSE:.*]]: !cir.ptr<!rec_HasSideEffects>{{.*}}, %[[ARG_INT:.*]]: !cir.ptr<!s32i> {{.*}})
    // CHECK-NEXT: %[[ARG_HSE_ALLOCA:.*]] = cir.alloca !cir.ptr<!rec_HasSideEffects>{{.*}}["ArgHSE
    // CHECK-NEXT: %[[ARG_INT_ALLOCA:.*]] = cir.alloca !cir.ptr<!s32i>{{.*}}["ArgInt
    // CHECK-NEXT: %[[LOC_HSE_ALLOCA:.*]] = cir.alloca !cir.ptr<!rec_HasSideEffects>{{.*}}["LocalHSE
    // CHECK-NEXT: %[[LOC_INT_ALLOCA:.*]] = cir.alloca !cir.ptr<!s32i>{{.*}}["LocalInt
    // CHECK-NEXT: cir.store
    // CHECK-NEXT: cir.store
    HasSideEffects *LocalHSE;
    int *LocalInt;
#pragma acc declare deviceptr(ArgHSE, ArgInt)
    // CHECK-NEXT: %[[DEV_PTR_ARG_HSE:.*]] = acc.deviceptr varPtr(%[[ARG_HSE_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_HasSideEffects>>) -> !cir.ptr<!cir.ptr<!rec_HasSideEffects>> {name = "ArgHSE"}
    // CHECK-NEXT: %[[DEV_PTR_ARG_INT:.*]] = acc.deviceptr varPtr(%[[ARG_INT_ALLOCA]] : !cir.ptr<!cir.ptr<!s32i>>) -> !cir.ptr<!cir.ptr<!s32i>> {name = "ArgInt"}
    // CHECK-NEXT: %[[ENTER1:.*]] = acc.declare_enter dataOperands(%[[DEV_PTR_ARG_HSE]], %[[DEV_PTR_ARG_INT]] : !cir.ptr<!cir.ptr<!rec_HasSideEffects>>, !cir.ptr<!cir.ptr<!s32i>>)
    {
      // CHECK-NEXT: cir.scope {
#pragma acc declare deviceptr(LocalHSE, LocalInt)
    // CHECK-NEXT: %[[DEV_PTR_LOC_HSE:.*]] = acc.deviceptr varPtr(%[[LOC_HSE_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_HasSideEffects>>) -> !cir.ptr<!cir.ptr<!rec_HasSideEffects>> {name = "LocalHSE"}
    // CHECK-NEXT: %[[DEV_PTR_LOC_INT:.*]] = acc.deviceptr varPtr(%[[LOC_INT_ALLOCA]] : !cir.ptr<!cir.ptr<!s32i>>) -> !cir.ptr<!cir.ptr<!s32i>> {name = "LocalInt"}
    // CHECK-NEXT: %[[ENTER2:.*]] = acc.declare_enter dataOperands(%[[DEV_PTR_LOC_HSE]], %[[DEV_PTR_LOC_INT]] : !cir.ptr<!cir.ptr<!rec_HasSideEffects>>, !cir.ptr<!cir.ptr<!s32i>>)
    do_thing();
    // CHECK-NEXT: cir.call @do_thing
    // CHECK-NEXT: acc.declare_exit token(%[[ENTER2]])

    }
    // CHECK-NEXT: }

    // Make sure that cleanup gets put in the right scope.
    do_thing();
    // CHECK-NEXT: cir.call @do_thing
    // CHECK-NEXT: acc.declare_exit token(%[[ENTER1]])
}

