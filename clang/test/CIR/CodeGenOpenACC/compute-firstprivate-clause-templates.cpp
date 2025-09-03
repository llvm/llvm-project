// RUN: %clang_cc1 -fopenacc -triple x86_64-linux-gnu -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir -triple x86_64-linux-pc %s -o - | FileCheck %s

struct CopyConstruct {
  CopyConstruct() = default;
  CopyConstruct(const CopyConstruct&);
};

struct NonDefaultCtor {
  NonDefaultCtor();
};

struct HasDtor {
  ~HasDtor();
};

// CHECK: acc.firstprivate.recipe @firstprivatization__ZTSi : !cir.ptr<!s32i> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!s32i> {{.*}}):
// CHECK-NEXT: cir.alloca !s32i, !cir.ptr<!s32i>, ["openacc.firstprivate.init"]
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } copy {
// CHECK-NEXT: ^bb0(%[[ARG_FROM:.*]]: !cir.ptr<!s32i> {{.*}}, %[[ARG_TO:.*]]: !cir.ptr<!s32i> {{.*}}):
// CHECK-NEXT: %[[LOAD:.*]] = cir.load {{.*}} %[[ARG_FROM]] : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT: cir.store{{.*}} %[[LOAD]], %[[ARG_TO]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
//
// CHECK-NEXT: acc.firstprivate.recipe @firstprivatization__ZTS7HasDtor : !cir.ptr<!rec_HasDtor> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_HasDtor> {{.*}}):
// CHECK-NEXT: cir.alloca !rec_HasDtor, !cir.ptr<!rec_HasDtor>, ["openacc.firstprivate.init"]
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } copy {
// CHECK-NEXT: ^bb0(%[[ARG_FROM:.*]]: !cir.ptr<!rec_HasDtor> {{.*}}, %[[ARG_TO:.*]]: !cir.ptr<!rec_HasDtor> {{.*}}):
// CHECK-NEXT: cir.call @_ZN7HasDtorC1ERKS_(%[[ARG_TO]], %[[ARG_FROM]]) nothrow : (!cir.ptr<!rec_HasDtor>, !cir.ptr<!rec_HasDtor>) -> ()
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ORIG:.*]]: !cir.ptr<!rec_HasDtor> {{.*}}, %[[ARG:.*]]: !cir.ptr<!rec_HasDtor> {{.*}}):
// CHECK-NEXT: cir.call @_ZN7HasDtorD1Ev(%[[ARG]]) nothrow : (!cir.ptr<!rec_HasDtor>) -> ()
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
//
// CHECK-NEXT: acc.firstprivate.recipe @firstprivatization__ZTS14NonDefaultCtor : !cir.ptr<!rec_NonDefaultCtor> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_NonDefaultCtor> {{.*}}):
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !rec_NonDefaultCtor, !cir.ptr<!rec_NonDefaultCtor>, ["openacc.firstprivate.init"]
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } copy {
// CHECK-NEXT: ^bb0(%[[ARG_FROM:.*]]: !cir.ptr<!rec_NonDefaultCtor> {{.*}}, %[[ARG_TO:.*]]: !cir.ptr<!rec_NonDefaultCtor> {{.*}}):
// CHECK-NEXT: cir.call @_ZN14NonDefaultCtorC1ERKS_(%[[ARG_TO]], %[[ARG_FROM]]) nothrow : (!cir.ptr<!rec_NonDefaultCtor>, !cir.ptr<!rec_NonDefaultCtor>) -> ()
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
//
// CHECK-NEXT: acc.firstprivate.recipe @firstprivatization__ZTS13CopyConstruct : !cir.ptr<!rec_CopyConstruct> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_CopyConstruct> {{.*}}):
// CHECK-NEXT: cir.alloca !rec_CopyConstruct, !cir.ptr<!rec_CopyConstruct>, ["openacc.firstprivate.init"]
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } copy {
// CHECK-NEXT: ^bb0(%[[ARG_FROM:.*]]: !cir.ptr<!rec_CopyConstruct> {{.*}}, %[[ARG_TO:.*]]: !cir.ptr<!rec_CopyConstruct> {{.*}}):
// CHECK-NEXT:  cir.call @_ZN13CopyConstructC1ERKS_(%[[ARG_TO]], %[[ARG_FROM]]) : (!cir.ptr<!rec_CopyConstruct>, !cir.ptr<!rec_CopyConstruct>) -> ()
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }

template<typename T, typename U, typename V, typename W>
void dependent_version(const T &cc, const U &ndc, const V &dtor, const W &someInt) {
  // CHECK: cir.func {{.*}}@_Z17dependent_versionI13CopyConstruct14NonDefaultCtor7HasDtoriEvRKT_RKT0_RKT1_RKT2_(%[[ARG0:.*]]: !cir.ptr<!rec_CopyConstruct> {{.*}}, %[[ARG1:.*]]: !cir.ptr<!rec_NonDefaultCtor> {{.*}}, %[[ARG2:.*]]: !cir.ptr<!rec_HasDtor> {{.*}}, %[[ARG3:.*]]: !cir.ptr<!s32i> {{.*}}) {
  // CHECK-NEXT: %[[CC:.*]] = cir.alloca !cir.ptr<!rec_CopyConstruct>, !cir.ptr<!cir.ptr<!rec_CopyConstruct>>, ["cc", init, const]
  // CHECK-NEXT: %[[NDC:.*]] = cir.alloca !cir.ptr<!rec_NonDefaultCtor>, !cir.ptr<!cir.ptr<!rec_NonDefaultCtor>>, ["ndc", init, const]
  // CHECK-NEXT: %[[DTOR:.*]] = cir.alloca !cir.ptr<!rec_HasDtor>, !cir.ptr<!cir.ptr<!rec_HasDtor>>, ["dtor", init, const]
  // CHECK-NEXT: %[[SOMEINT:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["someInt", init, const]
  //             %             3 = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["someInt", init, const]

#pragma acc parallel firstprivate(cc, ndc, dtor, someInt)
  ;
  // CHECK: %[[PRIV_LOAD:.*]] = cir.load %[[CC]] : !cir.ptr<!cir.ptr<!rec_CopyConstruct>>, !cir.ptr<!rec_CopyConstruct>
  // CHECK-NEXT: %[[PRIVATE1:.*]] = acc.firstprivate varPtr(%[[PRIV_LOAD]] : !cir.ptr<!rec_CopyConstruct>) -> !cir.ptr<!rec_CopyConstruct> {name = "cc"}
  // CHECK-NEXT: %[[PRIV_LOAD:.*]] = cir.load %[[NDC]] : !cir.ptr<!cir.ptr<!rec_NonDefaultCtor>>, !cir.ptr<!rec_NonDefaultCtor>
  // CHECK-NEXT: %[[PRIVATE2:.*]] = acc.firstprivate varPtr(%[[PRIV_LOAD]] : !cir.ptr<!rec_NonDefaultCtor>) -> !cir.ptr<!rec_NonDefaultCtor> {name = "ndc"}
  // CHECK-NEXT: %[[PRIV_LOAD:.*]] = cir.load %[[DTOR]] : !cir.ptr<!cir.ptr<!rec_HasDtor>>, !cir.ptr<!rec_HasDtor>
  // CHECK-NEXT: %[[PRIVATE3:.*]] = acc.firstprivate varPtr(%[[PRIV_LOAD]] : !cir.ptr<!rec_HasDtor>) -> !cir.ptr<!rec_HasDtor> {name = "dtor"}
  // CHECK-NEXT: %[[PRIV_LOAD:.*]] = cir.load %[[SOMEINT]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
  // CHECK-NEXT: %[[PRIVATE4:.*]] = acc.firstprivate varPtr(%[[PRIV_LOAD]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {name = "someInt"}

  // CHECK-NEXT: acc.parallel firstprivate(@firstprivatization__ZTS13CopyConstruct -> %[[PRIVATE1]] : !cir.ptr<!rec_CopyConstruct>, 
  // CHECK-SAME: @firstprivatization__ZTS14NonDefaultCtor -> %[[PRIVATE2]] : !cir.ptr<!rec_NonDefaultCtor>,
  // CHECK-SAME: @firstprivatization__ZTS7HasDtor -> %[[PRIVATE3]] : !cir.ptr<!rec_HasDtor>,
  // CHECK-SAME: @firstprivatization__ZTSi -> %[[PRIVATE4]] : !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
}

void use() {
  CopyConstruct cc;
  NonDefaultCtor ndc;
  HasDtor dtor;
  int i;
  dependent_version(cc, ndc, dtor, i);
}
