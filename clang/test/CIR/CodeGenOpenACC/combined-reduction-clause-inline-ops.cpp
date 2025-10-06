// RUN: %clang_cc1 -fopenacc -triple x86_64-linux-gnu -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir -triple x86_64-linux-pc %s -o - | FileCheck %s

struct HasOperatorsInline {
  int i;
  unsigned u;
  float f;
  double d;
  bool b;

  ~HasOperatorsInline();

  HasOperatorsInline &operator+=(HasOperatorsInline& other);
  HasOperatorsInline &operator*=(HasOperatorsInline& other);
  HasOperatorsInline &operator&=(HasOperatorsInline& other);
  HasOperatorsInline &operator|=(HasOperatorsInline& other);
  HasOperatorsInline &operator^=(HasOperatorsInline& other);
  bool &operator&&(HasOperatorsInline& other);
  bool &operator||(HasOperatorsInline& other);
  // For min/max
  HasOperatorsInline &operator<(HasOperatorsInline& other);
  HasOperatorsInline &operator=(HasOperatorsInline& other);
};

template<typename T>
void acc_combined() {
  T someVar;
  T someVarArr[5];
#pragma acc parallel loop reduction(+:someVar)
// CHECK: acc.reduction.recipe @reduction_add__ZTS18HasOperatorsInline : !cir.ptr<!rec_HasOperatorsInline> reduction_operator <add> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_HasOperatorsInline>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !rec_HasOperatorsInline, !cir.ptr<!rec_HasOperatorsInline>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[ALLOCA]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[ALLOCA]][1] {name = "u"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[ALLOCA]][2] {name = "f"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[ALLOCA]][4] {name = "d"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[ALLOCA]][5] {name = "b"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #false
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ORIG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}}, %[[ARG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}}):  
// CHECK-NEXT: cir.call @_ZN18HasOperatorsInlineD1Ev(%[[ARG]]) nothrow : (!cir.ptr<!rec_HasOperatorsInline>)
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  for(int i=0;i < 5; ++i);
#pragma acc parallel loop reduction(*:someVar)
// CHECK: acc.reduction.recipe @reduction_mul__ZTS18HasOperatorsInline : !cir.ptr<!rec_HasOperatorsInline> reduction_operator <mul> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_HasOperatorsInline>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !rec_HasOperatorsInline, !cir.ptr<!rec_HasOperatorsInline>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[ALLOCA]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[ALLOCA]][1] {name = "u"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[ALLOCA]][2] {name = "f"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[ALLOCA]][4] {name = "d"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[ALLOCA]][5] {name = "b"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ORIG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}}, %[[ARG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}}):  
// CHECK-NEXT: cir.call @_ZN18HasOperatorsInlineD1Ev(%[[ARG]]) nothrow : (!cir.ptr<!rec_HasOperatorsInline>)
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  for(int i=0;i < 5; ++i);
#pragma acc parallel loop reduction(max:someVar)
// CHECK-NEXT: acc.reduction.recipe @reduction_max__ZTS18HasOperatorsInline : !cir.ptr<!rec_HasOperatorsInline> reduction_operator <max> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_HasOperatorsInline>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !rec_HasOperatorsInline, !cir.ptr<!rec_HasOperatorsInline>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[ALLOCA]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.int<-2147483648> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[ALLOCA]][1] {name = "u"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.int<0> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[ALLOCA]][2] {name = "f"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.fp<-3.4{{.*}}E+38> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[ALLOCA]][4] {name = "d"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.fp<-1.7{{.*}}E+308> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[ALLOCA]][5] {name = "b"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #false
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ORIG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}}, %[[ARG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}}):  
// CHECK-NEXT: cir.call @_ZN18HasOperatorsInlineD1Ev(%[[ARG]]) nothrow : (!cir.ptr<!rec_HasOperatorsInline>)
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  for(int i=0;i < 5; ++i);
#pragma acc parallel loop reduction(min:someVar)
// CHECK-NEXT: acc.reduction.recipe @reduction_min__ZTS18HasOperatorsInline : !cir.ptr<!rec_HasOperatorsInline> reduction_operator <min> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_HasOperatorsInline>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !rec_HasOperatorsInline, !cir.ptr<!rec_HasOperatorsInline>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[ALLOCA]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.int<2147483647> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[ALLOCA]][1] {name = "u"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.int<4294967295> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[ALLOCA]][2] {name = "f"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.fp<3.4{{.*}}E+38> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[ALLOCA]][4] {name = "d"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.fp<1.7{{.*}}E+308> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[ALLOCA]][5] {name = "b"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ORIG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}}, %[[ARG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}}):  
// CHECK-NEXT: cir.call @_ZN18HasOperatorsInlineD1Ev(%[[ARG]]) nothrow : (!cir.ptr<!rec_HasOperatorsInline>)
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  for(int i=0;i < 5; ++i);
#pragma acc parallel loop reduction(&:someVar)
// CHECK-NEXT: acc.reduction.recipe @reduction_iand__ZTS18HasOperatorsInline : !cir.ptr<!rec_HasOperatorsInline> reduction_operator <iand> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_HasOperatorsInline>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !rec_HasOperatorsInline, !cir.ptr<!rec_HasOperatorsInline>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[ALLOCA]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #cir.int<-1> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[ALLOCA]][1] {name = "u"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #cir.int<4294967295> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[ALLOCA]][2] {name = "f"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #cir.fp<0xFF{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[ALLOCA]][4] {name = "d"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #cir.fp<0xFF{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[ALLOCA]][5] {name = "b"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ORIG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}}, %[[ARG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}}):  
// CHECK-NEXT: cir.call @_ZN18HasOperatorsInlineD1Ev(%[[ARG]]) nothrow : (!cir.ptr<!rec_HasOperatorsInline>)
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  for(int i=0;i < 5; ++i);
#pragma acc parallel loop reduction(|:someVar)
// CHECK-NEXT: acc.reduction.recipe @reduction_ior__ZTS18HasOperatorsInline : !cir.ptr<!rec_HasOperatorsInline> reduction_operator <ior> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_HasOperatorsInline>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !rec_HasOperatorsInline, !cir.ptr<!rec_HasOperatorsInline>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[ALLOCA]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[ALLOCA]][1] {name = "u"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[ALLOCA]][2] {name = "f"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[ALLOCA]][4] {name = "d"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[ALLOCA]][5] {name = "b"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #false
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ORIG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}}, %[[ARG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}}):  
// CHECK-NEXT: cir.call @_ZN18HasOperatorsInlineD1Ev(%[[ARG]]) nothrow : (!cir.ptr<!rec_HasOperatorsInline>)
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  for(int i=0;i < 5; ++i);
#pragma acc parallel loop reduction(^:someVar)
// CHECK-NEXT: acc.reduction.recipe @reduction_xor__ZTS18HasOperatorsInline : !cir.ptr<!rec_HasOperatorsInline> reduction_operator <xor> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_HasOperatorsInline>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !rec_HasOperatorsInline, !cir.ptr<!rec_HasOperatorsInline>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[ALLOCA]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[ALLOCA]][1] {name = "u"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[ALLOCA]][2] {name = "f"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[ALLOCA]][4] {name = "d"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[ALLOCA]][5] {name = "b"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #false
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ORIG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}}, %[[ARG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}}):  
// CHECK-NEXT: cir.call @_ZN18HasOperatorsInlineD1Ev(%[[ARG]]) nothrow : (!cir.ptr<!rec_HasOperatorsInline>)
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  for(int i=0;i < 5; ++i);
#pragma acc parallel loop reduction(&&:someVar)
// CHECK-NEXT: acc.reduction.recipe @reduction_land__ZTS18HasOperatorsInline : !cir.ptr<!rec_HasOperatorsInline> reduction_operator <land> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_HasOperatorsInline>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !rec_HasOperatorsInline, !cir.ptr<!rec_HasOperatorsInline>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[ALLOCA]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[ALLOCA]][1] {name = "u"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[ALLOCA]][2] {name = "f"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[ALLOCA]][4] {name = "d"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[ALLOCA]][5] {name = "b"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ORIG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}}, %[[ARG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}}):  
// CHECK-NEXT: cir.call @_ZN18HasOperatorsInlineD1Ev(%[[ARG]]) nothrow : (!cir.ptr<!rec_HasOperatorsInline>)
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  for(int i=0;i < 5; ++i);
#pragma acc parallel loop reduction(||:someVar)
// CHECK-NEXT: acc.reduction.recipe @reduction_lor__ZTS18HasOperatorsInline : !cir.ptr<!rec_HasOperatorsInline> reduction_operator <lor> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_HasOperatorsInline>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !rec_HasOperatorsInline, !cir.ptr<!rec_HasOperatorsInline>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[ALLOCA]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[ALLOCA]][1] {name = "u"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[ALLOCA]][2] {name = "f"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[ALLOCA]][4] {name = "d"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[ALLOCA]][5] {name = "b"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #false
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ORIG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}}, %[[ARG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}}):  
// CHECK-NEXT: cir.call @_ZN18HasOperatorsInlineD1Ev(%[[ARG]]) nothrow : (!cir.ptr<!rec_HasOperatorsInline>)
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  for(int i=0;i < 5; ++i);

#pragma acc parallel loop reduction(+:someVarArr)
// CHECK-NEXT: acc.reduction.recipe @reduction_add__ZTSA5_18HasOperatorsInline : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> reduction_operator <add> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!rec_HasOperatorsInline x 5>, !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[TEMP_ITR:.*]] = cir.alloca !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, ["arrayinit.temp"]
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.store {{.*}} %[[DECAY]], %[[TEMP_ITR]] : !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>
// CHECK-NEXT: %[[LAST_IDX:.*]] = cir.const #cir.int<5> : !s64i
// CHECK-NEXT: %[[END_ITR:.*]] = cir.ptr_stride %[[DECAY]], %[[LAST_IDX]] : (!cir.ptr<!rec_HasOperatorsInline>, !s64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load {{.*}} %[[TEMP_ITR]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[TEMP_LOAD]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[TEMP_LOAD]][1] {name = "u"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[TEMP_LOAD]][2] {name = "f"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[TEMP_LOAD]][4] {name = "d"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[TEMP_LOAD]][5] {name = "b"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #false
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s64i
// CHECK-NEXT: %[[NEXT_ITEM:.*]] = cir.ptr_stride %[[TEMP_LOAD]], %[[ONE]] : (!cir.ptr<!rec_HasOperatorsInline>, !s64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.store {{.*}} %[[NEXT_ITEM]], %[[TEMP_ITR]] : !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load {{.*}} %[[TEMP_ITR]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(ne, %[[TEMP_LOAD]], %[[END_ITR]]) : !cir.ptr<!rec_HasOperatorsInline>, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]]) 
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}):  
// CHECK-NEXT: %[[SIZE:.*]] = cir.const #cir.int<4>  : !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[LAST_ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[SIZE]] : (!cir.ptr<!rec_HasOperatorsInline>, !u64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[IDX:.*]] = cir.alloca !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, ["__array_idx"] {alignment = 1 : i64}
// CHECK-NEXT: cir.store %[[LAST_ELT]], %[[IDX]] : !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[CUR:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.call @_ZN18HasOperatorsInlineD1Ev(%[[CUR]]) nothrow : (!cir.ptr<!rec_HasOperatorsInline>)
// CHECK-NEXT: %[[NEG:.*]] = cir.const #cir.int<-1> : !s64i
// CHECK-NEXT: %[[NEW_ITEM:.*]] = cir.ptr_stride %[[CUR]], %[[NEG]] : (!cir.ptr<!rec_HasOperatorsInline>, !s64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.store %[[NEW_ITEM]], %[[IDX]] : !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[CUR_LOAD:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(ne, %[[CUR_LOAD]], %[[DECAY]]) : !cir.ptr<!rec_HasOperatorsInline>, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]])
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  for(int i=0;i < 5; ++i);
#pragma acc parallel loop reduction(*:someVarArr)
// CHECK-NEXT: acc.reduction.recipe @reduction_mul__ZTSA5_18HasOperatorsInline : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> reduction_operator <mul> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!rec_HasOperatorsInline x 5>, !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[DECAY]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[DECAY]][1] {name = "u"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[DECAY]][2] {name = "f"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[DECAY]][4] {name = "d"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[DECAY]][5] {name = "b"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
//
// CHECK-NEXT: %[[ONE_IDX:.*]] = cir.const #cir.int<1> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[ONE_IDX]] : (!cir.ptr<!rec_HasOperatorsInline>, !s64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[NEXT_ELT]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[NEXT_ELT]][1] {name = "u"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[NEXT_ELT]][2] {name = "f"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[NEXT_ELT]][4] {name = "d"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[NEXT_ELT]][5] {name = "b"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
//
// CHECK-NEXT: %[[TWO_IDX:.*]] = cir.const #cir.int<2> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[TWO_IDX]] : (!cir.ptr<!rec_HasOperatorsInline>, !s64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[NEXT_ELT]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[NEXT_ELT]][1] {name = "u"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[NEXT_ELT]][2] {name = "f"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[NEXT_ELT]][4] {name = "d"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[NEXT_ELT]][5] {name = "b"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
//
//
// CHECK-NEXT: %[[THREE_IDX:.*]] = cir.const #cir.int<3> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[THREE_IDX]] : (!cir.ptr<!rec_HasOperatorsInline>, !s64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[NEXT_ELT]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[NEXT_ELT]][1] {name = "u"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[NEXT_ELT]][2] {name = "f"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[NEXT_ELT]][4] {name = "d"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[NEXT_ELT]][5] {name = "b"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
//
// CHECK-NEXT: %[[FOUR_IDX:.*]] = cir.const #cir.int<4> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[FOUR_IDX]] : (!cir.ptr<!rec_HasOperatorsInline>, !s64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[NEXT_ELT]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[NEXT_ELT]][1] {name = "u"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[NEXT_ELT]][2] {name = "f"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[NEXT_ELT]][4] {name = "d"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[NEXT_ELT]][5] {name = "b"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
//
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ORIG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}):  
// CHECK-NEXT: %[[SIZE:.*]] = cir.const #cir.int<4>  : !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[LAST_ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[SIZE]] : (!cir.ptr<!rec_HasOperatorsInline>, !u64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[IDX:.*]] = cir.alloca !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, ["__array_idx"] {alignment = 1 : i64}
// CHECK-NEXT: cir.store %[[LAST_ELT]], %[[IDX]] : !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[CUR:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.call @_ZN18HasOperatorsInlineD1Ev(%[[CUR]]) nothrow : (!cir.ptr<!rec_HasOperatorsInline>)
// CHECK-NEXT: %[[NEG:.*]] = cir.const #cir.int<-1> : !s64i
// CHECK-NEXT: %[[NEW_ITEM:.*]] = cir.ptr_stride %[[CUR]], %[[NEG]] : (!cir.ptr<!rec_HasOperatorsInline>, !s64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.store %[[NEW_ITEM]], %[[IDX]] : !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[CUR_LOAD:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(ne, %[[CUR_LOAD]], %[[DECAY]]) : !cir.ptr<!rec_HasOperatorsInline>, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]])
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  for(int i=0;i < 5; ++i);
#pragma acc parallel loop reduction(max:someVarArr)
// CHECK-NEXT: acc.reduction.recipe @reduction_max__ZTSA5_18HasOperatorsInline : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> reduction_operator <max> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!rec_HasOperatorsInline x 5>, !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[DECAY]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.int<-2147483648> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[DECAY]][1] {name = "u"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.int<0> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[DECAY]][2] {name = "f"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.fp<-3.4{{.*}}E+38> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[DECAY]][4] {name = "d"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.fp<-1.7{{.*}}E+308> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[DECAY]][5] {name = "b"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #false
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
//
// CHECK-NEXT: %[[LEAST_IDX:.*]] = cir.const #cir.int<1> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[LEAST_IDX]] : (!cir.ptr<!rec_HasOperatorsInline>, !s64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[NEXT_ELT]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.int<-2147483648> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[NEXT_ELT]][1] {name = "u"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.int<0> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[NEXT_ELT]][2] {name = "f"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.fp<-3.4{{.*}}E+38> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[NEXT_ELT]][4] {name = "d"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.fp<-1.7{{.*}}E+308> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[NEXT_ELT]][5] {name = "b"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #false
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
//
// CHECK-NEXT: %[[TWO_IDX:.*]] = cir.const #cir.int<2> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[TWO_IDX]] : (!cir.ptr<!rec_HasOperatorsInline>, !s64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[NEXT_ELT]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.int<-2147483648> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[NEXT_ELT]][1] {name = "u"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.int<0> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[NEXT_ELT]][2] {name = "f"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.fp<-3.4{{.*}}E+38> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[NEXT_ELT]][4] {name = "d"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.fp<-1.7{{.*}}E+308> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[NEXT_ELT]][5] {name = "b"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #false
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
//
//
// CHECK-NEXT: %[[THREE_IDX:.*]] = cir.const #cir.int<3> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[THREE_IDX]] : (!cir.ptr<!rec_HasOperatorsInline>, !s64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[NEXT_ELT]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.int<-2147483648> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[NEXT_ELT]][1] {name = "u"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.int<0> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[NEXT_ELT]][2] {name = "f"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.fp<-3.4{{.*}}E+38> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[NEXT_ELT]][4] {name = "d"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.fp<-1.7{{.*}}E+308> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[NEXT_ELT]][5] {name = "b"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #false
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
//
// CHECK-NEXT: %[[FOUR_IDX:.*]] = cir.const #cir.int<4> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[FOUR_IDX]] : (!cir.ptr<!rec_HasOperatorsInline>, !s64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[NEXT_ELT]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.int<-2147483648> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[NEXT_ELT]][1] {name = "u"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.int<0> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[NEXT_ELT]][2] {name = "f"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.fp<-3.4{{.*}}E+38> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[NEXT_ELT]][4] {name = "d"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.fp<-1.7{{.*}}E+308> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[NEXT_ELT]][5] {name = "b"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #false
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
//
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ORIG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}):  
// CHECK-NEXT: %[[SIZE:.*]] = cir.const #cir.int<4>  : !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[LAST_ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[SIZE]] : (!cir.ptr<!rec_HasOperatorsInline>, !u64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[IDX:.*]] = cir.alloca !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, ["__array_idx"] {alignment = 1 : i64}
// CHECK-NEXT: cir.store %[[LAST_ELT]], %[[IDX]] : !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[CUR:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.call @_ZN18HasOperatorsInlineD1Ev(%[[CUR]]) nothrow : (!cir.ptr<!rec_HasOperatorsInline>)
// CHECK-NEXT: %[[NEG:.*]] = cir.const #cir.int<-1> : !s64i
// CHECK-NEXT: %[[NEW_ITEM:.*]] = cir.ptr_stride %[[CUR]], %[[NEG]] : (!cir.ptr<!rec_HasOperatorsInline>, !s64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.store %[[NEW_ITEM]], %[[IDX]] : !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[CUR_LOAD:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(ne, %[[CUR_LOAD]], %[[DECAY]]) : !cir.ptr<!rec_HasOperatorsInline>, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]])
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  for(int i=0;i < 5; ++i);
#pragma acc parallel loop reduction(min:someVarArr)
// CHECK-NEXT: acc.reduction.recipe @reduction_min__ZTSA5_18HasOperatorsInline : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> reduction_operator <min> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!rec_HasOperatorsInline x 5>, !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[DECAY]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.int<2147483647> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[DECAY]][1] {name = "u"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.int<4294967295> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[DECAY]][2] {name = "f"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.fp<3.4{{.*}}E+38> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[DECAY]][4] {name = "d"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.fp<1.7{{.*}}E+308> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[DECAY]][5] {name = "b"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
//
// CHECK-NEXT: %[[LARGEST_IDX:.*]] = cir.const #cir.int<1> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[LARGEST_IDX]] : (!cir.ptr<!rec_HasOperatorsInline>, !s64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[NEXT_ELT]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.int<2147483647> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[NEXT_ELT]][1] {name = "u"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.int<4294967295> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[NEXT_ELT]][2] {name = "f"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.fp<3.4{{.*}}E+38> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[NEXT_ELT]][4] {name = "d"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.fp<1.7{{.*}}E+308> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[NEXT_ELT]][5] {name = "b"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
//
// CHECK-NEXT: %[[TWO_IDX:.*]] = cir.const #cir.int<2> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[TWO_IDX]] : (!cir.ptr<!rec_HasOperatorsInline>, !s64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[NEXT_ELT]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.int<2147483647> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[NEXT_ELT]][1] {name = "u"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.int<4294967295> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[NEXT_ELT]][2] {name = "f"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.fp<3.4{{.*}}E+38> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[NEXT_ELT]][4] {name = "d"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.fp<1.7{{.*}}E+308> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[NEXT_ELT]][5] {name = "b"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
//
//
// CHECK-NEXT: %[[THREE_IDX:.*]] = cir.const #cir.int<3> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[THREE_IDX]] : (!cir.ptr<!rec_HasOperatorsInline>, !s64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[NEXT_ELT]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.int<2147483647> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[NEXT_ELT]][1] {name = "u"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.int<4294967295> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[NEXT_ELT]][2] {name = "f"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.fp<3.4{{.*}}E+38> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[NEXT_ELT]][4] {name = "d"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.fp<1.7{{.*}}E+308> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[NEXT_ELT]][5] {name = "b"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
//
// CHECK-NEXT: %[[FOUR_IDX:.*]] = cir.const #cir.int<4> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[FOUR_IDX]] : (!cir.ptr<!rec_HasOperatorsInline>, !s64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[NEXT_ELT]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.int<2147483647> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[NEXT_ELT]][1] {name = "u"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.int<4294967295> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[NEXT_ELT]][2] {name = "f"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.fp<3.4{{.*}}E+38> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[NEXT_ELT]][4] {name = "d"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.fp<1.7{{.*}}E+308> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[NEXT_ELT]][5] {name = "b"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
//
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ORIG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}):  
// CHECK-NEXT: %[[SIZE:.*]] = cir.const #cir.int<4>  : !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[LAST_ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[SIZE]] : (!cir.ptr<!rec_HasOperatorsInline>, !u64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[IDX:.*]] = cir.alloca !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, ["__array_idx"] {alignment = 1 : i64}
// CHECK-NEXT: cir.store %[[LAST_ELT]], %[[IDX]] : !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[CUR:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.call @_ZN18HasOperatorsInlineD1Ev(%[[CUR]]) nothrow : (!cir.ptr<!rec_HasOperatorsInline>)
// CHECK-NEXT: %[[NEG:.*]] = cir.const #cir.int<-1> : !s64i
// CHECK-NEXT: %[[NEW_ITEM:.*]] = cir.ptr_stride %[[CUR]], %[[NEG]] : (!cir.ptr<!rec_HasOperatorsInline>, !s64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.store %[[NEW_ITEM]], %[[IDX]] : !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[CUR_LOAD:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(ne, %[[CUR_LOAD]], %[[DECAY]]) : !cir.ptr<!rec_HasOperatorsInline>, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]])
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  for(int i=0;i < 5; ++i);
#pragma acc parallel loop reduction(&:someVarArr)
// CHECK-NEXT: acc.reduction.recipe @reduction_iand__ZTSA5_18HasOperatorsInline : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> reduction_operator <iand> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!rec_HasOperatorsInline x 5>, !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[DECAY]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #cir.int<-1> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[DECAY]][1] {name = "u"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #cir.int<4294967295> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[DECAY]][2] {name = "f"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #cir.fp<0xFF{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[DECAY]][4] {name = "d"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #cir.fp<0xFF{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[DECAY]][5] {name = "b"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
//
// CHECK-NEXT: %[[ALL_ONES_IDX:.*]] = cir.const #cir.int<1> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[ALL_ONES_IDX]] : (!cir.ptr<!rec_HasOperatorsInline>, !s64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[NEXT_ELT]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #cir.int<-1> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[NEXT_ELT]][1] {name = "u"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #cir.int<4294967295> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[NEXT_ELT]][2] {name = "f"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #cir.fp<0xFF{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[NEXT_ELT]][4] {name = "d"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #cir.fp<0xFF{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[NEXT_ELT]][5] {name = "b"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
//
// CHECK-NEXT: %[[TWO_IDX:.*]] = cir.const #cir.int<2> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[TWO_IDX]] : (!cir.ptr<!rec_HasOperatorsInline>, !s64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[NEXT_ELT]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #cir.int<-1> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[NEXT_ELT]][1] {name = "u"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #cir.int<4294967295> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[NEXT_ELT]][2] {name = "f"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #cir.fp<0xFF{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[NEXT_ELT]][4] {name = "d"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #cir.fp<0xFF{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[NEXT_ELT]][5] {name = "b"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
//
//
// CHECK-NEXT: %[[THREE_IDX:.*]] = cir.const #cir.int<3> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[THREE_IDX]] : (!cir.ptr<!rec_HasOperatorsInline>, !s64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[NEXT_ELT]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #cir.int<-1> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[NEXT_ELT]][1] {name = "u"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #cir.int<4294967295> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[NEXT_ELT]][2] {name = "f"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #cir.fp<0xFF{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[NEXT_ELT]][4] {name = "d"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #cir.fp<0xFF{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[NEXT_ELT]][5] {name = "b"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
//
// CHECK-NEXT: %[[FOUR_IDX:.*]] = cir.const #cir.int<4> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[FOUR_IDX]] : (!cir.ptr<!rec_HasOperatorsInline>, !s64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[NEXT_ELT]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #cir.int<-1> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[NEXT_ELT]][1] {name = "u"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #cir.int<4294967295> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[NEXT_ELT]][2] {name = "f"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #cir.fp<0xFF{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[NEXT_ELT]][4] {name = "d"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #cir.fp<0xFF{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[NEXT_ELT]][5] {name = "b"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
//
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ORIG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}):  
// CHECK-NEXT: %[[SIZE:.*]] = cir.const #cir.int<4>  : !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[LAST_ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[SIZE]] : (!cir.ptr<!rec_HasOperatorsInline>, !u64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[IDX:.*]] = cir.alloca !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, ["__array_idx"] {alignment = 1 : i64}
// CHECK-NEXT: cir.store %[[LAST_ELT]], %[[IDX]] : !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[CUR:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.call @_ZN18HasOperatorsInlineD1Ev(%[[CUR]]) nothrow : (!cir.ptr<!rec_HasOperatorsInline>)
// CHECK-NEXT: %[[NEG:.*]] = cir.const #cir.int<-1> : !s64i
// CHECK-NEXT: %[[NEW_ITEM:.*]] = cir.ptr_stride %[[CUR]], %[[NEG]] : (!cir.ptr<!rec_HasOperatorsInline>, !s64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.store %[[NEW_ITEM]], %[[IDX]] : !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[CUR_LOAD:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(ne, %[[CUR_LOAD]], %[[DECAY]]) : !cir.ptr<!rec_HasOperatorsInline>, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]])
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  for(int i=0;i < 5; ++i);
#pragma acc parallel loop reduction(|:someVarArr)
// CHECK-NEXT: acc.reduction.recipe @reduction_ior__ZTSA5_18HasOperatorsInline : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> reduction_operator <ior> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!rec_HasOperatorsInline x 5>, !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[TEMP_ITR:.*]] = cir.alloca !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, ["arrayinit.temp"]
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.store {{.*}} %[[DECAY]], %[[TEMP_ITR]] : !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>
// CHECK-NEXT: %[[LAST_IDX:.*]] = cir.const #cir.int<5> : !s64i
// CHECK-NEXT: %[[END_ITR:.*]] = cir.ptr_stride %[[DECAY]], %[[LAST_IDX]] : (!cir.ptr<!rec_HasOperatorsInline>, !s64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load {{.*}} %[[TEMP_ITR]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[TEMP_LOAD]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[TEMP_LOAD]][1] {name = "u"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[TEMP_LOAD]][2] {name = "f"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[TEMP_LOAD]][4] {name = "d"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[TEMP_LOAD]][5] {name = "b"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #false
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
//
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s64i
// CHECK-NEXT: %[[NEXT_ITEM:.*]] = cir.ptr_stride %[[TEMP_LOAD]], %[[ONE]] : (!cir.ptr<!rec_HasOperatorsInline>, !s64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.store {{.*}} %[[NEXT_ITEM]], %[[TEMP_ITR]] : !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load {{.*}} %[[TEMP_ITR]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(ne, %[[TEMP_LOAD]], %[[END_ITR]]) : !cir.ptr<!rec_HasOperatorsInline>, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]]) 
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ORIG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}):  
// CHECK-NEXT: %[[SIZE:.*]] = cir.const #cir.int<4>  : !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[LAST_ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[SIZE]] : (!cir.ptr<!rec_HasOperatorsInline>, !u64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[IDX:.*]] = cir.alloca !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, ["__array_idx"] {alignment = 1 : i64}
// CHECK-NEXT: cir.store %[[LAST_ELT]], %[[IDX]] : !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[CUR:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.call @_ZN18HasOperatorsInlineD1Ev(%[[CUR]]) nothrow : (!cir.ptr<!rec_HasOperatorsInline>)
// CHECK-NEXT: %[[NEG:.*]] = cir.const #cir.int<-1> : !s64i
// CHECK-NEXT: %[[NEW_ITEM:.*]] = cir.ptr_stride %[[CUR]], %[[NEG]] : (!cir.ptr<!rec_HasOperatorsInline>, !s64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.store %[[NEW_ITEM]], %[[IDX]] : !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[CUR_LOAD:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(ne, %[[CUR_LOAD]], %[[DECAY]]) : !cir.ptr<!rec_HasOperatorsInline>, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]])
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  for(int i=0;i < 5; ++i);
#pragma acc parallel loop reduction(^:someVarArr)
// CHECK-NEXT: acc.reduction.recipe @reduction_xor__ZTSA5_18HasOperatorsInline : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> reduction_operator <xor> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!rec_HasOperatorsInline x 5>, !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[TEMP_ITR:.*]] = cir.alloca !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, ["arrayinit.temp"]
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.store {{.*}} %[[DECAY]], %[[TEMP_ITR]] : !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>
// CHECK-NEXT: %[[LAST_IDX:.*]] = cir.const #cir.int<5> : !s64i
// CHECK-NEXT: %[[END_ITR:.*]] = cir.ptr_stride %[[DECAY]], %[[LAST_IDX]] : (!cir.ptr<!rec_HasOperatorsInline>, !s64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load {{.*}} %[[TEMP_ITR]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[TEMP_LOAD]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[TEMP_LOAD]][1] {name = "u"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[TEMP_LOAD]][2] {name = "f"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[TEMP_LOAD]][4] {name = "d"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[TEMP_LOAD]][5] {name = "b"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #false
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s64i
// CHECK-NEXT: %[[NEXT_ITEM:.*]] = cir.ptr_stride %[[TEMP_LOAD]], %[[ONE]] : (!cir.ptr<!rec_HasOperatorsInline>, !s64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.store {{.*}} %[[NEXT_ITEM]], %[[TEMP_ITR]] : !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load {{.*}} %[[TEMP_ITR]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(ne, %[[TEMP_LOAD]], %[[END_ITR]]) : !cir.ptr<!rec_HasOperatorsInline>, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]]) 
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ORIG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}):  
// CHECK-NEXT: %[[SIZE:.*]] = cir.const #cir.int<4>  : !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[LAST_ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[SIZE]] : (!cir.ptr<!rec_HasOperatorsInline>, !u64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[IDX:.*]] = cir.alloca !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, ["__array_idx"] {alignment = 1 : i64}
// CHECK-NEXT: cir.store %[[LAST_ELT]], %[[IDX]] : !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[CUR:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.call @_ZN18HasOperatorsInlineD1Ev(%[[CUR]]) nothrow : (!cir.ptr<!rec_HasOperatorsInline>)
// CHECK-NEXT: %[[NEG:.*]] = cir.const #cir.int<-1> : !s64i
// CHECK-NEXT: %[[NEW_ITEM:.*]] = cir.ptr_stride %[[CUR]], %[[NEG]] : (!cir.ptr<!rec_HasOperatorsInline>, !s64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.store %[[NEW_ITEM]], %[[IDX]] : !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[CUR_LOAD:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(ne, %[[CUR_LOAD]], %[[DECAY]]) : !cir.ptr<!rec_HasOperatorsInline>, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]])
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }

  for(int i=0;i < 5; ++i);
#pragma acc parallel loop reduction(&&:someVarArr)
// CHECK-NEXT: acc.reduction.recipe @reduction_land__ZTSA5_18HasOperatorsInline : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> reduction_operator <land> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!rec_HasOperatorsInline x 5>, !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[DECAY]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[DECAY]][1] {name = "u"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[DECAY]][2] {name = "f"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[DECAY]][4] {name = "d"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[DECAY]][5] {name = "b"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
//
// CHECK-NEXT: %[[ONE_IDX:.*]] = cir.const #cir.int<1> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[ONE_IDX]] : (!cir.ptr<!rec_HasOperatorsInline>, !s64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[NEXT_ELT]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[NEXT_ELT]][1] {name = "u"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[NEXT_ELT]][2] {name = "f"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[NEXT_ELT]][4] {name = "d"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[NEXT_ELT]][5] {name = "b"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
//
// CHECK-NEXT: %[[TWO_IDX:.*]] = cir.const #cir.int<2> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[TWO_IDX]] : (!cir.ptr<!rec_HasOperatorsInline>, !s64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[NEXT_ELT]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[NEXT_ELT]][1] {name = "u"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[NEXT_ELT]][2] {name = "f"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[NEXT_ELT]][4] {name = "d"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[NEXT_ELT]][5] {name = "b"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
//
//
// CHECK-NEXT: %[[THREE_IDX:.*]] = cir.const #cir.int<3> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[THREE_IDX]] : (!cir.ptr<!rec_HasOperatorsInline>, !s64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[NEXT_ELT]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[NEXT_ELT]][1] {name = "u"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[NEXT_ELT]][2] {name = "f"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[NEXT_ELT]][4] {name = "d"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[NEXT_ELT]][5] {name = "b"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
//
// CHECK-NEXT: %[[FOUR_IDX:.*]] = cir.const #cir.int<4> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[FOUR_IDX]] : (!cir.ptr<!rec_HasOperatorsInline>, !s64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[NEXT_ELT]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[NEXT_ELT]][1] {name = "u"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[NEXT_ELT]][2] {name = "f"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[NEXT_ELT]][4] {name = "d"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[NEXT_ELT]][5] {name = "b"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
//
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ORIG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}):  
// CHECK-NEXT: %[[SIZE:.*]] = cir.const #cir.int<4>  : !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[LAST_ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[SIZE]] : (!cir.ptr<!rec_HasOperatorsInline>, !u64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[IDX:.*]] = cir.alloca !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, ["__array_idx"] {alignment = 1 : i64}
// CHECK-NEXT: cir.store %[[LAST_ELT]], %[[IDX]] : !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[CUR:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.call @_ZN18HasOperatorsInlineD1Ev(%[[CUR]]) nothrow : (!cir.ptr<!rec_HasOperatorsInline>)
// CHECK-NEXT: %[[NEG:.*]] = cir.const #cir.int<-1> : !s64i
// CHECK-NEXT: %[[NEW_ITEM:.*]] = cir.ptr_stride %[[CUR]], %[[NEG]] : (!cir.ptr<!rec_HasOperatorsInline>, !s64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.store %[[NEW_ITEM]], %[[IDX]] : !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[CUR_LOAD:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(ne, %[[CUR_LOAD]], %[[DECAY]]) : !cir.ptr<!rec_HasOperatorsInline>, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]])
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  for(int i=0;i < 5; ++i);
#pragma acc parallel loop reduction(||:someVarArr)
// CHECK: acc.reduction.recipe @reduction_lor__ZTSA5_18HasOperatorsInline : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> reduction_operator <lor> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!rec_HasOperatorsInline x 5>, !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[TEMP_ITR:.*]] = cir.alloca !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, ["arrayinit.temp"]
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.store {{.*}} %[[DECAY]], %[[TEMP_ITR]] : !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>
// CHECK-NEXT: %[[LAST_IDX:.*]] = cir.const #cir.int<5> : !s64i
// CHECK-NEXT: %[[END_ITR:.*]] = cir.ptr_stride %[[DECAY]], %[[LAST_IDX]] : (!cir.ptr<!rec_HasOperatorsInline>, !s64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load {{.*}} %[[TEMP_ITR]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[TEMP_LOAD]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[TEMP_LOAD]][1] {name = "u"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[TEMP_LOAD]][2] {name = "f"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[TEMP_LOAD]][4] {name = "d"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[TEMP_LOAD]][5] {name = "b"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #false
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
//
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s64i
// CHECK-NEXT: %[[NEXT_ITEM:.*]] = cir.ptr_stride %[[TEMP_LOAD]], %[[ONE]] : (!cir.ptr<!rec_HasOperatorsInline>, !s64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.store {{.*}} %[[NEXT_ITEM]], %[[TEMP_ITR]] : !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load {{.*}} %[[TEMP_ITR]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(ne, %[[TEMP_LOAD]], %[[END_ITR]]) : !cir.ptr<!rec_HasOperatorsInline>, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]]) 
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ORIG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}):  
// CHECK-NEXT: %[[SIZE:.*]] = cir.const #cir.int<4>  : !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[LAST_ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[SIZE]] : (!cir.ptr<!rec_HasOperatorsInline>, !u64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[IDX:.*]] = cir.alloca !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, ["__array_idx"] {alignment = 1 : i64}
// CHECK-NEXT: cir.store %[[LAST_ELT]], %[[IDX]] : !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[CUR:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.call @_ZN18HasOperatorsInlineD1Ev(%[[CUR]]) nothrow : (!cir.ptr<!rec_HasOperatorsInline>)
// CHECK-NEXT: %[[NEG:.*]] = cir.const #cir.int<-1> : !s64i
// CHECK-NEXT: %[[NEW_ITEM:.*]] = cir.ptr_stride %[[CUR]], %[[NEG]] : (!cir.ptr<!rec_HasOperatorsInline>, !s64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.store %[[NEW_ITEM]], %[[IDX]] : !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[CUR_LOAD:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(ne, %[[CUR_LOAD]], %[[DECAY]]) : !cir.ptr<!rec_HasOperatorsInline>, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]])
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  for(int i=0;i < 5; ++i);

#pragma acc parallel loop reduction(+:someVarArr[2])
// CHECK-NEXT: acc.reduction.recipe @reduction_add__Bcnt1__ZTSA5_18HasOperatorsInline : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> reduction_operator <add> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>{{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!rec_HasOperatorsInline x 5>, !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>, ["openacc.reduction.init"]
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB:.*]] = acc.get_lowerbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB]] : index to !u64i
// CHECK-NEXT: %[[UB:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB]] : index to !u64i
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["iter"] {alignment = 8 : i64}
// CHECK-NEXT: cir.store %[[LB_CAST]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[UB_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride %[[DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!rec_HasOperatorsInline>, !u64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[STRIDE]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[STRIDE]][1] {name = "u"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[STRIDE]][2] {name = "f"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[STRIDE]][4] {name = "d"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[STRIDE]][5] {name = "b"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #false
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ORIG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}): 
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB:.*]] = acc.get_lowerbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB]] : index to !u64i
// CHECK-NEXT: %[[UB:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB]] : index to !u64i
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["iter"] {alignment = 8 : i64}
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u64i
// CHECK-NEXT: %[[LAST_SUB_ONE:.*]] = cir.binop(sub, %[[UB_CAST]], %[[ONE]]) : !u64i
// CHECK-NEXT: cir.store %[[LAST_SUB_ONE]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(ge, %[[ITR_LOAD]], %[[LB_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride %[[DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!rec_HasOperatorsInline>, !u64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.call @_ZN18HasOperatorsInlineD1Ev(%[[STRIDE]]) nothrow : (!cir.ptr<!rec_HasOperatorsInline>) -> ()
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[DEC:.*]] = cir.unary(dec, %[[ITR_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[DEC]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  for(int i=0;i < 5; ++i);
#pragma acc parallel loop reduction(*:someVarArr[2])
// CHECK-NEXT: acc.reduction.recipe @reduction_mul__Bcnt1__ZTSA5_18HasOperatorsInline : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> reduction_operator <mul> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>{{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!rec_HasOperatorsInline x 5>, !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>, ["openacc.reduction.init"]
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB:.*]] = acc.get_lowerbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB]] : index to !u64i
// CHECK-NEXT: %[[UB:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB]] : index to !u64i
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["iter"] {alignment = 8 : i64}
// CHECK-NEXT: cir.store %[[LB_CAST]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[UB_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride %[[DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!rec_HasOperatorsInline>, !u64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[STRIDE]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[STRIDE]][1] {name = "u"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[STRIDE]][2] {name = "f"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[STRIDE]][4] {name = "d"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[STRIDE]][5] {name = "b"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ORIG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}): 
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB:.*]] = acc.get_lowerbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB]] : index to !u64i
// CHECK-NEXT: %[[UB:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB]] : index to !u64i
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["iter"] {alignment = 8 : i64}
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u64i
// CHECK-NEXT: %[[LAST_SUB_ONE:.*]] = cir.binop(sub, %[[UB_CAST]], %[[ONE]]) : !u64i
// CHECK-NEXT: cir.store %[[LAST_SUB_ONE]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(ge, %[[ITR_LOAD]], %[[LB_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride %[[DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!rec_HasOperatorsInline>, !u64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.call @_ZN18HasOperatorsInlineD1Ev(%[[STRIDE]]) nothrow : (!cir.ptr<!rec_HasOperatorsInline>) -> ()
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[DEC:.*]] = cir.unary(dec, %[[ITR_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[DEC]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  for(int i=0;i < 5; ++i);
#pragma acc parallel loop reduction(max:someVarArr[2])
// CHECK-NEXT: acc.reduction.recipe @reduction_max__Bcnt1__ZTSA5_18HasOperatorsInline : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> reduction_operator <max> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>{{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!rec_HasOperatorsInline x 5>, !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>, ["openacc.reduction.init"]
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB:.*]] = acc.get_lowerbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB]] : index to !u64i
// CHECK-NEXT: %[[UB:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB]] : index to !u64i
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["iter"] {alignment = 8 : i64}
// CHECK-NEXT: cir.store %[[LB_CAST]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[UB_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride %[[DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!rec_HasOperatorsInline>, !u64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[STRIDE]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.int<-2147483648> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[STRIDE]][1] {name = "u"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.int<0> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[STRIDE]][2] {name = "f"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.fp<-3.4{{.*}}E+38> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[STRIDE]][4] {name = "d"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.fp<-1.7{{.*}}E+308> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[STRIDE]][5] {name = "b"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #false
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ORIG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}): 
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB:.*]] = acc.get_lowerbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB]] : index to !u64i
// CHECK-NEXT: %[[UB:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB]] : index to !u64i
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["iter"] {alignment = 8 : i64}
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u64i
// CHECK-NEXT: %[[LAST_SUB_ONE:.*]] = cir.binop(sub, %[[UB_CAST]], %[[ONE]]) : !u64i
// CHECK-NEXT: cir.store %[[LAST_SUB_ONE]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(ge, %[[ITR_LOAD]], %[[LB_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride %[[DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!rec_HasOperatorsInline>, !u64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.call @_ZN18HasOperatorsInlineD1Ev(%[[STRIDE]]) nothrow : (!cir.ptr<!rec_HasOperatorsInline>) -> ()
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[DEC:.*]] = cir.unary(dec, %[[ITR_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[DEC]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  for(int i=0;i < 5; ++i);
#pragma acc parallel loop reduction(min:someVarArr[2])
// CHECK-NEXT: acc.reduction.recipe @reduction_min__Bcnt1__ZTSA5_18HasOperatorsInline : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> reduction_operator <min> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>{{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!rec_HasOperatorsInline x 5>, !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>, ["openacc.reduction.init"]
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB:.*]] = acc.get_lowerbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB]] : index to !u64i
// CHECK-NEXT: %[[UB:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB]] : index to !u64i
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["iter"] {alignment = 8 : i64}
// CHECK-NEXT: cir.store %[[LB_CAST]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[UB_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride %[[DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!rec_HasOperatorsInline>, !u64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[STRIDE]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.int<2147483647> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[STRIDE]][1] {name = "u"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.int<4294967295> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[STRIDE]][2] {name = "f"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.fp<3.4{{.*}}E+38> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[STRIDE]][4] {name = "d"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.fp<1.7{{.*}}E+308> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[STRIDE]][5] {name = "b"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ORIG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}): 
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB:.*]] = acc.get_lowerbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB]] : index to !u64i
// CHECK-NEXT: %[[UB:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB]] : index to !u64i
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["iter"] {alignment = 8 : i64}
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u64i
// CHECK-NEXT: %[[LAST_SUB_ONE:.*]] = cir.binop(sub, %[[UB_CAST]], %[[ONE]]) : !u64i
// CHECK-NEXT: cir.store %[[LAST_SUB_ONE]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(ge, %[[ITR_LOAD]], %[[LB_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride %[[DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!rec_HasOperatorsInline>, !u64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.call @_ZN18HasOperatorsInlineD1Ev(%[[STRIDE]]) nothrow : (!cir.ptr<!rec_HasOperatorsInline>) -> ()
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[DEC:.*]] = cir.unary(dec, %[[ITR_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[DEC]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  for(int i=0;i < 5; ++i);
#pragma acc parallel loop reduction(&:someVarArr[2])
// CHECK-NEXT: acc.reduction.recipe @reduction_iand__Bcnt1__ZTSA5_18HasOperatorsInline : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> reduction_operator <iand> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>{{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!rec_HasOperatorsInline x 5>, !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>, ["openacc.reduction.init"]
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB:.*]] = acc.get_lowerbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB]] : index to !u64i
// CHECK-NEXT: %[[UB:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB]] : index to !u64i
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["iter"] {alignment = 8 : i64}
// CHECK-NEXT: cir.store %[[LB_CAST]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[UB_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride %[[DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!rec_HasOperatorsInline>, !u64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[STRIDE]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #cir.int<-1> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[STRIDE]][1] {name = "u"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #cir.int<4294967295> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[STRIDE]][2] {name = "f"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #cir.fp<0xFF{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[STRIDE]][4] {name = "d"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #cir.fp<0xFF{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[STRIDE]][5] {name = "b"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ORIG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}): 
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB:.*]] = acc.get_lowerbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB]] : index to !u64i
// CHECK-NEXT: %[[UB:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB]] : index to !u64i
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["iter"] {alignment = 8 : i64}
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u64i
// CHECK-NEXT: %[[LAST_SUB_ONE:.*]] = cir.binop(sub, %[[UB_CAST]], %[[ONE]]) : !u64i
// CHECK-NEXT: cir.store %[[LAST_SUB_ONE]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(ge, %[[ITR_LOAD]], %[[LB_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride %[[DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!rec_HasOperatorsInline>, !u64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.call @_ZN18HasOperatorsInlineD1Ev(%[[STRIDE]]) nothrow : (!cir.ptr<!rec_HasOperatorsInline>) -> ()
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[DEC:.*]] = cir.unary(dec, %[[ITR_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[DEC]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  for(int i=0;i < 5; ++i);
#pragma acc parallel loop reduction(|:someVarArr[2])
// CHECK-NEXT: acc.reduction.recipe @reduction_ior__Bcnt1__ZTSA5_18HasOperatorsInline : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> reduction_operator <ior> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>{{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!rec_HasOperatorsInline x 5>, !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>, ["openacc.reduction.init"]
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB:.*]] = acc.get_lowerbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB]] : index to !u64i
// CHECK-NEXT: %[[UB:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB]] : index to !u64i
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["iter"] {alignment = 8 : i64}
// CHECK-NEXT: cir.store %[[LB_CAST]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[UB_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride %[[DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!rec_HasOperatorsInline>, !u64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[STRIDE]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[STRIDE]][1] {name = "u"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[STRIDE]][2] {name = "f"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[STRIDE]][4] {name = "d"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[STRIDE]][5] {name = "b"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #false
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ORIG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}): 
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB:.*]] = acc.get_lowerbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB]] : index to !u64i
// CHECK-NEXT: %[[UB:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB]] : index to !u64i
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["iter"] {alignment = 8 : i64}
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u64i
// CHECK-NEXT: %[[LAST_SUB_ONE:.*]] = cir.binop(sub, %[[UB_CAST]], %[[ONE]]) : !u64i
// CHECK-NEXT: cir.store %[[LAST_SUB_ONE]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(ge, %[[ITR_LOAD]], %[[LB_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride %[[DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!rec_HasOperatorsInline>, !u64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.call @_ZN18HasOperatorsInlineD1Ev(%[[STRIDE]]) nothrow : (!cir.ptr<!rec_HasOperatorsInline>) -> ()
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[DEC:.*]] = cir.unary(dec, %[[ITR_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[DEC]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  for(int i=0;i < 5; ++i);
#pragma acc parallel loop reduction(^:someVarArr[2])
// CHECK-NEXT: acc.reduction.recipe @reduction_xor__Bcnt1__ZTSA5_18HasOperatorsInline : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> reduction_operator <xor> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>{{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!rec_HasOperatorsInline x 5>, !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>, ["openacc.reduction.init"]
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB:.*]] = acc.get_lowerbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB]] : index to !u64i
// CHECK-NEXT: %[[UB:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB]] : index to !u64i
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["iter"] {alignment = 8 : i64}
// CHECK-NEXT: cir.store %[[LB_CAST]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[UB_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride %[[DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!rec_HasOperatorsInline>, !u64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[STRIDE]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[STRIDE]][1] {name = "u"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[STRIDE]][2] {name = "f"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[STRIDE]][4] {name = "d"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[STRIDE]][5] {name = "b"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #false
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ORIG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}): 
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB:.*]] = acc.get_lowerbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB]] : index to !u64i
// CHECK-NEXT: %[[UB:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB]] : index to !u64i
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["iter"] {alignment = 8 : i64}
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u64i
// CHECK-NEXT: %[[LAST_SUB_ONE:.*]] = cir.binop(sub, %[[UB_CAST]], %[[ONE]]) : !u64i
// CHECK-NEXT: cir.store %[[LAST_SUB_ONE]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(ge, %[[ITR_LOAD]], %[[LB_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride %[[DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!rec_HasOperatorsInline>, !u64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.call @_ZN18HasOperatorsInlineD1Ev(%[[STRIDE]]) nothrow : (!cir.ptr<!rec_HasOperatorsInline>) -> ()
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[DEC:.*]] = cir.unary(dec, %[[ITR_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[DEC]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  for(int i=0;i < 5; ++i);
#pragma acc parallel loop reduction(&&:someVarArr[2])
// CHECK-NEXT: acc.reduction.recipe @reduction_land__Bcnt1__ZTSA5_18HasOperatorsInline : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> reduction_operator <land> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>{{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!rec_HasOperatorsInline x 5>, !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>, ["openacc.reduction.init"]
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB:.*]] = acc.get_lowerbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB]] : index to !u64i
// CHECK-NEXT: %[[UB:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB]] : index to !u64i
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["iter"] {alignment = 8 : i64}
// CHECK-NEXT: cir.store %[[LB_CAST]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[UB_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride %[[DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!rec_HasOperatorsInline>, !u64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[STRIDE]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[STRIDE]][1] {name = "u"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[STRIDE]][2] {name = "f"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[STRIDE]][4] {name = "d"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[STRIDE]][5] {name = "b"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ORIG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}): 
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB:.*]] = acc.get_lowerbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB]] : index to !u64i
// CHECK-NEXT: %[[UB:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB]] : index to !u64i
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["iter"] {alignment = 8 : i64}
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u64i
// CHECK-NEXT: %[[LAST_SUB_ONE:.*]] = cir.binop(sub, %[[UB_CAST]], %[[ONE]]) : !u64i
// CHECK-NEXT: cir.store %[[LAST_SUB_ONE]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(ge, %[[ITR_LOAD]], %[[LB_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride %[[DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!rec_HasOperatorsInline>, !u64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.call @_ZN18HasOperatorsInlineD1Ev(%[[STRIDE]]) nothrow : (!cir.ptr<!rec_HasOperatorsInline>) -> ()
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[DEC:.*]] = cir.unary(dec, %[[ITR_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[DEC]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  for(int i=0;i < 5; ++i);
#pragma acc parallel loop reduction(||:someVarArr[2])
// CHECK-NEXT: acc.reduction.recipe @reduction_lor__Bcnt1__ZTSA5_18HasOperatorsInline : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> reduction_operator <lor> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>{{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!rec_HasOperatorsInline x 5>, !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>, ["openacc.reduction.init"]
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB:.*]] = acc.get_lowerbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB]] : index to !u64i
// CHECK-NEXT: %[[UB:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB]] : index to !u64i
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["iter"] {alignment = 8 : i64}
// CHECK-NEXT: cir.store %[[LB_CAST]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[UB_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride %[[DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!rec_HasOperatorsInline>, !u64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[STRIDE]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[STRIDE]][1] {name = "u"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[STRIDE]][2] {name = "f"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[STRIDE]][4] {name = "d"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[STRIDE]][5] {name = "b"} : !cir.ptr<!rec_HasOperatorsInline> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #false
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ORIG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}): 
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB:.*]] = acc.get_lowerbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB]] : index to !u64i
// CHECK-NEXT: %[[UB:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB]] : index to !u64i
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["iter"] {alignment = 8 : i64}
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u64i
// CHECK-NEXT: %[[LAST_SUB_ONE:.*]] = cir.binop(sub, %[[UB_CAST]], %[[ONE]]) : !u64i
// CHECK-NEXT: cir.store %[[LAST_SUB_ONE]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(ge, %[[ITR_LOAD]], %[[LB_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride %[[DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!rec_HasOperatorsInline>, !u64i) -> !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.call @_ZN18HasOperatorsInlineD1Ev(%[[STRIDE]]) nothrow : (!cir.ptr<!rec_HasOperatorsInline>) -> ()
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[DEC:.*]] = cir.unary(dec, %[[ITR_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[DEC]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  for(int i=0;i < 5; ++i);

#pragma acc parallel loop reduction(+:someVarArr[1:1])
  for(int i=0;i < 5; ++i);
#pragma acc parallel loop reduction(*:someVarArr[1:1])
  for(int i=0;i < 5; ++i);
#pragma acc parallel loop reduction(max:someVarArr[1:1])
  for(int i=0;i < 5; ++i);
#pragma acc parallel loop reduction(min:someVarArr[1:1])
  for(int i=0;i < 5; ++i);
#pragma acc parallel loop reduction(&:someVarArr[1:1])
  for(int i=0;i < 5; ++i);
#pragma acc parallel loop reduction(|:someVarArr[1:1])
  for(int i=0;i < 5; ++i);
#pragma acc parallel loop reduction(^:someVarArr[1:1])
  for(int i=0;i < 5; ++i);
#pragma acc parallel loop reduction(&&:someVarArr[1:1])
  for(int i=0;i < 5; ++i);
#pragma acc parallel loop reduction(||:someVarArr[1:1])
  for(int i=0;i < 5; ++i);
  // CHECK-NEXT: cir.func {{.*}}@_Z12acc_combined
}

void uses() {
  acc_combined<HasOperatorsInline>();
}
