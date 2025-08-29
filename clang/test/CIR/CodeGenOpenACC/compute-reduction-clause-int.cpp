// RUN: not %clang_cc1 -fopenacc -triple x86_64-linux-gnu -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir -triple x86_64-linux-pc %s -o - | FileCheck %s


// CHECK: acc.reduction.recipe @reduction_lor__ZTSA5_i : !cir.ptr<!cir.array<!s32i x 5>> reduction_operator <lor> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>>{{.*}})
// CHECK-NEXT: cir.alloca !cir.array<!s32i x 5>, !cir.ptr<!cir.array<!s32i x 5>>, ["openacc.reduction.init"]
// TODO OpenACC: Expecting an initialization to... SOME value here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!s32i x 5>>
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_land__ZTSA5_i : !cir.ptr<!cir.array<!s32i x 5>> reduction_operator <land> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>>{{.*}})
// CHECK-NEXT: cir.alloca !cir.array<!s32i x 5>, !cir.ptr<!cir.array<!s32i x 5>>, ["openacc.reduction.init"]
// TODO OpenACC: Expecting an initialization to... SOME value here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!s32i x 5>>
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_xor__ZTSA5_i : !cir.ptr<!cir.array<!s32i x 5>> reduction_operator <xor> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>>{{.*}})
// CHECK-NEXT: cir.alloca !cir.array<!s32i x 5>, !cir.ptr<!cir.array<!s32i x 5>>, ["openacc.reduction.init"]
// TODO OpenACC: Expecting an initialization to... SOME value here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!s32i x 5>>
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_ior__ZTSA5_i : !cir.ptr<!cir.array<!s32i x 5>> reduction_operator <ior> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>>{{.*}})
// CHECK-NEXT: cir.alloca !cir.array<!s32i x 5>, !cir.ptr<!cir.array<!s32i x 5>>, ["openacc.reduction.init"]
// TODO OpenACC: Expecting an initialization to... SOME value here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!s32i x 5>>
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_iand__ZTSA5_i : !cir.ptr<!cir.array<!s32i x 5>> reduction_operator <iand> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>>{{.*}})
// CHECK-NEXT: cir.alloca !cir.array<!s32i x 5>, !cir.ptr<!cir.array<!s32i x 5>>, ["openacc.reduction.init"]
// TODO OpenACC: Expecting an initialization to... SOME value here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!s32i x 5>>
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_min__ZTSA5_i : !cir.ptr<!cir.array<!s32i x 5>> reduction_operator <min> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>>{{.*}})
// CHECK-NEXT: cir.alloca !cir.array<!s32i x 5>, !cir.ptr<!cir.array<!s32i x 5>>, ["openacc.reduction.init"]
// TODO OpenACC: Expecting an initialization to... SOME value here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!s32i x 5>>
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_max__ZTSA5_i : !cir.ptr<!cir.array<!s32i x 5>> reduction_operator <max> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>>{{.*}})
// CHECK-NEXT: cir.alloca !cir.array<!s32i x 5>, !cir.ptr<!cir.array<!s32i x 5>>, ["openacc.reduction.init"]
// TODO OpenACC: Expecting an initialization to... SOME value here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!s32i x 5>>
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_mul__ZTSA5_i : !cir.ptr<!cir.array<!s32i x 5>> reduction_operator <mul> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>>{{.*}})
// CHECK-NEXT: cir.alloca !cir.array<!s32i x 5>, !cir.ptr<!cir.array<!s32i x 5>>, ["openacc.reduction.init"]
// TODO OpenACC: Expecting an initialization to... SOME value here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!s32i x 5>>
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_add__ZTSA5_i : !cir.ptr<!cir.array<!s32i x 5>> reduction_operator <add> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>>{{.*}})
// CHECK-NEXT: cir.alloca !cir.array<!s32i x 5>, !cir.ptr<!cir.array<!s32i x 5>>, ["openacc.reduction.init"]
// TODO OpenACC: Expecting an initialization to... SOME value here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!s32i x 5>>
// CHECK-NEXT: }


// CHECK-NEXT: acc.reduction.recipe @reduction_lor__ZTSi : !cir.ptr<!s32i> reduction_operator <lor> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!s32i>{{.*}})
// CHECK-NEXT: cir.alloca !s32i, !cir.ptr<!s32i>, ["openacc.reduction.init"]
// TODO OpenACC: Expecting an initialization to... SOME value here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!s32i> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!s32i> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!s32i>
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_land__ZTSi : !cir.ptr<!s32i> reduction_operator <land> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!s32i>{{.*}})
// CHECK-NEXT: cir.alloca !s32i, !cir.ptr<!s32i>, ["openacc.reduction.init"]
// TODO OpenACC: Expecting an initialization to... SOME value here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!s32i> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!s32i> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!s32i>
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_xor__ZTSi : !cir.ptr<!s32i> reduction_operator <xor> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!s32i>{{.*}})
// CHECK-NEXT: cir.alloca !s32i, !cir.ptr<!s32i>, ["openacc.reduction.init"]
// TODO OpenACC: Expecting an initialization to... SOME value here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!s32i> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!s32i> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!s32i>
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_ior__ZTSi : !cir.ptr<!s32i> reduction_operator <ior> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!s32i>{{.*}})
// CHECK-NEXT: cir.alloca !s32i, !cir.ptr<!s32i>, ["openacc.reduction.init"]
// TODO OpenACC: Expecting an initialization to... SOME value here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!s32i> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!s32i> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!s32i>
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_iand__ZTSi : !cir.ptr<!s32i> reduction_operator <iand> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!s32i>{{.*}})
// CHECK-NEXT: cir.alloca !s32i, !cir.ptr<!s32i>, ["openacc.reduction.init"]
// TODO OpenACC: Expecting an initialization to... SOME value here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!s32i> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!s32i> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!s32i>
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_min__ZTSi : !cir.ptr<!s32i> reduction_operator <min> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!s32i>{{.*}})
// CHECK-NEXT: cir.alloca !s32i, !cir.ptr<!s32i>, ["openacc.reduction.init"]
// TODO OpenACC: Expecting an initialization to... SOME value here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!s32i> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!s32i> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!s32i>
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_max__ZTSi : !cir.ptr<!s32i> reduction_operator <max> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!s32i>{{.*}})
// CHECK-NEXT: cir.alloca !s32i, !cir.ptr<!s32i>, ["openacc.reduction.init"]
// TODO OpenACC: Expecting an initialization to... SOME value here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!s32i> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!s32i> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!s32i>
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_mul__ZTSi : !cir.ptr<!s32i> reduction_operator <mul> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!s32i>{{.*}})
// CHECK-NEXT: cir.alloca !s32i, !cir.ptr<!s32i>, ["openacc.reduction.init"]
// TODO OpenACC: Expecting an initialization to... SOME value here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!s32i> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!s32i> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!s32i>
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_add__ZTSi : !cir.ptr<!s32i> reduction_operator <add> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!s32i>{{.*}})
// CHECK-NEXT: cir.alloca !s32i, !cir.ptr<!s32i>, ["openacc.reduction.init"]
// TODO OpenACC: Expecting an initialization to... SOME value here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!s32i> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!s32i> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!s32i>
// CHECK-NEXT: }

template<typename T>
void acc_compute() {
  T someVar;
  T someVarArr[5];
#pragma acc parallel reduction(+:someVar)
  ;
#pragma acc parallel reduction(*:someVar)
  ;
#pragma acc parallel reduction(max:someVar)
  ;
#pragma acc parallel reduction(min:someVar)
  ;
#pragma acc parallel reduction(&:someVar)
  ;
#pragma acc parallel reduction(|:someVar)
  ;
#pragma acc parallel reduction(^:someVar)
  ;
#pragma acc parallel reduction(&&:someVar)
  ;
#pragma acc parallel reduction(||:someVar)
  ;

#pragma acc parallel reduction(+:someVarArr)
  ;
#pragma acc parallel reduction(*:someVarArr)
  ;
#pragma acc parallel reduction(max:someVarArr)
  ;
#pragma acc parallel reduction(min:someVarArr)
  ;
#pragma acc parallel reduction(&:someVarArr)
  ;
#pragma acc parallel reduction(|:someVarArr)
  ;
#pragma acc parallel reduction(^:someVarArr)
  ;
#pragma acc parallel reduction(&&:someVarArr)
  ;
#pragma acc parallel reduction(||:someVarArr)
  ;

#pragma acc parallel reduction(+:someVarArr[2])
  ;
#pragma acc parallel reduction(*:someVarArr[2])
  ;
#pragma acc parallel reduction(max:someVarArr[2])
  ;
#pragma acc parallel reduction(min:someVarArr[2])
  ;
#pragma acc parallel reduction(&:someVarArr[2])
  ;
#pragma acc parallel reduction(|:someVarArr[2])
  ;
#pragma acc parallel reduction(^:someVarArr[2])
  ;
#pragma acc parallel reduction(&&:someVarArr[2])
  ;
#pragma acc parallel reduction(||:someVarArr[2])
  ;

#pragma acc parallel reduction(+:someVarArr[1:1])
  ;
#pragma acc parallel reduction(*:someVarArr[1:1])
  ;
#pragma acc parallel reduction(max:someVarArr[1:1])
  ;
#pragma acc parallel reduction(min:someVarArr[1:1])
  ;
#pragma acc parallel reduction(&:someVarArr[1:1])
  ;
#pragma acc parallel reduction(|:someVarArr[1:1])
  ;
#pragma acc parallel reduction(^:someVarArr[1:1])
  ;
#pragma acc parallel reduction(&&:someVarArr[1:1])
  ;
#pragma acc parallel reduction(||:someVarArr[1:1])
  ;
}

void uses() {
  acc_compute<int>();
}
