// RUN: %clang_cc1 -fopenacc -triple x86_64-linux-gnu -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir -triple x86_64-linux-pc %s -o - | FileCheck %s

struct NoCopyConstruct {};

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
// CHECK-NEXT: %[[FROM_LOAD:.*]] = cir.load {{.*}}%[[ARG_FROM]] : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT: cir.store {{.*}} %[[FROM_LOAD]], %[[ARG_TO]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
//
// CHECK-NEXT: acc.firstprivate.recipe @firstprivatization__ZTSf : !cir.ptr<!cir.float> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.float> {{.*}}):
// CHECK-NEXT: cir.alloca !cir.float, !cir.ptr<!cir.float>, ["openacc.firstprivate.init"]
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } copy {
// CHECK-NEXT: ^bb0(%[[ARG_FROM:.*]]: !cir.ptr<!cir.float> {{.*}}, %[[ARG_TO:.*]]: !cir.ptr<!cir.float> {{.*}}):
// CHECK-NEXT: %[[FROM_LOAD:.*]] = cir.load {{.*}}%[[ARG_FROM]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[FROM_LOAD]], %[[ARG_TO]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
//
// CHECK-NEXT: acc.firstprivate.recipe @firstprivatization__ZTS15NoCopyConstruct : !cir.ptr<!rec_NoCopyConstruct> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_NoCopyConstruct> {{.*}}):
// CHECK-NEXT: cir.alloca !rec_NoCopyConstruct, !cir.ptr<!rec_NoCopyConstruct>, ["openacc.firstprivate.init"]
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } copy {
// CHECK-NEXT: ^bb0(%[[ARG_FROM:.*]]: !cir.ptr<!rec_NoCopyConstruct> {{.*}}, %[[ARG_TO:.*]]: !cir.ptr<!rec_NoCopyConstruct> {{.*}}):
// CHECK-NEXT: cir.call @_ZN15NoCopyConstructC1ERKS_(%[[ARG_TO]], %[[ARG_FROM]]) nothrow : (!cir.ptr<!rec_NoCopyConstruct>, !cir.ptr<!rec_NoCopyConstruct>) -> ()
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
//
// CHECK-NEXT: acc.firstprivate.recipe @firstprivatization__ZTS13CopyConstruct : !cir.ptr<!rec_CopyConstruct> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_CopyConstruct> {{.*}}):
// CHECK-NEXT: cir.alloca !rec_CopyConstruct, !cir.ptr<!rec_CopyConstruct>, ["openacc.firstprivate.init"]
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } copy {
// CHECK-NEXT: ^bb0(%[[ARG_FROM:.*]]: !cir.ptr<!rec_CopyConstruct> {{.*}}, %[[ARG_TO:.*]]: !cir.ptr<!rec_CopyConstruct> {{.*}}):
// CHECK-NEXT: cir.call @_ZN13CopyConstructC1ERKS_(%[[ARG_TO]], %[[ARG_FROM]]) : (!cir.ptr<!rec_CopyConstruct>, !cir.ptr<!rec_CopyConstruct>) -> ()
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
// CHECK-NEXT: acc.firstprivate.recipe @firstprivatization__Bcnt1__ZTSA5_i : !cir.ptr<!cir.array<!s32i x 5>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}):
// CHECK-NEXT: cir.alloca !cir.array<!s32i x 5>, !cir.ptr<!cir.array<!s32i x 5>>, ["openacc.firstprivate.init"]
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } copy {
// CHECK-NEXT: ^bb0(%[[ARG_FROM:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}}, %[[ARG_TO:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}):
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
// CHECK-NEXT: %[[DECAY_FROM:.*]] = cir.cast array_to_ptrdecay %[[ARG_FROM]] : !cir.ptr<!cir.array<!s32i x 5>> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[STRIDE_FROM:.*]] = cir.ptr_stride %[[DECAY_FROM]], %[[ITR_LOAD]] : (!cir.ptr<!s32i>, !u64i) -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[DECAY_TO:.*]] = cir.cast array_to_ptrdecay %[[ARG_TO]] : !cir.ptr<!cir.array<!s32i x 5>> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[STRIDE_TO:.*]] = cir.ptr_stride %[[DECAY_TO]], %[[ITR_LOAD]] : (!cir.ptr<!s32i>, !u64i) -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[FROM_LOAD:.*]] = cir.load{{.*}} %[[STRIDE_FROM]] : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT: cir.store{{.*}} %[[FROM_LOAD]], %[[STRIDE_TO]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
//
// CHECK-NEXT: acc.firstprivate.recipe @firstprivatization__Bcnt1__ZTSA5_f : !cir.ptr<!cir.array<!cir.float x 5>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}):
// CHECK-NEXT: cir.alloca !cir.array<!cir.float x 5>, !cir.ptr<!cir.array<!cir.float x 5>>, ["openacc.firstprivate.init"]
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } copy {
// CHECK-NEXT: ^bb0(%[[ARG_FROM:.*]]: !cir.ptr<!cir.array<!cir.float x 5>> {{.*}}, %[[ARG_TO:.*]]: !cir.ptr<!cir.array<!cir.float x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}):
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
// CHECK-NEXT: %[[DECAY_FROM:.*]] = cir.cast array_to_ptrdecay %[[ARG_FROM]] : !cir.ptr<!cir.array<!cir.float x 5>> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[STRIDE_FROM:.*]] = cir.ptr_stride %[[DECAY_FROM]], %[[ITR_LOAD]] : (!cir.ptr<!cir.float>, !u64i) -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[DECAY_TO:.*]] = cir.cast array_to_ptrdecay %[[ARG_TO]] : !cir.ptr<!cir.array<!cir.float x 5>> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[STRIDE_TO:.*]] = cir.ptr_stride %[[DECAY_TO]], %[[ITR_LOAD]] : (!cir.ptr<!cir.float>, !u64i) -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[FROM_LOAD:.*]] = cir.load{{.*}} %[[STRIDE_FROM]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: cir.store{{.*}} %[[FROM_LOAD]], %[[STRIDE_TO]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
//
// CHECK-NEXT: acc.firstprivate.recipe @firstprivatization__Bcnt1__ZTSA5_15NoCopyConstruct : !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}):
// CHECK-NEXT: cir.alloca !cir.array<!rec_NoCopyConstruct x 5>, !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>>, ["openacc.firstprivate.init"]
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } copy {
// CHECK-NEXT: ^bb0(%[[ARG_FROM:.*]]: !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>> {{.*}}, %[[ARG_TO:.*]]: !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}):
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
// CHECK-NEXT: %[[DECAY_FROM:.*]] = cir.cast array_to_ptrdecay %[[ARG_FROM]] : !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>> -> !cir.ptr<!rec_NoCopyConstruct>
// CHECK-NEXT: %[[STRIDE_FROM:.*]] = cir.ptr_stride %[[DECAY_FROM]], %[[ITR_LOAD]] : (!cir.ptr<!rec_NoCopyConstruct>, !u64i) -> !cir.ptr<!rec_NoCopyConstruct>
// CHECK-NEXT: %[[DECAY_TO:.*]] = cir.cast array_to_ptrdecay %[[ARG_TO]] : !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>> -> !cir.ptr<!rec_NoCopyConstruct>
// CHECK-NEXT: %[[STRIDE_TO:.*]] = cir.ptr_stride %[[DECAY_TO]], %[[ITR_LOAD]] : (!cir.ptr<!rec_NoCopyConstruct>, !u64i) -> !cir.ptr<!rec_NoCopyConstruct>
// CHECK-NEXT: cir.call @_ZN15NoCopyConstructC1ERKS_(%[[STRIDE_TO]], %[[STRIDE_FROM]]) nothrow : (!cir.ptr<!rec_NoCopyConstruct>, !cir.ptr<!rec_NoCopyConstruct>) -> ()
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
//
// CHECK-NEXT: acc.firstprivate.recipe @firstprivatization__Bcnt1__ZTSA5_13CopyConstruct : !cir.ptr<!cir.array<!rec_CopyConstruct x 5>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_CopyConstruct x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}):
// CHECK-NEXT: cir.alloca !cir.array<!rec_CopyConstruct x 5>, !cir.ptr<!cir.array<!rec_CopyConstruct x 5>>, ["openacc.firstprivate.init"]
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } copy {
// CHECK-NEXT: ^bb0(%[[ARG_FROM:.*]]: !cir.ptr<!cir.array<!rec_CopyConstruct x 5>> {{.*}}, %[[ARG_TO:.*]]: !cir.ptr<!cir.array<!rec_CopyConstruct x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}):
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
// CHECK-NEXT: %[[DECAY_FROM:.*]] = cir.cast array_to_ptrdecay %[[ARG_FROM]] : !cir.ptr<!cir.array<!rec_CopyConstruct x 5>> -> !cir.ptr<!rec_CopyConstruct>
// CHECK-NEXT: %[[STRIDE_FROM:.*]] = cir.ptr_stride %[[DECAY_FROM]], %[[ITR_LOAD]] : (!cir.ptr<!rec_CopyConstruct>, !u64i) -> !cir.ptr<!rec_CopyConstruct>
// CHECK-NEXT: %[[DECAY_TO:.*]] = cir.cast array_to_ptrdecay %[[ARG_TO]] : !cir.ptr<!cir.array<!rec_CopyConstruct x 5>> -> !cir.ptr<!rec_CopyConstruct>
// CHECK-NEXT: %[[STRIDE_TO:.*]] = cir.ptr_stride %[[DECAY_TO]], %[[ITR_LOAD]] : (!cir.ptr<!rec_CopyConstruct>, !u64i) -> !cir.ptr<!rec_CopyConstruct>
// CHECK-NEXT: cir.call @_ZN13CopyConstructC1ERKS_(%[[STRIDE_TO]], %[[STRIDE_FROM]]) : (!cir.ptr<!rec_CopyConstruct>, !cir.ptr<!rec_CopyConstruct>) -> ()
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
//
// CHECK-NEXT: acc.firstprivate.recipe @firstprivatization__Bcnt1__ZTSA5_14NonDefaultCtor : !cir.ptr<!cir.array<!rec_NonDefaultCtor x 5>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_NonDefaultCtor x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}):
// CHECK-NEXT: cir.alloca !cir.array<!rec_NonDefaultCtor x 5>, !cir.ptr<!cir.array<!rec_NonDefaultCtor x 5>>, ["openacc.firstprivate.init"]
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } copy {
// CHECK-NEXT: ^bb0(%[[ARG_FROM:.*]]: !cir.ptr<!cir.array<!rec_NonDefaultCtor x 5>> {{.*}}, %[[ARG_TO:.*]]: !cir.ptr<!cir.array<!rec_NonDefaultCtor x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}):
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
// CHECK-NEXT: %[[DECAY_FROM:.*]] = cir.cast array_to_ptrdecay %[[ARG_FROM]] : !cir.ptr<!cir.array<!rec_NonDefaultCtor x 5>> -> !cir.ptr<!rec_NonDefaultCtor>
// CHECK-NEXT: %[[STRIDE_FROM:.*]] = cir.ptr_stride %[[DECAY_FROM]], %[[ITR_LOAD]] : (!cir.ptr<!rec_NonDefaultCtor>, !u64i) -> !cir.ptr<!rec_NonDefaultCtor>
// CHECK-NEXT: %[[DECAY_TO:.*]] = cir.cast array_to_ptrdecay %[[ARG_TO]] : !cir.ptr<!cir.array<!rec_NonDefaultCtor x 5>> -> !cir.ptr<!rec_NonDefaultCtor>
// CHECK-NEXT: %[[STRIDE_TO:.*]] = cir.ptr_stride %[[DECAY_TO]], %[[ITR_LOAD]] : (!cir.ptr<!rec_NonDefaultCtor>, !u64i) -> !cir.ptr<!rec_NonDefaultCtor>
// CHECK-NEXT: cir.call @_ZN14NonDefaultCtorC1ERKS_(%[[STRIDE_TO]], %[[STRIDE_FROM]]) nothrow : (!cir.ptr<!rec_NonDefaultCtor>, !cir.ptr<!rec_NonDefaultCtor>) -> ()
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
//
// CHECK-NEXT: acc.firstprivate.recipe @firstprivatization__Bcnt1__ZTSA5_7HasDtor : !cir.ptr<!cir.array<!rec_HasDtor x 5>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasDtor x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}):
// CHECK-NEXT: cir.alloca !cir.array<!rec_HasDtor x 5>, !cir.ptr<!cir.array<!rec_HasDtor x 5>>, ["openacc.firstprivate.init"]
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } copy {
// CHECK-NEXT: ^bb0(%[[ARG_FROM:.*]]: !cir.ptr<!cir.array<!rec_HasDtor x 5>> {{.*}}, %[[ARG_TO:.*]]: !cir.ptr<!cir.array<!rec_HasDtor x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}):
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
// CHECK-NEXT: %[[DECAY_FROM:.*]] = cir.cast array_to_ptrdecay %[[ARG_FROM]] : !cir.ptr<!cir.array<!rec_HasDtor x 5>> -> !cir.ptr<!rec_HasDtor>
// CHECK-NEXT: %[[STRIDE_FROM:.*]] = cir.ptr_stride %[[DECAY_FROM]], %[[ITR_LOAD]] : (!cir.ptr<!rec_HasDtor>, !u64i) -> !cir.ptr<!rec_HasDtor>
// CHECK-NEXT: %[[DECAY_TO:.*]] = cir.cast array_to_ptrdecay %[[ARG_TO]] : !cir.ptr<!cir.array<!rec_HasDtor x 5>> -> !cir.ptr<!rec_HasDtor>
// CHECK-NEXT: %[[STRIDE_TO:.*]] = cir.ptr_stride %[[DECAY_TO]], %[[ITR_LOAD]] : (!cir.ptr<!rec_HasDtor>, !u64i) -> !cir.ptr<!rec_HasDtor>
// CHECK-NEXT: cir.call @_ZN7HasDtorC1ERKS_(%[[STRIDE_TO]], %[[STRIDE_FROM]]) nothrow : (!cir.ptr<!rec_HasDtor>, !cir.ptr<!rec_HasDtor>) -> ()
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ORIG:.*]]: !cir.ptr<!cir.array<!rec_HasDtor x 5>> {{.*}}, %[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasDtor x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}):
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
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ARG]] : !cir.ptr<!cir.array<!rec_HasDtor x 5>> -> !cir.ptr<!rec_HasDtor>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride %[[DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!rec_HasDtor>, !u64i) -> !cir.ptr<!rec_HasDtor>
// CHECK-NEXT: cir.call @_ZN7HasDtorD1Ev(%[[STRIDE]]) nothrow : (!cir.ptr<!rec_HasDtor>) -> ()
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
//

extern "C" void acc_compute() {
  // CHECK: cir.func{{.*}} @acc_compute() {

  int someInt;
  // CHECK-NEXT: %[[SOMEINT:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["someInt"]
  float someFloat;
  // CHECK-NEXT: %[[SOMEFLOAT:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["someFloat"]
  NoCopyConstruct noCopy;
  // CHECK-NEXT: %[[NOCOPY:.*]] = cir.alloca !rec_NoCopyConstruct, !cir.ptr<!rec_NoCopyConstruct>, ["noCopy"]
  CopyConstruct hasCopy;
  // CHECK-NEXT: %[[HASCOPY:.*]] = cir.alloca !rec_CopyConstruct, !cir.ptr<!rec_CopyConstruct>, ["hasCopy"]
  NonDefaultCtor notDefCtor;
  // CHECK-NEXT: %[[NOTDEFCTOR:.*]] = cir.alloca !rec_NonDefaultCtor, !cir.ptr<!rec_NonDefaultCtor>, ["notDefCtor", init]
  HasDtor dtor;
  // CHECK-NEXT: %[[DTOR:.*]] = cir.alloca !rec_HasDtor, !cir.ptr<!rec_HasDtor>, ["dtor"]
  int someIntArr[5];
  // CHECK-NEXT: %[[INTARR:.*]] = cir.alloca !cir.array<!s32i x 5>, !cir.ptr<!cir.array<!s32i x 5>>, ["someIntArr"]
  float someFloatArr[5];
  // CHECK-NEXT: %[[FLOATARR:.*]] = cir.alloca !cir.array<!cir.float x 5>, !cir.ptr<!cir.array<!cir.float x 5>>, ["someFloatArr"]
  NoCopyConstruct noCopyArr[5];
  // CHECK-NEXT: %[[NOCOPYARR:.*]] = cir.alloca !cir.array<!rec_NoCopyConstruct x 5>, !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>>, ["noCopyArr"]
  CopyConstruct hasCopyArr[5];
  // CHECK-NEXT: %[[HASCOPYARR:.*]] = cir.alloca !cir.array<!rec_CopyConstruct x 5>, !cir.ptr<!cir.array<!rec_CopyConstruct x 5>>, ["hasCopyArr"]
  NonDefaultCtor notDefCtorArr[5];
  // CHECK-NEXT: %[[NOTDEFCTORARR:.*]] = cir.alloca !cir.array<!rec_NonDefaultCtor x 5>, !cir.ptr<!cir.array<!rec_NonDefaultCtor x 5>>, ["notDefCtorArr", init]
  HasDtor dtorArr[5];
  // CHECK-NEXT: %[[DTORARR:.*]] = cir.alloca !cir.array<!rec_HasDtor x 5>, !cir.ptr<!cir.array<!rec_HasDtor x 5>>, ["dtorArr"]
  // CHECK-NEXT: cir.call @_ZN14NonDefaultCtorC1Ev(%[[NOTDEFCTOR]]) : (!cir.ptr<!rec_NonDefaultCtor>) -> ()

#pragma acc parallel firstprivate(someInt)
  ;
  // CHECK: %[[PRIVATE:.*]] = acc.firstprivate varPtr(%[[SOMEINT]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {name = "someInt"}
  // CHECK-NEXT: acc.parallel firstprivate(@firstprivatization__ZTSi -> %[[PRIVATE]] : !cir.ptr<!s32i>)
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
#pragma acc serial firstprivate(someFloat)
  ;
  // CHECK-NEXT: %[[PRIVATE:.*]] = acc.firstprivate varPtr(%[[SOMEFLOAT]] : !cir.ptr<!cir.float>) -> !cir.ptr<!cir.float> {name = "someFloat"}
  // CHECK-NEXT: acc.serial firstprivate(@firstprivatization__ZTSf -> %[[PRIVATE]] : !cir.ptr<!cir.float>)
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc parallel firstprivate(noCopy)
  ;
  // CHECK-NEXT: %[[PRIVATE:.*]] = acc.firstprivate varPtr(%[[NOCOPY]] : !cir.ptr<!rec_NoCopyConstruct>) -> !cir.ptr<!rec_NoCopyConstruct> {name = "noCopy"}
  // CHECK-NEXT: acc.parallel firstprivate(@firstprivatization__ZTS15NoCopyConstruct -> %[[PRIVATE]] : !cir.ptr<!rec_NoCopyConstruct>
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
#pragma acc serial firstprivate(hasCopy)
  ;
  // CHECK-NEXT: %[[PRIVATE:.*]] = acc.firstprivate varPtr(%[[HASCOPY]] : !cir.ptr<!rec_CopyConstruct>) -> !cir.ptr<!rec_CopyConstruct> {name = "hasCopy"}
  // CHECK-NEXT: acc.serial firstprivate(@firstprivatization__ZTS13CopyConstruct -> %[[PRIVATE]] : !cir.ptr<!rec_CopyConstruct>)
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
#pragma acc serial firstprivate(notDefCtor)
  ;
  // CHECK-NEXT: %[[PRIVATE:.*]] = acc.firstprivate varPtr(%[[NOTDEFCTOR]] : !cir.ptr<!rec_NonDefaultCtor>) -> !cir.ptr<!rec_NonDefaultCtor> {name = "notDefCtor"}
  // CHECK-NEXT: acc.serial firstprivate(@firstprivatization__ZTS14NonDefaultCtor -> %[[PRIVATE]] : !cir.ptr<!rec_NonDefaultCtor>)
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
#pragma acc serial firstprivate(dtor)
  ;
  // CHECK-NEXT: %[[PRIVATE:.*]] = acc.firstprivate varPtr(%[[DTOR]] : !cir.ptr<!rec_HasDtor>) -> !cir.ptr<!rec_HasDtor> {name = "dtor"}
  // CHECK-NEXT: acc.serial firstprivate(@firstprivatization__ZTS7HasDtor -> %[[PRIVATE]] : !cir.ptr<!rec_HasDtor>)
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc parallel firstprivate(someInt, someFloat, noCopy, hasCopy, notDefCtor, dtor)
  ;
  // CHECK: %[[PRIVATE1:.*]] = acc.firstprivate varPtr(%[[SOMEINT]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {name = "someInt"}
  // CHECK-NEXT: %[[PRIVATE2:.*]] = acc.firstprivate varPtr(%[[SOMEFLOAT]] : !cir.ptr<!cir.float>) -> !cir.ptr<!cir.float> {name = "someFloat"}
  // CHECK-NEXT: %[[PRIVATE3:.*]] = acc.firstprivate varPtr(%[[NOCOPY]] : !cir.ptr<!rec_NoCopyConstruct>) -> !cir.ptr<!rec_NoCopyConstruct> {name = "noCopy"}
  // CHECK-NEXT: %[[PRIVATE4:.*]] = acc.firstprivate varPtr(%[[HASCOPY]] : !cir.ptr<!rec_CopyConstruct>) -> !cir.ptr<!rec_CopyConstruct> {name = "hasCopy"}
  // CHECK-NEXT: %[[PRIVATE5:.*]] = acc.firstprivate varPtr(%[[NOTDEFCTOR]] : !cir.ptr<!rec_NonDefaultCtor>) -> !cir.ptr<!rec_NonDefaultCtor> {name = "notDefCtor"}
  // CHECK-NEXT: %[[PRIVATE6:.*]] = acc.firstprivate varPtr(%[[DTOR]] : !cir.ptr<!rec_HasDtor>) -> !cir.ptr<!rec_HasDtor> {name = "dtor"}
  // CHECK-NEXT: acc.parallel firstprivate(@firstprivatization__ZTSi -> %[[PRIVATE1]] : !cir.ptr<!s32i>,
  // CHECK-SAME: @firstprivatization__ZTSf -> %[[PRIVATE2]] : !cir.ptr<!cir.float>,
  // CHECK-SAME: @firstprivatization__ZTS15NoCopyConstruct -> %[[PRIVATE3]] : !cir.ptr<!rec_NoCopyConstruct>,
  // CHECK-SAME: @firstprivatization__ZTS13CopyConstruct -> %[[PRIVATE4]] : !cir.ptr<!rec_CopyConstruct>,
  // CHECK-SAME: @firstprivatization__ZTS14NonDefaultCtor -> %[[PRIVATE5]] : !cir.ptr<!rec_NonDefaultCtor>,
  // CHECK-SAME: @firstprivatization__ZTS7HasDtor -> %[[PRIVATE6]] : !cir.ptr<!rec_HasDtor>)
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc serial firstprivate(someIntArr[1])
  ;
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[PRIVATE:.*]] = acc.firstprivate varPtr(%[[INTARR]] : !cir.ptr<!cir.array<!s32i x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!s32i x 5>> {name = "someIntArr[1]"}
  // CHECK-NEXT: acc.serial firstprivate(@firstprivatization__Bcnt1__ZTSA5_i -> %[[PRIVATE]] : !cir.ptr<!cir.array<!s32i x 5>>)
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
#pragma acc parallel firstprivate(someFloatArr[1])
  ;
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[PRIVATE:.*]] = acc.firstprivate varPtr(%[[FLOATARR]] : !cir.ptr<!cir.array<!cir.float x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!cir.float x 5>> {name = "someFloatArr[1]"}
  // CHECK-NEXT: acc.parallel firstprivate(@firstprivatization__Bcnt1__ZTSA5_f -> %[[PRIVATE]] : !cir.ptr<!cir.array<!cir.float x 5>>)
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
#pragma acc serial firstprivate(noCopyArr[1])
  ;
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[PRIVATE:.*]] = acc.firstprivate varPtr(%[[NOCOPYARR]] : !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>> {name = "noCopyArr[1]"}
  // CHECK-NEXT: acc.serial firstprivate(@firstprivatization__Bcnt1__ZTSA5_15NoCopyConstruct -> %[[PRIVATE]] : !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>>)
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
#pragma acc parallel firstprivate(hasCopyArr[1])
  ;
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[PRIVATE:.*]] = acc.firstprivate varPtr(%[[HASCOPYARR]] : !cir.ptr<!cir.array<!rec_CopyConstruct x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!rec_CopyConstruct x 5>> {name = "hasCopyArr[1]"}
  // CHECK-NEXT: acc.parallel firstprivate(@firstprivatization__Bcnt1__ZTSA5_13CopyConstruct -> %[[PRIVATE]] : !cir.ptr<!cir.array<!rec_CopyConstruct x 5>>)
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
#pragma acc parallel firstprivate(notDefCtorArr[1])
  ;
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[PRIVATE:.*]] = acc.firstprivate varPtr(%[[NOTDEFCTORARR]] : !cir.ptr<!cir.array<!rec_NonDefaultCtor x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!rec_NonDefaultCtor x 5>> {name = "notDefCtorArr[1]"}
  // CHECK-NEXT: acc.parallel firstprivate(@firstprivatization__Bcnt1__ZTSA5_14NonDefaultCtor -> %[[PRIVATE]] : !cir.ptr<!cir.array<!rec_NonDefaultCtor x 5>>)
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
#pragma acc parallel firstprivate(dtorArr[1])
  ;
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[PRIVATE:.*]] = acc.firstprivate varPtr(%[[DTORARR]] : !cir.ptr<!cir.array<!rec_HasDtor x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!rec_HasDtor x 5>> {name = "dtorArr[1]"}
  // CHECK-NEXT: acc.parallel firstprivate(@firstprivatization__Bcnt1__ZTSA5_7HasDtor -> %[[PRIVATE]] : !cir.ptr<!cir.array<!rec_HasDtor x 5>>)
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
#pragma acc serial firstprivate(someIntArr[1], someFloatArr[1], noCopyArr[1], hasCopyArr[1], notDefCtorArr[1], dtorArr[1])
  ;
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[PRIVATE1:.*]] = acc.firstprivate varPtr(%[[INTARR]] : !cir.ptr<!cir.array<!s32i x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!s32i x 5>> {name = "someIntArr[1]"}
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[PRIVATE2:.*]] = acc.firstprivate varPtr(%[[FLOATARR]] : !cir.ptr<!cir.array<!cir.float x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!cir.float x 5>> {name = "someFloatArr[1]"}
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[PRIVATE3:.*]] = acc.firstprivate varPtr(%[[NOCOPYARR]] : !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>> {name = "noCopyArr[1]"}
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[PRIVATE4:.*]] = acc.firstprivate varPtr(%[[HASCOPYARR]] : !cir.ptr<!cir.array<!rec_CopyConstruct x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!rec_CopyConstruct x 5>> {name = "hasCopyArr[1]"}
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[PRIVATE5:.*]] = acc.firstprivate varPtr(%[[NOTDEFCTORARR]] : !cir.ptr<!cir.array<!rec_NonDefaultCtor x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!rec_NonDefaultCtor x 5>> {name = "notDefCtorArr[1]"}
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[PRIVATE6:.*]] = acc.firstprivate varPtr(%[[DTORARR]] : !cir.ptr<!cir.array<!rec_HasDtor x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!rec_HasDtor x 5>> {name = "dtorArr[1]"}
  // CHECK-NEXT: acc.serial firstprivate(@firstprivatization__Bcnt1__ZTSA5_i -> %[[PRIVATE1]] : !cir.ptr<!cir.array<!s32i x 5>>,
  // CHECK-SAME: @firstprivatization__Bcnt1__ZTSA5_f -> %[[PRIVATE2]] : !cir.ptr<!cir.array<!cir.float x 5>>,
  // CHECK-SAME: @firstprivatization__Bcnt1__ZTSA5_15NoCopyConstruct -> %[[PRIVATE3]] : !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>>,
  // CHECK-SAME: @firstprivatization__Bcnt1__ZTSA5_13CopyConstruct -> %[[PRIVATE4]] : !cir.ptr<!cir.array<!rec_CopyConstruct x 5>>,
  // CHECK-SAME: @firstprivatization__Bcnt1__ZTSA5_14NonDefaultCtor -> %[[PRIVATE5]] : !cir.ptr<!cir.array<!rec_NonDefaultCtor x 5>>,
  // CHECK-SAME: @firstprivatization__Bcnt1__ZTSA5_7HasDtor -> %[[PRIVATE6]] : !cir.ptr<!cir.array<!rec_HasDtor x 5>>)
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc parallel firstprivate(someIntArr[1:1])
  ;
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST2:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CAST2]] : si32) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[PRIVATE:.*]] = acc.firstprivate varPtr(%[[INTARR]] : !cir.ptr<!cir.array<!s32i x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!s32i x 5>> {name = "someIntArr[1:1]"}
  // CHECK-NEXT: acc.parallel firstprivate(@firstprivatization__Bcnt1__ZTSA5_i -> %[[PRIVATE]] : !cir.ptr<!cir.array<!s32i x 5>>)
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
#pragma acc serial firstprivate(someFloatArr[1:1])
  ;
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST2:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CAST2]] : si32) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[PRIVATE:.*]] = acc.firstprivate varPtr(%[[FLOATARR]] : !cir.ptr<!cir.array<!cir.float x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!cir.float x 5>> {name = "someFloatArr[1:1]"}
  // CHECK-NEXT: acc.serial firstprivate(@firstprivatization__Bcnt1__ZTSA5_f -> %[[PRIVATE]] : !cir.ptr<!cir.array<!cir.float x 5>>)
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
#pragma acc parallel firstprivate(noCopyArr[1:1])
  ;
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST2:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CAST2]] : si32) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[PRIVATE:.*]] = acc.firstprivate varPtr(%[[NOCOPYARR]] : !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>> {name = "noCopyArr[1:1]"}
  // CHECK-NEXT: acc.parallel firstprivate(@firstprivatization__Bcnt1__ZTSA5_15NoCopyConstruct -> %[[PRIVATE]] : !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>>)
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
#pragma acc serial firstprivate(hasCopyArr[1:1])
  ;
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST2:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CAST2]] : si32) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[PRIVATE:.*]] = acc.firstprivate varPtr(%[[HASCOPYARR]] : !cir.ptr<!cir.array<!rec_CopyConstruct x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!rec_CopyConstruct x 5>> {name = "hasCopyArr[1:1]"}
  // CHECK-NEXT: acc.serial firstprivate(@firstprivatization__Bcnt1__ZTSA5_13CopyConstruct -> %[[PRIVATE]] : !cir.ptr<!cir.array<!rec_CopyConstruct x 5>>)
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
#pragma acc parallel firstprivate(notDefCtorArr[1:1])
  ;
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST2:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CAST2]] : si32) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[PRIVATE:.*]] = acc.firstprivate varPtr(%[[NOTDEFCTORARR]] : !cir.ptr<!cir.array<!rec_NonDefaultCtor x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!rec_NonDefaultCtor x 5>> {name = "notDefCtorArr[1:1]"}
  // CHECK-NEXT: acc.parallel firstprivate(@firstprivatization__Bcnt1__ZTSA5_14NonDefaultCtor -> %[[PRIVATE]] : !cir.ptr<!cir.array<!rec_NonDefaultCtor x 5>>)
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
#pragma acc parallel firstprivate(dtorArr[1:1])
  ;
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST2:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CAST2]] : si32) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[PRIVATE:.*]] = acc.firstprivate varPtr(%[[DTORARR]] : !cir.ptr<!cir.array<!rec_HasDtor x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!rec_HasDtor x 5>> {name = "dtorArr[1:1]"}
  // CHECK-NEXT: acc.parallel firstprivate(@firstprivatization__Bcnt1__ZTSA5_7HasDtor -> %[[PRIVATE]] : !cir.ptr<!cir.array<!rec_HasDtor x 5>>)
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
#pragma acc parallel firstprivate(someIntArr[1:1], someFloatArr[1:1], noCopyArr[1:1], hasCopyArr[1:1], notDefCtorArr[1:1], dtorArr[1:1])
  ;
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST2:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CAST2]] : si32) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[PRIVATE1:.*]] = acc.firstprivate varPtr(%[[INTARR]] : !cir.ptr<!cir.array<!s32i x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!s32i x 5>> {name = "someIntArr[1:1]"}
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST2:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CAST2]] : si32) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[PRIVATE2:.*]] = acc.firstprivate varPtr(%[[FLOATARR]] : !cir.ptr<!cir.array<!cir.float x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!cir.float x 5>> {name = "someFloatArr[1:1]"}
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST2:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CAST2]] : si32) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[PRIVATE3:.*]] = acc.firstprivate varPtr(%[[NOCOPYARR]] : !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>> {name = "noCopyArr[1:1]"}
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST2:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CAST2]] : si32) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[PRIVATE4:.*]] = acc.firstprivate varPtr(%[[HASCOPYARR]] : !cir.ptr<!cir.array<!rec_CopyConstruct x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!rec_CopyConstruct x 5>> {name = "hasCopyArr[1:1]"}
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST2:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CAST2]] : si32) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[PRIVATE5:.*]] = acc.firstprivate varPtr(%[[NOTDEFCTORARR]] : !cir.ptr<!cir.array<!rec_NonDefaultCtor x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!rec_NonDefaultCtor x 5>> {name = "notDefCtorArr[1:1]"}
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST2:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CAST2]] : si32) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[PRIVATE6:.*]] = acc.firstprivate varPtr(%[[DTORARR]] : !cir.ptr<!cir.array<!rec_HasDtor x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!rec_HasDtor x 5>> {name = "dtorArr[1:1]"}
  // CHECK-NEXT: acc.parallel firstprivate(@firstprivatization__Bcnt1__ZTSA5_i -> %[[PRIVATE1]] : !cir.ptr<!cir.array<!s32i x 5>>,
  // CHECK-SAME: @firstprivatization__Bcnt1__ZTSA5_f -> %[[PRIVATE2]] : !cir.ptr<!cir.array<!cir.float x 5>>,
  // CHECK-SAME: @firstprivatization__Bcnt1__ZTSA5_15NoCopyConstruct -> %[[PRIVATE3]] : !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>>,
  // CHECK-SAME: @firstprivatization__Bcnt1__ZTSA5_13CopyConstruct -> %[[PRIVATE4]] : !cir.ptr<!cir.array<!rec_CopyConstruct x 5>>,
  // CHECK-SAME: @firstprivatization__Bcnt1__ZTSA5_14NonDefaultCtor -> %[[PRIVATE5]] : !cir.ptr<!cir.array<!rec_NonDefaultCtor x 5>>,
  // CHECK-SAME: @firstprivatization__Bcnt1__ZTSA5_7HasDtor -> %[[PRIVATE6]] : !cir.ptr<!cir.array<!rec_HasDtor x 5>>)
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
}
