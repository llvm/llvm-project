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

// CHECK: acc.firstprivate.recipe @firstprivatization__ZTSA5_7HasDtor : !cir.ptr<!cir.array<!rec_HasDtor x 5>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasDtor x 5>> {{.*}}):
// CHECK-NEXT: cir.alloca !cir.array<!rec_HasDtor x 5>, !cir.ptr<!cir.array<!rec_HasDtor x 5>>, ["openacc.firstprivate.init"]
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } copy {
// CHECK-NEXT: ^bb0(%[[ARG_FROM:.*]]: !cir.ptr<!cir.array<!rec_HasDtor x 5>> {{.*}}, %[[ARG_TO:.*]]: !cir.ptr<!cir.array<!rec_HasDtor x 5>> {{.*}}):
// CHECK-NEXT: %[[DECAY_TO:.*]] = cir.cast(array_to_ptrdecay, %[[ARG_TO]] : !cir.ptr<!cir.array<!rec_HasDtor x 5>>), !cir.ptr<!rec_HasDtor> 
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0>
// CHECK-NEXT: %[[DECAY_FROM:.*]] =  cir.cast(array_to_ptrdecay, %[[ARG_FROM]] : !cir.ptr<!cir.array<!rec_HasDtor x 5>>), !cir.ptr<!rec_HasDtor>
// CHECK-NEXT: %[[FROM_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_FROM]] : !cir.ptr<!rec_HasDtor>, %[[ZERO]] : !u64i), !cir.ptr<!rec_HasDtor>
// CHECK-NEXT: cir.call @_ZN7HasDtorC1ERKS_(%[[DECAY_TO]], %[[FROM_OFFSET]]) nothrow : (!cir.ptr<!rec_HasDtor>, !cir.ptr<!rec_HasDtor>) -> ()
//
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1>
// CHECK-NEXT: %[[TO_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_TO]] : !cir.ptr<!rec_HasDtor>, %[[ONE]] : !s64i), !cir.ptr<!rec_HasDtor>
// CHECK-NEXT: %[[ONE_2:.*]] = cir.const #cir.int<1>
// CHECK-NEXT: %[[DECAY_FROM:.*]] =  cir.cast(array_to_ptrdecay, %[[ARG_FROM]] : !cir.ptr<!cir.array<!rec_HasDtor x 5>>), !cir.ptr<!rec_HasDtor>
// CHECK-NEXT: %[[FROM_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_FROM]] : !cir.ptr<!rec_HasDtor>, %[[ONE_2]] : !u64i), !cir.ptr<!rec_HasDtor>
// CHECK-NEXT: cir.call @_ZN7HasDtorC1ERKS_(%[[TO_OFFSET]], %[[FROM_OFFSET]]) nothrow : (!cir.ptr<!rec_HasDtor>, !cir.ptr<!rec_HasDtor>) -> ()
//
// CHECK-NEXT: %[[TWO:.*]] = cir.const #cir.int<2>
// CHECK-NEXT: %[[TO_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_TO]] : !cir.ptr<!rec_HasDtor>, %[[TWO]] : !s64i), !cir.ptr<!rec_HasDtor>
// CHECK-NEXT: %[[TWO_2:.*]] = cir.const #cir.int<2>
// CHECK-NEXT: %[[DECAY_FROM:.*]] =  cir.cast(array_to_ptrdecay, %[[ARG_FROM]] : !cir.ptr<!cir.array<!rec_HasDtor x 5>>), !cir.ptr<!rec_HasDtor>
// CHECK-NEXT: %[[FROM_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_FROM]] : !cir.ptr<!rec_HasDtor>, %[[TWO_2]] : !u64i), !cir.ptr<!rec_HasDtor>
// CHECK-NEXT: cir.call @_ZN7HasDtorC1ERKS_(%[[TO_OFFSET]], %[[FROM_OFFSET]]) nothrow : (!cir.ptr<!rec_HasDtor>, !cir.ptr<!rec_HasDtor>) -> ()
//
// CHECK-NEXT: %[[THREE:.*]] = cir.const #cir.int<3>
// CHECK-NEXT: %[[TO_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_TO]] : !cir.ptr<!rec_HasDtor>, %[[THREE]] : !s64i), !cir.ptr<!rec_HasDtor>
// CHECK-NEXT: %[[THREE_2:.*]] = cir.const #cir.int<3>
// CHECK-NEXT: %[[DECAY_FROM:.*]] =  cir.cast(array_to_ptrdecay, %[[ARG_FROM]] : !cir.ptr<!cir.array<!rec_HasDtor x 5>>), !cir.ptr<!rec_HasDtor>
// CHECK-NEXT: %[[FROM_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_FROM]] : !cir.ptr<!rec_HasDtor>, %[[THREE_2]] : !u64i), !cir.ptr<!rec_HasDtor>
// CHECK-NEXT: cir.call @_ZN7HasDtorC1ERKS_(%[[TO_OFFSET]], %[[FROM_OFFSET]]) nothrow : (!cir.ptr<!rec_HasDtor>, !cir.ptr<!rec_HasDtor>) -> ()
//
// CHECK-NEXT: %[[FOUR:.*]] = cir.const #cir.int<4>
// CHECK-NEXT: %[[TO_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_TO]] : !cir.ptr<!rec_HasDtor>, %[[FOUR]] : !s64i), !cir.ptr<!rec_HasDtor>
// CHECK-NEXT: %[[FOUR_2:.*]] = cir.const #cir.int<4>
// CHECK-NEXT: %[[DECAY_FROM:.*]] =  cir.cast(array_to_ptrdecay, %[[ARG_FROM]] : !cir.ptr<!cir.array<!rec_HasDtor x 5>>), !cir.ptr<!rec_HasDtor>
// CHECK-NEXT: %[[FROM_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_FROM]] : !cir.ptr<!rec_HasDtor>, %[[FOUR_2]] : !u64i), !cir.ptr<!rec_HasDtor>
// CHECK-NEXT: cir.call @_ZN7HasDtorC1ERKS_(%[[TO_OFFSET]], %[[FROM_OFFSET]]) nothrow : (!cir.ptr<!rec_HasDtor>, !cir.ptr<!rec_HasDtor>) -> ()
//
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ORIG:.*]]: !cir.ptr<!cir.array<!rec_HasDtor x 5>> {{.*}}, %[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasDtor x 5>> {{.*}}):
// CHECK-NEXT: %[[LAST_IDX:.*]] = cir.const #cir.int<4> : !u64i
// CHECK-NEXT: %[[ARRPTR:.*]] = cir.cast(array_to_ptrdecay, %[[ARG]] : !cir.ptr<!cir.array<!rec_HasDtor x 5>>), !cir.ptr<!rec_HasDtor>
// CHECK-NEXT: %[[ELEM:.*]] = cir.ptr_stride(%[[ARRPTR]] : !cir.ptr<!rec_HasDtor>, %[[LAST_IDX]] : !u64i), !cir.ptr<!rec_HasDtor>
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !cir.ptr<!rec_HasDtor>, !cir.ptr<!cir.ptr<!rec_HasDtor>>, ["__array_idx"]
// CHECK-NEXT: cir.store %[[ELEM]], %[[ITR]] : !cir.ptr<!rec_HasDtor>, !cir.ptr<!cir.ptr<!rec_HasDtor>>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[ELEM_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!cir.ptr<!rec_HasDtor>>, !cir.ptr<!rec_HasDtor>
// CHECK-NEXT: cir.call @_ZN7HasDtorD1Ev(%[[ELEM_LOAD]]) nothrow : (!cir.ptr<!rec_HasDtor>) -> ()
// CHECK-NEXT: %[[NEG_ONE:.*]] =  cir.const #cir.int<-1> : !s64i
// CHECK-NEXT: %[[PREVELEM:.*]] = cir.ptr_stride(%[[ELEM_LOAD]] : !cir.ptr<!rec_HasDtor>, %[[NEG_ONE]] : !s64i), !cir.ptr<!rec_HasDtor>
// CHECK-NEXT: cir.store %[[PREVELEM]], %[[ITR]] : !cir.ptr<!rec_HasDtor>, !cir.ptr<!cir.ptr<!rec_HasDtor>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[ELEM_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!cir.ptr<!rec_HasDtor>>, !cir.ptr<!rec_HasDtor>
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(ne, %[[ELEM_LOAD]], %[[ARRPTR]]) : !cir.ptr<!rec_HasDtor>, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]])
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
//
// CHECK-NEXT: acc.firstprivate.recipe @firstprivatization__ZTSA5_14NonDefaultCtor : !cir.ptr<!cir.array<!rec_NonDefaultCtor x 5>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_NonDefaultCtor x 5>> {{.*}}):
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!rec_NonDefaultCtor x 5>, !cir.ptr<!cir.array<!rec_NonDefaultCtor x 5>>, ["openacc.firstprivate.init"]
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } copy {
// CHECK-NEXT: ^bb0(%[[ARG_FROM:.*]]: !cir.ptr<!cir.array<!rec_NonDefaultCtor x 5>> {{.*}}, %[[ARG_TO:.*]]: !cir.ptr<!cir.array<!rec_NonDefaultCtor x 5>> {{.*}}):
// CHECK-NEXT: %[[TO_DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[ARG_TO]] : !cir.ptr<!cir.array<!rec_NonDefaultCtor x 5>>), !cir.ptr<!rec_NonDefaultCtor>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0>
// CHECK-NEXT: %[[FROM_DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[ARG_FROM]] : !cir.ptr<!cir.array<!rec_NonDefaultCtor x 5>>), !cir.ptr<!rec_NonDefaultCtor>
// CHECK-NEXT: %[[FROM_OFFSET:.*]] = cir.ptr_stride(%[[FROM_DECAY]] : !cir.ptr<!rec_NonDefaultCtor>, %[[ZERO]] : !u64i), !cir.ptr<!rec_NonDefaultCtor>
// CHECK-NEXT: cir.call @_ZN14NonDefaultCtorC1ERKS_(%[[TO_DECAY]], %[[FROM_OFFSET]]) nothrow : (!cir.ptr<!rec_NonDefaultCtor>, !cir.ptr<!rec_NonDefaultCtor>) -> ()
//
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1>
// CHECK-NEXT: %[[TO_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_TO]] : !cir.ptr<!rec_NonDefaultCtor>, %[[ONE]] : !s64i), !cir.ptr<!rec_NonDefaultCtor>
// CHECK-NEXT: %[[ONE_2:.*]] = cir.const #cir.int<1>
// CHECK-NEXT: %[[DECAY_FROM:.*]] =  cir.cast(array_to_ptrdecay, %[[ARG_FROM]] : !cir.ptr<!cir.array<!rec_NonDefaultCtor x 5>>), !cir.ptr<!rec_NonDefaultCtor>
// CHECK-NEXT: %[[FROM_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_FROM]] : !cir.ptr<!rec_NonDefaultCtor>, %[[ONE_2]] : !u64i), !cir.ptr<!rec_NonDefaultCtor>
// CHECK-NEXT: cir.call @_ZN14NonDefaultCtorC1ERKS_(%[[TO_OFFSET]], %[[FROM_OFFSET]]) nothrow : (!cir.ptr<!rec_NonDefaultCtor>, !cir.ptr<!rec_NonDefaultCtor>) -> ()
//
// CHECK-NEXT: %[[TWO:.*]] = cir.const #cir.int<2>
// CHECK-NEXT: %[[TO_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_TO]] : !cir.ptr<!rec_NonDefaultCtor>, %[[TWO]] : !s64i), !cir.ptr<!rec_NonDefaultCtor>
// CHECK-NEXT: %[[TWO_2:.*]] = cir.const #cir.int<2>
// CHECK-NEXT: %[[DECAY_FROM:.*]] =  cir.cast(array_to_ptrdecay, %[[ARG_FROM]] : !cir.ptr<!cir.array<!rec_NonDefaultCtor x 5>>), !cir.ptr<!rec_NonDefaultCtor>
// CHECK-NEXT: %[[FROM_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_FROM]] : !cir.ptr<!rec_NonDefaultCtor>, %[[TWO_2]] : !u64i), !cir.ptr<!rec_NonDefaultCtor>
// CHECK-NEXT: cir.call @_ZN14NonDefaultCtorC1ERKS_(%[[TO_OFFSET]], %[[FROM_OFFSET]]) nothrow : (!cir.ptr<!rec_NonDefaultCtor>, !cir.ptr<!rec_NonDefaultCtor>) -> ()
//
// CHECK-NEXT: %[[THREE:.*]] = cir.const #cir.int<3>
// CHECK-NEXT: %[[TO_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_TO]] : !cir.ptr<!rec_NonDefaultCtor>, %[[THREE]] : !s64i), !cir.ptr<!rec_NonDefaultCtor>
// CHECK-NEXT: %[[THREE_2:.*]] = cir.const #cir.int<3>
// CHECK-NEXT: %[[DECAY_FROM:.*]] =  cir.cast(array_to_ptrdecay, %[[ARG_FROM]] : !cir.ptr<!cir.array<!rec_NonDefaultCtor x 5>>), !cir.ptr<!rec_NonDefaultCtor>
// CHECK-NEXT: %[[FROM_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_FROM]] : !cir.ptr<!rec_NonDefaultCtor>, %[[THREE_2]] : !u64i), !cir.ptr<!rec_NonDefaultCtor>
// CHECK-NEXT: cir.call @_ZN14NonDefaultCtorC1ERKS_(%[[TO_OFFSET]], %[[FROM_OFFSET]]) nothrow : (!cir.ptr<!rec_NonDefaultCtor>, !cir.ptr<!rec_NonDefaultCtor>) -> ()
//
// CHECK-NEXT: %[[FOUR:.*]] = cir.const #cir.int<4>
// CHECK-NEXT: %[[TO_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_TO]] : !cir.ptr<!rec_NonDefaultCtor>, %[[FOUR]] : !s64i), !cir.ptr<!rec_NonDefaultCtor>
// CHECK-NEXT: %[[FOUR_2:.*]] = cir.const #cir.int<4>
// CHECK-NEXT: %[[DECAY_FROM:.*]] =  cir.cast(array_to_ptrdecay, %[[ARG_FROM]] : !cir.ptr<!cir.array<!rec_NonDefaultCtor x 5>>), !cir.ptr<!rec_NonDefaultCtor>
// CHECK-NEXT: %[[FROM_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_FROM]] : !cir.ptr<!rec_NonDefaultCtor>, %[[FOUR_2]] : !u64i), !cir.ptr<!rec_NonDefaultCtor>
// CHECK-NEXT: cir.call @_ZN14NonDefaultCtorC1ERKS_(%[[TO_OFFSET]], %[[FROM_OFFSET]]) nothrow : (!cir.ptr<!rec_NonDefaultCtor>, !cir.ptr<!rec_NonDefaultCtor>) -> ()
//
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
//
// CHECK-NEXT: acc.firstprivate.recipe @firstprivatization__ZTSA5_13CopyConstruct : !cir.ptr<!cir.array<!rec_CopyConstruct x 5>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_CopyConstruct x 5>> {{.*}}):
// CHECK-NEXT: cir.alloca !cir.array<!rec_CopyConstruct x 5>, !cir.ptr<!cir.array<!rec_CopyConstruct x 5>>, ["openacc.firstprivate.init"]
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } copy {
// CHECK-NEXT: ^bb0(%[[ARG_FROM:.*]]: !cir.ptr<!cir.array<!rec_CopyConstruct x 5>> {{.*}}, %[[ARG_TO:.*]]: !cir.ptr<!cir.array<!rec_CopyConstruct x 5>> {{.*}}):
// CHECK-NEXT: %[[TO_DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[ARG_TO]] : !cir.ptr<!cir.array<!rec_CopyConstruct x 5>>), !cir.ptr<!rec_CopyConstruct>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0>
// CHECK-NEXT: %[[FROM_DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[ARG_FROM]] : !cir.ptr<!cir.array<!rec_CopyConstruct x 5>>), !cir.ptr<!rec_CopyConstruct>
// CHECK-NEXT: %[[FROM_OFFSET:.*]] = cir.ptr_stride(%[[FROM_DECAY]] : !cir.ptr<!rec_CopyConstruct>, %[[ZERO]] : !u64i), !cir.ptr<!rec_CopyConstruct>
// CHECK-NEXT: cir.call @_ZN13CopyConstructC1ERKS_(%[[TO_DECAY]], %[[FROM_OFFSET]]) : (!cir.ptr<!rec_CopyConstruct>, !cir.ptr<!rec_CopyConstruct>) -> ()
//
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1>
// CHECK-NEXT: %[[TO_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_TO]] : !cir.ptr<!rec_CopyConstruct>, %[[ONE]] : !s64i), !cir.ptr<!rec_CopyConstruct>
// CHECK-NEXT: %[[ONE_2:.*]] = cir.const #cir.int<1>
// CHECK-NEXT: %[[DECAY_FROM:.*]] =  cir.cast(array_to_ptrdecay, %[[ARG_FROM]] : !cir.ptr<!cir.array<!rec_CopyConstruct x 5>>), !cir.ptr<!rec_CopyConstruct>
// CHECK-NEXT: %[[FROM_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_FROM]] : !cir.ptr<!rec_CopyConstruct>, %[[ONE_2]] : !u64i), !cir.ptr<!rec_CopyConstruct>
// CHECK-NEXT: cir.call @_ZN13CopyConstructC1ERKS_(%[[TO_OFFSET]], %[[FROM_OFFSET]]) : (!cir.ptr<!rec_CopyConstruct>, !cir.ptr<!rec_CopyConstruct>) -> ()
//
// CHECK-NEXT: %[[TWO:.*]] = cir.const #cir.int<2>
// CHECK-NEXT: %[[TO_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_TO]] : !cir.ptr<!rec_CopyConstruct>, %[[TWO]] : !s64i), !cir.ptr<!rec_CopyConstruct>
// CHECK-NEXT: %[[TWO_2:.*]] = cir.const #cir.int<2>
// CHECK-NEXT: %[[DECAY_FROM:.*]] =  cir.cast(array_to_ptrdecay, %[[ARG_FROM]] : !cir.ptr<!cir.array<!rec_CopyConstruct x 5>>), !cir.ptr<!rec_CopyConstruct>
// CHECK-NEXT: %[[FROM_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_FROM]] : !cir.ptr<!rec_CopyConstruct>, %[[TWO_2]] : !u64i), !cir.ptr<!rec_CopyConstruct>
// CHECK-NEXT: cir.call @_ZN13CopyConstructC1ERKS_(%[[TO_OFFSET]], %[[FROM_OFFSET]]) : (!cir.ptr<!rec_CopyConstruct>, !cir.ptr<!rec_CopyConstruct>) -> ()
//
// CHECK-NEXT: %[[THREE:.*]] = cir.const #cir.int<3>
// CHECK-NEXT: %[[TO_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_TO]] : !cir.ptr<!rec_CopyConstruct>, %[[THREE]] : !s64i), !cir.ptr<!rec_CopyConstruct>
// CHECK-NEXT: %[[THREE_2:.*]] = cir.const #cir.int<3>
// CHECK-NEXT: %[[DECAY_FROM:.*]] =  cir.cast(array_to_ptrdecay, %[[ARG_FROM]] : !cir.ptr<!cir.array<!rec_CopyConstruct x 5>>), !cir.ptr<!rec_CopyConstruct>
// CHECK-NEXT: %[[FROM_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_FROM]] : !cir.ptr<!rec_CopyConstruct>, %[[THREE_2]] : !u64i), !cir.ptr<!rec_CopyConstruct>
// CHECK-NEXT: cir.call @_ZN13CopyConstructC1ERKS_(%[[TO_OFFSET]], %[[FROM_OFFSET]]) : (!cir.ptr<!rec_CopyConstruct>, !cir.ptr<!rec_CopyConstruct>) -> ()
//
// CHECK-NEXT: %[[FOUR:.*]] = cir.const #cir.int<4>
// CHECK-NEXT: %[[TO_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_TO]] : !cir.ptr<!rec_CopyConstruct>, %[[FOUR]] : !s64i), !cir.ptr<!rec_CopyConstruct>
// CHECK-NEXT: %[[FOUR_2:.*]] = cir.const #cir.int<4>
// CHECK-NEXT: %[[DECAY_FROM:.*]] =  cir.cast(array_to_ptrdecay, %[[ARG_FROM]] : !cir.ptr<!cir.array<!rec_CopyConstruct x 5>>), !cir.ptr<!rec_CopyConstruct>
// CHECK-NEXT: %[[FROM_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_FROM]] : !cir.ptr<!rec_CopyConstruct>, %[[FOUR_2]] : !u64i), !cir.ptr<!rec_CopyConstruct>
// CHECK-NEXT: cir.call @_ZN13CopyConstructC1ERKS_(%[[TO_OFFSET]], %[[FROM_OFFSET]]) : (!cir.ptr<!rec_CopyConstruct>, !cir.ptr<!rec_CopyConstruct>) -> ()
//
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
//
// CHECK-NEXT: acc.firstprivate.recipe @firstprivatization__ZTSA5_15NoCopyConstruct : !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>> {{.*}}):
// CHECK-NEXT: cir.alloca !cir.array<!rec_NoCopyConstruct x 5>, !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>>, ["openacc.firstprivate.init"]
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } copy {
// CHECK-NEXT: ^bb0(%[[ARG_FROM:.*]]: !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>> {{.*}}, %[[ARG_TO:.*]]: !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>> {{.*}}):
// CHECK-NEXT: %[[TO_DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[ARG_TO]] : !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>>), !cir.ptr<!rec_NoCopyConstruct>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0>
// CHECK-NEXT: %[[FROM_DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[ARG_FROM]] : !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>>), !cir.ptr<!rec_NoCopyConstruct>
// CHECK-NEXT: %[[FROM_OFFSET:.*]] = cir.ptr_stride(%[[FROM_DECAY]] : !cir.ptr<!rec_NoCopyConstruct>, %[[ZERO]] : !u64i), !cir.ptr<!rec_NoCopyConstruct>
// CHECK-NEXT: cir.call @_ZN15NoCopyConstructC1ERKS_(%[[TO_DECAY]], %[[FROM_OFFSET]]) nothrow : (!cir.ptr<!rec_NoCopyConstruct>, !cir.ptr<!rec_NoCopyConstruct>) -> ()
//
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1>
// CHECK-NEXT: %[[TO_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_TO]] : !cir.ptr<!rec_NoCopyConstruct>, %[[ONE]] : !s64i), !cir.ptr<!rec_NoCopyConstruct>
// CHECK-NEXT: %[[ONE_2:.*]] = cir.const #cir.int<1>
// CHECK-NEXT: %[[DECAY_FROM:.*]] =  cir.cast(array_to_ptrdecay, %[[ARG_FROM]] : !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>>), !cir.ptr<!rec_NoCopyConstruct>
// CHECK-NEXT: %[[FROM_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_FROM]] : !cir.ptr<!rec_NoCopyConstruct>, %[[ONE_2]] : !u64i), !cir.ptr<!rec_NoCopyConstruct>
// CHECK-NEXT: cir.call @_ZN15NoCopyConstructC1ERKS_(%[[TO_OFFSET]], %[[FROM_OFFSET]]) nothrow : (!cir.ptr<!rec_NoCopyConstruct>, !cir.ptr<!rec_NoCopyConstruct>) -> ()
//
// CHECK-NEXT: %[[TWO:.*]] = cir.const #cir.int<2>
// CHECK-NEXT: %[[TO_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_TO]] : !cir.ptr<!rec_NoCopyConstruct>, %[[TWO]] : !s64i), !cir.ptr<!rec_NoCopyConstruct>
// CHECK-NEXT: %[[TWO_2:.*]] = cir.const #cir.int<2>
// CHECK-NEXT: %[[DECAY_FROM:.*]] =  cir.cast(array_to_ptrdecay, %[[ARG_FROM]] : !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>>), !cir.ptr<!rec_NoCopyConstruct>
// CHECK-NEXT: %[[FROM_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_FROM]] : !cir.ptr<!rec_NoCopyConstruct>, %[[TWO_2]] : !u64i), !cir.ptr<!rec_NoCopyConstruct>
// CHECK-NEXT: cir.call @_ZN15NoCopyConstructC1ERKS_(%[[TO_OFFSET]], %[[FROM_OFFSET]]) nothrow : (!cir.ptr<!rec_NoCopyConstruct>, !cir.ptr<!rec_NoCopyConstruct>) -> ()
//
// CHECK-NEXT: %[[THREE:.*]] = cir.const #cir.int<3>
// CHECK-NEXT: %[[TO_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_TO]] : !cir.ptr<!rec_NoCopyConstruct>, %[[THREE]] : !s64i), !cir.ptr<!rec_NoCopyConstruct>
// CHECK-NEXT: %[[THREE_2:.*]] = cir.const #cir.int<3>
// CHECK-NEXT: %[[DECAY_FROM:.*]] =  cir.cast(array_to_ptrdecay, %[[ARG_FROM]] : !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>>), !cir.ptr<!rec_NoCopyConstruct>
// CHECK-NEXT: %[[FROM_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_FROM]] : !cir.ptr<!rec_NoCopyConstruct>, %[[THREE_2]] : !u64i), !cir.ptr<!rec_NoCopyConstruct>
// CHECK-NEXT: cir.call @_ZN15NoCopyConstructC1ERKS_(%[[TO_OFFSET]], %[[FROM_OFFSET]]) nothrow : (!cir.ptr<!rec_NoCopyConstruct>, !cir.ptr<!rec_NoCopyConstruct>) -> ()
//
// CHECK-NEXT: %[[FOUR:.*]] = cir.const #cir.int<4>
// CHECK-NEXT: %[[TO_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_TO]] : !cir.ptr<!rec_NoCopyConstruct>, %[[FOUR]] : !s64i), !cir.ptr<!rec_NoCopyConstruct>
// CHECK-NEXT: %[[FOUR_2:.*]] = cir.const #cir.int<4>
// CHECK-NEXT: %[[DECAY_FROM:.*]] =  cir.cast(array_to_ptrdecay, %[[ARG_FROM]] : !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>>), !cir.ptr<!rec_NoCopyConstruct>
// CHECK-NEXT: %[[FROM_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_FROM]] : !cir.ptr<!rec_NoCopyConstruct>, %[[FOUR_2]] : !u64i), !cir.ptr<!rec_NoCopyConstruct>
// CHECK-NEXT: cir.call @_ZN15NoCopyConstructC1ERKS_(%[[TO_OFFSET]], %[[FROM_OFFSET]]) nothrow : (!cir.ptr<!rec_NoCopyConstruct>, !cir.ptr<!rec_NoCopyConstruct>) -> ()
//
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
//
// CHECK-NEXT: acc.firstprivate.recipe @firstprivatization__ZTSA5_f : !cir.ptr<!cir.array<!cir.float x 5>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>> {{.*}}):
// CHECK-NEXT: cir.alloca !cir.array<!cir.float x 5>, !cir.ptr<!cir.array<!cir.float x 5>>, ["openacc.firstprivate.init"]
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } copy {
// CHECK-NEXT: ^bb0(%[[ARG_FROM:.*]]: !cir.ptr<!cir.array<!cir.float x 5>> {{.*}}, %[[ARG_TO:.*]]: !cir.ptr<!cir.array<!cir.float x 5>> {{.*}}):
// CHECK-NEXT: %[[TO_DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[ARG_TO]] : !cir.ptr<!cir.array<!cir.float x 5>>), !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0>
// CHECK-NEXT: %[[FROM_DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[ARG_FROM]] : !cir.ptr<!cir.array<!cir.float x 5>>), !cir.ptr<!cir.float>
// CHECK-NEXT: %[[FROM_OFFSET:.*]] = cir.ptr_stride(%[[FROM_DECAY]] : !cir.ptr<!cir.float>, %[[ZERO]] : !u64i), !cir.ptr<!cir.float>
// CHECK-NEXT: %[[FROM_LOAD:.*]] = cir.load {{.*}}%[[FROM_OFFSET]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[FROM_LOAD]], %[[TO_DECAY]] : !cir.float, !cir.ptr<!cir.float>
//
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1>
// CHECK-NEXT: %[[TO_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_TO]] : !cir.ptr<!cir.float>, %[[ONE]] : !s64i), !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE_2:.*]] = cir.const #cir.int<1>
// CHECK-NEXT: %[[DECAY_FROM:.*]] =  cir.cast(array_to_ptrdecay, %[[ARG_FROM]] : !cir.ptr<!cir.array<!cir.float x 5>>), !cir.ptr<!cir.float>
// CHECK-NEXT: %[[FROM_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_FROM]] : !cir.ptr<!cir.float>, %[[ONE_2]] : !u64i), !cir.ptr<!cir.float>
// CHECK-NEXT: %[[FROM_LOAD:.*]] = cir.load {{.*}}%[[FROM_OFFSET]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[FROM_LOAD]], %[[TO_OFFSET]] : !cir.float, !cir.ptr<!cir.float>
//
// CHECK-NEXT: %[[TWO:.*]] = cir.const #cir.int<2>
// CHECK-NEXT: %[[TO_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_TO]] : !cir.ptr<!cir.float>, %[[TWO]] : !s64i), !cir.ptr<!cir.float>
// CHECK-NEXT: %[[TWO_2:.*]] = cir.const #cir.int<2>
// CHECK-NEXT: %[[DECAY_FROM:.*]] =  cir.cast(array_to_ptrdecay, %[[ARG_FROM]] : !cir.ptr<!cir.array<!cir.float x 5>>), !cir.ptr<!cir.float>
// CHECK-NEXT: %[[FROM_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_FROM]] : !cir.ptr<!cir.float>, %[[TWO_2]] : !u64i), !cir.ptr<!cir.float>
// CHECK-NEXT: %[[FROM_LOAD:.*]] = cir.load {{.*}}%[[FROM_OFFSET]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[FROM_LOAD]], %[[TO_OFFSET]] : !cir.float, !cir.ptr<!cir.float>
//
// CHECK-NEXT: %[[THREE:.*]] = cir.const #cir.int<3>
// CHECK-NEXT: %[[TO_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_TO]] : !cir.ptr<!cir.float>, %[[THREE]] : !s64i), !cir.ptr<!cir.float>
// CHECK-NEXT: %[[THREE_2:.*]] = cir.const #cir.int<3>
// CHECK-NEXT: %[[DECAY_FROM:.*]] =  cir.cast(array_to_ptrdecay, %[[ARG_FROM]] : !cir.ptr<!cir.array<!cir.float x 5>>), !cir.ptr<!cir.float>
// CHECK-NEXT: %[[FROM_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_FROM]] : !cir.ptr<!cir.float>, %[[THREE_2]] : !u64i), !cir.ptr<!cir.float>
// CHECK-NEXT: %[[FROM_LOAD:.*]] = cir.load {{.*}}%[[FROM_OFFSET]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[FROM_LOAD]], %[[TO_OFFSET]] : !cir.float, !cir.ptr<!cir.float>
//
// CHECK-NEXT: %[[FOUR:.*]] = cir.const #cir.int<4>
// CHECK-NEXT: %[[TO_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_TO]] : !cir.ptr<!cir.float>, %[[FOUR]] : !s64i), !cir.ptr<!cir.float>
// CHECK-NEXT: %[[FOUR_2:.*]] = cir.const #cir.int<4>
// CHECK-NEXT: %[[DECAY_FROM:.*]] =  cir.cast(array_to_ptrdecay, %[[ARG_FROM]] : !cir.ptr<!cir.array<!cir.float x 5>>), !cir.ptr<!cir.float>
// CHECK-NEXT: %[[FROM_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_FROM]] : !cir.ptr<!cir.float>, %[[FOUR_2]] : !u64i), !cir.ptr<!cir.float>
// CHECK-NEXT: %[[FROM_LOAD:.*]] = cir.load {{.*}}%[[FROM_OFFSET]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[FROM_LOAD]], %[[TO_OFFSET]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
//
// CHECK-NEXT: acc.firstprivate.recipe @firstprivatization__ZTSA5_i : !cir.ptr<!cir.array<!s32i x 5>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}}):
// CHECK-NEXT: cir.alloca !cir.array<!s32i x 5>, !cir.ptr<!cir.array<!s32i x 5>>, ["openacc.firstprivate.init"]
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } copy {
// CHECK-NEXT: ^bb0(%[[ARG_FROM:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}}, %[[ARG_TO:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}}):
// CHECK-NEXT: %[[TO_DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[ARG_TO]] : !cir.ptr<!cir.array<!s32i x 5>>), !cir.ptr<!s32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0>
// CHECK-NEXT: %[[FROM_DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[ARG_FROM]] : !cir.ptr<!cir.array<!s32i x 5>>), !cir.ptr<!s32i>
// CHECK-NEXT: %[[FROM_OFFSET:.*]] = cir.ptr_stride(%[[FROM_DECAY]] : !cir.ptr<!s32i>, %[[ZERO]] : !u64i), !cir.ptr<!s32i>
// CHECK-NEXT: %[[FROM_LOAD:.*]] = cir.load {{.*}}%[[FROM_OFFSET]] : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT: cir.store {{.*}} %[[FROM_LOAD]], %[[TO_DECAY]] : !s32i, !cir.ptr<!s32i>
//
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1>
// CHECK-NEXT: %[[TO_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_TO]] : !cir.ptr<!s32i>, %[[ONE]] : !s64i), !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE_2:.*]] = cir.const #cir.int<1>
// CHECK-NEXT: %[[DECAY_FROM:.*]] =  cir.cast(array_to_ptrdecay, %[[ARG_FROM]] : !cir.ptr<!cir.array<!s32i x 5>>), !cir.ptr<!s32i>
// CHECK-NEXT: %[[FROM_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_FROM]] : !cir.ptr<!s32i>, %[[ONE_2]] : !u64i), !cir.ptr<!s32i>
// CHECK-NEXT: %[[FROM_LOAD:.*]] = cir.load {{.*}}%[[FROM_OFFSET]] : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT: cir.store {{.*}} %[[FROM_LOAD]], %[[TO_OFFSET]] : !s32i, !cir.ptr<!s32i>
//
// CHECK-NEXT: %[[TWO:.*]] = cir.const #cir.int<2>
// CHECK-NEXT: %[[TO_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_TO]] : !cir.ptr<!s32i>, %[[TWO]] : !s64i), !cir.ptr<!s32i>
// CHECK-NEXT: %[[TWO_2:.*]] = cir.const #cir.int<2>
// CHECK-NEXT: %[[DECAY_FROM:.*]] =  cir.cast(array_to_ptrdecay, %[[ARG_FROM]] : !cir.ptr<!cir.array<!s32i x 5>>), !cir.ptr<!s32i>
// CHECK-NEXT: %[[FROM_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_FROM]] : !cir.ptr<!s32i>, %[[TWO_2]] : !u64i), !cir.ptr<!s32i>
// CHECK-NEXT: %[[FROM_LOAD:.*]] = cir.load {{.*}}%[[FROM_OFFSET]] : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT: cir.store {{.*}} %[[FROM_LOAD]], %[[TO_OFFSET]] : !s32i, !cir.ptr<!s32i>
//
// CHECK-NEXT: %[[THREE:.*]] = cir.const #cir.int<3>
// CHECK-NEXT: %[[TO_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_TO]] : !cir.ptr<!s32i>, %[[THREE]] : !s64i), !cir.ptr<!s32i>
// CHECK-NEXT: %[[THREE_2:.*]] = cir.const #cir.int<3>
// CHECK-NEXT: %[[DECAY_FROM:.*]] =  cir.cast(array_to_ptrdecay, %[[ARG_FROM]] : !cir.ptr<!cir.array<!s32i x 5>>), !cir.ptr<!s32i>
// CHECK-NEXT: %[[FROM_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_FROM]] : !cir.ptr<!s32i>, %[[THREE_2]] : !u64i), !cir.ptr<!s32i>
// CHECK-NEXT: %[[FROM_LOAD:.*]] = cir.load {{.*}}%[[FROM_OFFSET]] : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT: cir.store {{.*}} %[[FROM_LOAD]], %[[TO_OFFSET]] : !s32i, !cir.ptr<!s32i>
//
// CHECK-NEXT: %[[FOUR:.*]] = cir.const #cir.int<4>
// CHECK-NEXT: %[[TO_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_TO]] : !cir.ptr<!s32i>, %[[FOUR]] : !s64i), !cir.ptr<!s32i>
// CHECK-NEXT: %[[FOUR_2:.*]] = cir.const #cir.int<4>
// CHECK-NEXT: %[[DECAY_FROM:.*]] =  cir.cast(array_to_ptrdecay, %[[ARG_FROM]] : !cir.ptr<!cir.array<!s32i x 5>>), !cir.ptr<!s32i>
// CHECK-NEXT: %[[FROM_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_FROM]] : !cir.ptr<!s32i>, %[[FOUR_2]] : !u64i), !cir.ptr<!s32i>
// CHECK-NEXT: %[[FROM_LOAD:.*]] = cir.load {{.*}}%[[FROM_OFFSET]] : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT: cir.store {{.*}} %[[FROM_LOAD]], %[[TO_OFFSET]] : !s32i, !cir.ptr<!s32i>
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
// CHECK-NEXT: cir.call @_ZN13CopyConstructC1ERKS_(%[[ARG_TO]], %[[ARG_FROM]]) : (!cir.ptr<!rec_CopyConstruct>, !cir.ptr<!rec_CopyConstruct>) -> ()
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
// CHECK-NEXT: acc.firstprivate.recipe @firstprivatization__ZTSi : !cir.ptr<!s32i> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!s32i> {{.*}}):
// CHECK-NEXT: cir.alloca !s32i, !cir.ptr<!s32i>, ["openacc.firstprivate.init"]
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } copy {
// CHECK-NEXT: ^bb0(%[[ARG_FROM:.*]]: !cir.ptr<!s32i> {{.*}}, %[[ARG_TO:.*]]: !cir.ptr<!s32i> {{.*}}):
// CHECK-NEXT: %[[FROM_LOAD:.*]] = cir.load {{.*}}%[[ARG_FROM]] : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT: cir.store {{.*}} %[[FROM_LOAD]], %[[ARG_TO]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }

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
  // CHECK-NEXT: acc.serial firstprivate(@firstprivatization__ZTSA5_i -> %[[PRIVATE]] : !cir.ptr<!cir.array<!s32i x 5>>)
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
  // CHECK-NEXT: acc.parallel firstprivate(@firstprivatization__ZTSA5_f -> %[[PRIVATE]] : !cir.ptr<!cir.array<!cir.float x 5>>)
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
  // CHECK-NEXT: acc.serial firstprivate(@firstprivatization__ZTSA5_15NoCopyConstruct -> %[[PRIVATE]] : !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>>)
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
  // CHECK-NEXT: acc.parallel firstprivate(@firstprivatization__ZTSA5_13CopyConstruct -> %[[PRIVATE]] : !cir.ptr<!cir.array<!rec_CopyConstruct x 5>>)
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
  // CHECK-NEXT: acc.parallel firstprivate(@firstprivatization__ZTSA5_14NonDefaultCtor -> %[[PRIVATE]] : !cir.ptr<!cir.array<!rec_NonDefaultCtor x 5>>)
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
  // CHECK-NEXT: acc.parallel firstprivate(@firstprivatization__ZTSA5_7HasDtor -> %[[PRIVATE]] : !cir.ptr<!cir.array<!rec_HasDtor x 5>>)
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
  // CHECK-NEXT: acc.serial firstprivate(@firstprivatization__ZTSA5_i -> %[[PRIVATE1]] : !cir.ptr<!cir.array<!s32i x 5>>,
  // CHECK-SAME: @firstprivatization__ZTSA5_f -> %[[PRIVATE2]] : !cir.ptr<!cir.array<!cir.float x 5>>,
  // CHECK-SAME: @firstprivatization__ZTSA5_15NoCopyConstruct -> %[[PRIVATE3]] : !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>>,
  // CHECK-SAME: @firstprivatization__ZTSA5_13CopyConstruct -> %[[PRIVATE4]] : !cir.ptr<!cir.array<!rec_CopyConstruct x 5>>,
  // CHECK-SAME: @firstprivatization__ZTSA5_14NonDefaultCtor -> %[[PRIVATE5]] : !cir.ptr<!cir.array<!rec_NonDefaultCtor x 5>>,
  // CHECK-SAME: @firstprivatization__ZTSA5_7HasDtor -> %[[PRIVATE6]] : !cir.ptr<!cir.array<!rec_HasDtor x 5>>)
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
  // CHECK-NEXT: acc.parallel firstprivate(@firstprivatization__ZTSA5_i -> %[[PRIVATE]] : !cir.ptr<!cir.array<!s32i x 5>>)
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
  // CHECK-NEXT: acc.serial firstprivate(@firstprivatization__ZTSA5_f -> %[[PRIVATE]] : !cir.ptr<!cir.array<!cir.float x 5>>)
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
  // CHECK-NEXT: acc.parallel firstprivate(@firstprivatization__ZTSA5_15NoCopyConstruct -> %[[PRIVATE]] : !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>>)
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
  // CHECK-NEXT: acc.serial firstprivate(@firstprivatization__ZTSA5_13CopyConstruct -> %[[PRIVATE]] : !cir.ptr<!cir.array<!rec_CopyConstruct x 5>>)
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
  // CHECK-NEXT: acc.parallel firstprivate(@firstprivatization__ZTSA5_14NonDefaultCtor -> %[[PRIVATE]] : !cir.ptr<!cir.array<!rec_NonDefaultCtor x 5>>)
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
  // CHECK-NEXT: acc.parallel firstprivate(@firstprivatization__ZTSA5_7HasDtor -> %[[PRIVATE]] : !cir.ptr<!cir.array<!rec_HasDtor x 5>>)
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
  // CHECK-NEXT: acc.parallel firstprivate(@firstprivatization__ZTSA5_i -> %[[PRIVATE1]] : !cir.ptr<!cir.array<!s32i x 5>>,
  // CHECK-SAME: @firstprivatization__ZTSA5_f -> %[[PRIVATE2]] : !cir.ptr<!cir.array<!cir.float x 5>>,
  // CHECK-SAME: @firstprivatization__ZTSA5_15NoCopyConstruct -> %[[PRIVATE3]] : !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>>,
  // CHECK-SAME: @firstprivatization__ZTSA5_13CopyConstruct -> %[[PRIVATE4]] : !cir.ptr<!cir.array<!rec_CopyConstruct x 5>>,
  // CHECK-SAME: @firstprivatization__ZTSA5_14NonDefaultCtor -> %[[PRIVATE5]] : !cir.ptr<!cir.array<!rec_NonDefaultCtor x 5>>,
  // CHECK-SAME: @firstprivatization__ZTSA5_7HasDtor -> %[[PRIVATE6]] : !cir.ptr<!cir.array<!rec_HasDtor x 5>>)
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
}
