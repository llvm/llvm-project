// RUN: not %clang_cc1 -fopenacc -triple x86_64-linux-gnu -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir -triple x86_64-linux-pc %s -o - | FileCheck %s

struct HasOperatorsInline {
  int i;
  float f;
  double d;

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


// CHECK: acc.reduction.recipe @reduction_lor__ZTSA5_18HasOperatorsInline : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> reduction_operator <lor> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>{{.*}})
// CHECK-NEXT: cir.alloca !cir.array<!rec_HasOperatorsInline x 5>, !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>, ["openacc.reduction.init"]
// TODO OpenACC: Expecting an initialization to... SOME value here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}):  
// CHECK-NEXT: %[[SIZE:.*]] = cir.const #cir.int<4>  : !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[ARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>), !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[LAST_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!rec_HasOperatorsInline>, %[[SIZE]] : !u64i), !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[IDX:.*]] = cir.alloca !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, ["__array_idx"] {alignment = 1 : i64}
// CHECK-NEXT: cir.store %[[LAST_ELT]], %[[IDX]] : !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[CUR:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.call @_ZN18HasOperatorsInlineD1Ev(%[[CUR]]) nothrow : (!cir.ptr<!rec_HasOperatorsInline>)
// CHECK-NEXT: %[[NEG:.*]] = cir.const #cir.int<-1> : !s64i
// CHECK-NEXT: %[[NEW_ITEM:.*]] = cir.ptr_stride(%[[CUR]] : !cir.ptr<!rec_HasOperatorsInline>, %[[NEG]] : !s64i), !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.store %[[NEW_ITEM]], %[[IDX]] : !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[CUR_LOAD:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(ne, %[[CUR_LOAD]], %[[DECAY]]) : !cir.ptr<!rec_HasOperatorsInline>, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]])
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_land__ZTSA5_18HasOperatorsInline : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> reduction_operator <land> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>{{.*}})
// CHECK-NEXT: cir.alloca !cir.array<!rec_HasOperatorsInline x 5>, !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>, ["openacc.reduction.init"]
// TODO OpenACC: Expecting an initialization to... SOME value here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}):  
// CHECK-NEXT: %[[SIZE:.*]] = cir.const #cir.int<4>  : !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[ARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>), !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[LAST_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!rec_HasOperatorsInline>, %[[SIZE]] : !u64i), !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[IDX:.*]] = cir.alloca !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, ["__array_idx"] {alignment = 1 : i64}
// CHECK-NEXT: cir.store %[[LAST_ELT]], %[[IDX]] : !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[CUR:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.call @_ZN18HasOperatorsInlineD1Ev(%[[CUR]]) nothrow : (!cir.ptr<!rec_HasOperatorsInline>)
// CHECK-NEXT: %[[NEG:.*]] = cir.const #cir.int<-1> : !s64i
// CHECK-NEXT: %[[NEW_ITEM:.*]] = cir.ptr_stride(%[[CUR]] : !cir.ptr<!rec_HasOperatorsInline>, %[[NEG]] : !s64i), !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.store %[[NEW_ITEM]], %[[IDX]] : !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[CUR_LOAD:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(ne, %[[CUR_LOAD]], %[[DECAY]]) : !cir.ptr<!rec_HasOperatorsInline>, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]])
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_xor__ZTSA5_18HasOperatorsInline : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> reduction_operator <xor> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>{{.*}})
// CHECK-NEXT: cir.alloca !cir.array<!rec_HasOperatorsInline x 5>, !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>, ["openacc.reduction.init"]
// TODO OpenACC: Expecting an initialization to... SOME value here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}):  
// CHECK-NEXT: %[[SIZE:.*]] = cir.const #cir.int<4>  : !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[ARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>), !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[LAST_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!rec_HasOperatorsInline>, %[[SIZE]] : !u64i), !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[IDX:.*]] = cir.alloca !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, ["__array_idx"] {alignment = 1 : i64}
// CHECK-NEXT: cir.store %[[LAST_ELT]], %[[IDX]] : !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[CUR:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.call @_ZN18HasOperatorsInlineD1Ev(%[[CUR]]) nothrow : (!cir.ptr<!rec_HasOperatorsInline>)
// CHECK-NEXT: %[[NEG:.*]] = cir.const #cir.int<-1> : !s64i
// CHECK-NEXT: %[[NEW_ITEM:.*]] = cir.ptr_stride(%[[CUR]] : !cir.ptr<!rec_HasOperatorsInline>, %[[NEG]] : !s64i), !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.store %[[NEW_ITEM]], %[[IDX]] : !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[CUR_LOAD:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(ne, %[[CUR_LOAD]], %[[DECAY]]) : !cir.ptr<!rec_HasOperatorsInline>, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]])
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_ior__ZTSA5_18HasOperatorsInline : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> reduction_operator <ior> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>{{.*}})
// CHECK-NEXT: cir.alloca !cir.array<!rec_HasOperatorsInline x 5>, !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>, ["openacc.reduction.init"]
// TODO OpenACC: Expecting an initialization to... SOME value here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}):  
// CHECK-NEXT: %[[SIZE:.*]] = cir.const #cir.int<4>  : !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[ARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>), !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[LAST_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!rec_HasOperatorsInline>, %[[SIZE]] : !u64i), !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[IDX:.*]] = cir.alloca !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, ["__array_idx"] {alignment = 1 : i64}
// CHECK-NEXT: cir.store %[[LAST_ELT]], %[[IDX]] : !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[CUR:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.call @_ZN18HasOperatorsInlineD1Ev(%[[CUR]]) nothrow : (!cir.ptr<!rec_HasOperatorsInline>)
// CHECK-NEXT: %[[NEG:.*]] = cir.const #cir.int<-1> : !s64i
// CHECK-NEXT: %[[NEW_ITEM:.*]] = cir.ptr_stride(%[[CUR]] : !cir.ptr<!rec_HasOperatorsInline>, %[[NEG]] : !s64i), !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.store %[[NEW_ITEM]], %[[IDX]] : !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[CUR_LOAD:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(ne, %[[CUR_LOAD]], %[[DECAY]]) : !cir.ptr<!rec_HasOperatorsInline>, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]])
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_iand__ZTSA5_18HasOperatorsInline : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> reduction_operator <iand> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>{{.*}})
// CHECK-NEXT: cir.alloca !cir.array<!rec_HasOperatorsInline x 5>, !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>, ["openacc.reduction.init"]
// TODO OpenACC: Expecting an initialization to... SOME value here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}):  
// CHECK-NEXT: %[[SIZE:.*]] = cir.const #cir.int<4>  : !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[ARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>), !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[LAST_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!rec_HasOperatorsInline>, %[[SIZE]] : !u64i), !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[IDX:.*]] = cir.alloca !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, ["__array_idx"] {alignment = 1 : i64}
// CHECK-NEXT: cir.store %[[LAST_ELT]], %[[IDX]] : !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[CUR:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.call @_ZN18HasOperatorsInlineD1Ev(%[[CUR]]) nothrow : (!cir.ptr<!rec_HasOperatorsInline>)
// CHECK-NEXT: %[[NEG:.*]] = cir.const #cir.int<-1> : !s64i
// CHECK-NEXT: %[[NEW_ITEM:.*]] = cir.ptr_stride(%[[CUR]] : !cir.ptr<!rec_HasOperatorsInline>, %[[NEG]] : !s64i), !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.store %[[NEW_ITEM]], %[[IDX]] : !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[CUR_LOAD:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(ne, %[[CUR_LOAD]], %[[DECAY]]) : !cir.ptr<!rec_HasOperatorsInline>, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]])
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_min__ZTSA5_18HasOperatorsInline : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> reduction_operator <min> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>{{.*}})
// CHECK-NEXT: cir.alloca !cir.array<!rec_HasOperatorsInline x 5>, !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>, ["openacc.reduction.init"]
// TODO OpenACC: Expecting an initialization to... SOME value here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}):  
// CHECK-NEXT: %[[SIZE:.*]] = cir.const #cir.int<4>  : !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[ARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>), !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[LAST_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!rec_HasOperatorsInline>, %[[SIZE]] : !u64i), !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[IDX:.*]] = cir.alloca !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, ["__array_idx"] {alignment = 1 : i64}
// CHECK-NEXT: cir.store %[[LAST_ELT]], %[[IDX]] : !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[CUR:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.call @_ZN18HasOperatorsInlineD1Ev(%[[CUR]]) nothrow : (!cir.ptr<!rec_HasOperatorsInline>)
// CHECK-NEXT: %[[NEG:.*]] = cir.const #cir.int<-1> : !s64i
// CHECK-NEXT: %[[NEW_ITEM:.*]] = cir.ptr_stride(%[[CUR]] : !cir.ptr<!rec_HasOperatorsInline>, %[[NEG]] : !s64i), !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.store %[[NEW_ITEM]], %[[IDX]] : !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[CUR_LOAD:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(ne, %[[CUR_LOAD]], %[[DECAY]]) : !cir.ptr<!rec_HasOperatorsInline>, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]])
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_max__ZTSA5_18HasOperatorsInline : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> reduction_operator <max> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>{{.*}})
// CHECK-NEXT: cir.alloca !cir.array<!rec_HasOperatorsInline x 5>, !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>, ["openacc.reduction.init"]
// TODO OpenACC: Expecting an initialization to... SOME value here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}):  
// CHECK-NEXT: %[[SIZE:.*]] = cir.const #cir.int<4>  : !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[ARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>), !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[LAST_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!rec_HasOperatorsInline>, %[[SIZE]] : !u64i), !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[IDX:.*]] = cir.alloca !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, ["__array_idx"] {alignment = 1 : i64}
// CHECK-NEXT: cir.store %[[LAST_ELT]], %[[IDX]] : !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[CUR:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.call @_ZN18HasOperatorsInlineD1Ev(%[[CUR]]) nothrow : (!cir.ptr<!rec_HasOperatorsInline>)
// CHECK-NEXT: %[[NEG:.*]] = cir.const #cir.int<-1> : !s64i
// CHECK-NEXT: %[[NEW_ITEM:.*]] = cir.ptr_stride(%[[CUR]] : !cir.ptr<!rec_HasOperatorsInline>, %[[NEG]] : !s64i), !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.store %[[NEW_ITEM]], %[[IDX]] : !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[CUR_LOAD:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(ne, %[[CUR_LOAD]], %[[DECAY]]) : !cir.ptr<!rec_HasOperatorsInline>, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]])
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_mul__ZTSA5_18HasOperatorsInline : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> reduction_operator <mul> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>{{.*}})
// CHECK-NEXT: cir.alloca !cir.array<!rec_HasOperatorsInline x 5>, !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>, ["openacc.reduction.init"]
// TODO OpenACC: Expecting an initialization to... SOME value here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}):  
// CHECK-NEXT: %[[SIZE:.*]] = cir.const #cir.int<4>  : !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[ARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>), !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[LAST_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!rec_HasOperatorsInline>, %[[SIZE]] : !u64i), !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[IDX:.*]] = cir.alloca !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, ["__array_idx"] {alignment = 1 : i64}
// CHECK-NEXT: cir.store %[[LAST_ELT]], %[[IDX]] : !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[CUR:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.call @_ZN18HasOperatorsInlineD1Ev(%[[CUR]]) nothrow : (!cir.ptr<!rec_HasOperatorsInline>)
// CHECK-NEXT: %[[NEG:.*]] = cir.const #cir.int<-1> : !s64i
// CHECK-NEXT: %[[NEW_ITEM:.*]] = cir.ptr_stride(%[[CUR]] : !cir.ptr<!rec_HasOperatorsInline>, %[[NEG]] : !s64i), !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.store %[[NEW_ITEM]], %[[IDX]] : !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[CUR_LOAD:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(ne, %[[CUR_LOAD]], %[[DECAY]]) : !cir.ptr<!rec_HasOperatorsInline>, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]])
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_add__ZTSA5_18HasOperatorsInline : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> reduction_operator <add> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>{{.*}})
// CHECK-NEXT: cir.alloca !cir.array<!rec_HasOperatorsInline x 5>, !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>, ["openacc.reduction.init"]
// TODO OpenACC: Expecting an initialization to... SOME value here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>> {{.*}}):  
// CHECK-NEXT: %[[SIZE:.*]] = cir.const #cir.int<4>  : !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[ARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsInline x 5>>), !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[LAST_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!rec_HasOperatorsInline>, %[[SIZE]] : !u64i), !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[IDX:.*]] = cir.alloca !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, ["__array_idx"] {alignment = 1 : i64}
// CHECK-NEXT: cir.store %[[LAST_ELT]], %[[IDX]] : !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[CUR:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.call @_ZN18HasOperatorsInlineD1Ev(%[[CUR]]) nothrow : (!cir.ptr<!rec_HasOperatorsInline>)
// CHECK-NEXT: %[[NEG:.*]] = cir.const #cir.int<-1> : !s64i
// CHECK-NEXT: %[[NEW_ITEM:.*]] = cir.ptr_stride(%[[CUR]] : !cir.ptr<!rec_HasOperatorsInline>, %[[NEG]] : !s64i), !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: cir.store %[[NEW_ITEM]], %[[IDX]] : !cir.ptr<!rec_HasOperatorsInline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[CUR_LOAD:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsInline>>, !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(ne, %[[CUR_LOAD]], %[[DECAY]]) : !cir.ptr<!rec_HasOperatorsInline>, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]])
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }


// CHECK-NEXT: acc.reduction.recipe @reduction_lor__ZTS18HasOperatorsInline : !cir.ptr<!rec_HasOperatorsInline> reduction_operator <lor> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_HasOperatorsInline>{{.*}})
// CHECK-NEXT: cir.alloca !rec_HasOperatorsInline, !cir.ptr<!rec_HasOperatorsInline>, ["openacc.reduction.init"]
// TODO OpenACC: Expecting an initialization to... SOME value here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}}):  
// CHECK-NEXT: cir.call @_ZN18HasOperatorsInlineD1Ev(%[[ARG]]) nothrow : (!cir.ptr<!rec_HasOperatorsInline>)
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_land__ZTS18HasOperatorsInline : !cir.ptr<!rec_HasOperatorsInline> reduction_operator <land> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_HasOperatorsInline>{{.*}})
// CHECK-NEXT: cir.alloca !rec_HasOperatorsInline, !cir.ptr<!rec_HasOperatorsInline>, ["openacc.reduction.init"]
// TODO OpenACC: Expecting an initialization to... SOME value here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}}):  
// CHECK-NEXT: cir.call @_ZN18HasOperatorsInlineD1Ev(%[[ARG]]) nothrow : (!cir.ptr<!rec_HasOperatorsInline>)
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_xor__ZTS18HasOperatorsInline : !cir.ptr<!rec_HasOperatorsInline> reduction_operator <xor> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_HasOperatorsInline>{{.*}})
// CHECK-NEXT: cir.alloca !rec_HasOperatorsInline, !cir.ptr<!rec_HasOperatorsInline>, ["openacc.reduction.init"]
// TODO OpenACC: Expecting an initialization to... SOME value here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}}):  
// CHECK-NEXT: cir.call @_ZN18HasOperatorsInlineD1Ev(%[[ARG]]) nothrow : (!cir.ptr<!rec_HasOperatorsInline>)
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_ior__ZTS18HasOperatorsInline : !cir.ptr<!rec_HasOperatorsInline> reduction_operator <ior> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_HasOperatorsInline>{{.*}})
// CHECK-NEXT: cir.alloca !rec_HasOperatorsInline, !cir.ptr<!rec_HasOperatorsInline>, ["openacc.reduction.init"]
// TODO OpenACC: Expecting an initialization to... SOME value here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}}):  
// CHECK-NEXT: cir.call @_ZN18HasOperatorsInlineD1Ev(%[[ARG]]) nothrow : (!cir.ptr<!rec_HasOperatorsInline>)
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_iand__ZTS18HasOperatorsInline : !cir.ptr<!rec_HasOperatorsInline> reduction_operator <iand> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_HasOperatorsInline>{{.*}})
// CHECK-NEXT: cir.alloca !rec_HasOperatorsInline, !cir.ptr<!rec_HasOperatorsInline>, ["openacc.reduction.init"]
// TODO OpenACC: Expecting an initialization to... SOME value here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}}):  
// CHECK-NEXT: cir.call @_ZN18HasOperatorsInlineD1Ev(%[[ARG]]) nothrow : (!cir.ptr<!rec_HasOperatorsInline>)
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_min__ZTS18HasOperatorsInline : !cir.ptr<!rec_HasOperatorsInline> reduction_operator <min> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_HasOperatorsInline>{{.*}})
// CHECK-NEXT: cir.alloca !rec_HasOperatorsInline, !cir.ptr<!rec_HasOperatorsInline>, ["openacc.reduction.init"]
// TODO OpenACC: Expecting an initialization to... SOME value here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}}):  
// CHECK-NEXT: cir.call @_ZN18HasOperatorsInlineD1Ev(%[[ARG]]) nothrow : (!cir.ptr<!rec_HasOperatorsInline>)
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_max__ZTS18HasOperatorsInline : !cir.ptr<!rec_HasOperatorsInline> reduction_operator <max> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_HasOperatorsInline>{{.*}})
// CHECK-NEXT: cir.alloca !rec_HasOperatorsInline, !cir.ptr<!rec_HasOperatorsInline>, ["openacc.reduction.init"]
// TODO OpenACC: Expecting an initialization to... SOME value here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}}):  
// CHECK-NEXT: cir.call @_ZN18HasOperatorsInlineD1Ev(%[[ARG]]) nothrow : (!cir.ptr<!rec_HasOperatorsInline>)
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_mul__ZTS18HasOperatorsInline : !cir.ptr<!rec_HasOperatorsInline> reduction_operator <mul> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_HasOperatorsInline>{{.*}})
// CHECK-NEXT: cir.alloca !rec_HasOperatorsInline, !cir.ptr<!rec_HasOperatorsInline>, ["openacc.reduction.init"]
// TODO OpenACC: Expecting an initialization to... SOME value here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}}):  
// CHECK-NEXT: cir.call @_ZN18HasOperatorsInlineD1Ev(%[[ARG]]) nothrow : (!cir.ptr<!rec_HasOperatorsInline>)
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_add__ZTS18HasOperatorsInline : !cir.ptr<!rec_HasOperatorsInline> reduction_operator <add> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_HasOperatorsInline>{{.*}})
// CHECK-NEXT: cir.alloca !rec_HasOperatorsInline, !cir.ptr<!rec_HasOperatorsInline>, ["openacc.reduction.init"]
// TODO OpenACC: Expecting an initialization to... SOME value here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!rec_HasOperatorsInline>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_HasOperatorsInline> {{.*}}):  
// CHECK-NEXT: cir.call @_ZN18HasOperatorsInlineD1Ev(%[[ARG]]) nothrow : (!cir.ptr<!rec_HasOperatorsInline>)
// CHECK-NEXT: acc.yield
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
  acc_compute<HasOperatorsInline>();
}
