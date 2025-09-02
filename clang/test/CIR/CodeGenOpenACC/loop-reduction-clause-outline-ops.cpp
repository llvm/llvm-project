// RUN: not %clang_cc1 -fopenacc -triple x86_64-linux-gnu -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir -triple x86_64-linux-pc %s -o - | FileCheck %s
struct HasOperatorsOutline {
  int i;
  float f;
  double d;

  ~HasOperatorsOutline();
  HasOperatorsOutline &operator=(const HasOperatorsOutline &);
};

HasOperatorsOutline &operator+=(HasOperatorsOutline &, HasOperatorsOutline &);
HasOperatorsOutline &operator*=(HasOperatorsOutline &, HasOperatorsOutline &);
HasOperatorsOutline &operator&=(HasOperatorsOutline &, HasOperatorsOutline &);
HasOperatorsOutline &operator|=(HasOperatorsOutline &, HasOperatorsOutline &);
HasOperatorsOutline &operator^=(HasOperatorsOutline &, HasOperatorsOutline &);
bool &operator&&(HasOperatorsOutline &, HasOperatorsOutline &);
bool &operator||(HasOperatorsOutline &, HasOperatorsOutline &);
// For min/max
HasOperatorsOutline &operator<(HasOperatorsOutline &, HasOperatorsOutline &);

// CHECK: acc.reduction.recipe @reduction_lor__ZTSA5_19HasOperatorsOutline : !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>> reduction_operator <lor> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!rec_HasOperatorsOutline x 5>, !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[TEMP_ITR:.*]] = cir.alloca !cir.ptr<!rec_HasOperatorsOutline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>, ["arrayinit.temp"]
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[ALLOCA]] : !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>>), !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: cir.store {{.*}} %[[DECAY]], %[[TEMP_ITR]] : !cir.ptr<!rec_HasOperatorsOutline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>
// CHECK-NEXT: %[[LAST_IDX:.*]] = cir.const #cir.int<5> : !s64i
// CHECK-NEXT: %[[END_ITR:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!rec_HasOperatorsOutline>, %[[LAST_IDX]] : !s64i), !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load {{.*}} %[[TEMP_ITR]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>, !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[TEMP_LOAD]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[TEMP_LOAD]][1] {name = "f"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[TEMP_LOAD]][2] {name = "d"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s64i
// CHECK-NEXT: %[[NEXT_ITEM:.*]] = cir.ptr_stride(%[[TEMP_LOAD]] : !cir.ptr<!rec_HasOperatorsOutline>, %[[ONE]] : !s64i), !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: cir.store {{.*}} %[[NEXT_ITEM]], %[[TEMP_ITR]] : !cir.ptr<!rec_HasOperatorsOutline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load {{.*}} %[[TEMP_ITR]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>, !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(ne, %[[TEMP_LOAD]], %[[END_ITR]]) : !cir.ptr<!rec_HasOperatorsOutline>, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]]) 
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>> {{.*}}):  
// CHECK-NEXT: %[[SIZE:.*]] = cir.const #cir.int<4>  : !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[ARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>>), !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: %[[LAST_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!rec_HasOperatorsOutline>, %[[SIZE]] : !u64i), !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: %[[IDX:.*]] = cir.alloca !cir.ptr<!rec_HasOperatorsOutline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>, ["__array_idx"] {alignment = 1 : i64}
// CHECK-NEXT: cir.store %[[LAST_ELT]], %[[IDX]] : !cir.ptr<!rec_HasOperatorsOutline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[CUR:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>, !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: cir.call @_ZN19HasOperatorsOutlineD1Ev(%[[CUR]]) nothrow : (!cir.ptr<!rec_HasOperatorsOutline>)
// CHECK-NEXT: %[[NEG:.*]] = cir.const #cir.int<-1> : !s64i
// CHECK-NEXT: %[[NEW_ITEM:.*]] = cir.ptr_stride(%[[CUR]] : !cir.ptr<!rec_HasOperatorsOutline>, %[[NEG]] : !s64i), !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: cir.store %[[NEW_ITEM]], %[[IDX]] : !cir.ptr<!rec_HasOperatorsOutline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[CUR_LOAD:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>, !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(ne, %[[CUR_LOAD]], %[[DECAY]]) : !cir.ptr<!rec_HasOperatorsOutline>, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]])
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_land__ZTSA5_19HasOperatorsOutline : !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>> reduction_operator <land> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!rec_HasOperatorsOutline x 5>, !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[ALLOCA]] : !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>>), !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[DECAY]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[DECAY]][1] {name = "f"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[DECAY]][2] {name = "d"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
//
// CHECK-NEXT: %[[ONE_IDX:.*]] = cir.const #cir.int<1> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!rec_HasOperatorsOutline>, %[[ONE_IDX]] : !s64i), !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[NEXT_ELT]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[NEXT_ELT]][1] {name = "f"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[NEXT_ELT]][2] {name = "d"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
//
// CHECK-NEXT: %[[TWO_IDX:.*]] = cir.const #cir.int<2> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!rec_HasOperatorsOutline>, %[[TWO_IDX]] : !s64i), !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[NEXT_ELT]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[NEXT_ELT]][1] {name = "f"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[NEXT_ELT]][2] {name = "d"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
//
// CHECK-NEXT: %[[THREE_IDX:.*]] = cir.const #cir.int<3> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!rec_HasOperatorsOutline>, %[[THREE_IDX]] : !s64i), !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[NEXT_ELT]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[NEXT_ELT]][1] {name = "f"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[NEXT_ELT]][2] {name = "d"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
//
// CHECK-NEXT: %[[FOUR_IDX:.*]] = cir.const #cir.int<4> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!rec_HasOperatorsOutline>, %[[FOUR_IDX]] : !s64i), !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[NEXT_ELT]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[NEXT_ELT]][1] {name = "f"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[NEXT_ELT]][2] {name = "d"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
//
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>> {{.*}}):  
// CHECK-NEXT: %[[SIZE:.*]] = cir.const #cir.int<4>  : !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[ARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>>), !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: %[[LAST_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!rec_HasOperatorsOutline>, %[[SIZE]] : !u64i), !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: %[[IDX:.*]] = cir.alloca !cir.ptr<!rec_HasOperatorsOutline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>, ["__array_idx"] {alignment = 1 : i64}
// CHECK-NEXT: cir.store %[[LAST_ELT]], %[[IDX]] : !cir.ptr<!rec_HasOperatorsOutline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[CUR:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>, !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: cir.call @_ZN19HasOperatorsOutlineD1Ev(%[[CUR]]) nothrow : (!cir.ptr<!rec_HasOperatorsOutline>)
// CHECK-NEXT: %[[NEG:.*]] = cir.const #cir.int<-1> : !s64i
// CHECK-NEXT: %[[NEW_ITEM:.*]] = cir.ptr_stride(%[[CUR]] : !cir.ptr<!rec_HasOperatorsOutline>, %[[NEG]] : !s64i), !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: cir.store %[[NEW_ITEM]], %[[IDX]] : !cir.ptr<!rec_HasOperatorsOutline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[CUR_LOAD:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>, !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(ne, %[[CUR_LOAD]], %[[DECAY]]) : !cir.ptr<!rec_HasOperatorsOutline>, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]])
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_xor__ZTSA5_19HasOperatorsOutline : !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>> reduction_operator <xor> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!rec_HasOperatorsOutline x 5>, !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[TEMP_ITR:.*]] = cir.alloca !cir.ptr<!rec_HasOperatorsOutline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>, ["arrayinit.temp"]
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[ALLOCA]] : !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>>), !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: cir.store {{.*}} %[[DECAY]], %[[TEMP_ITR]] : !cir.ptr<!rec_HasOperatorsOutline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>
// CHECK-NEXT: %[[LAST_IDX:.*]] = cir.const #cir.int<5> : !s64i
// CHECK-NEXT: %[[END_ITR:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!rec_HasOperatorsOutline>, %[[LAST_IDX]] : !s64i), !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load {{.*}} %[[TEMP_ITR]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>, !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[TEMP_LOAD]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[TEMP_LOAD]][1] {name = "f"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[TEMP_LOAD]][2] {name = "d"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s64i
// CHECK-NEXT: %[[NEXT_ITEM:.*]] = cir.ptr_stride(%[[TEMP_LOAD]] : !cir.ptr<!rec_HasOperatorsOutline>, %[[ONE]] : !s64i), !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: cir.store {{.*}} %[[NEXT_ITEM]], %[[TEMP_ITR]] : !cir.ptr<!rec_HasOperatorsOutline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load {{.*}} %[[TEMP_ITR]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>, !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(ne, %[[TEMP_LOAD]], %[[END_ITR]]) : !cir.ptr<!rec_HasOperatorsOutline>, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]]) 
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>> {{.*}}):  
// CHECK-NEXT: %[[SIZE:.*]] = cir.const #cir.int<4>  : !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[ARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>>), !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: %[[LAST_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!rec_HasOperatorsOutline>, %[[SIZE]] : !u64i), !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: %[[IDX:.*]] = cir.alloca !cir.ptr<!rec_HasOperatorsOutline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>, ["__array_idx"] {alignment = 1 : i64}
// CHECK-NEXT: cir.store %[[LAST_ELT]], %[[IDX]] : !cir.ptr<!rec_HasOperatorsOutline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[CUR:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>, !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: cir.call @_ZN19HasOperatorsOutlineD1Ev(%[[CUR]]) nothrow : (!cir.ptr<!rec_HasOperatorsOutline>)
// CHECK-NEXT: %[[NEG:.*]] = cir.const #cir.int<-1> : !s64i
// CHECK-NEXT: %[[NEW_ITEM:.*]] = cir.ptr_stride(%[[CUR]] : !cir.ptr<!rec_HasOperatorsOutline>, %[[NEG]] : !s64i), !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: cir.store %[[NEW_ITEM]], %[[IDX]] : !cir.ptr<!rec_HasOperatorsOutline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[CUR_LOAD:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>, !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(ne, %[[CUR_LOAD]], %[[DECAY]]) : !cir.ptr<!rec_HasOperatorsOutline>, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]])
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_ior__ZTSA5_19HasOperatorsOutline : !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>> reduction_operator <ior> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!rec_HasOperatorsOutline x 5>, !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[TEMP_ITR:.*]] = cir.alloca !cir.ptr<!rec_HasOperatorsOutline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>, ["arrayinit.temp"]
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[ALLOCA]] : !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>>), !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: cir.store {{.*}} %[[DECAY]], %[[TEMP_ITR]] : !cir.ptr<!rec_HasOperatorsOutline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>
// CHECK-NEXT: %[[LAST_IDX:.*]] = cir.const #cir.int<5> : !s64i
// CHECK-NEXT: %[[END_ITR:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!rec_HasOperatorsOutline>, %[[LAST_IDX]] : !s64i), !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load {{.*}} %[[TEMP_ITR]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>, !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[TEMP_LOAD]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[TEMP_LOAD]][1] {name = "f"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[TEMP_LOAD]][2] {name = "d"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s64i
// CHECK-NEXT: %[[NEXT_ITEM:.*]] = cir.ptr_stride(%[[TEMP_LOAD]] : !cir.ptr<!rec_HasOperatorsOutline>, %[[ONE]] : !s64i), !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: cir.store {{.*}} %[[NEXT_ITEM]], %[[TEMP_ITR]] : !cir.ptr<!rec_HasOperatorsOutline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load {{.*}} %[[TEMP_ITR]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>, !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(ne, %[[TEMP_LOAD]], %[[END_ITR]]) : !cir.ptr<!rec_HasOperatorsOutline>, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]]) 
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>> {{.*}}):  
// CHECK-NEXT: %[[SIZE:.*]] = cir.const #cir.int<4>  : !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[ARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>>), !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: %[[LAST_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!rec_HasOperatorsOutline>, %[[SIZE]] : !u64i), !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: %[[IDX:.*]] = cir.alloca !cir.ptr<!rec_HasOperatorsOutline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>, ["__array_idx"] {alignment = 1 : i64}
// CHECK-NEXT: cir.store %[[LAST_ELT]], %[[IDX]] : !cir.ptr<!rec_HasOperatorsOutline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[CUR:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>, !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: cir.call @_ZN19HasOperatorsOutlineD1Ev(%[[CUR]]) nothrow : (!cir.ptr<!rec_HasOperatorsOutline>)
// CHECK-NEXT: %[[NEG:.*]] = cir.const #cir.int<-1> : !s64i
// CHECK-NEXT: %[[NEW_ITEM:.*]] = cir.ptr_stride(%[[CUR]] : !cir.ptr<!rec_HasOperatorsOutline>, %[[NEG]] : !s64i), !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: cir.store %[[NEW_ITEM]], %[[IDX]] : !cir.ptr<!rec_HasOperatorsOutline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[CUR_LOAD:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>, !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(ne, %[[CUR_LOAD]], %[[DECAY]]) : !cir.ptr<!rec_HasOperatorsOutline>, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]])
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_iand__ZTSA5_19HasOperatorsOutline : !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>> reduction_operator <iand> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>>{{.*}})
// CHECK-NEXT: cir.alloca !cir.array<!rec_HasOperatorsOutline x 5>, !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>>, ["openacc.reduction.init"]
// TODO OpenACC: Expecting an initialization to... SOME value here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>> {{.*}}):  
// CHECK-NEXT: %[[SIZE:.*]] = cir.const #cir.int<4>  : !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[ARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>>), !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: %[[LAST_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!rec_HasOperatorsOutline>, %[[SIZE]] : !u64i), !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: %[[IDX:.*]] = cir.alloca !cir.ptr<!rec_HasOperatorsOutline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>, ["__array_idx"] {alignment = 1 : i64}
// CHECK-NEXT: cir.store %[[LAST_ELT]], %[[IDX]] : !cir.ptr<!rec_HasOperatorsOutline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[CUR:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>, !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: cir.call @_ZN19HasOperatorsOutlineD1Ev(%[[CUR]]) nothrow : (!cir.ptr<!rec_HasOperatorsOutline>)
// CHECK-NEXT: %[[NEG:.*]] = cir.const #cir.int<-1> : !s64i
// CHECK-NEXT: %[[NEW_ITEM:.*]] = cir.ptr_stride(%[[CUR]] : !cir.ptr<!rec_HasOperatorsOutline>, %[[NEG]] : !s64i), !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: cir.store %[[NEW_ITEM]], %[[IDX]] : !cir.ptr<!rec_HasOperatorsOutline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[CUR_LOAD:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>, !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(ne, %[[CUR_LOAD]], %[[DECAY]]) : !cir.ptr<!rec_HasOperatorsOutline>, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]])
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_min__ZTSA5_19HasOperatorsOutline : !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>> reduction_operator <min> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>>{{.*}})
// CHECK-NEXT: cir.alloca !cir.array<!rec_HasOperatorsOutline x 5>, !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>>, ["openacc.reduction.init"]
// TODO OpenACC: Expecting an initialization to... SOME value here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>> {{.*}}):  
// CHECK-NEXT: %[[SIZE:.*]] = cir.const #cir.int<4>  : !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[ARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>>), !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: %[[LAST_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!rec_HasOperatorsOutline>, %[[SIZE]] : !u64i), !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: %[[IDX:.*]] = cir.alloca !cir.ptr<!rec_HasOperatorsOutline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>, ["__array_idx"] {alignment = 1 : i64}
// CHECK-NEXT: cir.store %[[LAST_ELT]], %[[IDX]] : !cir.ptr<!rec_HasOperatorsOutline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[CUR:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>, !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: cir.call @_ZN19HasOperatorsOutlineD1Ev(%[[CUR]]) nothrow : (!cir.ptr<!rec_HasOperatorsOutline>)
// CHECK-NEXT: %[[NEG:.*]] = cir.const #cir.int<-1> : !s64i
// CHECK-NEXT: %[[NEW_ITEM:.*]] = cir.ptr_stride(%[[CUR]] : !cir.ptr<!rec_HasOperatorsOutline>, %[[NEG]] : !s64i), !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: cir.store %[[NEW_ITEM]], %[[IDX]] : !cir.ptr<!rec_HasOperatorsOutline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[CUR_LOAD:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>, !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(ne, %[[CUR_LOAD]], %[[DECAY]]) : !cir.ptr<!rec_HasOperatorsOutline>, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]])
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_max__ZTSA5_19HasOperatorsOutline : !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>> reduction_operator <max> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>>{{.*}})
// CHECK-NEXT: cir.alloca !cir.array<!rec_HasOperatorsOutline x 5>, !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>>, ["openacc.reduction.init"]
// TODO OpenACC: Expecting an initialization to... SOME value here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>> {{.*}}):  
// CHECK-NEXT: %[[SIZE:.*]] = cir.const #cir.int<4>  : !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[ARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>>), !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: %[[LAST_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!rec_HasOperatorsOutline>, %[[SIZE]] : !u64i), !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: %[[IDX:.*]] = cir.alloca !cir.ptr<!rec_HasOperatorsOutline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>, ["__array_idx"] {alignment = 1 : i64}
// CHECK-NEXT: cir.store %[[LAST_ELT]], %[[IDX]] : !cir.ptr<!rec_HasOperatorsOutline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[CUR:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>, !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: cir.call @_ZN19HasOperatorsOutlineD1Ev(%[[CUR]]) nothrow : (!cir.ptr<!rec_HasOperatorsOutline>)
// CHECK-NEXT: %[[NEG:.*]] = cir.const #cir.int<-1> : !s64i
// CHECK-NEXT: %[[NEW_ITEM:.*]] = cir.ptr_stride(%[[CUR]] : !cir.ptr<!rec_HasOperatorsOutline>, %[[NEG]] : !s64i), !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: cir.store %[[NEW_ITEM]], %[[IDX]] : !cir.ptr<!rec_HasOperatorsOutline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[CUR_LOAD:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>, !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(ne, %[[CUR_LOAD]], %[[DECAY]]) : !cir.ptr<!rec_HasOperatorsOutline>, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]])
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_mul__ZTSA5_19HasOperatorsOutline : !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>> reduction_operator <mul> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!rec_HasOperatorsOutline x 5>, !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[ALLOCA]] : !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>>), !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[DECAY]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[DECAY]][1] {name = "f"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[DECAY]][2] {name = "d"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
//
// CHECK-NEXT: %[[ONE_IDX:.*]] = cir.const #cir.int<1> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!rec_HasOperatorsOutline>, %[[ONE_IDX]] : !s64i), !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[NEXT_ELT]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[NEXT_ELT]][1] {name = "f"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[NEXT_ELT]][2] {name = "d"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
//
// CHECK-NEXT: %[[TWO_IDX:.*]] = cir.const #cir.int<2> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!rec_HasOperatorsOutline>, %[[TWO_IDX]] : !s64i), !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[NEXT_ELT]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[NEXT_ELT]][1] {name = "f"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[NEXT_ELT]][2] {name = "d"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
//
// CHECK-NEXT: %[[THREE_IDX:.*]] = cir.const #cir.int<3> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!rec_HasOperatorsOutline>, %[[THREE_IDX]] : !s64i), !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[NEXT_ELT]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[NEXT_ELT]][1] {name = "f"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[NEXT_ELT]][2] {name = "d"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
//
// CHECK-NEXT: %[[FOUR_IDX:.*]] = cir.const #cir.int<4> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!rec_HasOperatorsOutline>, %[[FOUR_IDX]] : !s64i), !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[NEXT_ELT]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[NEXT_ELT]][1] {name = "f"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[NEXT_ELT]][2] {name = "d"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: acc.yield 
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>> {{.*}}):  
// CHECK-NEXT: %[[SIZE:.*]] = cir.const #cir.int<4>  : !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[ARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>>), !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: %[[LAST_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!rec_HasOperatorsOutline>, %[[SIZE]] : !u64i), !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: %[[IDX:.*]] = cir.alloca !cir.ptr<!rec_HasOperatorsOutline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>, ["__array_idx"] {alignment = 1 : i64}
// CHECK-NEXT: cir.store %[[LAST_ELT]], %[[IDX]] : !cir.ptr<!rec_HasOperatorsOutline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[CUR:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>, !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: cir.call @_ZN19HasOperatorsOutlineD1Ev(%[[CUR]]) nothrow : (!cir.ptr<!rec_HasOperatorsOutline>)
// CHECK-NEXT: %[[NEG:.*]] = cir.const #cir.int<-1> : !s64i
// CHECK-NEXT: %[[NEW_ITEM:.*]] = cir.ptr_stride(%[[CUR]] : !cir.ptr<!rec_HasOperatorsOutline>, %[[NEG]] : !s64i), !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: cir.store %[[NEW_ITEM]], %[[IDX]] : !cir.ptr<!rec_HasOperatorsOutline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[CUR_LOAD:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>, !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(ne, %[[CUR_LOAD]], %[[DECAY]]) : !cir.ptr<!rec_HasOperatorsOutline>, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]])
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_add__ZTSA5_19HasOperatorsOutline : !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>> reduction_operator <add> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!rec_HasOperatorsOutline x 5>, !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[TEMP_ITR:.*]] = cir.alloca !cir.ptr<!rec_HasOperatorsOutline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>, ["arrayinit.temp"]
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[ALLOCA]] : !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>>), !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: cir.store {{.*}} %[[DECAY]], %[[TEMP_ITR]] : !cir.ptr<!rec_HasOperatorsOutline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>
// CHECK-NEXT: %[[LAST_IDX:.*]] = cir.const #cir.int<5> : !s64i
// CHECK-NEXT: %[[END_ITR:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!rec_HasOperatorsOutline>, %[[LAST_IDX]] : !s64i), !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load {{.*}} %[[TEMP_ITR]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>, !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[TEMP_LOAD]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[TEMP_LOAD]][1] {name = "f"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[TEMP_LOAD]][2] {name = "d"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s64i
// CHECK-NEXT: %[[NEXT_ITEM:.*]] = cir.ptr_stride(%[[TEMP_LOAD]] : !cir.ptr<!rec_HasOperatorsOutline>, %[[ONE]] : !s64i), !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: cir.store {{.*}} %[[NEXT_ITEM]], %[[TEMP_ITR]] : !cir.ptr<!rec_HasOperatorsOutline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load {{.*}} %[[TEMP_ITR]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>, !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(ne, %[[TEMP_LOAD]], %[[END_ITR]]) : !cir.ptr<!rec_HasOperatorsOutline>, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]]) 
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>> {{.*}}):  
// CHECK-NEXT: %[[SIZE:.*]] = cir.const #cir.int<4>  : !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[ARG]] : !cir.ptr<!cir.array<!rec_HasOperatorsOutline x 5>>), !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: %[[LAST_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!rec_HasOperatorsOutline>, %[[SIZE]] : !u64i), !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: %[[IDX:.*]] = cir.alloca !cir.ptr<!rec_HasOperatorsOutline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>, ["__array_idx"] {alignment = 1 : i64}
// CHECK-NEXT: cir.store %[[LAST_ELT]], %[[IDX]] : !cir.ptr<!rec_HasOperatorsOutline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[CUR:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>, !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: cir.call @_ZN19HasOperatorsOutlineD1Ev(%[[CUR]]) nothrow : (!cir.ptr<!rec_HasOperatorsOutline>)
// CHECK-NEXT: %[[NEG:.*]] = cir.const #cir.int<-1> : !s64i
// CHECK-NEXT: %[[NEW_ITEM:.*]] = cir.ptr_stride(%[[CUR]] : !cir.ptr<!rec_HasOperatorsOutline>, %[[NEG]] : !s64i), !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: cir.store %[[NEW_ITEM]], %[[IDX]] : !cir.ptr<!rec_HasOperatorsOutline>, !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[CUR_LOAD:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_HasOperatorsOutline>>, !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(ne, %[[CUR_LOAD]], %[[DECAY]]) : !cir.ptr<!rec_HasOperatorsOutline>, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]])
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }


// CHECK-NEXT: acc.reduction.recipe @reduction_lor__ZTS19HasOperatorsOutline : !cir.ptr<!rec_HasOperatorsOutline> reduction_operator <lor> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_HasOperatorsOutline>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !rec_HasOperatorsOutline, !cir.ptr<!rec_HasOperatorsOutline>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[ALLOCA]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[ALLOCA]][1] {name = "f"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[ALLOCA]][2] {name = "d"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!rec_HasOperatorsOutline> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!rec_HasOperatorsOutline> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_HasOperatorsOutline> {{.*}}):  
// CHECK-NEXT: cir.call @_ZN19HasOperatorsOutlineD1Ev(%[[ARG]]) nothrow : (!cir.ptr<!rec_HasOperatorsOutline>)
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_land__ZTS19HasOperatorsOutline : !cir.ptr<!rec_HasOperatorsOutline> reduction_operator <land> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_HasOperatorsOutline>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !rec_HasOperatorsOutline, !cir.ptr<!rec_HasOperatorsOutline>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[ALLOCA]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[ALLOCA]][1] {name = "f"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[ALLOCA]][2] {name = "d"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!rec_HasOperatorsOutline> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!rec_HasOperatorsOutline> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_HasOperatorsOutline> {{.*}}):  
// CHECK-NEXT: cir.call @_ZN19HasOperatorsOutlineD1Ev(%[[ARG]]) nothrow : (!cir.ptr<!rec_HasOperatorsOutline>)
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_xor__ZTS19HasOperatorsOutline : !cir.ptr<!rec_HasOperatorsOutline> reduction_operator <xor> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_HasOperatorsOutline>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !rec_HasOperatorsOutline, !cir.ptr<!rec_HasOperatorsOutline>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[ALLOCA]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[ALLOCA]][1] {name = "f"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[ALLOCA]][2] {name = "d"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!rec_HasOperatorsOutline> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!rec_HasOperatorsOutline> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_HasOperatorsOutline> {{.*}}):  
// CHECK-NEXT: cir.call @_ZN19HasOperatorsOutlineD1Ev(%[[ARG]]) nothrow : (!cir.ptr<!rec_HasOperatorsOutline>)
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_ior__ZTS19HasOperatorsOutline : !cir.ptr<!rec_HasOperatorsOutline> reduction_operator <ior> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_HasOperatorsOutline>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !rec_HasOperatorsOutline, !cir.ptr<!rec_HasOperatorsOutline>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[ALLOCA]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[ALLOCA]][1] {name = "f"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[ALLOCA]][2] {name = "d"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!rec_HasOperatorsOutline> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!rec_HasOperatorsOutline> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_HasOperatorsOutline> {{.*}}):  
// CHECK-NEXT: cir.call @_ZN19HasOperatorsOutlineD1Ev(%[[ARG]]) nothrow : (!cir.ptr<!rec_HasOperatorsOutline>)
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_iand__ZTS19HasOperatorsOutline : !cir.ptr<!rec_HasOperatorsOutline> reduction_operator <iand> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_HasOperatorsOutline>{{.*}})
// CHECK-NEXT: cir.alloca !rec_HasOperatorsOutline, !cir.ptr<!rec_HasOperatorsOutline>, ["openacc.reduction.init"]
// TODO OpenACC: Expecting an initialization to... SOME value here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!rec_HasOperatorsOutline> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!rec_HasOperatorsOutline> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_HasOperatorsOutline> {{.*}}):  
// CHECK-NEXT: cir.call @_ZN19HasOperatorsOutlineD1Ev(%[[ARG]]) nothrow : (!cir.ptr<!rec_HasOperatorsOutline>)
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_min__ZTS19HasOperatorsOutline : !cir.ptr<!rec_HasOperatorsOutline> reduction_operator <min> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_HasOperatorsOutline>{{.*}})
// CHECK-NEXT: cir.alloca !rec_HasOperatorsOutline, !cir.ptr<!rec_HasOperatorsOutline>, ["openacc.reduction.init"]
// TODO OpenACC: Expecting an initialization to... SOME value here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!rec_HasOperatorsOutline> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!rec_HasOperatorsOutline> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_HasOperatorsOutline> {{.*}}):  
// CHECK-NEXT: cir.call @_ZN19HasOperatorsOutlineD1Ev(%[[ARG]]) nothrow : (!cir.ptr<!rec_HasOperatorsOutline>)
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_max__ZTS19HasOperatorsOutline : !cir.ptr<!rec_HasOperatorsOutline> reduction_operator <max> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_HasOperatorsOutline>{{.*}})
// CHECK-NEXT: cir.alloca !rec_HasOperatorsOutline, !cir.ptr<!rec_HasOperatorsOutline>, ["openacc.reduction.init"]
// TODO OpenACC: Expecting an initialization to... SOME value here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!rec_HasOperatorsOutline> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!rec_HasOperatorsOutline> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_HasOperatorsOutline> {{.*}}):  
// CHECK-NEXT: cir.call @_ZN19HasOperatorsOutlineD1Ev(%[[ARG]]) nothrow : (!cir.ptr<!rec_HasOperatorsOutline>)
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_mul__ZTS19HasOperatorsOutline : !cir.ptr<!rec_HasOperatorsOutline> reduction_operator <mul> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_HasOperatorsOutline>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !rec_HasOperatorsOutline, !cir.ptr<!rec_HasOperatorsOutline>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[ALLOCA]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[ALLOCA]][1] {name = "f"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[ALLOCA]][2] {name = "d"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!rec_HasOperatorsOutline> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!rec_HasOperatorsOutline> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_HasOperatorsOutline> {{.*}}):  
// CHECK-NEXT: cir.call @_ZN19HasOperatorsOutlineD1Ev(%[[ARG]]) nothrow : (!cir.ptr<!rec_HasOperatorsOutline>)
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_add__ZTS19HasOperatorsOutline : !cir.ptr<!rec_HasOperatorsOutline> reduction_operator <add> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_HasOperatorsOutline>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !rec_HasOperatorsOutline, !cir.ptr<!rec_HasOperatorsOutline>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[ALLOCA]][0] {name = "i"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[ALLOCA]][1] {name = "f"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[ALLOCA]][2] {name = "d"} : !cir.ptr<!rec_HasOperatorsOutline> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!rec_HasOperatorsOutline> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!rec_HasOperatorsOutline> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!rec_HasOperatorsOutline>
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_HasOperatorsOutline> {{.*}}):  
// CHECK-NEXT: cir.call @_ZN19HasOperatorsOutlineD1Ev(%[[ARG]]) nothrow : (!cir.ptr<!rec_HasOperatorsOutline>)
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }

template<typename T>
void acc_loop() {
  T someVar;
  T someVarArr[5];
#pragma acc loop reduction(+:someVar)
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(*:someVar)
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(max:someVar)
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(min:someVar)
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(&:someVar)
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(|:someVar)
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(^:someVar)
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(&&:someVar)
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(||:someVar)
  for(int i = 0; i < 5; ++i);

#pragma acc loop reduction(+:someVarArr)
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(*:someVarArr)
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(max:someVarArr)
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(min:someVarArr)
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(&:someVarArr)
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(|:someVarArr)
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(^:someVarArr)
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(&&:someVarArr)
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(||:someVarArr)
  for(int i = 0; i < 5; ++i);

#pragma acc loop reduction(+:someVarArr[2])
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(*:someVarArr[2])
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(max:someVarArr[2])
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(min:someVarArr[2])
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(&:someVarArr[2])
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(|:someVarArr[2])
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(^:someVarArr[2])
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(&&:someVarArr[2])
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(||:someVarArr[2])
  for(int i = 0; i < 5; ++i);

#pragma acc loop reduction(+:someVarArr[1:1])
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(*:someVarArr[1:1])
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(max:someVarArr[1:1])
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(min:someVarArr[1:1])
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(&:someVarArr[1:1])
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(|:someVarArr[1:1])
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(^:someVarArr[1:1])
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(&&:someVarArr[1:1])
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(||:someVarArr[1:1])
  for(int i = 0; i < 5; ++i);
}

void uses() {
  acc_loop<HasOperatorsOutline>();
}
