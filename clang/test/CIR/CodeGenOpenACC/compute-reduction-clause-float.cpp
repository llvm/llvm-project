// RUN: %clang_cc1 -fopenacc -triple x86_64-linux-gnu -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir -triple x86_64-linux-pc %s -o - | FileCheck %s


// CHECK: acc.reduction.recipe @reduction_lor__ZTSA5_f : !cir.ptr<!cir.array<!cir.float x 5>> reduction_operator <lor> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!cir.float x 5>, !cir.ptr<!cir.array<!cir.float x 5>>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[TEMP_ITR:.*]] = cir.alloca !cir.ptr<!cir.float>, !cir.ptr<!cir.ptr<!cir.float>>, ["arrayinit.temp"]
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[ALLOCA]] : !cir.ptr<!cir.array<!cir.float x 5>>), !cir.ptr<!cir.float>
// CHECK-NEXT: cir.store {{.*}} %[[DECAY]], %[[TEMP_ITR]] : !cir.ptr<!cir.float>, !cir.ptr<!cir.ptr<!cir.float>>
// CHECK-NEXT: %[[LAST_IDX:.*]] = cir.const #cir.int<5> : !s64i
// CHECK-NEXT: %[[END_ITR:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!cir.float>, %[[LAST_IDX]] : !s64i), !cir.ptr<!cir.float>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load {{.*}} %[[TEMP_ITR]] : !cir.ptr<!cir.ptr<!cir.float>>, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[TEMP_LOAD]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s64i
// CHECK-NEXT: %[[NEXT_ITEM:.*]] = cir.ptr_stride(%[[TEMP_LOAD]] : !cir.ptr<!cir.float>, %[[ONE]] : !s64i), !cir.ptr<!cir.float>
// CHECK-NEXT: cir.store {{.*}} %[[NEXT_ITEM]], %[[TEMP_ITR]] : !cir.ptr<!cir.float>, !cir.ptr<!cir.ptr<!cir.float>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load {{.*}} %[[TEMP_ITR]] : !cir.ptr<!cir.ptr<!cir.float>>, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(ne, %[[TEMP_LOAD]], %[[END_ITR]]) : !cir.ptr<!cir.float>, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]]) 
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!cir.float x 5>>
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_land__ZTSA5_f : !cir.ptr<!cir.array<!cir.float x 5>> reduction_operator <land> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!cir.float x 5>, !cir.ptr<!cir.array<!cir.float x 5>>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[ALLOCA]] : !cir.ptr<!cir.array<!cir.float x 5>>), !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store{{.*}} %[[ONE]], %[[DECAY]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE_IDX:.*]] = cir.const #cir.int<1> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!cir.float>, %[[ONE_IDX]] : !s64i), !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store{{.*}} %[[ONE]], %[[NEXT_ELT]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[TWO_IDX:.*]] = cir.const #cir.int<2> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!cir.float>, %[[TWO_IDX]] : !s64i), !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store{{.*}} %[[ONE]], %[[NEXT_ELT]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[THREE_IDX:.*]] = cir.const #cir.int<3> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!cir.float>, %[[THREE_IDX]] : !s64i), !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store{{.*}} %[[ONE]], %[[NEXT_ELT]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[FOUR_IDX:.*]] = cir.const #cir.int<4> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!cir.float>, %[[FOUR_IDX]] : !s64i), !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store{{.*}} %[[ONE]], %[[NEXT_ELT]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!cir.float x 5>>
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_xor__ZTSA5_f : !cir.ptr<!cir.array<!cir.float x 5>> reduction_operator <xor> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!cir.float x 5>, !cir.ptr<!cir.array<!cir.float x 5>>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[TEMP_ITR:.*]] = cir.alloca !cir.ptr<!cir.float>, !cir.ptr<!cir.ptr<!cir.float>>, ["arrayinit.temp"]
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[ALLOCA]] : !cir.ptr<!cir.array<!cir.float x 5>>), !cir.ptr<!cir.float>
// CHECK-NEXT: cir.store {{.*}} %[[DECAY]], %[[TEMP_ITR]] : !cir.ptr<!cir.float>, !cir.ptr<!cir.ptr<!cir.float>>
// CHECK-NEXT: %[[LAST_IDX:.*]] = cir.const #cir.int<5> : !s64i
// CHECK-NEXT: %[[END_ITR:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!cir.float>, %[[LAST_IDX]] : !s64i), !cir.ptr<!cir.float>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load {{.*}} %[[TEMP_ITR]] : !cir.ptr<!cir.ptr<!cir.float>>, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[TEMP_LOAD]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s64i
// CHECK-NEXT: %[[NEXT_ITEM:.*]] = cir.ptr_stride(%[[TEMP_LOAD]] : !cir.ptr<!cir.float>, %[[ONE]] : !s64i), !cir.ptr<!cir.float>
// CHECK-NEXT: cir.store {{.*}} %[[NEXT_ITEM]], %[[TEMP_ITR]] : !cir.ptr<!cir.float>, !cir.ptr<!cir.ptr<!cir.float>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load {{.*}} %[[TEMP_ITR]] : !cir.ptr<!cir.ptr<!cir.float>>, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(ne, %[[TEMP_LOAD]], %[[END_ITR]]) : !cir.ptr<!cir.float>, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]]) 
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!cir.float x 5>>
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_ior__ZTSA5_f : !cir.ptr<!cir.array<!cir.float x 5>> reduction_operator <ior> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!cir.float x 5>, !cir.ptr<!cir.array<!cir.float x 5>>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[TEMP_ITR:.*]] = cir.alloca !cir.ptr<!cir.float>, !cir.ptr<!cir.ptr<!cir.float>>, ["arrayinit.temp"]
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[ALLOCA]] : !cir.ptr<!cir.array<!cir.float x 5>>), !cir.ptr<!cir.float>
// CHECK-NEXT: cir.store {{.*}} %[[DECAY]], %[[TEMP_ITR]] : !cir.ptr<!cir.float>, !cir.ptr<!cir.ptr<!cir.float>>
// CHECK-NEXT: %[[LAST_IDX:.*]] = cir.const #cir.int<5> : !s64i
// CHECK-NEXT: %[[END_ITR:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!cir.float>, %[[LAST_IDX]] : !s64i), !cir.ptr<!cir.float>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load {{.*}} %[[TEMP_ITR]] : !cir.ptr<!cir.ptr<!cir.float>>, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[TEMP_LOAD]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s64i
// CHECK-NEXT: %[[NEXT_ITEM:.*]] = cir.ptr_stride(%[[TEMP_LOAD]] : !cir.ptr<!cir.float>, %[[ONE]] : !s64i), !cir.ptr<!cir.float>
// CHECK-NEXT: cir.store {{.*}} %[[NEXT_ITEM]], %[[TEMP_ITR]] : !cir.ptr<!cir.float>, !cir.ptr<!cir.ptr<!cir.float>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load {{.*}} %[[TEMP_ITR]] : !cir.ptr<!cir.ptr<!cir.float>>, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(ne, %[[TEMP_LOAD]], %[[END_ITR]]) : !cir.ptr<!cir.float>, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]]) 
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!cir.float x 5>>
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_iand__ZTSA5_f : !cir.ptr<!cir.array<!cir.float x 5>> reduction_operator <iand> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!cir.float x 5>, !cir.ptr<!cir.array<!cir.float x 5>>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[ALLOCA]] : !cir.ptr<!cir.array<!cir.float x 5>>), !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #cir.fp<0xF{{.*}}> : !cir.float
// CHECK-NEXT: cir.store{{.*}} %[[ALL_ONES]], %[[DECAY]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE_IDX:.*]] = cir.const #cir.int<1> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!cir.float>, %[[ONE_IDX]] : !s64i), !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #cir.fp<0xF{{.*}}> : !cir.float
// CHECK-NEXT: cir.store{{.*}} %[[ALL_ONES]], %[[NEXT_ELT]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[TWO_IDX:.*]] = cir.const #cir.int<2> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!cir.float>, %[[TWO_IDX]] : !s64i), !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #cir.fp<0xF{{.*}}> : !cir.float
// CHECK-NEXT: cir.store{{.*}} %[[ALL_ONES]], %[[NEXT_ELT]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[THREE_IDX:.*]] = cir.const #cir.int<3> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!cir.float>, %[[THREE_IDX]] : !s64i), !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #cir.fp<0xF{{.*}}> : !cir.float
// CHECK-NEXT: cir.store{{.*}} %[[ALL_ONES]], %[[NEXT_ELT]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[FOUR_IDX:.*]] = cir.const #cir.int<4> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!cir.float>, %[[FOUR_IDX]] : !s64i), !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #cir.fp<0xF{{.*}}> : !cir.float
// CHECK-NEXT: cir.store{{.*}} %[[ALL_ONES]], %[[NEXT_ELT]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!cir.float x 5>>
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_min__ZTSA5_f : !cir.ptr<!cir.array<!cir.float x 5>> reduction_operator <min> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!cir.float x 5>, !cir.ptr<!cir.array<!cir.float x 5>>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[ALLOCA]] : !cir.ptr<!cir.array<!cir.float x 5>>), !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.fp<3.4{{.*}}E+38> : !cir.float
// CHECK-NEXT: cir.store{{.*}} %[[LARGEST]], %[[DECAY]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE_IDX:.*]] = cir.const #cir.int<1> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!cir.float>, %[[ONE_IDX]] : !s64i), !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.fp<3.4{{.*}}E+38> : !cir.float
// CHECK-NEXT: cir.store{{.*}} %[[LARGEST]], %[[NEXT_ELT]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[TWO_IDX:.*]] = cir.const #cir.int<2> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!cir.float>, %[[TWO_IDX]] : !s64i), !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.fp<3.4{{.*}}E+38> : !cir.float
// CHECK-NEXT: cir.store{{.*}} %[[LARGEST]], %[[NEXT_ELT]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[THREE_IDX:.*]] = cir.const #cir.int<3> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!cir.float>, %[[THREE_IDX]] : !s64i), !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.fp<3.4{{.*}}E+38> : !cir.float
// CHECK-NEXT: cir.store{{.*}} %[[LARGEST]], %[[NEXT_ELT]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[FOUR_IDX:.*]] = cir.const #cir.int<4> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!cir.float>, %[[FOUR_IDX]] : !s64i), !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.fp<3.4{{.*}}E+38> : !cir.float
// CHECK-NEXT: cir.store{{.*}} %[[LARGEST]], %[[NEXT_ELT]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!cir.float x 5>>
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_max__ZTSA5_f : !cir.ptr<!cir.array<!cir.float x 5>> reduction_operator <max> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!cir.float x 5>, !cir.ptr<!cir.array<!cir.float x 5>>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[ALLOCA]] : !cir.ptr<!cir.array<!cir.float x 5>>), !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.fp<-3.4{{.*}}E+38> : !cir.float
// CHECK-NEXT: cir.store{{.*}} %[[LEAST]], %[[DECAY]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE_IDX:.*]] = cir.const #cir.int<1> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!cir.float>, %[[ONE_IDX]] : !s64i), !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.fp<-3.4{{.*}}E+38> : !cir.float
// CHECK-NEXT: cir.store{{.*}} %[[LEAST]], %[[NEXT_ELT]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[TWO_IDX:.*]] = cir.const #cir.int<2> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!cir.float>, %[[TWO_IDX]] : !s64i), !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.fp<-3.4{{.*}}E+38> : !cir.float
// CHECK-NEXT: cir.store{{.*}} %[[LEAST]], %[[NEXT_ELT]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[THREE_IDX:.*]] = cir.const #cir.int<3> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!cir.float>, %[[THREE_IDX]] : !s64i), !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.fp<-3.4{{.*}}E+38> : !cir.float
// CHECK-NEXT: cir.store{{.*}} %[[LEAST]], %[[NEXT_ELT]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[FOUR_IDX:.*]] = cir.const #cir.int<4> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!cir.float>, %[[FOUR_IDX]] : !s64i), !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.fp<-3.4{{.*}}E+38> : !cir.float
// CHECK-NEXT: cir.store{{.*}} %[[LEAST]], %[[NEXT_ELT]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!cir.float x 5>>
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_mul__ZTSA5_f : !cir.ptr<!cir.array<!cir.float x 5>> reduction_operator <mul> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!cir.float x 5>, !cir.ptr<!cir.array<!cir.float x 5>>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[ALLOCA]] : !cir.ptr<!cir.array<!cir.float x 5>>), !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store{{.*}} %[[ONE]], %[[DECAY]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE_IDX:.*]] = cir.const #cir.int<1> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!cir.float>, %[[ONE_IDX]] : !s64i), !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store{{.*}} %[[ONE]], %[[NEXT_ELT]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[TWO_IDX:.*]] = cir.const #cir.int<2> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!cir.float>, %[[TWO_IDX]] : !s64i), !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store{{.*}} %[[ONE]], %[[NEXT_ELT]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[THREE_IDX:.*]] = cir.const #cir.int<3> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!cir.float>, %[[THREE_IDX]] : !s64i), !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store{{.*}} %[[ONE]], %[[NEXT_ELT]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[FOUR_IDX:.*]] = cir.const #cir.int<4> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!cir.float>, %[[FOUR_IDX]] : !s64i), !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store{{.*}} %[[ONE]], %[[NEXT_ELT]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!cir.float x 5>>
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_add__ZTSA5_f : !cir.ptr<!cir.array<!cir.float x 5>> reduction_operator <add> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!cir.float x 5>, !cir.ptr<!cir.array<!cir.float x 5>>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[TEMP_ITR:.*]] = cir.alloca !cir.ptr<!cir.float>, !cir.ptr<!cir.ptr<!cir.float>>, ["arrayinit.temp"]
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[ALLOCA]] : !cir.ptr<!cir.array<!cir.float x 5>>), !cir.ptr<!cir.float>
// CHECK-NEXT: cir.store {{.*}} %[[DECAY]], %[[TEMP_ITR]] : !cir.ptr<!cir.float>, !cir.ptr<!cir.ptr<!cir.float>>
// CHECK-NEXT: %[[LAST_IDX:.*]] = cir.const #cir.int<5> : !s64i
// CHECK-NEXT: %[[END_ITR:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!cir.float>, %[[LAST_IDX]] : !s64i), !cir.ptr<!cir.float>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load {{.*}} %[[TEMP_ITR]] : !cir.ptr<!cir.ptr<!cir.float>>, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[TEMP_LOAD]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s64i
// CHECK-NEXT: %[[NEXT_ITEM:.*]] = cir.ptr_stride(%[[TEMP_LOAD]] : !cir.ptr<!cir.float>, %[[ONE]] : !s64i), !cir.ptr<!cir.float>
// CHECK-NEXT: cir.store {{.*}} %[[NEXT_ITEM]], %[[TEMP_ITR]] : !cir.ptr<!cir.float>, !cir.ptr<!cir.ptr<!cir.float>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load {{.*}} %[[TEMP_ITR]] : !cir.ptr<!cir.ptr<!cir.float>>, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(ne, %[[TEMP_LOAD]], %[[END_ITR]]) : !cir.ptr<!cir.float>, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]]) 
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!cir.float x 5>>
// CHECK-NEXT: }


// CHECK-NEXT: acc.reduction.recipe @reduction_lor__ZTSf : !cir.ptr<!cir.float> reduction_operator <lor> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.float>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[ALLOCA]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.float> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.float> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.float>
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_land__ZTSf : !cir.ptr<!cir.float> reduction_operator <land> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.float>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store{{.*}} %[[ONE]], %[[ALLOCA]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.float> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.float> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.float>
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_xor__ZTSf : !cir.ptr<!cir.float> reduction_operator <xor> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.float>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[ALLOCA]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.float> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.float> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.float>
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_ior__ZTSf : !cir.ptr<!cir.float> reduction_operator <ior> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.float>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[ALLOCA]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.float> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.float> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.float>
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_iand__ZTSf : !cir.ptr<!cir.float> reduction_operator <iand> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.float>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #cir.fp<0xF{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[ALLOCA]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.float> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.float> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.float>
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_min__ZTSf : !cir.ptr<!cir.float> reduction_operator <min> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.float>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.fp<3.4{{.*}}E+38> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[ALLOCA]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.float> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.float> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.float>
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_max__ZTSf : !cir.ptr<!cir.float> reduction_operator <max> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.float>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.fp<-3.4{{.*}}E+38> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[ALLOCA]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.float> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.float> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.float>
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_mul__ZTSf : !cir.ptr<!cir.float> reduction_operator <mul> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.float>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store{{.*}} %[[ONE]], %[[ALLOCA]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.float> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.float> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.float>
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_add__ZTSf : !cir.ptr<!cir.float> reduction_operator <add> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.float>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[ALLOCA]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.float> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.float> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.float>
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
  acc_compute<float>();
}
