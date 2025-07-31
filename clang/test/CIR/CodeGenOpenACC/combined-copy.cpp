// RUN: %clang_cc1 -fopenacc -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir %s -o - | FileCheck %s

struct InnerStructTy {
  int Member[5];
};
struct StructTy {
  int scalarMember;
  int arrayMember[5];
  short twoDArrayMember[5][3];
  InnerStructTy iSTy;

void InlineFunc() {
  // CHECK: cir.func {{.*}}InlineFunc{{.*}}
  // CHECK-NEXT: %[[THIS:.*]] = cir.alloca !cir.ptr<!rec_StructTy>, !cir.ptr<!cir.ptr<!rec_StructTy>>, ["this", init]
  // CHECK-NEXT: cir.store %[[THIS_ARG:.*]], %[[THIS]] : !cir.ptr<!rec_StructTy>, !cir.ptr<!cir.ptr<!rec_StructTy>>
  // CHECK-NEXT: %[[THIS_LOAD:.*]] = cir.load %[[THIS]] : !cir.ptr<!cir.ptr<!rec_StructTy>>, !cir.ptr<!rec_StructTy>

#pragma acc parallel loop copy(scalarMember)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[GETSCALARMEM:.*]] = cir.get_member %[[THIS_LOAD]][0] {name = "scalarMember"} : !cir.ptr<!rec_StructTy> -> !cir.ptr<!s32i>
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[GETSCALARMEM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copy>, name = "this->scalarMember"}
  // CHECK-NEXT: acc.parallel combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!s32i>) to varPtr(%[[GETSCALARMEM]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copy>, name = "this->scalarMember"}

#pragma acc kernels loop copy(arrayMember[2])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
  // CHECK-NEXT: %[[TWO_CAST:.*]] = builtin.unrealized_conversion_cast %[[TWO]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS1:.*]] = acc.bounds lowerbound(%[[TWO_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[GETARRAYMEM:.*]] = cir.get_member %[[THIS_LOAD]][1] {name = "arrayMember"} : !cir.ptr<!rec_StructTy> -> !cir.ptr<!cir.array<!s32i x 5>>
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[GETARRAYMEM]] : !cir.ptr<!cir.array<!s32i x 5>>) bounds(%[[BOUNDS1]]) -> !cir.ptr<!cir.array<!s32i x 5>> {dataClause = #acc<data_clause acc_copy>, name = "this->arrayMember[2]"}
  // CHECK-NEXT: acc.kernels combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!s32i x 5>>) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!s32i x 5>>) bounds(%[[BOUNDS1]]) to varPtr(%[[GETARRAYMEM]] : !cir.ptr<!cir.array<!s32i x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "this->arrayMember[2]"}

#pragma acc kernels loop copy(twoDArrayMember[1][2])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
  // CHECK-NEXT: %[[TWO_CAST:.*]] = builtin.unrealized_conversion_cast %[[TWO]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS1:.*]] = acc.bounds lowerbound(%[[TWO_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS2:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[GET2DARRAYMEM:.*]] = cir.get_member %[[THIS_LOAD]][2] {name = "twoDArrayMember"} : !cir.ptr<!rec_StructTy> -> !cir.ptr<!cir.array<!cir.array<!s16i x 3> x 5>>
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[GET2DARRAYMEM]] : !cir.ptr<!cir.array<!cir.array<!s16i x 3> x 5>>) bounds(%[[BOUNDS1]], %[[BOUNDS2]]) -> !cir.ptr<!cir.array<!cir.array<!s16i x 3> x 5>> {dataClause = #acc<data_clause acc_copy>, name = "this->twoDArrayMember[1][2]"}
  // CHECK-NEXT: acc.kernels combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.array<!s16i x 3> x 5>>) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.array<!s16i x 3> x 5>>) bounds(%[[BOUNDS1]], %[[BOUNDS2]]) to varPtr(%[[GET2DARRAYMEM]] : !cir.ptr<!cir.array<!cir.array<!s16i x 3> x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "this->twoDArrayMember[1][2]"}

#pragma acc kernels loop copy(iSTy)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[GETSTRUCTMEM:.*]] = cir.get_member %[[THIS_LOAD]][3] {name = "iSTy"} : !cir.ptr<!rec_StructTy> -> !cir.ptr<!rec_InnerStructTy>
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[GETSTRUCTMEM]] : !cir.ptr<!rec_InnerStructTy>) -> !cir.ptr<!rec_InnerStructTy> {dataClause = #acc<data_clause acc_copy>, name = "this->iSTy"}
  // CHECK-NEXT: acc.kernels combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!rec_InnerStructTy>) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!rec_InnerStructTy>) to varPtr(%[[GETSTRUCTMEM]] : !cir.ptr<!rec_InnerStructTy>) {dataClause = #acc<data_clause acc_copy>, name = "this->iSTy"}

#pragma acc parallel loop copy(iSTy.Member)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[GETSTRUCTMEM:.*]] = cir.get_member %[[THIS_LOAD]][3] {name = "iSTy"} : !cir.ptr<!rec_StructTy> -> !cir.ptr<!rec_InnerStructTy>
  // CHECK-NEXT: %[[GETMEMOFSTRUCT:.*]] = cir.get_member %[[GETSTRUCTMEM]][0] {name = "Member"} : !cir.ptr<!rec_InnerStructTy> -> !cir.ptr<!cir.array<!s32i x 5>>
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[GETMEMOFSTRUCT]] : !cir.ptr<!cir.array<!s32i x 5>>) -> !cir.ptr<!cir.array<!s32i x 5>> {dataClause = #acc<data_clause acc_copy>, name = "this->iSTy.Member"}
  // CHECK-NEXT: acc.parallel combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!s32i x 5>>) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!s32i x 5>>) to varPtr(%[[GETMEMOFSTRUCT]] : !cir.ptr<!cir.array<!s32i x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "this->iSTy.Member"}

#pragma acc serial loop copy(iSTy.Member[1])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS1:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[GETSTRUCTMEM:.*]] = cir.get_member %[[THIS_LOAD]][3] {name = "iSTy"} : !cir.ptr<!rec_StructTy> -> !cir.ptr<!rec_InnerStructTy>
  // CHECK-NEXT: %[[GETMEMOFSTRUCT:.*]] = cir.get_member %[[GETSTRUCTMEM]][0] {name = "Member"} : !cir.ptr<!rec_InnerStructTy> -> !cir.ptr<!cir.array<!s32i x 5>>
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[GETMEMOFSTRUCT]] : !cir.ptr<!cir.array<!s32i x 5>>) bounds(%[[BOUNDS1]]) -> !cir.ptr<!cir.array<!s32i x 5>> {dataClause = #acc<data_clause acc_copy>, name = "this->iSTy.Member[1]"}
  // CHECK-NEXT: acc.serial combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!s32i x 5>>) {
  // CHECK-NEXT: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!s32i x 5>>) bounds(%[[BOUNDS1]]) to varPtr(%[[GETMEMOFSTRUCT]] : !cir.ptr<!cir.array<!s32i x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "this->iSTy.Member[1]"}

#pragma acc parallel loop copy(this->scalarMember)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[GETSCALARMEM:.*]] = cir.get_member %[[THIS_LOAD]][0] {name = "scalarMember"} : !cir.ptr<!rec_StructTy> -> !cir.ptr<!s32i>
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[GETSCALARMEM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copy>, name = "this->scalarMember"}
  // CHECK-NEXT: acc.parallel combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!s32i>) to varPtr(%[[GETSCALARMEM]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copy>, name = "this->scalarMember"}

#pragma acc kernels loop copy(this->arrayMember[2])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
  // CHECK-NEXT: %[[TWO_CAST:.*]] = builtin.unrealized_conversion_cast %[[TWO]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS1:.*]] = acc.bounds lowerbound(%[[TWO_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[GETARRAYMEM:.*]] = cir.get_member %[[THIS_LOAD]][1] {name = "arrayMember"} : !cir.ptr<!rec_StructTy> -> !cir.ptr<!cir.array<!s32i x 5>>
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[GETARRAYMEM]] : !cir.ptr<!cir.array<!s32i x 5>>) bounds(%[[BOUNDS1]]) -> !cir.ptr<!cir.array<!s32i x 5>> {dataClause = #acc<data_clause acc_copy>, name = "this->arrayMember[2]"}
  // CHECK-NEXT: acc.kernels combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!s32i x 5>>) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!s32i x 5>>) bounds(%[[BOUNDS1]]) to varPtr(%[[GETARRAYMEM]] : !cir.ptr<!cir.array<!s32i x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "this->arrayMember[2]"}
#pragma acc kernels loop copy(this->twoDArrayMember[1][2])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
  // CHECK-NEXT: %[[TWO_CAST:.*]] = builtin.unrealized_conversion_cast %[[TWO]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS1:.*]] = acc.bounds lowerbound(%[[TWO_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS2:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[GET2DARRAYMEM:.*]] = cir.get_member %[[THIS_LOAD]][2] {name = "twoDArrayMember"} : !cir.ptr<!rec_StructTy> -> !cir.ptr<!cir.array<!cir.array<!s16i x 3> x 5>>
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[GET2DARRAYMEM]] : !cir.ptr<!cir.array<!cir.array<!s16i x 3> x 5>>) bounds(%[[BOUNDS1]], %[[BOUNDS2]]) -> !cir.ptr<!cir.array<!cir.array<!s16i x 3> x 5>> {dataClause = #acc<data_clause acc_copy>, name = "this->twoDArrayMember[1][2]"}
  // CHECK-NEXT: acc.kernels combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.array<!s16i x 3> x 5>>) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.array<!s16i x 3> x 5>>) bounds(%[[BOUNDS1]], %[[BOUNDS2]]) to varPtr(%[[GET2DARRAYMEM]] : !cir.ptr<!cir.array<!cir.array<!s16i x 3> x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "this->twoDArrayMember[1][2]"}
 
#pragma acc kernels loop copy(this->iSTy)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[GETSTRUCTMEM:.*]] = cir.get_member %[[THIS_LOAD]][3] {name = "iSTy"} : !cir.ptr<!rec_StructTy> -> !cir.ptr<!rec_InnerStructTy>
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[GETSTRUCTMEM]] : !cir.ptr<!rec_InnerStructTy>) -> !cir.ptr<!rec_InnerStructTy> {dataClause = #acc<data_clause acc_copy>, name = "this->iSTy"}
  // CHECK-NEXT: acc.kernels combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!rec_InnerStructTy>) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!rec_InnerStructTy>) to varPtr(%[[GETSTRUCTMEM]] : !cir.ptr<!rec_InnerStructTy>) {dataClause = #acc<data_clause acc_copy>, name = "this->iSTy"}

#pragma acc parallel loop copy(this->iSTy.Member)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[GETSTRUCTMEM:.*]] = cir.get_member %[[THIS_LOAD]][3] {name = "iSTy"} : !cir.ptr<!rec_StructTy> -> !cir.ptr<!rec_InnerStructTy>
  // CHECK-NEXT: %[[GETMEMOFSTRUCT:.*]] = cir.get_member %[[GETSTRUCTMEM]][0] {name = "Member"} : !cir.ptr<!rec_InnerStructTy> -> !cir.ptr<!cir.array<!s32i x 5>>
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[GETMEMOFSTRUCT]] : !cir.ptr<!cir.array<!s32i x 5>>) -> !cir.ptr<!cir.array<!s32i x 5>> {dataClause = #acc<data_clause acc_copy>, name = "this->iSTy.Member"}
  // CHECK-NEXT: acc.parallel combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!s32i x 5>>) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!s32i x 5>>) to varPtr(%[[GETMEMOFSTRUCT]] : !cir.ptr<!cir.array<!s32i x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "this->iSTy.Member"}

#pragma acc serial loop copy(this->iSTy.Member[1])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS1:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[GETSTRUCTMEM:.*]] = cir.get_member %[[THIS_LOAD]][3] {name = "iSTy"} : !cir.ptr<!rec_StructTy> -> !cir.ptr<!rec_InnerStructTy>
  // CHECK-NEXT: %[[GETMEMOFSTRUCT:.*]] = cir.get_member %[[GETSTRUCTMEM]][0] {name = "Member"} : !cir.ptr<!rec_InnerStructTy> -> !cir.ptr<!cir.array<!s32i x 5>>
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[GETMEMOFSTRUCT]] : !cir.ptr<!cir.array<!s32i x 5>>) bounds(%[[BOUNDS1]]) -> !cir.ptr<!cir.array<!s32i x 5>> {dataClause = #acc<data_clause acc_copy>, name = "this->iSTy.Member[1]"}
  // CHECK-NEXT: acc.serial combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!s32i x 5>>) {
  // CHECK-NEXT: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!s32i x 5>>) bounds(%[[BOUNDS1]]) to varPtr(%[[GETMEMOFSTRUCT]] : !cir.ptr<!cir.array<!s32i x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "this->iSTy.Member[1]"}
}

void OutlineFunc();
};

void InlineUse() {
  StructTy s;
  s.InlineFunc();
}

void StructTy::OutlineFunc() {
  // CHECK: cir.func {{.*}}OutlineFunc{{.*}}
  // CHECK-NEXT: %[[THIS:.*]] = cir.alloca !cir.ptr<!rec_StructTy>, !cir.ptr<!cir.ptr<!rec_StructTy>>, ["this", init]
  // CHECK-NEXT: cir.store %[[THIS_ARG:.*]], %[[THIS]] : !cir.ptr<!rec_StructTy>, !cir.ptr<!cir.ptr<!rec_StructTy>>
  // CHECK-NEXT: %[[THIS_LOAD:.*]] = cir.load %[[THIS]] : !cir.ptr<!cir.ptr<!rec_StructTy>>, !cir.ptr<!rec_StructTy>
#pragma acc parallel loop copy(scalarMember)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[GETSCALARMEM:.*]] = cir.get_member %[[THIS_LOAD]][0] {name = "scalarMember"} : !cir.ptr<!rec_StructTy> -> !cir.ptr<!s32i>
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[GETSCALARMEM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copy>, name = "this->scalarMember"}
  // CHECK-NEXT: acc.parallel combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!s32i>) to varPtr(%[[GETSCALARMEM]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copy>, name = "this->scalarMember"}
#pragma acc kernels loop copy(arrayMember[2])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
  // CHECK-NEXT: %[[TWO_CAST:.*]] = builtin.unrealized_conversion_cast %[[TWO]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS1:.*]] = acc.bounds lowerbound(%[[TWO_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[GETARRAYMEM:.*]] = cir.get_member %[[THIS_LOAD]][1] {name = "arrayMember"} : !cir.ptr<!rec_StructTy> -> !cir.ptr<!cir.array<!s32i x 5>>
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[GETARRAYMEM]] : !cir.ptr<!cir.array<!s32i x 5>>) bounds(%[[BOUNDS1]]) -> !cir.ptr<!cir.array<!s32i x 5>> {dataClause = #acc<data_clause acc_copy>, name = "this->arrayMember[2]"}
  // CHECK-NEXT: acc.kernels combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!s32i x 5>>) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!s32i x 5>>) bounds(%[[BOUNDS1]]) to varPtr(%[[GETARRAYMEM]] : !cir.ptr<!cir.array<!s32i x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "this->arrayMember[2]"}
#pragma acc kernels loop copy(twoDArrayMember[1][2])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
  // CHECK-NEXT: %[[TWO_CAST:.*]] = builtin.unrealized_conversion_cast %[[TWO]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS1:.*]] = acc.bounds lowerbound(%[[TWO_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS2:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[GET2DARRAYMEM:.*]] = cir.get_member %[[THIS_LOAD]][2] {name = "twoDArrayMember"} : !cir.ptr<!rec_StructTy> -> !cir.ptr<!cir.array<!cir.array<!s16i x 3> x 5>>
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[GET2DARRAYMEM]] : !cir.ptr<!cir.array<!cir.array<!s16i x 3> x 5>>) bounds(%[[BOUNDS1]], %[[BOUNDS2]]) -> !cir.ptr<!cir.array<!cir.array<!s16i x 3> x 5>> {dataClause = #acc<data_clause acc_copy>, name = "this->twoDArrayMember[1][2]"}
  // CHECK-NEXT: acc.kernels combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.array<!s16i x 3> x 5>>) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.array<!s16i x 3> x 5>>) bounds(%[[BOUNDS1]], %[[BOUNDS2]]) to varPtr(%[[GET2DARRAYMEM]] : !cir.ptr<!cir.array<!cir.array<!s16i x 3> x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "this->twoDArrayMember[1][2]"}
#pragma acc kernels loop copy(iSTy)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[GETSTRUCTMEM:.*]] = cir.get_member %[[THIS_LOAD]][3] {name = "iSTy"} : !cir.ptr<!rec_StructTy> -> !cir.ptr<!rec_InnerStructTy>
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[GETSTRUCTMEM]] : !cir.ptr<!rec_InnerStructTy>) -> !cir.ptr<!rec_InnerStructTy> {dataClause = #acc<data_clause acc_copy>, name = "this->iSTy"}
  // CHECK-NEXT: acc.kernels combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!rec_InnerStructTy>) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!rec_InnerStructTy>) to varPtr(%[[GETSTRUCTMEM]] : !cir.ptr<!rec_InnerStructTy>) {dataClause = #acc<data_clause acc_copy>, name = "this->iSTy"}

#pragma acc parallel loop copy(iSTy.Member)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[GETSTRUCTMEM:.*]] = cir.get_member %[[THIS_LOAD]][3] {name = "iSTy"} : !cir.ptr<!rec_StructTy> -> !cir.ptr<!rec_InnerStructTy>
  // CHECK-NEXT: %[[GETMEMOFSTRUCT:.*]] = cir.get_member %[[GETSTRUCTMEM]][0] {name = "Member"} : !cir.ptr<!rec_InnerStructTy> -> !cir.ptr<!cir.array<!s32i x 5>>
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[GETMEMOFSTRUCT]] : !cir.ptr<!cir.array<!s32i x 5>>) -> !cir.ptr<!cir.array<!s32i x 5>> {dataClause = #acc<data_clause acc_copy>, name = "this->iSTy.Member"}
  // CHECK-NEXT: acc.parallel combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!s32i x 5>>) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!s32i x 5>>) to varPtr(%[[GETMEMOFSTRUCT]] : !cir.ptr<!cir.array<!s32i x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "this->iSTy.Member"}

#pragma acc serial loop copy(iSTy.Member[1])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS1:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[GETSTRUCTMEM:.*]] = cir.get_member %[[THIS_LOAD]][3] {name = "iSTy"} : !cir.ptr<!rec_StructTy> -> !cir.ptr<!rec_InnerStructTy>
  // CHECK-NEXT: %[[GETMEMOFSTRUCT:.*]] = cir.get_member %[[GETSTRUCTMEM]][0] {name = "Member"} : !cir.ptr<!rec_InnerStructTy> -> !cir.ptr<!cir.array<!s32i x 5>>
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[GETMEMOFSTRUCT]] : !cir.ptr<!cir.array<!s32i x 5>>) bounds(%[[BOUNDS1]]) -> !cir.ptr<!cir.array<!s32i x 5>> {dataClause = #acc<data_clause acc_copy>, name = "this->iSTy.Member[1]"}
  // CHECK-NEXT: acc.serial combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!s32i x 5>>) {
  // CHECK-NEXT: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!s32i x 5>>) bounds(%[[BOUNDS1]]) to varPtr(%[[GETMEMOFSTRUCT]] : !cir.ptr<!cir.array<!s32i x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "this->iSTy.Member[1]"}

#pragma acc parallel loop copy(this->scalarMember)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[GETSCALARMEM:.*]] = cir.get_member %[[THIS_LOAD]][0] {name = "scalarMember"} : !cir.ptr<!rec_StructTy> -> !cir.ptr<!s32i>
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[GETSCALARMEM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copy>, name = "this->scalarMember"}
  // CHECK-NEXT: acc.parallel combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!s32i>) to varPtr(%[[GETSCALARMEM]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copy>, name = "this->scalarMember"}
#pragma acc kernels loop copy(this->arrayMember[2])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
  // CHECK-NEXT: %[[TWO_CAST:.*]] = builtin.unrealized_conversion_cast %[[TWO]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS1:.*]] = acc.bounds lowerbound(%[[TWO_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[GETARRAYMEM:.*]] = cir.get_member %[[THIS_LOAD]][1] {name = "arrayMember"} : !cir.ptr<!rec_StructTy> -> !cir.ptr<!cir.array<!s32i x 5>>
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[GETARRAYMEM]] : !cir.ptr<!cir.array<!s32i x 5>>) bounds(%[[BOUNDS1]]) -> !cir.ptr<!cir.array<!s32i x 5>> {dataClause = #acc<data_clause acc_copy>, name = "this->arrayMember[2]"}
  // CHECK-NEXT: acc.kernels combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!s32i x 5>>) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!s32i x 5>>) bounds(%[[BOUNDS1]]) to varPtr(%[[GETARRAYMEM]] : !cir.ptr<!cir.array<!s32i x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "this->arrayMember[2]"}

#pragma acc kernels loop copy(this->twoDArrayMember[1][2])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
  // CHECK-NEXT: %[[TWO_CAST:.*]] = builtin.unrealized_conversion_cast %[[TWO]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS1:.*]] = acc.bounds lowerbound(%[[TWO_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS2:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[GET2DARRAYMEM:.*]] = cir.get_member %[[THIS_LOAD]][2] {name = "twoDArrayMember"} : !cir.ptr<!rec_StructTy> -> !cir.ptr<!cir.array<!cir.array<!s16i x 3> x 5>>
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[GET2DARRAYMEM]] : !cir.ptr<!cir.array<!cir.array<!s16i x 3> x 5>>) bounds(%[[BOUNDS1]], %[[BOUNDS2]]) -> !cir.ptr<!cir.array<!cir.array<!s16i x 3> x 5>> {dataClause = #acc<data_clause acc_copy>, name = "this->twoDArrayMember[1][2]"}
  // CHECK-NEXT: acc.kernels combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.array<!s16i x 3> x 5>>) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.array<!s16i x 3> x 5>>) bounds(%[[BOUNDS1]], %[[BOUNDS2]]) to varPtr(%[[GET2DARRAYMEM]] : !cir.ptr<!cir.array<!cir.array<!s16i x 3> x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "this->twoDArrayMember[1][2]"}

#pragma acc kernels loop copy(this->iSTy)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[GETSTRUCTMEM:.*]] = cir.get_member %[[THIS_LOAD]][3] {name = "iSTy"} : !cir.ptr<!rec_StructTy> -> !cir.ptr<!rec_InnerStructTy>
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[GETSTRUCTMEM]] : !cir.ptr<!rec_InnerStructTy>) -> !cir.ptr<!rec_InnerStructTy> {dataClause = #acc<data_clause acc_copy>, name = "this->iSTy"}
  // CHECK-NEXT: acc.kernels combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!rec_InnerStructTy>) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!rec_InnerStructTy>) to varPtr(%[[GETSTRUCTMEM]] : !cir.ptr<!rec_InnerStructTy>) {dataClause = #acc<data_clause acc_copy>, name = "this->iSTy"}

#pragma acc parallel loop copy(this->iSTy.Member)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[GETSTRUCTMEM:.*]] = cir.get_member %[[THIS_LOAD]][3] {name = "iSTy"} : !cir.ptr<!rec_StructTy> -> !cir.ptr<!rec_InnerStructTy>
  // CHECK-NEXT: %[[GETMEMOFSTRUCT:.*]] = cir.get_member %[[GETSTRUCTMEM]][0] {name = "Member"} : !cir.ptr<!rec_InnerStructTy> -> !cir.ptr<!cir.array<!s32i x 5>>
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[GETMEMOFSTRUCT]] : !cir.ptr<!cir.array<!s32i x 5>>) -> !cir.ptr<!cir.array<!s32i x 5>> {dataClause = #acc<data_clause acc_copy>, name = "this->iSTy.Member"}
  // CHECK-NEXT: acc.parallel combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!s32i x 5>>) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!s32i x 5>>) to varPtr(%[[GETMEMOFSTRUCT]] : !cir.ptr<!cir.array<!s32i x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "this->iSTy.Member"}

#pragma acc serial loop copy(this->iSTy.Member[1])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS1:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[GETSTRUCTMEM:.*]] = cir.get_member %[[THIS_LOAD]][3] {name = "iSTy"} : !cir.ptr<!rec_StructTy> -> !cir.ptr<!rec_InnerStructTy>
  // CHECK-NEXT: %[[GETMEMOFSTRUCT:.*]] = cir.get_member %[[GETSTRUCTMEM]][0] {name = "Member"} : !cir.ptr<!rec_InnerStructTy> -> !cir.ptr<!cir.array<!s32i x 5>>
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[GETMEMOFSTRUCT]] : !cir.ptr<!cir.array<!s32i x 5>>) bounds(%[[BOUNDS1]]) -> !cir.ptr<!cir.array<!s32i x 5>> {dataClause = #acc<data_clause acc_copy>, name = "this->iSTy.Member[1]"}
  // CHECK-NEXT: acc.serial combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!s32i x 5>>) {
  // CHECK-NEXT: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!s32i x 5>>) bounds(%[[BOUNDS1]]) to varPtr(%[[GETMEMOFSTRUCT]] : !cir.ptr<!cir.array<!s32i x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "this->iSTy.Member[1]"}
}
