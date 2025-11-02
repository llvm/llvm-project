// RUN: %clang_cc1 -fopenacc -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir %s -o - | FileCheck %s

int global;
void acc_compute(int parmVar) {
  // CHECK: cir.func{{.*}} @acc_compute(%[[ARG:.*]]: !s32i{{.*}})
  // CHECK-NEXT: %[[PARM:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["parmVar", init]
  int localVar1;
  short localVar2;
  float localVar3;
  // CHECK-NEXT: %[[LOCAL1:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["localVar1"]
  // CHECK-NEXT: %[[LOCAL2:.*]] = cir.alloca !s16i, !cir.ptr<!s16i>, ["localVar2"]
  // CHECK-NEXT: %[[LOCAL3:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["localVar3"] 
  // CHECK-NEXT: %[[LOCALPTR:.*]] = cir.alloca !cir.ptr<!s16i>, !cir.ptr<!cir.ptr<!s16i>>, ["localPointer"]
  // CHECK-NEXT: %[[LOCALARRAY:.*]] = cir.alloca !cir.array<!cir.float x 5>, !cir.ptr<!cir.array<!cir.float x 5>>, ["localArray"]
  // CHECK-NEXT: %[[LOCALARRAYOFPTRS:.*]] = cir.alloca !cir.array<!cir.ptr<!cir.float> x 5>, !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>, ["localArrayOfPtrs"]
  // CHECK-NEXT: %[[THREEDARRAY:.*]] = cir.alloca !cir.array<!cir.array<!cir.array<!cir.double x 7> x 6> x 5>, !cir.ptr<!cir.array<!cir.array<!cir.array<!cir.double x 7> x 6> x 5>>, ["threeDArray"]
  // CHECK-NEXT: cir.store %[[ARG]], %[[PARM]] : !s32i, !cir.ptr<!s32i>

#pragma acc parallel loop copy(localVar1, parmVar) copy(localVar2) copy(localVar3, parmVar)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCAL1]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copy>, name = "localVar1"} loc
  // CHECK-NEXT: %[[COPYIN2:.*]] = acc.copyin varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copy>, name = "parmVar"} loc
  // CHECK-NEXT: %[[COPYIN3:.*]] = acc.copyin varPtr(%[[LOCAL2]] : !cir.ptr<!s16i>) -> !cir.ptr<!s16i> {dataClause = #acc<data_clause acc_copy>, name = "localVar2"} loc
  // CHECK-NEXT: %[[COPYIN4:.*]] = acc.copyin varPtr(%[[LOCAL3]] : !cir.ptr<!cir.float>) -> !cir.ptr<!cir.float> {dataClause = #acc<data_clause acc_copy>, name = "localVar3"} loc
  // CHECK-NEXT: %[[COPYIN5:.*]] = acc.copyin varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copy>, name = "parmVar"} loc
  // CHECK-NEXT: acc.parallel combined(loop) dataOperands(%[[COPYIN1]], %[[COPYIN2]], %[[COPYIN3]], %[[COPYIN4]], %[[COPYIN5]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>, !cir.ptr<!s16i>, !cir.ptr<!cir.float>, !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN5]] : !cir.ptr<!s32i>) to varPtr(%[[PARM]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copy>, name = "parmVar"} loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN4]] : !cir.ptr<!cir.float>) to varPtr(%[[LOCAL3]] : !cir.ptr<!cir.float>) {dataClause = #acc<data_clause acc_copy>, name = "localVar3"} loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN3]] : !cir.ptr<!s16i>) to varPtr(%[[LOCAL2]] : !cir.ptr<!s16i>) {dataClause = #acc<data_clause acc_copy>, name = "localVar2"} loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN2]] : !cir.ptr<!s32i>) to varPtr(%[[PARM]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copy>, name = "parmVar"} loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!s32i>) to varPtr(%[[LOCAL1]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copy>, name = "localVar1"} loc

#pragma acc serial loop copy(localVar1, parmVar) copy(localVar2) copy(localVar3, parmVar)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCAL1]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copy>, name = "localVar1"} loc
  // CHECK-NEXT: %[[COPYIN2:.*]] = acc.copyin varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copy>, name = "parmVar"} loc
  // CHECK-NEXT: %[[COPYIN3:.*]] = acc.copyin varPtr(%[[LOCAL2]] : !cir.ptr<!s16i>) -> !cir.ptr<!s16i> {dataClause = #acc<data_clause acc_copy>, name = "localVar2"} loc
  // CHECK-NEXT: %[[COPYIN4:.*]] = acc.copyin varPtr(%[[LOCAL3]] : !cir.ptr<!cir.float>) -> !cir.ptr<!cir.float> {dataClause = #acc<data_clause acc_copy>, name = "localVar3"} loc
  // CHECK-NEXT: %[[COPYIN5:.*]] = acc.copyin varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copy>, name = "parmVar"} loc
  // CHECK-NEXT: acc.serial combined(loop) dataOperands(%[[COPYIN1]], %[[COPYIN2]], %[[COPYIN3]], %[[COPYIN4]], %[[COPYIN5]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>, !cir.ptr<!s16i>, !cir.ptr<!cir.float>, !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN5]] : !cir.ptr<!s32i>) to varPtr(%[[PARM]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copy>, name = "parmVar"} loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN4]] : !cir.ptr<!cir.float>) to varPtr(%[[LOCAL3]] : !cir.ptr<!cir.float>) {dataClause = #acc<data_clause acc_copy>, name = "localVar3"} loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN3]] : !cir.ptr<!s16i>) to varPtr(%[[LOCAL2]] : !cir.ptr<!s16i>) {dataClause = #acc<data_clause acc_copy>, name = "localVar2"} loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN2]] : !cir.ptr<!s32i>) to varPtr(%[[PARM]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copy>, name = "parmVar"} loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!s32i>) to varPtr(%[[LOCAL1]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copy>, name = "localVar1"} loc

#pragma acc kernels loop copy(localVar1, parmVar) copy(localVar2) copy(localVar3, parmVar)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCAL1]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copy>, name = "localVar1"} loc
  // CHECK-NEXT: %[[COPYIN2:.*]] = acc.copyin varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copy>, name = "parmVar"} loc
  // CHECK-NEXT: %[[COPYIN3:.*]] = acc.copyin varPtr(%[[LOCAL2]] : !cir.ptr<!s16i>) -> !cir.ptr<!s16i> {dataClause = #acc<data_clause acc_copy>, name = "localVar2"} loc
  // CHECK-NEXT: %[[COPYIN4:.*]] = acc.copyin varPtr(%[[LOCAL3]] : !cir.ptr<!cir.float>) -> !cir.ptr<!cir.float> {dataClause = #acc<data_clause acc_copy>, name = "localVar3"} loc
  // CHECK-NEXT: %[[COPYIN5:.*]] = acc.copyin varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copy>, name = "parmVar"} loc
  // CHECK-NEXT: acc.kernels combined(loop) dataOperands(%[[COPYIN1]], %[[COPYIN2]], %[[COPYIN3]], %[[COPYIN4]], %[[COPYIN5]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>, !cir.ptr<!s16i>, !cir.ptr<!cir.float>, !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN5]] : !cir.ptr<!s32i>) to varPtr(%[[PARM]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copy>, name = "parmVar"} loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN4]] : !cir.ptr<!cir.float>) to varPtr(%[[LOCAL3]] : !cir.ptr<!cir.float>) {dataClause = #acc<data_clause acc_copy>, name = "localVar3"} loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN3]] : !cir.ptr<!s16i>) to varPtr(%[[LOCAL2]] : !cir.ptr<!s16i>) {dataClause = #acc<data_clause acc_copy>, name = "localVar2"} loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN2]] : !cir.ptr<!s32i>) to varPtr(%[[PARM]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copy>, name = "parmVar"} loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!s32i>) to varPtr(%[[LOCAL1]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copy>, name = "localVar1"} loc

  // TODO: OpenACC: Represent alwaysin/alwaysout/always correctly. For now,
  // these do nothing to the IR.
#pragma acc parallel loop copy(alwaysin: localVar1) copy(alwaysout: localVar2) copy(always: localVar3)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCAL1]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copy>, modifiers = #acc<data_clause_modifier alwaysin>, name = "localVar1"} loc
  // CHECK-NEXT: %[[COPYIN2:.*]] = acc.copyin varPtr(%[[LOCAL2]] : !cir.ptr<!s16i>) -> !cir.ptr<!s16i> {dataClause = #acc<data_clause acc_copy>, modifiers = #acc<data_clause_modifier alwaysout>, name = "localVar2"} loc
  // CHECK-NEXT: %[[COPYIN3:.*]] = acc.copyin varPtr(%[[LOCAL3]] : !cir.ptr<!cir.float>) -> !cir.ptr<!cir.float> {dataClause = #acc<data_clause acc_copy>, modifiers = #acc<data_clause_modifier always>, name = "localVar3"} loc
  // CHECK-NEXT: acc.parallel combined(loop) dataOperands(%[[COPYIN1]], %[[COPYIN2]], %[[COPYIN3]] : !cir.ptr<!s32i>, !cir.ptr<!s16i>, !cir.ptr<!cir.float>) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN3]] : !cir.ptr<!cir.float>) to varPtr(%[[LOCAL3]] : !cir.ptr<!cir.float>) {dataClause = #acc<data_clause acc_copy>, modifiers = #acc<data_clause_modifier always>,  name = "localVar3"} loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN2]] : !cir.ptr<!s16i>) to varPtr(%[[LOCAL2]] : !cir.ptr<!s16i>) {dataClause = #acc<data_clause acc_copy>, modifiers = #acc<data_clause_modifier alwaysout>, name = "localVar2"} loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!s32i>) to varPtr(%[[LOCAL1]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copy>, modifiers = #acc<data_clause_modifier alwaysin>, name = "localVar1"} loc

#pragma acc serial loop copy(always, alwaysin, alwaysout: localVar1)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCAL1]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copy>, modifiers = #acc<data_clause_modifier always>, name = "localVar1"} loc
  // CHECK-NEXT: acc.serial combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!s32i>) to varPtr(%[[LOCAL1]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copy>, modifiers = #acc<data_clause_modifier always>, name = "localVar1"} loc

  short *localPointer;
  float localArray[5];

#pragma acc kernels loop copy(localArray, localPointer, global)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCALARRAY]] : !cir.ptr<!cir.array<!cir.float x 5>>) -> !cir.ptr<!cir.array<!cir.float x 5>> {dataClause = #acc<data_clause acc_copy>, name = "localArray"} loc
  // CHECK-NEXT: %[[COPYIN2:.*]] = acc.copyin varPtr(%[[LOCALPTR]] : !cir.ptr<!cir.ptr<!s16i>>) -> !cir.ptr<!cir.ptr<!s16i>> {dataClause = #acc<data_clause acc_copy>, name = "localPointer"} loc
  // CHECK-NEXT: %[[GLOBAL_REF:.*]] = cir.get_global @global : !cir.ptr<!s32i>
  // CHECK-NEXT: %[[COPYIN3:.*]] = acc.copyin varPtr(%[[GLOBAL_REF]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copy>, name = "global"} loc
  // CHECK-NEXT: acc.kernels combined(loop) dataOperands(%[[COPYIN1]], %[[COPYIN2]], %[[COPYIN3]] : !cir.ptr<!cir.array<!cir.float x 5>>, !cir.ptr<!cir.ptr<!s16i>>, !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN3]] : !cir.ptr<!s32i>) to varPtr(%[[GLOBAL_REF]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copy>, name = "global"} loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN2]] : !cir.ptr<!cir.ptr<!s16i>>) to varPtr(%[[LOCALPTR]] : !cir.ptr<!cir.ptr<!s16i>>) {dataClause = #acc<data_clause acc_copy>, name = "localPointer"} loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.float x 5>>) to varPtr(%[[LOCALARRAY]] : !cir.ptr<!cir.array<!cir.float x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "localArray"} loc

#pragma acc parallel loop copy(localVar1) async
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCAL1]] : !cir.ptr<!s32i>) async -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copy>, name = "localVar1"} loc
  // CHECK-NEXT: acc.parallel combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!s32i>) async {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!s32i>) async to varPtr(%[[LOCAL1]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copy>, name = "localVar1"} loc

#pragma acc serial loop async copy(localVar1, localVar2)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCAL1]] : !cir.ptr<!s32i>) async -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copy>, name = "localVar1"} loc
  // CHECK-NEXT: %[[COPYIN2:.*]] = acc.copyin varPtr(%[[LOCAL2]] : !cir.ptr<!s16i>) async -> !cir.ptr<!s16i> {dataClause = #acc<data_clause acc_copy>, name = "localVar2"} loc
  // CHECK-NEXT: acc.serial combined(loop) dataOperands(%[[COPYIN1]], %[[COPYIN2]] : !cir.ptr<!s32i>, !cir.ptr<!s16i>) async {
  // CHECK-NEXT: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN2]] : !cir.ptr<!s16i>) async to varPtr(%[[LOCAL2]] : !cir.ptr<!s16i>) {dataClause = #acc<data_clause acc_copy>, name = "localVar2"} loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!s32i>) async to varPtr(%[[LOCAL1]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copy>, name = "localVar1"} loc

#pragma acc kernels loop copy(localVar1, localVar2) async(1)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCAL1]] : !cir.ptr<!s32i>) async(%[[ONE_CAST]] : si32) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copy>, name = "localVar1"} loc
  // CHECK-NEXT: %[[COPYIN2:.*]] = acc.copyin varPtr(%[[LOCAL2]] : !cir.ptr<!s16i>) async(%[[ONE_CAST]] : si32) -> !cir.ptr<!s16i> {dataClause = #acc<data_clause acc_copy>, name = "localVar2"} loc
  // CHECK-NEXT: acc.kernels combined(loop) dataOperands(%[[COPYIN1]], %[[COPYIN2]] : !cir.ptr<!s32i>, !cir.ptr<!s16i>) async(%[[ONE_CAST]] : si32) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN2]] : !cir.ptr<!s16i>) async(%[[ONE_CAST]] : si32) to varPtr(%[[LOCAL2]] : !cir.ptr<!s16i>) {dataClause = #acc<data_clause acc_copy>, name = "localVar2"} loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!s32i>) async(%[[ONE_CAST]] : si32) to varPtr(%[[LOCAL1]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copy>, name = "localVar1"} loc

#pragma acc parallel loop async(1) copy(localVar1)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCAL1]] : !cir.ptr<!s32i>) async(%[[ONE_CAST]] : si32) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copy>, name = "localVar1"} loc
  // CHECK-NEXT: acc.parallel combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!s32i>) async(%[[ONE_CAST]] : si32) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!s32i>) async(%[[ONE_CAST]] : si32) to varPtr(%[[LOCAL1]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copy>, name = "localVar1"} loc

#pragma acc serial loop copy(localVar1) device_type(nvidia, radeon) async
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCAL1]] : !cir.ptr<!s32i>) async([#acc.device_type<nvidia>, #acc.device_type<radeon>]) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copy>, name = "localVar1"} loc
  // CHECK-NEXT: acc.serial combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!s32i>) async([#acc.device_type<nvidia>, #acc.device_type<radeon>]) {
  // CHECK-NEXT: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!s32i>)  async([#acc.device_type<nvidia>, #acc.device_type<radeon>]) to varPtr(%[[LOCAL1]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copy>, name = "localVar1"} loc

#pragma acc kernels loop copy(localVar1) device_type(nvidia, radeon) async(1)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCAL1]] : !cir.ptr<!s32i>) async(%[[ONE_CAST]] : si32 [#acc.device_type<nvidia>], %[[ONE_CAST]] : si32 [#acc.device_type<radeon>]) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copy>, name = "localVar1"} loc
  // CHECK-NEXT: acc.kernels combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!s32i>) async(%[[ONE_CAST]] : si32 [#acc.device_type<nvidia>], %[[ONE_CAST]] : si32 [#acc.device_type<radeon>]) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!s32i>)  async(%[[ONE_CAST]] : si32 [#acc.device_type<nvidia>], %[[ONE_CAST]] : si32 [#acc.device_type<radeon>]) to varPtr(%[[LOCAL1]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copy>, name = "localVar1"} loc

#pragma acc parallel loop copy(localVar1) async device_type(nvidia, radeon) async
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCAL1]] : !cir.ptr<!s32i>) async([#acc.device_type<none>, #acc.device_type<nvidia>, #acc.device_type<radeon>]) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copy>, name = "localVar1"} loc
  // CHECK-NEXT: acc.parallel combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!s32i>) async([#acc.device_type<none>, #acc.device_type<nvidia>, #acc.device_type<radeon>]) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!s32i>)  async([#acc.device_type<none>, #acc.device_type<nvidia>, #acc.device_type<radeon>]) to varPtr(%[[LOCAL1]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copy>, name = "localVar1"} loc

#pragma acc serial loop copy(localVar1) async(0) device_type(nvidia, radeon) async(1)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
  // CHECK-NEXT: %[[ZERO_CAST:.*]] = builtin.unrealized_conversion_cast %[[ZERO]] : !s32i to si32
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCAL1]] : !cir.ptr<!s32i>) async(%[[ZERO_CAST]] : si32, %[[ONE_CAST]] : si32 [#acc.device_type<nvidia>], %[[ONE_CAST]] : si32 [#acc.device_type<radeon>]) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copy>, name = "localVar1"} loc
  // CHECK-NEXT: acc.serial combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!s32i>) async(%[[ZERO_CAST]] : si32, %[[ONE_CAST]] : si32 [#acc.device_type<nvidia>], %[[ONE_CAST]] : si32 [#acc.device_type<radeon>]) {
  // CHECK-NEXT: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!s32i>)  async(%[[ZERO_CAST]] : si32, %[[ONE_CAST]] : si32 [#acc.device_type<nvidia>], %[[ONE_CAST]] : si32 [#acc.device_type<radeon>]) to varPtr(%[[LOCAL1]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copy>, name = "localVar1"} loc

#pragma acc kernels loop copy(localVar1) async device_type(nvidia, radeon) async(1)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCAL1]] : !cir.ptr<!s32i>) async([#acc.device_type<none>], %[[ONE_CAST]] : si32 [#acc.device_type<nvidia>], %[[ONE_CAST]] : si32 [#acc.device_type<radeon>]) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copy>, name = "localVar1"} loc
  // CHECK-NEXT: acc.kernels combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!s32i>) async([#acc.device_type<none>], %[[ONE_CAST]] : si32 [#acc.device_type<nvidia>], %[[ONE_CAST]] : si32 [#acc.device_type<radeon>]) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!s32i>)  async([#acc.device_type<none>], %[[ONE_CAST]] : si32 [#acc.device_type<nvidia>], %[[ONE_CAST]] : si32 [#acc.device_type<radeon>]) to varPtr(%[[LOCAL1]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copy>, name = "localVar1"} loc

#pragma acc parallel loop copy(localVar1) async(1) device_type(nvidia, radeon) async
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCAL1]] : !cir.ptr<!s32i>) async([#acc.device_type<nvidia>, #acc.device_type<radeon>], %[[ONE_CAST]] : si32) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copy>, name = "localVar1"} loc

  // CHECK-NEXT: acc.parallel combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!s32i>) async([#acc.device_type<nvidia>, #acc.device_type<radeon>], %[[ONE_CAST]] : si32) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!s32i>) async([#acc.device_type<nvidia>, #acc.device_type<radeon>], %[[ONE_CAST]] : si32) to varPtr(%[[LOCAL1]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copy>, name = "localVar1"} loc

#pragma acc serial loop copy(localVar1) async(0) device_type(nvidia, radeon) async(1)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
  // CHECK-NEXT: %[[ZERO_CAST:.*]] = builtin.unrealized_conversion_cast %[[ZERO]] : !s32i to si32
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCAL1]] : !cir.ptr<!s32i>) async(%[[ZERO_CAST]] : si32, %[[ONE_CAST]] : si32 [#acc.device_type<nvidia>], %[[ONE_CAST]] : si32 [#acc.device_type<radeon>]) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copy>, name = "localVar1"} loc
  // CHECK-NEXT: acc.serial combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!s32i>) async(%[[ZERO_CAST]] : si32, %[[ONE_CAST]] : si32 [#acc.device_type<nvidia>], %[[ONE_CAST]] : si32 [#acc.device_type<radeon>]) {
  // CHECK-NEXT: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!s32i>)  async(%[[ZERO_CAST]] : si32, %[[ONE_CAST]] : si32 [#acc.device_type<nvidia>], %[[ONE_CAST]] : si32 [#acc.device_type<radeon>]) to varPtr(%[[LOCAL1]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copy>, name = "localVar1"} loc

#pragma acc parallel loop copy(localArray[3])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[THREE:.*]] = cir.const #cir.int<3> : !s32i
  // CHECK-NEXT: %[[THREE_CAST:.*]] = builtin.unrealized_conversion_cast %[[THREE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[THREE_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64) loc
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCALARRAY]] : !cir.ptr<!cir.array<!cir.float x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!cir.float x 5>> {dataClause = #acc<data_clause acc_copy>, name = "localArray[3]"} loc
  // CHECK-NEXT: acc.parallel combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.float x 5>>) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.float x 5>>) bounds(%[[BOUNDS]]) to varPtr(%[[LOCALARRAY]] : !cir.ptr<!cir.array<!cir.float x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "localArray[3]"} loc
  
#pragma acc serial loop copy(localArray[1:3])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[THREE:.*]] = cir.const #cir.int<3> : !s32i
  // CHECK-NEXT: %[[THREE_CAST:.*]] = builtin.unrealized_conversion_cast %[[THREE]] : !s32i to si32
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[THREE_CAST]] : si32) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64) loc
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCALARRAY]] : !cir.ptr<!cir.array<!cir.float x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!cir.float x 5>> {dataClause = #acc<data_clause acc_copy>, name = "localArray[1:3]"} loc
  // CHECK-NEXT: acc.serial combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.float x 5>>) {
  // CHECK-NEXT: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.float x 5>>) bounds(%[[BOUNDS]]) to varPtr(%[[LOCALARRAY]] : !cir.ptr<!cir.array<!cir.float x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "localArray[1:3]"} loc

#pragma acc kernels loop copy(localArray[:3])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[THREE:.*]] = cir.const #cir.int<3> : !s32i
  // CHECK-NEXT: %[[THREE_CAST:.*]] = builtin.unrealized_conversion_cast %[[THREE]] : !s32i to si32
  // CHECK-NEXT: %[[ZERO_CONST2:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ZERO_CONST]] : i64) extent(%[[THREE_CAST]] : si32) stride(%[[ONE_CONST]] : i64) startIdx(%[[ZERO_CONST2]] : i64) loc
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCALARRAY]] : !cir.ptr<!cir.array<!cir.float x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!cir.float x 5>> {dataClause = #acc<data_clause acc_copy>, name = "localArray[:3]"} loc
  // CHECK-NEXT: acc.kernels combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.float x 5>>) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.float x 5>>) bounds(%[[BOUNDS]]) to varPtr(%[[LOCALARRAY]] : !cir.ptr<!cir.array<!cir.float x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "localArray[:3]"} loc

#pragma acc parallel loop copy(localArray[1:])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[FOUR_CONST:.*]] = arith.constant 4 : i64
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) upperbound(%[[FOUR_CONST]] : i64) stride(%[[ONE_CONST]] : i64) startIdx(%[[ZERO_CONST]] : i64) loc
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCALARRAY]] : !cir.ptr<!cir.array<!cir.float x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!cir.float x 5>> {dataClause = #acc<data_clause acc_copy>, name = "localArray[1:]"} loc
  // CHECK-NEXT: acc.parallel combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.float x 5>>) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.float x 5>>) bounds(%[[BOUNDS]]) to varPtr(%[[LOCALARRAY]] : !cir.ptr<!cir.array<!cir.float x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "localArray[1:]"} loc

#pragma acc serial loop copy(localArray[localVar1:localVar2])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[LV1:.*]] = cir.load{{.*}} %[[LOCAL1]] : !cir.ptr<!s32i>, !s32i 
  // CHECK-NEXT: %[[LV1_CAST:.*]] = builtin.unrealized_conversion_cast %[[LV1]] : !s32i to si32
  // CHECK-NEXT: %[[LV2:.*]] = cir.load{{.*}} %[[LOCAL2]] : !cir.ptr<!s16i>, !s16i 
  // CHECK-NEXT: %[[LV2_CAST:.*]] = builtin.unrealized_conversion_cast %[[LV2]] : !s16i to si16
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[LV1_CAST]] : si32) extent(%[[LV2_CAST]] : si16) stride(%[[ONE_CONST]] : i64) startIdx(%[[ZERO_CONST]] : i64) loc
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCALARRAY]] : !cir.ptr<!cir.array<!cir.float x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!cir.float x 5>> {dataClause = #acc<data_clause acc_copy>, name = "localArray[localVar1:localVar2]"} loc
  // CHECK-NEXT: acc.serial combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.float x 5>>) {
  // CHECK-NEXT: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.float x 5>>) bounds(%[[BOUNDS]]) to varPtr(%[[LOCALARRAY]] : !cir.ptr<!cir.array<!cir.float x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "localArray[localVar1:localVar2]"} loc

#pragma acc kernels loop copy(localArray[:localVar2])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[LV2:.*]] = cir.load{{.*}} %[[LOCAL2]] : !cir.ptr<!s16i>, !s16i 
  // CHECK-NEXT: %[[LV2_CAST:.*]] = builtin.unrealized_conversion_cast %[[LV2]] : !s16i to si16
  // CHECK-NEXT: %[[ZERO_CONST2:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ZERO_CONST]] : i64) extent(%[[LV2_CAST]] : si16) stride(%[[ONE_CONST]] : i64) startIdx(%[[ZERO_CONST2]] : i64) loc
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCALARRAY]] : !cir.ptr<!cir.array<!cir.float x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!cir.float x 5>> {dataClause = #acc<data_clause acc_copy>, name = "localArray[:localVar2]"} loc
  // CHECK-NEXT: acc.kernels combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.float x 5>>) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.float x 5>>) bounds(%[[BOUNDS]]) to varPtr(%[[LOCALARRAY]] : !cir.ptr<!cir.array<!cir.float x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "localArray[:localVar2]"} loc

#pragma acc parallel loop copy(localArray[localVar1:])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[LV1:.*]] = cir.load{{.*}} %[[LOCAL1]] : !cir.ptr<!s32i>, !s32i 
  // CHECK-NEXT: %[[LV1_CAST:.*]] = builtin.unrealized_conversion_cast %[[LV1]] : !s32i to si32
  // CHECK-NEXT: %[[FOUR_CONST:.*]] = arith.constant 4 : i64
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[LV1_CAST]] : si32) upperbound(%[[FOUR_CONST]] : i64) stride(%[[ONE_CONST]] : i64) startIdx(%[[ZERO_CONST]] : i64) loc
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCALARRAY]] : !cir.ptr<!cir.array<!cir.float x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!cir.float x 5>> {dataClause = #acc<data_clause acc_copy>, name = "localArray[localVar1:]"} loc
  // CHECK-NEXT: acc.parallel combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.float x 5>>) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.float x 5>>) bounds(%[[BOUNDS]]) to varPtr(%[[LOCALARRAY]] : !cir.ptr<!cir.array<!cir.float x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "localArray[localVar1:]"} loc

#pragma acc serial loop copy(localPointer[3])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[THREE:.*]] = cir.const #cir.int<3> : !s32i
  // CHECK-NEXT: %[[THREE_CAST:.*]] = builtin.unrealized_conversion_cast %[[THREE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[THREE_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64) loc
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCALPTR]] : !cir.ptr<!cir.ptr<!s16i>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.ptr<!s16i>> {dataClause = #acc<data_clause acc_copy>, name = "localPointer[3]"} loc
  // CHECK-NEXT: acc.serial combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.ptr<!s16i>>) {
  // CHECK-NEXT: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.ptr<!s16i>>) bounds(%[[BOUNDS]]) to varPtr(%[[LOCALPTR]] : !cir.ptr<!cir.ptr<!s16i>>) {dataClause = #acc<data_clause acc_copy>, name = "localPointer[3]"} loc

#pragma acc kernels loop copy(localPointer[1:3])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[THREE:.*]] = cir.const #cir.int<3> : !s32i
  // CHECK-NEXT: %[[THREE_CAST:.*]] = builtin.unrealized_conversion_cast %[[THREE]] : !s32i to si32
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[THREE_CAST]] : si32) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64) loc
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCALPTR]] : !cir.ptr<!cir.ptr<!s16i>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.ptr<!s16i>> {dataClause = #acc<data_clause acc_copy>, name = "localPointer[1:3]"} loc
  // CHECK-NEXT: acc.kernels combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.ptr<!s16i>>) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.ptr<!s16i>>) bounds(%[[BOUNDS]]) to varPtr(%[[LOCALPTR]] : !cir.ptr<!cir.ptr<!s16i>>) {dataClause = #acc<data_clause acc_copy>, name = "localPointer[1:3]"} loc

#pragma acc parallel loop copy(localPointer[:3])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[THREE:.*]] = cir.const #cir.int<3> : !s32i
  // CHECK-NEXT: %[[THREE_CAST:.*]] = builtin.unrealized_conversion_cast %[[THREE]] : !s32i to si32
  // CHECK-NEXT: %[[ZERO_CONST2:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ZERO_CONST]] : i64) extent(%[[THREE_CAST]] : si32) stride(%[[ONE_CONST]] : i64) startIdx(%[[ZERO_CONST2]] : i64) loc
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCALPTR]] : !cir.ptr<!cir.ptr<!s16i>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.ptr<!s16i>> {dataClause = #acc<data_clause acc_copy>, name = "localPointer[:3]"} loc
  // CHECK-NEXT: acc.parallel combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.ptr<!s16i>>) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.ptr<!s16i>>) bounds(%[[BOUNDS]]) to varPtr(%[[LOCALPTR]] : !cir.ptr<!cir.ptr<!s16i>>) {dataClause = #acc<data_clause acc_copy>, name = "localPointer[:3]"} loc

#pragma acc serial loop copy(localPointer[localVar1:localVar2])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[LV1:.*]] = cir.load{{.*}} %[[LOCAL1]] : !cir.ptr<!s32i>, !s32i 
  // CHECK-NEXT: %[[LV1_CAST:.*]] = builtin.unrealized_conversion_cast %[[LV1]] : !s32i to si32
  // CHECK-NEXT: %[[LV2:.*]] = cir.load{{.*}} %[[LOCAL2]] : !cir.ptr<!s16i>, !s16i 
  // CHECK-NEXT: %[[LV2_CAST:.*]] = builtin.unrealized_conversion_cast %[[LV2]] : !s16i to si16
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[LV1_CAST]] : si32) extent(%[[LV2_CAST]] : si16) stride(%[[ONE_CONST]] : i64) startIdx(%[[ZERO_CONST]] : i64) loc
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCALPTR]] : !cir.ptr<!cir.ptr<!s16i>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.ptr<!s16i>> {dataClause = #acc<data_clause acc_copy>, name = "localPointer[localVar1:localVar2]"} loc
  // CHECK-NEXT: acc.serial combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.ptr<!s16i>>) {
  // CHECK-NEXT: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.ptr<!s16i>>) bounds(%[[BOUNDS]]) to varPtr(%[[LOCALPTR]] : !cir.ptr<!cir.ptr<!s16i>>) {dataClause = #acc<data_clause acc_copy>, name = "localPointer[localVar1:localVar2]"} loc

#pragma acc kernels loop copy(localPointer[:localVar2])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[LV2:.*]] = cir.load{{.*}} %[[LOCAL2]] : !cir.ptr<!s16i>, !s16i 
  // CHECK-NEXT: %[[LV2_CAST:.*]] = builtin.unrealized_conversion_cast %[[LV2]] : !s16i to si16
  // CHECK-NEXT: %[[ZERO_CONST2:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ZERO_CONST]] : i64) extent(%[[LV2_CAST]] : si16) stride(%[[ONE_CONST]] : i64) startIdx(%[[ZERO_CONST2]] : i64) loc
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCALPTR]] : !cir.ptr<!cir.ptr<!s16i>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.ptr<!s16i>> {dataClause = #acc<data_clause acc_copy>, name = "localPointer[:localVar2]"} loc
  // CHECK-NEXT: acc.kernels combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.ptr<!s16i>>) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.ptr<!s16i>>) bounds(%[[BOUNDS]]) to varPtr(%[[LOCALPTR]] : !cir.ptr<!cir.ptr<!s16i>>) {dataClause = #acc<data_clause acc_copy>, name = "localPointer[:localVar2]"} loc

  float *localArrayOfPtrs[5];
#pragma acc parallel loop copy(localArrayOfPtrs[3])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[THREE:.*]] = cir.const #cir.int<3> : !s32i
  // CHECK-NEXT: %[[THREE_CAST:.*]] = builtin.unrealized_conversion_cast %[[THREE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[THREE_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64) loc
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCALARRAYOFPTRS]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>> {dataClause = #acc<data_clause acc_copy>, name = "localArrayOfPtrs[3]"} loc
  // CHECK-NEXT: acc.parallel combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) bounds(%[[BOUNDS]]) to varPtr(%[[LOCALARRAYOFPTRS]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "localArrayOfPtrs[3]"} loc

#pragma acc serial loop copy(localArrayOfPtrs[3][2])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
  // CHECK-NEXT: %[[TWO_CAST:.*]] = builtin.unrealized_conversion_cast %[[TWO]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[TWO_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64) loc
  // CHECK-NEXT: %[[THREE:.*]] = cir.const #cir.int<3> : !s32i
  // CHECK-NEXT: %[[THREE_CAST:.*]] = builtin.unrealized_conversion_cast %[[THREE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS2:.*]] = acc.bounds lowerbound(%[[THREE_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64) loc
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCALARRAYOFPTRS]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) bounds(%[[BOUNDS]], %[[BOUNDS2]]) -> !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>> {dataClause = #acc<data_clause acc_copy>, name = "localArrayOfPtrs[3][2]"} loc
  // CHECK-NEXT: acc.serial combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) {
  // CHECK-NEXT: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) bounds(%[[BOUNDS]], %[[BOUNDS2]]) to varPtr(%[[LOCALARRAYOFPTRS]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "localArrayOfPtrs[3][2]"} loc

#pragma acc kernels loop copy(localArrayOfPtrs[localVar1:localVar2])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[LV1:.*]] = cir.load{{.*}} %[[LOCAL1]] : !cir.ptr<!s32i>, !s32i 
  // CHECK-NEXT: %[[LV1_CAST:.*]] = builtin.unrealized_conversion_cast %[[LV1]] : !s32i to si32
  // CHECK-NEXT: %[[LV2:.*]] = cir.load{{.*}} %[[LOCAL2]] : !cir.ptr<!s16i>, !s16i 
  // CHECK-NEXT: %[[LV2_CAST:.*]] = builtin.unrealized_conversion_cast %[[LV2]] : !s16i to si16
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[LV1_CAST]] : si32) extent(%[[LV2_CAST]] : si16) stride(%[[ONE_CONST]] : i64) startIdx(%[[ZERO_CONST]] : i64) loc
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCALARRAYOFPTRS]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>> {dataClause = #acc<data_clause acc_copy>, name = "localArrayOfPtrs[localVar1:localVar2]"} loc
  // CHECK-NEXT: acc.kernels combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) bounds(%[[BOUNDS]]) to varPtr(%[[LOCALARRAYOFPTRS]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "localArrayOfPtrs[localVar1:localVar2]"} loc

#pragma acc parallel loop copy(localArrayOfPtrs[localVar1:])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[LV1:.*]] = cir.load{{.*}} %[[LOCAL1]] : !cir.ptr<!s32i>, !s32i 
  // CHECK-NEXT: %[[LV1_CAST:.*]] = builtin.unrealized_conversion_cast %[[LV1]] : !s32i to si32
  // CHECK-NEXT: %[[FOUR_CONST:.*]] = arith.constant 4 : i64
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[LV1_CAST]] : si32) upperbound(%[[FOUR_CONST]] : i64) stride(%[[ONE_CONST]] : i64) startIdx(%[[ZERO_CONST]] : i64) loc
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCALARRAYOFPTRS]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>> {dataClause = #acc<data_clause acc_copy>, name = "localArrayOfPtrs[localVar1:]"} loc
  // CHECK-NEXT: acc.parallel combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) bounds(%[[BOUNDS]]) to varPtr(%[[LOCALARRAYOFPTRS]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "localArrayOfPtrs[localVar1:]"} loc

#pragma acc serial loop copy(localArrayOfPtrs[:localVar2])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[LV2:.*]] = cir.load{{.*}} %[[LOCAL2]] : !cir.ptr<!s16i>, !s16i 
  // CHECK-NEXT: %[[LV2_CAST:.*]] = builtin.unrealized_conversion_cast %[[LV2]] : !s16i to si16
  // CHECK-NEXT: %[[ZERO_CONST2:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ZERO_CONST]] : i64) extent(%[[LV2_CAST]] : si16) stride(%[[ONE_CONST]] : i64) startIdx(%[[ZERO_CONST2]] : i64) loc
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCALARRAYOFPTRS]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>> {dataClause = #acc<data_clause acc_copy>, name = "localArrayOfPtrs[:localVar2]"} loc
  // CHECK-NEXT: acc.serial combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) {
  // CHECK-NEXT: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) bounds(%[[BOUNDS]]) to varPtr(%[[LOCALARRAYOFPTRS]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "localArrayOfPtrs[:localVar2]"} loc

#pragma acc kernels loop copy(localArrayOfPtrs[localVar1])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[LV1:.*]] = cir.load{{.*}} %[[LOCAL1]] : !cir.ptr<!s32i>, !s32i 
  // CHECK-NEXT: %[[LV1_CAST:.*]] = builtin.unrealized_conversion_cast %[[LV1]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[LV1_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64) loc
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCALARRAYOFPTRS]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>> {dataClause = #acc<data_clause acc_copy>, name = "localArrayOfPtrs[localVar1]"} loc
  // CHECK-NEXT: acc.kernels combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) bounds(%[[BOUNDS]]) to varPtr(%[[LOCALARRAYOFPTRS]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "localArrayOfPtrs[localVar1]"} loc

#pragma acc parallel loop copy(localArrayOfPtrs[localVar1][localVar2])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[LV2:.*]] = cir.load{{.*}} %[[LOCAL2]] : !cir.ptr<!s16i>, !s16i 
  // CHECK-NEXT: %[[LV2_CAST:.*]] = builtin.unrealized_conversion_cast %[[LV2]] : !s16i to si16
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[LV2_CAST]] : si16) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64) loc
  // CHECK-NEXT: %[[LV1:.*]] = cir.load{{.*}} %[[LOCAL1]] : !cir.ptr<!s32i>, !s32i 
  // CHECK-NEXT: %[[LV1_CAST:.*]] = builtin.unrealized_conversion_cast %[[LV1]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS2:.*]] = acc.bounds lowerbound(%[[LV1_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64) loc
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCALARRAYOFPTRS]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) bounds(%[[BOUNDS]], %[[BOUNDS2]]) -> !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>> {dataClause = #acc<data_clause acc_copy>, name = "localArrayOfPtrs[localVar1][localVar2]"} loc
  // CHECK-NEXT: acc.parallel combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) bounds(%[[BOUNDS]], %[[BOUNDS2]]) to varPtr(%[[LOCALARRAYOFPTRS]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "localArrayOfPtrs[localVar1][localVar2]"} loc
  
#pragma acc serial loop copy(localArrayOfPtrs[localVar1][localVar2:parmVar])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[LV2:.*]] = cir.load{{.*}} %[[LOCAL2]] : !cir.ptr<!s16i>, !s16i 
  // CHECK-NEXT: %[[LV2_CAST:.*]] = builtin.unrealized_conversion_cast %[[LV2]] : !s16i to si16
  // CHECK-NEXT: %[[PV:.*]] = cir.load{{.*}} %[[PARM]] : !cir.ptr<!s32i>, !s32i 
  // CHECK-NEXT: %[[PV_CAST:.*]] = builtin.unrealized_conversion_cast %[[PV]] : !s32i to si32
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[LV2_CAST]] : si16) extent(%[[PV_CAST]] : si32) stride(%[[ONE_CONST]] : i64) startIdx(%[[ZERO_CONST]] : i64) loc
  // CHECK-NEXT: %[[LV1:.*]] = cir.load{{.*}} %[[LOCAL1]] : !cir.ptr<!s32i>, !s32i 
  // CHECK-NEXT: %[[LV1_CAST:.*]] = builtin.unrealized_conversion_cast %[[LV1]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS2:.*]] = acc.bounds lowerbound(%[[LV1_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64) loc
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCALARRAYOFPTRS]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) bounds(%[[BOUNDS]], %[[BOUNDS2]]) -> !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>> {dataClause = #acc<data_clause acc_copy>, name = "localArrayOfPtrs[localVar1][localVar2:parmVar]"} loc
  // CHECK-NEXT: acc.serial combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) {
  // CHECK-NEXT: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) bounds(%[[BOUNDS]], %[[BOUNDS2]]) to varPtr(%[[LOCALARRAYOFPTRS]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "localArrayOfPtrs[localVar1][localVar2:parmVar]"} loc
 
#pragma acc kernels loop copy(localArrayOfPtrs[localVar1][:parmVar])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[PV:.*]] = cir.load{{.*}} %[[PARM]] : !cir.ptr<!s32i>, !s32i 
  // CHECK-NEXT: %[[PV_CAST:.*]] = builtin.unrealized_conversion_cast %[[PV]] : !s32i to si32
  // CHECK-NEXT: %[[ZERO_CONST2:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ZERO_CONST]] : i64) extent(%[[PV_CAST]] : si32) stride(%[[ONE_CONST]] : i64) startIdx(%[[ZERO_CONST2]] : i64) loc
  // CHECK-NEXT: %[[LV1:.*]] = cir.load{{.*}} %[[LOCAL1]] : !cir.ptr<!s32i>, !s32i 
  // CHECK-NEXT: %[[LV1_CAST:.*]] = builtin.unrealized_conversion_cast %[[LV1]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS2:.*]] = acc.bounds lowerbound(%[[LV1_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64) loc
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCALARRAYOFPTRS]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) bounds(%[[BOUNDS]], %[[BOUNDS2]]) -> !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>> {dataClause = #acc<data_clause acc_copy>, name = "localArrayOfPtrs[localVar1][:parmVar]"} loc
  // CHECK-NEXT: acc.kernels combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) bounds(%[[BOUNDS]], %[[BOUNDS2]]) to varPtr(%[[LOCALARRAYOFPTRS]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "localArrayOfPtrs[localVar1][:parmVar]"} loc

#pragma acc parallel loop copy(localArrayOfPtrs[localVar1:localVar2][:1])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ZERO_CONST2:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ZERO_CONST]] : i64) extent(%[[ONE_CAST]] : si32) stride(%[[ONE_CONST]] : i64) startIdx(%[[ZERO_CONST2]] : i64) loc
  // CHECK-NEXT: %[[LV1:.*]] = cir.load{{.*}} %[[LOCAL1]] : !cir.ptr<!s32i>, !s32i 
  // CHECK-NEXT: %[[LV1_CAST:.*]] = builtin.unrealized_conversion_cast %[[LV1]] : !s32i to si32
  // CHECK-NEXT: %[[LV2:.*]] = cir.load{{.*}} %[[LOCAL2]] : !cir.ptr<!s16i>, !s16i 
  // CHECK-NEXT: %[[LV2_CAST:.*]] = builtin.unrealized_conversion_cast %[[LV2]] : !s16i to si16
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS2:.*]] = acc.bounds lowerbound(%[[LV1_CAST]] : si32) extent(%[[LV2_CAST]] : si16) stride(%[[ONE_CONST]] : i64) startIdx(%[[ZERO_CONST]] : i64) loc
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCALARRAYOFPTRS]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) bounds(%[[BOUNDS]], %[[BOUNDS2]]) -> !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>> {dataClause = #acc<data_clause acc_copy>, name = "localArrayOfPtrs[localVar1:localVar2][:1]"} loc
  // CHECK-NEXT: acc.parallel combined(loop)  dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) bounds(%[[BOUNDS]], %[[BOUNDS2]]) to varPtr(%[[LOCALARRAYOFPTRS]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "localArrayOfPtrs[localVar1:localVar2][:1]"} loc

#pragma acc serial loop copy(localArrayOfPtrs[localVar1:localVar2][1:1])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST2:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ZERO_CONST2:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CAST2]] : si32) stride(%[[ONE_CONST]] : i64) startIdx(%[[ZERO_CONST2]] : i64) loc
  // CHECK-NEXT: %[[LV1:.*]] = cir.load{{.*}} %[[LOCAL1]] : !cir.ptr<!s32i>, !s32i 
  // CHECK-NEXT: %[[LV1_CAST:.*]] = builtin.unrealized_conversion_cast %[[LV1]] : !s32i to si32
  // CHECK-NEXT: %[[LV2:.*]] = cir.load{{.*}} %[[LOCAL2]] : !cir.ptr<!s16i>, !s16i 
  // CHECK-NEXT: %[[LV2_CAST:.*]] = builtin.unrealized_conversion_cast %[[LV2]] : !s16i to si16
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS2:.*]] = acc.bounds lowerbound(%[[LV1_CAST]] : si32) extent(%[[LV2_CAST]] : si16) stride(%[[ONE_CONST]] : i64) startIdx(%[[ZERO_CONST]] : i64) loc
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCALARRAYOFPTRS]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) bounds(%[[BOUNDS]], %[[BOUNDS2]]) -> !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>> {dataClause = #acc<data_clause acc_copy>, name = "localArrayOfPtrs[localVar1:localVar2][1:1]"} loc
  // CHECK-NEXT: acc.serial combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) {
  // CHECK-NEXT: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) bounds(%[[BOUNDS]], %[[BOUNDS2]]) to varPtr(%[[LOCALARRAYOFPTRS]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "localArrayOfPtrs[localVar1:localVar2][1:1]"} loc

  double threeDArray[5][6][7];
#pragma acc kernels loop copy(threeDArray[1][2][3])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[THREE:.*]] = cir.const #cir.int<3> : !s32i
  // CHECK-NEXT: %[[THREE_CAST:.*]] = builtin.unrealized_conversion_cast %[[THREE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[THREE_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64) loc
  // CHECK-NEXT: %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
  // CHECK-NEXT: %[[TWO_CAST:.*]] = builtin.unrealized_conversion_cast %[[TWO]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS2:.*]] = acc.bounds lowerbound(%[[TWO_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64) loc
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS3:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64) loc
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[THREEDARRAY]] : !cir.ptr<!cir.array<!cir.array<!cir.array<!cir.double x 7> x 6> x 5>>) bounds(%[[BOUNDS]], %[[BOUNDS2]], %[[BOUNDS3]]) -> !cir.ptr<!cir.array<!cir.array<!cir.array<!cir.double x 7> x 6> x 5>> {dataClause = #acc<data_clause acc_copy>, name = "threeDArray[1][2][3]"} loc
  // CHECK-NEXT: acc.kernels combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.array<!cir.array<!cir.double x 7> x 6> x 5>>) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.array<!cir.array<!cir.double x 7> x 6> x 5>>) bounds(%[[BOUNDS]], %[[BOUNDS2]], %[[BOUNDS3]]) to varPtr(%[[THREEDARRAY]] : !cir.ptr<!cir.array<!cir.array<!cir.array<!cir.double x 7> x 6> x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "threeDArray[1][2][3]"} loc

#pragma acc parallel loop copy(threeDArray[1:1][2:1][3:1])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[THREE:.*]] = cir.const #cir.int<3> : !s32i
  // CHECK-NEXT: %[[THREE_CAST:.*]] = builtin.unrealized_conversion_cast %[[THREE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[THREE_CAST]] : si32) extent(%[[ONE_CAST]] : si32) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64) loc
  // CHECK-NEXT: %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
  // CHECK-NEXT: %[[TWO_CAST:.*]] = builtin.unrealized_conversion_cast %[[TWO]] : !s32i to si32
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS2:.*]] = acc.bounds lowerbound(%[[TWO_CAST]] : si32) extent(%[[ONE_CAST]] : si32) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64) loc
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST2:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS3:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CAST2]] : si32) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64) loc
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[THREEDARRAY]] : !cir.ptr<!cir.array<!cir.array<!cir.array<!cir.double x 7> x 6> x 5>>) bounds(%[[BOUNDS]], %[[BOUNDS2]], %[[BOUNDS3]]) -> !cir.ptr<!cir.array<!cir.array<!cir.array<!cir.double x 7> x 6> x 5>> {dataClause = #acc<data_clause acc_copy>, name = "threeDArray[1:1][2:1][3:1]"} loc
  // CHECK-NEXT: acc.parallel combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.array<!cir.array<!cir.double x 7> x 6> x 5>>) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.array<!cir.array<!cir.double x 7> x 6> x 5>>) bounds(%[[BOUNDS]], %[[BOUNDS2]], %[[BOUNDS3]]) to varPtr(%[[THREEDARRAY]] : !cir.ptr<!cir.array<!cir.array<!cir.array<!cir.double x 7> x 6> x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "threeDArray[1:1][2:1][3:1]"} loc
}

typedef struct StructTy {
  int scalarMember;
  int arrayMember[5];
  short twoDArrayMember[5][3];
  float *ptrArrayMember[5];
  double **ptrPtrMember;
} Struct ;

void acc_compute_members() {
  // CHECK: cir.func{{.*}} @acc_compute_members()
  Struct localStruct;
  // CHECK-NEXT: %[[LOCALSTRUCT:.*]] = cir.alloca !rec_StructTy, !cir.ptr<!rec_StructTy>, ["localStruct"]

#pragma acc parallel loop copy(localStruct)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCALSTRUCT]] : !cir.ptr<!rec_StructTy>) -> !cir.ptr<!rec_StructTy> {dataClause = #acc<data_clause acc_copy>, name = "localStruct"}
  // CHECK-NEXT: acc.parallel combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!rec_StructTy>) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!rec_StructTy>) to varPtr(%[[LOCALSTRUCT]] : !cir.ptr<!rec_StructTy>) {dataClause = #acc<data_clause acc_copy>, name = "localStruct"}

#pragma acc serial loop copy(localStruct.scalarMember)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[GETMEMBER:.*]] = cir.get_member %[[LOCALSTRUCT]][0] {name = "scalarMember"} : !cir.ptr<!rec_StructTy> -> !cir.ptr<!s32i>
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[GETMEMBER]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copy>, name = "localStruct.scalarMember"}
  // CHECK-NEXT: acc.serial combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!s32i>) to varPtr(%[[GETMEMBER]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copy>, name = "localStruct.scalarMember"}

#pragma acc kernels loop copy(localStruct.arrayMember)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[GETARRAYMEMBER:.*]] = cir.get_member %[[LOCALSTRUCT]][1] {name = "arrayMember"} : !cir.ptr<!rec_StructTy> -> !cir.ptr<!cir.array<!s32i x 5>>
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[GETARRAYMEMBER]] : !cir.ptr<!cir.array<!s32i x 5>>) -> !cir.ptr<!cir.array<!s32i x 5>> {dataClause = #acc<data_clause acc_copy>, name = "localStruct.arrayMember"}
  // CHECK-NEXT: acc.kernels combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!s32i x 5>>) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!s32i x 5>>) to varPtr(%[[GETARRAYMEMBER]] : !cir.ptr<!cir.array<!s32i x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "localStruct.arrayMember"} loc

#pragma acc parallel loop copy(localStruct.arrayMember[2])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[TWO_CONST:.*]] = cir.const #cir.int<2> : !s32i
  // CHECK-NEXT: %[[TWO_CAST:.*]] = builtin.unrealized_conversion_cast %[[TWO_CONST]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS:.*]]  = acc.bounds lowerbound(%[[TWO_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64) 
  // CHECK-NEXT: %[[GETARRAYMEMBER:.*]] = cir.get_member %[[LOCALSTRUCT]][1] {name = "arrayMember"} : !cir.ptr<!rec_StructTy> -> !cir.ptr<!cir.array<!s32i x 5>>
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[GETARRAYMEMBER]] : !cir.ptr<!cir.array<!s32i x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!s32i x 5>> {dataClause = #acc<data_clause acc_copy>, name = "localStruct.arrayMember[2]"} 
  // CHECK-NEXT: acc.parallel combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!s32i x 5>>) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield 
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!s32i x 5>>) bounds(%[[BOUNDS]]) to varPtr(%[[GETARRAYMEMBER]] : !cir.ptr<!cir.array<!s32i x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "localStruct.arrayMember[2]"} loc

#pragma acc serial loop copy(localStruct.arrayMember[1:2])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
  // CHECK-NEXT: %[[TWO_CAST:.*]] = builtin.unrealized_conversion_cast %[[TWO]] : !s32i to si32
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS:.*]]  = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[TWO_CAST]] : si32) stride(%[[ONE_CONST]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[GETARRAYMEMBER:.*]] = cir.get_member %[[LOCALSTRUCT]][1] {name = "arrayMember"} : !cir.ptr<!rec_StructTy> -> !cir.ptr<!cir.array<!s32i x 5>>
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[GETARRAYMEMBER]] : !cir.ptr<!cir.array<!s32i x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!s32i x 5>> {dataClause = #acc<data_clause acc_copy>, name = "localStruct.arrayMember[1:2]"} 
  // CHECK-NEXT: acc.serial combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!s32i x 5>>) {
  // CHECK-NEXT: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield 
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!s32i x 5>>) bounds(%[[BOUNDS]]) to varPtr(%[[GETARRAYMEMBER]] : !cir.ptr<!cir.array<!s32i x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "localStruct.arrayMember[1:2]"} loc

#pragma acc kernels loop copy(localStruct.arrayMember[1:])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[FOUR_CONST:.*]] = arith.constant 4 : i64
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS:.*]]  = acc.bounds lowerbound(%[[ONE_CAST]] : si32) upperbound(%[[FOUR_CONST]] : i64) stride(%[[ONE_CONST]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[GETARRAYMEMBER:.*]] = cir.get_member %[[LOCALSTRUCT]][1] {name = "arrayMember"} : !cir.ptr<!rec_StructTy> -> !cir.ptr<!cir.array<!s32i x 5>>
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[GETARRAYMEMBER]] : !cir.ptr<!cir.array<!s32i x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!s32i x 5>> {dataClause = #acc<data_clause acc_copy>, name = "localStruct.arrayMember[1:]"} 
  // CHECK-NEXT: acc.kernels combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!s32i x 5>>) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!s32i x 5>>) bounds(%[[BOUNDS]]) to varPtr(%[[GETARRAYMEMBER]] : !cir.ptr<!cir.array<!s32i x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "localStruct.arrayMember[1:]"} loc

#pragma acc parallel loop copy(localStruct.arrayMember[:2])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
  // CHECK-NEXT: %[[TWO_CAST:.*]] = builtin.unrealized_conversion_cast %[[TWO]] : !s32i to si32
  // CHECK-NEXT: %[[ZERO_CONST2:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS:.*]]  = acc.bounds lowerbound(%[[ZERO_CONST]] : i64) extent(%[[TWO_CAST]] : si32) stride(%[[ONE_CONST]] : i64) startIdx(%[[ZERO_CONST2]] : i64)
  // CHECK-NEXT: %[[GETARRAYMEMBER:.*]] = cir.get_member %[[LOCALSTRUCT]][1] {name = "arrayMember"} : !cir.ptr<!rec_StructTy> -> !cir.ptr<!cir.array<!s32i x 5>>
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[GETARRAYMEMBER]] : !cir.ptr<!cir.array<!s32i x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!s32i x 5>> {dataClause = #acc<data_clause acc_copy>, name = "localStruct.arrayMember[:2]"} 
  // CHECK-NEXT: acc.parallel combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!s32i x 5>>) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield 
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!s32i x 5>>) bounds(%[[BOUNDS]]) to varPtr(%[[GETARRAYMEMBER]] : !cir.ptr<!cir.array<!s32i x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "localStruct.arrayMember[:2]"} loc

#pragma acc serial loop copy(localStruct.twoDArrayMember)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[GET2DARRAYMEMBER:.*]] = cir.get_member %[[LOCALSTRUCT]][2] {name = "twoDArrayMember"} : !cir.ptr<!rec_StructTy> -> !cir.ptr<!cir.array<!cir.array<!s16i x 3> x 5>>
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[GET2DARRAYMEMBER]] : !cir.ptr<!cir.array<!cir.array<!s16i x 3> x 5>>) -> !cir.ptr<!cir.array<!cir.array<!s16i x 3> x 5>> {dataClause = #acc<data_clause acc_copy>, name = "localStruct.twoDArrayMember"} 
  // CHECK-NEXT: acc.serial combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.array<!s16i x 3> x 5>>) {
  // CHECK-NEXT: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield 
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.array<!s16i x 3> x 5>>) to varPtr(%[[GET2DARRAYMEMBER]] : !cir.ptr<!cir.array<!cir.array<!s16i x 3> x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "localStruct.twoDArrayMember"} 

#pragma acc kernels loop copy(localStruct.twoDArrayMember[3][2])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
  // CHECK-NEXT: %[[TWO_CAST:.*]] = builtin.unrealized_conversion_cast %[[TWO]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS1:.*]] = acc.bounds lowerbound(%[[TWO_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[THREE:.*]] = cir.const #cir.int<3> : !s32i
  // CHECK-NEXT: %[[THREE_CAST:.*]] = builtin.unrealized_conversion_cast %[[THREE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS2:.*]] = acc.bounds lowerbound(%[[THREE_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[GET2DARRAYMEMBER:.*]] = cir.get_member %[[LOCALSTRUCT]][2] {name = "twoDArrayMember"} : !cir.ptr<!rec_StructTy> -> !cir.ptr<!cir.array<!cir.array<!s16i x 3> x 5>>
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[GET2DARRAYMEMBER]] : !cir.ptr<!cir.array<!cir.array<!s16i x 3> x 5>>) bounds(%[[BOUNDS1]], %[[BOUNDS2]]) -> !cir.ptr<!cir.array<!cir.array<!s16i x 3> x 5>> {dataClause = #acc<data_clause acc_copy>, name = "localStruct.twoDArrayMember[3][2]"} 
  // CHECK-NEXT: acc.kernels combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.array<!s16i x 3> x 5>>) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.array<!s16i x 3> x 5>>) bounds(%[[BOUNDS1]], %[[BOUNDS2]]) to varPtr(%[[GET2DARRAYMEMBER]] : !cir.ptr<!cir.array<!cir.array<!s16i x 3> x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "localStruct.twoDArrayMember[3][2]"} 

#pragma acc parallel loop copy(localStruct.twoDArrayMember[1:3][1:1])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST2:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS1:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CAST2]] : si32) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[THREE:.*]] = cir.const #cir.int<3> : !s32i
  // CHECK-NEXT: %[[THREE_CAST:.*]] = builtin.unrealized_conversion_cast %[[THREE]] : !s32i to si32
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS2:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[THREE_CAST]] : si32) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[GET2DARRAYMEMBER:.*]] = cir.get_member %[[LOCALSTRUCT]][2] {name = "twoDArrayMember"} : !cir.ptr<!rec_StructTy> -> !cir.ptr<!cir.array<!cir.array<!s16i x 3> x 5>>
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[GET2DARRAYMEMBER]] : !cir.ptr<!cir.array<!cir.array<!s16i x 3> x 5>>) bounds(%[[BOUNDS1]], %[[BOUNDS2]]) -> !cir.ptr<!cir.array<!cir.array<!s16i x 3> x 5>> {dataClause = #acc<data_clause acc_copy>, name = "localStruct.twoDArrayMember[1:3][1:1]"} 
  // CHECK-NEXT: acc.parallel combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.array<!s16i x 3> x 5>>) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.array<!s16i x 3> x 5>>) bounds(%[[BOUNDS1]], %[[BOUNDS2]]) to varPtr(%[[GET2DARRAYMEMBER]] : !cir.ptr<!cir.array<!cir.array<!s16i x 3> x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "localStruct.twoDArrayMember[1:3][1:1]"} 

#pragma acc serial loop copy(localStruct.ptrArrayMember)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[GETPTRARRAYMEMBER:.*]] = cir.get_member %[[LOCALSTRUCT]][3] {name = "ptrArrayMember"} : !cir.ptr<!rec_StructTy> -> !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[GETPTRARRAYMEMBER]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) -> !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>> {dataClause = #acc<data_clause acc_copy>, name = "localStruct.ptrArrayMember"}
  // CHECK-NEXT: acc.serial combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) {
  // CHECK-NEXT: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT:  acc.yield
  // CHECK-NEXT:  } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) to varPtr(%[[GETPTRARRAYMEMBER]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "localStruct.ptrArrayMember"}

#pragma acc kernels loop copy(localStruct.ptrArrayMember[3][2])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
  // CHECK-NEXT: %[[TWO_CAST:.*]] = builtin.unrealized_conversion_cast %[[TWO]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS1:.*]] = acc.bounds lowerbound(%[[TWO_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[THREE:.*]] = cir.const #cir.int<3> : !s32i
  // CHECK-NEXT: %[[THREE_CAST:.*]] = builtin.unrealized_conversion_cast %[[THREE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS2:.*]] = acc.bounds lowerbound(%[[THREE_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[GETPTRARRAYMEMBER:.*]] = cir.get_member %[[LOCALSTRUCT]][3] {name = "ptrArrayMember"} : !cir.ptr<!rec_StructTy> -> !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[GETPTRARRAYMEMBER]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) bounds(%[[BOUNDS1]], %[[BOUNDS2]]) -> !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>> {dataClause = #acc<data_clause acc_copy>, name = "localStruct.ptrArrayMember[3][2]"}
  // CHECK-NEXT: acc.kernels combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT:  acc.terminator
  // CHECK-NEXT:  } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) bounds(%[[BOUNDS1]], %[[BOUNDS2]]) to varPtr(%[[GETPTRARRAYMEMBER]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "localStruct.ptrArrayMember[3][2]"}

#pragma acc parallel loop copy(localStruct.ptrArrayMember[1:3][1:1])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST2:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS1:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CAST2]] : si32) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[THREE:.*]] = cir.const #cir.int<3> : !s32i
  // CHECK-NEXT: %[[THREE_CAST:.*]] = builtin.unrealized_conversion_cast %[[THREE]] : !s32i to si32
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS2:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[THREE_CAST]] : si32) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[GETPTRARRAYMEMBER:.*]] = cir.get_member %[[LOCALSTRUCT]][3] {name = "ptrArrayMember"} : !cir.ptr<!rec_StructTy> -> !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[GETPTRARRAYMEMBER]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) bounds(%[[BOUNDS1]], %[[BOUNDS2]]) -> !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>> {dataClause = #acc<data_clause acc_copy>, name = "localStruct.ptrArrayMember[1:3][1:1]"}
  // CHECK-NEXT: acc.parallel combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT:  acc.yield
  // CHECK-NEXT:  } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) bounds(%[[BOUNDS1]], %[[BOUNDS2]]) to varPtr(%[[GETPTRARRAYMEMBER]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "localStruct.ptrArrayMember[1:3][1:1]"}

#pragma acc serial loop copy(localStruct.ptrPtrMember)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[GETPTRPTRMEMBER:.*]] = cir.get_member %[[LOCALSTRUCT]][4] {name = "ptrPtrMember"} : !cir.ptr<!rec_StructTy> -> !cir.ptr<!cir.ptr<!cir.ptr<!cir.double>>>
  // CHECK-NEXT:  %[[COPYIN1:.*]] = acc.copyin varPtr(%[[GETPTRPTRMEMBER]] : !cir.ptr<!cir.ptr<!cir.ptr<!cir.double>>>) -> !cir.ptr<!cir.ptr<!cir.ptr<!cir.double>>> {dataClause = #acc<data_clause acc_copy>, name = "localStruct.ptrPtrMember"}
  // CHECK-NEXT:  acc.serial combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.ptr<!cir.ptr<!cir.double>>>) {
  // CHECK-NEXT: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT:  acc.yield
  // CHECK-NEXT:  } loc
  // CHECK-NEXT:  acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.ptr<!cir.ptr<!cir.double>>>) to varPtr(%[[GETPTRPTRMEMBER]] : !cir.ptr<!cir.ptr<!cir.ptr<!cir.double>>>) {dataClause = #acc<data_clause acc_copy>, name = "localStruct.ptrPtrMember"}

#pragma acc kernels loop copy(localStruct.ptrPtrMember[3][2])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
  // CHECK-NEXT: %[[TWO_CAST:.*]] = builtin.unrealized_conversion_cast %[[TWO]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS1:.*]] = acc.bounds lowerbound(%[[TWO_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[THREE:.*]] = cir.const #cir.int<3> : !s32i
  // CHECK-NEXT: %[[THREE_CAST:.*]] = builtin.unrealized_conversion_cast %[[THREE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS2:.*]] = acc.bounds lowerbound(%[[THREE_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[GETPTRPTRMEMBER:.*]] = cir.get_member %[[LOCALSTRUCT]][4] {name = "ptrPtrMember"} : !cir.ptr<!rec_StructTy> -> !cir.ptr<!cir.ptr<!cir.ptr<!cir.double>>>
  // CHECK-NEXT:  %[[COPYIN1:.*]] = acc.copyin varPtr(%[[GETPTRPTRMEMBER]] : !cir.ptr<!cir.ptr<!cir.ptr<!cir.double>>>) bounds(%[[BOUNDS1]], %[[BOUNDS2]]) -> !cir.ptr<!cir.ptr<!cir.ptr<!cir.double>>> {dataClause = #acc<data_clause acc_copy>, name = "localStruct.ptrPtrMember[3][2]"}
  // CHECK-NEXT:  acc.kernels combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.ptr<!cir.ptr<!cir.double>>>) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT:  acc.terminator
  // CHECK-NEXT:  } loc
  // CHECK-NEXT:  acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.ptr<!cir.ptr<!cir.double>>>) bounds(%[[BOUNDS1]], %[[BOUNDS2]]) to varPtr(%[[GETPTRPTRMEMBER]] : !cir.ptr<!cir.ptr<!cir.ptr<!cir.double>>>) {dataClause = #acc<data_clause acc_copy>, name = "localStruct.ptrPtrMember[3][2]"}

#pragma acc parallel loop copy(localStruct.ptrPtrMember[1:3][1:1])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST2:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS1:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CAST2]] : si32) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[THREE:.*]] = cir.const #cir.int<3> : !s32i
  // CHECK-NEXT: %[[THREE_CAST:.*]] = builtin.unrealized_conversion_cast %[[THREE]] : !s32i to si32
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS2:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[THREE_CAST]] : si32) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[GETPTRPTRMEMBER:.*]] = cir.get_member %[[LOCALSTRUCT]][4] {name = "ptrPtrMember"} : !cir.ptr<!rec_StructTy> -> !cir.ptr<!cir.ptr<!cir.ptr<!cir.double>>>
  // CHECK-NEXT:  %[[COPYIN1:.*]] = acc.copyin varPtr(%[[GETPTRPTRMEMBER]] : !cir.ptr<!cir.ptr<!cir.ptr<!cir.double>>>) bounds(%[[BOUNDS1]], %[[BOUNDS2]]) -> !cir.ptr<!cir.ptr<!cir.ptr<!cir.double>>> {dataClause = #acc<data_clause acc_copy>, name = "localStruct.ptrPtrMember[1:3][1:1]"}
  // CHECK-NEXT:  acc.parallel combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.ptr<!cir.ptr<!cir.double>>>) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT:  acc.yield
  // CHECK-NEXT:  } loc
  // CHECK-NEXT:  acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.ptr<!cir.ptr<!cir.double>>>) bounds(%[[BOUNDS1]], %[[BOUNDS2]]) to varPtr(%[[GETPTRPTRMEMBER]] : !cir.ptr<!cir.ptr<!cir.ptr<!cir.double>>>) {dataClause = #acc<data_clause acc_copy>, name = "localStruct.ptrPtrMember[1:3][1:1]"}
}

typedef struct InnerTy {
  int a;
  int b;
} Inner;

typedef struct OuterTy {
  Inner inner[4];
} Outer;

void copy_member_of_array_element_member() {
  // CHECK: cir.func{{.*}} @copy_member_of_array_element_member()
  Outer outer;
  // CHECK-NEXT: %[[OUTER:.*]] = cir.alloca !rec_OuterTy, !cir.ptr<!rec_OuterTy>, ["outer"]

  #pragma acc parallel loop copy(outer.inner[2].b)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
  // CHECK-NEXT: %[[GETINNER:.*]] = cir.get_member %[[OUTER]][0] {name = "inner"} : !cir.ptr<!rec_OuterTy> -> !cir.ptr<!cir.array<!rec_InnerTy x 4>>
  // CHECK-NEXT: %[[INNERDECAY:.*]] = cir.cast array_to_ptrdecay %[[GETINNER]] : !cir.ptr<!cir.array<!rec_InnerTy x 4>> -> !cir.ptr<!rec_InnerTy>
  // CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride %[[INNERDECAY]], %[[TWO]] : (!cir.ptr<!rec_InnerTy>, !s32i) -> !cir.ptr<!rec_InnerTy>
  // CHECK-NEXT: %[[GETB:.*]] = cir.get_member %[[STRIDE]][1] {name = "b"} : !cir.ptr<!rec_InnerTy> -> !cir.ptr<!s32i>
  // CHECK-NEXT:  %[[COPYIN1:.*]] = acc.copyin varPtr(%[[GETB]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copy>, name = "outer.inner[2].b"}
  // CHECK-NEXT:  acc.parallel combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT:  acc.yield
  // CHECK-NEXT:  } loc
  // CHECK-NEXT:  acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!s32i>) to varPtr(%[[GETB]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copy>, name = "outer.inner[2].b"}
}

void modifier_list() {
  // CHECK: cir.func{{.*}} @modifier_list()
  int localVar;
  // CHECK-NEXT: %[[LOCALVAR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["localVar"]

#pragma acc parallel loop copy(always:localVar)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCALVAR]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copy>, modifiers = #acc<data_clause_modifier always>, name = "localVar"}
  // CHECK-NEXT: acc.parallel combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!s32i>) to varPtr(%[[LOCALVAR]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copy>, modifiers = #acc<data_clause_modifier always>, name = "localVar"}
#pragma acc serial loop copy(alwaysin:localVar)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCALVAR]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copy>, modifiers = #acc<data_clause_modifier alwaysin>, name = "localVar"}
  // CHECK-NEXT: acc.serial combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!s32i>) to varPtr(%[[LOCALVAR]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copy>, modifiers = #acc<data_clause_modifier alwaysin>, name = "localVar"}
#pragma acc kernels loop copy(alwaysout:localVar)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCALVAR]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copy>, modifiers = #acc<data_clause_modifier alwaysout>, name = "localVar"}
  // CHECK-NEXT: acc.kernels combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!s32i>) to varPtr(%[[LOCALVAR]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copy>, modifiers = #acc<data_clause_modifier alwaysout>, name = "localVar"}
#pragma acc parallel loop copy(capture:localVar)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCALVAR]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copy>, modifiers = #acc<data_clause_modifier capture>, name = "localVar"}
  // CHECK-NEXT: acc.parallel combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!s32i>) to varPtr(%[[LOCALVAR]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copy>, modifiers = #acc<data_clause_modifier capture>, name = "localVar"}
#pragma acc serial loop copy(capture, always, alwaysin, alwaysout:localVar)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCALVAR]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copy>, modifiers = #acc<data_clause_modifier always,capture>, name = "localVar"}
  // CHECK-NEXT: acc.serial combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!s32i>) to varPtr(%[[LOCALVAR]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copy>, modifiers = #acc<data_clause_modifier always,capture>, name = "localVar"}
}
