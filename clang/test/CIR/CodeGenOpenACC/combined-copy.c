// RUN: %clang_cc1 -fopenacc -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir %s -o - | FileCheck %s

int global;
void acc_compute(int parmVar) {
  // CHECK: cir.func @acc_compute(%[[ARG:.*]]: !s32i{{.*}}) {
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
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCAL1]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copy>, name = "localVar1"} loc
  // CHECK-NEXT: %[[COPYIN2:.*]] = acc.copyin varPtr(%[[LOCAL2]] : !cir.ptr<!s16i>) -> !cir.ptr<!s16i> {dataClause = #acc<data_clause acc_copy>, name = "localVar2"} loc
  // CHECK-NEXT: %[[COPYIN3:.*]] = acc.copyin varPtr(%[[LOCAL3]] : !cir.ptr<!cir.float>) -> !cir.ptr<!cir.float> {dataClause = #acc<data_clause acc_copy>, name = "localVar3"} loc
  // CHECK-NEXT: acc.parallel combined(loop) dataOperands(%[[COPYIN1]], %[[COPYIN2]], %[[COPYIN3]] : !cir.ptr<!s32i>, !cir.ptr<!s16i>, !cir.ptr<!cir.float>) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN3]] : !cir.ptr<!cir.float>) to varPtr(%[[LOCAL3]] : !cir.ptr<!cir.float>) {dataClause = #acc<data_clause acc_copy>, name = "localVar3"} loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN2]] : !cir.ptr<!s16i>) to varPtr(%[[LOCAL2]] : !cir.ptr<!s16i>) {dataClause = #acc<data_clause acc_copy>, name = "localVar2"} loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!s32i>) to varPtr(%[[LOCAL1]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copy>, name = "localVar1"} loc

#pragma acc serial loop copy(always, alwaysin, alwaysout: localVar1)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCAL1]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copy>, name = "localVar1"} loc
  // CHECK-NEXT: acc.serial combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!s32i>) to varPtr(%[[LOCAL1]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copy>, name = "localVar1"} loc

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
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCALARRAY]] : !cir.ptr<!cir.array<!cir.float x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!cir.float x 5>> {dataClause = #acc<data_clause acc_copy>, name = "localArray"} loc
  // CHECK-NEXT: acc.parallel combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.float x 5>>) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.float x 5>>) bounds(%[[BOUNDS]]) to varPtr(%[[LOCALARRAY]] : !cir.ptr<!cir.array<!cir.float x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "localArray"} loc
  
#pragma acc serial loop copy(localArray[1:3])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[THREE:.*]] = cir.const #cir.int<3> : !s32i
  // CHECK-NEXT: %[[THREE_CAST:.*]] = builtin.unrealized_conversion_cast %[[THREE]] : !s32i to si32
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[THREE_CAST]] : si32) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64) loc
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCALARRAY]] : !cir.ptr<!cir.array<!cir.float x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!cir.float x 5>> {dataClause = #acc<data_clause acc_copy>, name = "localArray"} loc
  // CHECK-NEXT: acc.serial combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.float x 5>>) {
  // CHECK-NEXT: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.float x 5>>) bounds(%[[BOUNDS]]) to varPtr(%[[LOCALARRAY]] : !cir.ptr<!cir.array<!cir.float x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "localArray"} loc

#pragma acc kernels loop copy(localArray[:3])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[THREE:.*]] = cir.const #cir.int<3> : !s32i
  // CHECK-NEXT: %[[THREE_CAST:.*]] = builtin.unrealized_conversion_cast %[[THREE]] : !s32i to si32
  // CHECK-NEXT: %[[ZERO_CONST2:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ZERO_CONST]] : i64) extent(%[[THREE_CAST]] : si32) stride(%[[ONE_CONST]] : i64) startIdx(%[[ZERO_CONST2]] : i64) loc
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCALARRAY]] : !cir.ptr<!cir.array<!cir.float x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!cir.float x 5>> {dataClause = #acc<data_clause acc_copy>, name = "localArray"} loc
  // CHECK-NEXT: acc.kernels combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.float x 5>>) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.float x 5>>) bounds(%[[BOUNDS]]) to varPtr(%[[LOCALARRAY]] : !cir.ptr<!cir.array<!cir.float x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "localArray"} loc

#pragma acc parallel loop copy(localArray[1:])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[FOUR_CONST:.*]] = arith.constant 4 : i64
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) upperbound(%[[FOUR_CONST]] : i64) stride(%[[ONE_CONST]] : i64) startIdx(%[[ZERO_CONST]] : i64) loc
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCALARRAY]] : !cir.ptr<!cir.array<!cir.float x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!cir.float x 5>> {dataClause = #acc<data_clause acc_copy>, name = "localArray"} loc
  // CHECK-NEXT: acc.parallel combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.float x 5>>) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.float x 5>>) bounds(%[[BOUNDS]]) to varPtr(%[[LOCALARRAY]] : !cir.ptr<!cir.array<!cir.float x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "localArray"} loc

#pragma acc serial loop copy(localArray[localVar1:localVar2])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[LV1:.*]] = cir.load{{.*}} %[[LOCAL1]] : !cir.ptr<!s32i>, !s32i 
  // CHECK-NEXT: %[[LV1_CAST:.*]] = builtin.unrealized_conversion_cast %[[LV1]] : !s32i to si32
  // CHECK-NEXT: %[[LV2:.*]] = cir.load{{.*}} %[[LOCAL2]] : !cir.ptr<!s16i>, !s16i 
  // CHECK-NEXT: %[[LV2_CAST:.*]] = builtin.unrealized_conversion_cast %[[LV2]] : !s16i to si16
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[LV1_CAST]] : si32) extent(%[[LV2_CAST]] : si16) stride(%[[ONE_CONST]] : i64) startIdx(%[[ZERO_CONST]] : i64) loc
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCALARRAY]] : !cir.ptr<!cir.array<!cir.float x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!cir.float x 5>> {dataClause = #acc<data_clause acc_copy>, name = "localArray"} loc
  // CHECK-NEXT: acc.serial combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.float x 5>>) {
  // CHECK-NEXT: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.float x 5>>) bounds(%[[BOUNDS]]) to varPtr(%[[LOCALARRAY]] : !cir.ptr<!cir.array<!cir.float x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "localArray"} loc

#pragma acc kernels loop copy(localArray[:localVar2])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[LV2:.*]] = cir.load{{.*}} %[[LOCAL2]] : !cir.ptr<!s16i>, !s16i 
  // CHECK-NEXT: %[[LV2_CAST:.*]] = builtin.unrealized_conversion_cast %[[LV2]] : !s16i to si16
  // CHECK-NEXT: %[[ZERO_CONST2:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ZERO_CONST]] : i64) extent(%[[LV2_CAST]] : si16) stride(%[[ONE_CONST]] : i64) startIdx(%[[ZERO_CONST2]] : i64) loc
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCALARRAY]] : !cir.ptr<!cir.array<!cir.float x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!cir.float x 5>> {dataClause = #acc<data_clause acc_copy>, name = "localArray"} loc
  // CHECK-NEXT: acc.kernels combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.float x 5>>) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.float x 5>>) bounds(%[[BOUNDS]]) to varPtr(%[[LOCALARRAY]] : !cir.ptr<!cir.array<!cir.float x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "localArray"} loc

#pragma acc parallel loop copy(localArray[localVar1:])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[LV1:.*]] = cir.load{{.*}} %[[LOCAL1]] : !cir.ptr<!s32i>, !s32i 
  // CHECK-NEXT: %[[LV1_CAST:.*]] = builtin.unrealized_conversion_cast %[[LV1]] : !s32i to si32
  // CHECK-NEXT: %[[FOUR_CONST:.*]] = arith.constant 4 : i64
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[LV1_CAST]] : si32) upperbound(%[[FOUR_CONST]] : i64) stride(%[[ONE_CONST]] : i64) startIdx(%[[ZERO_CONST]] : i64) loc
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCALARRAY]] : !cir.ptr<!cir.array<!cir.float x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!cir.float x 5>> {dataClause = #acc<data_clause acc_copy>, name = "localArray"} loc
  // CHECK-NEXT: acc.parallel combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.float x 5>>) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.float x 5>>) bounds(%[[BOUNDS]]) to varPtr(%[[LOCALARRAY]] : !cir.ptr<!cir.array<!cir.float x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "localArray"} loc

#pragma acc serial loop copy(localPointer[3])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[THREE:.*]] = cir.const #cir.int<3> : !s32i
  // CHECK-NEXT: %[[THREE_CAST:.*]] = builtin.unrealized_conversion_cast %[[THREE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[THREE_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64) loc
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCALPTR]] : !cir.ptr<!cir.ptr<!s16i>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.ptr<!s16i>> {dataClause = #acc<data_clause acc_copy>, name = "localPointer"} loc
  // CHECK-NEXT: acc.serial combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.ptr<!s16i>>) {
  // CHECK-NEXT: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.ptr<!s16i>>) bounds(%[[BOUNDS]]) to varPtr(%[[LOCALPTR]] : !cir.ptr<!cir.ptr<!s16i>>) {dataClause = #acc<data_clause acc_copy>, name = "localPointer"} loc

#pragma acc kernels loop copy(localPointer[1:3])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[THREE:.*]] = cir.const #cir.int<3> : !s32i
  // CHECK-NEXT: %[[THREE_CAST:.*]] = builtin.unrealized_conversion_cast %[[THREE]] : !s32i to si32
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[THREE_CAST]] : si32) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64) loc
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCALPTR]] : !cir.ptr<!cir.ptr<!s16i>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.ptr<!s16i>> {dataClause = #acc<data_clause acc_copy>, name = "localPointer"} loc
  // CHECK-NEXT: acc.kernels combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.ptr<!s16i>>) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.ptr<!s16i>>) bounds(%[[BOUNDS]]) to varPtr(%[[LOCALPTR]] : !cir.ptr<!cir.ptr<!s16i>>) {dataClause = #acc<data_clause acc_copy>, name = "localPointer"} loc

#pragma acc parallel loop copy(localPointer[:3])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[THREE:.*]] = cir.const #cir.int<3> : !s32i
  // CHECK-NEXT: %[[THREE_CAST:.*]] = builtin.unrealized_conversion_cast %[[THREE]] : !s32i to si32
  // CHECK-NEXT: %[[ZERO_CONST2:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ZERO_CONST]] : i64) extent(%[[THREE_CAST]] : si32) stride(%[[ONE_CONST]] : i64) startIdx(%[[ZERO_CONST2]] : i64) loc
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCALPTR]] : !cir.ptr<!cir.ptr<!s16i>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.ptr<!s16i>> {dataClause = #acc<data_clause acc_copy>, name = "localPointer"} loc
  // CHECK-NEXT: acc.parallel combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.ptr<!s16i>>) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.ptr<!s16i>>) bounds(%[[BOUNDS]]) to varPtr(%[[LOCALPTR]] : !cir.ptr<!cir.ptr<!s16i>>) {dataClause = #acc<data_clause acc_copy>, name = "localPointer"} loc

#pragma acc serial loop copy(localPointer[localVar1:localVar2])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[LV1:.*]] = cir.load{{.*}} %[[LOCAL1]] : !cir.ptr<!s32i>, !s32i 
  // CHECK-NEXT: %[[LV1_CAST:.*]] = builtin.unrealized_conversion_cast %[[LV1]] : !s32i to si32
  // CHECK-NEXT: %[[LV2:.*]] = cir.load{{.*}} %[[LOCAL2]] : !cir.ptr<!s16i>, !s16i 
  // CHECK-NEXT: %[[LV2_CAST:.*]] = builtin.unrealized_conversion_cast %[[LV2]] : !s16i to si16
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[LV1_CAST]] : si32) extent(%[[LV2_CAST]] : si16) stride(%[[ONE_CONST]] : i64) startIdx(%[[ZERO_CONST]] : i64) loc
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCALPTR]] : !cir.ptr<!cir.ptr<!s16i>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.ptr<!s16i>> {dataClause = #acc<data_clause acc_copy>, name = "localPointer"} loc
  // CHECK-NEXT: acc.serial combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.ptr<!s16i>>) {
  // CHECK-NEXT: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.ptr<!s16i>>) bounds(%[[BOUNDS]]) to varPtr(%[[LOCALPTR]] : !cir.ptr<!cir.ptr<!s16i>>) {dataClause = #acc<data_clause acc_copy>, name = "localPointer"} loc

#pragma acc kernels loop copy(localPointer[:localVar2])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[LV2:.*]] = cir.load{{.*}} %[[LOCAL2]] : !cir.ptr<!s16i>, !s16i 
  // CHECK-NEXT: %[[LV2_CAST:.*]] = builtin.unrealized_conversion_cast %[[LV2]] : !s16i to si16
  // CHECK-NEXT: %[[ZERO_CONST2:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ZERO_CONST]] : i64) extent(%[[LV2_CAST]] : si16) stride(%[[ONE_CONST]] : i64) startIdx(%[[ZERO_CONST2]] : i64) loc
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCALPTR]] : !cir.ptr<!cir.ptr<!s16i>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.ptr<!s16i>> {dataClause = #acc<data_clause acc_copy>, name = "localPointer"} loc
  // CHECK-NEXT: acc.kernels combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.ptr<!s16i>>) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.ptr<!s16i>>) bounds(%[[BOUNDS]]) to varPtr(%[[LOCALPTR]] : !cir.ptr<!cir.ptr<!s16i>>) {dataClause = #acc<data_clause acc_copy>, name = "localPointer"} loc

  float *localArrayOfPtrs[5];
#pragma acc parallel loop copy(localArrayOfPtrs[3])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[THREE:.*]] = cir.const #cir.int<3> : !s32i
  // CHECK-NEXT: %[[THREE_CAST:.*]] = builtin.unrealized_conversion_cast %[[THREE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[THREE_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64) loc
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCALARRAYOFPTRS]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>> {dataClause = #acc<data_clause acc_copy>, name = "localArrayOfPtrs"} loc
  // CHECK-NEXT: acc.parallel combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) bounds(%[[BOUNDS]]) to varPtr(%[[LOCALARRAYOFPTRS]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "localArrayOfPtrs"} loc

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
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCALARRAYOFPTRS]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) bounds(%[[BOUNDS]], %[[BOUNDS2]]) -> !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>> {dataClause = #acc<data_clause acc_copy>, name = "localArrayOfPtrs"} loc
  // CHECK-NEXT: acc.serial combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) {
  // CHECK-NEXT: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) bounds(%[[BOUNDS]], %[[BOUNDS2]]) to varPtr(%[[LOCALARRAYOFPTRS]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "localArrayOfPtrs"} loc

#pragma acc kernels loop copy(localArrayOfPtrs[localVar1:localVar2])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[LV1:.*]] = cir.load{{.*}} %[[LOCAL1]] : !cir.ptr<!s32i>, !s32i 
  // CHECK-NEXT: %[[LV1_CAST:.*]] = builtin.unrealized_conversion_cast %[[LV1]] : !s32i to si32
  // CHECK-NEXT: %[[LV2:.*]] = cir.load{{.*}} %[[LOCAL2]] : !cir.ptr<!s16i>, !s16i 
  // CHECK-NEXT: %[[LV2_CAST:.*]] = builtin.unrealized_conversion_cast %[[LV2]] : !s16i to si16
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[LV1_CAST]] : si32) extent(%[[LV2_CAST]] : si16) stride(%[[ONE_CONST]] : i64) startIdx(%[[ZERO_CONST]] : i64) loc
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCALARRAYOFPTRS]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>> {dataClause = #acc<data_clause acc_copy>, name = "localArrayOfPtrs"} loc
  // CHECK-NEXT: acc.kernels combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) bounds(%[[BOUNDS]]) to varPtr(%[[LOCALARRAYOFPTRS]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "localArrayOfPtrs"} loc

#pragma acc parallel loop copy(localArrayOfPtrs[localVar1:])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[LV1:.*]] = cir.load{{.*}} %[[LOCAL1]] : !cir.ptr<!s32i>, !s32i 
  // CHECK-NEXT: %[[LV1_CAST:.*]] = builtin.unrealized_conversion_cast %[[LV1]] : !s32i to si32
  // CHECK-NEXT: %[[FOUR_CONST:.*]] = arith.constant 4 : i64
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[LV1_CAST]] : si32) upperbound(%[[FOUR_CONST]] : i64) stride(%[[ONE_CONST]] : i64) startIdx(%[[ZERO_CONST]] : i64) loc
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCALARRAYOFPTRS]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>> {dataClause = #acc<data_clause acc_copy>, name = "localArrayOfPtrs"} loc
  // CHECK-NEXT: acc.parallel combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) bounds(%[[BOUNDS]]) to varPtr(%[[LOCALARRAYOFPTRS]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "localArrayOfPtrs"} loc

#pragma acc serial loop copy(localArrayOfPtrs[:localVar2])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[LV2:.*]] = cir.load{{.*}} %[[LOCAL2]] : !cir.ptr<!s16i>, !s16i 
  // CHECK-NEXT: %[[LV2_CAST:.*]] = builtin.unrealized_conversion_cast %[[LV2]] : !s16i to si16
  // CHECK-NEXT: %[[ZERO_CONST2:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ZERO_CONST]] : i64) extent(%[[LV2_CAST]] : si16) stride(%[[ONE_CONST]] : i64) startIdx(%[[ZERO_CONST2]] : i64) loc
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCALARRAYOFPTRS]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>> {dataClause = #acc<data_clause acc_copy>, name = "localArrayOfPtrs"} loc
  // CHECK-NEXT: acc.serial combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) {
  // CHECK-NEXT: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) bounds(%[[BOUNDS]]) to varPtr(%[[LOCALARRAYOFPTRS]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "localArrayOfPtrs"} loc

#pragma acc kernels loop copy(localArrayOfPtrs[localVar1])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[LV1:.*]] = cir.load{{.*}} %[[LOCAL1]] : !cir.ptr<!s32i>, !s32i 
  // CHECK-NEXT: %[[LV1_CAST:.*]] = builtin.unrealized_conversion_cast %[[LV1]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[LV1_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64) loc
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCALARRAYOFPTRS]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>> {dataClause = #acc<data_clause acc_copy>, name = "localArrayOfPtrs"} loc
  // CHECK-NEXT: acc.kernels combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) bounds(%[[BOUNDS]]) to varPtr(%[[LOCALARRAYOFPTRS]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "localArrayOfPtrs"} loc

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
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCALARRAYOFPTRS]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) bounds(%[[BOUNDS]], %[[BOUNDS2]]) -> !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>> {dataClause = #acc<data_clause acc_copy>, name = "localArrayOfPtrs"} loc
  // CHECK-NEXT: acc.parallel combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) bounds(%[[BOUNDS]], %[[BOUNDS2]]) to varPtr(%[[LOCALARRAYOFPTRS]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "localArrayOfPtrs"} loc
  
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
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCALARRAYOFPTRS]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) bounds(%[[BOUNDS]], %[[BOUNDS2]]) -> !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>> {dataClause = #acc<data_clause acc_copy>, name = "localArrayOfPtrs"} loc
  // CHECK-NEXT: acc.serial combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) {
  // CHECK-NEXT: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) bounds(%[[BOUNDS]], %[[BOUNDS2]]) to varPtr(%[[LOCALARRAYOFPTRS]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "localArrayOfPtrs"} loc
 
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
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCALARRAYOFPTRS]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) bounds(%[[BOUNDS]], %[[BOUNDS2]]) -> !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>> {dataClause = #acc<data_clause acc_copy>, name = "localArrayOfPtrs"} loc
  // CHECK-NEXT: acc.kernels combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) bounds(%[[BOUNDS]], %[[BOUNDS2]]) to varPtr(%[[LOCALARRAYOFPTRS]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "localArrayOfPtrs"} loc

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
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCALARRAYOFPTRS]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) bounds(%[[BOUNDS]], %[[BOUNDS2]]) -> !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>> {dataClause = #acc<data_clause acc_copy>, name = "localArrayOfPtrs"} loc
  // CHECK-NEXT: acc.parallel combined(loop)  dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) bounds(%[[BOUNDS]], %[[BOUNDS2]]) to varPtr(%[[LOCALARRAYOFPTRS]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "localArrayOfPtrs"} loc

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
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[LOCALARRAYOFPTRS]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) bounds(%[[BOUNDS]], %[[BOUNDS2]]) -> !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>> {dataClause = #acc<data_clause acc_copy>, name = "localArrayOfPtrs"} loc
  // CHECK-NEXT: acc.serial combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) {
  // CHECK-NEXT: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) bounds(%[[BOUNDS]], %[[BOUNDS2]]) to varPtr(%[[LOCALARRAYOFPTRS]] : !cir.ptr<!cir.array<!cir.ptr<!cir.float> x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "localArrayOfPtrs"} loc

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
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[THREEDARRAY]] : !cir.ptr<!cir.array<!cir.array<!cir.array<!cir.double x 7> x 6> x 5>>) bounds(%[[BOUNDS]], %[[BOUNDS2]], %[[BOUNDS3]]) -> !cir.ptr<!cir.array<!cir.array<!cir.array<!cir.double x 7> x 6> x 5>> {dataClause = #acc<data_clause acc_copy>, name = "threeDArray"} loc
  // CHECK-NEXT: acc.kernels combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.array<!cir.array<!cir.double x 7> x 6> x 5>>) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.array<!cir.array<!cir.double x 7> x 6> x 5>>) bounds(%[[BOUNDS]], %[[BOUNDS2]], %[[BOUNDS3]]) to varPtr(%[[THREEDARRAY]] : !cir.ptr<!cir.array<!cir.array<!cir.array<!cir.double x 7> x 6> x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "threeDArray"} loc

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
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[THREEDARRAY]] : !cir.ptr<!cir.array<!cir.array<!cir.array<!cir.double x 7> x 6> x 5>>) bounds(%[[BOUNDS]], %[[BOUNDS2]], %[[BOUNDS3]]) -> !cir.ptr<!cir.array<!cir.array<!cir.array<!cir.double x 7> x 6> x 5>> {dataClause = #acc<data_clause acc_copy>, name = "threeDArray"} loc
  // CHECK-NEXT: acc.parallel combined(loop) dataOperands(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.array<!cir.array<!cir.double x 7> x 6> x 5>>) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!cir.array<!cir.array<!cir.array<!cir.double x 7> x 6> x 5>>) bounds(%[[BOUNDS]], %[[BOUNDS2]], %[[BOUNDS3]]) to varPtr(%[[THREEDARRAY]] : !cir.ptr<!cir.array<!cir.array<!cir.array<!cir.double x 7> x 6> x 5>>) {dataClause = #acc<data_clause acc_copy>, name = "threeDArray"} loc
}
