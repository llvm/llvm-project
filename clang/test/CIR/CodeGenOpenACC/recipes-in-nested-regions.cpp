// RUN: %clang_cc1 -fopenacc -triple x86_64-linux-gnu -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// CHECK: acc.firstprivate.recipe @firstprivatization__ZTSi : !cir.ptr<!s32i> init {
// CHECK: acc.private.recipe @privatization__ZTSi : !cir.ptr<!s32i> init {
// CHECK: acc.reduction.recipe @reduction_max__ZTSi : !cir.ptr<!s32i> reduction_operator <max> init {

void in_while(int c, int x) {
  // CHECK-LABEL: cir.func{{.*}} @_Z8in_whileii(
  // CHECK: %[[X:.*]] = cir.alloca "x"
  // CHECK: cir.while {
  // CHECK: } do {
  // CHECK: acc.parallel combined(loop) {
  // CHECK-NEXT: %[[RED:.*]] = acc.reduction varPtr(%[[X]] : !cir.ptr<!s32i>) recipe(@reduction_max__ZTSi) name("x") -> !cir.ptr<!s32i>
  // CHECK-NEXT: acc.loop combined(parallel) reduction(%[[RED]] : !cir.ptr<!s32i>) {
  while (c) {
#pragma acc parallel loop reduction(max:x)
    for (int i = 0; i < 4; ++i)
      x = i;
  }
}

void in_if(int c, int x) {
  // CHECK-LABEL: cir.func{{.*}} @_Z5in_ifii(
  // CHECK: %[[X:.*]] = cir.alloca "x"
  // CHECK: cir.if %{{.*}} {
  // CHECK-NEXT: %[[PRIV:.*]] = acc.private varPtr(%[[X]] : !cir.ptr<!s32i>) recipe(@privatization__ZTSi) name("x") -> !cir.ptr<!s32i>
  // CHECK-NEXT: acc.parallel private(%[[PRIV]] : !cir.ptr<!s32i>) {
  if (c) {
#pragma acc parallel private(x)
    x = 1;
  }
}

void in_do(int c, int x) {
  // CHECK-LABEL: cir.func{{.*}} @_Z5in_doii(
  // CHECK: %[[X:.*]] = cir.alloca "x"
  // CHECK: cir.do {
  // CHECK: %[[FPRIV:.*]] = acc.firstprivate varPtr(%[[X]] : !cir.ptr<!s32i>) recipe(@firstprivatization__ZTSi) name("x") -> !cir.ptr<!s32i>
  // CHECK-NEXT: acc.serial firstprivate(%[[FPRIV]] : !cir.ptr<!s32i>) {
  // CHECK: } while {
  do {
#pragma acc serial firstprivate(x)
    x = 1;
  } while (c);
}
