// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR --implicit-check-not=externNotCalled \
// RUN:   --implicit-check-not=internalNotCalled --implicit-check-not=inlineNotCalled

extern int externCalled();
extern int externNotCalled();

namespace {
  int internalCalled() { return 1; }
  int internalNotCalled() { return 2; }
}

struct S {
  int inlineCalled() { return 3; }
  int inlineNotCalled() { return 4; }
};

void use() {
  S s;
  externCalled();
  internalCalled();
  s.inlineCalled();
}

// CIR: cir.func{{.*}} @_Z12externCalledv
// This shouldn't have a body.
// CIR-NOT: cir.return

// CIR: cir.func{{.*}} @_ZN12_GLOBAL__N_114internalCalledEv
// CIR:   %[[ONE:.*]] = cir.const #cir.int<1>
// CIR:   cir.store %[[ONE]], %[[RET_ADDR:.*]]

// CIR: cir.func{{.*}} @_ZN1S12inlineCalledEv
// CIR:   %[[THIS:.*]] = cir.alloca !cir.ptr<!rec_S>, !cir.ptr<!cir.ptr<!rec_S>>, ["this", init]
// CIR:   %[[THREE:.*]] = cir.const #cir.int<3>
// CIR:   cir.store %[[THREE]], %[[RET_ADDR:.*]]

// CIR: cir.func{{.*}} @_Z3usev()
