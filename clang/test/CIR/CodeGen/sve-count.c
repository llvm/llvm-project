// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve \
// RUN:   -target-feature +sme2 -fclangir -emit-cir -o %t.cir %s
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve \
// RUN:   -target-feature +sme2 -fclangir -emit-llvm -o %t-cir.ll %s
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve \
// RUN:   -target-feature +sme2 -emit-llvm -o %t.ll %s
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// REQUIRES: aarch64-registered-target

void copy_count(void) {
  __SVCount_t c;
  __SVCount_t d = c;
}

// CIR-LABEL: cir.func {{.*}} @copy_count()
// CIR:         %[[C:.*]] = cir.alloca "c" align(2) : !cir.ptr<!cir.sve_count>
// CIR:         %[[D:.*]] = cir.alloca "d" align(2) init : !cir.ptr<!cir.sve_count>
// CIR:         %[[VAL:.*]] = cir.load align(2) %[[C]] : !cir.ptr<!cir.sve_count>, !cir.sve_count
// CIR:         cir.store align(2) %[[VAL]], %[[D]] : !cir.sve_count, !cir.ptr<!cir.sve_count>

// LLVM-LABEL: define dso_local void @copy_count()
// LLVM:         %[[C:.*]] = alloca target("aarch64.svcount"), align 2
// LLVM:         %[[D:.*]] = alloca target("aarch64.svcount"), align 2
// LLVM:         %[[VAL:.*]] = load target("aarch64.svcount"), ptr %[[C]], align 2
// LLVM:         store target("aarch64.svcount") %[[VAL]], ptr %[[D]], align 2
