// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

struct StructWithDestructor {
  ~StructWithDestructor();
};

void after();

void test_labeled_block() {
lbl: {
    StructWithDestructor a;
  }
  after();
}

// CIR-LABEL: cir.func {{.*}} @_Z18test_labeled_blockv(
// CIR:         cir.label "lbl"
// CIR-NEXT:    cir.scope {
// CIR-NEXT:      %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!rec_StructWithDestructor>
// CIR-NEXT:      cir.cleanup.scope {
// CIR-NEXT:        cir.yield
// CIR-NEXT:      } cleanup normal {
// CIR-NEXT:        cir.call @_ZN20StructWithDestructorD1Ev(%[[A_ADDR]])
// CIR-NEXT:        cir.yield
// CIR-NEXT:      }
// CIR-NEXT:    }
// CIR-NEXT:    cir.call @_Z5afterv()
// CIR-NEXT:    cir.return

// LLVM-LABEL: define{{.*}} void @_Z18test_labeled_blockv()
// LLVM:         %[[A_ADDR:.*]] = alloca %struct.StructWithDestructor
// LLVM:         call void @_ZN20StructWithDestructorD1Ev(ptr {{.*}} %[[A_ADDR]])
// LLVM:         call void @_Z5afterv()
// LLVM-NEXT:    ret void

void test_attributed_block() {
  [[likely]] {
    StructWithDestructor a;
  }
  after();
}

// CIR-LABEL: cir.func {{.*}} @_Z21test_attributed_blockv(
// CIR-NEXT:    cir.scope {
// CIR-NEXT:      %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!rec_StructWithDestructor>
// CIR-NEXT:      cir.cleanup.scope {
// CIR-NEXT:        cir.yield
// CIR-NEXT:      } cleanup normal {
// CIR-NEXT:        cir.call @_ZN20StructWithDestructorD1Ev(%[[A_ADDR]])
// CIR-NEXT:        cir.yield
// CIR-NEXT:      }
// CIR-NEXT:    }
// CIR-NEXT:    cir.call @_Z5afterv()
// CIR-NEXT:    cir.return

// LLVM-LABEL: define{{.*}} void @_Z21test_attributed_blockv()
// LLVM:         %[[A_ADDR:.*]] = alloca %struct.StructWithDestructor
// LLVM:         call void @_ZN20StructWithDestructorD1Ev(ptr {{.*}} %[[A_ADDR]])
// LLVM:         call void @_Z5afterv()
// LLVM-NEXT:    ret void
