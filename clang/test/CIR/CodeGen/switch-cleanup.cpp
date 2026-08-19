// RUN: %clang_cc1 -std=c++23 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++23 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM,LLVMCIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -std=c++23 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM,OGCG --input-file=%t.ll %s

struct SmallVector {
  ~SmallVector();
};
struct BitVector {
  SmallVector Bits;
};

// The switch body is a single DeclStmt (no compound statement / explicit scope).
void simple_stmt_body(int x) {
  switch (x)
    BitVector bv;
}

// CIR-LABEL: cir.func{{.*}} @_Z16simple_stmt_bodyi
// CIR:         cir.scope {
// CIR:           cir.alloca "bv" {{.*}} : !cir.ptr<!rec_BitVector>
// CIR:           cir.switch(%{{.*}} : !s32i) {
// CIR:             cir.cleanup.scope {
// CIR:               cir.yield
// CIR:             } cleanup normal {
// CIR:               cir.call @_ZN9BitVectorD1Ev
// CIR:               cir.yield
// CIR:             }
// CIR:             cir.yield
// CIR:           }
// CIR:         }
// CIR:         cir.return

// LLVM-LABEL: define{{.*}} void @_Z16simple_stmt_bodyi
// LLVM:         alloca %struct.BitVector
// LLVM:         switch i32 %{{.*}}, label %[[EPILOG:.*]] [
// LLVM:         ]
// The CIR pipeline emits the (dead) destructor cleanup; OGCG drops it entirely.
// This is inside an unreachable block in LLVMCIR.
// LLVMCIR:      call void @_ZN9BitVectorD1Ev
// OGCG:       [[EPILOG]]:
// LLVM:         ret void

// The switch body is a CompoundStmt, so it introduces an explicit scope and the
// declaration is emitted through the unassociated-statement path.
void compound_stmt_body(int x) {
  switch (x) {
    BitVector bv;
  }
}

// CIR-LABEL: cir.func{{.*}} @_Z18compound_stmt_bodyi
// CIR:         cir.scope {
// CIR:           cir.alloca "bv" {{.*}} : !cir.ptr<!rec_BitVector>
// CIR:           cir.switch(%{{.*}} : !s32i) {
// CIR:             cir.cleanup.scope {
// CIR:               cir.yield
// CIR:             } cleanup normal {
// CIR:               cir.call @_ZN9BitVectorD1Ev
// CIR:               cir.yield
// CIR:             }
// CIR:             cir.yield
// CIR:           }
// CIR:         }
// CIR:         cir.return

// LLVM-LABEL: define{{.*}} void @_Z18compound_stmt_bodyi
// LLVM:         alloca %struct.BitVector
// LLVM:         switch i32 %{{.*}}, label %[[EPILOG:.*]] [
// LLVM:         ]
// The CIR pipeline emits the (dead) destructor cleanup; OGCG drops it entirely.
// This is inside an unreachable block in LLVMCIR.
// LLVMCIR:      call void @_ZN9BitVectorD1Ev
// OGCG:       [[EPILOG]]:
// LLVM:         ret void

void cleanup_in_case(int x) {
  switch (x) {
    default:
      break;
    case 0:
      BitVector bv;
      break;
  }
}

// CIR-LABEL: cir.func{{.*}} @_Z15cleanup_in_casei
// CIR:         cir.scope {
// CIR:           cir.alloca "bv" {{.*}} : !cir.ptr<!rec_BitVector>
// CIR:           cir.switch(%{{.*}} : !s32i) {
// CIR:             cir.case(default, []) {
// CIR:               cir.break
// CIR:             }
// CIR:             cir.case(equal, [#cir.int<0> : !s32i]) {
// CIR:               cir.cleanup.scope {
// CIR:                 cir.break
// CIR:               ^bb[[DEAD_BLOCK:.*]]:
// CIR:                 cir.yield
// CIR:               } cleanup normal {
// CIR:                 cir.call @_ZN9BitVectorD1Ev
// CIR:                 cir.yield
// CIR:               }
// CIR:               cir.yield
// CIR:             }
// CIR:             cir.yield
// CIR:           }
// CIR:         }
// CIR:         cir.return

// LLVM-LABEL: define{{.*}} void @_Z15cleanup_in_casei
// LLVMCIR:      %[[CLEANUP_FLAG:.*]] = alloca i32
// LLVM:         alloca %struct.BitVector
// LLVM:         switch i32 %{{.*}}, label %[[DEFAULT:.*]] [
// LLVM:           i32 0, label %[[CASE_ZERO:.*]]
// LLVM:         ]
// LLVM:       [[DEFAULT]]:
// LLVM:         br label %[[EPILOG:.*]]
// LLVM:       [[CASE_ZERO]]:
// LLVMCIR:      store i32 1, ptr %[[CLEANUP_FLAG]]
// LLVMCIR:      br label %[[CASE_ZERO_CLEANUP:.*]]
// LLVMCIR:    [[DEAD_BLOCK:.*]]:
// LLVMCIR:      store i32 0, ptr %[[CLEANUP_FLAG]]
// LLVMCIR:      br label %[[CASE_ZERO_CLEANUP]]
// LLVMCIR:    [[CASE_ZERO_CLEANUP]]:
// LLVM:         call void @_ZN9BitVectorD1Ev
// LLVMCIR:      br label %[[CLEANUP_CONTINUE:.*]]
// LLVM:         br label %[[EPILOG]]
// LLVM:       [[EPILOG]]:
// LLVM:         ret void

void cleanup_in_while_in_switch(int x) {
  switch (x)
    while (x) {
      default:
        break;
      case 0:
        BitVector bv;
        continue;
    }
}

// CIR-LABEL: cir.func{{.*}} @_Z26cleanup_in_while_in_switchi
// CIR:         cir.scope {
// CIR:           cir.switch(%{{.*}} : !s32i) {
// CIR:             cir.scope {
// CIR:               cir.while {
// CIR:                 cir.condition
// CIR:               } do {
// CIR:                 cir.scope {
// CIR:                   cir.alloca "bv" {{.*}} : !cir.ptr<!rec_BitVector>
// CIR:                   cir.case(default, []) {
// CIR:                     cir.break
// CIR:                   }
// CIR:                   cir.case(equal, [#cir.int<0> : !s32i]) {
// CIR:                     cir.cleanup.scope {
// CIR:                       cir.yield
// CIR:                     } cleanup normal {
// CIR:                       cir.call @_ZN9BitVectorD1Ev
// CIR:                       cir.yield
// CIR:                     }
// CIR:                     cir.yield
// CIR:                   }
// CIR:                   cir.continue
// CIR:                 }
// CIR:                 cir.yield
// CIR:               }
// CIR:             }
// CIR:             cir.yield
// CIR:           }
// CIR:         }
// CIR:         cir.return

// LLVM-LABEL: define{{.*}} void @_Z26cleanup_in_while_in_switchi
// LLVM:         alloca %struct.BitVector
// LLVM:         switch i32 %{{.*}}, label %[[DEFAULT:.*]] [
// LLVM:           i32 0, label %[[CASE_ZERO:.*]]
// LLVM:         ]
// LLVM:       [[DEAD_BLOCK:.*]]:
// LLVMCIR:      br label %[[SWITCH_BODY:.*]]
// OGCG:         br label %[[WHILE_COND:.*]]
// LLVMCIR:    [[SWITCH_BODY]]:
// LLVMCIR:      br label %[[WHILE_COND:.*]]
// LLVM:       [[WHILE_COND]]:
// LLVM:         icmp
// LLVM:         br i1 %{{.*}}, label %[[WHILE_BODY:.*]], label %[[WHILE_END:.*]]
// LLVM:       [[WHILE_BODY]]:
// LLVM:         br label %[[DEFAULT]]
// LLVM:       [[DEFAULT]]:
// LLVM:         br label %[[WHILE_END]]
// LLVM:       [[CASE_ZERO]]:
// LLVM:         call void @_ZN9BitVectorD1Ev
// LLVM:         br label %[[WHILE_COND]]
// LLVM:       [[WHILE_END]]:
// LLVM:         br label %[[EPILOG:.*]]
// LLVM:       [[EPILOG]]:
// LLVM:         ret void
