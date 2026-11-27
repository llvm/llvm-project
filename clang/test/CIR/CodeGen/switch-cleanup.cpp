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

// The switch body is the loop that holds the cases, so the loop is hoisted out
// of the switch as pre-case code. Each case stores its destination id and
// leaves the switch, and the ids route back into the loop body past the
// condition.
// CIR-LABEL: cir.func{{.*}} @_Z26cleanup_in_while_in_switchi
// CIR:         %[[SLOT:.*]] = cir.alloca "__goto_dest_slot"
// CIR:         cir.scope {
// CIR:           cir.br ^[[SWITCH_BB:bb[0-9]+]](%{{.*}} : !s32i)
// CIR:         ^{{bb[0-9]+}}:
// CIR:           cir.br ^[[LOOP_BB:bb[0-9]+]]
// CIR:         ^[[LOOP_BB]]:
// CIR-NEXT:      cir.scope {
// CIR:             cir.switch.flat %{{.*}} : !s32i, ^{{bb[0-9]+}} [
// CIR-NEXT:          [[DEFAULT_ID:[0-9]+]]: ^{{bb[0-9]+}},
// CIR-NEXT:          [[CASE_ZERO_ID:[0-9]+]]: ^{{bb[0-9]+}}
// CIR:             cir.while {
// CIR:               cir.switch.flat %{{.*}} : !s32i, ^{{bb[0-9]+}} [
// CIR-NEXT:            [[DEFAULT_ID]]: ^[[SKIP_COND_D:bb[0-9]+]],
// CIR-NEXT:            [[CASE_ZERO_ID]]: ^[[SKIP_COND_Z:bb[0-9]+]]
// CIR:             ^[[SKIP_COND_D]]:
// CIR-NEXT:          %{{.*}} = cir.const #true
// CIR:             ^[[SKIP_COND_Z]]:
// CIR-NEXT:          %{{.*}} = cir.const #true
// CIR:               cir.condition
// CIR:             } do {
// CIR:               cir.switch.flat %{{.*}} : !s32i, ^{{bb[0-9]+}} [
// CIR-NEXT:            [[DEFAULT_ID]]: ^{{bb[0-9]+}},
// CIR-NEXT:            [[CASE_ZERO_ID]]: ^{{bb[0-9]+}}
// CIR:               cir.scope {
// CIR:                 cir.alloca "bv" {{.*}} : !cir.ptr<!rec_BitVector>
// CIR:                 cir.switch.flat %{{.*}} : !s32i, ^[[BODY_NORMAL:bb[0-9]+]] [
// CIR-NEXT:              [[DEFAULT_ID]]: ^[[TO_DEFAULT:bb[0-9]+]],
// CIR-NEXT:              [[CASE_ZERO_ID]]: ^[[TO_CASE_ZERO:bb[0-9]+]]
// CIR:               ^[[TO_DEFAULT]]:
// CIR:                 cir.br ^[[DEFAULT_BODY:bb[0-9]+]]
// CIR:               ^[[TO_CASE_ZERO]]:
// CIR:                 cir.br ^[[CASE_ZERO_BODY:bb[0-9]+]]
// CIR:               ^[[BODY_NORMAL]]:
// CIR-NEXT:            cir.br ^[[DEFAULT_BODY]]
// CIR:               ^[[DEFAULT_BODY]]:
// CIR-NEXT:            cir.scope {
// CIR-NEXT:              cir.break
// CIR:                 }
// CIR-NEXT:            cir.br ^[[CASE_ZERO_BODY]]
// CIR:               ^[[CASE_ZERO_BODY]]:
// CIR-NEXT:            cir.scope {
// CIR-NEXT:              cir.cleanup.scope {
// CIR:                     cir.yield
// CIR:                   } cleanup normal {
// CIR:                     cir.call @_ZN9BitVectorD1Ev
// CIR:                     cir.yield
// CIR:                   }
// CIR:                 }
// CIR:                 cir.continue
// CIR-NEXT:          }
// CIR-NEXT:          cir.yield
// CIR-NEXT:        }
// CIR-NEXT:        cir.yield
// CIR-NEXT:      }
// CIR-NEXT:      cir.br ^[[EXIT:bb[0-9]+]]
// CIR:         ^[[SWITCH_BB]](%[[COND:[0-9]+]]: !s32i
// CIR-NEXT:      cir.switch(%[[COND]] : !s32i) {
// CIR-NEXT:        cir.case(default, []) {
// CIR:               %[[DEFAULT_VAL:.*]] = cir.const #cir.int<[[DEFAULT_ID]]> : !s32i
// CIR-NEXT:          cir.store %[[DEFAULT_VAL]], %[[SLOT]]
// CIR-NEXT:          cir.break
// CIR-NEXT:        }
// CIR-NEXT:        cir.case(equal, [#cir.int<0> : !s32i]) {
// CIR-NEXT:          %[[CASE_ZERO_VAL:.*]] = cir.const #cir.int<[[CASE_ZERO_ID]]> : !s32i
// CIR-NEXT:          cir.store %[[CASE_ZERO_VAL]], %[[SLOT]]
// CIR-NEXT:          cir.break
// CIR-NEXT:        }
// CIR-NEXT:        cir.yield
// CIR-NEXT:      }
// CIR-NEXT:      %[[DEST:.*]] = cir.load %[[SLOT]]
// CIR-NEXT:      cir.switch.flat %[[DEST]] : !s32i, ^[[NO_CASE:bb[0-9]+]] [
// CIR-NEXT:        [[DEFAULT_ID]]: ^[[ENTER_DEFAULT:bb[0-9]+]],
// CIR-NEXT:        [[CASE_ZERO_ID]]: ^[[ENTER_CASE_ZERO:bb[0-9]+]]
// CIR:         ^[[ENTER_DEFAULT]]:
// CIR-NEXT:      cir.br ^[[LOOP_BB]]
// CIR:         ^[[ENTER_CASE_ZERO]]:
// CIR-NEXT:      cir.br ^[[LOOP_BB]]
// CIR:         ^[[NO_CASE]]:
// CIR-NEXT:      cir.br ^[[EXIT]]
// CIR:         ^[[EXIT]]:
// CIR-NEXT:      cir.yield
// CIR:         cir.return

// LLVM-LABEL: define{{.*}} void @_Z26cleanup_in_while_in_switchi
// LLVM:         alloca %struct.BitVector
// LLVMCIR:      load i32
// LLVMCIR:      br label %{{.*}}
// LLVMCIR:    [[TO_DEFAULT:[0-9]+]]:
// LLVMCIR:      br label %[[DEFAULT_BODY:.*]]
// LLVMCIR:    [[TO_CASE_ZERO:[0-9]+]]:
// LLVMCIR:      br label %[[CASE_ZERO_BODY:.*]]
// LLVMCIR:    [[WHILE_COND:[0-9]+]]:
// LLVMCIR:      icmp
// LLVMCIR:      br i1 %{{.*}}, label %[[WHILE_BODY:.*]], label %[[WHILE_END:.*]]
// LLVMCIR:    [[DEFAULT_BODY]]:
// LLVMCIR:      br label %[[BREAK:.*]]
// LLVMCIR:    [[WHILE_BODY]]:
// LLVMCIR:      br label %[[BREAK]]
// LLVMCIR:    [[BREAK]]:
// LLVMCIR:      br label %[[WHILE_END]]
// LLVMCIR:    [[CASE_ZERO_BODY]]:
// LLVMCIR:      call void @_ZN9BitVectorD1Ev
// LLVMCIR:      br label %[[WHILE_COND]]
// LLVMCIR:      switch i32 %{{.*}}, label %[[TO_DEFAULT]] [
// LLVMCIR:        i32 0, label %[[TO_CASE_ZERO]]
// LLVMCIR:      ]
// LLVMCIR:    [[WHILE_END]]:
// OGCG:         switch i32 %{{.*}}, label %[[DEFAULT:.*]] [
// OGCG:           i32 0, label %[[CASE_ZERO:.*]]
// OGCG:         ]
// OGCG:       [[DEAD_BLOCK:.*]]:
// OGCG:         br label %[[WHILE_COND:.*]]
// OGCG:       [[WHILE_COND]]:
// OGCG:         icmp
// OGCG:         br i1 %{{.*}}, label %[[WHILE_BODY:.*]], label %[[WHILE_END:.*]]
// OGCG:       [[WHILE_BODY]]:
// OGCG:         br label %[[DEFAULT]]
// OGCG:       [[DEFAULT]]:
// OGCG:         br label %[[WHILE_END]]
// OGCG:       [[CASE_ZERO]]:
// OGCG:         call void @_ZN9BitVectorD1Ev
// OGCG:         br label %[[WHILE_COND]]
// OGCG:       [[WHILE_END]]:
// OGCG:         br label %[[EPILOG:.*]]
// OGCG:       [[EPILOG]]:
// LLVM:         ret void
