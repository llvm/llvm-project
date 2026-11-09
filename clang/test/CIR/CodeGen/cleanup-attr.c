// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefixes=LLVM,LLVMCIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM

void cl(int *p);
void vcl(void *p);

void simple(void) {
  __attribute__((cleanup(cl))) int x = 1;
  x++;
}

// CIR: cir.func {{.*}}@simple()
// CIR: %[[X:.*]] = cir.alloca "x" align(4) init : !cir.ptr<!s32i>
// CIR: %[[ONE:.*]] = cir.const #cir.int<1>
// CIR: cir.store align(4) %[[ONE]], %[[X]]
// CIR: cir.cleanup.scope {
// CIR:   cir.inc nsw
// CIR: } cleanup normal {
// CIR:   cir.call @cl(%[[X]]) : (!cir.ptr<!s32i> {llvm.noundef}) -> ()
// CIR:   cir.yield
// CIR: }

// LLVM: define dso_local void @simple()
// LLVM: %[[X:.*]] = alloca i32
// LLVMCIR: br label %[[DO_INC:.*]]
// LLVMCIR: [[DO_INC]]:
// LLVM: add nsw i32
// LLVMCIR: br label %[[CLEANUP:.*]]
// LLVMCIR: [[CLEANUP]]:
// LLVM: call void @cl(ptr noundef %[[X]])

// The cleanup function takes a different pointer type than the variable, and
// runs on early exits, in reverse order of declaration.
void multiple(int c) {
  __attribute__((cleanup(cl))) int x = 1;
  __attribute__((cleanup(vcl))) void *y = 0;
  if (c)
    return;
  x++;
}

// CIR: cir.func {{.*}}@multiple(
// CIR: %[[X:.*]] = cir.alloca "x" align(4) init
// CIR: %[[Y:.*]] = cir.alloca "y" align(8) init
// CIR: cir.cleanup.scope {
// CIR:   %[[NULL:.*]] = cir.const #cir.ptr<null>
// CIR:   cir.store align(8) %[[NULL]], %[[Y]]
// CIR:   cir.cleanup.scope {
// CIR:     cir.scope {
// CIR:       cir.if %{{.*}} {
// CIR:         cir.return
// CIR:       }
// CIR:     }
// CIR:     cir.inc nsw
// CIR:     cir.yield
// CIR:   } cleanup normal {
// CIR:     %[[Y_CAST:.*]] = cir.cast bitcast %[[Y]] : !cir.ptr<!cir.ptr<!void>> -> !cir.ptr<!void>
// CIR:     cir.call @vcl(%[[Y_CAST]]) : (!cir.ptr<!void> {llvm.noundef}) -> ()
// CIR:     cir.yield
// CIR:   }
// CIR: } cleanup normal {
// CIR:   cir.call @cl(%[[X]]) : (!cir.ptr<!s32i> {llvm.noundef}) -> ()
// CIR:   cir.yield
// CIR: }

// LLVM: define dso_local void @multiple(i32 noundef %[[C_ARG:.*]])
// LLVM: store i32 %[[C_ARG]], ptr %[[C:[^,]+]]
// LLVM: store i32 1, ptr %[[X:[^,]+]]
// LLVM: store ptr null, ptr %[[Y:[^,]+]]
// LLVM: %[[C_VAL:.*]] = load i32, ptr %[[C]]
// LLVM: %[[COND:.*]] = icmp ne i32 %[[C_VAL]], 0
// LLVM: br i1 %[[COND]], label %[[EARLY:.*]], label %[[FALLTHRU:.*]]
//
// Early return: record the exit and run the cleanups.
// LLVM: [[EARLY]]:
// LLVM: store i32 1, ptr %[[DEST:[^,]+]]
// LLVM: br label %[[CLEANUP:.*]]
//
// Fall-through: x++, then run the cleanups.
// LLVM: [[FALLTHRU]]:
// LLVM: %[[X_VAL:.*]] = load i32, ptr %[[X]]
// LLVM: %[[X_INC:.*]] = add nsw i32 %[[X_VAL]], 1
// LLVM: store i32 %[[X_INC]], ptr %[[X]]
// LLVM: store i32 0, ptr %[[DEST]]
// LLVM: br label %[[CLEANUP]]
//
// y was declared last, so its cleanup runs before x's.
// LLVM: [[CLEANUP]]:
// LLVM: call void @vcl(ptr noundef %[[Y]])
// LLVM: call void @cl(ptr noundef %[[X]])
// LLVM: ret void
