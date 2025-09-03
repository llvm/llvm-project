// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -o - -emit-llvm -O1 \
// RUN:     -fexceptions -fcxx-exceptions -mllvm -simplifycfg-sink-common=false | FileCheck %s
//
// We should emit lifetime.ends for these temporaries in both the 'exception'
// and 'normal' paths in functions.
//
// -O1 is necessary to make lifetime markers appear.

struct Large {
  int cs[32];
};

Large getLarge();

// Used to ensure we emit invokes.
struct NontrivialDtor {
  int i;
  ~NontrivialDtor();
};

// CHECK-LABEL: define{{.*}} void @_Z33cleanupsAreEmittedWithoutTryCatchv
void cleanupsAreEmittedWithoutTryCatch() {
// CHECK: call void @llvm.lifetime.start.p0(ptr nonnull %[[CLEAN:.*]])
// CHECK: call void @llvm.lifetime.start.p0(ptr nonnull %[[T1:.*]])
// CHECK-NEXT: invoke void @_Z8getLargev
// CHECK-NEXT:     to label %[[CONT:[^ ]+]] unwind label %[[LPAD:[^ ]+]]
//
// CHECK: [[CONT]]:
// CHECK: call void @llvm.lifetime.end.p0(ptr nonnull %[[T1]])
// CHECK: call void @llvm.lifetime.start.p0(ptr nonnull %[[T2:.*]])
// CHECK-NEXT: invoke void @_Z8getLargev
// CHECK-NEXT:     to label %[[CONT2:[^ ]+]] unwind label %[[LPAD2:.+]]
//
// CHECK: [[CONT2]]:
// CHECK: call void @llvm.lifetime.end.p0(ptr nonnull %[[T2]])
// CHECK: call void @llvm.lifetime.end.p0(ptr nonnull %[[CLEAN]])
// CHECK: ret void
//
// CHECK: [[LPAD]]:
// CHECK: call void @llvm.lifetime.end.p0(ptr nonnull %[[T1]])
// CHECK: br label %[[EHCLEANUP:.+]]
//
// CHECK: [[LPAD2]]:
// CHECK: call void @llvm.lifetime.end.p0(ptr nonnull %[[T2]])
// CHECK: br label %[[EHCLEANUP]]
//
// CHECK: [[EHCLEANUP]]:
// CHECK: call void @llvm.lifetime.end.p0(ptr nonnull %[[CLEAN]])

  NontrivialDtor clean;

  getLarge();
  getLarge();
}

// CHECK-LABEL: define{{.*}} void @_Z30cleanupsAreEmittedWithTryCatchv
void cleanupsAreEmittedWithTryCatch() {
// CHECK: call void @llvm.lifetime.start.p0(ptr nonnull %[[CLEAN:.*]])
// CHECK: call void @llvm.lifetime.start.p0(ptr nonnull %[[T1:.*]])
// CHECK-NEXT: invoke void @_Z8getLargev
// CHECK-NEXT:     to label %[[CONT:[^ ]+]] unwind label %[[LPAD:[^ ]+]]
//
// CHECK: [[CONT]]:
// CHECK: call void @llvm.lifetime.end.p0(ptr nonnull %[[T1]])
// CHECK: call void @llvm.lifetime.start.p0(ptr nonnull %[[T2:.*]])
// CHECK-NEXT: invoke void @_Z8getLargev
// CHECK-NEXT:     to label %[[CONT2:[^ ]+]] unwind label %[[LPAD2:.+]]
//
// CHECK: [[CONT2]]:
// CHECK: call void @llvm.lifetime.end.p0(ptr nonnull %[[T2]])
// CHECK: br label %[[TRY_CONT:.+]]
//
// CHECK: [[LPAD]]:
// CHECK: call void @llvm.lifetime.end.p0(ptr nonnull %[[T1]])
// CHECK: br label %[[CATCH:.+]]
//
// CHECK: [[LPAD2]]:
// CHECK: call void @llvm.lifetime.end.p0(ptr nonnull %[[T2]])
// CHECK: br label %[[CATCH]]
//
// CHECK: [[CATCH]]:
// CHECK-NOT: call void @llvm.lifetime
// CHECK: invoke void
// CHECK-NEXT: to label %[[TRY_CONT]] unwind label %[[OUTER_LPAD:.+]]
//
// CHECK: [[TRY_CONT]]:
// CHECK: call void @llvm.lifetime.start.p0(ptr nonnull %[[T_OUTER:.*]])
// CHECK-NEXT: invoke void @_Z8getLargev
// CHECK-NEXT:     to label %[[OUTER_CONT:[^ ]+]] unwind label %[[OUTER_LPAD2:.+]]
//
// CHECK: [[OUTER_CONT]]:
// CHECK: call void @llvm.lifetime.end.p0(ptr nonnull %[[T_OUTER]])
// CHECK: call void @llvm.lifetime.end.p0(ptr nonnull %[[CLEAN]])
// CHECK: ret void
//
// CHECK: [[OUTER_LPAD]]:
// CHECK-NOT: call void @llvm.lifetime
// CHECK: br label %[[EHCLEANUP:.+]]
//
// CHECK: [[OUTER_LPAD2]]:
// CHECK: call void @llvm.lifetime.end.p0(ptr nonnull %[[T_OUTER]])
// CHECK: br label %[[EHCLEANUP]]
//
// CHECK: [[EHCLEANUP]]:
// CHECK: call void @llvm.lifetime.end.p0(ptr nonnull %[[CLEAN]])

  NontrivialDtor clean;

  try {
    getLarge();
    getLarge();
  } catch (...) {}

  getLarge();
}

// CHECK-LABEL: define{{.*}} void @_Z39cleanupInTryHappensBeforeCleanupInCatchv
void cleanupInTryHappensBeforeCleanupInCatch() {
// CHECK: call void @llvm.lifetime.start.p0(ptr nonnull %[[T1:.*]])
// CHECK-NEXT: invoke void @_Z8getLargev
// CHECK-NEXT:     to label %[[CONT:[^ ]+]] unwind label %[[LPAD:[^ ]+]]
//
// CHECK: [[CONT]]:
// CHECK: call void @llvm.lifetime.end.p0(ptr nonnull %[[T1]])
// CHECK: br label %[[TRY_CONT]]
//
// CHECK: [[LPAD]]:
// CHECK: call void @llvm.lifetime.end.p0(ptr nonnull %[[T1]])
// CHECK: br i1 {{[^,]+}}, label %[[CATCH_INT_MATCH:[^,]+]], label %[[CATCH_ALL:.+]]
//
// CHECK: [[CATCH_INT_MATCH]]:
// CHECK: call void @llvm.lifetime.start.p0(ptr nonnull %[[T2:.*]])
// CHECK-NEXT: invoke void @_Z8getLargev
// CHECK-NEXT:     to label %[[CATCH_INT_CONT:[^ ]+]] unwind label %[[CATCH_INT_LPAD:[^ ]+]]
//
// CHECK: [[CATCH_INT_CONT]]:
// CHECK: call void @llvm.lifetime.end.p0(ptr nonnull %[[T2]])
// CHECK: br label %[[TRY_CONT]]
//
// CHECK: [[TRY_CONT]]:
// CHECK: ret void
//
// CHECK: [[CATCH_ALL]]:
// CHECK: call void @llvm.lifetime.start.p0(ptr nonnull %[[T3:.*]])
// CHECK-NEXT: invoke void @_Z8getLargev
// CHECK-NEXT:     to label %[[CATCH_ALL_CONT:[^ ]+]] unwind label %[[CATCH_ALL_LPAD:[^ ]+]]
//
// CHECK: [[CATCH_ALL_CONT]]:
// CHECK: call void @llvm.lifetime.end.p0(ptr nonnull %[[T3]])
// CHECK: br label %[[TRY_CONT]]
//
// CHECK: [[CATCH_ALL_LPAD]]:
// CHECK: call void @llvm.lifetime.end.p0(ptr nonnull %[[T3]])
//
// CHECK: [[CATCH_INT_LPAD]]:
// CHECK: call void @llvm.lifetime.end.p0(ptr nonnull %[[T2]])
// CHECK-NOT: call void @llvm.lifetime

  try {
    getLarge();
  } catch (const int &) {
    getLarge();
  } catch (...) {
    getLarge();
  }
}

// FIXME: We don't currently emit lifetime markers for aggregate by-value
// temporaries (e.g. given a function `Large combine(Large, Large);`
// combine(getLarge(), getLarge()) "leaks" two `Large`s). We probably should. We
// also don't emit markers for things like:
//
// {
//   Large L = getLarge();
//   combine(L, L);
// }
//
// Though this arguably isn't as bad, since we pass a pointer to `L` as one of
// the two args.
