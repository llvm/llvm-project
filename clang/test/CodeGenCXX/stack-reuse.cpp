// RUN: %clang_cc1 -triple armv7-unknown-linux-gnueabihf %s -o - -emit-llvm -O2 | FileCheck %s

// Stack should be reused when possible, no need to allocate two separate slots
// if they have disjoint lifetime.

// Sizes of objects are related to previously existed threshold of 32.  In case
// of S_large stack size is rounded to 40 bytes.

// 32B
struct S_small {
  int a[8];
};

// 36B
struct S_large {
  int a[9];
};

// Helper class for lifetime scope absence testing
struct Combiner {
  S_large a, b;

  Combiner(S_large);
  Combiner f();
};

extern S_small foo_small();
extern S_large foo_large();
extern void bar_small(S_small*);
extern void bar_large(S_large*);

// Prevent mangling of function names.
extern "C" {

void small_rvoed_unnamed_temporary_object() {
// CHECK-LABEL: define{{.*}} void @small_rvoed_unnamed_temporary_object
// CHECK: call void @llvm.lifetime.start
// CHECK: call void @_Z9foo_smallv
// CHECK: call void @llvm.lifetime.end
// CHECK: call void @llvm.lifetime.start
// CHECK: call void @_Z9foo_smallv
// CHECK: call void @llvm.lifetime.end

  foo_small();
  foo_small();
}

void large_rvoed_unnamed_temporary_object() {
// CHECK-LABEL: define{{.*}} void @large_rvoed_unnamed_temporary_object
// CHECK: call void @llvm.lifetime.start
// CHECK: call void @_Z9foo_largev
// CHECK: call void @llvm.lifetime.end
// CHECK: call void @llvm.lifetime.start
// CHECK: call void @_Z9foo_largev
// CHECK: call void @llvm.lifetime.end

  foo_large();
  foo_large();
}

void small_rvoed_named_temporary_object() {
// CHECK-LABEL: define{{.*}} void @small_rvoed_named_temporary_object
// CHECK: call void @llvm.lifetime.start
// CHECK: call void @_Z9foo_smallv
// CHECK: call void @llvm.lifetime.end
// CHECK: call void @llvm.lifetime.start
// CHECK: call void @_Z9foo_smallv
// CHECK: call void @llvm.lifetime.end

  {
    S_small s = foo_small();
  }
  {
    S_small s = foo_small();
  }
}

void large_rvoed_named_temporary_object() {
// CHECK-LABEL: define{{.*}} void @large_rvoed_named_temporary_object
// CHECK: call void @llvm.lifetime.start
// CHECK: call void @_Z9foo_largev
// CHECK: call void @llvm.lifetime.end
// CHECK: call void @llvm.lifetime.start
// CHECK: call void @_Z9foo_largev
// CHECK: call void @llvm.lifetime.end

  {
    S_large s = foo_large();
  }
  {
    S_large s = foo_large();
  }
}

void small_auto_object() {
// CHECK-LABEL: define{{.*}} void @small_auto_object
// CHECK: call void @llvm.lifetime.start
// CHECK: call void @_Z9bar_smallP7S_small
// CHECK: call void @llvm.lifetime.end
// CHECK: call void @llvm.lifetime.start
// CHECK: call void @_Z9bar_smallP7S_small
// CHECK: call void @llvm.lifetime.end

  {
    S_small s;
    bar_small(&s);
  }
  {
    S_small s;
    bar_small(&s);
  }
}

void large_auto_object() {
// CHECK-LABEL: define{{.*}} void @large_auto_object
// CHECK: call void @llvm.lifetime.start
// CHECK: call void @_Z9bar_largeP7S_large
// CHECK: call void @llvm.lifetime.end
// CHECK: call void @llvm.lifetime.start
// CHECK: call void @_Z9bar_largeP7S_large
// CHECK: call void @llvm.lifetime.end

  {
    S_large s;
    bar_large(&s);
  }
  {
    S_large s;
    bar_large(&s);
  }
}

int large_combiner_test(S_large s) {
// CHECK-LABEL: define{{.*}} i32 @large_combiner_test
// CHECK: [[T2:%.*]] = alloca %struct.Combiner
// CHECK: [[T1:%.*]] = alloca %struct.Combiner
// CHECK: [[T3:%.*]] = call noundef ptr @_ZN8CombinerC1E7S_large(ptr {{[^,]*}} [[T1]], [9 x i32] %s.coerce)
// CHECK: call void @_ZN8Combiner1fEv(ptr nonnull sret(%struct.Combiner) align 4 [[T2]], ptr {{[^,]*}} [[T1]])
// CHECK: [[T5:%.*]] = load i32, ptr [[T2]]
// CHECK: ret i32 [[T5]]

  return Combiner(s).f().a.a[0];
}

}
