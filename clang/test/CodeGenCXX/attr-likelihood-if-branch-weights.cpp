// RUN: %clang_cc1 -O1 -emit-llvm %s -o - -triple=x86_64-linux-gnu | FileCheck -DLIKELY=2000 -DUNLIKELY=1 %s
// RUN: %clang_cc1 -O1 -emit-llvm %s -triple=x86_64-linux-gnu -mllvm -likely-branch-weight=99 -mllvm -unlikely-branch-weight=42 -o - | FileCheck -DLIKELY=99 -DUNLIKELY=42 %s

extern volatile bool b;
extern volatile int i;
extern bool A();
extern bool B();

bool f() {
  // CHECK-LABEL: define{{.*}} zeroext i1 @_Z1fv
  // CHECK: br {{.*}} !prof ![[PROF_LIKELY:[0-9]+]]
  if (b)
    [[likely]] {
      return A();
    }
  return B();
}

bool g() {
  // CHECK-LABEL: define{{.*}} zeroext i1 @_Z1gv
  // CHECK: br {{.*}} !prof ![[PROF_UNLIKELY:[0-9]+]]
  if (b)
    [[unlikely]] {
      return A();
    }

  return B();
}

bool h() {
  // CHECK-LABEL: define{{.*}} zeroext i1 @_Z1hv
  // CHECK: br {{.*}} !prof ![[PROF_UNLIKELY]]
  if (b)
    [[unlikely]] return A();

  return B();
}

void NullStmt() {
  // CHECK-LABEL: define{{.*}}NullStmt
  // CHECK: br {{.*}} !prof ![[PROF_UNLIKELY]]
  if (b)
    [[unlikely]];
  else {
    // Make sure the branches aren't optimized away.
    b = true;
  }
}

void IfStmt() {
  // CHECK-LABEL: define{{.*}}IfStmt
  // CHECK: br {{.*}} !prof ![[PROF_UNLIKELY]]
  if (b)
    [[unlikely]] if (B()) {}

  // CHECK-NOT: br {{.*}} !prof
  // CHECK: br {{.*}} !prof
  if (b) {
    if (B())
      [[unlikely]] { b = false; }
  }
}

void WhileStmt() {
  // CHECK-LABEL: define{{.*}}WhileStmt
  // CHECK: br {{.*}} !prof ![[PROF_UNLIKELY]]
  if (b)
    [[unlikely]] while (B()) {}

  // CHECK-NOT: br {{.*}} %if.end{{.*}} !prof
  if (b)
    // CHECK: br {{.*}} !prof ![[PROF_LIKELY]]
    while (B())
      [[unlikely]] { b = false; }
}

void DoStmt() {
  // CHECK-LABEL: define{{.*}}DoStmt
  // CHECK: br {{.*}} !prof ![[PROF_UNLIKELY]]
  if (b)
    [[unlikely]] do {}
    while (B())
      ;

  // CHECK-NOT: br {{.*}} %if.end{{.*}} !prof
  if (b)
    do
      [[unlikely]] {}
    while (B());
}

void ForStmt() {
  // CHECK-LABEL: define{{.*}}ForStmt
  // CHECK: br {{.*}} !prof ![[PROF_UNLIKELY]]
  if (b)
    [[unlikely]] for (; B();) {}

  // CHECK-NOT: br {{.*}} %if.end{{.*}} !prof
  if (b)
    // CHECK: br {{.*}} !prof ![[PROF_LIKELY]]
    for (; B();)
      [[unlikely]] {}
}

void GotoStmt() {
  // CHECK-LABEL: define{{.*}}GotoStmt
  // CHECK: br {{.*}} !prof ![[PROF_UNLIKELY]]
  if (b)
    [[unlikely]] goto end;
  else {
    // Make sure the branches aren't optimized away.
    b = true;
  }
end:;
}

void ReturnStmt() {
  // CHECK-LABEL: define{{.*}}ReturnStmt
  // CHECK: br {{.*}} !prof ![[PROF_UNLIKELY]]
  if (b)
    [[unlikely]] return;
  else {
    // Make sure the branches aren't optimized away.
    b = true;
  }
}

void SwitchStmt() {
  // CHECK-LABEL: define{{.*}}SwitchStmt
  // CHECK: br {{.*}} !prof ![[PROF_UNLIKELY]]
  if (b)
    [[unlikely]] switch (i) {}
  else {
    // Make sure the branches aren't optimized away.
    b = true;
  }
  // CHECK-NOT: br {{.*}} %if.end{{.*}} !prof
  if (b)
    switch (i)
      [[unlikely]] {}
  else {
    // Make sure the branches aren't optimized away.
    b = true;
  }
}

// CHECK: ![[PROF_LIKELY]] = !{!"branch_weights", i32 [[UNLIKELY]], i32 [[LIKELY]]}
// CHECK: ![[PROF_UNLIKELY]] = !{!"branch_weights", i32 [[LIKELY]], i32 [[UNLIKELY]]}
