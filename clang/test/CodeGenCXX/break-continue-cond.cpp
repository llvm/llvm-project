// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++20 -emit-llvm -o - %s | FileCheck %s --enable-var-scope

bool cond();
void step();

// CHECK-LABEL: define {{.*}}void @_Z24while_condition_continuev(
// CHECK:       while.cond:
// CHECK:       while.cond1:
// CHECK-NEXT:    br label %while.cond
void while_condition_continue() {
  while (cond())
    while (int n = ({ continue; 1; })) {}
}

// CHECK-LABEL: define {{.*}}void @_Z22for_condition_continuev(
// CHECK:       while.cond:
// CHECK:       for.cond:
// CHECK-NEXT:    br label %while.cond
void for_condition_continue() {
  while (cond())
    for (; int n = ({ continue; 1; }); ++n) {}
}

struct Cleanup {
  Cleanup();
  ~Cleanup();
  explicit operator bool();
};

// CHECK-LABEL: define {{.*}}void @_Z17increment_cleanupv(
// CHECK:         [[INIT:%.*]] = alloca %struct.Cleanup
// CHECK-NEXT:    [[CONDITION:%.*]] = alloca %struct.Cleanup
// CHECK:       while.cond:
// CHECK:       for.inc:
// CHECK-NEXT:    store i32 [[DEST:[0-9]+]], ptr [[SLOT:%.*]], align 4
// CHECK-NEXT:    br label %cleanup
// CHECK:       cleanup:
// CHECK-NEXT:    call void @_ZN7CleanupD1Ev(ptr {{.*}} [[CONDITION]])
// CHECK-NEXT:    call void @_ZN7CleanupD1Ev(ptr {{.*}} [[INIT]])
// CHECK-NEXT:    [[VALUE:%.*]] = load i32, ptr [[SLOT]], align 4
// CHECK-NEXT:    switch i32 [[VALUE]],
// CHECK:           i32 [[DEST]], label %while.cond
void increment_cleanup() {
  while (cond())
    for (Cleanup init; Cleanup condition{}; ({ continue; })) {}
}

// CHECK-LABEL: define {{.*}}void @_Z23increment_break_cleanupv(
// CHECK:         [[INIT:%.*]] = alloca %struct.Cleanup
// CHECK-NEXT:    [[CONDITION:%.*]] = alloca %struct.Cleanup
// CHECK:       for.inc:
// CHECK-NEXT:    store i32 [[DEST:[0-9]+]], ptr [[SLOT:%.*]], align 4
// CHECK-NEXT:    br label %cleanup
// CHECK:       cleanup:
// CHECK-NEXT:    call void @_ZN7CleanupD1Ev(ptr {{.*}} [[CONDITION]])
// CHECK-NEXT:    call void @_ZN7CleanupD1Ev(ptr {{.*}} [[INIT]])
// CHECK-NEXT:    [[VALUE:%.*]] = load i32, ptr [[SLOT]], align 4
// CHECK-NEXT:    switch i32 [[VALUE]],
// CHECK:           i32 [[DEST]], label %while.end
// CHECK:       for.end:
// CHECK:       while.end:
void increment_break_cleanup() {
  while (cond())
    for (Cleanup init; Cleanup condition{}; ({ break; })) {}
}

