// RUN: %clang_cc1 -triple x86_64-windows-msvc -emit-llvm -O0 -fms-extensions -fexceptions -fcxx-exceptions -o - %s | FileCheck %s

int freed = 0;
void myfree(int *p) {
  ++freed;
}

// CHECK-LABEL: define dso_local i32 @main(
int main() {
  int x = 0;
  int *p = &x;
  __try {
    return 0;
  } __finally {
    myfree(p);
  }
}

// Check that a guard flag is allocated to prevent double execution
// CHECK: %finally.executed = alloca i1
// CHECK: store i1 false, ptr %finally.executed

// Check the main function has guard logic to prevent double execution
// CHECK: %finally.executed{{.*}} = load i1, ptr %finally.executed
// CHECK: br i1 %finally.executed{{.*}}, label %finally.skip, label %finally.run
// CHECK: finally.run:
// CHECK: store i1 true, ptr %finally.executed
// CHECK: call void @"?fin$0@0@main@@"
// CHECK: finally.skip:

// Check the finally helper function is called only once
// CHECK-LABEL: define internal void @"?fin$0@0@main@@"
// CHECK: call void @myfree
// CHECK-NOT: call void @myfree
