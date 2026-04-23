// RUN: %clang_cc1 -x c -triple x86_64-windows-msvc -emit-llvm -O0 -fms-extensions -fexceptions -o - %s | FileCheck %s

// Global state to track resource cleanup
int freed = 0;
void* allocated_buffer = 0;

// External functions that prevent optimization
extern void external_operation(void);
extern int external_condition(void);
extern void external_cleanup(void*);

// Declare SEH exception functions
void RaiseException(unsigned long code, unsigned long flags, unsigned long argc, void* argv);

// Simulate complex resource allocation/cleanup
void* allocate_buffer(int size) {
  // Simulate allocation that could fail
  if (external_condition()) {
    allocated_buffer = (void*)0x12345678;  // Mock allocation
    return allocated_buffer;
  }
  return 0;
}

void free_buffer(void* buffer) {
  if (buffer && freed == 0) {
    freed = 1;
    allocated_buffer = 0;
    external_cleanup(buffer);  // External cleanup prevents inlining
  }
}


int complex_operation_with_finally(int operation_type) {
  void* buffer = 0;
  int result = 0;
  
  __try {
    // Multiple operations that could throw exceptions
    buffer = allocate_buffer(1024);
    if (!buffer) {
      result = -1;
      __leave;  // Early exit - finally should still run
    }
    
    // Simulate complex operations that could throw
    external_operation();  // Could throw
    
    if (operation_type == 1) {
      external_operation();  // Another potential throw point
    }
    
    result = 0;  // Success
  } __finally {
    // Critical cleanup that must run exactly once
    if (buffer) {
      free_buffer(buffer);
    }
  }
  
  // Exception raised after finally block has already executed
  // This is the pattern that causes double execution in resource cleanup
  if (operation_type == 2) {
    RaiseException(0xC0000005, 0, 0, 0);
  }
  
  return result;
}

// CHECK: define dso_local i32 @complex_operation_with_finally(i32 noundef %operation_type)
// CHECK:   %finally.executed = alloca i1, align 1
// CHECK:   store i1 false, ptr %finally.executed, align 1

// Normal path: check if finally already ran.
// CHECK-LABEL: __try.__leave:
// CHECK:   %[[finally_executed:.+]] = load i1, ptr %finally.executed, align 1
// CHECK:   br i1 %[[finally_executed]], label %finally.skip, label %finally.run

// Normal path: run finally and set flag.
// CHECK-LABEL: finally.run:
// CHECK:   store i1 true, ptr %finally.executed, align 1
// CHECK:   call void @"?fin$0@0@complex_operation_with_finally@@"(i8 noundef
// CHECK:   br label %finally.skip

// Normal path: skip finally.
// CHECK-LABEL: finally.skip:
// CHECK:   %[[cmp:.+]] = icmp eq i32 {{.+}}, 2
// CHECK:   br i1 %[[cmp]], label %if.then10, label %if.end11

// Exception path: check if finally already ran.
// CHECK-LABEL: ehcleanup:
// CHECK:   %[[finally_executed_eh:.+]] = load i1, ptr %finally.executed, align 1
// CHECK:   br i1 %[[finally_executed_eh]], label %finally.skip8, label %finally.run7

// Exception path: run finally and set flag.
// CHECK-LABEL: finally.run7:
// CHECK:   store i1 true, ptr %finally.executed, align 1
// CHECK:   call void @"?fin$0@0@complex_operation_with_finally@@"(i8 noundef 1
// CHECK:   br label %finally.skip8

// Exception path: skip finally.
// CHECK-LABEL: finally.skip8:
// CHECK:   cleanupret from {{.+}} unwind to caller

// CHECK: define internal void @"?fin$0@0@complex_operation_with_finally@@"(i8 noundef %abnormal_termination, ptr noundef %frame_pointer)
// CHECK:   call void @free_buffer(ptr noundef

// CHECK-LABEL: @main
int main() {
  // This tests that the finally is not executed twice when an exception
  // is raised after the finally has already run.
  int result = complex_operation_with_finally(2);
  return result;
}
