// RUN: %clang_cc1 -triple x86_64-windows -fms-extensions -Wno-implicit-function-declaration -S -emit-llvm %s -o - | FileCheck %s

// CHECK: %[[dst:[0-9-]+]] = call ptr @llvm.eh.recoverfp(ptr @"?fin$0@0@main@@", ptr %frame_pointer)
// CHECK-NEXT: %[[dst1:[0-9-]+]] = call ptr @llvm.localrecover(ptr @"?fin$0@0@main@@", ptr %[[dst]], i32 0)
// CHECK-NEXT: = load ptr, ptr %[[dst1]], align 8

int
main(int argc, char *argv[])
{
    int Counter = 0;
    //
    // Try/except within the finally clause of a try/finally.
    //
    __try {
      Counter -= 1;
    }
    __finally {
      __try {
        Counter += 2;
        // RtlRaiseStatus(STATUS_INTEGER_OVERFLOW);
      } __except(Counter) {
        __try {
          Counter += 3;
        }
        __finally {
          if (abnormal_termination() == 1) {
            Counter += 5;
          }
        }
      }
    }
    // expect Counter == 9
    return 1;
}

