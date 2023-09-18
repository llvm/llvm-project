// RUN: %clang_cc1 -triple x86_64-windows -fasync-exceptions -fcxx-exceptions -fexceptions -fms-extensions -x c++ -Wno-implicit-function-declaration -S -emit-llvm %s -o - | FileCheck %s

// CHECK-LABEL: @main()
// CHECK: invoke void @llvm.seh.try.begin()
// CHECK: invoke void @llvm.seh.try.begin()
// CHECK: %[[src:[0-9-]+]] = load volatile i32, ptr %i
// CHECK-NEXT: i32 noundef %[[src]]
// CHECK: invoke void @llvm.seh.try.end()
// CHECK: invoke void @llvm.seh.try.end()

// CHECK: define internal void @"?fin$0@0@main@@"(i8 noundef %abnormal_termination
// CHECK: invoke void @llvm.seh.try.begin()
// CHECK: invoke void @llvm.seh.try.end()

// *****************************************************************************
// Abstract:     Test __Try in __finally under SEH -EHa option
void printf(...);
int volatile *NullPtr = 0;
int main() {
  for (int i = 0; i < 3; i++) {
    printf(" --- Test _Try in _finally --- i = %d \n", i);
    __try {
      __try {
        printf("  In outer _try i = %d \n", i);
        if (i == 0)
          *NullPtr = 0;
      } __finally {
        __try {
          printf("  In outer _finally i = %d \n", i);
          if (i == 1)
            *NullPtr = 0;
        } __finally {
          printf("  In Inner _finally i = %d \n", i);
          if (i == 2)
            *NullPtr = 0;
        }
      }
    } __except (1) {
      printf(" --- In outer except handler i = %d \n", i);
    }
  }
  return 0;
}

// CHECK-LABEL:@"?foo@@YAXXZ"()
// CHECK: invoke.cont:
// CHECK: invoke void @llvm.seh.try.begin()
// CHECK: store volatile i32 1, ptr %cleanup.dest.slot
// CHECK: invoke void @llvm.seh.try.end()
// CHECK: invoke.cont2:
// CHECK: %cleanup.dest = load i32, ptr %cleanup.dest.slot
// CHECK: %1 = icmp ne i32 %cleanup.dest, 0
// CHECK: %2 = zext i1 %1 to i8
// CHECK: call void @"?fin$0@0@foo@@"(i8 noundef %2, ptr noundef %0)
// CHECK: ehcleanup:
// CHECK: call void @"?fin$0@0@foo@@"(i8 noundef 1, ptr noundef %4)
void foo()
{
  __try {
    return;
  }
  __finally {
    if (_abnormal_termination()) {
      printf("Passed\n");
    } else {
      printf("Failed\n");
    }
  }
}
