// RUN: %clang_cc1 -triple x86_64-windows -fasync-exceptions -fcxx-exceptions -fexceptions -fms-extensions -x c++ -Wno-implicit-function-declaration -S -emit-llvm %s -o - | FileCheck %s

// CHECK: invoke void @llvm.seh.try.begin()
// CHECK: invoke void @llvm.seh.try.begin()
// CHECK: %[[src:[0-9-]+]] = load volatile i32, i32* %i
// CHECK-NEXT: i32 %[[src]]
// CHECK: invoke void @llvm.seh.try.end()
// CHECK: invoke void @llvm.seh.try.end()

// CHECK: define internal void @"?fin$0@0@main@@"(i8 %abnormal_termination
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
