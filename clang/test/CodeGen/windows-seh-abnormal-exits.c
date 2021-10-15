// RUN: %clang_cc1 -triple x86_64-windows -fms-extensions -Wno-implicit-function-declaration -S -emit-llvm %s -o - | FileCheck %s

// CHECK: %[[src:[0-9-]+]] = call i8* @llvm.localaddress()
// CHECK-NEXT: %cleanup.dest = load i32, i32* %cleanup.dest.slot, align 4
// CHECK-NEXT: %[[src2:[0-9-]+]] = icmp ne i32 %cleanup.dest, 0
// CHECK-NEXT: %[[src3:[0-9-]+]] = zext i1 %[[src2]] to i8
// CHECK-NEXT: call void @"?fin$0@0@seh_abnormal_exits@@"(i8 noundef %[[src3]], i8* noundef %[[src]])

void seh_abnormal_exits(int *Counter) {
  for (int i = 0; i < 5; i++) {
    __try {
      if (i == 0)
        continue;   // abnormal termination
      else if (i == 1)
        goto t10;   // abnormal termination
      else if (i == 2)
        __leave;  // normal execution
      else if (i == 4)
        return;  // abnormal termination
    }
    __finally {
      if (AbnormalTermination()) {
        *Counter += 1;
      }
    }
  t10:;
  }
  return; // *Counter == 3
}

