

struct Struct {
  int field; // CHECK: rename [[@LINE]]:7 -> [[@LINE]]:12
};

struct Struct2 {
  Struct /*range array=*/array[4][2]; // ARRAYCHECK: rename [[@LINE]]:26 -> [[@LINE]]:31
};

void foo() {
  (void)__builtin_offsetof(Struct, field); // CHECK: rename [[@LINE]]:36 -> [[@LINE]]:41
  (void)__builtin_offsetof(Struct2, /*range array=*/array[1][0]./*range f=*/field); // CHECK: rename [[@LINE]]:77 -> [[@LINE]]:82
} // ARRAYCHECK: rename [[@LINE-1]]:53 -> [[@LINE-1]]:58

// RUN: clang-refactor-test rename-initiate -at=%s:4:7 -at=%s:12:36 -new-name=Bar %s | FileCheck %s
// RUN: clang-refactor-test rename-initiate -at=%s:8:26 -at=%s:13:53 -new-name=Bar %s | FileCheck --check-prefix=ARRAYCHECK %s
