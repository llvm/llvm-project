// RUN: %clang -E -fsanitize=undefined %s -o - | FileCheck --check-prefix=CHECK-UBSAN %s
// RUN: %clang -E -fsanitize=alignment %s -o - | FileCheck --check-prefix=CHECK-ALIGNMENT %s
// RUN: %clang -E  %s -o - | FileCheck --check-prefix=CHECK-NO-UBSAN %s

#if __has_feature(undefined_behavior_sanitizer)
int UBSanEnabled();
#else
int UBSanDisabled();
#endif

// CHECK-UBSAN: UBSanEnabled
// CHECK-ALIGNMENT: UBSanEnabled
// CHECK-NO-UBSAN: UBSanDisabled
