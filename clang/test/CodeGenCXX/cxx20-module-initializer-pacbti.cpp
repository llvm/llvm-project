// RUN: %clang_cc1 -triple thumbv8.1m.main-unknown-none-eabi -emit-module-interface -target-feature +pacbti -msign-return-address=all -mbranch-target-enforce -std=c++20 %s -o %t.pcm
// RUN: %clang_cc1 -triple thumbv8.1m.main-unknown-none-eabi -std=c++20 %t.pcm -emit-llvm -o - | \
// RUN: FileCheck --check-prefixes=CHECK,CHECK-PAC-ARM %s

// RUN: %clang_cc1 -triple aarch64-unknown-none-elf -emit-module-interface -target-feature +pacbti -msign-return-address=all -msign-return-address-key=b_key -mbranch-target-enforce -std=c++20 %s -o %t.pcm
// RUN: %clang_cc1 -triple aarch64-unknown-none-elf -std=c++20 %t.pcm -emit-llvm -o - | \
// RUN: FileCheck --check-prefixes=CHECK,CHECK-PAC-AARCH64 %s

// CHECK: define void @_ZGIW3foo() #0
// CHECK-PAC-ARM: attributes #0 = { noinline nounwind "branch-target-enforcement" "no-trapping-math"="true" "sign-return-address"="all" "sign-return-address-key"="a_key" "stack-protector-buffer-size"="8" "target-features"="+armv8.1-m.main,+pacbti,+thumb-mode" }
// CHECK-PAC-AARCH64: attributes #0 = { noinline nounwind "branch-target-enforcement" "no-trapping-math"="true" "sign-return-address"="all" "sign-return-address-key"="b_key" "stack-protector-buffer-size"="8" "target-features"="+pacbti" }

module;

export module foo;

export void func();
