// RUN: %clang_cc1 -triple thumbv8.1m.main-unknown-none-eabi -emit-module-interface -target-feature +pacbti -mbranch-target-enforce -std=c++20 %s -o %t.pcm
// RUN: %clang_cc1 -triple thumbv8.1m.main-unknown-none-eabi -std=c++20 %t.pcm -emit-llvm -o - | \
// RUN: FileCheck --check-prefixes=CHECK,CHECK-BTI %s

// RUN: %clang_cc1 -triple thumbv8.1m.main-unknown-none-eabi -emit-module-interface -target-feature +pacbti -msign-return-address=non-leaf -std=c++20 %s -o %t.pcm
// RUN: %clang_cc1 -triple thumbv8.1m.main-unknown-none-eabi -std=c++20 %t.pcm -emit-llvm -o - | \
// RUN: FileCheck --check-prefixes=CHECK,CHECK-PAC %s

// RUN: %clang_cc1 -triple thumbv8.1m.main-unknown-none-eabi -emit-module-interface -target-feature +pacbti -msign-return-address=all -std=c++20 %s -o %t.pcm
// RUN: %clang_cc1 -triple thumbv8.1m.main-unknown-none-eabi -std=c++20 %t.pcm -emit-llvm -o - | \
// RUN: FileCheck --check-prefixes=CHECK,CHECK-PAC-ALL %s

// RUN: %clang_cc1 -triple thumbv8.1m.main-unknown-none-eabi -emit-module-interface -target-feature +pacbti -msign-return-address=non-leaf -mbranch-target-enforce -std=c++20 %s -o %t.pcm
// RUN: %clang_cc1 -triple thumbv8.1m.main-unknown-none-eabi -std=c++20 %t.pcm -emit-llvm -o - | \
// RUN: FileCheck --check-prefixes=CHECK,CHECK-PAC-BTI %s

// RUN: %clang_cc1 -triple thumbv8.1m.main-unknown-none-eabi -emit-module-interface -target-feature +pacbti -msign-return-address=all -mbranch-target-enforce -std=c++20 %s -o %t.pcm
// RUN: %clang_cc1 -triple thumbv8.1m.main-unknown-none-eabi -std=c++20 %t.pcm -emit-llvm -o - | \
// RUN: FileCheck --check-prefixes=CHECK,CHECK-PAC-BTI-ALL %s

// RUN: %clang_cc1 -triple aarch64-unknown-none-elf -emit-module-interface -target-feature +pacbti -mbranch-target-enforce -std=c++20 %s -o %t.pcm
// RUN: %clang_cc1 -triple aarch64-unknown-none-elf -std=c++20 %t.pcm -emit-llvm -o - | \
// RUN: FileCheck --check-prefixes=CHECK,CHECK-BTI %s

// RUN: %clang_cc1 -triple aarch64-unknown-none-elf -emit-module-interface -target-feature +pacbti -msign-return-address=non-leaf -std=c++20 %s -o %t.pcm
// RUN: %clang_cc1 -triple aarch64-unknown-none-elf -std=c++20 %t.pcm -emit-llvm -o - | \
// RUN: FileCheck --check-prefixes=CHECK,CHECK-PAC %s

// RUN: %clang_cc1 -triple aarch64-unknown-none-elf -emit-module-interface -target-feature +pacbti -msign-return-address=non-leaf -msign-return-address-key=b_key -std=c++20 %s -o %t.pcm
// RUN: %clang_cc1 -triple aarch64-unknown-none-elf -std=c++20 %t.pcm -emit-llvm -o - | \
// RUN: FileCheck --check-prefixes=CHECK,CHECK-PAC-BKEY %s

// RUN: %clang_cc1 -triple aarch64-unknown-none-elf -emit-module-interface -target-feature +pacbti -msign-return-address=all -std=c++20 %s -o %t.pcm
// RUN: %clang_cc1 -triple aarch64-unknown-none-elf -std=c++20 %t.pcm -emit-llvm -o - | \
// RUN: FileCheck --check-prefixes=CHECK,CHECK-PAC-ALL %s

// RUN: %clang_cc1 -triple aarch64-unknown-none-elf -emit-module-interface -target-feature +pacbti -msign-return-address=all -msign-return-address-key=b_key -std=c++20 %s -o %t.pcm
// RUN: %clang_cc1 -triple aarch64-unknown-none-elf -std=c++20 %t.pcm -emit-llvm -o - | \
// RUN: FileCheck --check-prefixes=CHECK,CHECK-PAC-ALL-BKEY %s

// RUN: %clang_cc1 -triple aarch64-unknown-none-elf -emit-module-interface -target-feature +pacbti -msign-return-address=non-leaf -mbranch-target-enforce -std=c++20 %s -o %t.pcm
// RUN: %clang_cc1 -triple aarch64-unknown-none-elf -std=c++20 %t.pcm -emit-llvm -o - | \
// RUN: FileCheck --check-prefixes=CHECK,CHECK-PAC-BTI %s

// RUN: %clang_cc1 -triple aarch64-unknown-none-elf -emit-module-interface -target-feature +pacbti -msign-return-address=non-leaf -msign-return-address-key=b_key -mbranch-target-enforce -std=c++20 %s -o %t.pcm
// RUN: %clang_cc1 -triple aarch64-unknown-none-elf -std=c++20 %t.pcm -emit-llvm -o - | \
// RUN: FileCheck --check-prefixes=CHECK,CHECK-PAC-BKEY-BTI %s

// RUN: %clang_cc1 -triple aarch64-unknown-none-elf -emit-module-interface -target-feature +pacbti -msign-return-address=all -mbranch-target-enforce -std=c++20 %s -o %t.pcm
// RUN: %clang_cc1 -triple aarch64-unknown-none-elf -std=c++20 %t.pcm -emit-llvm -o - | \
// RUN: FileCheck --check-prefixes=CHECK,CHECK-PAC-BTI-ALL %s

// RUN: %clang_cc1 -triple aarch64-unknown-none-elf -emit-module-interface -target-feature +pacbti -msign-return-address=all -msign-return-address-key=b_key -mbranch-target-enforce -std=c++20 %s -o %t.pcm
// RUN: %clang_cc1 -triple aarch64-unknown-none-elf -std=c++20 %t.pcm -emit-llvm -o - | \
// RUN: FileCheck --check-prefixes=CHECK,CHECK-PAC-BKEY-BTI-ALL %s

// CHECK: define void @_ZGIW3foo() #0
// CHECK-BTI: attributes #0 = { nounwind "branch-target-enforcement" }
// CHECK-PAC: attributes #0 = { nounwind "sign-return-address"="non-leaf" "sign-return-address-key"="a_key" }
// CHECK-PAC-BKEY: attributes #0 = { nounwind "sign-return-address"="non-leaf" "sign-return-address-key"="b_key" }
// CHECK-PAC-ALL: attributes #0 = { nounwind "sign-return-address"="all" "sign-return-address-key"="a_key" }
// CHECK-PAC-ALL-BKEY: attributes #0 = { nounwind "sign-return-address"="all" "sign-return-address-key"="b_key" }
// CHECK-PAC-BTI: attributes #0 = { nounwind "branch-target-enforcement" "sign-return-address"="non-leaf" "sign-return-address-key"="a_key" }
// CHECK-PAC-BKEY-BTI: attributes #0 = { nounwind "branch-target-enforcement" "sign-return-address"="non-leaf" "sign-return-address-key"="b_key" }
// CHECK-PAC-BTI-ALL: attributes #0 = { nounwind "branch-target-enforcement" "sign-return-address"="all" "sign-return-address-key"="a_key" }
// CHECK-PAC-BKEY-BTI-ALL: attributes #0 = { nounwind "branch-target-enforcement" "sign-return-address"="all" "sign-return-address-key"="b_key" }

module;

export module foo;

export void func();
