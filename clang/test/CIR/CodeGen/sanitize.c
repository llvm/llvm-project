// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir \
// RUN:   -fsanitize=address %s -o - | FileCheck %s --check-prefix=ASAN
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir \
// RUN:   %s -o - | FileCheck %s --check-prefix=NO-ASAN
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm \
// RUN:   -fsanitize=address %s -o - | FileCheck %s --check-prefix=LLVM-ASAN

// ASAN: cir.sanitize = #cir.sanitize<[address]>

// NO-ASAN: module
// NO-ASAN-NOT: cir.sanitize

void foo(void) {}
// ASAN-LABEL: cir.func{{.*}}@foo
// ASAN-SAME: sanitize(#cir.sanitize<[address]>)

// LLVM-ASAN: define {{.*}}void @foo() #[[SANITIZE_ATTR:[0-9]+]]

__attribute__((no_sanitize("address")))
void no_sanitize_address(void) {}
// ASAN-LABEL: cir.func{{.*}}@no_sanitize_address
// ASAN-NOT: sanitize(#cir.sanitize

// LLVM-ASAN: define {{.*}}void @no_sanitize_address()
// LLVM-ASAN-NOT: #[[SANITIZE_ATTR]]

// LLVM-ASAN: attributes #[[SANITIZE_ATTR]] = {{.*}}sanitize_address
