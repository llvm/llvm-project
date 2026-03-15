// RUN: %clang_cc1 %s -triple x86_64-unknown-linux-gnu -fsanitize=address -emit-llvm -o - | FileCheck -check-prefix=CHECK-ASAN %s

// CHECK-ASAN: @myGlobal1 = global { i32, [28 x i8] } zeroinitializer, align 32 #[[ATTR0:[0-9]+]]
// CHECK-ASAN: @myGlobal2 = global i32 0, no_sanitize_address, align 4
// CHECK-NOT: #[[ATTR1:[0-9]+]]
// CHECK-ASAN: attributes #[[ATTR0]] = { sanitized_padded_global }

int myGlobal1;
int __attribute__((no_sanitize("address"))) myGlobal2;

int main() {
    myGlobal1 = 0;
    myGlobal2 = 0;
    return 0;
}
