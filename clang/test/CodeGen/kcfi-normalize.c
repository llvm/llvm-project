// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -fsanitize=kcfi -fsanitize-cfi-icall-experimental-normalize-integers -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -fsanitize=kcfi -fsanitize-cfi-icall-experimental-normalize-integers -x c++ -o - %s | FileCheck %s
#if !__has_feature(kcfi)
#error Missing kcfi?
#endif

// Test that normalized type metadata for functions are emitted for cross-language KCFI support with
// other languages that can't represent and encode C/C++ integer types.

void foo(void (*fn)(int), int arg) {
    // CHECK-LABEL: define{{.*}}foo
    // CHECK-SAME: {{.*}}!kcfi_type ![[TYPE1:[0-9]+]]
    // KCFI ID = 0x2A548E59
    // CHECK: call void %0(i32 noundef %1){{.*}}[ "kcfi"(i32 710184537) ]
    fn(arg);
}

void bar(void (*fn)(int, int), int arg1, int arg2) {
    // CHECK-LABEL: define{{.*}}bar
    // CHECK-SAME: {{.*}}!kcfi_type ![[TYPE2:[0-9]+]]
    // KCFI ID = 0xD5A52C2A
    // CHECK: call void %0(i32 noundef %1, i32 noundef %2){{.*}}[ "kcfi"(i32 -710595542) ]
    fn(arg1, arg2);
}

void baz(void (*fn)(int, int, int), int arg1, int arg2, int arg3) {
    // CHECK-LABEL: define{{.*}}baz
    // CHECK-SAME: {{.*}}!kcfi_type ![[TYPE3:[0-9]+]]
    // KCFI ID = 0x2EA2BF3B
    // CHECK: call void %0(i32 noundef %1, i32 noundef %2, i32 noundef %3){{.*}}[ "kcfi"(i32 782417723) ]
    fn(arg1, arg2, arg3);
}

// CHECK: ![[#]] = !{i32 4, !"cfi-normalize-integers", i32 1}
// KCFI ID = DEEB3EA2
// CHECK: ![[TYPE1]] = !{i32 -555008350}
// KCFI ID = 24372DCB
// CHECK: ![[TYPE2]] = !{i32 607595979}
// KCFI ID = 0x60D0180C
// CHECK: ![[TYPE3]] = !{i32 1624250380}
