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
    // CHECK: call void %0(i32 noundef %1){{.*}}[ "kcfi"(i32 1162514891) ]
    fn(arg);
}

void bar(void (*fn)(int, int), int arg1, int arg2) {
    // CHECK-LABEL: define{{.*}}bar
    // CHECK-SAME: {{.*}}!kcfi_type ![[TYPE2:[0-9]+]]
    // CHECK: call void %0(i32 noundef %1, i32 noundef %2){{.*}}[ "kcfi"(i32 448046469) ]
    fn(arg1, arg2);
}

void baz(void (*fn)(int, int, int), int arg1, int arg2, int arg3) {
    // CHECK-LABEL: define{{.*}}baz
    // CHECK-SAME: {{.*}}!kcfi_type ![[TYPE3:[0-9]+]]
    // CHECK: call void %0(i32 noundef %1, i32 noundef %2, i32 noundef %3){{.*}}[ "kcfi"(i32 -2049681433) ]
    fn(arg1, arg2, arg3);
}

// CHECK: ![[TYPE1]] = !{i32 -1143117868}
// CHECK: ![[TYPE2]] = !{i32 -460921415}
// CHECK: ![[TYPE3]] = !{i32 -333839615}
