// RUN: %clang_cc1 -triple x86_64-unknown-linux -fsanitize=cfi-icall -fsanitize-trap=cfi-icall -fsanitize-cfi-icall-experimental-normalize-integers -emit-llvm -o - %s | FileCheck %s

// Test that normalized type metadata for functions are emitted for cross-language CFI support with
// other languages that can't represent and encode C/C++ integer types.

void foo(void (*fn)(int), int arg) {
    // CHECK-LABEL: define{{.*}}foo
    // CHECK-SAME: {{.*}}!type ![[TYPE1:[0-9]+]] !type !{{[0-9]+}}
    // CHECK: call i1 @llvm.type.test({{i8\*|ptr}} {{%f|%0}}, metadata !"_ZTSFvu3i32E.normalized")
    fn(arg);
}

void bar(void (*fn)(int, int), int arg1, int arg2) {
    // CHECK-LABEL: define{{.*}}bar
    // CHECK-SAME: {{.*}}!type ![[TYPE2:[0-9]+]] !type !{{[0-9]+}}
    // CHECK: call i1 @llvm.type.test({{i8\*|ptr}} {{%f|%0}}, metadata !"_ZTSFvu3i32S_E.normalized")
    fn(arg1, arg2);
}

void baz(void (*fn)(int, int, int), int arg1, int arg2, int arg3) {
    // CHECK-LABEL: define{{.*}}baz
    // CHECK-SAME: {{.*}}!type ![[TYPE3:[0-9]+]] !type !{{[0-9]+}}
    // CHECK: call i1 @llvm.type.test({{i8\*|ptr}} {{%f|%0}}, metadata !"_ZTSFvu3i32S_S_E.normalized")
    fn(arg1, arg2, arg3);
}

union Union {
  char *c;
  long *n;
} __attribute__((transparent_union));

void uni(void (*fn)(union Union), union Union arg1) {
    // CHECK-LABEL: define{{.*}}uni
    // CHECK-SAME: {{.*}}!type ![[TYPE4:[0-9]+]] !type !{{[0-9]+}}
    // CHECK: call i1 @llvm.type.test({{i8\*|ptr}} {{%f|%0}}, metadata !"_ZTSFv5UnionE.normalized")
    fn(arg1);
}

// CHECK: ![[TYPE1]] = !{i64 0, !"_ZTSFvPFvu3i32ES_E.normalized"}
// CHECK: ![[TYPE2]] = !{i64 0, !"_ZTSFvPFvu3i32S_ES_S_E.normalized"}
// CHECK: ![[TYPE3]] = !{i64 0, !"_ZTSFvPFvu3i32S_S_ES_S_S_E.normalized"}
// CHECK: ![[TYPE4]] = !{i64 0, !"_ZTSFvPFv5UnionES_E.normalized"}

