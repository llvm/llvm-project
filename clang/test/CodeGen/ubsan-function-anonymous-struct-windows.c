// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -fsanitize=function -emit-llvm -o - %s | FileCheck %s

// CHECK: define internal {{.*}} @sample_tuple(){{.*}} !func_sanitize !{{[0-9]+}}

static struct { int first; float second; } sample_tuple(void) {
    return (typeof(sample_tuple())){ .first = 10, .second = 0.1f };
}

void call_sample(void) {
    sample_tuple();
}
