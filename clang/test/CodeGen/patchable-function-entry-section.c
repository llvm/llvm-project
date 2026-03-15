// RUN: %clang_cc1 -triple aarch64 -emit-llvm %s -o - | FileCheck --check-prefixes=COMMON,NODEFAULT %s
// RUN: %clang_cc1 -triple x86_64 -emit-llvm %s -fpatchable-function-entry=1 -fpatchable-function-entry-section=__default_section -o - | FileCheck --check-prefixes=COMMON,DEFAULT %s

// COMMON: define{{.*}} void @f0() #0
__attribute__((patchable_function_entry(0))) void f0(void) {}

// COMMON: define{{.*}} void @f00() #0
__attribute__((patchable_function_entry(0, 0, "__unused_section"))) void f00(void) {}

// COMMON: define{{.*}} void @f2() #1
__attribute__((patchable_function_entry(2))) void f2(void) {}

// COMMON: define{{.*}} void @f20() #2
__attribute__((patchable_function_entry(2, 0, "__attr_section"))) void f20(void) {}

// COMMON: define{{.*}} void @f44() #3
__attribute__((patchable_function_entry(4, 4))) void f44(void) {}

// COMMON: define{{.*}} void @f52() #4
__attribute__((patchable_function_entry(5, 2, "__attr_section"))) void f52(void) {}

// OPT: define{{.*}} void @f() #5
void f(void) {}

/// No need to emit "patchable-function-entry" and thus also "patchable-function-entry-section"
// COMMON: attributes #0 = { {{.*}}
// COMMON-NOT: "patchable-function-entry-section"

// NODEFAULT: attributes #1 = { {{.*}} "patchable-function-entry"="2"
// NODEFAULT-NOT: "patchable-function-entry-section"
// DEFAULT: attributes #1 = { {{.*}} "patchable-function-entry"="2" "patchable-function-entry-section"="__default_section"

// COMMON: attributes #2 = { {{.*}} "patchable-function-entry"="2" "patchable-function-entry-section"="__attr_section"

// NODEFAULT: attributes #3 = { {{.*}} "patchable-function-entry"="0" "patchable-function-prefix"="4"
// NODEFAULT-NOT: "patchable-function-entry-section"
// DEFAULT: attributes #3 = { {{.*}} "patchable-function-entry"="0" "patchable-function-entry-section"="__default_section" "patchable-function-prefix"="4"

// COMMON: attributes #4 = { {{.*}} "patchable-function-entry"="3" "patchable-function-entry-section"="__attr_section" "patchable-function-prefix"="2"

// DEFAULT:   attributes #5 = { {{.*}} "patchable-function-entry"="1" "patchable-function-entry-section"="__default_section"
