// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -o - -fsanitize=daddress %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -o - -O1 -fsanitize=daddress %s | FileCheck %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -emit-llvm -o - -fsanitize=daddress %s | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -emit-llvm -o - -O1 -fsanitize=daddress %s | FileCheck %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -emit-llvm -o - -O1 -fsanitize=daddress %s | FileCheck %s --check-prefix=ABORT
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -emit-llvm -o - -O1 -fsanitize=daddress -mllvm -dasan-trap-on-error %s | FileCheck %s --check-prefix=TRAP
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -emit-llvm -o - -O1 -fsanitize=daddress -fsanitize-recover=daddress %s | FileCheck %s --check-prefix=RECOVER
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -emit-llvm -o - -fsanitize=daddress %s | FileCheck %s --check-prefix=ABORT-SHARED

int foo(int *a, int *b) { return *a + *b; }
int bar(int __attribute__((address_space(3))) *s) { return *s; }

// CHECK: sanitize_device_address
// CHECK: !{i32 4, !"nosanitize_device_address", i32 1}

// ABORT-LABEL: define {{.*}} @foo(
// ABORT:       notail call void @__dasan_report_load(ptr %a, i64 4, i64 {{.*}}, i64 {{.*}}) [[SITE:#[0-9]+]]
// ABORT-NEXT:  unreachable
// ABORT:       notail call void @__dasan_report_load(ptr %b, i64 4, i64 {{.*}}, i64 {{.*}}) [[SITE]]
// ABORT-NEXT:  unreachable
// ABORT:       declare void @__dasan_report_load(ptr, i64, i64, i64) [[ATTR:#[0-9]+]]
// ABORT:       attributes [[ATTR]] = { cold noreturn nounwind }
// ABORT:       attributes [[SITE]] = { noinline nomerge }

// TRAP:       br i1 {{.*}}, label %dasan.trap,
// TRAP:       br i1 {{.*}}, label %dasan.trap,
// TRAP:       dasan.trap:
// TRAP:       call void @llvm.trap()
// TRAP-NEXT:  unreachable

// RECOVER:     notail call void @__dasan_report_load_noabort(ptr %a, {{.*}}) [[SITE:#[0-9]+]]
// RECOVER-NOT: unreachable
// RECOVER:     notail call void @__dasan_report_load_noabort(ptr %b, {{.*}}) [[SITE]]
// RECOVER:     declare void @__dasan_report_load_noabort(ptr, i64, i64, i64) [[RATTR:#[0-9]+]]
// RECOVER:     attributes [[RATTR]] = { cold nounwind }
// RECOVER:     attributes [[SITE]] = { noinline nomerge }

// Unsized LDS needs the dispatch pointer, which -O1 infers away on this TU.
// ABORT-SHARED:      define {{.*}} @bar(
// ABORT-SHARED:      notail call void @__dasan_report_shared_load(
// ABORT-SHARED-NEXT: unreachable
