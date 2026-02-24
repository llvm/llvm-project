// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -o - %s | FileCheck %s

// Verify __int256 global/static/extern variable declarations and access.

// CHECK-DAG: @global_s = global i256 0, align 16
__int256_t global_s;

// CHECK-DAG: @global_u = global i256 42, align 16
__uint256_t global_u = 42;

// CHECK-DAG: @static_s = internal global i256 0, align 16
static __int256_t static_s;

// CHECK-DAG: @extern_s = external global i256, align 16
extern __int256_t extern_s;

// CHECK-LABEL: define{{.*}} void @read_global(ptr{{.*}}sret(i256)
// CHECK: load i256, ptr @global_s, align 16
__int256_t read_global(void) { return global_s; }

// CHECK-LABEL: define{{.*}} void @write_global(ptr{{.*}}byval(i256) align 16
// CHECK: store i256 %{{.*}}, ptr @global_s, align 16
void write_global(__int256_t v) { global_s = v; }

// CHECK-LABEL: define{{.*}} void @read_static(ptr{{.*}}sret(i256)
// CHECK: load i256, ptr @static_s, align 16
__int256_t read_static(void) { return static_s; }

// CHECK-LABEL: define{{.*}} void @write_static(ptr{{.*}}byval(i256) align 16
// CHECK: store i256 %{{.*}}, ptr @static_s, align 16
void write_static(__int256_t v) { static_s = v; }

// CHECK-LABEL: define{{.*}} void @read_extern(ptr{{.*}}sret(i256)
// CHECK: load i256, ptr @extern_s, align 16
__int256_t read_extern(void) { return extern_s; }

// CHECK-LABEL: define{{.*}} void @read_global_u(ptr{{.*}}sret(i256)
// CHECK: load i256, ptr @global_u, align 16
__uint256_t read_global_u(void) { return global_u; }
