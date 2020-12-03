// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix SIMD-ONLY
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix SIMD-ONLY
// SIMD-ONLY-NOT: {{__kmpc|__tgt}}
// REQUIRES: powerpc-registered-target

// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// CHECK-LABEL: define {{.*}}void @{{.+}}gtid_test
void gtid_test() {
// CHECK: call void @__kmpc_push_target_tripcount(%struct.ident_t* @{{.+}}, i64 -1, i64 100)
// CHECK: %0 = call i32 @__tgt_target_teams_mapper(%struct.ident_t* @{{.+}}, i64 -1, i8* @{{.+}}, i32 0, i8** null, i8** null, i64* null, i64* null, i8** null, i8** null, i32 0, i32 0)
// CHECK: call void [[TARGET_OUTLINE:@.+]]()
// CHECK: ret void
#pragma omp target teams distribute parallel for order(concurrent)
  for(int i = 0 ; i < 100; i++) {}
}

// CHECK: define internal void [[TARGET_OUTLINE]]()
// CHECK: call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_teams(%struct.ident_t* @{{.+}}, i32 0, void (i32*, i32*, ...)* bitcast (void (i32*, i32*)* [[TEAMS_OUTLINE:@.+]] to void (i32*, i32*, ...)*))
// CHECK: ret void

// CHECK: define internal void [[TEAMS_OUTLINE]](i32* {{.+}}, i32* {{.+}})
// CHECK: call void @__kmpc_for_static_init_4(
// CHECK-NOT: {{store|load}}{{.+}}!llvm.access.group !
// CHECK: call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* @{{.+}}, i32 2, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, i64, i64)* [[PARALLEL_OUTLINE:@.+]] to void (i32*, i32*, ...)*), i64 %{{.+}}, i64 %{{.+}})
// CHECK-NOT: {{store|load}}{{.+}}!llvm.access.group !
// CHECK: call void @__kmpc_for_static_fini(

// CHECK: define internal void [[PARALLEL_OUTLINE]](i32* {{.+}}, i32* {{.+}}, i64 {{.+}}, i64 {{.+}})
// CHECK: call void @__kmpc_for_static_init_4(
// CHECK: {{store|load}}{{.+}}!llvm.access.group ![[AG:[0-9]+]]
// CHECK: call void @__kmpc_for_static_fini(
// CHECK: ret void

// CHECK: ![[AG]] = distinct !{}
// CHECK: !{!"llvm.loop.parallel_accesses", ![[AG]]}
#endif
