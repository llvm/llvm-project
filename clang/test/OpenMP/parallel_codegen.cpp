// RUN: %clang_cc1 -verify -fopenmp -x c++ -emit-llvm %s -triple %itanium_abi_triple -fexceptions -fcxx-exceptions -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -debug-info-kind=limited -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix=CHECK-DEBUG %s

// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -emit-llvm %s -triple %itanium_abi_triple -fexceptions -fcxx-exceptions -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -debug-info-kind=limited -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics
#ifndef HEADER
#define HEADER
// CHECK-DAG: %struct.ident_t = type { i32, i32, i32, i32, i8* }
// CHECK-DAG: [[STR:@.+]] = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00"
// CHECK-DAG: [[DEF_LOC_2:@.+]] = private unnamed_addr global %struct.ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([23 x i8], [23 x i8]* [[STR]], i32 0, i32 0) }
// CHECK-DEBUG-DAG: %struct.ident_t = type { i32, i32, i32, i32, i8* }
// CHECK-DEBUG-DAG: [[STR:@.+]] = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00"
// CHECK-DEBUG-DAG: [[DEF_LOC_2:@.+]] = private unnamed_addr global %struct.ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([23 x i8], [23 x i8]* [[STR]], i32 0, i32 0) }
// CHECK-DEBUG-DAG: [[LOC1:@.+]] = private unnamed_addr constant [{{.+}} x i8] c";{{.*}}parallel_codegen.cpp;main;[[@LINE+19]];1;;\00"
// CHECK-DEBUG-DAG: [[LOC2:@.+]] = private unnamed_addr constant [{{.+}} x i8] c";{{.*}}parallel_codegen.cpp;tmain;[[@LINE+8]];1;;\00"

template <class T>
void foo(T argc) {}

template <typename T>
int tmain(T argc) {
  typedef double (*chunk_t)[argc[0][0]];
#pragma omp parallel
  {
  foo(argc);
  chunk_t var;(void)var[0][0];
  }
  return 0;
}

int main (int argc, char **argv) {
  int a[argc];
#pragma omp parallel
  foo(a[1]);
  return tmain(argv);
}

// CHECK-LABEL: define {{[a-z\_\b]*[ ]?i32}} @main({{i32[ ]?[a-z]*}} %argc, i8** %argv)
// CHECK: store i32 %argc, i32* [[ARGC_ADDR:%.+]],
// CHECK: [[VLA:%.+]] = alloca i32, i{{[0-9]+}} [[VLA_SIZE:%[^,]+]],
// CHECK:  call {{.*}}void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* [[DEF_LOC_2]], i32 2, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, i{{[0-9]+}}, i32*)* [[OMP_OUTLINED:@.+]] to void (i32*, i32*, ...)*), i{{[0-9]+}} [[VLA_SIZE]], i32* [[VLA]])
// CHECK-NEXT:  [[ARGV:%.+]] = load i8**, i8*** {{%[a-z0-9.]+}}
// CHECK-NEXT:  [[RET:%.+]] = call {{[a-z\_\b]*[ ]?i32}} [[TMAIN:@.+tmain.+]](i8** [[ARGV]])
// CHECK:       ret i32
// CHECK-NEXT:  }
// CHECK-DEBUG-LABEL: define i32 @main(i32 %argc, i8** %argv)
// CHECK-DEBUG:       [[LOC_2_ADDR:%.+]] = alloca %struct.ident_t
// CHECK-DEBUG:       [[KMPC_LOC_VOIDPTR:%.+]] = bitcast %struct.ident_t* [[LOC_2_ADDR]] to i8*
// CHECK-DEBUG-NEXT:  [[KMPC_DEFAULT_LOC_VOIDPTR:%.+]] = bitcast %struct.ident_t* [[DEF_LOC_2]] to i8*
// CHECK-DEBUG-NEXT:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[KMPC_LOC_VOIDPTR]], i8* align 8 [[KMPC_DEFAULT_LOC_VOIDPTR]], i64 24, i1 false)
// CHECK-DEBUG:       store i32 %argc, i32* [[ARGC_ADDR:%.+]],
// CHECK-DEBUG:       [[VLA:%.+]] = alloca i32, i64 [[VLA_SIZE:%[^,]+]],
// CHECK-DEBUG:       [[KMPC_LOC_PSOURCE_REF:%.+]] = getelementptr inbounds %struct.ident_t, %struct.ident_t* [[LOC_2_ADDR]], i32 0, i32 4
// CHECK-DEBUG-NEXT:  store i8* getelementptr inbounds ([{{.+}} x i8], [{{.+}} x i8]* [[LOC1]], i32 0, i32 0), i8** [[KMPC_LOC_PSOURCE_REF]]
// CHECK-DEBUG:       call {{.*}}void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* [[LOC_2_ADDR]], i32 2, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, i64, i32*)* [[OMP_OUTLINED:@.+]] to void (i32*, i32*, ...)*), i64 [[VLA_SIZE]], i32* [[VLA]])
// CHECK-DEBUG-NEXT:  [[ARGV:%.+]] = load i8**, i8*** {{%[a-z0-9.]+}}
// CHECK-DEBUG-NEXT:  [[RET:%.+]] = call i32 [[TMAIN:@.+tmain.+]](i8** [[ARGV]])
// CHECK-DEBUG:       ret i32
// CHECK-DEBUG-NEXT:  }

// CHECK:       define internal {{.*}}void [[OMP_OUTLINED]](i32* noalias %.global_tid., i32* noalias %.bound_tid., i{{[0-9]+}}{{.*}} [[VLA_SIZE:%.+]], i32* {{.+}} [[VLA_ADDR:%[^)]+]])
// CHECK-SAME:       #[[FN_ATTRS:[0-9]+]]
// CHECK:       store i32* [[VLA_ADDR]], i32** [[VLA_PTR_ADDR:%.+]],
// CHECK:       [[VLA_REF:%.+]] = load i32*, i32** [[VLA_PTR_ADDR]]
// CHECK:       [[VLA_ELEM_REF:%.+]] = getelementptr inbounds i32, i32* [[VLA_REF]], i{{[0-9]+}} 1
// CHECK-NEXT:  [[VLA_ELEM:%.+]] = load i32, i32* [[VLA_ELEM_REF]]
// CHECK-NEXT:  invoke {{.*}}void [[FOO:@.+foo.+]](i32{{[ ]?[a-z]*}} [[VLA_ELEM]])
// CHECK:       ret void
// CHECK:       call {{.*}}void @{{.+terminate.*|abort}}(
// CHECK-NEXT:  unreachable
// CHECK-NEXT:  }
// CHECK-DEBUG:       define internal void [[OMP_OUTLINED_DEBUG:@.+]](i32* noalias %.global_tid., i32* noalias %.bound_tid., i64 [[VLA_SIZE:%.+]], i32* {{.+}} [[VLA_ADDR:%[^)]+]])
// CHECK-DEBUG-SAME:  #[[FN_ATTRS:[0-9]+]]
// CHECK-DEBUG:       store i32* [[VLA_ADDR]], i32** [[VLA_PTR_ADDR:%.+]],
// CHECK-DEBUG:       [[VLA_REF:%.+]] = load i32*, i32** [[VLA_PTR_ADDR]]
// CHECK-DEBUG:       [[VLA_ELEM_REF:%.+]] = getelementptr inbounds i32, i32* [[VLA_REF]], i64 1
// CHECK-DEBUG-NEXT:  [[VLA_ELEM:%.+]] = load i32, i32* [[VLA_ELEM_REF]]
// CHECK-DEBUG-NEXT:  invoke void [[FOO:@.+foo.+]](i32 [[VLA_ELEM]])
// CHECK-DEBUG:       ret void
// CHECK-DEBUG:       call void @{{.+terminate.*|abort}}(
// CHECK-DEBUG-NEXT:  unreachable
// CHECK-DEBUG-NEXT:  }

// CHECK-DAG: define linkonce_odr {{.*}}void [[FOO]]({{i32[ ]?[a-z]*}} %argc)
// CHECK-DAG: declare !callback ![[cbid:[0-9]+]] {{.*}}void @__kmpc_fork_call(%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...)
// CHECK-DEBUG-DAG: define linkonce_odr void [[FOO]](i32 %argc)
// CHECK-DEBUG-DAG: declare !callback ![[cbid:[0-9]+]] void @__kmpc_fork_call(%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...)
// CHECK-DEBUG-DAG:       define internal void [[OMP_OUTLINED]](i32* noalias %.global_tid., i32* noalias %.bound_tid., i64 [[VLA_SIZE:%.+]], i32* {{.+}} [[VLA_ADDR:%[^)]+]])
// CHECK-DEBUG-DAG:       call void [[OMP_OUTLINED_DEBUG]]

// CHECK:       define linkonce_odr {{[a-z\_\b]*[ ]?i32}} [[TMAIN]](i8** %argc)
// CHECK:       store i8** %argc, i8*** [[ARGC_ADDR:%.+]],
// CHECK:       call {{.*}}void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* [[DEF_LOC_2]], i32 2, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, i8***, i{{64|32}})* [[OMP_OUTLINED:@.+]] to void (i32*, i32*, ...)*), i8*** [[ARGC_ADDR]], i{{64|32}} %{{.+}})
// CHECK-NEXT:  ret i32 0
// CHECK-NEXT:  }
// CHECK-DEBUG:       define linkonce_odr i32 [[TMAIN]](i8** %argc)
// CHECK-DEBUG-DAG:   [[LOC_2_ADDR:%.+]] = alloca %struct.ident_t
// CHECK-DEBUG:       [[KMPC_LOC_VOIDPTR:%.+]] = bitcast %struct.ident_t* [[LOC_2_ADDR]] to i8*
// CHECK-DEBUG-NEXT:  [[KMPC_DEFAULT_LOC_VOIDPTR:%.+]] = bitcast %struct.ident_t* [[DEF_LOC_2]] to i8*
// CHECK-DEBUG-NEXT:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[KMPC_LOC_VOIDPTR]], i8* align 8 [[KMPC_DEFAULT_LOC_VOIDPTR]], i64 24, i1 false)
// CHECK-DEBUG-NEXT:  store i8** %argc, i8*** [[ARGC_ADDR:%.+]],
// CHECK-DEBUG:  [[KMPC_LOC_PSOURCE_REF:%.+]] = getelementptr inbounds %struct.ident_t, %struct.ident_t* [[LOC_2_ADDR]], i32 0, i32 4
// CHECK-DEBUG-NEXT:  store i8* getelementptr inbounds ([{{.+}} x i8], [{{.+}} x i8]* [[LOC2]], i32 0, i32 0), i8** [[KMPC_LOC_PSOURCE_REF]]
// CHECK-DEBUG-NEXT:  call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* [[LOC_2_ADDR]], i32 2, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, i8***, i64)* [[OMP_OUTLINED:@.+]] to void (i32*, i32*, ...)*), i8*** [[ARGC_ADDR]], i64 %{{.+}})
// CHECK-DEBUG-NEXT:  ret i32 0
// CHECK-DEBUG-NEXT:  }

// CHECK:       define internal {{.*}}void [[OMP_OUTLINED]](i32* noalias %.global_tid., i32* noalias %.bound_tid., i8*** dereferenceable({{4|8}}) %argc, i{{64|32}}{{.*}} %{{.+}})
// CHECK:       store i8*** %argc, i8**** [[ARGC_PTR_ADDR:%.+]],
// CHECK:       [[ARGC_REF:%.+]] = load i8***, i8**** [[ARGC_PTR_ADDR]]
// CHECK:       [[ARGC:%.+]] = load i8**, i8*** [[ARGC_REF]]
// CHECK-NEXT:  invoke {{.*}}void [[FOO1:@.+foo.+]](i8** [[ARGC]])
// CHECK:       ret void
// CHECK:       call {{.*}}void @{{.+terminate.*|abort}}(
// CHECK-NEXT:  unreachable
// CHECK-NEXT:  }
// CHECK-DEBUG:       define internal void [[OMP_OUTLINED_DEBUG:@.+]](i32* noalias %.global_tid., i32* noalias %.bound_tid., i8*** dereferenceable({{4|8}}) %argc, i64 %{{.+}})
// CHECK-DEBUG:       store i8*** %argc, i8**** [[ARGC_PTR_ADDR:%.+]],
// CHECK-DEBUG:       [[ARGC_REF:%.+]] = load i8***, i8**** [[ARGC_PTR_ADDR]]
// CHECK-DEBUG:       [[ARGC:%.+]] = load i8**, i8*** [[ARGC_REF]]
// CHECK-DEBUG-NEXT:  invoke void [[FOO1:@.+foo.+]](i8** [[ARGC]])
// CHECK-DEBUG:       ret void
// CHECK-DEBUG:       call void @{{.+terminate.*|abort}}(
// CHECK-DEBUG-NEXT:  unreachable
// CHECK-DEBUG-NEXT:  }

// CHECK: define linkonce_odr {{.*}}void [[FOO1]](i8** %argc)
// CHECK-DEBUG-DAG: define linkonce_odr void [[FOO1]](i8** %argc)
// CHECK-DEBUG-DAG: define internal void [[OMP_OUTLINED]](i32* noalias %.global_tid., i32* noalias %.bound_tid., i8*** dereferenceable({{4|8}}) %argc, i64 %{{.+}})
// CHECK-DEBUG-DAG: call void [[OMP_OUTLINED_DEBUG]]({{[^)]+}}){{[^,]*}}, !dbg

// CHECK: attributes #[[FN_ATTRS]] = {{.+}} nounwind
// CHECK-DEBUG: attributes #[[FN_ATTRS]] = {{.+}} nounwind
// CHECK: ![[cbid]] = !{![[cbidb:[0-9]+]]}
// CHECK: ![[cbidb]] = !{i64 2, i64 -1, i64 -1, i1 true}
#endif
