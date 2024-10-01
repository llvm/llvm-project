// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -fopenmp -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -triple x86_64-apple-darwin10 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -triple x86_64-apple-darwin10 -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -fopenmp-simd -emit-llvm -o - %s | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -triple x86_64-apple-darwin10 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -triple x86_64-apple-darwin10 -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

// CHECK-DAG: [[MAIN_A:@.+]] = internal global ptr null,
// CHECK-DAG: [[TMAIN_A:@.+]] = linkonce_odr global ptr null,

typedef void *omp_depend_t;

void foo() {}

template <class T>
T tmain(T argc) {
  static T a;
  int *argv;
#pragma omp depobj(a) depend(in:argv, ([3][*(int*)argv][4])argv)
#pragma omp depobj(argc) destroy
#pragma omp depobj(argc) update(inout)
  return argc;
}

int main(int argc, char **argv) {
  static omp_depend_t a;
  omp_depend_t b;
#pragma omp depobj(a) depend(out:argc, argv)
#pragma omp depobj(b) destroy
#pragma omp depobj(b) update(mutexinoutset)
#pragma omp depobj(a) depend(iterator(char *p = argv[argc]:argv[0]:-1), out: p[0])
  (void)tmain(a), tmain(b);
  return 0;
}

// CHECK-LABEL: @main
// CHECK: [[B_ADDR:%b]] = alloca ptr,
// CHECK: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num(
// CHECK: [[DEP_ADDR_VOID:%.+]] = call ptr @__kmpc_alloc(i32 [[GTID]], i64 72, ptr null)
// CHECK: [[SZ_BASE:%.+]] = getelementptr inbounds nuw %struct.kmp_depend_info, ptr [[DEP_ADDR_VOID]], i{{.+}} 0, i{{.+}} 0
// CHECK: store i64 2, ptr [[SZ_BASE]], align 8
// CHECK: [[BASE_ADDR:%.+]] = getelementptr %struct.kmp_depend_info, ptr [[DEP_ADDR_VOID]], i{{.+}} 1
// CHECK: [[ADDR:%.+]] = getelementptr inbounds nuw %struct.kmp_depend_info, ptr [[BASE_ADDR]], i{{.+}} 0, i{{.+}} 0
// CHECK: store i64 %{{.+}}, ptr [[ADDR]], align 8
// CHECK: [[SZ_ADDR:%.+]] = getelementptr inbounds nuw %struct.kmp_depend_info, ptr [[BASE_ADDR]], i{{.+}} 0, i{{.+}} 1
// CHECK: store i64 4, ptr [[SZ_ADDR]], align 8
// CHECK: [[FLAGS_ADDR:%.+]] = getelementptr inbounds nuw %struct.kmp_depend_info, ptr [[BASE_ADDR]], i{{.+}} 0, i{{.+}} 2
// CHECK: store i8 3, ptr [[FLAGS_ADDR]], align 8
// CHECK: [[BASE_ADDR:%.+]] = getelementptr %struct.kmp_depend_info, ptr [[DEP_ADDR_VOID]], i{{.+}} 2
// CHECK: [[ADDR:%.+]] = getelementptr inbounds nuw %struct.kmp_depend_info, ptr [[BASE_ADDR]], i{{.+}} 0, i{{.+}} 0
// CHECK: store i64 %{{.+}}, ptr [[ADDR]], align 8
// CHECK: [[SZ_ADDR:%.+]] = getelementptr inbounds nuw %struct.kmp_depend_info, ptr [[BASE_ADDR]], i{{.+}} 0, i{{.+}} 1
// CHECK: store i64 8, ptr [[SZ_ADDR]], align 8
// CHECK: [[FLAGS_ADDR:%.+]] = getelementptr inbounds nuw %struct.kmp_depend_info, ptr [[BASE_ADDR]], i{{.+}} 0, i{{.+}} 2
// CHECK: store i8 3, ptr [[FLAGS_ADDR]], align 8
// CHECK: [[BASE_ADDR:%.+]] = getelementptr %struct.kmp_depend_info, ptr [[DEP_ADDR_VOID]], i{{.+}} 1
// CHECK: store ptr [[BASE_ADDR]], ptr [[MAIN_A]], align 8
// CHECK: [[B:%.+]] = load ptr, ptr [[B_ADDR]], align 8
// CHECK: [[B_REF:%.+]] = getelementptr %struct.kmp_depend_info, ptr [[B]], i{{.+}} -1
// CHECK: call void @__kmpc_free(i32 [[GTID]], ptr [[B_REF]], ptr null)
// CHECK: [[B_BASE:%.+]] = load ptr, ptr [[B_ADDR]], align 8
// CHECK: [[NUMDEPS_BASE:%.+]] = getelementptr %struct.kmp_depend_info, ptr [[B_BASE]], i64 -1
// CHECK: [[NUMDEPS_ADDR:%.+]] = getelementptr inbounds nuw %struct.kmp_depend_info, ptr [[NUMDEPS_BASE]], i{{.+}} 0, i{{.+}} 0
// CHECK: [[NUMDEPS:%.+]] = load i64, ptr [[NUMDEPS_ADDR]], align 8
// CHECK: [[END:%.+]] = getelementptr %struct.kmp_depend_info, ptr [[B_BASE]], i64 [[NUMDEPS]]
// CHECK: br label %[[BODY:.+]]
// CHECK: [[BODY]]:
// CHECK: [[EL:%.+]] = phi ptr [ [[B_BASE]], %{{.+}} ], [ [[EL_NEXT:%.+]], %[[BODY]] ]
// CHECK: [[FLAG_BASE:%.+]] = getelementptr inbounds nuw %struct.kmp_depend_info, ptr [[EL]], i{{.+}} 0, i{{.+}} 2
// CHECK: store i8 4, ptr [[FLAG_BASE]], align 8
// CHECK: [[EL_NEXT]] = getelementptr %struct.kmp_depend_info, ptr [[EL]], i{{.+}} 1
// CHECK: [[IS_DONE:%.+]] = icmp eq ptr [[EL_NEXT]], [[END]]
// CHECK: br i1 [[IS_DONE]], label %[[DONE:.+]], label %[[BODY]]
// CHECK: [[DONE]]:

// Claculate toal number of elements.
// (argv[argc]-argv[0]-(-1)-1) / -(-1);
// CHECK: [[ARGV:%.+]] = load ptr, ptr [[ARGV_ADDR:%.+]], align 8
// CHECK: [[ARGC:%.+]] = load i32, ptr [[ARGC_ADDR:%.+]], align 4
// CHECK: [[IDX:%.+]] = sext i32 [[ARGC]] to i64
// CHECK: [[BEGIN_ADDR:%.+]] = getelementptr inbounds ptr, ptr [[ARGV]], i64 [[IDX]]
// CHECK: [[BEGIN:%.+]] = load ptr, ptr [[BEGIN_ADDR]], align 8
// CHECK: [[ARGV:%.+]] = load ptr, ptr [[ARGV_ADDR]], align 8
// CHECK: [[END_ADDR:%.+]] = getelementptr inbounds ptr, ptr [[ARGV]], i64 0
// CHECK: [[END:%.+]] = load ptr, ptr [[END_ADDR]], align 8
// CHECK: [[BEGIN_INT:%.+]] = ptrtoint ptr [[BEGIN]] to i64
// CHECK: [[END_INT:%.+]] = ptrtoint ptr [[END]] to i64
// CHECK: [[BE_SUB:%.+]] = sub i64 [[BEGIN_INT]], [[END_INT]]
// CHECK: [[BE_SUB_ST_SUB:%.+]] = add nsw i64 [[BE_SUB]], 1
// CHECK: [[BE_SUB_ST_SUB_1_SUB:%.+]] = sub nsw i64 [[BE_SUB_ST_SUB]], 1
// CHECK: [[BE_SUB_ST_SUB_1_SUB_1_DIV:%.+]] = sdiv i64 [[BE_SUB_ST_SUB_1_SUB]], 1
// CHECK: [[NELEMS:%.+]] = mul nuw i64 1, [[BE_SUB_ST_SUB_1_SUB_1_DIV]]

// Allocate size is (NELEMS + 1) * sizeof(%struct.kmp_depend_info).
// sizeof(%struct.kmp_depend_info) == 24;
// CHECK: [[EXTRA_SZ:%.+]] = add nuw i64 1, [[NELEMS]]
// CHECK: [[SIZE:%.+]] = mul nuw i64 [[EXTRA_SZ]], 24

// Allocate memory
// kmp_depend_info* dep = (kmp_depend_info*)kmpc_alloc(SIZE);
// CHECK: [[DEP_ADDR_VOID:%.+]] = call ptr @__kmpc_alloc(i32 %{{.+}}, i64 [[SIZE]], ptr null)

// dep[0].base_addr = NELEMS.
// CHECK: [[BASE_ADDR_ADDR:%.+]] = getelementptr inbounds nuw %struct.kmp_depend_info, ptr [[DEP_ADDR_VOID]], i{{.+}} 0, i{{.+}} 0
// CHECK: store i64 [[NELEMS]], ptr [[BASE_ADDR_ADDR]], align 8

// iterator_counter = 1;
// CHECK: store i64 1, ptr [[ITERATOR_COUNTER_ADDR:%.+]], align 8

// NITER = (argv[argc]-argv[0]-(-1)-1) / -(-1);
// CHECK: [[ARGV:%.+]] = load ptr, ptr [[ARGV_ADDR]], align 8
// CHECK: [[ARGC:%.+]] = load i32, ptr [[ARGC_ADDR]], align 4
// CHECK: [[IDX:%.+]] = sext i32 [[ARGC]] to i64
// CHECK: [[BEGIN_ADDR:%.+]] = getelementptr inbounds ptr, ptr [[ARGV]], i64 [[IDX]]
// CHECK: [[BEGIN:%.+]] = load ptr, ptr [[BEGIN_ADDR]], align 8
// CHECK: [[ARGV:%.+]] = load ptr, ptr [[ARGV_ADDR]], align 8
// CHECK: [[END_ADDR:%.+]] = getelementptr inbounds ptr, ptr [[ARGV]], i64 0
// CHECK: [[END:%.+]] = load ptr, ptr [[END_ADDR]], align 8
// CHECK: [[BEGIN_INT:%.+]] = ptrtoint ptr [[BEGIN]] to i64
// CHECK: [[END_INT:%.+]] = ptrtoint ptr [[END]] to i64
// CHECK: [[BE_SUB:%.+]] = sub i64 [[BEGIN_INT]], [[END_INT]]
// CHECK: [[BE_SUB_ST_SUB:%.+]] = add nsw i64 [[BE_SUB]], 1
// CHECK: [[BE_SUB_ST_SUB_1_SUB:%.+]] = sub nsw i64 [[BE_SUB_ST_SUB]], 1
// CHECK: [[NITER:%.+]] = sdiv i64 [[BE_SUB_ST_SUB_1_SUB]], 1

// Loop.
// CHECK: store i64 0, ptr [[COUNTER_ADDR:%.+]], align 8
// CHECK: br label %[[CONT:.+]]

// CHECK: [[CONT]]:
// CHECK: [[COUNTER:%.+]] = load i64, ptr [[COUNTER_ADDR]], align 8
// CHECK: [[CMP:%.+]] = icmp slt i64 [[COUNTER]], [[NITER]]
// CHECK: br i1 [[CMP]], label %[[BODY:.+]], label %[[EXIT:.+]]

// CHECK: [[BODY]]:

// p = BEGIN + COUNTER * STEP;
// CHECK: [[ARGV:%.+]] = load ptr, ptr [[ARGV_ADDR]], align 8
// CHECK: [[ARGC:%.+]] = load i32, ptr [[ARGC_ADDR]], align 4
// CHECK: [[IDX:%.+]] = sext i32 [[ARGC]] to i64
// CHECK: [[BEGIN_ADDR:%.+]] = getelementptr inbounds ptr, ptr [[ARGV]], i64 [[IDX]]
// CHECK: [[BEGIN:%.+]] = load ptr, ptr [[BEGIN_ADDR]], align 8
// CHECK: [[COUNTER:%.+]] = load i64, ptr [[COUNTER_ADDR]], align 8
// CHECK: [[CS_MUL:%.+]] = mul nsw i64 [[COUNTER]], -1
// CHECK: [[CS_MUL_BEGIN_ADD:%.+]] = getelementptr inbounds i8, ptr [[BEGIN]], i64 [[CS_MUL]]
// CHECK: store ptr [[CS_MUL_BEGIN_ADD]], ptr [[P_ADDR:%.+]], align 8

// &p[0]
// CHECK: [[P:%.+]] = load ptr, ptr [[P_ADDR]], align 8
// CHECK: [[P0:%.+]] = getelementptr inbounds i8, ptr [[P]], i64 0
// CHECK: [[P0_ADDR:%.+]] = ptrtoint ptr [[P0]] to i64

// dep[ITERATOR_COUNTER].base_addr = &p[0];
// CHECK: [[ITERATOR_COUNTER:%.+]] = load i64, ptr [[ITERATOR_COUNTER_ADDR]], align 8
// CHECK: [[DEP_IC:%.+]] = getelementptr %struct.kmp_depend_info, ptr [[DEP_ADDR_VOID]], i64 [[ITERATOR_COUNTER]]
// CHECK: [[DEP_IC_BASE_ADDR:%.+]] = getelementptr inbounds nuw %struct.kmp_depend_info, ptr [[DEP_IC]], i{{.+}} 0, i{{.+}} 0
// CHECK: store i64 [[P0_ADDR]], ptr [[DEP_IC_BASE_ADDR]], align 8

// dep[ITERATOR_COUNTER].size = sizeof(p[0]);
// CHECK: [[DEP_IC_SIZE:%.+]] = getelementptr inbounds nuw %struct.kmp_depend_info, ptr [[DEP_IC]], i{{.+}} 0, i{{.+}} 1
// CHECK: store i64 1, ptr [[DEP_IC_SIZE]], align 8
// dep[ITERATOR_COUNTER].flags = in_out;
// CHECK: [[DEP_IC_FLAGS:%.+]] = getelementptr inbounds nuw %struct.kmp_depend_info, ptr [[DEP_IC]], i{{.+}} 0, i{{.+}} 2
// CHECK: store i8 3, ptr [[DEP_IC_FLAGS]], align 8

// ITERATOR_COUNTER = ITERATOR_COUNTER + 1;
// CHECK: [[ITERATOR_COUNTER:%.+]] = load i64, ptr [[ITERATOR_COUNTER_ADDR]], align 8
// CHECK: [[INC:%.+]] = add nuw i64 [[ITERATOR_COUNTER]], 1
// CHECK: store i64 [[INC]], ptr [[ITERATOR_COUNTER_ADDR]], align 8

// COUNTER = COUNTER + 1;
// CHECK: [[COUNTER:%.+]] = load i64, ptr [[COUNTER_ADDR]], align 8
// CHECK: [[INC:%.+]] = add nsw i64 [[COUNTER]], 1
// CHECK: store i64 [[INC]], ptr [[COUNTER_ADDR]], align 8
// CHECK: br label %[[CONT]]

// CHECK: [[EXIT]]:

// a = &dep[1];
// CHECK: [[DEP_BEGIN:%.+]] = getelementptr %struct.kmp_depend_info, ptr [[DEP_ADDR_VOID]], i64 1
// CHECK: store ptr [[DEP_BEGIN]], ptr [[MAIN_A]], align 8

// CHECK-LABEL: tmain
// CHECK: [[ARGC_ADDR:%.+]] = alloca ptr,
// CHECK: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num(
// CHECK: [[DEP_ADDR_VOID:%.+]] = call ptr @__kmpc_alloc(i32 [[GTID]], i64 72, ptr null)
// CHECK: [[SZ_BASE:%.+]] = getelementptr inbounds nuw %struct.kmp_depend_info, ptr [[DEP_ADDR_VOID]], i{{.+}} 0, i{{.+}} 0
// CHECK: store i64 2, ptr [[SZ_BASE]], align 8
// CHECK: [[BASE_ADDR:%.+]] = getelementptr %struct.kmp_depend_info, ptr [[DEP_ADDR_VOID]], i{{.+}} 1
// CHECK: [[ADDR:%.+]] = getelementptr inbounds nuw %struct.kmp_depend_info, ptr [[BASE_ADDR]], i{{.+}} 0, i{{.+}} 0
// CHECK: store i64 %{{.+}}, ptr [[ADDR]], align 8
// CHECK: [[SZ_ADDR:%.+]] = getelementptr inbounds nuw %struct.kmp_depend_info, ptr [[BASE_ADDR]], i{{.+}} 0, i{{.+}} 1
// CHECK: store i64 8, ptr [[SZ_ADDR]], align 8
// CHECK: [[FLAGS_ADDR:%.+]] = getelementptr inbounds nuw %struct.kmp_depend_info, ptr [[BASE_ADDR]], i{{.+}} 0, i{{.+}} 2
// CHECK: store i8 1, ptr [[FLAGS_ADDR]], align 8
// CHECK: [[SHAPE_ADDR:%.+]] = load ptr, ptr [[ARGV_ADDR:%.+]], align 8
// CHECK: [[SZ1:%.+]] = mul nuw i64 12, %{{.+}}
// CHECK: [[SZ:%.+]] = mul nuw i64 [[SZ1]], 4
// CHECK: [[SHAPE:%.+]] = ptrtoint ptr [[SHAPE_ADDR]] to i64
// CHECK: [[BASE_ADDR:%.+]] = getelementptr %struct.kmp_depend_info, ptr [[DEP_ADDR_VOID]], i{{.+}} 2
// CHECK: [[ADDR:%.+]] = getelementptr inbounds nuw %struct.kmp_depend_info, ptr [[BASE_ADDR]], i{{.+}} 0, i{{.+}} 0
// CHECK: store i64 [[SHAPE]], ptr [[ADDR]], align 8
// CHECK: [[SZ_ADDR:%.+]] = getelementptr inbounds nuw %struct.kmp_depend_info, ptr [[BASE_ADDR]], i{{.+}} 0, i{{.+}} 1
// CHECK: store i64 [[SZ]], ptr [[SZ_ADDR]], align 8
// CHECK: [[FLAGS_ADDR:%.+]] = getelementptr inbounds nuw %struct.kmp_depend_info, ptr [[BASE_ADDR]], i{{.+}} 0, i{{.+}} 2
// CHECK: store i8 1, ptr [[FLAGS_ADDR]], align 8
// CHECK: [[BASE_ADDR:%.+]] = getelementptr %struct.kmp_depend_info, ptr [[DEP_ADDR_VOID]], i{{.+}} 1
// CHECK: store ptr [[BASE_ADDR]], ptr [[TMAIN_A]], align 8
// CHECK: [[ARGC:%.+]] = load ptr, ptr [[ARGC_ADDR]], align 8
// CHECK: [[ARGC_REF:%.+]] = getelementptr %struct.kmp_depend_info, ptr [[ARGC]], i{{.+}} -1
// CHECK: call void @__kmpc_free(i32 [[GTID]], ptr [[ARGC_REF]], ptr null)
// CHECK: [[ARGC_BASE:%.+]] = load ptr, ptr [[ARGC_ADDR]], align 8
// CHECK: [[NUMDEPS_BASE:%.+]] = getelementptr %struct.kmp_depend_info, ptr [[ARGC_BASE]], i64 -1
// CHECK: [[NUMDEPS_ADDR:%.+]] = getelementptr inbounds nuw %struct.kmp_depend_info, ptr [[NUMDEPS_BASE]], i{{.+}} 0, i{{.+}} 0
// CHECK: [[NUMDEPS:%.+]] = load i64, ptr [[NUMDEPS_ADDR]], align 8
// CHECK: [[END:%.+]] = getelementptr %struct.kmp_depend_info, ptr [[ARGC_BASE]], i64 [[NUMDEPS]]
// CHECK: br label %[[BODY:.+]]
// CHECK: [[BODY]]:
// CHECK: [[EL:%.+]] = phi ptr [ [[ARGC_BASE]], %{{.+}} ], [ [[EL_NEXT:%.+]], %[[BODY]] ]
// CHECK: [[FLAG_BASE:%.+]] = getelementptr inbounds nuw %struct.kmp_depend_info, ptr [[EL]], i{{.+}} 0, i{{.+}} 2
// CHECK: store i8 3, ptr [[FLAG_BASE]], align 8
// CHECK: [[EL_NEXT]] = getelementptr %struct.kmp_depend_info, ptr [[EL]], i{{.+}} 1
// CHECK: [[IS_DONE:%.+]] = icmp eq ptr [[EL_NEXT]], [[END]]
// CHECK: br i1 [[IS_DONE]], label %[[DONE:.+]], label %[[BODY]]
// CHECK: [[DONE]]:

#endif
