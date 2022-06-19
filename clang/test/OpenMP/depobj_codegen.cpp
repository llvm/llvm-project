// RUN: %clang_cc1 -no-opaque-pointers -verify -triple x86_64-apple-darwin10 -fopenmp -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -triple x86_64-apple-darwin10 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -triple x86_64-apple-darwin10 -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -no-opaque-pointers -verify -triple x86_64-apple-darwin10 -fopenmp-simd -emit-llvm -o - %s | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -triple x86_64-apple-darwin10 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -triple x86_64-apple-darwin10 -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

// CHECK-DAG: [[MAIN_A:@.+]] = internal global i8* null,
// CHECK-DAG: [[TMAIN_A:@.+]] = linkonce_odr global i8* null,

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
// CHECK: [[B_ADDR:%.+]] = alloca i8*,
// CHECK: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num(
// CHECK: [[DEP_ADDR_VOID:%.+]] = call i8* @__kmpc_alloc(i32 [[GTID]], i64 72, i8* null)
// CHECK: [[DEP_ADDR:%.+]] = bitcast i8* [[DEP_ADDR_VOID]] to %struct.kmp_depend_info*
// CHECK: [[SZ_BASE:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEP_ADDR]], i{{.+}} 0, i{{.+}} 0
// CHECK: store i64 2, i64* [[SZ_BASE]], align 8
// CHECK: [[BASE_ADDR:%.+]] = getelementptr %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEP_ADDR]], i{{.+}} 1
// CHECK: [[ADDR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[BASE_ADDR]], i{{.+}} 0, i{{.+}} 0
// CHECK: store i64 %{{.+}}, i64* [[ADDR]], align 8
// CHECK: [[SZ_ADDR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[BASE_ADDR]], i{{.+}} 0, i{{.+}} 1
// CHECK: store i64 4, i64* [[SZ_ADDR]], align 8
// CHECK: [[FLAGS_ADDR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[BASE_ADDR]], i{{.+}} 0, i{{.+}} 2
// CHECK: store i8 3, i8* [[FLAGS_ADDR]], align 8
// CHECK: [[BASE_ADDR:%.+]] = getelementptr %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEP_ADDR]], i{{.+}} 2
// CHECK: [[ADDR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[BASE_ADDR]], i{{.+}} 0, i{{.+}} 0
// CHECK: store i64 %{{.+}}, i64* [[ADDR]], align 8
// CHECK: [[SZ_ADDR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[BASE_ADDR]], i{{.+}} 0, i{{.+}} 1
// CHECK: store i64 8, i64* [[SZ_ADDR]], align 8
// CHECK: [[FLAGS_ADDR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[BASE_ADDR]], i{{.+}} 0, i{{.+}} 2
// CHECK: store i8 3, i8* [[FLAGS_ADDR]], align 8
// CHECK: [[BASE_ADDR:%.+]] = getelementptr %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEP_ADDR]], i{{.+}} 1
// CHECK: [[DEP:%.+]] = bitcast %struct.kmp_depend_info* [[BASE_ADDR]] to i8*
// CHECK: store i8* [[DEP]], i8** [[MAIN_A]], align 8
// CHECK: [[B:%.+]] = load i8*, i8** [[B_ADDR]], align 8
// CHECK: [[B_BASE:%.+]] = bitcast i8* [[B]] to %struct.kmp_depend_info*
// CHECK: [[B_REF:%.+]] = getelementptr %struct.kmp_depend_info, %struct.kmp_depend_info* [[B_BASE]], i{{.+}} -1
// CHECK: [[B:%.+]] = bitcast %struct.kmp_depend_info* [[B_REF]] to i8*
// CHECK: call void @__kmpc_free(i32 [[GTID]], i8* [[B]], i8* null)
// CHECK: [[B_ADDR_CAST:%.+]] = bitcast i8** [[B_ADDR]] to %struct.kmp_depend_info**
// CHECK: [[B_BASE:%.+]] = load %struct.kmp_depend_info*, %struct.kmp_depend_info** [[B_ADDR_CAST]], align 8
// CHECK: [[NUMDEPS_BASE:%.+]] = getelementptr %struct.kmp_depend_info, %struct.kmp_depend_info* [[B_BASE]], i64 -1
// CHECK: [[NUMDEPS_ADDR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[NUMDEPS_BASE]], i{{.+}} 0, i{{.+}} 0
// CHECK: [[NUMDEPS:%.+]] = load i64, i64* [[NUMDEPS_ADDR]], align 8
// CHECK: [[END:%.+]] = getelementptr %struct.kmp_depend_info, %struct.kmp_depend_info* [[B_BASE]], i64 [[NUMDEPS]]
// CHECK: br label %[[BODY:.+]]
// CHECK: [[BODY]]:
// CHECK: [[EL:%.+]] = phi %struct.kmp_depend_info* [ [[B_BASE]], %{{.+}} ], [ [[EL_NEXT:%.+]], %[[BODY]] ]
// CHECK: [[FLAG_BASE:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[EL]], i{{.+}} 0, i{{.+}} 2
// CHECK: store i8 4, i8* [[FLAG_BASE]], align 8
// CHECK: [[EL_NEXT]] = getelementptr %struct.kmp_depend_info, %struct.kmp_depend_info* [[EL]], i{{.+}} 1
// CHECK: [[IS_DONE:%.+]] = icmp eq %struct.kmp_depend_info* [[EL_NEXT]], [[END]]
// CHECK: br i1 [[IS_DONE]], label %[[DONE:.+]], label %[[BODY]]
// CHECK: [[DONE]]:

// Claculate toal number of elements.
// (argv[argc]-argv[0]-(-1)-1) / -(-1);
// CHECK: [[ARGV:%.+]] = load i8**, i8*** [[ARGV_ADDR:%.+]], align 8
// CHECK: [[ARGC:%.+]] = load i32, i32* [[ARGC_ADDR:%.+]], align 4
// CHECK: [[IDX:%.+]] = sext i32 [[ARGC]] to i64
// CHECK: [[BEGIN_ADDR:%.+]] = getelementptr inbounds i8*, i8** [[ARGV]], i64 [[IDX]]
// CHECK: [[BEGIN:%.+]] = load i8*, i8** [[BEGIN_ADDR]], align 8
// CHECK: [[ARGV:%.+]] = load i8**, i8*** [[ARGV_ADDR]], align 8
// CHECK: [[END_ADDR:%.+]] = getelementptr inbounds i8*, i8** [[ARGV]], i64 0
// CHECK: [[END:%.+]] = load i8*, i8** [[END_ADDR]], align 8
// CHECK: [[BEGIN_INT:%.+]] = ptrtoint i8* [[BEGIN]] to i64
// CHECK: [[END_INT:%.+]] = ptrtoint i8* [[END]] to i64
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
// CHECK: [[DEP_ADDR_VOID:%.+]] = call i8* @__kmpc_alloc(i32 %{{.+}}, i64 [[SIZE]], i8* null)
// CHECK: [[DEP_ADDR:%.+]] = bitcast i8* [[DEP_ADDR_VOID]] to %struct.kmp_depend_info*

// dep[0].base_addr = NELEMS.
// CHECK: [[BASE_ADDR_ADDR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEP_ADDR]], i{{.+}} 0, i{{.+}} 0
// CHECK: store i64 [[NELEMS]], i64* [[BASE_ADDR_ADDR]], align 8

// iterator_counter = 1;
// CHECK: store i64 1, i64* [[ITERATOR_COUNTER_ADDR:%.+]], align 8

// NITER = (argv[argc]-argv[0]-(-1)-1) / -(-1);
// CHECK: [[ARGV:%.+]] = load i8**, i8*** [[ARGV_ADDR]], align 8
// CHECK: [[ARGC:%.+]] = load i32, i32* [[ARGC_ADDR]], align 4
// CHECK: [[IDX:%.+]] = sext i32 [[ARGC]] to i64
// CHECK: [[BEGIN_ADDR:%.+]] = getelementptr inbounds i8*, i8** [[ARGV]], i64 [[IDX]]
// CHECK: [[BEGIN:%.+]] = load i8*, i8** [[BEGIN_ADDR]], align 8
// CHECK: [[ARGV:%.+]] = load i8**, i8*** [[ARGV_ADDR]], align 8
// CHECK: [[END_ADDR:%.+]] = getelementptr inbounds i8*, i8** [[ARGV]], i64 0
// CHECK: [[END:%.+]] = load i8*, i8** [[END_ADDR]], align 8
// CHECK: [[BEGIN_INT:%.+]] = ptrtoint i8* [[BEGIN]] to i64
// CHECK: [[END_INT:%.+]] = ptrtoint i8* [[END]] to i64
// CHECK: [[BE_SUB:%.+]] = sub i64 [[BEGIN_INT]], [[END_INT]]
// CHECK: [[BE_SUB_ST_SUB:%.+]] = add nsw i64 [[BE_SUB]], 1
// CHECK: [[BE_SUB_ST_SUB_1_SUB:%.+]] = sub nsw i64 [[BE_SUB_ST_SUB]], 1
// CHECK: [[NITER:%.+]] = sdiv i64 [[BE_SUB_ST_SUB_1_SUB]], 1

// Loop.
// CHECK: store i64 0, i64* [[COUNTER_ADDR:%.+]], align 8
// CHECK: br label %[[CONT:.+]]

// CHECK: [[CONT]]:
// CHECK: [[COUNTER:%.+]] = load i64, i64* [[COUNTER_ADDR]], align 8
// CHECK: [[CMP:%.+]] = icmp slt i64 [[COUNTER]], [[NITER]]
// CHECK: br i1 [[CMP]], label %[[BODY:.+]], label %[[EXIT:.+]]

// CHECK: [[BODY]]:

// p = BEGIN + COUNTER * STEP;
// CHECK: [[ARGV:%.+]] = load i8**, i8*** [[ARGV_ADDR]], align 8
// CHECK: [[ARGC:%.+]] = load i32, i32* [[ARGC_ADDR]], align 4
// CHECK: [[IDX:%.+]] = sext i32 [[ARGC]] to i64
// CHECK: [[BEGIN_ADDR:%.+]] = getelementptr inbounds i8*, i8** [[ARGV]], i64 [[IDX]]
// CHECK: [[BEGIN:%.+]] = load i8*, i8** [[BEGIN_ADDR]], align 8
// CHECK: [[COUNTER:%.+]] = load i64, i64* [[COUNTER_ADDR]], align 8
// CHECK: [[CS_MUL:%.+]] = mul nsw i64 [[COUNTER]], -1
// CHECK: [[CS_MUL_BEGIN_ADD:%.+]] = getelementptr inbounds i8, i8* [[BEGIN]], i64 [[CS_MUL]]
// CHECK: store i8* [[CS_MUL_BEGIN_ADD]], i8** [[P_ADDR:%.+]], align 8

// &p[0]
// CHECK: [[P:%.+]] = load i8*, i8** [[P_ADDR]], align 8
// CHECK: [[P0:%.+]] = getelementptr inbounds i8, i8* [[P]], i64 0
// CHECK: [[P0_ADDR:%.+]] = ptrtoint i8* [[P0]] to i64

// dep[ITERATOR_COUNTER].base_addr = &p[0];
// CHECK: [[ITERATOR_COUNTER:%.+]] = load i64, i64* [[ITERATOR_COUNTER_ADDR]], align 8
// CHECK: [[DEP_IC:%.+]] = getelementptr %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEP_ADDR]], i64 [[ITERATOR_COUNTER]]
// CHECK: [[DEP_IC_BASE_ADDR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEP_IC]], i{{.+}} 0, i{{.+}} 0
// CHECK: store i64 [[P0_ADDR]], i64* [[DEP_IC_BASE_ADDR]], align 8

// dep[ITERATOR_COUNTER].size = sizeof(p[0]);
// CHECK: [[DEP_IC_SIZE:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEP_IC]], i{{.+}} 0, i{{.+}} 1
// CHECK: store i64 1, i64* [[DEP_IC_SIZE]], align 8
// dep[ITERATOR_COUNTER].flags = in_out;
// CHECK: [[DEP_IC_FLAGS:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEP_IC]], i{{.+}} 0, i{{.+}} 2
// CHECK: store i8 3, i8* [[DEP_IC_FLAGS]], align 8

// ITERATOR_COUNTER = ITERATOR_COUNTER + 1;
// CHECK: [[ITERATOR_COUNTER:%.+]] = load i64, i64* [[ITERATOR_COUNTER_ADDR]], align 8
// CHECK: [[INC:%.+]] = add nuw i64 [[ITERATOR_COUNTER]], 1
// CHECK: store i64 [[INC]], i64* [[ITERATOR_COUNTER_ADDR]], align 8

// COUNTER = COUNTER + 1;
// CHECK: [[COUNTER:%.+]] = load i64, i64* [[COUNTER_ADDR]], align 8
// CHECK: [[INC:%.+]] = add nsw i64 [[COUNTER]], 1
// CHECK: store i64 [[INC]], i64* [[COUNTER_ADDR]], align 8
// CHECK: br label %[[CONT]]

// CHECK: [[EXIT]]:

// a = &dep[1];
// CHECK: [[DEP_BEGIN:%.+]] = getelementptr %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEP_ADDR]], i64 1
// CHECK: [[DEP:%.+]] = bitcast %struct.kmp_depend_info* [[DEP_BEGIN]] to i8*
// CHECK: store i8* [[DEP]], i8** [[MAIN_A]], align 8

// CHECK-LABEL: tmain
// CHECK: [[ARGC_ADDR:%.+]] = alloca i8*,
// CHECK: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num(
// CHECK: [[DEP_ADDR_VOID:%.+]] = call i8* @__kmpc_alloc(i32 [[GTID]], i64 72, i8* null)
// CHECK: [[DEP_ADDR:%.+]] = bitcast i8* [[DEP_ADDR_VOID]] to %struct.kmp_depend_info*
// CHECK: [[SZ_BASE:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEP_ADDR]], i{{.+}} 0, i{{.+}} 0
// CHECK: store i64 2, i64* [[SZ_BASE]], align 8
// CHECK: [[BASE_ADDR:%.+]] = getelementptr %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEP_ADDR]], i{{.+}} 1
// CHECK: [[ADDR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[BASE_ADDR]], i{{.+}} 0, i{{.+}} 0
// CHECK: store i64 %{{.+}}, i64* [[ADDR]], align 8
// CHECK: [[SZ_ADDR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[BASE_ADDR]], i{{.+}} 0, i{{.+}} 1
// CHECK: store i64 8, i64* [[SZ_ADDR]], align 8
// CHECK: [[FLAGS_ADDR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[BASE_ADDR]], i{{.+}} 0, i{{.+}} 2
// CHECK: store i8 1, i8* [[FLAGS_ADDR]], align 8
// CHECK: [[SHAPE_ADDR:%.+]] = load i32*, i32** [[ARGV_ADDR:%.+]], align 8
// CHECK: [[SZ1:%.+]] = mul nuw i64 12, %{{.+}}
// CHECK: [[SZ:%.+]] = mul nuw i64 [[SZ1]], 4
// CHECK: [[SHAPE:%.+]] = ptrtoint i32* [[SHAPE_ADDR]] to i64
// CHECK: [[BASE_ADDR:%.+]] = getelementptr %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEP_ADDR]], i{{.+}} 2
// CHECK: [[ADDR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[BASE_ADDR]], i{{.+}} 0, i{{.+}} 0
// CHECK: store i64 [[SHAPE]], i64* [[ADDR]], align 8
// CHECK: [[SZ_ADDR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[BASE_ADDR]], i{{.+}} 0, i{{.+}} 1
// CHECK: store i64 [[SZ]], i64* [[SZ_ADDR]], align 8
// CHECK: [[FLAGS_ADDR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[BASE_ADDR]], i{{.+}} 0, i{{.+}} 2
// CHECK: store i8 1, i8* [[FLAGS_ADDR]], align 8
// CHECK: [[BASE_ADDR:%.+]] = getelementptr %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEP_ADDR]], i{{.+}} 1
// CHECK: [[DEP:%.+]] = bitcast %struct.kmp_depend_info* [[BASE_ADDR]] to i8*
// CHECK: store i8* [[DEP]], i8** [[TMAIN_A]], align 8
// CHECK: [[ARGC:%.+]] = load i8*, i8** [[ARGC_ADDR]], align 8
// CHECK: [[ARGC_BASE:%.+]] = bitcast i8* [[ARGC]] to %struct.kmp_depend_info*
// CHECK: [[ARGC_REF:%.+]] = getelementptr %struct.kmp_depend_info, %struct.kmp_depend_info* [[ARGC_BASE]], i{{.+}} -1
// CHECK: [[ARGC:%.+]] = bitcast %struct.kmp_depend_info* [[ARGC_REF]] to i8*
// CHECK: call void @__kmpc_free(i32 [[GTID]], i8* [[ARGC]], i8* null)
// CHECK: [[ARGC_ADDR_CAST:%.+]] = bitcast i8** [[ARGC_ADDR]] to %struct.kmp_depend_info**
// CHECK: [[ARGC_BASE:%.+]] = load %struct.kmp_depend_info*, %struct.kmp_depend_info** [[ARGC_ADDR_CAST]], align 8
// CHECK: [[NUMDEPS_BASE:%.+]] = getelementptr %struct.kmp_depend_info, %struct.kmp_depend_info* [[ARGC_BASE]], i64 -1
// CHECK: [[NUMDEPS_ADDR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[NUMDEPS_BASE]], i{{.+}} 0, i{{.+}} 0
// CHECK: [[NUMDEPS:%.+]] = load i64, i64* [[NUMDEPS_ADDR]], align 8
// CHECK: [[END:%.+]] = getelementptr %struct.kmp_depend_info, %struct.kmp_depend_info* [[ARGC_BASE]], i64 [[NUMDEPS]]
// CHECK: br label %[[BODY:.+]]
// CHECK: [[BODY]]:
// CHECK: [[EL:%.+]] = phi %struct.kmp_depend_info* [ [[ARGC_BASE]], %{{.+}} ], [ [[EL_NEXT:%.+]], %[[BODY]] ]
// CHECK: [[FLAG_BASE:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[EL]], i{{.+}} 0, i{{.+}} 2
// CHECK: store i8 3, i8* [[FLAG_BASE]], align 8
// CHECK: [[EL_NEXT]] = getelementptr %struct.kmp_depend_info, %struct.kmp_depend_info* [[EL]], i{{.+}} 1
// CHECK: [[IS_DONE:%.+]] = icmp eq %struct.kmp_depend_info* [[EL_NEXT]], [[END]]
// CHECK: br i1 [[IS_DONE]], label %[[DONE:.+]], label %[[BODY]]
// CHECK: [[DONE]]:

#endif
