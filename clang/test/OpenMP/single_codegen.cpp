// RUN: %clang_cc1 -verify -fopenmp -fnoopenmp-use-tls -x c++ -std=c++11 -triple x86_64-unknown-unknown -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fnoopenmp-use-tls -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fnoopenmp-use-tls -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -std=c++11 -fopenmp -fnoopenmp-use-tls -fexceptions -fcxx-exceptions -debug-info-kind=line-tables-only -x c++ -emit-llvm %s -o - | FileCheck %s --check-prefix=TERM_DEBUG
// RUN: %clang_cc1 -verify -fopenmp -fnoopenmp-use-tls -x c++ -std=c++11 -DARRAY -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck -check-prefix=ARRAY %s

// RUN: %clang_cc1 -verify -fopenmp-simd -fnoopenmp-use-tls -x c++ -std=c++11 -triple x86_64-unknown-unknown -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -fnoopenmp-use-tls -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fnoopenmp-use-tls -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -std=c++11 -fopenmp-simd -fnoopenmp-use-tls -fexceptions -fcxx-exceptions -debug-info-kind=line-tables-only -x c++ -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -fnoopenmp-use-tls -x c++ -std=c++11 -DARRAY -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics
#ifndef ARRAY
#ifndef HEADER
#define HEADER

class TestClass {
public:
  int a;
  TestClass() : a(0) {}
  TestClass(const TestClass &C) : a(C.a) {}
  TestClass &operator=(const TestClass &) { return *this;}
  ~TestClass(){};
};

// CHECK-DAG:   [[TEST_CLASS_TY:%.+]] = type { i{{[0-9]+}} }
// CHECK-DAG:   [[SST_TY:%.+]] = type { double }
// CHECK-DAG:   [[SS_TY:%.+]] = type { i32, i8, i32* }
// CHECK-DAG:   [[IDENT_T_TY:%.+]] = type { i32, i32, i32, i32, i8* }
// CHECK:       [[IMPLICIT_BARRIER_SINGLE_LOC:@.+]] = private unnamed_addr constant %{{.+}} { i32 0, i32 322, i32 0, i32 0, i8*

// CHECK:       define void [[FOO:@.+]]()

TestClass tc;
TestClass tc2[2];
#pragma omp threadprivate(tc, tc2)

void foo() {}

struct SS {
  int a;
  int b : 4;
  int &c;
  SS(int &d) : a(0), b(0), c(d) {
#pragma omp parallel firstprivate(a, b, c)
#pragma omp single copyprivate(a, this->b, (this)->c)
    [&]() {
      ++this->a, --b, (this)->c /= 1;
#pragma omp parallel firstprivate(a, b, c)
#pragma omp single copyprivate(a, this->b, (this)->c)
      ++(this)->a, --b, this->c /= 1;
    }();
  }
};

template<typename T>
struct SST {
  T a;
  SST() : a(T()) {
#pragma omp parallel firstprivate(a)
#pragma omp single copyprivate(this->a)
    [&]() {
      [&]() {
        ++this->a;
#pragma omp parallel firstprivate(a)
#pragma omp single copyprivate((this)->a)
        ++(this)->a;
      }();
    }();
  }
};

// CHECK-LABEL: @main
// TERM_DEBUG-LABEL: @main
int main() {
  // CHECK-DAG: [[A_ADDR:%.+]] = alloca i8
  // CHECK-DAG: [[A2_ADDR:%.+]] = alloca [2 x i8]
  // CHECK-DAG: [[C_ADDR:%.+]] = alloca [[TEST_CLASS_TY]]
  char a;
  char a2[2];
  TestClass &c = tc;
  SST<double> sst;
  SS ss(c.a);

// CHECK:       [[GTID:%.+]] = call i32 @__kmpc_global_thread_num([[IDENT_T_TY]]* [[DEFAULT_LOC:@.+]])
// CHECK-DAG:   [[DID_IT:%.+]] = alloca i32,
// CHECK-DAG:   [[COPY_LIST:%.+]] = alloca [5 x i8*],

// CHECK:       [[RES:%.+]] = call i32 @__kmpc_single([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]])
// CHECK-NEXT:  [[IS_SINGLE:%.+]] = icmp ne i32 [[RES]], 0
// CHECK-NEXT:  br i1 [[IS_SINGLE]], label {{%?}}[[THEN:.+]], label {{%?}}[[EXIT:.+]]
// CHECK:       [[THEN]]
// CHECK-NEXT:  store i8 2, i8* [[A_ADDR]]
// CHECK-NEXT:  call void @__kmpc_end_single([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]])
// CHECK-NEXT:  br label {{%?}}[[EXIT]]
// CHECK:       [[EXIT]]
// CHECK-NOT:   call {{.+}} @__kmpc_cancel_barrier
#pragma omp single nowait
  a = 2;
// CHECK:       [[RES:%.+]] = call i32 @__kmpc_single([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]])
// CHECK-NEXT:  [[IS_SINGLE:%.+]] = icmp ne i32 [[RES]], 0
// CHECK-NEXT:  br i1 [[IS_SINGLE]], label {{%?}}[[THEN:.+]], label {{%?}}[[EXIT:.+]]
// CHECK:       [[THEN]]
// CHECK-NEXT:  store i8 2, i8* [[A_ADDR]]
// CHECK-NEXT:  call void @__kmpc_end_single([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]])
// CHECK-NEXT:  br label {{%?}}[[EXIT]]
// CHECK:       [[EXIT]]
// CHECK:       call{{.*}} @__kmpc_barrier([[IDENT_T_TY]]* [[IMPLICIT_BARRIER_SINGLE_LOC]], i32 [[GTID]])
#pragma omp single
  a = 2;
// CHECK:       store i32 0, i32* [[DID_IT]]
// CHECK:       [[RES:%.+]] = call i32 @__kmpc_single([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]])
// CHECK-NEXT:  [[IS_SINGLE:%.+]] = icmp ne i32 [[RES]], 0
// CHECK-NEXT:  br i1 [[IS_SINGLE]], label {{%?}}[[THEN:.+]], label {{%?}}[[EXIT:.+]]
// CHECK:       [[THEN]]
// CHECK-NEXT:  invoke void [[FOO]]()
// CHECK:       to label {{%?}}[[CONT:.+]] unwind
// CHECK:       [[CONT]]
// CHECK:       call void @__kmpc_end_single([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]])
// CHECK:       store i32 1, i32* [[DID_IT]]
// CHECK-NEXT:  br label {{%?}}[[EXIT]]
// CHECK:       [[EXIT]]
// CHECK:       [[A_PTR_REF:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[COPY_LIST]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK:       store i8* [[A_ADDR]], i8** [[A_PTR_REF]],
// CHECK:       [[C_PTR_REF:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[COPY_LIST]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
// CHECK:       store i8* {{.+}}, i8** [[C_PTR_REF]],
// CHECK:       [[TC_PTR_REF:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[COPY_LIST]], i{{[0-9]+}} 0, i{{[0-9]+}} 2
// CHECK:       [[TC_THREADPRIVATE_ADDR_VOID_PTR:%.+]] = call{{.*}} i8* @__kmpc_threadprivate_cached
// CHECK:       [[TC_THREADPRIVATE_ADDR:%.+]] = bitcast i8* [[TC_THREADPRIVATE_ADDR_VOID_PTR]] to [[TEST_CLASS_TY]]*
// CHECK:       [[TC_PTR_REF_VOID_PTR:%.+]] = bitcast [[TEST_CLASS_TY]]* [[TC_THREADPRIVATE_ADDR]] to i8*
// CHECK:       store i8* [[TC_PTR_REF_VOID_PTR]], i8** [[TC_PTR_REF]],
// CHECK:       [[A2_PTR_REF:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[COPY_LIST]], i{{[0-9]+}} 0, i{{[0-9]+}} 3
// CHECK:       [[BITCAST:%.+]] = bitcast [2 x i8]* [[A2_ADDR]] to i8*
// CHECK:       store i8* [[BITCAST]], i8** [[A2_PTR_REF]],
// CHECK:       [[TC2_PTR_REF:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[COPY_LIST]], i{{[0-9]+}} 0, i{{[0-9]+}} 4
// CHECK:       [[TC2_THREADPRIVATE_ADDR_VOID_PTR:%.+]] = call{{.*}} i8* @__kmpc_threadprivate_cached
// CHECK:       [[TC2_THREADPRIVATE_ADDR:%.+]] = bitcast i8* [[TC2_THREADPRIVATE_ADDR_VOID_PTR]] to [2 x [[TEST_CLASS_TY]]]*
// CHECK:       [[TC2_PTR_REF_VOID_PTR:%.+]] = bitcast [2 x [[TEST_CLASS_TY]]]* [[TC2_THREADPRIVATE_ADDR]] to i8*
// CHECK:       store i8* [[TC2_PTR_REF_VOID_PTR]], i8** [[TC2_PTR_REF]],
// CHECK:       [[COPY_LIST_VOID_PTR:%.+]] = bitcast [5 x i8*]* [[COPY_LIST]] to i8*
// CHECK:       [[DID_IT_VAL:%.+]] = load i32, i32* [[DID_IT]],
// CHECK:       call void @__kmpc_copyprivate([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]], i64 40, i8* [[COPY_LIST_VOID_PTR]], void (i8*, i8*)* [[COPY_FUNC:@.+]], i32 [[DID_IT_VAL]])
// CHECK-NOT:   call {{.+}} @__kmpc_cancel_barrier
#pragma omp single copyprivate(a, c, tc, a2, tc2)
  foo();
// CHECK-NOT:   call i32 @__kmpc_single
// CHECK-NOT:   call void @__kmpc_end_single
  return a;
}

// CHECK: void [[COPY_FUNC]](i8*, i8*)
// CHECK: store i8* %0, i8** [[DST_ADDR_REF:%.+]],
// CHECK: store i8* %1, i8** [[SRC_ADDR_REF:%.+]],
// CHECK: [[DST_ADDR_VOID_PTR:%.+]] = load i8*, i8** [[DST_ADDR_REF]],
// CHECK: [[DST_ADDR:%.+]] = bitcast i8* [[DST_ADDR_VOID_PTR]] to [5 x i8*]*
// CHECK: [[SRC_ADDR_VOID_PTR:%.+]] = load i8*, i8** [[SRC_ADDR_REF]],
// CHECK: [[SRC_ADDR:%.+]] = bitcast i8* [[SRC_ADDR_VOID_PTR]] to [5 x i8*]*
// CHECK: [[DST_A_ADDR_REF:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[DST_ADDR]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK: [[DST_A_ADDR:%.+]] = load i8*, i8** [[DST_A_ADDR_REF]],
// CHECK: [[SRC_A_ADDR_REF:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[SRC_ADDR]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK: [[SRC_A_ADDR:%.+]] = load i8*, i8** [[SRC_A_ADDR_REF]],
// CHECK: [[SRC_A_VAL:%.+]] = load i8, i8* [[SRC_A_ADDR]],
// CHECK: store i8 [[SRC_A_VAL]], i8* [[DST_A_ADDR]],
// CHECK: [[DST_C_ADDR_REF:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[DST_ADDR]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
// CHECK: [[DST_C_ADDR_VOID_PTR:%.+]] = load i8*, i8** [[DST_C_ADDR_REF]],
// CHECK: [[DST_C_ADDR:%.+]] = bitcast i8* [[DST_C_ADDR_VOID_PTR]] to [[TEST_CLASS_TY]]*
// CHECK: [[SRC_C_ADDR_REF:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[SRC_ADDR]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
// CHECK: [[SRC_C_ADDR_VOID_PTR:%.+]] = load i8*, i8** [[SRC_C_ADDR_REF]],
// CHECK: [[SRC_C_ADDR:%.+]] = bitcast i8* [[SRC_C_ADDR_VOID_PTR]] to [[TEST_CLASS_TY]]*
// CHECK: call{{.*}} [[TEST_CLASS_TY_ASSIGN:@.+]]([[TEST_CLASS_TY]]* [[DST_C_ADDR]], [[TEST_CLASS_TY]]* {{.*}}[[SRC_C_ADDR]])
// CHECK: [[DST_TC_ADDR_REF:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[DST_ADDR]], i{{[0-9]+}} 0, i{{[0-9]+}} 2
// CHECK: [[DST_TC_ADDR_VOID_PTR:%.+]] = load i8*, i8** [[DST_TC_ADDR_REF]],
// CHECK: [[DST_TC_ADDR:%.+]] = bitcast i8* [[DST_TC_ADDR_VOID_PTR]] to [[TEST_CLASS_TY]]*
// CHECK: [[SRC_TC_ADDR_REF:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[SRC_ADDR]], i{{[0-9]+}} 0, i{{[0-9]+}} 2
// CHECK: [[SRC_TC_ADDR_VOID_PTR:%.+]] = load i8*, i8** [[SRC_TC_ADDR_REF]],
// CHECK: [[SRC_TC_ADDR:%.+]] = bitcast i8* [[SRC_TC_ADDR_VOID_PTR]] to [[TEST_CLASS_TY]]*
// CHECK: call{{.*}} [[TEST_CLASS_TY_ASSIGN]]([[TEST_CLASS_TY]]* [[DST_TC_ADDR]], [[TEST_CLASS_TY]]* {{.*}}[[SRC_TC_ADDR]])
// CHECK: [[DST_A2_ADDR_REF:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[DST_ADDR]], i{{[0-9]+}} 0, i{{[0-9]+}} 3
// CHECK: [[DST_A2_ADDR:%.+]] = load i8*, i8** [[DST_A2_ADDR_REF]],
// CHECK: [[SRC_A2_ADDR_REF:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[SRC_ADDR]], i{{[0-9]+}} 0, i{{[0-9]+}} 3
// CHECK: [[SRC_A2_ADDR:%.+]] = load i8*, i8** [[SRC_A2_ADDR_REF]],
// CHECK: call void @llvm.memcpy.{{.+}}(i8* [[DST_A2_ADDR]], i8* [[SRC_A2_ADDR]], i{{[0-9]+}} 2, i{{[0-9]+}} 1, i1 false)
// CHECK: [[DST_TC2_ADDR_REF:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[DST_ADDR]], i{{[0-9]+}} 0, i{{[0-9]+}} 4
// CHECK: [[DST_TC2_ADDR_VOID_PTR:%.+]] = load i8*, i8** [[DST_TC2_ADDR_REF]],
// CHECK: [[DST_TC2_ADDR:%.+]] = bitcast i8* [[DST_TC2_ADDR_VOID_PTR]] to [[TEST_CLASS_TY]]*
// CHECK: [[SRC_TC2_ADDR_REF:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[SRC_ADDR]], i{{[0-9]+}} 0, i{{[0-9]+}} 4
// CHECK: [[SRC_TC2_ADDR_VOID_PTR:%.+]] = load i8*, i8** [[SRC_TC2_ADDR_REF]],
// CHECK: [[SRC_TC2_ADDR:%.+]] = bitcast i8* [[SRC_TC2_ADDR_VOID_PTR]] to [[TEST_CLASS_TY]]*
// CHECK: br i1
// CHECK: call{{.*}} [[TEST_CLASS_TY_ASSIGN]]([[TEST_CLASS_TY]]* %{{.+}}, [[TEST_CLASS_TY]]* {{.*}})
// CHECK: br i1
// CHECK: ret void

// CHECK-LABEL:      parallel_single
// TERM_DEBUG-LABEL: parallel_single
void parallel_single() {
#pragma omp parallel
#pragma omp single
  // TERM_DEBUG-NOT: __kmpc_global_thread_num
  // TERM_DEBUG:     call i32 @__kmpc_single({{.+}}), !dbg [[DBG_LOC_START:![0-9]+]]
  // TERM_DEBUG:     invoke void {{.*}}foo{{.*}}()
  // TERM_DEBUG:     unwind label %[[TERM_LPAD:.+]],
  // TERM_DEBUG-NOT: __kmpc_global_thread_num
  // TERM_DEBUG:     call void @__kmpc_end_single({{.+}}), !dbg [[DBG_LOC_END:![0-9]+]]
  // TERM_DEBUG:     [[TERM_LPAD]]
  // TERM_DEBUG:     call void @__clang_call_terminate
  // TERM_DEBUG:     unreachable
  foo();
}
// TERM_DEBUG-DAG: [[DBG_LOC_START]] = !DILocation(line: [[@LINE-12]],
// TERM_DEBUG-DAG: [[DBG_LOC_END]] = !DILocation(line: [[@LINE-3]],
#endif
#else
// ARRAY-LABEL: array_func
struct St {
  int a, b;
  St() : a(0), b(0) {}
  St &operator=(const St &) { return *this; };
  ~St() {}
};

void array_func(int n, int a[n], St s[2]) {
// ARRAY: call void @__kmpc_copyprivate(%ident_t* @{{.+}}, i32 %{{.+}}, i64 16, i8* %{{.+}}, void (i8*, i8*)* [[CPY:@.+]], i32 %{{.+}})
#pragma omp single copyprivate(a, s)
  ;
}
// ARRAY: define internal void [[CPY]]
// ARRAY: store i32* %{{.+}}, i32** %{{.+}},
// ARRAY: store %struct.St* %{{.+}}, %struct.St** %{{.+}},
#endif

// CHECK-LABEL:@_ZN2SSC2ERi(
// CHECK: call void ([[IDENT_T_TY]]*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call([[IDENT_T_TY]]* @{{.+}}, i32 4, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, [[SS_TY]]*, i64, i64, i64)* [[SS_MICROTASK:@.+]] to void
// CHECK-NEXT: ret void

// CHECK: define internal void [[SS_MICROTASK]](i32* {{[^,]+}}, i32* {{[^,]+}}, [[SS_TY]]* {{.+}}, i64 {{.+}}, i64 {{.+}}, i64 {{.+}})
// Private a
// CHECK: alloca i64,
// Private b
// CHECK: alloca i64,
// Private c
// CHECK: alloca i64,
// CHECK: alloca i32*,
// CHECK: alloca i32*,
// CHECK: alloca i32*,
// CHECK: alloca i32*,
// CHECK: [[DID_IT:%.+]] = alloca i32,
// CHECK: bitcast i64* %{{.+}} to i32*
// CHECK: bitcast i64* %{{.+}} to i32*
// CHECK: bitcast i64* %{{.+}} to i32*
// CHECK: store i32 0, i32* [[DID_IT]],
// CHECK: [[RES:%.+]] = call i32 @__kmpc_single([[IDENT_T_TY]]* @{{.+}}, i32 %{{.+}})
// CHECK-NEXT: icmp ne i32 [[RES]], 0
// CHECK-NEXT: br i1

// CHECK: getelementptr inbounds [[CAP_TY:%.+]], [[CAP_TY]]* [[CAP:%.+]], i32 0, i32 0
// CHECK: getelementptr inbounds [[CAP_TY]], [[CAP_TY]]* [[CAP]], i32 0, i32 1
// CHECK-NEXT: load i32*, i32** %
// CHECK-NEXT: store i32* %
// CHECK-NEXT: getelementptr inbounds [[CAP_TY]], [[CAP_TY]]* [[CAP]], i32 0, i32 2
// CHECK-NEXT: store i32* %
// CHECK-NEXT: getelementptr inbounds [[CAP_TY]], [[CAP_TY]]* [[CAP]], i32 0, i32 3
// CHECK-NEXT: load i32*, i32** %
// CHECK-NEXT: store i32* %
// CHECK-LABEL: invoke void @_ZZN2SSC1ERiENKUlvE_clEv(
// CHECK-SAME: [[CAP_TY]]* [[CAP]])

// CHECK: call void @__kmpc_end_single([[IDENT_T_TY]]* @{{.+}}, i32 %{{.+}})
// CHECK: store i32 1, i32* [[DID_IT]],
// CHECK: br label

// CHECK: call void @__kmpc_end_single(%{{.+}}* @{{.+}}, i32 %{{.+}})
// CHECK: br label

// CHECK: getelementptr inbounds [3 x i8*], [3 x i8*]* [[LIST:%.+]], i64 0, i64 0
// CHECK: load i32*, i32** %
// CHECK-NEXT: bitcast i32* %
// CHECK-NEXT: store i8* %
// CHECK: getelementptr inbounds [3 x i8*], [3 x i8*]* [[LIST]], i64 0, i64 1
// CHECK-NEXT: bitcast i32* %
// CHECK-NEXT: store i8* %
// CHECK: getelementptr inbounds [3 x i8*], [3 x i8*]* [[LIST]], i64 0, i64 2
// CHECK: load i32*, i32** %
// CHECK-NEXT: bitcast i32* %
// CHECK-NEXT: store i8* %
// CHECK-NEXT: bitcast [3 x i8*]* [[LIST]] to i8*
// CHECK-NEXT: load i32, i32* [[DID_IT]],
// CHECK-NEXT: call void @__kmpc_copyprivate([[IDENT_T_TY]]* @{{.+}}, i32 %{{.+}}, i64 24, i8* %{{.+}}, void (i8*, i8*)* [[COPY_FUNC:@[^,]+]], i32 %{{.+}})
// CHECK-NEXT: ret void

// CHECK-LABEL: @_ZZN2SSC1ERiENKUlvE_clEv(
// CHECK: getelementptr inbounds [[CAP_TY]], [[CAP_TY]]* [[CAP:%.+]], i32 0, i32 1
// CHECK-NEXT: load i32*, i32** %
// CHECK-NEXT: load i32, i32* %
// CHECK-NEXT: add nsw i32 %{{.+}}, 1
// CHECK-NEXT: store i32 %
// CHECK-NEXT: getelementptr inbounds [[CAP_TY]], [[CAP_TY]]* [[CAP]], i32 0, i32 2
// CHECK-NEXT: load i32*, i32** %
// CHECK-NEXT: load i32, i32* %
// CHECK-NEXT: add nsw i32 %{{.+}}, -1
// CHECK-NEXT: store i32 %
// CHECK-NEXT: getelementptr inbounds [[CAP_TY]], [[CAP_TY]]* [[CAP]], i32 0, i32 3
// CHECK-NEXT: load i32*, i32** %
// CHECK-NEXT: load i32, i32* %
// CHECK-NEXT: sdiv i32 %{{.+}}, 1
// CHECK-NEXT: store i32 %
// CHECK-NEXT: getelementptr inbounds [[CAP_TY]], [[CAP_TY]]* [[CAP]], i32 0, i32 1
// CHECK-NEXT: load i32*, i32** %
// CHECK-NEXT: load i32, i32* %
// CHECK-NEXT: bitcast i64* %
// CHECK-NEXT: store i32 %{{.+}}, i32* %
// CHECK-NEXT: load i64, i64* %
// CHECK-NEXT: getelementptr inbounds [[CAP_TY]], [[CAP_TY]]* [[CAP]], i32 0, i32 2
// CHECK-NEXT: load i32*, i32** %
// CHECK-NEXT: load i32, i32* %
// CHECK-NEXT: bitcast i64* %
// CHECK-NEXT: store i32 %{{.+}}, i32* %
// CHECK-NEXT: load i64, i64* %
// CHECK-NEXT: getelementptr inbounds [[CAP_TY]], [[CAP_TY]]* [[CAP]], i32 0, i32 3
// CHECK-NEXT: load i32*, i32** %
// CHECK-NEXT: load i32, i32* %
// CHECK-NEXT: bitcast i64* %
// CHECK-NEXT: store i32 %{{.+}}, i32* %
// CHECK-NEXT: load i64, i64* %
// CHECK-NEXT: call void ([[IDENT_T_TY]]*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call([[IDENT_T_TY]]* @{{.+}}, i32 4, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, [[SS_TY]]*, i64, i64, i64)* [[SS_MICROTASK1:@.+]] to void
// CHECK-NEXT: ret void

// CHECK: define internal void [[COPY_FUNC]](i8*, i8*)
// CHECK: ret void

// CHECK: define internal void [[SS_MICROTASK1]](i32* {{[^,]+}}, i32* {{[^,]+}}, [[SS_TY]]* {{.+}}, i64 {{.+}}, i64 {{.+}}, i64 {{.+}})
// Private a
// CHECK: alloca i64,
// Private b
// CHECK: alloca i64,
// Private c
// CHECK: alloca i64,
// CHECK: alloca i32*,
// CHECK: alloca i32*,
// CHECK: alloca i32*,
// CHECK: alloca i32*,
// CHECK: [[DID_IT:%.+]] = alloca i32,
// CHECK: bitcast i64* %{{.+}} to i32*
// CHECK: bitcast i64* %{{.+}} to i32*
// CHECK: bitcast i64* %{{.+}} to i32*
// CHECK: [[RES:%.+]] = call i32 @__kmpc_single([[IDENT_T_TY]]* @{{.+}}, i32 %{{.+}})
// CHECK-NEXT: icmp ne i32 [[RES]], 0
// CHECK-NEXT: br i1

// CHECK-NOT: getelementptr inbounds
// CHECK: load i32*, i32** %
// CHECK-NEXT: load i32, i32* %
// CHECK-NEXT: add nsw i32 %{{.+}}, 1
// CHECK-NEXT: store i32 %
// CHECK-NOT: getelementptr inbounds
// CHECK: load i32, i32* %
// CHECK-NEXT: add nsw i32 %{{.+}}, -1
// CHECK-NEXT: store i32 %
// CHECK-NOT: getelementptr inbounds
// CHECK: load i32*, i32** %
// CHECK-NEXT: load i32, i32* %
// CHECK-NEXT: sdiv i32 %{{.+}}, 1
// CHECK-NEXT: store i32 %
// CHECK-NEXT: call void @__kmpc_end_single([[IDENT_T_TY]]* @{{.+}}, i32 %{{.+}})
// CHECK-NEXT: store i32 1, i32* [[DID_IT]],
// CHECK-NEXT: br label

// CHECK: getelementptr inbounds [3 x i8*], [3 x i8*]* [[LIST:%.+]], i64 0, i64 0
// CHECK: load i32*, i32** %
// CHECK-NEXT: bitcast i32* %
// CHECK-NEXT: store i8* %
// CHECK: getelementptr inbounds [3 x i8*], [3 x i8*]* [[LIST]], i64 0, i64 1
// CHECK-NEXT: bitcast i32* %
// CHECK-NEXT: store i8* %
// CHECK: getelementptr inbounds [3 x i8*], [3 x i8*]* [[LIST]], i64 0, i64 2
// CHECK: load i32*, i32** %
// CHECK-NEXT: bitcast i32* %
// CHECK-NEXT: store i8* %
// CHECK-NEXT: bitcast [3 x i8*]* [[LIST]] to i8*
// CHECK-NEXT: load i32, i32* [[DID_IT]],
// CHECK-NEXT: call void @__kmpc_copyprivate([[IDENT_T_TY]]* @{{.+}}, i32 %{{.+}}, i64 24, i8* %{{.+}}, void (i8*, i8*)* [[COPY_FUNC:@[^,]+]], i32 %{{.+}})
// CHECK-NEXT:  ret void

// CHECK: define internal void [[COPY_FUNC]](i8*, i8*)
// CHECK: ret void

// CHECK-LABEL: @_ZN3SSTIdEC2Ev
// CHECK: getelementptr inbounds [[SST_TY]], [[SST_TY]]* %{{.+}}, i32 0, i32 0
// CHECK-NEXT: store double 0.000000e+00, double* %
// CHECK-NEXT: getelementptr inbounds [[SST_TY]], [[SST_TY]]* %{{.+}}, i32 0, i32 0
// CHECK-NEXT: store double* %{{.+}}, double** %
// CHECK-NEXT: load double*, double** %
// CHECK-NEXT: load double, double* %
// CHECK-NEXT: bitcast i64* %{{.+}} to double*
// CHECK-NEXT: store double %{{.+}}, double* %
// CHECK-NEXT: load i64, i64* %
// CHECK-NEXT: call void ([[IDENT_T_TY]]*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call([[IDENT_T_TY]]* @{{.+}}, i32 2, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, [[SST_TY]]*, i64)* [[SST_MICROTASK:@.+]] to void
// CHECK-NEXT: ret void

// CHECK: define internal void [[SST_MICROTASK]](i32* {{[^,]+}}, i32* {{[^,]+}}, [[SST_TY]]* {{.+}}, i64 {{.+}})
// CHECK: [[RES:%.+]] = call i32 @__kmpc_single([[IDENT_T_TY]]* @{{.+}}, i32 %{{.+}})
// CHECK-NEXT: icmp ne i32 [[RES]], 0
// CHECK-NEXT: br i1

// CHECK: getelementptr inbounds %{{.+}}, %{{.+}}* %{{.+}}, i32 0, i32 1
// CHECK-NEXT: load double*, double** %
// CHECK-NEXT: store double* %
// CHECK-LABEL: invoke void @_ZZN3SSTIdEC1EvENKUlvE_clEv(

// CHECK: call void @__kmpc_end_single([[IDENT_T_TY]]* @{{.+}}, i32 %{{.+}})
// CHECK-NEXT: store i32 1, i32* [[DID_IT]],
// CHECK-NEXT: br label

// CHECK: call void @__kmpc_end_single([[IDENT_T_TY]]* @{{.+}}, i32 %{{.+}})
// CHECK-NEXT: br label

// CHECK: getelementptr inbounds [1 x i8*], [1 x i8*]* [[LIST:%.+]], i64 0, i64 0
// CHECK: load double*, double** %
// CHECK-NEXT: bitcast double* %
// CHECK-NEXT: store i8* %
// CHECK-NEXT: bitcast [1 x i8*]* [[LIST]] to i8*
// CHECK-NEXT: load i32, i32* [[DID_IT]],
// CHECK-NEXT: call void @__kmpc_copyprivate([[IDENT_T_TY]]* @{{.+}}, i32 %{{.+}}, i64 8, i8* %{{.+}}, void (i8*, i8*)* [[COPY_FUNC:@[^,]+]], i32 %{{.+}})
// CHECK-NEXT:  ret void

// CHECK-LABEL: @_ZZN3SSTIdEC1EvENKUlvE_clEv(
// CHECK: getelementptr inbounds %{{.+}}, %{{.+}}* %{{.+}}, i32 0, i32 1
// CHECK-NEXT: getelementptr inbounds %{{.+}}, %{{.+}}* %{{.+}}, i32 0, i32 1
// CHECK-NEXT: load double*, double** %
// CHECK-NEXT: store double* %
// CHECK-LABEL: call void @_ZZZN3SSTIdEC1EvENKUlvE_clEvENKUlvE_clEv(
// CHECK-NEXT: ret void

// CHECK: define internal void [[COPY_FUNC]](i8*, i8*)
// CHECK: ret void

// CHECK-LABEL: @_ZZZN3SSTIdEC1EvENKUlvE_clEvENKUlvE_clEv(
