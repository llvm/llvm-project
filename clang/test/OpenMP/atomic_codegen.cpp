// RUN: %clang_cc1 -no-opaque-pointers -verify -triple x86_64-apple-darwin10 -fopenmp -fexceptions -fcxx-exceptions -x c++ -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -verify -triple x86_64-apple-darwin10 -fopenmp -fexceptions -fcxx-exceptions -x c++ -emit-llvm -std=c++98 %s -o - | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -verify -triple x86_64-apple-darwin10 -fopenmp -fexceptions -fcxx-exceptions -x c++ -emit-llvm -std=c++11 %s -o - | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -verify -triple x86_64-apple-darwin10 -fopenmp -fexceptions -fcxx-exceptions -debug-info-kind=line-tables-only -x c++ -emit-llvm %s -o - | FileCheck %s --check-prefix=TERM_DEBUG

// RUN: %clang_cc1 -no-opaque-pointers -verify -triple x86_64-apple-darwin10 -fopenmp-simd -fexceptions -fcxx-exceptions -x c++ -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -no-opaque-pointers -verify -triple x86_64-apple-darwin10 -fopenmp-simd -fexceptions -fcxx-exceptions -x c++ -emit-llvm -std=c++98 %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -no-opaque-pointers -verify -triple x86_64-apple-darwin10 -fopenmp-simd -fexceptions -fcxx-exceptions -x c++ -emit-llvm -std=c++11 %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -no-opaque-pointers -verify -triple x86_64-apple-darwin10 -fopenmp-simd -fexceptions -fcxx-exceptions -debug-info-kind=line-tables-only -x c++ -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics

int a;
int b;

struct St {
  unsigned long field;
  St() {}
  ~St() {}
  int &get() { return a; }
};

// CHECK-LABEL: parallel_atomic_ewc
void parallel_atomic_ewc() {
  St s;
#pragma omp parallel
  {
      // CHECK: invoke void @_ZN2StC1Ev(%struct.St* {{[^,]*}} [[TEMP_ST_ADDR:%.+]])
      // CHECK: [[SCALAR_ADDR:%.+]] = invoke noundef nonnull align 4 dereferenceable(4) i32* @_ZN2St3getEv(%struct.St* {{[^,]*}} [[TEMP_ST_ADDR]])
      // CHECK: [[SCALAR_VAL:%.+]] = load atomic i32, i32* [[SCALAR_ADDR]] monotonic, align 4
      // CHECK: store i32 [[SCALAR_VAL]], i32* @b
      // CHECK98: invoke void @_ZN2StD1Ev(%struct.St* {{[^,]*}} [[TEMP_ST_ADDR]])
      // CHECK11: call void @_ZN2StD1Ev(%struct.St* {{[^,]*}} [[TEMP_ST_ADDR]])
#pragma omp atomic read
      b = St().get();
      // CHECK-DAG: invoke void @_ZN2StC1Ev(%struct.St* {{[^,]*}} [[TEMP_ST_ADDR:%.+]])
      // CHECK-DAG: [[SCALAR_ADDR:%.+]] = invoke noundef nonnull align 4 dereferenceable(4) i32* @_ZN2St3getEv(%struct.St* {{[^,]*}} [[TEMP_ST_ADDR]])
      // CHECK-DAG: [[B_VAL:%.+]] = load i32, i32* @b
      // CHECK: store atomic i32 [[B_VAL]], i32* [[SCALAR_ADDR]] monotonic, align 4
      // CHECK: {{invoke|call}} void @_ZN2StD1Ev(%struct.St* {{[^,]*}} [[TEMP_ST_ADDR]])
#pragma omp atomic write
      St().get() = b;
      // CHECK: invoke void @_ZN2StC1Ev(%struct.St* {{[^,]*}} [[TEMP_ST_ADDR:%.+]])
      // CHECK: [[SCALAR_ADDR:%.+]] = invoke noundef nonnull align 4 dereferenceable(4) i32* @_ZN2St3getEv(%struct.St* {{[^,]*}} [[TEMP_ST_ADDR]])
      // CHECK: [[B_VAL:%.+]] = load i32, i32* @b
      // CHECK: [[OLD_VAL:%.+]] = load atomic i32, i32* [[SCALAR_ADDR]] monotonic, align 4
      // CHECK: br label %[[OMP_UPDATE:.+]]
      // CHECK: [[OMP_UPDATE]]
      // CHECK: [[OLD_PHI_VAL:%.+]] = phi i32 [ [[OLD_VAL]], %{{.+}} ], [ [[NEW_OLD_VAL:%.+]], %[[OMP_UPDATE]] ]
      // CHECK: [[NEW_VAL:%.+]] = srem i32 [[OLD_PHI_VAL]], [[B_VAL]]
      // CHECK: store i32 [[NEW_VAL]], i32* [[TEMP:%.+]],
      // CHECK: [[NEW_VAL:%.+]] = load i32, i32* [[TEMP]],
      // CHECK: [[RES:%.+]] = cmpxchg i32* [[SCALAR_ADDR]], i32 [[OLD_PHI_VAL]], i32 [[NEW_VAL]] monotonic monotonic, align 4
      // CHECK: [[NEW_OLD_VAL]] = extractvalue { i32, i1 } [[RES]], 0
      // CHECK: [[COND:%.+]] = extractvalue { i32, i1 } [[RES]], 1
      // CHECK: br i1 [[COND]], label %[[OMP_DONE:.+]], label %[[OMP_UPDATE]]
      // CHECK: [[OMP_DONE]]
      // CHECK: {{invoke|call}} void @_ZN2StD1Ev(%struct.St* {{[^,]*}} [[TEMP_ST_ADDR]])
#pragma omp atomic
      St().get() %= b;
#pragma omp atomic hint(6)
      s.field++;
      // CHECK: invoke void @_ZN2StC1Ev(%struct.St* {{[^,]*}} [[TEMP_ST_ADDR:%.+]])
      // CHECK: [[SCALAR_ADDR:%.+]] = invoke noundef nonnull align 4 dereferenceable(4) i32* @_ZN2St3getEv(%struct.St* {{[^,]*}} [[TEMP_ST_ADDR]])
      // CHECK: [[B_VAL:%.+]] = load i32, i32* @b
      // CHECK: [[OLD_VAL:%.+]] = load atomic i32, i32* [[SCALAR_ADDR]] monotonic, align 4
      // CHECK: br label %[[OMP_UPDATE:.+]]
      // CHECK: [[OMP_UPDATE]]
      // CHECK: [[OLD_PHI_VAL:%.+]] = phi i32 [ [[OLD_VAL]], %{{.+}} ], [ [[NEW_OLD_VAL:%.+]], %[[OMP_UPDATE]] ]
      // CHECK: [[NEW_CALC_VAL:%.+]] = srem i32 [[OLD_PHI_VAL]], [[B_VAL]]
      // CHECK: store i32 [[NEW_CALC_VAL]], i32* [[TEMP:%.+]],
      // CHECK: [[NEW_VAL:%.+]] = load i32, i32* [[TEMP]],
      // CHECK: [[RES:%.+]] = cmpxchg i32* [[SCALAR_ADDR]], i32 [[OLD_PHI_VAL]], i32 [[NEW_VAL]] monotonic monotonic, align 4
      // CHECK: [[NEW_OLD_VAL]] = extractvalue { i32, i1 } [[RES]], 0
      // CHECK: [[COND:%.+]] = extractvalue { i32, i1 } [[RES]], 1
      // CHECK: br i1 [[COND]], label %[[OMP_DONE:.+]], label %[[OMP_UPDATE]]
      // CHECK: [[OMP_DONE]]
      // CHECK: store i32 [[NEW_CALC_VAL]], i32* @a,
      // CHECK: {{invoke|call}} void @_ZN2StD1Ev(%struct.St* {{[^,]*}} [[TEMP_ST_ADDR]])
#pragma omp atomic capture
      a = St().get() %= b;
    }
}

int &foo() { extern void mayThrow(); mayThrow(); return a; }

// TERM_DEBUG-LABEL: parallel_atomic
void parallel_atomic() {
#pragma omp parallel
  {
#pragma omp atomic read
    // TERM_DEBUG-NOT: __kmpc_global_thread_num
    // TERM_DEBUG:     invoke {{.*}}foo{{.*}}()
    // TERM_DEBUG:     unwind label %[[TERM_LPAD:.+]],
    // TERM_DEBUG:     load atomic i32, i32* @{{.+}} monotonic, align 4, !dbg [[READ_LOC:![0-9]+]]
    foo() = a;
#pragma omp atomic write
    // TERM_DEBUG-NOT: __kmpc_global_thread_num
    // TERM_DEBUG:     invoke {{.*}}foo{{.*}}()
    // TERM_DEBUG:     unwind label %[[TERM_LPAD:.+]],
    // TERM_DEBUG-NOT: __kmpc_global_thread_num
    // TERM_DEBUG:     store atomic i32 {{%.+}}, i32* @{{.+}} monotonic, align 4, !dbg [[WRITE_LOC:![0-9]+]]
    a = foo();
#pragma omp atomic update
    // TERM_DEBUG-NOT: __kmpc_global_thread_num
    // TERM_DEBUG:     invoke {{.*}}foo{{.*}}()
    // TERM_DEBUG:     unwind label %[[TERM_LPAD:.+]],
    // TERM_DEBUG-NOT: __kmpc_global_thread_num
    // TERM_DEBUG:     atomicrmw add i32* @{{.+}}, i32 %{{.+}} monotonic, align 4, !dbg [[UPDATE_LOC:![0-9]+]]
    a += foo();
#pragma omp atomic capture
    // TERM_DEBUG-NOT: __kmpc_global_thread_num
    // TERM_DEBUG:     invoke {{.*}}foo{{.*}}()
    // TERM_DEBUG:     unwind label %[[TERM_LPAD:.+]],
    // TERM_DEBUG-NOT: __kmpc_global_thread_num
    // TERM_DEBUG:     [[OLD_VAL:%.+]] = atomicrmw add i32* @{{.+}}, i32 %{{.+}} monotonic, align 4, !dbg [[CAPTURE_LOC:![0-9]+]]
    // TERM_DEBUG:     store i32 [[OLD_VAL]], i32* @b,
    {b = a; a += foo(); }
  }
  // TERM_DEBUG:     [[TERM_LPAD]]
  // TERM_DEBUG:     call void @__clang_call_terminate
  // TERM_DEBUG:     unreachable
}
// TERM_DEBUG-DAG: [[READ_LOC]] = !DILocation(line: [[@LINE-28]],
// TERM_DEBUG-DAG: [[WRITE_LOC]] = !DILocation(line: [[@LINE-22]],
// TERM_DEBUG-DAG: [[UPDATE_LOC]] = !DILocation(line: [[@LINE-16]],
// TERM_DEBUG-DAG: [[CAPTURE_LOC]] = !DILocation(line: [[@LINE-9]],
