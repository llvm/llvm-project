// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp -fopenmp-version=50 -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-version=50 -x c++ -triple x86_64-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-version=50 -x c++ -triple x86_64-unknown-unknown -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp-simd -fopenmp-version=50 -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -fopenmp-version=50 -x c++ -triple x86_64-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -fopenmp-version=50 -x c++ -triple x86_64-unknown-unknown -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
//
// expected-no-diagnostics
#ifndef HEADER
#define HEADER
void foo();
void bar();

// CHECK-LABEL: baz
void baz() {
  int a = 0;

  // CHECK: store i32 0, i32* [[A_ADDR:%.+]],
  // CHECK: store i32 0, i32* [[OMP_CNT:%.+]],
  // CHECK: br label %[[OMP_HEADER:.+]]

  // CHECK: [[OMP_HEADER]]:
  // CHECK: [[CNT_VAL:%.+]] = load i32, i32* [[OMP_CNT]],
  // CHECK: [[CMP:%.+]] = icmp slt i32 [[CNT_VAL]], 10
  // CHECK: br i1 [[CMP]], label %[[OMP_BODY:.+]], label %[[OMP_END:.+]]
#pragma omp simd reduction(inscan, + : a)
  for (int i = 0; i < 10; ++i) {
    // CHECK: [[OMP_BODY]]:

    // i = OMP_CNT*1 + 0;
    // CHECK: [[CNT_VAL:%.+]] = load i32, i32* [[OMP_CNT]],
    // CHECK: [[MUL:%.+]] = mul nsw i32 [[CNT_VAL]], 1
    // CHECK: [[ADD:%.+]] = add nsw i32 0, [[MUL]]
    // CHECK: store i32 [[ADD]], i32* [[I_ADDR:%.+]],

    // A_PRIV = 0;
    // CHECK: store i32 0, i32* [[A_PRIV_ADDR:%.+]],

    // goto DISPATCH;
    // CHECK: br label %[[DISPATCH:[^,]+]]

    // INPUT_PHASE:
    // foo();
    // goto REDUCE;
    // CHECK: [[INPUT_PHASE:.+]]:
    // CHECK: call void @{{.*}}foo{{.*}}()
    // CHECK: br label %[[REDUCE:[^,]+]]
    foo();

    // DISPATCH:
    // goto INPUT_PHASE;
    // CHECK: [[DISPATCH]]:
    // CHECK: br label %[[INPUT_PHASE]]

    // REDUCE:
    // A = A_PRIV + A;
    // A_PRIV = A;
    // goto SCAN_PHASE;
    // CHECK: [[REDUCE]]:
    // CHECK: [[A:%.+]] = load i32, i32* [[A_ADDR]],
    // CHECK: [[A_PRIV:%.+]] = load i32, i32* [[A_PRIV_ADDR]],
    // CHECK: [[SUM:%.+]] = add nsw i32 [[A]], [[A_PRIV]]
    // CHECK: store i32 [[SUM]], i32* [[A_ADDR]],
    // CHECK: [[A:%.+]] = load i32, i32* [[A_ADDR]],
    // CHECK: store i32 [[A]], i32* [[A_PRIV_ADDR]],
    // CHECK: br label %[[SCAN_PHASE:[^,]+]]
#pragma omp scan inclusive(a)

    // SCAN_PHASE:
    // bar();
    // goto CONTINUE;
    // CHECK: [[SCAN_PHASE]]:
    // CHECK: call void @{{.*}}bar{{.*}}()
    // CHECK: br label %[[CONTINUE:[^,]+]]
    bar();

    // CHECK: [[CONTINUE]]:
    // CHECK: br label %[[INC_BLOCK:[^,]+]]

    // ++OMP_CNT;
    // CHECK: [[INC_BLOCK]]:
    // CHECK: [[CNT:%.+]] = load i32, i32* [[OMP_CNT]],
    // CHECK: [[INC:%.+]] = add nsw i32 [[CNT]], 1
    // CHECK: store i32 [[INC]], i32* [[OMP_CNT]],
    // CHECK: br label %[[OMP_HEADER]]
  }
  // CHECK: [[OMP_END]]:
}

struct S {
  int a;
  S() {}
  ~S() {}
  S& operator+(const S&);
  S& operator=(const S&);
};

// CHECK-LABEL: xyz
void xyz() {
  S s[2];

  // CHECK: [[S_BEGIN:%.+]] = getelementptr inbounds [2 x %struct.S], [2 x %struct.S]* [[S_ADDR:%.+]], i{{.+}} 0, i{{.+}} 0
  // CHECK: [[S_END:%.+]] = getelementptr {{.*}}%struct.S, %struct.S* [[S_BEGIN]], i{{.+}} 2
  // CHECK: br label %[[ARRAY_INIT:.+]]
  // CHECK: [[ARRAY_INIT]]:
  // CHECK: [[S_CUR:%.+]] = phi %struct.S* [ [[S_BEGIN]], %{{.+}} ], [ [[S_NEXT:%.+]], %[[ARRAY_INIT]] ]
  // CHECK: call void [[CONSTR:@.+]](%struct.S* {{[^,]*}} [[S_CUR]])
  // CHECK: [[S_NEXT]] = getelementptr inbounds %struct.S, %struct.S* [[S_CUR]], i{{.+}} 1
  // CHECK: [[IS_DONE:%.+]] = icmp eq %struct.S* [[S_NEXT]], [[S_END]]
  // CHECK: br i1 [[IS_DONE]], label %[[DONE:.+]], label %[[ARRAY_INIT]]
  // CHECK: [[DONE]]:
  // CHECK: store i32 0, i32* [[OMP_CNT:%.+]],
  // CHECK: br label %[[OMP_HEADER:.+]]

  // CHECK: [[OMP_HEADER]]:
  // CHECK: [[CNT_VAL:%.+]] = load i32, i32* [[OMP_CNT]],
  // CHECK: [[CMP:%.+]] = icmp slt i32 [[CNT_VAL]], 10
  // CHECK: br i1 [[CMP]], label %[[OMP_BODY:.+]], label %[[OMP_END:.+]]
#pragma omp simd reduction(inscan, + : s)
  for (int i = 0; i < 10; ++i) {
    // CHECK: [[OMP_BODY]]:

    // i = OMP_CNT*1 + 0;
    // CHECK: [[CNT_VAL:%.+]] = load i32, i32* [[OMP_CNT]],
    // CHECK: [[MUL:%.+]] = mul nsw i32 [[CNT_VAL]], 1
    // CHECK: [[ADD:%.+]] = add nsw i32 0, [[MUL]]
    // CHECK: store i32 [[ADD]], i32* [[I_ADDR:%.+]],

    // S S_PRIV[2];
    // CHECK: [[S_BEGIN:%.+]] = getelementptr inbounds [2 x %struct.S], [2 x %struct.S]* [[S_PRIV_ADDR:%.+]], i{{.+}} 0, i{{.+}} 0
    // CHECK: [[S_END:%.+]] = getelementptr {{.*}}%struct.S, %struct.S* [[S_BEGIN]], i{{.+}} 2
    // CHECK: [[IS_DONE:%.+]] = icmp eq %struct.S* [[S_BEGIN]], [[S_END]]
    // CHECK: br i1 [[IS_DONE]], label %[[DONE:.+]], label %[[ARRAY_INIT:[^,]+]]
    // CHECK: [[ARRAY_INIT]]:
    // CHECK: [[S_CUR:%.+]] = phi %struct.S* [ [[S_BEGIN]], %[[OMP_BODY]] ], [ [[S_NEXT:%.+]], %[[ARRAY_INIT]] ]
    // CHECK: call void [[CONSTR]](%struct.S* {{[^,]*}} [[S_CUR]])
    // CHECK: [[S_NEXT]] = getelementptr {{.*}}%struct.S, %struct.S* [[S_CUR]], i{{.+}} 1
    // CHECK: [[IS_DONE:%.+]] = icmp eq %struct.S* [[S_NEXT]], [[S_END]]
    // CHECK: br i1 [[IS_DONE]], label %[[DONE:.+]], label %[[ARRAY_INIT]]
    // CHECK: [[DONE]]:
    // CHECK: [[LHS_BEGIN:%.+]] = bitcast [2 x %struct.S]* [[S_ADDR]] to %struct.S*
    // CHECK: [[RHS_BEGIN:%.+]] = bitcast [2 x %struct.S]* [[S_PRIV_ADDR]] to %struct.S*

    // goto DISPATCH;
    // CHECK: br label %[[DISPATCH:[^,]+]]

    // SCAN_PHASE:
    // foo();
    // goto CONTINUE;
    // CHECK: [[SCAN_PHASE:.+]]:
    // CHECK: call void @{{.*}}foo{{.*}}()
    // CHECK: br label %[[CONTINUE:[^,]+]]
    foo();

    // DISPATCH:
    // goto INPUT_PHASE;
    // CHECK: [[DISPATCH]]:
    // CHECK: br label %[[INPUT_PHASE:[^,]+]]

    // REDUCE:
    // TEMP = S;
    // S = S_PRIV + S;
    // S_PRIV = TEMP;
    // goto SCAN_PHASE;
    // CHECK: [[REDUCE:.+]]:

    // S TEMP[2];
    // CHECK: [[TEMP_ARR_BEG:%.+]] = getelementptr inbounds [2 x %struct.S], [2 x %struct.S]* [[TEMP_ARR:%.+]], i32 0, i32 0
    // CHECK: [[TEMP_ARR_END:%.+]] = getelementptr inbounds %struct.S, %struct.S* [[TEMP_ARR_BEG]], i64 2
    // CHECK: br label %[[BODY:[^,]+]]
    // CHECK: [[BODY]]:
    // CHECK: [[CUR:%.+]] = phi %struct.S* [ [[TEMP_ARR_BEG]], %[[REDUCE]] ], [ [[NEXT:%.+]], %[[BODY]] ]
    // CHECK: call void [[CONSTR]](%struct.S* {{[^,]*}} [[CUR]])
    // CHECK: [[NEXT]] = getelementptr inbounds %struct.S, %struct.S* [[CUR]], i64 1
    // CHECK: [[IS_DONE:%.+]] = icmp eq %struct.S* [[NEXT]], [[TEMP_ARR_END]]
    // CHECK: br i1 [[IS_DONE]], label %[[EXIT:[^,]+]], label %[[BODY]]
    // CHECK: [[EXIT]]:

    // TEMP = S;
    // CHECK: [[TEMP_ARR_BEG:%.+]] = getelementptr inbounds [2 x %struct.S], [2 x %struct.S]* [[TEMP_ARR]], i32 0, i32 0
    // CHECK: [[TEMP_ARR_END:%.+]] = getelementptr %struct.S, %struct.S* [[TEMP_ARR_BEG]], i64 2
    // CHECK: [[IS_EMPTY:%.+]] = icmp eq %struct.S* [[TEMP_ARR_BEG]], [[TEMP_ARR_END]]
    // CHECK: br i1 [[IS_EMPTY]], label %[[EXIT:[^,]+]], label %[[BODY:[^,]+]]
    // CHECK: [[BODY]]:
    // CHECK: [[CUR_SRC:%.+]] = phi %struct.S* [ [[LHS_BEGIN]], %{{.+}} ], [ [[SRC_NEXT:%.+]], %[[BODY]] ]
    // CHECK: [[CUR_DEST:%.+]] = phi %struct.S* [ [[TEMP_ARR_BEG]], %{{.+}} ], [ [[DEST_NEXT:%.+]], %[[BODY]] ]
    // CHECK: call {{.*}}%struct.S* [[S_COPY:@.+]](%struct.S* {{[^,]*}} [[CUR_DEST]], %struct.S* {{.*}}[[CUR_SRC]])
    // CHECK: [[DEST_NEXT:%.+]] = getelementptr %struct.S, %struct.S* [[CUR_DEST]], i32 1
    // CHECK: [[SRC_NEXT:%.+]] = getelementptr %struct.S, %struct.S* [[CUR_SRC]], i32 1
    // CHECK: [[IS_DONE:%.+]] = icmp eq %struct.S* [[DEST_NEXT]], [[TEMP_ARR_END]]
    // CHECK: br i1 [[IS_DONE]], label %[[EXIT]], label %[[BODY]]
    // CHECK: [[EXIT]]:

    // S = S_PRIV + S;
    // CHECK: [[LHS_END:%.+]] = getelementptr {{.*}}%struct.S, %struct.S* [[LHS_BEGIN]], i{{.+}} 2
    // CHECK: [[IS_DONE:%.+]] = icmp eq %struct.S* [[LHS_BEGIN]], [[LHS_END]]
    // CHECK: br i1 [[IS_DONE]], label %[[DONE:.+]], label %[[ARRAY_REDUCE_COPY:[^,]+]]
    // CHECK: [[ARRAY_REDUCE_COPY]]:
    // CHECK: [[SRC_CUR:%.+]] = phi %struct.S* [ [[RHS_BEGIN]], %[[EXIT]] ], [ [[SRC_NEXT:%.+]], %[[ARRAY_REDUCE_COPY]] ]
    // CHECK: [[DEST_CUR:%.+]] = phi %struct.S* [ [[LHS_BEGIN]], %[[EXIT]] ], [ [[DEST_NEXT:%.+]], %[[ARRAY_REDUCE_COPY]] ]
    // CHECK: [[SUM:%.+]] = call {{.*}}%struct.S* @{{.+}}(%struct.S* {{[^,]*}} [[DEST_CUR]], %struct.S* {{.*}}[[SRC_CUR]])
    // CHECK: call {{.*}}%struct.S* [[S_COPY]](%struct.S* {{[^,]*}} [[DEST_CUR]], %struct.S* {{.*}}[[SUM]])
    // CHECK: [[DEST_NEXT]] = getelementptr {{.*}}%struct.S, %struct.S* [[DEST_CUR]], i{{.+}} 1
    // CHECK: [[SRC_NEXT]] = getelementptr {{.*}}%struct.S, %struct.S* [[SRC_CUR]], i{{.+}} 1
    // CHECK: [[IS_DONE:%.+]] = icmp eq %struct.S* [[DEST_NEXT]], [[LHS_END]]
    // CHECK: br i1 [[IS_DONE]], label %[[DONE:.+]], label %[[ARRAY_REDUCE_COPY]]
    // CHECK: [[DONE]]:

    // S_PRIV = TEMP;
    // CHECK: [[TEMP_ARR_BEG:%.+]] = bitcast [2 x %struct.S]* [[TEMP_ARR]] to %struct.S*
    // CHECK: [[RHS_END:%.+]] = getelementptr %struct.S, %struct.S* [[RHS_BEGIN]], i64 2
    // CHECK: [[IS_EMPTY:%.+]] = icmp eq %struct.S* [[RHS_BEGIN]], [[RHS_END]]
    // CHECK: br i1 [[IS_EMPTY]], label %[[EXIT:[^,]+]], label %[[BODY:[^,]+]]
    // CHECK: [[BODY]]:
    // CHECK: [[CUR_SRC:%.+]] = phi %struct.S* [ [[TEMP_ARR_BEG]], %[[DONE]] ], [ [[SRC_NEXT:%.+]], %[[BODY]] ]
    // CHECK: [[CUR_DEST:%.+]] = phi %struct.S* [ [[RHS_BEGIN]], %[[DONE]] ], [ [[DEST_NEXT:%.+]], %[[BODY]] ]
    // CHECK: call {{.*}}%struct.S* [[S_COPY]](%struct.S* {{[^,]*}} [[CUR_DEST]], %struct.S* {{.*}}[[CUR_SRC]])
    // CHECK: [[DEST_NEXT]] = getelementptr %struct.S, %struct.S* [[CUR_DEST]], i32 1
    // CHECK: [[SRC_NEXT]] = getelementptr %struct.S, %struct.S* [[CUR_SRC]], i32 1
    // CHECK: [[IS_DONE:%.+]] = icmp eq %struct.S* [[DEST_NEXT]], [[RHS_END]]
    // CHECK: br i1 [[IS_DONE]], label %[[DONE:[^,]+]], label %[[BODY]]
    // CHECK: [[DONE]]:

    // TEMP.~S()
    // CHECK: [[TEMP_ARR_BEG:%.+]] = getelementptr inbounds [2 x %struct.S], [2 x %struct.S]* [[TEMP_ARR]], i32 0, i32 0
    // CHECK: [[TEMP_ARR_END:%.+]] = getelementptr inbounds %struct.S, %struct.S* [[TEMP_ARR_BEG]], i64 2
    // CHECK: br label %[[BODY:[^,]+]]
    // CHECK: [[BODY]]:
    // CHECK: [[CUR:%.+]] = phi %struct.S* [ [[TEMP_ARR_END]], %[[DONE]] ], [ [[PREV:%.+]], %[[BODY]] ]
    // CHECK: [[PREV]] = getelementptr inbounds %struct.S, %struct.S* [[CUR]], i64 -1
    // CHECK: call void [[DESTR:@.+]](%struct.S* {{[^,]*}} [[PREV]])
    // CHECK: [[IS_DONE:%.+]] = icmp eq %struct.S* [[PREV]], [[TEMP_ARR_BEG]]
    // CHECK: br i1 [[IS_DONE]], label %[[EXIT:[^,]+]], label %[[BODY]]
    // CHECK: [[EXIT]]:

    // goto SCAN_PHASE;
    // CHECK: br label %[[SCAN_PHASE]]
#pragma omp scan exclusive(s)

    // INPUT_PHASE:
    // bar();
    // goto REDUCE;
    // CHECK: [[INPUT_PHASE]]:
    // CHECK: call void @{{.*}}bar{{.*}}()
    // CHECK: br label %[[REDUCE]]
    bar();

    // CHECK: [[CONTINUE]]:

    // S_PRIV[2].~S();
    // CHECK: [[S_BEGIN:%.+]] = getelementptr inbounds [2 x %struct.S], [2 x %struct.S]* [[S_PRIV_ADDR]], i{{.+}} 0, i{{.+}} 0
    // CHECK: [[S_END:%.+]] = getelementptr {{.*}}%struct.S, %struct.S* [[S_BEGIN]], i{{.+}} 2
    // CHECK: br label %[[ARRAY_DESTR:[^,]+]]
    // CHECK: [[ARRAY_DESTR]]:
    // CHECK: [[S_CUR:%.+]] = phi %struct.S* [ [[S_END]], %[[CONTINUE]] ], [ [[S_PREV:%.+]], %[[ARRAY_DESTR]] ]
    // CHECK: [[S_PREV]] = getelementptr {{.*}}%struct.S, %struct.S* [[S_CUR]], i{{.+}} -1
    // CHECK: call void [[DESTR]](%struct.S* {{[^,]*}} [[S_PREV]])
    // CHECK: [[IS_DONE:%.+]] = icmp eq %struct.S* [[S_PREV]], [[S_BEGIN]]
    // CHECK: br i1 [[IS_DONE]], label %[[DONE:.+]], label %[[ARRAY_DESTR]]
    // CHECK: [[DONE]]:
    // CHECK: br label %[[INC_BLOCK:[^,]+]]

    // ++OMP_CNT;
    // CHECK: [[INC_BLOCK]]:
    // CHECK: [[CNT:%.+]] = load i32, i32* [[OMP_CNT]],
    // CHECK: [[INC:%.+]] = add nsw i32 [[CNT]], 1
    // CHECK: store i32 [[INC]], i32* [[OMP_CNT]],
    // CHECK: br label %[[OMP_HEADER]]
  }
  // CHECK: [[OMP_END]]:
}

// CHECK-NOT: !{!"llvm.loop.parallel_accesses"

#endif // HEADER
