// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -fexceptions -fcxx-exceptions -o - -fopenmp-version=45 | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s -fopenmp-version=45
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -debug-info-kind=limited -std=c++11 -include-pch %t -verify %s -emit-llvm -o - -fopenmp-version=45 | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -verify -triple x86_64-apple-darwin10 -fopenmp -fexceptions -fcxx-exceptions -debug-info-kind=line-tables-only -x c++ -emit-llvm %s -o - -fopenmp-version=45 | FileCheck %s --check-prefix=TERM_DEBUG
// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -fexceptions -fcxx-exceptions -o - -fopenmp-version=50 -DOMP5 | FileCheck %s --check-prefix=CHECK --check-prefix=OMP50 --check-prefix=OMP50RT
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s -fopenmp-version=50 -DOMP5
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -debug-info-kind=limited -std=c++11 -include-pch %t -verify %s -emit-llvm -o - -fopenmp-version=50 -DOMP5 | FileCheck %s --check-prefix=CHECK --check-prefix=OMP50 --check-prefix=OMP50RT

// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp-simd -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -fexceptions -fcxx-exceptions -o - -fopenmp-version=45 | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s -fopenmp-version=45
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -debug-info-kind=limited -std=c++11 -include-pch %t -verify %s -emit-llvm -o - -fopenmp-version=45 | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -verify -triple x86_64-apple-darwin10 -fopenmp-simd -fexceptions -fcxx-exceptions -debug-info-kind=line-tables-only -x c++ -emit-llvm %s -o - -fopenmp-version=45 | FileCheck --check-prefix=TERM_DEBUG %s
// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp-simd -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -fexceptions -fcxx-exceptions -o - -fopenmp-version=50 -DOMP5 | FileCheck %s --check-prefix=CHECK --check-prefix=OMP50
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s -fopenmp-version=50 -DOMP5
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -debug-info-kind=limited -std=c++11 -include-pch %t -verify %s -emit-llvm -o - -fopenmp-version=50 -DOMP5 | FileCheck %s --check-prefix=CHECK --check-prefix=OMP50
// expected-no-diagnostics
 #ifndef HEADER
 #define HEADER

#ifdef OMP5
#define CONDITIONAL conditional :
#else
#define CONDITIONAL
#endif //OMP5
// CHECK: [[SS_TY:%.+]] = type { i32 }
// OMP50-DAG: [[LAST_IV:@.+]] = {{.*}}common global i64 0
// OMP50-DAG: [[LAST_A:@.+]] = {{.*}}common global i32 0

long long get_val() { extern void mayThrow(); mayThrow(); return 0; }
double *g_ptr;

struct S {
  int a, b;
};

// CHECK-LABEL: define {{.*void}} @{{.*}}simple{{.*}}(float* noundef {{.+}}, float* noundef {{.+}}, float* noundef {{.+}}, float* noundef {{.+}})
void simple(float *a, float *b, float *c, float *d) {
  S s, *p;
#ifdef OMP5
  #pragma omp simd if (simd: true) nontemporal(a, b, c, d, s)
#else
  #pragma omp simd
#endif
  // CHECK: store i32 0, i32* [[OMP_IV:%[^,]+]]

// CHECK: [[IV:%.+]] = load i32, i32* [[OMP_IV]]{{.*}}!llvm.access.group
// CHECK-NEXT: [[CMP:%.+]] = icmp slt i32 [[IV]], 6
// CHECK-NEXT: br i1 [[CMP]], label %[[SIMPLE_LOOP1_BODY:.+]], label %[[SIMPLE_LOOP1_END:[^,]+]]
  for (int i = 3; i < 32; i += 5) {
// CHECK: [[SIMPLE_LOOP1_BODY]]:
// Start of body: calculate i from IV:
// CHECK: [[IV1_1:%.+]] = load i32, i32* [[OMP_IV]]{{.*}}!llvm.access.group
// CHECK: [[CALC_I_1:%.+]] = mul nsw i32 [[IV1_1]], 5
// CHECK-NEXT: [[CALC_I_2:%.+]] = add nsw i32 3, [[CALC_I_1]]
// CHECK-NEXT: store i32 [[CALC_I_2]], i32* [[LC_I:.+]]{{.*}}!llvm.access.group
// ... loop body ...
// End of body: store into a[i]:
// OMP45-NOT: load float*,{{.*}}!nontemporal
// CHECK-NOT: load float,{{.*}}!nontemporal
// OMP50: load float*,{{.*}}!nontemporal
// OMP50: load float*,{{.*}}!nontemporal
// OMP50: load float*,{{.*}}!nontemporal
// OMP50: load i32,{{.*}}!nontemporal
// OMP50-NOT: load i32,{{.*}}!nontemporal
// OMP50: load float*,{{.*}}!nontemporal
// CHECK-NOT: load float,{{.*}}!nontemporal
// CHECK: store float [[RESULT:%.+]], float* {{%.+}}{{.*}}!llvm.access.group
    a[i] = b[i] * c[i] * d[i] + s.a + p->a;
// CHECK: [[IV1_2:%.+]] = load i32, i32* [[OMP_IV]]{{.*}}!llvm.access.group
// CHECK-NEXT: [[ADD1_2:%.+]] = add nsw i32 [[IV1_2]], 1
// CHECK-NEXT: store i32 [[ADD1_2]], i32* [[OMP_IV]]{{.*}}!llvm.access.group
// br label %{{.+}}, !llvm.loop !{{.+}}
  }
// CHECK: [[SIMPLE_LOOP1_END]]:

  long long k = get_val();

  #pragma omp simd linear(k : 3)
// CHECK: [[K0:%.+]] = call {{.*}}i64 @{{.*}}get_val
// CHECK-NEXT: store i64 [[K0]], i64* [[K_VAR:%[^,]+]]
// CHECK: store i32 0, i32* [[OMP_IV2:%[^,]+]]
// CHECK: [[K0LOAD:%.+]] = load i64, i64* [[K_VAR]]
// CHECK-NEXT: store i64 [[K0LOAD]], i64* [[LIN0:%[^,]+]]

// CHECK: [[IV2:%.+]] = load i32, i32* [[OMP_IV2]]{{.*}}!llvm.access.group
// CHECK-NEXT: [[CMP2:%.+]] = icmp slt i32 [[IV2]], 9
// CHECK-NEXT: br i1 [[CMP2]], label %[[SIMPLE_LOOP2_BODY:.+]], label %[[SIMPLE_LOOP2_END:[^,]+]]
  for (int i = 10; i > 1; i--) {
// CHECK: [[SIMPLE_LOOP2_BODY]]:
// Start of body: calculate i from IV:
// CHECK: [[IV2_0:%.+]] = load i32, i32* [[OMP_IV2]]{{.*}}!llvm.access.group
// FIXME: It is interesting, why the following "mul 1" was not constant folded?
// CHECK-NEXT: [[IV2_1:%.+]] = mul nsw i32 [[IV2_0]], 1
// CHECK-NEXT: [[LC_I_1:%.+]] = sub nsw i32 10, [[IV2_1]]
// CHECK-NEXT: store i32 [[LC_I_1]], i32* {{.+}}, !llvm.access.group
//
// CHECK-NEXT: [[LIN0_1:%.+]] = load i64, i64* [[LIN0]]{{.*}}!llvm.access.group
// CHECK-NEXT: [[IV2_2:%.+]] = load i32, i32* [[OMP_IV2]]{{.*}}!llvm.access.group
// CHECK-NEXT: [[LIN_MUL1:%.+]] = mul nsw i32 [[IV2_2]], 3
// CHECK-NEXT: [[LIN_EXT1:%.+]] = sext i32 [[LIN_MUL1]] to i64
// CHECK-NEXT: [[LIN_ADD1:%.+]] = add nsw i64 [[LIN0_1]], [[LIN_EXT1]]
// Update of the privatized version of linear variable!
// CHECK-NEXT: store i64 [[LIN_ADD1]], i64* [[K_PRIVATIZED:%[^,]+]]
    a[k]++;
    k = k + 3;
// CHECK: [[IV2_2:%.+]] = load i32, i32* [[OMP_IV2]]{{.*}}!llvm.access.group
// CHECK-NEXT: [[ADD2_2:%.+]] = add nsw i32 [[IV2_2]], 1
// CHECK-NEXT: store i32 [[ADD2_2]], i32* [[OMP_IV2]]{{.*}}!llvm.access.group
// br label {{.+}}, !llvm.loop ![[SIMPLE_LOOP2_ID]]
  }
// CHECK: [[SIMPLE_LOOP2_END]]:
//
// Update linear vars after loop, as the loop was operating on a private version.
// CHECK: [[LIN0_2:%.+]] = load i64, i64* [[K_PRIVATIZED]],
// CHECK-NEXT: store i64 [[LIN0_2]], i64* [[K_VAR]]
//

  int lin = 12;
  #pragma omp simd linear(lin : get_val()), linear(g_ptr)

// Init linear private var.
// CHECK: store i32 12, i32* [[LIN_VAR:%[^,]+]]
// CHECK: store i64 0, i64* [[OMP_IV3:%[^,]+]]

// CHECK: [[LIN_LOAD:%.+]] = load i32, i32* [[LIN_VAR]]
// CHECK-NEXT: store i32 [[LIN_LOAD]], i32* [[LIN_START:%[^,]+]]
// Remember linear step.
// CHECK: [[CALL_VAL:%.+]] = invoke
// CHECK: store i64 [[CALL_VAL]], i64* [[LIN_STEP:%[^,]+]]

// CHECK: [[GLIN_LOAD:%.+]] = load double*, double** [[GLIN_VAR:@[^,]+]]
// CHECK-NEXT: store double* [[GLIN_LOAD]], double** [[GLIN_START:%[^,]+]]

// CHECK: [[IV3:%.+]] = load i64, i64* [[OMP_IV3]]{{.*}}!llvm.access.group
// CHECK-NEXT: [[CMP3:%.+]] = icmp ult i64 [[IV3]], 4
// CHECK-NEXT: br i1 [[CMP3]], label %[[SIMPLE_LOOP3_BODY:.+]], label %[[SIMPLE_LOOP3_END:[^,]+]]
  for (unsigned long long it = 2000; it >= 600; it-=400) {
// CHECK: [[SIMPLE_LOOP3_BODY]]:
// Start of body: calculate it from IV:
// CHECK: [[IV3_0:%.+]] = load i64, i64* [[OMP_IV3]]{{.*}}!llvm.access.group
// CHECK-NEXT: [[LC_IT_1:%.+]] = mul i64 [[IV3_0]], 400
// CHECK-NEXT: [[LC_IT_2:%.+]] = sub i64 2000, [[LC_IT_1]]
// CHECK-NEXT: store i64 [[LC_IT_2]], i64* {{.+}}, !llvm.access.group
//
// Linear start and step are used to calculate current value of the linear variable.
// CHECK: [[LINSTART:.+]] = load i32, i32* [[LIN_START]]{{.*}}!llvm.access.group
// CHECK: [[LINSTEP:.+]] = load i64, i64* [[LIN_STEP]]{{.*}}!llvm.access.group
// CHECK-NOT: store i32 {{.+}}, i32* [[LIN_VAR]],{{.*}}!llvm.access.group
// CHECK: store i32 {{.+}}, i32* [[LIN_PRIV:%.+]],{{.*}}!llvm.access.group
// CHECK: [[GLINSTART:.+]] = load double*, double** [[GLIN_START]]{{.*}}!llvm.access.group
// CHECK-NEXT: [[IV3_1:%.+]] = load i64, i64* [[OMP_IV3]]{{.*}}!llvm.access.group
// CHECK-NEXT: [[MUL:%.+]] = mul i64 [[IV3_1]], 1
// CHECK: [[GEP:%.+]] = getelementptr{{.*}}[[GLINSTART]]
// CHECK-NEXT: store double* [[GEP]], double** [[G_PTR_CUR:%[^,]+]]{{.*}}!llvm.access.group
    *g_ptr++ = 0.0;
// CHECK: [[GEP_VAL:%.+]] = load double{{.*}}[[G_PTR_CUR]]{{.*}}!llvm.access.group
// CHECK: store double{{.*}}[[GEP_VAL]]{{.*}}!llvm.access.group
    a[it + lin]++;
// CHECK: [[FLT_INC:%.+]] = fadd float
// CHECK-NEXT: store float [[FLT_INC]],{{.*}}!llvm.access.group
// CHECK: [[IV3_2:%.+]] = load i64, i64* [[OMP_IV3]]{{.*}}!llvm.access.group
// CHECK-NEXT: [[ADD3_2:%.+]] = add i64 [[IV3_2]], 1
// CHECK-NEXT: store i64 [[ADD3_2]], i64* [[OMP_IV3]]{{.*}}!llvm.access.group
  }
// CHECK: [[SIMPLE_LOOP3_END]]:
//
// Linear start and step are used to calculate final value of the linear variables.
// CHECK: [[LIN:%.+]] = load i32, i32* [[LIN_PRIV]]
// CHECK-NEXT: store i32 [[LIN]], i32* [[LIN_VAR]],
// CHECK: [[GLIN:%.+]] = load double*, double** [[G_PTR_CUR]]
// CHECK-NEXT: store double* [[GLIN]], double** [[GLIN_VAR]]

  #pragma omp simd
// CHECK: store i32 0, i32* [[OMP_IV4:%[^,]+]]

// CHECK: [[IV4:%.+]] = load i32, i32* [[OMP_IV4]]{{.*}}!llvm.access.group
// CHECK-NEXT: [[CMP4:%.+]] = icmp slt i32 [[IV4]], 4
// CHECK-NEXT: br i1 [[CMP4]], label %[[SIMPLE_LOOP4_BODY:.+]], label %[[SIMPLE_LOOP4_END:[^,]+]]
  for (short it = 6; it <= 20; it-=-4) {
// CHECK: [[SIMPLE_LOOP4_BODY]]:
// Start of body: calculate it from IV:
// CHECK: [[IV4_0:%.+]] = load i32, i32* [[OMP_IV4]]{{.*}}!llvm.access.group
// CHECK-NEXT: [[LC_IT_1:%.+]] = mul nsw i32 [[IV4_0]], 4
// CHECK-NEXT: [[LC_IT_2:%.+]] = add nsw i32 6, [[LC_IT_1]]
// CHECK-NEXT: [[LC_IT_3:%.+]] = trunc i32 [[LC_IT_2]] to i16
// CHECK-NEXT: store i16 [[LC_IT_3]], i16* {{.+}}, !llvm.access.group

// CHECK: [[IV4_2:%.+]] = load i32, i32* [[OMP_IV4]]{{.*}}!llvm.access.group
// CHECK-NEXT: [[ADD4_2:%.+]] = add nsw i32 [[IV4_2]], 1
// CHECK-NEXT: store i32 [[ADD4_2]], i32* [[OMP_IV4]]{{.*}}!llvm.access.group
  }
// CHECK: [[SIMPLE_LOOP4_END]]:

  #pragma omp simd
// CHECK: store i32 0, i32* [[OMP_IV5:%[^,]+]]

// CHECK: [[IV5:%.+]] = load i32, i32* [[OMP_IV5]]{{.*}}!llvm.access.group
// CHECK-NEXT: [[CMP5:%.+]] = icmp slt i32 [[IV5]], 26
// CHECK-NEXT: br i1 [[CMP5]], label %[[SIMPLE_LOOP5_BODY:.+]], label %[[SIMPLE_LOOP5_END:[^,]+]]
  for (unsigned char it = 'z'; it >= 'a'; it+=-1) {
// CHECK: [[SIMPLE_LOOP5_BODY]]:
// Start of body: calculate it from IV:
// CHECK: [[IV5_0:%.+]] = load i32, i32* [[OMP_IV5]]{{.*}}!llvm.access.group
// CHECK-NEXT: [[IV5_1:%.+]] = mul nsw i32 [[IV5_0]], 1
// CHECK-NEXT: [[LC_IT_1:%.+]] = sub nsw i32 122, [[IV5_1]]
// CHECK-NEXT: [[LC_IT_2:%.+]] = trunc i32 [[LC_IT_1]] to i8
// CHECK-NEXT: store i8 [[LC_IT_2]], i8* {{.+}}, !llvm.access.group

// CHECK: [[IV5_2:%.+]] = load i32, i32* [[OMP_IV5]]{{.*}}!llvm.access.group
// CHECK-NEXT: [[ADD5_2:%.+]] = add nsw i32 [[IV5_2]], 1
// CHECK-NEXT: store i32 [[ADD5_2]], i32* [[OMP_IV5]]{{.*}}!llvm.access.group
  }
// CHECK: [[SIMPLE_LOOP5_END]]:

// CHECK-NOT: mul i32 %{{.+}}, 10
  #pragma omp simd
  for (unsigned i=100; i<10; i+=10) {
  }

  int A;
  // CHECK: store i32 -1, i32* [[A:%.+]],
  A = -1;
  #pragma omp simd lastprivate(CONDITIONAL A)
// CHECK: store i64 0, i64* [[OMP_IV7:%[^,]+]]
// CHECK: br label %[[SIMD_LOOP7_COND:[^,]+]]
// CHECK: [[SIMD_LOOP7_COND]]:
// CHECK-NEXT: [[IV7:%.+]] = load i64, i64* [[OMP_IV7]]{{.*}}!llvm.access.group
// CHECK-NEXT: [[CMP7:%.+]] = icmp slt i64 [[IV7]], 7
// CHECK-NEXT: br i1 [[CMP7]], label %[[SIMPLE_LOOP7_BODY:.+]], label %[[SIMPLE_LOOP7_END:[^,]+]]
  for (long long i = -10; i < 10; i += 3) {
// CHECK: [[SIMPLE_LOOP7_BODY]]:
// Start of body: calculate i from IV:
// CHECK: [[IV7_0:%.+]] = load i64, i64* [[OMP_IV7]]{{.*}}!llvm.access.group
// CHECK-NEXT: [[LC_IT_1:%.+]] = mul nsw i64 [[IV7_0]], 3
// CHECK-NEXT: [[LC_IT_2:%.+]] = add nsw i64 -10, [[LC_IT_1]]
// CHECK-NEXT: store i64 [[LC_IT_2]], i64* [[LC:%[^,]+]],{{.+}}!llvm.access.group
// CHECK-NEXT: [[LC_VAL:%.+]] = load i64, i64* [[LC]]{{.+}}!llvm.access.group
// CHECK-NEXT: [[A_VAL:%.+]] = load i32, i32* [[A_PRIV:%[^,]+]],{{.+}}!llvm.access.group
// CHECK-NEXT: [[CAST:%.+]] = sext i32 [[A_VAL]] to i64
// CHECK-NEXT: [[ADD:%.+]] = add nsw i64 [[CAST]], [[LC_VAL]]
// CHECK-NEXT: [[CONV:%.+]] = trunc i64 [[ADD]] to i32
// CHECK-NEXT: store i32 [[CONV]], i32* [[A_PRIV]],{{.+}}!llvm.access.group
// OMP50-NEXT: [[IV:%.+]] = load i64, i64* [[OMP_IV7]],{{.+}}!llvm.access.group
// OMP50RT:    call void @__kmpc_critical(%struct.ident_t* {{.+}}, i32 [[GTID:%.+]], [8 x i32]* [[A_REGION:@.+]]),{{.+}}!llvm.access.group
// OMP50-NEXT: [[LAST_IV_VAL:%.+]] = load i64, i64* [[LAST_IV]],{{.+}}!llvm.access.group
// OMP50-NEXT: [[CMP:%.+]] = icmp sle i64 [[LAST_IV_VAL]], [[IV]]
// OMP50-NEXT: br i1 [[CMP]], label %[[LP_THEN:.+]], label %[[LP_DONE:[^,]+]]
// OMP50:      [[LP_THEN]]:
// OMP50-NEXT: store i64 [[IV]], i64* [[LAST_IV]],{{.+}}!llvm.access.group
// OMP50-NEXT: [[A_VAL:%.+]] = load i32, i32* [[A_PRIV]],{{.+}}!llvm.access.group
// OMP50-NEXT: store i32 [[A_VAL]], i32* [[LAST_A]],{{.+}}!llvm.access.group
// OMP50-NEXT: br label %[[LP_DONE]]
// OMP50:      [[LP_DONE]]:
// OMP50RT-NEXT: call void @__kmpc_end_critical(%struct.ident_t* {{.+}}, i32 [[GTID]], [8 x i32]* [[A_REGION]]),{{.+}}!llvm.access.group
    A += i;
// CHECK: [[IV7_2:%.+]] = load i64, i64* [[OMP_IV7]]{{.*}}!llvm.access.group
// CHECK-NEXT: [[ADD7_2:%.+]] = add nsw i64 [[IV7_2]], 1
// CHECK-NEXT: store i64 [[ADD7_2]], i64* [[OMP_IV7]]{{.*}}!llvm.access.group
  }
// CHECK: [[SIMPLE_LOOP7_END]]:
// CHECK-NEXT: store i64 11, i64*
// OMP50-NEXT: [[LAST_A_VAL:%.+]] = load i32, i32* [[LAST_A]],
// OMP50-NEXT: store i32 [[LAST_A_VAL]], i32* [[A_PRIV]],
// CHECK-NEXT: [[A_PRIV_VAL:%.+]] = load i32, i32* [[A_PRIV]],
// CHECK-NEXT: store i32 [[A_PRIV_VAL]], i32* [[A]],
  int R;
  // CHECK: store i32 -1, i32* [[R:%[^,]+]],
  R = -1;
// CHECK: store i64 0, i64* [[OMP_IV8:%[^,]+]],
// CHECK: store i32 1, i32* [[R_PRIV:%[^,]+]],
#ifdef OMP5
  #pragma omp simd reduction(*:R) if(A)
#else
  #pragma omp simd reduction(*:R)
#endif
// OMP50:      [[A_VAL:%.+]] = load i32, i32* [[A]],
// OMP50-NEXT: [[COND:%.+]] = icmp ne i32 [[A_VAL]], 0
// OMP50-NEXT: br i1 [[COND]], label {{%?}}[[THEN:[^,]+]], label {{%?}}[[ELSE:[^,]+]]
// OMP50:      [[THEN]]:

// CHECK: br label %[[SIMD_LOOP8_COND:[^,]+]]
// CHECK: [[SIMD_LOOP8_COND]]:
// CHECK-NEXT: [[IV8:%.+]] = load i64, i64* [[OMP_IV8]]{{.*}}!llvm.access.group
// CHECK-NEXT: [[CMP8:%.+]] = icmp slt i64 [[IV8]], 7
// CHECK-NEXT: br i1 [[CMP8]], label %[[SIMPLE_LOOP8_BODY:.+]], label %[[SIMPLE_LOOP8_END:[^,]+]]
  for (long long i = -10; i < 10; i += 3) {
// CHECK: [[SIMPLE_LOOP8_BODY]]:
// Start of body: calculate i from IV:
// CHECK: [[IV8_0:%.+]] = load i64, i64* [[OMP_IV8]]{{.*}}!llvm.access.group
// CHECK-NEXT: [[LC_IT_1:%.+]] = mul nsw i64 [[IV8_0]], 3
// CHECK-NEXT: [[LC_IT_2:%.+]] = add nsw i64 -10, [[LC_IT_1]]
// CHECK-NEXT: store i64 [[LC_IT_2]], i64* [[LC:%[^,]+]],{{.+}}!llvm.access.group
// CHECK-NEXT: [[LC_VAL:%.+]] = load i64, i64* [[LC]]{{.+}}!llvm.access.group
// CHECK: store i32 %{{.+}}, i32* [[R_PRIV]],{{.+}}!llvm.access.group
    R *= i;
// CHECK: [[IV8_2:%.+]] = load i64, i64* [[OMP_IV8]]{{.*}}!llvm.access.group
// CHECK-NEXT: [[ADD8_2:%.+]] = add nsw i64 [[IV8_2]], 1
// CHECK-NEXT: store i64 [[ADD8_2]], i64* [[OMP_IV8]]{{.*}}!llvm.access.group
  }
// CHECK: [[SIMPLE_LOOP8_END]]:
// OMP50: br label {{%?}}[[EXIT:[^,]+]]
// OMP50: br label %[[SIMD_LOOP8_COND:[^,]+]]
// OMP50: [[SIMD_LOOP8_COND]]:
// OMP50-NEXT: [[IV8:%.+]] = load i64, i64* [[OMP_IV8]],{{[^!]*}}
// OMP50-NEXT: [[CMP8:%.+]] = icmp slt i64 [[IV8]], 7
// OMP50-NEXT: br i1 [[CMP8]], label %[[SIMPLE_LOOP8_BODY:.+]], label %[[SIMPLE_LOOP8_END:[^,]+]]
// OMP50: [[SIMPLE_LOOP8_BODY]]:
// Start of body: calculate i from IV:
// OMP50: [[IV8_0:%.+]] = load i64, i64* [[OMP_IV8]],{{[^!]*}}
// OMP50-NEXT: [[LC_IT_1:%.+]] = mul nsw i64 [[IV8_0]], 3
// OMP50-NEXT: [[LC_IT_2:%.+]] = add nsw i64 -10, [[LC_IT_1]]
// OMP50-NEXT: store i64 [[LC_IT_2]], i64* [[LC:%[^,]+]],{{[^!]*}}
// OMP50-NEXT: [[LC_VAL:%.+]] = load i64, i64* [[LC]],{{[^!]*}}
// OMP50: store i32 %{{.+}}, i32* [[R_PRIV]],{{[^!]*}}
// OMP50: [[IV8_2:%.+]] = load i64, i64* [[OMP_IV8]],{{[^!]*}}
// OMP50-NEXT: [[ADD8_2:%.+]] = add nsw i64 [[IV8_2]], 1
// OMP50-NEXT: store i64 [[ADD8_2]], i64* [[OMP_IV8]],{{[^!]*}}
// OMP50:      br label {{%?}}[[SIMD_LOOP8_COND]], {{.*}}!llvm.loop ![[DISABLE_VECT:.+]]
// OMP50: [[SIMPLE_LOOP8_END]]:
// OMP50: br label {{%?}}[[EXIT]]
// OMP50: [[EXIT]]:

// CHECK-DAG: [[R_VAL:%.+]] = load i32, i32* [[R]],
// CHECK-DAG: [[R_PRIV_VAL:%.+]] = load i32, i32* [[R_PRIV]],
// CHECK: [[RED:%.+]] = mul nsw i32 [[R_VAL]], [[R_PRIV_VAL]]
// CHECK-NEXT: store i32 [[RED]], i32* [[R]],
// CHECK-NEXT: ret void
}

template <class T, unsigned K> T tfoo(T a) { return a + K; }

template <typename T, unsigned N>
int templ1(T a, T *z) {
  #pragma omp simd collapse(N)
  for (int i = 0; i < N * 2; i++) {
    for (long long j = 0; j < (N + N + N + N); j += 2) {
      z[i + j] = a + tfoo<T, N>(i + j);
    }
  }
  return 0;
}

// Instatiation templ1<float,2>
// CHECK-LABEL: define {{.*i32}} @{{.*}}templ1{{.*}}(float noundef {{.+}}, float* noundef {{.+}})
// CHECK: store i64 0, i64* [[T1_OMP_IV:[^,]+]]
// ...
// CHECK: [[IV:%.+]] = load i64, i64* [[T1_OMP_IV]]{{.*}}!llvm.access.group
// CHECK-NEXT: [[CMP1:%.+]] = icmp slt i64 [[IV]], 16
// CHECK-NEXT: br i1 [[CMP1]], label %[[T1_BODY:.+]], label %[[T1_END:[^,]+]]
// CHECK: [[T1_BODY]]:
// Loop counters i and j updates:
// CHECK: [[IV1:%.+]] = load i64, i64* [[T1_OMP_IV]]{{.*}}!llvm.access.group
// CHECK-NEXT: [[I_1:%.+]] = sdiv i64 [[IV1]], 4
// CHECK-NEXT: [[I_1_MUL1:%.+]] = mul nsw i64 [[I_1]], 1
// CHECK-NEXT: [[I_1_ADD0:%.+]] = add nsw i64 0, [[I_1_MUL1]]
// CHECK-NEXT: [[I_2:%.+]] = trunc i64 [[I_1_ADD0]] to i32
// CHECK-NEXT: store i32 [[I_2]], i32* {{%.+}}{{.*}}!llvm.access.group
// CHECK: [[IV2:%.+]] = load i64, i64* [[T1_OMP_IV]]{{.*}}!llvm.access.group
// CHECK: [[IV2_1:%.+]] = load i64, i64* [[T1_OMP_IV]]{{.*}}!llvm.access.group
// CHECK-NEXT: [[J_1_DIV1:%.+]] = sdiv i64 [[IV2_1]], 4
// CHECK-NEXT: [[J_1_MUL1:%.+]] = mul nsw i64 [[J_1_DIV1]], 4
// CHECK-NEXT: [[J_1_SUB0:%.+]] = sub nsw i64 [[IV2]], [[J_1_MUL1]]
// CHECK-NEXT: [[J_2:%.+]] = mul nsw i64 [[J_1_SUB0]], 2
// CHECK-NEXT: [[J_2_ADD0:%.+]] = add nsw i64 0, [[J_2]]
// CHECK-NEXT: store i64 [[J_2_ADD0]], i64* {{%.+}}{{.*}}!llvm.access.group
// simd.for.inc:
// CHECK: [[IV3:%.+]] = load i64, i64* [[T1_OMP_IV]]{{.*}}!llvm.access.group
// CHECK-NEXT: [[INC:%.+]] = add nsw i64 [[IV3]], 1
// CHECK-NEXT: store i64 [[INC]], i64* [[T1_OMP_IV]]{{.*}}!llvm.access.group
// CHECK-NEXT: br label {{%.+}}
// CHECK: [[T1_END]]:
// CHECK: ret i32 0
//
void inst_templ1() {
  float a;
  float z[100];
  templ1<float,2> (a, z);
}


typedef int MyIdx;

class IterDouble {
  double *Ptr;
public:
  IterDouble operator++ () const {
    IterDouble n;
    n.Ptr = Ptr + 1;
    return n;
  }
  bool operator < (const IterDouble &that) const {
    return Ptr < that.Ptr;
  }
  double & operator *() const {
    return *Ptr;
  }
  MyIdx operator - (const IterDouble &that) const {
    return (MyIdx) (Ptr - that.Ptr);
  }
  IterDouble operator + (int Delta) {
    IterDouble re;
    re.Ptr = Ptr + Delta;
    return re;
  }

  ///~IterDouble() {}
};

// CHECK-LABEL: define {{.*void}} @{{.*}}iter_simple{{.*}}
void iter_simple(IterDouble ia, IterDouble ib, IterDouble ic) {
//
// Calculate number of iterations before the loop body.
// CHECK: [[DIFF1:%.+]] = invoke {{.*}}i32 @{{.*}}IterDouble{{.*}}
// CHECK: [[DIFF2:%.+]] = sub nsw i32 [[DIFF1]], 1
// CHECK-NEXT: [[DIFF3:%.+]] = add nsw i32 [[DIFF2]], 1
// CHECK-NEXT: [[DIFF4:%.+]] = sdiv i32 [[DIFF3]], 1
// CHECK-NEXT: [[DIFF5:%.+]] = sub nsw i32 [[DIFF4]], 1
// CHECK-NEXT: store i32 [[DIFF5]], i32* [[OMP_LAST_IT:%[^,]+]]{{.+}}
// CHECK: store i32 0, i32* [[IT_OMP_IV:%[^,]+]]
  #pragma omp simd

// CHECK: [[IV:%.+]] = load i32, i32* [[IT_OMP_IV]]{{.+}} !llvm.access.group
// CHECK-NEXT: [[LAST_IT:%.+]] = load i32, i32* [[OMP_LAST_IT]]{{.+}}!llvm.access.group
// CHECK-NEXT: [[NUM_IT:%.+]] = add nsw i32 [[LAST_IT]], 1
// CHECK-NEXT: [[CMP:%.+]] = icmp slt i32 [[IV]], [[NUM_IT]]
// CHECK-NEXT: br i1 [[CMP]], label %[[IT_BODY:[^,]+]], label %[[IT_END:[^,]+]]
  for (IterDouble i = ia; i < ib; ++i) {
// CHECK: [[IT_BODY]]:
// Start of body: calculate i from index:
// CHECK: [[IV1:%.+]] = load i32, i32* [[IT_OMP_IV]]{{.+}}!llvm.access.group
// Call of operator+ (i, IV).
// CHECK: {{%.+}} = invoke {{.+}} @{{.*}}IterDouble{{.*}}
// ... loop body ...
   *i = *ic * 0.5;
// Float multiply and save result.
// CHECK: [[MULR:%.+]] = fmul double {{%.+}}, 5.000000e-01
// CHECK-NEXT: invoke {{.+}} @{{.*}}IterDouble{{.*}}
// CHECK: store double [[MULR:%.+]], double* [[RESULT_ADDR:%.+]], !llvm.access.group
   ++ic;
//
// CHECK: [[IV2:%.+]] = load i32, i32* [[IT_OMP_IV]]{{.+}}!llvm.access.group
// CHECK-NEXT: [[ADD2:%.+]] = add nsw i32 [[IV2]], 1
// CHECK-NEXT: store i32 [[ADD2]], i32* [[IT_OMP_IV]]{{.+}}!llvm.access.group
// br label %{{.*}}, !llvm.loop ![[ITER_LOOP_ID]]
  }
// CHECK: [[IT_END]]:
// CHECK: ret void
}


// CHECK-LABEL: define {{.*void}} @{{.*}}collapsed{{.*}}
void collapsed(float *a, float *b, float *c, float *d) {
  int i; // outer loop counter
  unsigned j; // middle loop couter, leads to unsigned icmp in loop header.
  // k declared in the loop init below
  short l; // inner loop counter
// CHECK: store i32 0, i32* [[OMP_IV:[^,]+]]
//
  #pragma omp simd collapse(4)

// CHECK: [[IV:%.+]] = load i32, i32* [[OMP_IV]]{{.+}}!llvm.access.group
// CHECK-NEXT: [[CMP:%.+]] = icmp ult i32 [[IV]], 120
// CHECK-NEXT: br i1 [[CMP]], label %[[COLL1_BODY:[^,]+]], label %[[COLL1_END:[^,]+]]
  for (i = 1; i < 3; i++) // 2 iterations
    for (j = 2u; j < 5u; j++) //3 iterations
      for (int k = 3; k <= 6; k++) // 4 iterations
        for (l = 4; l < 9; ++l) // 5 iterations
        {
// CHECK: [[COLL1_BODY]]:
// Start of body: calculate i from index:
// CHECK: [[IV1:%.+]] = load i32, i32* [[OMP_IV]]{{.+}}!llvm.access.group
// Calculation of the loop counters values.
// CHECK: [[CALC_I_1:%.+]] = udiv i32 [[IV1]], 60
// CHECK-NEXT: [[CALC_I_1_MUL1:%.+]] = mul i32 [[CALC_I_1]], 1
// CHECK-NEXT: [[CALC_I_2:%.+]] = add i32 1, [[CALC_I_1_MUL1]]
// CHECK-NEXT: store i32 [[CALC_I_2]], i32* [[LC_I:.+]]

// CHECK: [[IV1_2:%.+]] = load i32, i32* [[OMP_IV]]{{.+}}!llvm.access.group
// CHECK: [[IV1_2_1:%.+]] = load i32, i32* [[OMP_IV]]{{.+}}!llvm.access.group
// CHECK-NEXT: [[CALC_J_1:%.+]] = udiv i32 [[IV1_2_1]], 60
// CHECK-NEXT: [[MUL_1:%.+]] = mul i32 [[CALC_J_1]], 60
// CHECK-NEXT: [[SUB_3:%.+]] = sub i32 [[IV1_2]], [[MUL_1]]
// CHECK-NEXT: [[CALC_J_2:%.+]] = udiv i32 [[SUB_3]], 20
// CHECK-NEXT: [[CALC_J_2_MUL1:%.+]] = mul i32 [[CALC_J_2]], 1
// CHECK-NEXT: [[CALC_J_3:%.+]] = add i32 2, [[CALC_J_2_MUL1]]
// CHECK-NEXT: store i32 [[CALC_J_3]], i32* [[LC_J:.+]]

// CHECK: [[IV1_3:%.+]] = load i32, i32* [[OMP_IV]]{{.+}}!llvm.access.group
// CHECK: [[IV1_3_1:%.+]] = load i32, i32* [[OMP_IV]]{{.+}}!llvm.access.group
// CHECK-NEXT: [[DIV_1:%.+]] = udiv i32 [[IV1_3_1]], 60
// CHECK-NEXT: [[MUL_2:%.+]] = mul i32 [[DIV_1]], 60
// CHECK-NEXT: [[ADD_3:%.+]] = sub i32 [[IV1_3]], [[MUL_2]]

// CHECK: [[IV1_4:%.+]] = load i32, i32* [[OMP_IV]]
// CHECK: [[IV1_4_1:%.+]] = load i32, i32* [[OMP_IV]]
// CHECK-NEXT: [[DIV_2:%.+]] = udiv i32 [[IV1_4_1]], 60
// CHECK-NEXT: [[MUL_3:%.+]] = mul i32 [[DIV_2]], 60
// CHECK-NEXT: [[SUB_6:%.+]] = sub i32 [[IV1_4]], [[MUL_3]]
// CHECK-NEXT: [[DIV_3:%.+]] = udiv i32 [[SUB_6]], 20
// CHECK-NEXT: [[MUL_4:%.+]] = mul i32 [[DIV_3]], 20
// CHECK-NEXT: [[ADD_5:%.+]] = sub i32 [[ADD_3]], [[MUL_4]]
// CHECK-NEXT: [[DIV_4:%.+]] = udiv i32 [[ADD_5]], 5
// CHECK-NEXT: [[MUL_5:%.+]] = mul i32 [[DIV_4]], 1
// CHECK-NEXT: [[ADD_6:%.+]] = add i32 3, [[MUL_5]]
// CHECK-NEXT: store i32 [[ADD_6]], i32* [[LC_K:.+]]

// CHECK: [[IV1_5:%.+]] = load i32, i32* [[OMP_IV]]{{.+}}!llvm.access.group
// CHECK: [[IV1_5_1:%.+]] = load i32, i32* [[OMP_IV]]{{.+}}!llvm.access.group
// CHECK-NEXT: [[DIV_5:%.+]] = udiv i32 [[IV1_5_1]], 60
// CHECK-NEXT: [[MUL_6:%.+]] = mul i32 [[DIV_5]], 60
// CHECK-NEXT: [[SUB_7:%.+]] = sub i32 [[IV1_5]], [[MUL_6]]

// CHECK: [[IV1_6:%.+]] = load i32, i32* [[OMP_IV]]
// CHECK: [[IV1_6_1:%.+]] = load i32, i32* [[OMP_IV]]
// CHECK-NEXT: [[DIV_6:%.+]] = udiv i32 [[IV1_6_1]], 60
// CHECK-NEXT: [[MUL_7:%.+]] = mul i32 [[DIV_6]], 60
// CHECK-NEXT: [[SUB_10:%.+]] = sub i32 [[IV1_6]], [[MUL_7]]
// CHECK-NEXT: [[DIV_7:%.+]] = udiv i32 [[SUB_10]], 20
// CHECK-NEXT: [[MUL_8:%.+]] = mul i32 [[DIV_7]], 20
// CHECK-NEXT: [[SUB_11:%.+]] = sub i32 [[SUB_7]], [[MUL_8]]

// CHECK: [[IV1_7:%.+]] = load i32, i32* [[OMP_IV]]
// CHECK: [[IV1_7_1:%.+]] = load i32, i32* [[OMP_IV]]
// CHECK-NEXT: [[DIV_8:%.+]] = udiv i32 [[IV1_7_1]], 60
// CHECK-NEXT: [[MUL_9:%.+]] = mul i32 [[DIV_8]], 60
// CHECK-NEXT: [[SUB_12:%.+]] = sub i32 [[IV1_7]], [[MUL_9]]

// CHECK: [[IV1_8:%.+]] = load i32, i32* [[OMP_IV]]
// CHECK: [[IV1_8_1:%.+]] = load i32, i32* [[OMP_IV]]
// CHECK-NEXT: [[DIV_3:%.+]] = udiv i32 [[IV1_8_1]], 60
// CHECK-NEXT: [[MUL_4:%.+]] = mul i32 [[DIV_3]], 60
// CHECK-NEXT: [[SUB_7:%.+]] = sub i32 [[IV1_8]], [[MUL_4]]
// CHECK-NEXT: [[DIV_4:%.+]] = udiv i32 [[SUB_7]], 20
// CHECK-NEXT: [[MUL_5:%.+]] = mul i32 [[DIV_4]], 20
// CHECK-NEXT: [[SUB_8:%.+]] = sub i32 [[SUB_12]], [[MUL_5]]
// CHECK-NEXT: [[DIV_5:%.+]] = udiv i32 [[SUB_8]], 5
// CHECK-NEXT: [[MUL_6:%.+]] = mul i32 [[DIV_5]], 5
// CHECK-NEXT: [[SUB_9:%.+]] = sub i32 [[SUB_11]], [[MUL_6]]
// CHECK-NEXT: [[MUL_6:%.+]] = mul i32 [[SUB_9]], 1
// CHECK-NEXT: [[CALC_L_2:%.+]] = add i32 4, [[MUL_6]]
// CHECK-NEXT: [[CALC_L_3:%.+]] = trunc i32 [[CALC_L_2]] to i16
// CHECK-NEXT: store i16 [[CALC_L_3]], i16* [[LC_L:.+]]
// ... loop body ...
// End of body: store into a[i]:
// CHECK: store float [[RESULT:%.+]], float* [[RESULT_ADDR:%.+]]{{.+}}!llvm.access.group
    float res = b[j] * c[k];
    a[i] = res * d[l];
// CHECK: [[IV2:%.+]] = load i32, i32* [[OMP_IV]]{{.*}}!llvm.access.group
// CHECK-NEXT: [[ADD2:%.+]] = add i32 [[IV2]], 1
// CHECK-NEXT: store i32 [[ADD2]], i32* [[OMP_IV]]{{.*}}!llvm.access.group
// br label %{{[^,]+}}, !llvm.loop ![[COLL1_LOOP_ID]]
// CHECK: [[COLL1_END]]:
  }
// i,j,l are updated; k is not updated.
// CHECK: store i32 3, i32*
// CHECK-NEXT: store i32 5, i32*
// CHECK-NEXT: store i32 7, i32*
// CHECK-NEXT: store i16 9, i16*
// CHECK: ret void
}

extern char foo();
extern double globalfloat;

// CHECK-LABEL: define {{.*void}} @{{.*}}widened{{.*}}
void widened(float *a, float *b, float *c, float *d) {
  int i; // outer loop counter
  short j; // inner loop counter
  globalfloat = 1.0;
  int localint = 1;
// CHECK: store double {{.+}}, double* [[GLOBALFLOAT:@.+]]
// Counter is widened to 64 bits.
// CHECK: store i64 0, i64* [[OMP_IV:[^,]+]]
//
  #pragma omp simd collapse(2) private(globalfloat, localint)

// CHECK: [[IV:%.+]] = load i64, i64* [[OMP_IV]]{{.+}}!llvm.access.group
// CHECK-NEXT: [[LI:%.+]] = load i64, i64* [[OMP_LI:%[^,]+]]{{.+}}!llvm.access.group
// CHECK-NEXT: [[NUMIT:%.+]] = add nsw i64 [[LI]], 1
// CHECK-NEXT: [[CMP:%.+]] = icmp slt i64 [[IV]], [[NUMIT]]
// CHECK-NEXT: br i1 [[CMP]], label %[[WIDE1_BODY:[^,]+]], label %[[WIDE1_END:[^,]+]]
  for (i = 1; i < 3; i++) // 2 iterations
    for (j = 0; j < foo(); j++) // foo() iterations
  {
// CHECK: [[WIDE1_BODY]]:
// Start of body: calculate i from index:
// CHECK: [[IV1:%.+]] = load i64, i64* [[OMP_IV]]{{.+}}!llvm.access.group
// Calculation of the loop counters values...
// CHECK: store i32 {{[^,]+}}, i32* [[LC_I:.+]]
// CHECK: [[IV1_2:%.+]] = load i64, i64* [[OMP_IV]]{{.+}}!llvm.access.group
// CHECK: store i16 {{[^,]+}}, i16* [[LC_J:.+]]
// ... loop body ...
//
// Here we expect store into private double var, not global
// CHECK-NOT: store double {{.+}}, double* [[GLOBALFLOAT]]
    globalfloat = (float)j/i;
    float res = b[j] * c[j];
// Store into a[i]:
// CHECK: store float [[RESULT:%.+]], float* [[RESULT_ADDR:%.+]]{{.+}}!llvm.access.group
    a[i] = res * d[i];
// Then there's a store into private var localint:
// CHECK: store i32 {{.+}}, i32* [[LOCALINT:%[^,]+]]{{.+}}!llvm.access.group
    localint = (int)j;
// CHECK: [[IV2:%.+]] = load i64, i64* [[OMP_IV]]{{.*}}!llvm.access.group
// CHECK-NEXT: [[ADD2:%.+]] = add nsw i64 [[IV2]], 1
// CHECK-NEXT: store i64 [[ADD2]], i64* [[OMP_IV]]{{.*}}!llvm.access.group
//
// br label %{{[^,]+}}, !llvm.loop ![[WIDE1_LOOP_ID]]
// CHECK: [[WIDE1_END]]:
  }
// i,j are updated.
// CHECK: store i32 3, i32* [[I:%[^,]+]]
// CHECK: store i16
//
// Here we expect store into original localint, not its privatized version.
// CHECK-NOT: store i32 {{.+}}, i32* [[LOCALINT]]
  localint = (int)j;
// CHECK: ret void
}

// CHECK-LABEL: define {{.*void}} @{{.*}}linear{{.*}}(float* noundef {{.+}})
void linear(float *a) {
  // CHECK: [[VAL_ADDR:%.+]] = alloca i64,
  // CHECK: [[K_ADDR:%.+]] = alloca i64*,
  long long val = 0;
  long long &k = val;

  #pragma omp simd linear(k : 3)
// CHECK: store i64* [[VAL_ADDR]], i64** [[K_ADDR]],
// CHECK: [[VAL_REF:%.+]] = load i64*, i64** [[K_ADDR]],
// CHECK: store i64* [[VAL_REF]], i64** [[K_ADDR_REF:%.+]],
// CHECK: store i32 0, i32* [[OMP_IV:%[^,]+]]
// CHECK: [[K_REF:%.+]] = load i64*, i64** [[K_ADDR_REF]],
// CHECK: [[K0LOAD:%.+]] = load i64, i64* [[K_REF]]
// CHECK-NEXT: store i64 [[K0LOAD]], i64* [[LIN0:%[^,]+]]

// CHECK: [[IV:%.+]] = load i32, i32* [[OMP_IV]]{{.*}}!llvm.access.group
// CHECK-NEXT: [[CMP2:%.+]] = icmp slt i32 [[IV]], 9
// CHECK-NEXT: br i1 [[CMP2]], label %[[SIMPLE_LOOP_BODY:.+]], label %[[SIMPLE_LOOP_END:[^,]+]]
  for (int i = 10; i > 1; i--) {
// CHECK: [[SIMPLE_LOOP_BODY]]:
// Start of body: calculate i from IV:
// CHECK: [[IV_0:%.+]] = load i32, i32* [[OMP_IV]]{{.*}}!llvm.access.group
// FIXME: It is interesting, why the following "mul 1" was not constant folded?
// CHECK-NEXT: [[IV_1:%.+]] = mul nsw i32 [[IV_0]], 1
// CHECK-NEXT: [[LC_I_1:%.+]] = sub nsw i32 10, [[IV_1]]
// CHECK-NEXT: store i32 [[LC_I_1]], i32* {{.+}}, !llvm.access.group
//
// CHECK-NEXT: [[LIN0_1:%.+]] = load i64, i64* [[LIN0]]{{.*}}!llvm.access.group
// CHECK-NEXT: [[IV_2:%.+]] = load i32, i32* [[OMP_IV]]{{.*}}!llvm.access.group
// CHECK-NEXT: [[LIN_MUL1:%.+]] = mul nsw i32 [[IV_2]], 3
// CHECK-NEXT: [[LIN_EXT1:%.+]] = sext i32 [[LIN_MUL1]] to i64
// CHECK-NEXT: [[LIN_ADD1:%.+]] = add nsw i64 [[LIN0_1]], [[LIN_EXT1]]
// Update of the privatized version of linear variable!
// CHECK-NEXT: store i64 [[LIN_ADD1]], i64* [[K_PRIVATIZED:%[^,]+]]
    a[k]++;
    k = k + 3;
// CHECK: [[IV_2:%.+]] = load i32, i32* [[OMP_IV]]{{.*}}!llvm.access.group
// CHECK-NEXT: [[ADD2_2:%.+]] = add nsw i32 [[IV_2]], 1
// CHECK-NEXT: store i32 [[ADD2_2]], i32* [[OMP_IV]]{{.*}}!llvm.access.group
// br label {{.+}}, !llvm.loop ![[SIMPLE_LOOP_ID]]
  }
// CHECK: [[SIMPLE_LOOP_END]]:
//
// Update linear vars after loop, as the loop was operating on a private version.
// CHECK: [[K_REF:%.+]] = load i64*, i64** [[K_ADDR_REF]],
// CHECK: store i64* [[K_REF]], i64** [[K_PRIV_REF:%.+]],
// CHECK: [[LIN0_2:%.+]] = load i64, i64* [[K_PRIVATIZED]]
// CHECK-NEXT: [[K_REF:%.+]] = load i64*, i64** [[K_PRIV_REF]],
// CHECK-NEXT: store i64 [[LIN0_2]], i64* [[K_REF]]
//

  #pragma omp simd linear(val(k) : 3)
// CHECK: [[VAL_REF:%.+]] = load i64*, i64** [[K_ADDR]],
// CHECK: store i64* [[VAL_REF]], i64** [[K_ADDR_REF:%.+]],
// CHECK: store i32 0, i32* [[OMP_IV:%[^,]+]]
// CHECK: [[K_REF:%.+]] = load i64*, i64** [[K_ADDR_REF]],
// CHECK: [[K0LOAD:%.+]] = load i64, i64* [[K_REF]]
// CHECK-NEXT: store i64 [[K0LOAD]], i64* [[LIN0:%[^,]+]]

// CHECK: [[IV:%.+]] = load i32, i32* [[OMP_IV]]{{.*}}!llvm.access.group
// CHECK-NEXT: [[CMP2:%.+]] = icmp slt i32 [[IV]], 9
// CHECK-NEXT: br i1 [[CMP2]], label %[[SIMPLE_LOOP_BODY:.+]], label %[[SIMPLE_LOOP_END:[^,]+]]
  for (int i = 10; i > 1; i--) {
// CHECK: [[SIMPLE_LOOP_BODY]]:
// Start of body: calculate i from IV:
// CHECK: [[IV_0:%.+]] = load i32, i32* [[OMP_IV]]{{.*}}!llvm.access.group
// FIXME: It is interesting, why the following "mul 1" was not constant folded?
// CHECK-NEXT: [[IV_1:%.+]] = mul nsw i32 [[IV_0]], 1
// CHECK-NEXT: [[LC_I_1:%.+]] = sub nsw i32 10, [[IV_1]]
// CHECK-NEXT: store i32 [[LC_I_1]], i32* {{.+}}, !llvm.access.group
//
// CHECK-NEXT: [[LIN0_1:%.+]] = load i64, i64* [[LIN0]]{{.*}}!llvm.access.group
// CHECK-NEXT: [[IV_2:%.+]] = load i32, i32* [[OMP_IV]]{{.*}}!llvm.access.group
// CHECK-NEXT: [[LIN_MUL1:%.+]] = mul nsw i32 [[IV_2]], 3
// CHECK-NEXT: [[LIN_EXT1:%.+]] = sext i32 [[LIN_MUL1]] to i64
// CHECK-NEXT: [[LIN_ADD1:%.+]] = add nsw i64 [[LIN0_1]], [[LIN_EXT1]]
// Update of the privatized version of linear variable!
// CHECK-NEXT: store i64 [[LIN_ADD1]], i64* [[K_PRIVATIZED:%[^,]+]]
    a[k]++;
    k = k + 3;
// CHECK: [[IV_2:%.+]] = load i32, i32* [[OMP_IV]]{{.*}}!llvm.access.group
// CHECK-NEXT: [[ADD2_2:%.+]] = add nsw i32 [[IV_2]], 1
// CHECK-NEXT: store i32 [[ADD2_2]], i32* [[OMP_IV]]{{.*}}!llvm.access.group
// br label {{.+}}, !llvm.loop ![[SIMPLE_LOOP_ID]]
  }
// CHECK: [[SIMPLE_LOOP_END]]:
//
// Update linear vars after loop, as the loop was operating on a private version.
// CHECK: [[K_REF:%.+]] = load i64*, i64** [[K_ADDR_REF]],
// CHECK: store i64* [[K_REF]], i64** [[K_PRIV_REF:%.+]],
// CHECK: [[LIN0_2:%.+]] = load i64, i64* [[K_PRIVATIZED]]
// CHECK-NEXT: [[K_REF:%.+]] = load i64*, i64** [[K_PRIV_REF]],
// CHECK-NEXT: store i64 [[LIN0_2]], i64* [[K_REF]]
//
  #pragma omp simd linear(uval(k) : 3)
// CHECK: store i32 0, i32* [[OMP_IV:%[^,]+]]
// CHECK: [[K0LOAD:%.+]] = load i64, i64* [[VAL_ADDR]]
// CHECK-NEXT: store i64 [[K0LOAD]], i64* [[LIN0:%[^,]+]]

// CHECK: [[IV:%.+]] = load i32, i32* [[OMP_IV]]{{.*}}!llvm.access.group
// CHECK-NEXT: [[CMP2:%.+]] = icmp slt i32 [[IV]], 9
// CHECK-NEXT: br i1 [[CMP2]], label %[[SIMPLE_LOOP_BODY:.+]], label %[[SIMPLE_LOOP_END:[^,]+]]
  for (int i = 10; i > 1; i--) {
// CHECK: [[SIMPLE_LOOP_BODY]]:
// Start of body: calculate i from IV:
// CHECK: [[IV_0:%.+]] = load i32, i32* [[OMP_IV]]{{.*}}!llvm.access.group
// FIXME: It is interesting, why the following "mul 1" was not constant folded?
// CHECK-NEXT: [[IV_1:%.+]] = mul nsw i32 [[IV_0]], 1
// CHECK-NEXT: [[LC_I_1:%.+]] = sub nsw i32 10, [[IV_1]]
// CHECK-NEXT: store i32 [[LC_I_1]], i32* {{.+}}, !llvm.access.group
//
// CHECK-NEXT: [[LIN0_1:%.+]] = load i64, i64* [[LIN0]]{{.*}}!llvm.access.group
// CHECK-NEXT: [[IV_2:%.+]] = load i32, i32* [[OMP_IV]]{{.*}}!llvm.access.group
// CHECK-NEXT: [[LIN_MUL1:%.+]] = mul nsw i32 [[IV_2]], 3
// CHECK-NEXT: [[LIN_EXT1:%.+]] = sext i32 [[LIN_MUL1]] to i64
// CHECK-NEXT: [[LIN_ADD1:%.+]] = add nsw i64 [[LIN0_1]], [[LIN_EXT1]]
// Update of the privatized version of linear variable!
// CHECK-NEXT: store i64 [[LIN_ADD1]], i64* [[K_PRIVATIZED:%[^,]+]]
    a[k]++;
    k = k + 3;
// CHECK: [[IV_2:%.+]] = load i32, i32* [[OMP_IV]]{{.*}}!llvm.access.group
// CHECK-NEXT: [[ADD2_2:%.+]] = add nsw i32 [[IV_2]], 1
// CHECK-NEXT: store i32 [[ADD2_2]], i32* [[OMP_IV]]{{.*}}!llvm.access.group
// br label {{.+}}, !llvm.loop ![[SIMPLE_LOOP_ID]]
  }
// CHECK: [[SIMPLE_LOOP_END]]:
//
// Update linear vars after loop, as the loop was operating on a private version.
// CHECK: [[LIN0_2:%.+]] = load i64, i64* [[K_PRIVATIZED]]
// CHECK-NEXT: store i64 [[LIN0_2]], i64* [[VAL_ADDR]]
//
}

#ifdef OMP5
// OMP50-LABEL: inner_simd
void inner_simd() {
  double a, b;
#pragma omp simd nontemporal(a)
  for (int i = 0; i < 10; ++i) {
#pragma omp simd nontemporal(b)
    for (int k = 0; k < 10; ++k) {
      // OMP50: load double,{{.*}}!nontemporal
      // OMP50: store double{{.*}}!nontemporal
      a = b;
    }
    // OMP50-NOT: load double,{{.*}}!nontemporal
    // OMP50: load double,
    // OMP50: store double{{.*}}!nontemporal
    a = b;
  }
}

extern struct T t;
struct Base {
  float a;
};
struct T : public Base {
  void foo() {
#pragma omp simd nontemporal(Base::a)
    for (int i = 0; i < 10; ++i) {
    // OMP50: store float{{.*}}!nontemporal
    // OMP50-NOT: nontemporal
    // OMP50-NEXT: store float
      Base::a = 0;
      t.a = 0;
    }
  }
} t;

void bartfoo() {
  t.foo();
}

#endif // OMP5
// TERM_DEBUG-LABEL: bar
int bar() { extern void mayThrow(); mayThrow(); return 0; };

// TERM_DEBUG-LABEL: parallel_simd
void parallel_simd(float *a) {
#pragma omp parallel
#pragma omp simd
  // TERM_DEBUG-NOT: __kmpc_global_thread_num
  // TERM_DEBUG:     invoke noundef i32 {{.*}}bar{{.*}}()
  // TERM_DEBUG:     unwind label %[[TERM_LPAD:[^,]+]],
  // TERM_DEBUG-NOT: __kmpc_global_thread_num
  // TERM_DEBUG:     [[TERM_LPAD]]
  // TERM_DEBUG:     call void @__clang_call_terminate
  // TERM_DEBUG:     unreachable
  for (unsigned i = 131071; i <= 2147483647; i += 127)
    a[i] += bar();
}
// TERM_DEBUG: !{{[0-9]+}} = !DILocation(line: [[@LINE-11]],

// CHECK-LABEL: S8
// CHECK-DAG: call void @llvm.assume(i1
// CHECK-DAG: call void @llvm.assume(i1
// CHECK-DAG: call void @llvm.assume(i1
// CHECK-DAG: call void @llvm.assume(i1
struct SS {
  SS(): a(0) {}
  SS(int v) : a(v) {}
  int a;
  typedef int type;
};

template <typename T>
class S7 : public T {
protected:
  T *a;
  T b[2];
  S7() : a(0) {}

public:
  S7(typename T::type &v) : a((T*)&v) {
#pragma omp simd aligned(a)
    for (int k = 0; k < a->a; ++k)
      ++this->a->a;
#pragma omp simd aligned(this->b : 8)
    for (int k = 0; k < a->a; ++k)
      ++a->a;
  }
};

class S8 : private IterDouble, public S7<SS> {
  S8() {}

public:
  S8(int v) : S7<SS>(v){
#pragma omp parallel private(a)
#pragma omp simd aligned(S7<SS>::a)
    for (int k = 0; k < a->a; ++k)
      ++this->a->a;
#pragma omp parallel shared(b)
#pragma omp simd aligned(this->b: 4)
    for (int k = 0; k < a->a; ++k)
      ++a->a;
  }
};
S8 s8(0);

// TERM_DEBUG-NOT: line: 0,
// TERM_DEBUG: distinct !DISubprogram(linkageName: "_GLOBAL__sub_I_simd_codegen.cpp",
// OMP50-DAG: ![[NOVECT:.+]] = !{!"llvm.loop.vectorize.enable", i1 false}
// OMP50-DAG: ![[DISABLE_VECT]] = distinct !{{.*}}![[NOVECT]]{{[,}]}}
#endif // HEADER

