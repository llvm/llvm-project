// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

typedef union {
  _Complex float cf;
  long long ll;
} ucf;

// CHECK-LABEL: define{{.*}} void @foo(
// CHECK:         [[DEC:%.*]] = fadd float [[OLD_R:%.*]], -1.000000e+00
// CHECK-NEXT:    [[CF_R:%.*]] = getelementptr inbounds nuw { float, float }, ptr %cf, i32 0, i32 0
// CHECK-NEXT:    [[CF_I:%.*]] = getelementptr inbounds nuw { float, float }, ptr %cf, i32 0, i32 1
// CHECK-NEXT:    store float [[DEC]], ptr [[CF_R]], align 4
// CHECK-NEXT:    store float [[OLD_I:%.*]], ptr [[CF_I]], align 4
// CHECK-NEXT:    [[TMP_R:%.*]] = getelementptr inbounds nuw { float, float }, ptr [[TMP:%.*]], i32 0, i32 0
// CHECK-NEXT:    [[TMP_I:%.*]] = getelementptr inbounds nuw { float, float }, ptr [[TMP]], i32 0, i32 1
// CHECK-NEXT:    store float [[OLD_R]], ptr [[TMP_R]], align 4
// CHECK-NEXT:    store float [[OLD_I]], ptr [[TMP_I]], align 4
// CHECK-NEXT:    [[V:%.*]] = load i64, ptr [[TMP]], align 4
// CHECK-NEXT:    call i64 asm "", "=r,r,~{dirflag},~{fpsr},~{flags}"(i64 [[V]])
void foo(ucf *in, ucf *out, _Complex float r) {
  int i;
  ucf ucf1;
  _Complex float cf;

  ucf1.ll = in[i].ll;
  __asm("" : "=r"(cf) : "r"(ucf1.ll));
  cf *= r;
  __asm("" : "=r"(ucf1.ll) : "r"(cf--));
  out[i].ll = ucf1.ll;
}

// CHECK-LABEL: define{{.*}} i64 @lvalue(
// CHECK:         %cf = alloca { float, float }, align 4
// CHECK-NOT:     alloca { float, float }
// CHECK:         [[V:%.*]] = load i64, ptr %cf, align 4
// CHECK-NEXT:    call i64 asm "", "=r,r,~{dirflag},~{fpsr},~{flags}"(i64 [[V]])
long long lvalue(_Complex float cf) {
  long long x;
  __asm("" : "=r"(x) : "r"(cf));
  return x;
}

_Complex float getcf(void);

// CHECK-LABEL: define{{.*}} i64 @call(
// CHECK:         call <2 x float> @getcf()
// CHECK:         [[R:%.*]] = load float, ptr %coerce.realp, align 4
// CHECK:         [[I:%.*]] = load float, ptr %coerce.imagp, align 4
// CHECK-NEXT:    [[TMP_R:%.*]] = getelementptr inbounds nuw { float, float }, ptr [[TMP:%.*]], i32 0, i32 0
// CHECK-NEXT:    [[TMP_I:%.*]] = getelementptr inbounds nuw { float, float }, ptr [[TMP]], i32 0, i32 1
// CHECK-NEXT:    store float [[R]], ptr [[TMP_R]], align 4
// CHECK-NEXT:    store float [[I]], ptr [[TMP_I]], align 4
// CHECK-NEXT:    [[V:%.*]] = load i64, ptr [[TMP]], align 4
// CHECK-NEXT:    call i64 asm "", "=r,r,~{dirflag},~{fpsr},~{flags}"(i64 [[V]])
long long call(void) {
  long long x;
  __asm("" : "=r"(x) : "r"(getcf()));
  return x;
}

// CHECK-LABEL: define{{.*}} i64 @binop(
// CHECK:         [[R:%.*]] = fadd float
// CHECK-NEXT:    [[I:%.*]] = fadd float
// CHECK-NEXT:    [[TMP_R:%.*]] = getelementptr inbounds nuw { float, float }, ptr [[TMP:%.*]], i32 0, i32 0
// CHECK-NEXT:    [[TMP_I:%.*]] = getelementptr inbounds nuw { float, float }, ptr [[TMP]], i32 0, i32 1
// CHECK-NEXT:    store float [[R]], ptr [[TMP_R]], align 4
// CHECK-NEXT:    store float [[I]], ptr [[TMP_I]], align 4
// CHECK-NEXT:    [[V:%.*]] = load i64, ptr [[TMP]], align 4
// CHECK-NEXT:    call i64 asm "", "=r,r,~{dirflag},~{fpsr},~{flags}"(i64 [[V]])
long long binop(_Complex float a, _Complex float b) {
  long long x;
  __asm("" : "=r"(x) : "r"(a + b));
  return x;
}

// CHECK-LABEL: define{{.*}} void @other_rvalues(
// CHECK:         load atomic i64, ptr {{.*}} seq_cst, align 8
// CHECK:         call void asm sideeffect "", "r,r,r,r,r,~{dirflag},~{fpsr},~{flags}"(i64 {{.*}}, i64 {{.*}}, i64 {{.*}}, i64 {{.*}}, i64 {{.*}})
void other_rvalues(int c, float f, _Complex float a, _Complex float b,
                   _Atomic _Complex float *p) {
  __asm volatile("" : : "r"(-a), "r"(c ? a : b), "r"((_Complex float)f),
                 "r"(({ a; })), "r"(*p));
}

// CHECK-LABEL: define{{.*}} void @large_rm(
// CHECK:         [[R:%.*]] = fadd double
// CHECK-NEXT:    [[I:%.*]] = fadd double
// CHECK-NEXT:    [[TMP_R:%.*]] = getelementptr inbounds nuw { double, double }, ptr [[TMP:%.*]], i32 0, i32 0
// CHECK-NEXT:    [[TMP_I:%.*]] = getelementptr inbounds nuw { double, double }, ptr [[TMP]], i32 0, i32 1
// CHECK-NEXT:    store double [[R]], ptr [[TMP_R]], align 8
// CHECK-NEXT:    store double [[I]], ptr [[TMP_I]], align 8
// CHECK-NEXT:    call void asm sideeffect "", "*rm,~{dirflag},~{fpsr},~{flags}"(ptr elementtype({ double, double }) [[TMP]])
void large_rm(_Complex double a, _Complex double b) {
  __asm volatile("" : : "rm"(a + b));
}

struct S { int a, b; };
struct S get_s(void);

// CHECK-LABEL: define{{.*}} i64 @struct_call(
// CHECK:         [[CALL:%.*]] = call i64 @get_s()
// CHECK-NEXT:    store i64 [[CALL]], ptr [[TMP:%.*]], align 4
// CHECK-NEXT:    [[V:%.*]] = load i64, ptr [[TMP]], align 4
// CHECK-NEXT:    call i64 asm "", "=r,r,~{dirflag},~{fpsr},~{flags}"(i64 [[V]])
long long struct_call(void) {
  long long x;
  __asm("" : "=r"(x) : "r"(get_s()));
  return x;
}

// CHECK-LABEL: define{{.*}} i64 @struct_atomic_load(
// CHECK:         load atomic i64, ptr {{.*}} seq_cst, align 8
// CHECK:         call void @llvm.memcpy.p0.p0.i64(ptr align 4 [[TMP:%.*]], ptr align 8
// CHECK-NEXT:    [[V:%.*]] = load i64, ptr [[TMP]], align 4
// CHECK-NEXT:    call i64 asm "", "=r,r,~{dirflag},~{fpsr},~{flags}"(i64 [[V]])
long long struct_atomic_load(_Atomic(struct S) *p) {
  long long x;
  __asm("" : "=r"(x) : "r"(__c11_atomic_load(p, 5)));
  return x;
}
