// RUN: %clang_cc1 -Wno-return-type -Wno-unused-value -emit-llvm %s -w -o - | FileCheck %s

// CHECK: @i = {{(dso_local )?}}global [[INT:i[0-9]+]] 0
volatile int i, j, k;
volatile int ar[5];
volatile char c;
// CHECK: @ci = {{(dso_local )?}}global [[CINT:.*]] zeroinitializer
volatile _Complex int ci;
volatile struct S {
#ifdef __cplusplus
  void operator =(volatile struct S&o) volatile;
#endif
  int i;
} a, b;

//void operator =(volatile struct S&o1, volatile struct S&o2) volatile;
int printf(const char *, ...);


// Note that these test results are very much specific to C!
// Assignments in C++ yield l-values, not r-values, and the situations
// that do implicit lvalue-to-rvalue conversion are substantially
// reduced.

// CHECK-LABEL: define {{.*}}void @test()
void test(void) {
  // CHECK: load volatile [[INT]], ptr @i
  i;
  // CHECK-NEXT: load volatile [[INT]], ptr @ci, align 4
  // CHECK-NEXT: load volatile [[INT]], ptr getelementptr inbounds ([[CINT]], ptr @ci, i32 0, i32 1), align 4
  // CHECK-NEXT: sitofp [[INT]]
  (float)(ci);
  // CHECK-NEXT: load volatile [[INT]], ptr @ci, align 4
  // CHECK-NEXT: load volatile [[INT]], ptr getelementptr inbounds ([[CINT]], ptr @ci, i32 0, i32 1), align 4
  (void)ci;
  // CHECK-NEXT: memcpy
  (void)a;
  // CHECK-NEXT: [[R:%.*]] = load volatile [[INT]], ptr @ci, align 4
  // CHECK-NEXT: [[I:%.*]] = load volatile [[INT]], ptr getelementptr inbounds ([[CINT]], ptr @ci, i32 0, i32 1), align 4
  // CHECK-NEXT: store volatile [[INT]] [[R]], ptr @ci, align 4
  // CHECK-NEXT: store volatile [[INT]] [[I]], ptr getelementptr inbounds ([[CINT]], ptr @ci, i32 0, i32 1), align 4
  (void)(ci=ci);
  // CHECK-NEXT: [[T:%.*]] = load volatile [[INT]], ptr @j
  // CHECK-NEXT: store volatile [[INT]] [[T]], ptr @i
  (void)(i=j);
  // CHECK-NEXT: [[R1:%.*]] = load volatile [[INT]], ptr @ci, align 4
  // CHECK-NEXT: [[I1:%.*]] = load volatile [[INT]], ptr getelementptr inbounds ([[CINT]], ptr @ci, i32 0, i32 1), align 4
  // CHECK-NEXT: [[R2:%.*]] = load volatile [[INT]], ptr @ci, align 4
  // CHECK-NEXT: [[I2:%.*]] = load volatile [[INT]], ptr getelementptr inbounds ([[CINT]], ptr @ci, i32 0, i32 1), align 4
  // Not sure why they're ordered this way.
  // CHECK-NEXT: [[R:%.*]] = add [[INT]] [[R2]], [[R1]]
  // CHECK-NEXT: [[I:%.*]] = add [[INT]] [[I2]], [[I1]]
  // CHECK-NEXT: store volatile [[INT]] [[R]], ptr @ci, align 4
  // CHECK-NEXT: store volatile [[INT]] [[I]], ptr getelementptr inbounds ([[CINT]], ptr @ci, i32 0, i32 1), align 4
  ci+=ci;

  // CHECK-NEXT: [[R1:%.*]] = load volatile [[INT]], ptr @ci, align 4
  // CHECK-NEXT: [[I1:%.*]] = load volatile [[INT]], ptr getelementptr inbounds ([[CINT]], ptr @ci, i32 0, i32 1), align 4
  // CHECK-NEXT: [[R2:%.*]] = load volatile [[INT]], ptr @ci, align 4
  // CHECK-NEXT: [[I2:%.*]] = load volatile [[INT]], ptr getelementptr inbounds ([[CINT]], ptr @ci, i32 0, i32 1), align 4
  // CHECK-NEXT: [[R:%.*]] = add [[INT]] [[R2]], [[R1]]
  // CHECK-NEXT: [[I:%.*]] = add [[INT]] [[I2]], [[I1]]
  // CHECK-NEXT: store volatile [[INT]] [[R]], ptr @ci, align 4
  // CHECK-NEXT: store volatile [[INT]] [[I]], ptr getelementptr inbounds ([[CINT]], ptr @ci, i32 0, i32 1), align 4
  // CHECK-NEXT: [[R2:%.*]] = load volatile [[INT]], ptr @ci, align 4
  // CHECK-NEXT: [[I2:%.*]] = load volatile [[INT]], ptr getelementptr inbounds ([[CINT]], ptr @ci, i32 0, i32 1), align 4
  // These additions can be elided
  // CHECK-NEXT: add [[INT]] [[R]], [[R2]]
  // CHECK-NEXT: add [[INT]] [[I]], [[I2]]
  (ci += ci) + ci;
  // CHECK-NEXT: call void asm
  asm("nop");
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: add nsw [[INT]]
  // CHECK-NEXT: store volatile
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: add nsw [[INT]]
  (i += j) + k;
  // CHECK-NEXT: call void asm
  asm("nop");
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: add nsw [[INT]]
  // CHECK-NEXT: store volatile
  // CHECK-NEXT: add nsw [[INT]]
  (i += j) + 1;
  // CHECK-NEXT: call void asm
  asm("nop");
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: add [[INT]]
  // CHECK-NEXT: add [[INT]]
  ci+ci;

  // CHECK-NEXT: load volatile
  __real i;
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: load volatile
  +ci;
  // CHECK-NEXT: call void asm
  asm("nop");
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: store volatile
  (void)(i=i);
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: store volatile
  // CHECK-NEXT: sitofp
  (float)(i=i);
  // CHECK-NEXT: load volatile
  (void)i;
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: store volatile
  i=i;
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: store volatile
  // CHECK-NEXT: store volatile
  i=i=i;
#ifndef __cplusplus
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: store volatile
  (void)__builtin_choose_expr(0, i=i, j=j);
#endif
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: icmp
  // CHECK-NEXT: br i1
  // CHECK: load volatile
  // CHECK-NEXT: store volatile
  // CHECK-NEXT: br label
  // CHECK: load volatile
  // CHECK-NEXT: store volatile
  // CHECK-NEXT: br label
  k ? (i=i) : (j=j);
  // CHECK: phi
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: store volatile
  (void)(i,(i=i));
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: store volatile
  // CHECK-NEXT: load volatile
  i=i,i;
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: store volatile
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: store volatile
  (i=j,k=j);
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: store volatile
  // CHECK-NEXT: load volatile
  (i=j,k);
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: load volatile
  (i,j);
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: trunc
  // CHECK-NEXT: store volatile
  // CHECK-NEXT: sext
  // CHECK-NEXT: store volatile
  i=c=k;
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: add nsw [[INT]]
  // CHECK-NEXT: store volatile
  i+=k;
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: load volatile
  ci;
#ifndef __cplusplus
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: load volatile
  (int)ci;
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: icmp ne
  // CHECK-NEXT: icmp ne
  // CHECK-NEXT: or i1
  (_Bool)ci;
#endif
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: store volatile
  // CHECK-NEXT: store volatile
  ci=ci;
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: store volatile
  // CHECK-NEXT: store volatile
  // CHECK-NEXT: store volatile
  // CHECK-NEXT: store volatile
  ci=ci=ci;
  // CHECK-NEXT: [[T:%.*]] = load volatile [[INT]], ptr getelementptr inbounds ([[CINT]], ptr @ci, i32 0, i32 1)
  // CHECK-NEXT: store volatile [[INT]] [[T]], ptr getelementptr inbounds ([[CINT]], ptr @ci, i32 0, i32 1)
  // CHECK-NEXT: store volatile [[INT]] [[T]], ptr getelementptr inbounds ([[CINT]], ptr @ci, i32 0, i32 1)
  __imag ci = __imag ci = __imag ci;
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: store volatile
  __real (i = j);
  // CHECK-NEXT: load volatile
  __imag i;
  
  // ============================================================
  // FIXME: Test cases we get wrong.

  // A use.  We load all of a into a copy of a, then load i.  gcc forgets to do
  // the assignment.
  // (a = a).i;

  // ============================================================
  // Test cases where we intentionally differ from gcc, due to suspected bugs in
  // gcc.

  // Not a use.  gcc forgets to do the assignment.
  // CHECK-NEXT: call void @llvm.memcpy{{.*}}, i1 true
  // CHECK-NEXT: call void @llvm.memcpy{{.*}}, i1 true
  ((a=a),a);

  // Not a use.  gcc gets this wrong, it doesn't emit the copy!  
  // (void)(a=a);

  // Not a use.  gcc got this wrong in 4.2 and omitted the side effects
  // entirely, but it is fixed in 4.4.0.
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: store volatile
  __imag (i = j);

#ifndef __cplusplus
  // A use of the real part
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: store volatile
  // CHECK-NEXT: store volatile
  // CHECK-NEXT: sitofp
  (float)(ci=ci);
  // Not a use, bug?  gcc treats this as not a use, that's probably a bug due to
  // tree folding ignoring volatile.
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: store volatile
  // CHECK-NEXT: store volatile
  (int)(ci=ci);
#endif

  // A use.
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: store volatile
  // CHECK-NEXT: sitofp
  (float)(i=i);
  // A use.  gcc treats this as not a use, that's probably a bug due to tree
  // folding ignoring volatile.
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: store volatile
  (int)(i=i);

  // A use.
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: store volatile
  // CHECK-NEXT: sub
  -(i=j);
  // A use.  gcc treats this a not a use, that's probably a bug due to tree
  // folding ignoring volatile.
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: store volatile
  +(i=k);

  // A use. gcc treats this a not a use, that's probably a bug due to tree
  // folding ignoring volatile.
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: store volatile
  // CHECK-NEXT: store volatile
  __real (ci=ci);

  // A use.
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: add
  i + 0;
  // A use.
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: store volatile
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: add
  (i=j) + i;
  // A use.  gcc treats this as not a use, that's probably a bug due to tree
  // folding ignoring volatile.
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: store volatile
  // CHECK-NEXT: add
  (i=j) + 0;

#ifdef __cplusplus
  (i,j)=k;
  (j=k,i)=i;
  struct { int x; } s, s1;
  printf("s is at %p\n", &s);
  printf("s is at %p\n", &(s = s1));
  printf("s.x is at %p\n", &((s = s1).x));
#endif
}

extern volatile enum X x;
// CHECK-LABEL: define {{.*}}void @test1()
void test1(void) {
  extern void test1_helper(void);
  test1_helper();
  // CHECK: call {{.*}}void @test1_helper()
  // CHECK-NEXT: ret void
  x;
  (void) x;
  return x;
}

// CHECK: define {{.*}} @test2()
int test2(void) {
  // CHECK: load volatile i32, ptr
  // CHECK-NEXT: load volatile i32, ptr
  // CHECK-NEXT: load volatile i32, ptr
  // CHECK-NEXT: add i32
  // CHECK-NEXT: add i32
  // CHECK-NEXT: store volatile i32
  // CHECK-NEXT: ret i32
  return i += ci;
}
