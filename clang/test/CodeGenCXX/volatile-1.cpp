// RUN: %clang_cc1 -Wno-unused-value -emit-llvm %s -o - | FileCheck %s

// CHECK: @i = global [[INT:i[0-9]+]] 0
volatile int i, j, k;
volatile int ar[5];
volatile char c;
// CHECK: @ci = global [[CINT:.*]] zeroinitializer
volatile _Complex int ci;
volatile struct S {
#ifdef __cplusplus
  void operator =(volatile struct S&o) volatile;
#endif
  int i;
} a, b;

//void operator =(volatile struct S&o1, volatile struct S&o2) volatile;
int printf(const char *, ...);


// CHECK: define void @{{.*}}test
void test() {

  asm("nop"); // CHECK: call void asm

  // should not load
  i;

  (float)(ci);
  // CHECK-NEXT: load volatile [[INT]]* getelementptr inbounds ([[CINT]]* @ci, i32 0, i32 0)
  // CHECK-NEXT: load volatile [[INT]]* getelementptr inbounds ([[CINT]]* @ci, i32 0, i32 1)
  // CHECK-NEXT: sitofp [[INT]]

  // These are not uses in C++:
  //   [expr.static.cast]p6:
  //     The lvalue-to-rvalue . . . conversions are not applied to the expression.
  (void)ci;
  (void)a;

  (void)(ci=ci);
  // CHECK-NEXT: [[R:%.*]] = load volatile [[INT]]* getelementptr inbounds ([[CINT]]* @ci, i32 0, i32 0)
  // CHECK-NEXT: [[I:%.*]] = load volatile [[INT]]* getelementptr inbounds ([[CINT]]* @ci, i32 0, i32 1)
  // CHECK-NEXT: store volatile [[INT]] [[R]], [[INT]]* getelementptr inbounds ([[CINT]]* @ci, i32 0, i32 0)
  // CHECK-NEXT: store volatile [[INT]] [[I]], [[INT]]* getelementptr inbounds ([[CINT]]* @ci, i32 0, i32 1)

  (void)(i=j);
  // CHECK-NEXT: [[T:%.*]] = load volatile [[INT]]* @j
  // CHECK-NEXT: store volatile [[INT]] [[T]], [[INT]]* @i

  ci+=ci;
  // CHECK-NEXT: [[R1:%.*]] = load volatile [[INT]]* getelementptr inbounds ([[CINT]]* @ci, i32 0, i32 0)
  // CHECK-NEXT: [[I1:%.*]] = load volatile [[INT]]* getelementptr inbounds ([[CINT]]* @ci, i32 0, i32 1)
  // CHECK-NEXT: [[R2:%.*]] = load volatile [[INT]]* getelementptr inbounds ([[CINT]]* @ci, i32 0, i32 0)
  // CHECK-NEXT: [[I2:%.*]] = load volatile [[INT]]* getelementptr inbounds ([[CINT]]* @ci, i32 0, i32 1)
  // Not sure why they're ordered this way.
  // CHECK-NEXT: [[R:%.*]] = add [[INT]] [[R2]], [[R1]]
  // CHECK-NEXT: [[I:%.*]] = add [[INT]] [[I2]], [[I1]]
  // CHECK-NEXT: store volatile [[INT]] [[R]], [[INT]]* getelementptr inbounds ([[CINT]]* @ci, i32 0, i32 0)
  // CHECK-NEXT: store volatile [[INT]] [[I]], [[INT]]* getelementptr inbounds ([[CINT]]* @ci, i32 0, i32 1)

  // Note that C++ requires an extra load volatile over C from the LHS of the '+'.
  (ci += ci) + ci;
  // CHECK-NEXT: [[R1:%.*]] = load volatile [[INT]]* getelementptr inbounds ([[CINT]]* @ci, i32 0, i32 0)
  // CHECK-NEXT: [[I1:%.*]] = load volatile [[INT]]* getelementptr inbounds ([[CINT]]* @ci, i32 0, i32 1)
  // CHECK-NEXT: [[R2:%.*]] = load volatile [[INT]]* getelementptr inbounds ([[CINT]]* @ci, i32 0, i32 0)
  // CHECK-NEXT: [[I2:%.*]] = load volatile [[INT]]* getelementptr inbounds ([[CINT]]* @ci, i32 0, i32 1)
  // CHECK-NEXT: [[R:%.*]] = add [[INT]] [[R2]], [[R1]]
  // CHECK-NEXT: [[I:%.*]] = add [[INT]] [[I2]], [[I1]]
  // CHECK-NEXT: store volatile [[INT]] [[R]], [[INT]]* getelementptr inbounds ([[CINT]]* @ci, i32 0, i32 0)
  // CHECK-NEXT: store volatile [[INT]] [[I]], [[INT]]* getelementptr inbounds ([[CINT]]* @ci, i32 0, i32 1)
  // CHECK-NEXT: [[R1:%.*]] = load volatile [[INT]]* getelementptr inbounds ([[CINT]]* @ci, i32 0, i32 0)
  // CHECK-NEXT: [[I1:%.*]] = load volatile [[INT]]* getelementptr inbounds ([[CINT]]* @ci, i32 0, i32 1)
  // CHECK-NEXT: [[R2:%.*]] = load volatile [[INT]]* getelementptr inbounds ([[CINT]]* @ci, i32 0, i32 0)
  // CHECK-NEXT: [[I2:%.*]] = load volatile [[INT]]* getelementptr inbounds ([[CINT]]* @ci, i32 0, i32 1)
  // These additions can be elided.
  // CHECK-NEXT: add [[INT]] [[R1]], [[R2]]
  // CHECK-NEXT: add [[INT]] [[I1]], [[I2]]

  asm("nop"); // CHECK-NEXT: call void asm

  // Extra load volatile in C++.
  (i += j) + k;
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: add nsw [[INT]]
  // CHECK-NEXT: store volatile
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: add nsw [[INT]]

  asm("nop"); // CHECK-NEXT: call void asm

  // Extra load volatile in C++.
  (i += j) + 1;
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: add nsw [[INT]]
  // CHECK-NEXT: store volatile
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: add nsw [[INT]]

  asm("nop"); // CHECK-NEXT: call void asm

  ci+ci;
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: add [[INT]]
  // CHECK-NEXT: add [[INT]]

  __real i;

  +ci;
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: load volatile

  asm("nop"); // CHECK-NEXT: call void asm

  (void)(i=i);
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: store volatile

  (float)(i=i);
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: store volatile
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: sitofp

  (void)i;

  i=i;
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: store volatile

  // Extra load volatile in C++.
  i=i=i;
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: store volatile
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: store volatile

  (void)__builtin_choose_expr(0, i=i, j=j);
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: store volatile

  k ? (i=i) : (j=j);
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: icmp
  // CHECK-NEXT: br i1
  // CHECK: load volatile
  // CHECK-NEXT: store volatile
  // CHECK-NEXT: br label
  // CHECK: load volatile
  // CHECK-NEXT: store volatile
  // CHECK-NEXT: br label
  // CHECK:      phi

  (void)(i,(i=i));
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: store volatile

  i=i,k;
  // CHECK-NEXT: load volatile [[INT]]* @i
  // CHECK-NEXT: store volatile {{.*}}, [[INT]]* @i

  (i=j,k=j);
  // CHECK-NEXT: load volatile [[INT]]* @j
  // CHECK-NEXT: store volatile {{.*}}, [[INT]]* @i
  // CHECK-NEXT: load volatile [[INT]]* @j
  // CHECK-NEXT: store volatile {{.*}}, [[INT]]* @k

  (i=j,k);
  // CHECK-NEXT: load volatile [[INT]]* @j
  // CHECK-NEXT: store volatile {{.*}}, [[INT]]* @i

  (i,j);

  // Extra load in C++.
  i=c=k;
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: trunc
  // CHECK-NEXT: store volatile
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: sext
  // CHECK-NEXT: store volatile

  i+=k;
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: add nsw [[INT]]
  // CHECK-NEXT: store volatile

  ci;

  asm("nop"); // CHECK-NEXT: call void asm

  (int)ci;
  // CHECK-NEXT: load volatile {{.*}} @ci, i32 0, i32 0
  // CHECK-NEXT: load volatile {{.*}} @ci, i32 0, i32 1

  (bool)ci;
  // CHECK-NEXT: load volatile {{.*}} @ci, i32 0, i32 0
  // CHECK-NEXT: load volatile {{.*}} @ci, i32 0, i32 1
  // CHECK-NEXT: icmp ne
  // CHECK-NEXT: icmp ne
  // CHECK-NEXT: or i1

  ci=ci;
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: store volatile
  // CHECK-NEXT: store volatile

  asm("nop"); // CHECK-NEXT: call void asm

  // Extra load in C++.
  ci=ci=ci;
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: store volatile
  // CHECK-NEXT: store volatile
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: store volatile
  // CHECK-NEXT: store volatile

  __imag ci = __imag ci = __imag ci;
  // CHECK-NEXT: [[T:%.*]] = load volatile [[INT]]* getelementptr inbounds ([[CINT]]* @ci, i32 0, i32 1)
  // CHECK-NEXT: store volatile [[INT]] [[T]], [[INT]]* getelementptr inbounds ([[CINT]]* @ci, i32 0, i32 1)
  // CHECK-NEXT: [[T:%.*]] = load volatile [[INT]]* getelementptr inbounds ([[CINT]]* @ci, i32 0, i32 1)
  // CHECK-NEXT: store volatile [[INT]] [[T]], [[INT]]* getelementptr inbounds ([[CINT]]* @ci, i32 0, i32 1)

  __real (i = j);
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: store volatile

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
  // CHECK-NEXT: call
  ((a=a),a);

  // Not a use.  gcc gets this wrong, it doesn't emit the copy!  
  // CHECK-NEXT: call
  (void)(a=a);

  // Not a use.  gcc got this wrong in 4.2 and omitted the side effects
  // entirely, but it is fixed in 4.4.0.
  __imag (i = j);
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: store volatile

  // C++ does an extra load here.  Note that we have to do full loads.
  (float)(ci=ci);
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: store volatile
  // CHECK-NEXT: store volatile
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: sitofp

  // Not a use, bug?  gcc treats this as not a use, that's probably a
  // bug due to tree folding ignoring volatile.
  (int)(ci=ci);
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: store volatile
  // CHECK-NEXT: store volatile
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: load volatile

  // A use.
  (float)(i=i);
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: store volatile
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: sitofp

  // A use.  gcc treats this as not a use, that's probably a bug due to tree
  // folding ignoring volatile.
  (int)(i=i);
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: store volatile
  // CHECK-NEXT: load volatile

  // A use.
  -(i=j);
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: store volatile
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: sub

  // A use.  gcc treats this a not a use, that's probably a bug due to tree
  // folding ignoring volatile.
  +(i=k);
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: store volatile
  // CHECK-NEXT: load volatile

  // A use. gcc treats this a not a use, that's probably a bug due to tree
  // folding ignoring volatile.
  __real (ci=ci);
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: store volatile
  // CHECK-NEXT: store volatile

  // A use.
  i + 0;
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: add

  // A use.
  (i=j) + i;
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: store volatile
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: add

  // A use.  gcc treats this as not a use, that's probably a bug due to tree
  // folding ignoring volatile.
  (i=j) + 0;
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: store volatile
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: add

  (i,j)=k;
  // CHECK-NEXT: load volatile [[INT]]* @k
  // CHECK-NEXT: store volatile {{.*}}, [[INT]]* @j

  (j=k,i)=i;
  // CHECK-NEXT: load volatile [[INT]]* @i
  // CHECK-NEXT: load volatile [[INT]]* @k
  // CHECK-NEXT: store volatile {{.*}}, [[INT]]* @j
  // CHECK-NEXT: store volatile {{.*}}, [[INT]]* @i

  // CHECK-NEXT: ret void
}
