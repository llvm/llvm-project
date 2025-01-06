// RUN: %clang_cc1 -Wno-unused-value -triple %itanium_abi_triple -emit-llvm %s -std=c++98 -o - | FileCheck %s
// RUN: %clang_cc1 -Wno-unused-value -triple %itanium_abi_triple -emit-llvm %s -std=c++11 -o - | FileCheck -check-prefix=CHECK -check-prefix=CHECK11 %s

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


// CHECK: define {{.*}}void @{{.*}}test
void test() {

  asm("nop"); // CHECK: call void asm

  // should not load in C++98
  i;
  // CHECK11-NEXT: load volatile [[INT]], ptr @i

  (float)(ci);
  // CHECK-NEXT: load volatile [[INT]], ptr @ci
  // CHECK-NEXT: load volatile [[INT]], ptr getelementptr inbounds nuw ([[CINT]], ptr @ci, i32 0, i32 1)
  // CHECK-NEXT: sitofp [[INT]]

  // These are not uses in C++98:
  //   [expr.static.cast]p6:
  //     The lvalue-to-rvalue . . . conversions are not applied to the expression.
  (void)ci;
  // CHECK11-NEXT: load volatile [[INT]], ptr @ci
  // CHECK11-NEXT: load volatile [[INT]], ptr getelementptr inbounds nuw ([[CINT]], ptr @ci, i32 0, i32 1)

  (void)a;

  (void)(ci=ci);
  // CHECK-NEXT: [[R:%.*]] = load volatile [[INT]], ptr @ci
  // CHECK-NEXT: [[I:%.*]] = load volatile [[INT]], ptr getelementptr inbounds nuw ([[CINT]], ptr @ci, i32 0, i32 1)
  // CHECK-NEXT: store volatile [[INT]] [[R]], ptr @ci
  // CHECK-NEXT: store volatile [[INT]] [[I]], ptr getelementptr inbounds nuw ([[CINT]], ptr @ci, i32 0, i32 1)

  (void)(i=j);
  // CHECK-NEXT: [[T:%.*]] = load volatile [[INT]], ptr @j
  // CHECK-NEXT: store volatile [[INT]] [[T]], ptr @i

  ci+=ci;
  // CHECK-NEXT: [[R1:%.*]] = load volatile [[INT]], ptr @ci
  // CHECK-NEXT: [[I1:%.*]] = load volatile [[INT]], ptr getelementptr inbounds nuw ([[CINT]], ptr @ci, i32 0, i32 1)
  // CHECK-NEXT: [[R2:%.*]] = load volatile [[INT]], ptr @ci
  // CHECK-NEXT: [[I2:%.*]] = load volatile [[INT]], ptr getelementptr inbounds nuw ([[CINT]], ptr @ci, i32 0, i32 1)
  // Not sure why they're ordered this way.
  // CHECK-NEXT: [[R:%.*]] = add [[INT]] [[R2]], [[R1]]
  // CHECK-NEXT: [[I:%.*]] = add [[INT]] [[I2]], [[I1]]
  // CHECK-NEXT: store volatile [[INT]] [[R]], ptr @ci
  // CHECK-NEXT: store volatile [[INT]] [[I]], ptr getelementptr inbounds nuw ([[CINT]], ptr @ci, i32 0, i32 1)

  // Note that C++ requires an extra load volatile over C from the LHS of the '+'.
  (ci += ci) + ci;
  // CHECK-NEXT: [[R1:%.*]] = load volatile [[INT]], ptr @ci
  // CHECK-NEXT: [[I1:%.*]] = load volatile [[INT]], ptr getelementptr inbounds nuw ([[CINT]], ptr @ci, i32 0, i32 1)
  // CHECK-NEXT: [[R2:%.*]] = load volatile [[INT]], ptr @ci
  // CHECK-NEXT: [[I2:%.*]] = load volatile [[INT]], ptr getelementptr inbounds nuw ([[CINT]], ptr @ci, i32 0, i32 1)
  // CHECK-NEXT: [[R:%.*]] = add [[INT]] [[R2]], [[R1]]
  // CHECK-NEXT: [[I:%.*]] = add [[INT]] [[I2]], [[I1]]
  // CHECK-NEXT: store volatile [[INT]] [[R]], ptr @ci
  // CHECK-NEXT: store volatile [[INT]] [[I]], ptr getelementptr inbounds nuw ([[CINT]], ptr @ci, i32 0, i32 1)
  // CHECK-NEXT: [[R1:%.*]] = load volatile [[INT]], ptr @ci
  // CHECK-NEXT: [[I1:%.*]] = load volatile [[INT]], ptr getelementptr inbounds nuw ([[CINT]], ptr @ci, i32 0, i32 1)
  // CHECK-NEXT: [[R2:%.*]] = load volatile [[INT]], ptr @ci
  // CHECK-NEXT: [[I2:%.*]] = load volatile [[INT]], ptr getelementptr inbounds nuw ([[CINT]], ptr @ci, i32 0, i32 1)
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

  (void)i; // This is now a load in C++11
  // CHECK11-NEXT: load volatile

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

  (void)(i,(i=i)); // first i is also a load in C++11
  // CHECK11-NEXT: load volatile
  // CHECK-NEXT: load volatile
  // CHECK-NEXT: store volatile

  i=i,k; // k is also a load in C++11
  // CHECK-NEXT: load volatile [[INT]], ptr @i
  // CHECK-NEXT: store volatile {{.*}}, ptr @i
  // CHECK11-NEXT: load volatile [[INT]], ptr @k

  (i=j,k=j);
  // CHECK-NEXT: load volatile [[INT]], ptr @j
  // CHECK-NEXT: store volatile {{.*}}, ptr @i
  // CHECK-NEXT: load volatile [[INT]], ptr @j
  // CHECK-NEXT: store volatile {{.*}}, ptr @k

  (i=j,k); // k is also a load in C++11
  // CHECK-NEXT: load volatile [[INT]], ptr @j
  // CHECK-NEXT: store volatile {{.*}}, ptr @i
  // CHECK11-NEXT: load volatile [[INT]], ptr @k

  (i,j); // i and j both are loads in C++11
  // CHECK11-NEXT: load volatile [[INT]], ptr @i
  // CHECK11-NEXT: load volatile [[INT]], ptr @j

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

  ci; // ci is a load in C++11
  // CHECK11-NEXT: load volatile {{.*}} @ci
  // CHECK11-NEXT: load volatile {{.*}} @ci, i32 0, i32 1

  asm("nop"); // CHECK-NEXT: call void asm

  (int)ci;
  // CHECK-NEXT: load volatile {{.*}} @ci
  // CHECK-NEXT: load volatile {{.*}} @ci, i32 0, i32 1

  (bool)ci;
  // CHECK-NEXT: load volatile {{.*}} @ci
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
  // CHECK-NEXT: [[T:%.*]] = load volatile [[INT]], ptr getelementptr inbounds nuw ([[CINT]], ptr @ci, i32 0, i32 1)
  // CHECK-NEXT: store volatile [[INT]] [[T]], ptr getelementptr inbounds nuw ([[CINT]], ptr @ci, i32 0, i32 1)
  // CHECK-NEXT: [[T:%.*]] = load volatile [[INT]], ptr getelementptr inbounds nuw ([[CINT]], ptr @ci, i32 0, i32 1)
  // CHECK-NEXT: store volatile [[INT]] [[T]], ptr getelementptr inbounds nuw ([[CINT]], ptr @ci, i32 0, i32 1)

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
  // CHECK-NEXT: call {{.*}}void
  ((a=a),a);

  // Not a use.  gcc gets this wrong, it doesn't emit the copy!  
  // CHECK-NEXT: call {{.*}}void
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

  (i,j)=k; // i is also a load in C++11
  // CHECK-NEXT: load volatile [[INT]], ptr @k
  // CHECK11-NEXT: load volatile [[INT]], ptr @i
  // CHECK-NEXT: store volatile {{.*}}, ptr @j

  (j=k,i)=i;
  // CHECK-NEXT: load volatile [[INT]], ptr @i
  // CHECK-NEXT: load volatile [[INT]], ptr @k
  // CHECK-NEXT: store volatile {{.*}}, ptr @j
  // CHECK-NEXT: store volatile {{.*}}, ptr @i

  // CHECK-NEXT: ret void
}
