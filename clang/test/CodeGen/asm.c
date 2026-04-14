// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -O0 %s -o - | FileCheck %s -check-prefix=X86_64-O0
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -O2 %s -o - | FileCheck %s -check-prefix=X86_64-O2
// RUN: %clang_cc1 -triple i386-unknown-unknown   -emit-llvm -O0 %s -o - | FileCheck %s -check-prefix=I386-O0
// RUN: %clang_cc1 -triple i386-unknown-unknown   -emit-llvm -O2 %s -o - | FileCheck %s -check-prefix=I386-O2

// PR10415:
//
// X86_64-O0:      module asm "foo1"
// X86_64-O0-NEXT: module asm "foo2"
// X86_64-O0-NEXT: module asm "foo3"
//
// X86_64-O2:      module asm "foo1"
// X86_64-O2-NEXT: module asm "foo2"
// X86_64-O2-NEXT: module asm "foo3"
//
// I386-O0:        module asm "foo1"
// I386-O0-NEXT:   module asm "foo2"
// I386-O0-NEXT:   module asm "foo3"
//
// I386-O2:        module asm "foo1"
// I386-O2-NEXT:   module asm "foo2"
// I386-O2-NEXT:   module asm "foo3"
__asm__ ("foo1");
__asm__ ("foo2");
__asm__ ("foo3");

void t1(int len) {
  // X86_64-O0-LABEL: define{{.*}} void @t1
  // X86_64-O0:         call { i32, i32 } asm sideeffect "", "=&r,=&r,1,~{dirflag},~{fpsr},~{flags}"
  // X86_64-O0-SAME:      (i32 [[T1:%[.a-z0-9]+]])
  //
  // X86_64-O2-LABEL: define{{.*}} void @t1
  // X86_64-O2:         call { i32, i32 } asm sideeffect "", "=&r,=&r,1,~{dirflag},~{fpsr},~{flags}"
  // X86_64-O2-SAME:      (i32 [[T1:%[.a-z0-9]+]])
  //
  // I386-O0-LABEL:   define{{.*}} void @t1
  // I386-O0:           call { i32, i32 } asm sideeffect "", "=&r,=&r,1,~{dirflag},~{fpsr},~{flags}"
  // I386-O0-SAME:        (i32 [[T1:%[.a-z0-9]+]])
  //
  // I386-O2-LABEL:   define{{.*}} void @t1
  // I386-O2:           call { i32, i32 } asm sideeffect "", "=&r,=&r,1,~{dirflag},~{fpsr},~{flags}"
  // I386-O2-SAME:        (i32 [[T1:%[.a-z0-9]+]])
  __asm__ volatile ("" : "=&r" (len), "+&r" (len));
}

void t2(unsigned long long t)  {
  // X86_64-O0-LABEL: define{{.*}} void @t2
  // X86_64-O0:         call void asm sideeffect "", "=*m,*m,~{dirflag},~{fpsr},~{flags}"
  // X86_64-O0-SAME:      (ptr elementtype(i64) [[T2:%[a-z0-9.]+]], ptr elementtype(i64) [[T2]])
  //
  // X86_64-O2-LABEL: define{{.*}} void @t2
  // X86_64-O2:         call void asm sideeffect "", "=*m,*m,~{dirflag},~{fpsr},~{flags}"
  // X86_64-O2-SAME:      (ptr nonnull elementtype(i64) [[T2:%[.a-z0-9]+]], ptr nonnull elementtype(i64) [[T2]])
  //
  // I386-O0-LABEL:   define{{.*}} void @t2
  // I386-O0:           call void asm sideeffect "", "=*m,*m,~{dirflag},~{fpsr},~{flags}"
  // I386-O0-SAME:        (ptr elementtype(i64) [[T2:%[.0-9a-z]+]], ptr elementtype(i64) [[T2]])
  //
  // I386-O2-LABEL:   define{{.*}} void @t2
  // I386-O2:           call void asm sideeffect "", "=*m,*m,~{dirflag},~{fpsr},~{flags}"
  // I386-O2-SAME:        (ptr nonnull elementtype(i64) [[T2:%[.a-z0-9]+]], ptr nonnull elementtype(i64) [[T2]])
  __asm__ volatile ("" : "+m" (t));
}

void t3(unsigned char *src, unsigned long long temp) {
  // X86_64-O0-LABEL: define{{.*}} void @t3
  // X86_64-O0:         call ptr asm sideeffect "", "=*m,=r,*m,1,~{dirflag},~{fpsr},~{flags}"
  // X86_64-O0-SAME:      (ptr elementtype(i64) [[T3:%[a-z0-9.]+]], ptr elementtype(i64) [[T3]], ptr %{{.*}})
  //
  // X86_64-O2-LABEL: define{{.*}} void @t3
  // X86_64-O2:         call ptr asm sideeffect "", "=*m,=r,*m,1,~{dirflag},~{fpsr},~{flags}"
  // X86_64-O2-SAME:      (ptr nonnull elementtype(i64) [[T3:%[a-z0-9.]+]], ptr nonnull elementtype(i64) [[T3]], ptr %{{.*}})
  //
  // I386-O0-LABEL:   define{{.*}} void @t3
  // I386-O0:           call ptr asm sideeffect "", "=*m,=r,*m,1,~{dirflag},~{fpsr},~{flags}"
  // I386-O0-SAME:        (ptr elementtype(i64) [[T3:%[a-z0-9.]+]], ptr elementtype(i64) [[T3]], ptr %{{.*}})
  //
  // I386-O2-LABEL:   define{{.*}} void @t3
  // I386-O2:           call ptr asm sideeffect "", "=*m,=r,*m,1,~{dirflag},~{fpsr},~{flags}"
  // I386-O2-SAME:        (ptr nonnull elementtype(i64) [[T3:%[a-z0-9.]+]], ptr nonnull elementtype(i64) [[T3]], ptr %{{.*}})
  __asm__ volatile ("" : "+m" (temp), "+r" (src));
}

void t4(void) {
  // X86_64-O0-LABEL: define{{.*}} void @t4
  // X86_64-O0:         call void asm sideeffect "", "*m,*m,~{dirflag},~{fpsr},~{flags}"
  // X86_64-O0-SAME:      (ptr elementtype(i64) %{{.*}}, ptr elementtype(%struct.reg) %{{.*}})
  //
  // X86_64-O2-LABEL: define{{.*}} void @t4
  // X86_64-O2:         call void asm sideeffect "", "*m,*m,~{dirflag},~{fpsr},~{flags}"
  // X86_64-O2-SAME:      (ptr nonnull elementtype(i64) %{{.*}}, ptr nonnull elementtype(%struct.reg) %{{.*}})
  //
  // I386-O0-LABEL:   define{{.*}} void @t4
  // I386-O0:           call void asm sideeffect "", "*m,*m,~{dirflag},~{fpsr},~{flags}"
  // I386-O0-SAME:        (ptr elementtype(i64) %{{.*}}, ptr elementtype(%struct.reg) %{{.*}})
  //
  // I386-O2-LABEL:   define{{.*}} void @t4
  // I386-O2:           call void asm sideeffect "", "*m,*m,~{dirflag},~{fpsr},~{flags}"
  // I386-O2-SAME:        (ptr nonnull elementtype(i64) %{{.*}}, ptr nonnull elementtype(%struct.reg) %{{.*}})
  unsigned long long a;
  struct reg { unsigned long long a, b; } b;

  __asm__ volatile ("":: "m" (a), "m" (b));
}

// PR3417
void t5(int i) {
  // X86_64-O0-LABEL: define{{.*}} void @t5
  // X86_64-O0:         call ptr asm "nop", "=r,0,~{dirflag},~{fpsr},~{flags}"(ptr @t5)
  //
  // X86_64-O2-LABEL: define{{.*}} void @t5
  // X86_64-O2:         call ptr asm "nop", "=r,0,~{dirflag},~{fpsr},~{flags}"(ptr nonnull @t5)
  //
  // I386-O0-LABEL:   define{{.*}} void @t5
  // I386-O0:           call i32 asm "nop", "=r,0,~{dirflag},~{fpsr},~{flags}"(ptr @t5)
  //
  // I386-O2-LABEL:   define{{.*}} void @t5
  // I386-O2:           call i32 asm "nop", "=r,0,~{dirflag},~{fpsr},~{flags}"(ptr nonnull @t5)
  asm ("nop" : "=r" (i) : "0" (t5));
}

// PR3641
void t6(void) {
  // X86_64-O0-LABEL: define{{.*}} void @t6
  // X86_64-O0:         call void asm sideeffect "", "i,~{dirflag},~{fpsr},~{flags}"(ptr @{{.*}})
  //
  // X86_64-O2-LABEL: define{{.*}} void @t6
  // X86_64-O2:         call void asm sideeffect "", "i,~{dirflag},~{fpsr},~{flags}"(ptr nonnull @{{.*}})
  //
  // I386-O0-LABEL:   define{{.*}} void @t6
  // I386-O0:           call void asm sideeffect "", "i,~{dirflag},~{fpsr},~{flags}"(ptr @{{.*}})
  //
  // I386-O2-LABEL:   define{{.*}} void @t6
  // I386-O2:           call void asm sideeffect "", "i,~{dirflag},~{fpsr},~{flags}"(ptr nonnull @{{.*}})
  __asm__ volatile ("" : : "i" (t6));
}

void t7(int a) {
  // X86_64-O0-LABEL: define{{.*}} void @t7
  // X86_64-O0:         call i32 asm sideeffect "T7 NAMED: $1", "=r,i,0,~{dirflag},~{fpsr},~{flags}"
  // X86_64-O0-SAME:      (i32 4, i32 %{{.*}})
  //
  // X86_64-O2-LABEL: define{{.*}} void @t7
  // X86_64-O2:         call i32 asm sideeffect "T7 NAMED: $1", "=r,i,0,~{dirflag},~{fpsr},~{flags}"
  // X86_64-O2-SAME:      (i32 4, i32 %{{.*}})
  //
  // I386-O0-LABEL:   define{{.*}} void @t7
  // I386-O0:           call i32 asm sideeffect "T7 NAMED: $1", "=r,i,0,~{dirflag},~{fpsr},~{flags}"
  // I386-O0-SAME:        (i32 4, i32 %{{.*}})
  //
  // I386-O2-LABEL:   define{{.*}} void @t7
  // I386-O2:           call i32 asm sideeffect "T7 NAMED: $1", "=r,i,0,~{dirflag},~{fpsr},~{flags}"
  // I386-O2-SAME:        (i32 4, i32 %{{.*}})
  __asm__ volatile ("T7 NAMED: %[input]" : "+r" (a): [input] "i" (4));
}

void t8(void) {
  // X86_64-O0-LABEL: define{{.*}} void @t8
  // X86_64-O0:         call void asm sideeffect "T8 NAMED MODIFIER: ${0:c}", "i,~{dirflag},~{fpsr},~{flags}"(i32 4)
  //
  // X86_64-O2-LABEL: define{{.*}} void @t8
  // X86_64-O2:         call void asm sideeffect "T8 NAMED MODIFIER: ${0:c}", "i,~{dirflag},~{fpsr},~{flags}"(i32 4)
  //
  // I386-O0-LABEL:   define{{.*}} void @t8
  // I386-O0:           call void asm sideeffect "T8 NAMED MODIFIER: ${0:c}", "i,~{dirflag},~{fpsr},~{flags}"(i32 4)
  //
  // I386-O2-LABEL:   define{{.*}} void @t8
  // I386-O2:           call void asm sideeffect "T8 NAMED MODIFIER: ${0:c}", "i,~{dirflag},~{fpsr},~{flags}"(i32 4)
  __asm__ volatile ("T8 NAMED MODIFIER: %c[input]" :: [input] "i" (4));
}

// PR3682
unsigned t9(unsigned int a) {
  // X86_64-O0-LABEL: define{{.*}} i32 @t9
  // X86_64-O0:         call i32 asm "bswap $0 $1", "=r,0,~{dirflag},~{fpsr},~{flags}"(i32 %{{.*}})
  //
  // X86_64-O2-LABEL: define{{.*}} i32 @t9
  // X86_64-O2:         call i32 asm "bswap $0 $1", "=r,0,~{dirflag},~{fpsr},~{flags}"(i32 %{{.*}})
  //
  // I386-O0-LABEL:   define{{.*}} i32 @t9
  // I386-O0:           call i32 asm "bswap $0 $1", "=r,0,~{dirflag},~{fpsr},~{flags}"(i32 %{{.*}})
  //
  // I386-O2-LABEL:   define{{.*}} i32 @t9
  // I386-O2:           call i32 asm "bswap $0 $1", "=r,0,~{dirflag},~{fpsr},~{flags}"(i32 %{{.*}})
  asm ("bswap %0 %1" : "+r" (a));
  return a;
}

// PR3908
void t10(int r) {
  // X86_64-O0-LABEL: define{{.*}} void @t10
  // X86_64-O0:         call i32 asm "PR3908 $1 $3 $2 $0", "=r,mx,mr,x,0,~{dirflag},~{fpsr},~{flags}"
  // X86_64-O0-SAME:      (i32 0, i32 0, double 0.000000e+00, i32 %{{.*}})
  //
  // X86_64-O2-LABEL: define{{.*}} void @t10
  // X86_64-O2:         call i32 asm "PR3908 $1 $3 $2 $0", "=r,mx,mr,x,0,~{dirflag},~{fpsr},~{flags}"
  // X86_64-O2-SAME:      (i32 0, i32 0, double 0.000000e+00, i32 %{{.*}})
  //
  // I386-O0-LABEL:   define{{.*}} void @t10
  // I386-O0:           call i32 asm "PR3908 $1 $3 $2 $0", "=r,mx,mr,x,0,~{dirflag},~{fpsr},~{flags}"
  // I386-O0-SAME:        (i32 0, i32 0, double 0.000000e+00, i32 %{{.*}})
  //
  // I386-O2-LABEL:   define{{.*}} void @t10
  // I386-O2:           call i32 asm "PR3908 $1 $3 $2 $0", "=r,mx,mr,x,0,~{dirflag},~{fpsr},~{flags}"
  // I386-O2-SAME:        (i32 0, i32 0, double 0.000000e+00, i32 %{{.*}})
  __asm__ ("PR3908 %[lf] %[xx] %[li] %[r]" : [r] "+r" (r) : [lf] "mx" (0), [li] "mr" (0), [xx] "x" ((double)(0)));
}

// PR3373
unsigned t11(signed char input) {
  // X86_64-O0-LABEL: define{{.*}} i32 @t11
  // X86_64-O0:         call i32 asm "xyz", "={ax},0,~{dirflag},~{fpsr},~{flags}"(i32 %{{.*}})
  //
  // X86_64-O2-LABEL: define{{.*}} i32 @t11
  // X86_64-O2:         call i32 asm "xyz", "={ax},0,~{dirflag},~{fpsr},~{flags}"(i32 %{{.*}})
  //
  // I386-O0-LABEL:   define{{.*}} i32 @t11
  // I386-O0:           call i32 asm "xyz", "={ax},0,~{dirflag},~{fpsr},~{flags}"(i32 %{{.*}})
  //
  // I386-O2-LABEL:   define{{.*}} i32 @t11
  // I386-O2:           call i32 asm "xyz", "={ax},0,~{dirflag},~{fpsr},~{flags}"(i32 %{{.*}})
  unsigned  output;

  __asm__ ("xyz" : "=a" (output) : "0" (input));
  return output;
}

// PR3373
unsigned char t12(unsigned input) {
  // X86_64-O0-LABEL: define{{.*}} i8 @t12
  // X86_64-O0:         call i32 asm "xyz", "={ax},0,~{dirflag},~{fpsr},~{flags}"(i32 %{{.*}})
  //
  // X86_64-O2-LABEL: define{{.*}} i8 @t12
  // X86_64-O2:         call i32 asm "xyz", "={ax},0,~{dirflag},~{fpsr},~{flags}"(i32 %{{.*}})
  //
  // I386-O0-LABEL:   define{{.*}} i8 @t12
  // I386-O0:           call i32 asm "xyz", "={ax},0,~{dirflag},~{fpsr},~{flags}"(i32 %{{.*}})
  //
  // I386-O2-LABEL:   define{{.*}} i8 @t12
  // I386-O2:           call i32 asm "xyz", "={ax},0,~{dirflag},~{fpsr},~{flags}"(i32 %{{.*}})
  unsigned char output;

  __asm__ ("xyz" : "=a" (output) : "0" (input));
  return output;
}

unsigned char t13(unsigned input) {
  // X86_64-O0-LABEL: define{{.*}} i8 @t13
  // X86_64-O0:         call i32 asm "xyz $1", "={ax},0,~{dirflag},~{fpsr},~{flags}"(i32 %{{.*}})
  //
  // X86_64-O2-LABEL: define{{.*}} i8 @t13
  // X86_64-O2:         call i32 asm "xyz $1", "={ax},0,~{dirflag},~{fpsr},~{flags}"(i32 %{{.*}})
  //
  // I386-O0-LABEL:   define{{.*}} i8 @t13
  // I386-O0:           call i32 asm "xyz $1", "={ax},0,~{dirflag},~{fpsr},~{flags}"(i32 %{{.*}})
  //
  // I386-O2-LABEL:   define{{.*}} i8 @t13
  // I386-O2:           call i32 asm "xyz $1", "={ax},0,~{dirflag},~{fpsr},~{flags}"(i32 %{{.*}})
  unsigned char output;

  __asm__ ("xyz %1" : "=a" (output) : "0" (input));
  return output;
}

// bitfield destination of an asm.
struct S {
  int a : 4;
};

void t14(struct S *P) {
  // X86_64-O0-LABEL: define{{.*}} void @t14
  // X86_64-O0:         call i32 asm "abc $0", "=r,~{dirflag},~{fpsr},~{flags}"()
  //
  // X86_64-O2-LABEL: define{{.*}} void @t14
  // X86_64-O2:         call i32 asm "abc $0", "=r,~{dirflag},~{fpsr},~{flags}"()
  //
  // I386-O0-LABEL:   define{{.*}} void @t14
  // I386-O0:           call i32 asm "abc $0", "=r,~{dirflag},~{fpsr},~{flags}"()
  //
  // I386-O2-LABEL:   define{{.*}} void @t14
  // I386-O2:           call i32 asm "abc $0", "=r,~{dirflag},~{fpsr},~{flags}"()
  __asm__ ("abc %0" : "=r" (P->a));
}

struct large {
  int x[1000];
};

unsigned long t15(int x, struct large *P) {
  // X86_64-O0-LABEL: define{{.*}} i64 @t15
  // X86_64-O0:         call i32 asm "xyz ", "=r,*m,0,~{dirflag},~{fpsr},~{flags}"
  // X86_64-O0-SAME:      (ptr elementtype(%struct.large) %{{.*}}, i32 %{{.*}})
  //
  // X86_64-O2-LABEL: define{{.*}} i64 @t15
  // X86_64-O2:         call i32 asm "xyz ", "=r,*m,0,~{dirflag},~{fpsr},~{flags}"
  // X86_64-O2-SAME:      (ptr elementtype(%struct.large) %P, i32 %x)
  //
  // I386-O0-LABEL:   define{{.*}} i32 @t15
  // I386-O0:           call i32 asm "xyz ", "=r,*m,0,~{dirflag},~{fpsr},~{flags}"
  // I386-O0-SAME:        (ptr elementtype(%struct.large) %{{.*}}, i32 %{{.*}})
  //
  // I386-O2-LABEL:   define{{.*}} i32 @t15
  // I386-O2:           call i32 asm "xyz ", "=r,*m,0,~{dirflag},~{fpsr},~{flags}"
  // I386-O2-SAME:        (ptr elementtype(%struct.large) %P, i32 %x)
  __asm__ ("xyz " : "=r" (x) : "m" (*P), "0" (x));
  return x;
}

// PR4938
int t16(void) {
  // X86_64-O0-LABEL: define{{.*}} i32 @t16
  // X86_64-O0:         call i32 asm "nop;", "=%{cx},r,~{dirflag},~{fpsr},~{flags}"(i32 %{{.*}})
  //
  // X86_64-O2-LABEL: define{{.*}} i32 @t16
  // X86_64-O2:         call i32 asm "nop;", "=%{cx},r,~{dirflag},~{fpsr},~{flags}"(i32 undef)
  //
  // I386-O0-LABEL:   define{{.*}} i32 @t16
  // I386-O0:           call i32 asm "nop;", "=%{cx},r,~{dirflag},~{fpsr},~{flags}"(i32 %{{.*}})
  //
  // I386-O2-LABEL:   define{{.*}} i32 @t16
  // I386-O2:           call i32 asm "nop;", "=%{cx},r,~{dirflag},~{fpsr},~{flags}"(i32 undef)
  int a, b;

  asm ("nop;" :"=%c" (a) : "r" (b));
  return 0;
}

// PR6475
void t17(void) {
  // X86_64-O0-LABEL: define{{.*}} void @t17
  // X86_64-O0:         call void asm "nop", "=*m,~{dirflag},~{fpsr},~{flags}"(ptr elementtype(i32) %{{.*}})
  //
  // X86_64-O2-LABEL: define{{.*}} void @t17
  // X86_64-O2:         call void asm "nop", "=*m,~{dirflag},~{fpsr},~{flags}"(ptr nonnull elementtype(i32) %{{.*}})
  //
  // I386-O0-LABEL:   define{{.*}} void @t17
  // I386-O0:           call void asm "nop", "=*m,~{dirflag},~{fpsr},~{flags}"(ptr elementtype(i32) %{{.*}})
  //
  // I386-O2-LABEL:   define{{.*}} void @t17
  // I386-O2:           call void asm "nop", "=*m,~{dirflag},~{fpsr},~{flags}"(ptr nonnull elementtype(i32) %{{.*}})
  int i;

  __asm__ ("nop": "=m" (i));
}

int t18(unsigned data) {
  // X86_64-O0-LABEL: define{{.*}} i32 @t18
  // X86_64-O0:         [[ASM_RES:%[a-z0-9.]+]] ={{.*}} call { i32, i32 }
  // X86_64-O0-SAME:      asm "xyz", "={ax},={dx},{ax},~{dirflag},~{fpsr},~{flags}"(i32 {{.*}})
  // X86_64-O0-NEXT:    extractvalue { i32, i32 } [[ASM_RES]], 0
  // X86_64-O0-NEXT:    extractvalue { i32, i32 } [[ASM_RES]], 1
  //
  // X86_64-O2-LABEL: define{{.*}} i32 @t18
  // X86_64-O2:         [[ASM_RES:%[a-z0-9.]+]] ={{.*}} call { i32, i32 }
  // X86_64-O2-SAME:      asm "xyz", "={ax},={dx},{ax},~{dirflag},~{fpsr},~{flags}"(i32 {{.*}})
  // X86_64-O2-NEXT:    extractvalue { i32, i32 } [[ASM_RES]], 0
  // X86_64-O2-NEXT:    extractvalue { i32, i32 } [[ASM_RES]], 1
  //
  // I386-O0-LABEL:   define{{.*}} i32 @t18
  // I386-O0:           [[ASM_RES:%[a-z0-9.]+]] ={{.*}} call { i32, i32 }
  // I386-O0-SAME:        asm "xyz", "={ax},={dx},{ax},~{dirflag},~{fpsr},~{flags}"(i32 {{.*}})
  // I386-O0-NEXT:      extractvalue { i32, i32 } [[ASM_RES]], 0
  // I386-O0-NEXT:      extractvalue { i32, i32 } [[ASM_RES]], 1
  //
  // I386-O2-LABEL:   define{{.*}} i32 @t18
  // I386-O2:           [[ASM_RES:%[a-z0-9.]+]] ={{.*}} call { i32, i32 }
  // I386-O2-SAME:        asm "xyz", "={ax},={dx},{ax},~{dirflag},~{fpsr},~{flags}"(i32 {{.*}})
  // I386-O2-NEXT:      extractvalue { i32, i32 } [[ASM_RES]], 0
  // I386-O2-NEXT:      extractvalue { i32, i32 } [[ASM_RES]], 1
  int a, b;

  asm ("xyz" : "=a" (a), "=d" (b) : "a" (data));
  return a + b;
}

// PR6780
int t19(unsigned data) {
  // X86_64-O0-LABEL: define{{.*}} i32 @t19
  // X86_64-O0:         call i32 asm "x$(abc$|def$|ghi$)z", "=r,r,~{dirflag},~{fpsr},~{flags}"(i32 {{.*}})
  //
  // X86_64-O2-LABEL: define{{.*}} i32 @t19
  // X86_64-O2:         call i32 asm "x$(abc$|def$|ghi$)z", "=r,r,~{dirflag},~{fpsr},~{flags}"(i32 {{.*}})
  //
  // I386-O0-LABEL:   define{{.*}} i32 @t19
  // I386-O0:           call i32 asm "x$(abc$|def$|ghi$)z", "=r,r,~{dirflag},~{fpsr},~{flags}"(i32 {{.*}})
  //
  // I386-O2-LABEL:   define{{.*}} i32 @t19
  // I386-O2:           call i32 asm "x$(abc$|def$|ghi$)z", "=r,r,~{dirflag},~{fpsr},~{flags}"(i32 {{.*}})
  int a, b;

  asm ("x{abc|def|ghi}z" : "=r" (a) : "r" (data));
  return a + b;
}

// PR6845 - Mismatching source/dest fp types.
double t20(double x) {
  // X86_64-O0-LABEL: define{{.*}} double @t20
  // X86_64-O0:         fpext double {{.*}} to x86_fp80
  // X86_64-O0-NEXT:    call x86_fp80 asm sideeffect "frndint", "={st},0,~{dirflag},~{fpsr},~{flags}"(x86_fp80 {{.*}})
  // X86_64-O0:         fptrunc x86_fp80 {{.*}} to double
  //
  // X86_64-O2-LABEL: define{{.*}} double @t20
  // X86_64-O2:         fpext double {{.*}} to x86_fp80
  // X86_64-O2-NEXT:    call x86_fp80 asm sideeffect "frndint", "={st},0,~{dirflag},~{fpsr},~{flags}"(x86_fp80 {{.*}})
  // X86_64-O2-NEXT:    fptrunc x86_fp80 {{.*}} to double
  //
  // I386-O0-LABEL:   define{{.*}} double @t20
  // I386-O0:           fpext double {{.*}} to x86_fp80
  // I386-O0-NEXT:      call x86_fp80 asm sideeffect "frndint", "={st},0,~{dirflag},~{fpsr},~{flags}"(x86_fp80 {{.*}})
  // I386-O0:           fptrunc x86_fp80 {{.*}} to double
  //
  // I386-O2-LABEL:   define{{.*}} double @t20
  // I386-O2:           fpext double {{.*}} to x86_fp80
  // I386-O2-NEXT:      call x86_fp80 asm sideeffect "frndint", "={st},0,~{dirflag},~{fpsr},~{flags}"(x86_fp80 {{.*}})
  // I386-O2:           fptrunc x86_fp80 {{.*}} to double
  register long double result;

  __asm __volatile ("frndint"  : "=t" (result) : "0" (x));
  return result;
}

float t21(long double x) {
  // X86_64-O0-LABEL: define{{.*}} float @t21
  // X86_64-O0:         call x86_fp80 asm sideeffect "frndint", "={st},0,~{dirflag},~{fpsr},~{flags}"(x86_fp80 {{.*}})
  // X86_64-O0-NEXT:    fptrunc x86_fp80 {{.*}} to float
  //
  // X86_64-O2-LABEL: define{{.*}} float @t21
  // X86_64-O2:         call x86_fp80 asm sideeffect "frndint", "={st},0,~{dirflag},~{fpsr},~{flags}"(x86_fp80 {{.*}})
  // X86_64-O2-NEXT:    fptrunc x86_fp80 {{.*}} to float
  //
  // I386-O0-LABEL:   define{{.*}} float @t21
  // I386-O0:           call x86_fp80 asm sideeffect "frndint", "={st},0,~{dirflag},~{fpsr},~{flags}"(x86_fp80 {{.*}})
  // I386-O0-NEXT:      fptrunc x86_fp80 {{.*}} to float
  //
  // I386-O2-LABEL:   define{{.*}} float @t21
  // I386-O2:           call x86_fp80 asm sideeffect "frndint", "={st},0,~{dirflag},~{fpsr},~{flags}"(x86_fp80 {{.*}})
  // I386-O2-NEXT:      fptrunc x86_fp80 {{.*}} to float
  register float result;

  __asm __volatile ("frndint"  : "=t" (result) : "0" (x));
  return result;
}

// accept 'l' constraint
unsigned char t22(unsigned char a, unsigned char b) {
  // X86_64-O0-LABEL: define{{.*}} i8 @t22
  // X86_64-O0:         call i32 asm "0:\0A1:\0A", "=l{ax},0,{cx},~{edx},~{cc},~{dirflag},~{fpsr},~{flags}"
  // X86_64-O0-SAME:      (i32 {{.*}}, i32 {{.*}})
  //
  // X86_64-O2-LABEL: define{{.*}} i8 @t22
  // X86_64-O2:         call i32 asm "0:\0A1:\0A", "=l{ax},0,{cx},~{edx},~{cc},~{dirflag},~{fpsr},~{flags}"
  // X86_64-O2-SAME:      (i32 {{.*}}, i32 {{.*}})
  //
  // I386-O0-LABEL:   define{{.*}} i8 @t22
  // I386-O0:           call i32 asm "0:\0A1:\0A", "=l{ax},0,{cx},~{edx},~{cc},~{dirflag},~{fpsr},~{flags}"
  // I386-O0-SAME:        (i32 {{.*}}, i32 {{.*}})
  //
  // I386-O2-LABEL:   define{{.*}} i8 @t22
  // I386-O2:           call i32 asm "0:\0A1:\0A", "=l{ax},0,{cx},~{edx},~{cc},~{dirflag},~{fpsr},~{flags}"
  // I386-O2-SAME:        (i32 {{.*}}, i32 {{.*}})
  unsigned int la = a;
  unsigned int lb = b;
  unsigned int bigres;
  unsigned char res;

  __asm__ ("0:\n1:\n" : [bigres] "=la"(bigres) : [la] "0"(la), [lb] "c"(lb) : "edx", "cc");
  res = bigres;
  return res;
}

// accept 'l' constraint
unsigned char t23(unsigned char a, unsigned char b) {
  // X86_64-O0-LABEL: define{{.*}} i8 @t23
  // X86_64-O0:         call i32 asm "0:\0A1:\0A", "=l{ax},0,{cx},~{edx},~{cc},~{dirflag},~{fpsr},~{flags}"
  // X86_64-O0-SAME:      (i32 {{.*}}, i32 {{.*}})
  //
  // X86_64-O2-LABEL: define{{.*}} i8 @t23
  // X86_64-O2:         call i32 asm "0:\0A1:\0A", "=l{ax},0,{cx},~{edx},~{cc},~{dirflag},~{fpsr},~{flags}"
  // X86_64-O2-SAME:      (i32 {{.*}}, i32 {{.*}})
  //
  // I386-O0-LABEL:   define{{.*}} i8 @t23
  // I386-O0:           call i32 asm "0:\0A1:\0A", "=l{ax},0,{cx},~{edx},~{cc},~{dirflag},~{fpsr},~{flags}"
  // I386-O0-SAME:        (i32 {{.*}}, i32 {{.*}})
  //
  // I386-O2-LABEL:   define{{.*}} i8 @t23
  // I386-O2:           call i32 asm "0:\0A1:\0A", "=l{ax},0,{cx},~{edx},~{cc},~{dirflag},~{fpsr},~{flags}"
  // I386-O2-SAME:        (i32 {{.*}}, i32 {{.*}})
  unsigned int la = a;
  unsigned int lb = b;
  unsigned char res;

  __asm__ ("0:\n1:\n" : [res] "=la" (res) : [la] "0" (la), [lb] "c" (lb) : "edx", "cc");
  return res;
}

void *t24(char c) {
  // X86_64-O0-LABEL: define{{.*}} ptr @t24
  // X86_64-O0:         [[C:%[a-z0-9.]+]] = zext i8 {{.*}} to i64
  // X86_64-O0-NEXT:    call ptr asm "foobar", "={ax},0,~{dirflag},~{fpsr},~{flags}"(i64 [[C]])
  //
  // X86_64-O2-LABEL: define{{.*}} ptr @t24
  // X86_64-O2:         [[C:%[a-z0-9.]+]] = zext i8 {{.*}} to i64
  // X86_64-O2-NEXT:    call ptr asm "foobar", "={ax},0,~{dirflag},~{fpsr},~{flags}"(i64 [[C]])
  //
  // I386-O0-LABEL:   define{{.*}} ptr @t24
  // I386-O0:           [[C:%[a-z0-9.]+]] = zext i8 {{.*}} to i32
  // I386-O0-NEXT:      call ptr asm "foobar", "={ax},0,~{dirflag},~{fpsr},~{flags}"(i32 [[C]])
  //
  // I386-O2-LABEL:   define{{.*}} ptr @t24
  // I386-O2:           [[C:%[a-z0-9.]+]] = zext i8 {{.*}} to i32
  // I386-O2-NEXT:      call ptr asm "foobar", "={ax},0,~{dirflag},~{fpsr},~{flags}"(i32 [[C]])
  void *addr;

  __asm__ ("foobar" : "=a" (addr) : "0" (c));
  return addr;
}

// PR10299 - fpsr, fpcr
void t25(void) {
  // X86_64-O0-LABEL: define{{.*}} void @t25
  // X86_64-O0:         call void asm sideeffect "finit", "~{st},~{st(1)},~{st(2)},~{st(3)},~{st(4)},~{st(5)},~{st(6)},~{st(7)},~{fpsr},~{fpcr},~{dirflag},~{fpsr},~{flags}"()
  //
  // X86_64-O2-LABEL: define{{.*}} void @t25
  // X86_64-O2:         call void asm sideeffect "finit", "~{st},~{st(1)},~{st(2)},~{st(3)},~{st(4)},~{st(5)},~{st(6)},~{st(7)},~{fpsr},~{fpcr},~{dirflag},~{fpsr},~{flags}"()
  //
  // I386-O0-LABEL:   define{{.*}} void @t25
  // I386-O0:           call void asm sideeffect "finit", "~{st},~{st(1)},~{st(2)},~{st(3)},~{st(4)},~{st(5)},~{st(6)},~{st(7)},~{fpsr},~{fpcr},~{dirflag},~{fpsr},~{flags}"()
  //
  // I386-O2-LABEL:   define{{.*}} void @t25
  // I386-O2:           call void asm sideeffect "finit", "~{st},~{st(1)},~{st(2)},~{st(3)},~{st(4)},~{st(5)},~{st(6)},~{st(7)},~{fpsr},~{fpcr},~{dirflag},~{fpsr},~{flags}"()
  __asm__ __volatile__ ("finit" : : :
                        "st", "st(1)", "st(2)", "st(3)", "st(4)", "st(5)",
                        "st(6)", "st(7)", "fpsr", "fpcr");
}

// AVX registers
typedef long long __m256i __attribute__((__vector_size__(32)));

void t26 (__m256i *p) {
  // X86_64-O0-LABEL: define{{.*}} void @t26
  // X86_64-O0:         call void asm sideeffect "vmovaps  $0, %ymm0", "*m,~{ymm0},~{dirflag},~{fpsr},~{flags}"(ptr elementtype(<4 x i64>) {{.*}})
  //
  // X86_64-O2-LABEL: define{{.*}} void @t26
  // X86_64-O2:         call void asm sideeffect "vmovaps  $0, %ymm0", "*m,~{ymm0},~{dirflag},~{fpsr},~{flags}"(ptr elementtype(<4 x i64>) {{.*}})
  //
  // I386-O0-LABEL:   define{{.*}} void @t26
  // I386-O0:           call void asm sideeffect "vmovaps  $0, %ymm0", "*m,~{ymm0},~{dirflag},~{fpsr},~{flags}"(ptr elementtype(<4 x i64>) {{.*}})
  //
  // I386-O2-LABEL:   define{{.*}} void @t26
  // I386-O2:           call void asm sideeffect "vmovaps  $0, %ymm0", "*m,~{ymm0},~{dirflag},~{fpsr},~{flags}"(ptr elementtype(<4 x i64>) {{.*}})
  __asm__ volatile ("vmovaps  %0, %%ymm0" :: "m" (*(__m256i*)p) : "ymm0");
}

// Check to make sure the inline asm non-standard dialect attribute _not_ is
// emitted.
void t27(void) {
  // X86_64-O0-LABEL: define{{.*}} void @t27
  // X86_64-O0:         call void asm sideeffect "nop", "~{dirflag},~{fpsr},~{flags}"()
  // X86_64-O0-NOT:     ia_nsdialect
  // X86_64-O0:         ret void
  //
  // X86_64-O2-LABEL: define{{.*}} void @t27
  // X86_64-O2:         call void asm sideeffect "nop", "~{dirflag},~{fpsr},~{flags}"()
  // X86_64-O2-NOT:     ia_nsdialect
  // X86_64-O2:         ret void
  //
  // I386-O0-LABEL:   define{{.*}} void @t27
  // I386-O0:           call void asm sideeffect "nop", "~{dirflag},~{fpsr},~{flags}"()
  // I386-O0-NOT:       ia_nsdialect
  // I386-O0:           ret void
  //
  // I386-O2-LABEL:   define{{.*}} void @t27
  // I386-O2:           call void asm sideeffect "nop", "~{dirflag},~{fpsr},~{flags}"()
  // I386-O2-NOT:       ia_nsdialect
  // I386-O2:           ret void
  asm volatile ("nop");
}

// Check handling of '*' and '#' constraint modifiers.
void t28(void) {
  // X86_64-O0-LABEL: define{{.*}} void @t28
  // X86_64-O0:         call void asm sideeffect "/* $0 */", "i|r,~{dirflag},~{fpsr},~{flags}"(i32 1)
  //
  // X86_64-O2-LABEL: define{{.*}} void @t28
  // X86_64-O2:         call void asm sideeffect "/* $0 */", "i|r,~{dirflag},~{fpsr},~{flags}"(i32 1)
  //
  // I386-O0-LABEL:   define{{.*}} void @t28
  // I386-O0:           call void asm sideeffect "/* $0 */", "i|r,~{dirflag},~{fpsr},~{flags}"(i32 1)
  //
  // I386-O2-LABEL:   define{{.*}} void @t28
  // I386-O2:           call void asm sideeffect "/* $0 */", "i|r,~{dirflag},~{fpsr},~{flags}"(i32 1)
  asm volatile ("/* %0 */" : : "i#*X,*r" (1));
}

static unsigned t29_var[1];

void t29(void) {
  // X86_64-O0-LABEL: define{{.*}} void @t29
  // X86_64-O0:         call void asm sideeffect "movl %eax, $0", "*m,~{dirflag},~{fpsr},~{flags}"
  // X86_64-O0-SAME:      (ptr elementtype([1 x i32]) @t29_var)
  //
  // X86_64-O2-LABEL: define{{.*}} void @t29
  // X86_64-O2:         call void asm sideeffect "movl %eax, $0", "*m,~{dirflag},~{fpsr},~{flags}"
  // X86_64-O2-SAME:      (ptr nonnull elementtype([1 x i32]) @t29_var)
  //
  // I386-O0-LABEL:   define{{.*}} void @t29
  // I386-O0:           call void asm sideeffect "movl %eax, $0", "*m,~{dirflag},~{fpsr},~{flags}"
  // I386-O0-SAME:        (ptr elementtype([1 x i32]) @t29_var)
  //
  // I386-O2-LABEL:   define{{.*}} void @t29
  // I386-O2:           call void asm sideeffect "movl %eax, $0", "*m,~{dirflag},~{fpsr},~{flags}"
  // I386-O2-SAME:        (ptr nonnull elementtype([1 x i32]) @t29_var)
  asm volatile ("movl %%eax, %0" : : "m" (t29_var));
}

void t30(int len) {
  // X86_64-O0-LABEL: define{{.*}} void @t30
  // X86_64-O0:         call void asm sideeffect "", "=*&rm,0,~{dirflag},~{fpsr},~{flags}"
  //
  // X86_64-O2-LABEL: define{{.*}} void @t30
  // X86_64-O2:         call void asm sideeffect "", "=*&rm,0,~{dirflag},~{fpsr},~{flags}"
  //
  // I386-O0-LABEL:   define{{.*}} void @t30
  // I386-O0:           call void asm sideeffect "", "=*&rm,0,~{dirflag},~{fpsr},~{flags}"
  //
  // I386-O2-LABEL:   define{{.*}} void @t30
  // I386-O2:           call void asm sideeffect "", "=*&rm,0,~{dirflag},~{fpsr},~{flags}"
  __asm__ volatile ("" : "+&&rm" (len));
}

void t31(int len) {
  // X86_64-O0-LABEL: define{{.*}} void @t31
  // X86_64-O0:         call void asm sideeffect "", "=*%rm,=*rm,0,1,~{dirflag},~{fpsr},~{flags}"
  //
  // X86_64-O2-LABEL: define{{.*}} void @t31
  // X86_64-O2:         call void asm sideeffect "", "=*%rm,=*rm,0,1,~{dirflag},~{fpsr},~{flags}"
  //
  // I386-O0-LABEL:   define{{.*}} void @t31
  // I386-O0:           call void asm sideeffect "", "=*%rm,=*rm,0,1,~{dirflag},~{fpsr},~{flags}"
  //
  // I386-O2-LABEL:   define{{.*}} void @t31
  // I386-O2:           call void asm sideeffect "", "=*%rm,=*rm,0,1,~{dirflag},~{fpsr},~{flags}"
  __asm__ volatile("" : "+%%rm" (len), "+rm" (len));
}

int t32(int cond) {
  // X86_64-O0-LABEL: define{{.*}} i32 @t32
  // X86_64-O0:         callbr void asm sideeffect "testl $0, $0; jne ${1:l};", "r,!i,!i,~{dirflag},~{fpsr},~{flags}"(i32 {{.*}})
  // X86_64-O0-NEXT:            to label %asm.fallthrough [label %label_true, label %loop]
  //
  // X86_64-O2-LABEL: define{{.*}} i32 @t32
  // X86_64-O2:         callbr void asm sideeffect "testl $0, $0; jne ${1:l};", "r,!i,!i,~{dirflag},~{fpsr},~{flags}"(i32 {{.*}})
  // X86_64-O2-NEXT:            to label %return [label %label_true, label %return]
  //
  // I386-O0-LABEL:   define{{.*}} i32 @t32
  // I386-O0:           callbr void asm sideeffect "testl $0, $0; jne ${1:l};", "r,!i,!i,~{dirflag},~{fpsr},~{flags}"(i32 {{.*}})
  // I386-O0-NEXT:              to label %asm.fallthrough [label %label_true, label %loop]
  //
  // I386-O2-LABEL:   define{{.*}} i32 @t32
  // I386-O2:           callbr void asm sideeffect "testl $0, $0; jne ${1:l};", "r,!i,!i,~{dirflag},~{fpsr},~{flags}"(i32 {{.*}})
  // I386-O2-NEXT:              to label %return [label %label_true, label %return]
  asm goto ("testl %0, %0; jne %l1;" : : "r" (cond) : : label_true, loop);
  return 0;
loop:
  return 0;
label_true:
  return 1;
}

void *t33(void *ptr) {
  // X86_64-O0-LABEL: define{{.*}} ptr @t33
  // X86_64-O0:         call ptr asm "lea $1, $0", "=r,p,~{dirflag},~{fpsr},~{flags}"(ptr {{.*}})
  //
  // X86_64-O2-LABEL: define{{.*}} ptr @t33
  // X86_64-O2:         call ptr asm "lea $1, $0", "=r,p,~{dirflag},~{fpsr},~{flags}"(ptr {{.*}})
  //
  // I386-O0-LABEL:   define{{.*}} ptr @t33
  // I386-O0:           call ptr asm "lea $1, $0", "=r,p,~{dirflag},~{fpsr},~{flags}"(ptr {{.*}})
  //
  // I386-O2-LABEL:   define{{.*}} ptr @t33
  // I386-O2:           call ptr asm "lea $1, $0", "=r,p,~{dirflag},~{fpsr},~{flags}"(ptr {{.*}})
  void *ret;

  asm ("lea %1, %0" : "=r" (ret) : "p" (ptr));
  return ret;
}

void t34(void) {
  // X86_64-O0-LABEL: define{{.*}} void @t34
  // X86_64-O0:         call void asm sideeffect "T34 CC NAMED MODIFIER: ${0:c}", "i,~{dirflag},~{fpsr},~{flags}"
  //
  // X86_64-O2-LABEL: define{{.*}} void @t34
  // X86_64-O2:         call void asm sideeffect "T34 CC NAMED MODIFIER: ${0:c}", "i,~{dirflag},~{fpsr},~{flags}"
  //
  // I386-O0-LABEL:   define{{.*}} void @t34
  // I386-O0:           call void asm sideeffect "T34 CC NAMED MODIFIER: ${0:c}", "i,~{dirflag},~{fpsr},~{flags}"
  //
  // I386-O2-LABEL:   define{{.*}} void @t34
  // I386-O2:           call void asm sideeffect "T34 CC NAMED MODIFIER: ${0:c}", "i,~{dirflag},~{fpsr},~{flags}"
  __asm__ volatile ("T34 CC NAMED MODIFIER: %cc[input]" : : [input] "i"  (4));
}
