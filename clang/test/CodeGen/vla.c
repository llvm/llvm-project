// RUN: %clang_cc1 -Wno-int-conversion -triple i386-unknown-unknown %s -emit-llvm -o - | FileCheck %s -check-prefixes=CHECK,NULL-INVALID
// RUN: %clang_cc1 -Wno-int-conversion -triple i386-unknown-unknown %s -emit-llvm -fno-delete-null-pointer-checks -o - | FileCheck %s -check-prefixes=CHECK,NULL-VALID

int b(char* x);

// Extremely basic VLA test
void a(int x) {
  char arry[x];
  arry[0] = 10;
  b(arry);
}

int c(int n)
{
  return sizeof(int[n]);
}

int f0(int x) {
  int vla[x];
  return vla[x-1];
}

void
f(int count)
{
 int a[count];

  do {  } while (0);

  if (a[0] != 3) {
  }
}

void g(int count) {
  // Make sure we emit sizes correctly in some obscure cases
  int (*a[5])[count];
  int (*b)[][count];
}

// CHECK-LABEL: define{{.*}} void @f_8403108
void f_8403108(unsigned x) {
  // CHECK: call ptr @llvm.stacksave.p0()
  char s1[x];
  while (1) {
    // CHECK: call ptr @llvm.stacksave.p0()
    char s2[x];
    if (1)
      break;
  // CHECK: call void @llvm.stackrestore.p0(ptr
  }
  // CHECK: call void @llvm.stackrestore.p0(ptr
}

// pr7827
void function(short width, int data[][width]) {} // expected-note {{passing argument to parameter 'data' here}}

void test(void) {
     int bork[4][13];
     // CHECK: call void @function(i16 noundef signext 1, ptr noundef null)
     function(1, 0);
     // CHECK: call void @function(i16 noundef signext 1, ptr noundef inttoptr
     function(1, 0xbadbeef); // expected-warning {{incompatible integer to pointer conversion passing}}
     // CHECK: call void @function(i16 noundef  signext 1, ptr noundef {{.*}})
     function(1, bork);
}

void function1(short width, int data[][width][width]) {}
void test1(void) {
     int bork[4][13][15];
     // CHECK: call void @function1(i16 noundef signext 1, ptr noundef {{.*}})
     function1(1, bork);
     // CHECK: call void @function(i16 noundef signext 1, ptr noundef {{.*}})
     function(1, bork[2]);
}

static int GLOB;
int test2(int n)
{
  GLOB = 0;
  char b[1][n+3];			/* Variable length array.  */
  // CHECK:  [[tmp_1:%.*]] = load i32, ptr @GLOB, align 4
  // CHECK-NEXT: add nsw i32 [[tmp_1]], 1
  __typeof__(b[GLOB++]) c;
  return GLOB;
}

// http://llvm.org/PR8567
// CHECK-LABEL: define{{.*}} double @test_PR8567
double test_PR8567(int n, double (*p)[n][5]) {
  // CHECK:      [[NV:%.*]] = alloca i32, align 4
  // CHECK-NEXT: [[PV:%.*]] = alloca ptr, align 4
  // CHECK-NEXT: store
  // CHECK-NEXT: store
  // CHECK-NEXT: [[N:%.*]] = load i32, ptr [[NV]], align 4
  // CHECK-NEXT: [[P:%.*]] = load ptr, ptr [[PV]], align 4
  // CHECK-NEXT: [[T0:%.*]] = mul nsw i32 1, [[N]]
  // CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [5 x double], ptr [[P]], i32 [[T0]]
  // CHECK-NEXT: [[T2:%.*]] = getelementptr inbounds [5 x double], ptr [[T1]], i32 2
  // CHECK-NEXT: [[T3:%.*]] = getelementptr inbounds [5 x double], ptr [[T2]], i32 0, i32 3
  // CHECK-NEXT: [[T4:%.*]] = load double, ptr [[T3]]
  // CHECK-NEXT: ret double [[T4]]
 return p[1][2][3];
}

int test4(unsigned n, char (*p)[n][n+1][6]) {
  // CHECK-LABEL:    define{{.*}} i32 @test4(
  // CHECK:      [[N:%.*]] = alloca i32, align 4
  // CHECK-NEXT: [[P:%.*]] = alloca ptr, align 4
  // CHECK-NEXT: [[P2:%.*]] = alloca ptr, align 4
  // CHECK-NEXT: store i32
  // CHECK-NEXT: store ptr

  // VLA captures.
  // CHECK-NEXT: [[DIM0:%.*]] = load i32, ptr [[N]], align 4
  // CHECK-NEXT: [[T0:%.*]] = load i32, ptr [[N]], align 4
  // CHECK-NEXT: [[DIM1:%.*]] = add i32 [[T0]], 1

  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[P]], align 4
  // CHECK-NEXT: [[T1:%.*]] = load i32, ptr [[N]], align 4
  // CHECK-NEXT: [[T2:%.*]] = udiv i32 [[T1]], 2
  // CHECK-NEXT: [[T3:%.*]] = mul nuw i32 [[DIM0]], [[DIM1]]
  // CHECK-NEXT: [[T4:%.*]] = mul nsw i32 [[T2]], [[T3]]
  // CHECK-NEXT: [[T5:%.*]] = getelementptr inbounds nuw [6 x i8], ptr [[T0]], i32 [[T4]]
  // CHECK-NEXT: [[T6:%.*]] = load i32, ptr [[N]], align 4
  // CHECK-NEXT: [[T7:%.*]] = udiv i32 [[T6]], 4
  // CHECK-NEXT: [[T8:%.*]] = sub i32 0, [[T7]]
  // CHECK-NEXT: [[T9:%.*]] = mul nuw i32 [[DIM0]], [[DIM1]]
  // CHECK-NEXT: [[T10:%.*]] = mul nsw i32 [[T8]], [[T9]]
  // CHECK-NEXT: [[T11:%.*]] = getelementptr inbounds [6 x i8], ptr [[T5]], i32 [[T10]]
  // CHECK-NEXT: store ptr [[T11]], ptr [[P2]], align 4
  __typeof(p) p2 = (p + n/2) - n/4;

  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[P2]], align 4
  // CHECK-NEXT: [[T1:%.*]] = load ptr, ptr [[P]], align 4
  // CHECK-NEXT: [[T2:%.*]] = ptrtoint ptr [[T0]] to i32
  // CHECK-NEXT: [[T3:%.*]] = ptrtoint ptr [[T1]] to i32
  // CHECK-NEXT: [[T4:%.*]] = sub i32 [[T2]], [[T3]]
  // CHECK-NEXT: [[T5:%.*]] = mul nuw i32 [[DIM0]], [[DIM1]]
  // CHECK-NEXT: [[T6:%.*]] = mul nuw i32 6, [[T5]]
  // CHECK-NEXT: [[T7:%.*]] = sdiv exact i32 [[T4]], [[T6]]
  // CHECK-NEXT: ret i32 [[T7]]
  return p2 - p;
}

void test5(void)
{
  // CHECK-LABEL: define{{.*}} void @test5(
  int a[5], i = 0;
  // CHECK: [[A:%.*]] = alloca [5 x i32], align 4
  // CHECK-NEXT: [[I:%.*]] = alloca i32, align 4
  // CHECK-NEXT: [[CL:%.*]] = alloca ptr, align 4
  // CHECK-NEXT: store i32 0, ptr [[I]], align 4

  (typeof(++i, (int (*)[i])a)){&a} += 0;
  // CHECK-NEXT: [[Z:%.*]] = load i32, ptr [[I]], align 4
  // CHECK-NEXT: [[INC:%.*]]  = add nsw i32 [[Z]], 1
  // CHECK-NEXT: store i32 [[INC]], ptr [[I]], align 4
  // CHECK-NEXT: [[O:%.*]] = load i32, ptr [[I]], align 4
  // CHECK-NEXT: [[AR:%.*]] = getelementptr inbounds [5 x i32], ptr [[A]], i32 0, i32 0
  // CHECK-NEXT: store ptr [[A]], ptr [[CL]]
  // CHECK-NEXT: [[TH:%.*]] = load ptr, ptr [[CL]]
  // CHECK-NEXT: [[VLAIX:%.*]] = mul nsw i32 0, [[O]]
  // CHECK-NEXT: [[ADDPTR:%.*]] = getelementptr inbounds i32, ptr [[TH]], i32 [[VLAIX]]
  // CHECK-NEXT: store ptr [[ADDPTR]], ptr [[CL]]
}

void test6(void)
{
  // CHECK-LABEL: define{{.*}} void @test6(
  int n = 20, **a, i=0;
  // CHECK: [[N:%.*]] = alloca i32, align 4
  // CHECK-NEXT: [[A:%.*]] = alloca ptr, align 4
  // CHECK-NEXT: [[I:%.*]] = alloca i32, align 4
 (int (**)[i]){&a}[0][1][5] = 0;
  // CHECK-NEXT: [[CL:%.*]] = alloca ptr, align 4
  // CHECK-NEXT: store i32 20, ptr [[N]], align 4
  // CHECK-NEXT: store i32 0, ptr [[I]], align 4
  // CHECK-NEXT: [[Z:%.*]] = load i32, ptr [[I]], align 4
  // CHECK-NEXT: store ptr [[A]], ptr [[CL]]
  // CHECK-NEXT: [[T:%.*]] = load ptr, ptr [[CL]]
  // CHECK-NEXT: [[IX:%.*]] = getelementptr inbounds ptr, ptr [[T]], i32 0
  // CHECK-NEXT: [[TH:%.*]] = load ptr, ptr [[IX]], align 4
  // CHECK-NEXT: [[F:%.*]] = mul nsw i32 1, [[Z]]
  // CHECK-NEXT: [[IX1:%.*]] = getelementptr inbounds i32, ptr [[TH]], i32 [[F]]
  // CHECK-NEXT: [[IX2:%.*]] = getelementptr inbounds i32, ptr [[IX1]], i32 5
  // CHECK-NEXT: store i32 0, ptr [[IX2]], align 4
}

// Follow gcc's behavior for VLAs in parameter lists.  PR9559.
void test7(int a[b(0)]) {
  // CHECK-LABEL: define{{.*}} void @test7(
  // CHECK: call i32 @b(ptr noundef null)
}

// Make sure we emit dereferenceable or nonnull when the static keyword is
// provided.
void test8(int a[static 3]) { }
// CHECK: define{{.*}} void @test8(ptr noundef align 4 dereferenceable(12) %a)

void test9(int n, int a[static n]) { }
// NULL-INVALID: define{{.*}} void @test9(i32 noundef %n, ptr noundef nonnull align 4 %a)
// NULL-VALID: define{{.*}} void @test9(i32 noundef %n, ptr noundef align 4 %a)

// Make sure a zero-sized static array extent is still required to be nonnull.
void test10(int a[static 0]) {}
// NULL-INVALID: define{{.*}} void @test10(ptr noundef nonnull align 4 %a)
// NULL-VALID: define{{.*}} void @test10(ptr noundef align 4 %a)

const int constant = 32;
// CHECK: define {{.*}}pr44406(
int pr44406(void) {
  int n = 0;
  // Do not fold this VLA to an array of constant bound; that would miscompile
  // this testcase.
  char c[1][(constant - constant) + 3];
  // CHECK: store i32 1,
  sizeof(c[n = 1]);
  return n;
}
