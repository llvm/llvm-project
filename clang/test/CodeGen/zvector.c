// RUN: %clang_cc1 -triple s390x-linux-gnu -target-cpu z13 -fzvector \
// RUN:  -O -emit-llvm -o - -W -Wall -Werror %s | FileCheck %s

volatile vector signed char sc, sc2;
volatile vector unsigned char uc, uc2;
volatile vector bool char bc, bc2;

volatile vector signed short ss, ss2;
volatile vector unsigned short us, us2;
volatile vector bool short bs, bs2;

volatile vector signed int si, si2;
volatile vector unsigned int ui, ui2;
volatile vector bool int bi, bi2;

volatile vector signed long long sl, sl2;
volatile vector unsigned long long ul, ul2;
volatile vector bool long long bl, bl2;

volatile vector double fd, fd2;

volatile int cnt;

void test_assign (void)
{
// CHECK-LABEL: test_assign

// CHECK: [[VAL:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc2
// CHECK: store volatile <16 x i8> [[VAL]], <16 x i8>* @sc
  sc = sc2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc2
// CHECK: store volatile <16 x i8> [[VAL]], <16 x i8>* @uc
  uc = uc2;

// CHECK: [[VAL:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss2
// CHECK: store volatile <8 x i16> [[VAL]], <8 x i16>* @ss
  ss = ss2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us2
// CHECK: store volatile <8 x i16> [[VAL]], <8 x i16>* @us
  us = us2;

// CHECK: [[VAL:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si2
// CHECK: store volatile <4 x i32> [[VAL]], <4 x i32>* @si
  si = si2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui2
// CHECK: store volatile <4 x i32> [[VAL]], <4 x i32>* @ui
  ui = ui2;

// CHECK: [[VAL:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl2
// CHECK: store volatile <2 x i64> [[VAL]], <2 x i64>* @sl
  sl = sl2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul2
// CHECK: store volatile <2 x i64> [[VAL]], <2 x i64>* @ul
  ul = ul2;

// CHECK: [[VAL:%[^ ]+]] = load volatile <2 x double>, <2 x double>* @fd2
// CHECK: store volatile <2 x double> [[VAL]], <2 x double>* @fd
  fd = fd2;
}

void test_pos (void)
{
// CHECK-LABEL: test_pos

// CHECK: [[VAL:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc2
// CHECK: store volatile <16 x i8> [[VAL]], <16 x i8>* @sc
  sc = +sc2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc2
// CHECK: store volatile <16 x i8> [[VAL]], <16 x i8>* @uc
  uc = +uc2;

// CHECK: [[VAL:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss2
// CHECK: store volatile <8 x i16> [[VAL]], <8 x i16>* @ss
  ss = +ss2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us2
// CHECK: store volatile <8 x i16> [[VAL]], <8 x i16>* @us
  us = +us2;

// CHECK: [[VAL:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si2
// CHECK: store volatile <4 x i32> [[VAL]], <4 x i32>* @si
  si = +si2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui2
// CHECK: store volatile <4 x i32> [[VAL]], <4 x i32>* @ui
  ui = +ui2;

// CHECK: [[VAL:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl2
// CHECK: store volatile <2 x i64> [[VAL]], <2 x i64>* @sl
  sl = +sl2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul2
// CHECK: store volatile <2 x i64> [[VAL]], <2 x i64>* @ul
  ul = +ul2;

// CHECK: [[VAL:%[^ ]+]] = load volatile <2 x double>, <2 x double>* @fd2
// CHECK: store volatile <2 x double> [[VAL]], <2 x double>* @fd
  fd = +fd2;
}

void test_neg (void)
{
// CHECK-LABEL: test_neg

// CHECK: [[VAL:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc2
// CHECK: %{{.*}} = sub <16 x i8> zeroinitializer, [[VAL]]
  sc = -sc2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss2
// CHECK: %{{.*}} = sub <8 x i16> zeroinitializer, [[VAL]]
  ss = -ss2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si2
// CHECK: %{{.*}} = sub <4 x i32> zeroinitializer, [[VAL]]
  si = -si2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl2
// CHECK: %{{.*}} = sub <2 x i64> zeroinitializer, [[VAL]]
  sl = -sl2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <2 x double>, <2 x double>* @fd2
// CHECK: %{{.*}} = fsub <2 x double> <double -0.000000e+00, double -0.000000e+00>, [[VAL]]
  fd = -fd2;
}

void test_preinc (void)
{
// CHECK-LABEL: test_preinc

// CHECK: [[VAL:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc2
// CHECK: %{{.*}} = add <16 x i8> [[VAL]], <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  ++sc2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc2
// CHECK: %{{.*}} = add <16 x i8> [[VAL]], <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  ++uc2;

// CHECK: [[VAL:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss2
// CHECK: %{{.*}} = add <8 x i16> [[VAL]], <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  ++ss2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us2
// CHECK: %{{.*}} = add <8 x i16> [[VAL]], <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  ++us2;

// CHECK: [[VAL:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si2
// CHECK: %{{.*}} = add <4 x i32> [[VAL]], <i32 1, i32 1, i32 1, i32 1>
  ++si2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui2
// CHECK: %{{.*}} = add <4 x i32> [[VAL]], <i32 1, i32 1, i32 1, i32 1>
  ++ui2;

// CHECK: [[VAL:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl2
// CHECK: %{{.*}} = add <2 x i64> [[VAL]], <i64 1, i64 1>
  ++sl2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul2
// CHECK: %{{.*}} = add <2 x i64> [[VAL]], <i64 1, i64 1>
  ++ul2;

// CHECK: [[VAL:%[^ ]+]] = load volatile <2 x double>, <2 x double>* @fd2
// CHECK: %{{.*}} = fadd <2 x double> [[VAL]], <double 1.000000e+00, double 1.000000e+00>
  ++fd2;
}

void test_postinc (void)
{
// CHECK-LABEL: test_postinc

// CHECK: [[VAL:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc2
// CHECK: %{{.*}} = add <16 x i8> [[VAL]], <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  sc2++;
// CHECK: [[VAL:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc2
// CHECK: %{{.*}} = add <16 x i8> [[VAL]], <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  uc2++;

// CHECK: [[VAL:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss2
// CHECK: %{{.*}} = add <8 x i16> [[VAL]], <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  ss2++;
// CHECK: [[VAL:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us2
// CHECK: %{{.*}} = add <8 x i16> [[VAL]], <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  us2++;

// CHECK: [[VAL:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si2
// CHECK: %{{.*}} = add <4 x i32> [[VAL]], <i32 1, i32 1, i32 1, i32 1>
  si2++;
// CHECK: [[VAL:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui2
// CHECK: %{{.*}} = add <4 x i32> [[VAL]], <i32 1, i32 1, i32 1, i32 1>
  ui2++;

// CHECK: [[VAL:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl2
// CHECK: %{{.*}} = add <2 x i64> [[VAL]], <i64 1, i64 1>
  sl2++;
// CHECK: [[VAL:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul2
// CHECK: %{{.*}} = add <2 x i64> [[VAL]], <i64 1, i64 1>
  ul2++;

// CHECK: [[VAL:%[^ ]+]] = load volatile <2 x double>, <2 x double>* @fd2
// CHECK: %{{.*}} = fadd <2 x double> [[VAL]], <double 1.000000e+00, double 1.000000e+00>
  fd2++;
}

void test_predec (void)
{
// CHECK-LABEL: test_predec

// CHECK: [[VAL:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc2
// CHECK: %{{.*}} = add <16 x i8> [[VAL]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
  --sc2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc2
// CHECK: %{{.*}} = add <16 x i8> [[VAL]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
  --uc2;

// CHECK: [[VAL:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss2
// CHECK: %{{.*}} = add <8 x i16> [[VAL]], <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
  --ss2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us2
// CHECK: %{{.*}} = add <8 x i16> [[VAL]], <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
  --us2;

// CHECK: [[VAL:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si2
// CHECK: %{{.*}} = add <4 x i32> [[VAL]], <i32 -1, i32 -1, i32 -1, i32 -1>
  --si2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui2
// CHECK: %{{.*}} = add <4 x i32> [[VAL]], <i32 -1, i32 -1, i32 -1, i32 -1>
  --ui2;

// CHECK: [[VAL:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl2
// CHECK: %{{.*}} = add <2 x i64> [[VAL]], <i64 -1, i64 -1>
  --sl2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul2
// CHECK: %{{.*}} = add <2 x i64> [[VAL]], <i64 -1, i64 -1>
  --ul2;

// CHECK: [[VAL:%[^ ]+]] = load volatile <2 x double>, <2 x double>* @fd2
// CHECK: %{{.*}} = fadd <2 x double> [[VAL]], <double -1.000000e+00, double -1.000000e+00>
  --fd2;
}

void test_postdec (void)
{
// CHECK-LABEL: test_postdec

// CHECK: [[VAL:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc2
// CHECK: %{{.*}} = add <16 x i8> [[VAL]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
  sc2--;
// CHECK: [[VAL:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc2
// CHECK: %{{.*}} = add <16 x i8> [[VAL]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
  uc2--;

// CHECK: [[VAL:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss2
// CHECK: %{{.*}} = add <8 x i16> [[VAL]], <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
  ss2--;
// CHECK: [[VAL:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us2
// CHECK: %{{.*}} = add <8 x i16> [[VAL]], <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
  us2--;

// CHECK: [[VAL:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si2
// CHECK: %{{.*}} = add <4 x i32> [[VAL]], <i32 -1, i32 -1, i32 -1, i32 -1>
  si2--;
// CHECK: [[VAL:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui2
// CHECK: %{{.*}} = add <4 x i32> [[VAL]], <i32 -1, i32 -1, i32 -1, i32 -1>
  ui2--;

// CHECK: [[VAL:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl2
// CHECK: %{{.*}} = add <2 x i64> [[VAL]], <i64 -1, i64 -1>
  sl2--;
// CHECK: [[VAL:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul2
// CHECK: %{{.*}} = add <2 x i64> [[VAL]], <i64 -1, i64 -1>
  ul2--;

// CHECK: [[VAL:%[^ ]+]] = load volatile <2 x double>, <2 x double>* @fd2
// CHECK: %{{.*}} = fadd <2 x double> [[VAL]], <double -1.000000e+00, double -1.000000e+00>
  fd2--;
}

void test_add (void)
{
// CHECK-LABEL: test_add

// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc2
// CHECK: %{{.*}} = add <16 x i8> [[VAL2]], [[VAL1]]
  sc = sc + sc2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc2
// CHECK: %{{.*}} = add <16 x i8> [[VAL2]], [[VAL1]]
  sc = sc + bc2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc2
// CHECK: %{{.*}} = add <16 x i8> [[VAL2]], [[VAL1]]
  sc = bc + sc2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc2
// CHECK: %{{.*}} = add <16 x i8> [[VAL2]], [[VAL1]]
  uc = uc + uc2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc2
// CHECK: %{{.*}} = add <16 x i8> [[VAL2]], [[VAL1]]
  uc = uc + bc2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc2
// CHECK: %{{.*}} = add <16 x i8> [[VAL2]], [[VAL1]]
  uc = bc + uc2;

// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss2
// CHECK: %{{.*}} = add <8 x i16> [[VAL2]], [[VAL1]]
  ss = ss + ss2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs2
// CHECK: %{{.*}} = add <8 x i16> [[VAL2]], [[VAL1]]
  ss = ss + bs2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss2
// CHECK: %{{.*}} = add <8 x i16> [[VAL2]], [[VAL1]]
  ss = bs + ss2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us2
// CHECK: %{{.*}} = add <8 x i16> [[VAL2]], [[VAL1]]
  us = us + us2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs2
// CHECK: %{{.*}} = add <8 x i16> [[VAL2]], [[VAL1]]
  us = us + bs2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us2
// CHECK: %{{.*}} = add <8 x i16> [[VAL2]], [[VAL1]]
  us = bs + us2;

// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si2
// CHECK: %{{.*}} = add <4 x i32> [[VAL2]], [[VAL1]]
  si = si + si2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi2
// CHECK: %{{.*}} = add <4 x i32> [[VAL2]], [[VAL1]]
  si = si + bi2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si2
// CHECK: %{{.*}} = add <4 x i32> [[VAL2]], [[VAL1]]
  si = bi + si2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui2
// CHECK: %{{.*}} = add <4 x i32> [[VAL2]], [[VAL1]]
  ui = ui + ui2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi2
// CHECK: %{{.*}} = add <4 x i32> [[VAL2]], [[VAL1]]
  ui = ui + bi2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui2
// CHECK: %{{.*}} = add <4 x i32> [[VAL2]], [[VAL1]]
  ui = bi + ui2;

// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl2
// CHECK: %{{.*}} = add <2 x i64> [[VAL2]], [[VAL1]]
  sl = sl + sl2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl2
// CHECK: %{{.*}} = add <2 x i64> [[VAL2]], [[VAL1]]
  sl = sl + bl2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl2
// CHECK: %{{.*}} = add <2 x i64> [[VAL2]], [[VAL1]]
  sl = bl + sl2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul2
// CHECK: %{{.*}} = add <2 x i64> [[VAL2]], [[VAL1]]
  ul = ul + ul2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl2
// CHECK: %{{.*}} = add <2 x i64> [[VAL2]], [[VAL1]]
  ul = ul + bl2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul2
// CHECK: %{{.*}} = add <2 x i64> [[VAL2]], [[VAL1]]
  ul = bl + ul2;

// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x double>, <2 x double>* @fd
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x double>, <2 x double>* @fd2
// CHECK: %{{.*}} = fadd <2 x double> [[VAL1]], [[VAL2]]
  fd = fd + fd2;
}

void test_add_assign (void)
{
// CHECK-LABEL: test_add_assign

// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc
// CHECK: %{{.*}} = add <16 x i8> [[VAL1]], [[VAL2]]
  sc += sc2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc
// CHECK: %{{.*}} = add <16 x i8> [[VAL1]], [[VAL2]]
  sc += bc2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc
// CHECK: %{{.*}} = add <16 x i8> [[VAL1]], [[VAL2]]
  uc += uc2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc
// CHECK: %{{.*}} = add <16 x i8> [[VAL1]], [[VAL2]]
  uc += bc2;

// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss
// CHECK: %{{.*}} = add <8 x i16> [[VAL1]], [[VAL2]]
  ss += ss2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss
// CHECK: %{{.*}} = add <8 x i16> [[VAL1]], [[VAL2]]
  ss += bs2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us
// CHECK: %{{.*}} = add <8 x i16> [[VAL1]], [[VAL2]]
  us += us2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us
// CHECK: %{{.*}} = add <8 x i16> [[VAL1]], [[VAL2]]
  us += bs2;

// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si
// CHECK: %{{.*}} = add <4 x i32> [[VAL1]], [[VAL2]]
  si += si2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si
// CHECK: %{{.*}} = add <4 x i32> [[VAL1]], [[VAL2]]
  si += bi2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui
// CHECK: %{{.*}} = add <4 x i32> [[VAL1]], [[VAL2]]
  ui += ui2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui
// CHECK: %{{.*}} = add <4 x i32> [[VAL1]], [[VAL2]]
  ui += bi2;

// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl
// CHECK: %{{.*}} = add <2 x i64> [[VAL1]], [[VAL2]]
  sl += sl2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl
// CHECK: %{{.*}} = add <2 x i64> [[VAL1]], [[VAL2]]
  sl += bl2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul
// CHECK: %{{.*}} = add <2 x i64> [[VAL1]], [[VAL2]]
  ul += ul2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul
// CHECK: %{{.*}} = add <2 x i64> [[VAL1]], [[VAL2]]
  ul += bl2;

// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x double>, <2 x double>* @fd2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x double>, <2 x double>* @fd
// CHECK: %{{.*}} = fadd <2 x double> [[VAL2]], [[VAL1]]
  fd += fd2;
}

void test_sub (void)
{
// CHECK-LABEL: test_sub

// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc2
// CHECK: %{{.*}} = sub <16 x i8> [[VAL1]], [[VAL2]]
  sc = sc - sc2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc2
// CHECK: %{{.*}} = sub <16 x i8> [[VAL1]], [[VAL2]]
  sc = sc - bc2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc2
// CHECK: %{{.*}} = sub <16 x i8> [[VAL1]], [[VAL2]]
  sc = bc - sc2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc2
// CHECK: %{{.*}} = sub <16 x i8> [[VAL1]], [[VAL2]]
  uc = uc - uc2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc2
// CHECK: %{{.*}} = sub <16 x i8> [[VAL1]], [[VAL2]]
  uc = uc - bc2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc2
// CHECK: %{{.*}} = sub <16 x i8> [[VAL1]], [[VAL2]]
  uc = bc - uc2;

// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss2
// CHECK: %{{.*}} = sub <8 x i16> [[VAL1]], [[VAL2]]
  ss = ss - ss2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs2
// CHECK: %{{.*}} = sub <8 x i16> [[VAL1]], [[VAL2]]
  ss = ss - bs2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss2
// CHECK: %{{.*}} = sub <8 x i16> [[VAL1]], [[VAL2]]
  ss = bs - ss2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us2
// CHECK: %{{.*}} = sub <8 x i16> [[VAL1]], [[VAL2]]
  us = us - us2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs2
// CHECK: %{{.*}} = sub <8 x i16> [[VAL1]], [[VAL2]]
  us = us - bs2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us2
// CHECK: %{{.*}} = sub <8 x i16> [[VAL1]], [[VAL2]]
  us = bs - us2;

// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si2
// CHECK: %{{.*}} = sub <4 x i32> [[VAL1]], [[VAL2]]
  si = si - si2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi2
// CHECK: %{{.*}} = sub <4 x i32> [[VAL1]], [[VAL2]]
  si = si - bi2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si2
// CHECK: %{{.*}} = sub <4 x i32> [[VAL1]], [[VAL2]]
  si = bi - si2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui2
// CHECK: %{{.*}} = sub <4 x i32> [[VAL1]], [[VAL2]]
  ui = ui - ui2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi2
// CHECK: %{{.*}} = sub <4 x i32> [[VAL1]], [[VAL2]]
  ui = ui - bi2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui2
// CHECK: %{{.*}} = sub <4 x i32> [[VAL1]], [[VAL2]]
  ui = bi - ui2;

// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl2
// CHECK: %{{.*}} = sub <2 x i64> [[VAL1]], [[VAL2]]
  sl = sl - sl2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl2
// CHECK: %{{.*}} = sub <2 x i64> [[VAL1]], [[VAL2]]
  sl = sl - bl2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl2
// CHECK: %{{.*}} = sub <2 x i64> [[VAL1]], [[VAL2]]
  sl = bl - sl2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul2
// CHECK: %{{.*}} = sub <2 x i64> [[VAL1]], [[VAL2]]
  ul = ul - ul2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl2
// CHECK: %{{.*}} = sub <2 x i64> [[VAL1]], [[VAL2]]
  ul = ul - bl2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul2
// CHECK: %{{.*}} = sub <2 x i64> [[VAL1]], [[VAL2]]
  ul = bl - ul2;

// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x double>, <2 x double>* @fd
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x double>, <2 x double>* @fd2
// CHECK: %{{.*}} = fsub <2 x double> [[VAL1]], [[VAL2]]
  fd = fd - fd2;
}

void test_sub_assign (void)
{
// CHECK-LABEL: test_sub_assign

// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc
// CHECK: %{{.*}} = sub <16 x i8> [[VAL1]], [[VAL2]]
  sc -= sc2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc
// CHECK: %{{.*}} = sub <16 x i8> [[VAL1]], [[VAL2]]
  sc -= bc2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc
// CHECK: %{{.*}} = sub <16 x i8> [[VAL1]], [[VAL2]]
  uc -= uc2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc
// CHECK: %{{.*}} = sub <16 x i8> [[VAL1]], [[VAL2]]
  uc -= bc2;

// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss
// CHECK: %{{.*}} = sub <8 x i16> [[VAL1]], [[VAL2]]
  ss -= ss2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss
// CHECK: %{{.*}} = sub <8 x i16> [[VAL1]], [[VAL2]]
  ss -= bs2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us
// CHECK: %{{.*}} = sub <8 x i16> [[VAL1]], [[VAL2]]
  us -= us2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us
// CHECK: %{{.*}} = sub <8 x i16> [[VAL1]], [[VAL2]]
  us -= bs2;

// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si
// CHECK: %{{.*}} = sub <4 x i32> [[VAL1]], [[VAL2]]
  si -= si2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si
// CHECK: %{{.*}} = sub <4 x i32> [[VAL1]], [[VAL2]]
  si -= bi2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui
// CHECK: %{{.*}} = sub <4 x i32> [[VAL1]], [[VAL2]]
  ui -= ui2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui
// CHECK: %{{.*}} = sub <4 x i32> [[VAL1]], [[VAL2]]
  ui -= bi2;

// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl
// CHECK: %{{.*}} = sub <2 x i64> [[VAL1]], [[VAL2]]
  sl -= sl2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl
// CHECK: %{{.*}} = sub <2 x i64> [[VAL1]], [[VAL2]]
  sl -= bl2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul
// CHECK: %{{.*}} = sub <2 x i64> [[VAL1]], [[VAL2]]
  ul -= ul2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul
// CHECK: %{{.*}} = sub <2 x i64> [[VAL1]], [[VAL2]]
  ul -= bl2;

// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x double>, <2 x double>* @fd2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x double>, <2 x double>* @fd
// CHECK: %{{.*}} = fsub <2 x double> [[VAL1]], [[VAL2]]
  fd -= fd2;
}

void test_mul (void)
{
// CHECK-LABEL: test_mul

// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc2
// CHECK: %{{.*}} = mul <16 x i8> [[VAL2]], [[VAL1]]
  sc = sc * sc2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc2
// CHECK: %{{.*}} = mul <16 x i8> [[VAL2]], [[VAL1]]
  uc = uc * uc2;

// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss2
// CHECK: %{{.*}} = mul <8 x i16> [[VAL2]], [[VAL1]]
  ss = ss * ss2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us2
// CHECK: %{{.*}} = mul <8 x i16> [[VAL2]], [[VAL1]]
  us = us * us2;

// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si2
// CHECK: %{{.*}} = mul <4 x i32> [[VAL2]], [[VAL1]]
  si = si * si2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui2
// CHECK: %{{.*}} = mul <4 x i32> [[VAL2]], [[VAL1]]
  ui = ui * ui2;

// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl2
// CHECK: %{{.*}} = mul <2 x i64> [[VAL2]], [[VAL1]]
  sl = sl * sl2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul2
// CHECK: %{{.*}} = mul <2 x i64> [[VAL2]], [[VAL1]]
  ul = ul * ul2;

// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x double>, <2 x double>* @fd
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x double>, <2 x double>* @fd2
// CHECK: %{{.*}} = fmul <2 x double> [[VAL1]], [[VAL2]]
  fd = fd * fd2;
}

void test_mul_assign (void)
{
// CHECK-LABEL: test_mul_assign

// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc
// CHECK: %{{.*}} = mul <16 x i8> [[VAL1]], [[VAL2]]
  sc *= sc2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc
// CHECK: %{{.*}} = mul <16 x i8> [[VAL1]], [[VAL2]]
  uc *= uc2;

// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss
// CHECK: %{{.*}} = mul <8 x i16> [[VAL1]], [[VAL2]]
  ss *= ss2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us
// CHECK: %{{.*}} = mul <8 x i16> [[VAL1]], [[VAL2]]
  us *= us2;

// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si
// CHECK: %{{.*}} = mul <4 x i32> [[VAL1]], [[VAL2]]
  si *= si2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui
// CHECK: %{{.*}} = mul <4 x i32> [[VAL1]], [[VAL2]]
  ui *= ui2;

// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl
// CHECK: %{{.*}} = mul <2 x i64> [[VAL1]], [[VAL2]]
  sl *= sl2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul
// CHECK: %{{.*}} = mul <2 x i64> [[VAL1]], [[VAL2]]
  ul *= ul2;

// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x double>, <2 x double>* @fd2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x double>, <2 x double>* @fd
// CHECK: %{{.*}} = fmul <2 x double> [[VAL2]], [[VAL1]]
  fd *= fd2;
}

void test_div (void)
{
// CHECK-LABEL: test_div

// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc2
// CHECK: %{{.*}} = sdiv <16 x i8> [[VAL1]], [[VAL2]]
  sc = sc / sc2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc2
// CHECK: %{{.*}} = udiv <16 x i8> [[VAL1]], [[VAL2]]
  uc = uc / uc2;

// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss2
// CHECK: %{{.*}} = sdiv <8 x i16> [[VAL1]], [[VAL2]]
  ss = ss / ss2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us2
// CHECK: %{{.*}} = udiv <8 x i16> [[VAL1]], [[VAL2]]
  us = us / us2;

// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si2
// CHECK: %{{.*}} = sdiv <4 x i32> [[VAL1]], [[VAL2]]
  si = si / si2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui2
// CHECK: %{{.*}} = udiv <4 x i32> [[VAL1]], [[VAL2]]
  ui = ui / ui2;

// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl2
// CHECK: %{{.*}} = sdiv <2 x i64> [[VAL1]], [[VAL2]]
  sl = sl / sl2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul2
// CHECK: %{{.*}} = udiv <2 x i64> [[VAL1]], [[VAL2]]
  ul = ul / ul2;

// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x double>, <2 x double>* @fd
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x double>, <2 x double>* @fd2
// CHECK: %{{.*}} = fdiv <2 x double> [[VAL1]], [[VAL2]]
  fd = fd / fd2;
}

void test_div_assign (void)
{
// CHECK-LABEL: test_div_assign

// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc
// CHECK: %{{.*}} = sdiv <16 x i8> [[VAL1]], [[VAL2]]
  sc /= sc2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc
// CHECK: %{{.*}} = udiv <16 x i8> [[VAL1]], [[VAL2]]
  uc /= uc2;

// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss
// CHECK: %{{.*}} = sdiv <8 x i16> [[VAL1]], [[VAL2]]
  ss /= ss2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us
// CHECK: %{{.*}} = udiv <8 x i16> [[VAL1]], [[VAL2]]
  us /= us2;

// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si
// CHECK: %{{.*}} = sdiv <4 x i32> [[VAL1]], [[VAL2]]
  si /= si2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui
// CHECK: %{{.*}} = udiv <4 x i32> [[VAL1]], [[VAL2]]
  ui /= ui2;

// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl
// CHECK: %{{.*}} = sdiv <2 x i64> [[VAL1]], [[VAL2]]
  sl /= sl2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul
// CHECK: %{{.*}} = udiv <2 x i64> [[VAL1]], [[VAL2]]
  ul /= ul2;

// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x double>, <2 x double>* @fd2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x double>, <2 x double>* @fd
// CHECK: %{{.*}} = fdiv <2 x double> [[VAL1]], [[VAL2]]
  fd /= fd2;
}

void test_rem (void)
{
// CHECK-LABEL: test_rem

// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc2
// CHECK: %{{.*}} = srem <16 x i8> [[VAL1]], [[VAL2]]
  sc = sc % sc2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc2
// CHECK: %{{.*}} = urem <16 x i8> [[VAL1]], [[VAL2]]
  uc = uc % uc2;

// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss2
// CHECK: %{{.*}} = srem <8 x i16> [[VAL1]], [[VAL2]]
  ss = ss % ss2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us2
// CHECK: %{{.*}} = urem <8 x i16> [[VAL1]], [[VAL2]]
  us = us % us2;

// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si2
// CHECK: %{{.*}} = srem <4 x i32> [[VAL1]], [[VAL2]]
  si = si % si2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui2
// CHECK: %{{.*}} = urem <4 x i32> [[VAL1]], [[VAL2]]
  ui = ui % ui2;

// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl2
// CHECK: %{{.*}} = srem <2 x i64> [[VAL1]], [[VAL2]]
  sl = sl % sl2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul2
// CHECK: %{{.*}} = urem <2 x i64> [[VAL1]], [[VAL2]]
  ul = ul % ul2;
}

void test_rem_assign (void)
{
// CHECK-LABEL: test_rem_assign

// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc
// CHECK: %{{.*}} = srem <16 x i8> [[VAL1]], [[VAL2]]
  sc %= sc2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc
// CHECK: %{{.*}} = urem <16 x i8> [[VAL1]], [[VAL2]]
  uc %= uc2;

// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss
// CHECK: %{{.*}} = srem <8 x i16> [[VAL1]], [[VAL2]]
  ss %= ss2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us
// CHECK: %{{.*}} = urem <8 x i16> [[VAL1]], [[VAL2]]
  us %= us2;

// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si
// CHECK: %{{.*}} = srem <4 x i32> [[VAL1]], [[VAL2]]
  si %= si2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui
// CHECK: %{{.*}} = urem <4 x i32> [[VAL1]], [[VAL2]]
  ui %= ui2;

// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl
// CHECK: %{{.*}} = srem <2 x i64> [[VAL1]], [[VAL2]]
  sl %= sl2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul
// CHECK: %{{.*}} = urem <2 x i64> [[VAL1]], [[VAL2]]
  ul %= ul2;
}

void test_not (void)
{
// CHECK-LABEL: test_not

// CHECK: [[VAL:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc2
// CHECK: %{{.*}} = xor <16 x i8> [[VAL]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
  sc = ~sc2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc2
// CHECK: %{{.*}} = xor <16 x i8> [[VAL]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
  uc = ~uc2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc2
// CHECK: %{{.*}} = xor <16 x i8> [[VAL]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
  bc = ~bc2;

// CHECK: [[VAL:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss2
// CHECK: %{{.*}} = xor <8 x i16> [[VAL]], <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
  ss = ~ss2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us2
// CHECK: %{{.*}} = xor <8 x i16> [[VAL]], <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
  us = ~us2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs2
// CHECK: %{{.*}} = xor <8 x i16> [[VAL]], <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
  bs = ~bs2;

// CHECK: [[VAL:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si2
// CHECK: %{{.*}} = xor <4 x i32> [[VAL]], <i32 -1, i32 -1, i32 -1, i32 -1>
  si = ~si2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui2
// CHECK: %{{.*}} = xor <4 x i32> [[VAL]], <i32 -1, i32 -1, i32 -1, i32 -1>
  ui = ~ui2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi2
// CHECK: %{{.*}} = xor <4 x i32> [[VAL]], <i32 -1, i32 -1, i32 -1, i32 -1>
  bi = ~bi2;

// CHECK: [[VAL:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl2
// CHECK: %{{.*}} = xor <2 x i64> [[VAL]], <i64 -1, i64 -1>
  sl = ~sl2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul2
// CHECK: %{{.*}} = xor <2 x i64> [[VAL]], <i64 -1, i64 -1>
  ul = ~ul2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl2
// CHECK: %{{.*}} = xor <2 x i64> [[VAL]], <i64 -1, i64 -1>
  bl = ~bl2;
}

void test_and (void)
{
// CHECK-LABEL: test_and

// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc2
// CHECK: %{{.*}} = and <16 x i8> [[VAL2]], [[VAL1]]
  sc = sc & sc2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc2
// CHECK: %{{.*}} = and <16 x i8> [[VAL2]], [[VAL1]]
  sc = sc & bc2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc2
// CHECK: %{{.*}} = and <16 x i8> [[VAL2]], [[VAL1]]
  sc = bc & sc2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc2
// CHECK: %{{.*}} = and <16 x i8> [[VAL2]], [[VAL1]]
  uc = uc & uc2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc2
// CHECK: %{{.*}} = and <16 x i8> [[VAL2]], [[VAL1]]
  uc = uc & bc2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc2
// CHECK: %{{.*}} = and <16 x i8> [[VAL2]], [[VAL1]]
  uc = bc & uc2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc2
// CHECK: %{{.*}} = and <16 x i8> [[VAL2]], [[VAL1]]
  bc = bc & bc2;

// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss2
// CHECK: %{{.*}} = and <8 x i16> [[VAL2]], [[VAL1]]
  ss = ss & ss2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs2
// CHECK: %{{.*}} = and <8 x i16> [[VAL2]], [[VAL1]]
  ss = ss & bs2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss2
// CHECK: %{{.*}} = and <8 x i16> [[VAL2]], [[VAL1]]
  ss = bs & ss2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us2
// CHECK: %{{.*}} = and <8 x i16> [[VAL2]], [[VAL1]]
  us = us & us2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs2
// CHECK: %{{.*}} = and <8 x i16> [[VAL2]], [[VAL1]]
  us = us & bs2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us2
// CHECK: %{{.*}} = and <8 x i16> [[VAL2]], [[VAL1]]
  us = bs & us2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs2
// CHECK: %{{.*}} = and <8 x i16> [[VAL2]], [[VAL1]]
  bs = bs & bs2;

// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si2
// CHECK: %{{.*}} = and <4 x i32> [[VAL2]], [[VAL1]]
  si = si & si2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi2
// CHECK: %{{.*}} = and <4 x i32> [[VAL2]], [[VAL1]]
  si = si & bi2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si2
// CHECK: %{{.*}} = and <4 x i32> [[VAL2]], [[VAL1]]
  si = bi & si2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui2
// CHECK: %{{.*}} = and <4 x i32> [[VAL2]], [[VAL1]]
  ui = ui & ui2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi2
// CHECK: %{{.*}} = and <4 x i32> [[VAL2]], [[VAL1]]
  ui = ui & bi2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui2
// CHECK: %{{.*}} = and <4 x i32> [[VAL2]], [[VAL1]]
  ui = bi & ui2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi2
// CHECK: %{{.*}} = and <4 x i32> [[VAL2]], [[VAL1]]
  bi = bi & bi2;

// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl2
// CHECK: %{{.*}} = and <2 x i64> [[VAL2]], [[VAL1]]
  sl = sl & sl2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl2
// CHECK: %{{.*}} = and <2 x i64> [[VAL2]], [[VAL1]]
  sl = sl & bl2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl2
// CHECK: %{{.*}} = and <2 x i64> [[VAL2]], [[VAL1]]
  sl = bl & sl2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul2
// CHECK: %{{.*}} = and <2 x i64> [[VAL2]], [[VAL1]]
  ul = ul & ul2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl2
// CHECK: %{{.*}} = and <2 x i64> [[VAL2]], [[VAL1]]
  ul = ul & bl2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul2
// CHECK: %{{.*}} = and <2 x i64> [[VAL2]], [[VAL1]]
  ul = bl & ul2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl2
// CHECK: %{{.*}} = and <2 x i64> [[VAL2]], [[VAL1]]
  bl = bl & bl2;
}

void test_and_assign (void)
{
// CHECK-LABEL: test_and_assign

// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc
// CHECK: %{{.*}} = and <16 x i8> [[VAL1]], [[VAL2]]
  sc &= sc2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc
// CHECK: %{{.*}} = and <16 x i8> [[VAL1]], [[VAL2]]
  sc &= bc2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc
// CHECK: %{{.*}} = and <16 x i8> [[VAL1]], [[VAL2]]
  uc &= uc2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc
// CHECK: %{{.*}} = and <16 x i8> [[VAL1]], [[VAL2]]
  uc &= bc2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc
// CHECK: %{{.*}} = and <16 x i8> [[VAL1]], [[VAL2]]
  bc &= bc2;

// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss
// CHECK: %{{.*}} = and <8 x i16> [[VAL1]], [[VAL2]]
  ss &= ss2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss
// CHECK: %{{.*}} = and <8 x i16> [[VAL1]], [[VAL2]]
  ss &= bs2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us
// CHECK: %{{.*}} = and <8 x i16> [[VAL1]], [[VAL2]]
  us &= us2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us
// CHECK: %{{.*}} = and <8 x i16> [[VAL1]], [[VAL2]]
  us &= bs2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs
// CHECK: %{{.*}} = and <8 x i16> [[VAL1]], [[VAL2]]
  bs &= bs2;

// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si
// CHECK: %{{.*}} = and <4 x i32> [[VAL1]], [[VAL2]]
  si &= si2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si
// CHECK: %{{.*}} = and <4 x i32> [[VAL1]], [[VAL2]]
  si &= bi2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui
// CHECK: %{{.*}} = and <4 x i32> [[VAL1]], [[VAL2]]
  ui &= ui2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui
// CHECK: %{{.*}} = and <4 x i32> [[VAL1]], [[VAL2]]
  ui &= bi2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi
// CHECK: %{{.*}} = and <4 x i32> [[VAL1]], [[VAL2]]
  bi &= bi2;

// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl
// CHECK: %{{.*}} = and <2 x i64> [[VAL1]], [[VAL2]]
  sl &= sl2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl
// CHECK: %{{.*}} = and <2 x i64> [[VAL1]], [[VAL2]]
  sl &= bl2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul
// CHECK: %{{.*}} = and <2 x i64> [[VAL1]], [[VAL2]]
  ul &= ul2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul
// CHECK: %{{.*}} = and <2 x i64> [[VAL1]], [[VAL2]]
  ul &= bl2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl
// CHECK: %{{.*}} = and <2 x i64> [[VAL1]], [[VAL2]]
  bl &= bl2;
}

void test_or (void)
{
// CHECK-LABEL: test_or

// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc2
// CHECK: %{{.*}} = or <16 x i8> [[VAL2]], [[VAL1]]
  sc = sc | sc2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc2
// CHECK: %{{.*}} = or <16 x i8> [[VAL2]], [[VAL1]]
  sc = sc | bc2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc2
// CHECK: %{{.*}} = or <16 x i8> [[VAL2]], [[VAL1]]
  sc = bc | sc2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc2
// CHECK: %{{.*}} = or <16 x i8> [[VAL2]], [[VAL1]]
  uc = uc | uc2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc2
// CHECK: %{{.*}} = or <16 x i8> [[VAL2]], [[VAL1]]
  uc = uc | bc2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc2
// CHECK: %{{.*}} = or <16 x i8> [[VAL2]], [[VAL1]]
  uc = bc | uc2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc2
// CHECK: %{{.*}} = or <16 x i8> [[VAL2]], [[VAL1]]
  bc = bc | bc2;

// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss2
// CHECK: %{{.*}} = or <8 x i16> [[VAL2]], [[VAL1]]
  ss = ss | ss2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs2
// CHECK: %{{.*}} = or <8 x i16> [[VAL2]], [[VAL1]]
  ss = ss | bs2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss2
// CHECK: %{{.*}} = or <8 x i16> [[VAL2]], [[VAL1]]
  ss = bs | ss2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us2
// CHECK: %{{.*}} = or <8 x i16> [[VAL2]], [[VAL1]]
  us = us | us2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs2
// CHECK: %{{.*}} = or <8 x i16> [[VAL2]], [[VAL1]]
  us = us | bs2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us2
// CHECK: %{{.*}} = or <8 x i16> [[VAL2]], [[VAL1]]
  us = bs | us2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs2
// CHECK: %{{.*}} = or <8 x i16> [[VAL2]], [[VAL1]]
  bs = bs | bs2;

// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si2
// CHECK: %{{.*}} = or <4 x i32> [[VAL2]], [[VAL1]]
  si = si | si2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi2
// CHECK: %{{.*}} = or <4 x i32> [[VAL2]], [[VAL1]]
  si = si | bi2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si2
// CHECK: %{{.*}} = or <4 x i32> [[VAL2]], [[VAL1]]
  si = bi | si2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui2
// CHECK: %{{.*}} = or <4 x i32> [[VAL2]], [[VAL1]]
  ui = ui | ui2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi2
// CHECK: %{{.*}} = or <4 x i32> [[VAL2]], [[VAL1]]
  ui = ui | bi2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui2
// CHECK: %{{.*}} = or <4 x i32> [[VAL2]], [[VAL1]]
  ui = bi | ui2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi2
// CHECK: %{{.*}} = or <4 x i32> [[VAL2]], [[VAL1]]
  bi = bi | bi2;

// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl2
// CHECK: %{{.*}} = or <2 x i64> [[VAL2]], [[VAL1]]
  sl = sl | sl2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl2
// CHECK: %{{.*}} = or <2 x i64> [[VAL2]], [[VAL1]]
  sl = sl | bl2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl2
// CHECK: %{{.*}} = or <2 x i64> [[VAL2]], [[VAL1]]
  sl = bl | sl2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul2
// CHECK: %{{.*}} = or <2 x i64> [[VAL2]], [[VAL1]]
  ul = ul | ul2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl2
// CHECK: %{{.*}} = or <2 x i64> [[VAL2]], [[VAL1]]
  ul = ul | bl2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul2
// CHECK: %{{.*}} = or <2 x i64> [[VAL2]], [[VAL1]]
  ul = bl | ul2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl2
// CHECK: %{{.*}} = or <2 x i64> [[VAL2]], [[VAL1]]
  bl = bl | bl2;
}

void test_or_assign (void)
{
// CHECK-LABEL: test_or_assign

// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc
// CHECK: %{{.*}} = or <16 x i8> [[VAL1]], [[VAL2]]
  sc |= sc2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc
// CHECK: %{{.*}} = or <16 x i8> [[VAL1]], [[VAL2]]
  sc |= bc2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc
// CHECK: %{{.*}} = or <16 x i8> [[VAL1]], [[VAL2]]
  uc |= uc2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc
// CHECK: %{{.*}} = or <16 x i8> [[VAL1]], [[VAL2]]
  uc |= bc2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc
// CHECK: %{{.*}} = or <16 x i8> [[VAL1]], [[VAL2]]
  bc |= bc2;

// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss
// CHECK: %{{.*}} = or <8 x i16> [[VAL1]], [[VAL2]]
  ss |= ss2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss
// CHECK: %{{.*}} = or <8 x i16> [[VAL1]], [[VAL2]]
  ss |= bs2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us
// CHECK: %{{.*}} = or <8 x i16> [[VAL1]], [[VAL2]]
  us |= us2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us
// CHECK: %{{.*}} = or <8 x i16> [[VAL1]], [[VAL2]]
  us |= bs2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs
// CHECK: %{{.*}} = or <8 x i16> [[VAL1]], [[VAL2]]
  bs |= bs2;

// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si
// CHECK: %{{.*}} = or <4 x i32> [[VAL1]], [[VAL2]]
  si |= si2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si
// CHECK: %{{.*}} = or <4 x i32> [[VAL1]], [[VAL2]]
  si |= bi2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui
// CHECK: %{{.*}} = or <4 x i32> [[VAL1]], [[VAL2]]
  ui |= ui2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui
// CHECK: %{{.*}} = or <4 x i32> [[VAL1]], [[VAL2]]
  ui |= bi2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi
// CHECK: %{{.*}} = or <4 x i32> [[VAL1]], [[VAL2]]
  bi |= bi2;

// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl
// CHECK: %{{.*}} = or <2 x i64> [[VAL1]], [[VAL2]]
  sl |= sl2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl
// CHECK: %{{.*}} = or <2 x i64> [[VAL1]], [[VAL2]]
  sl |= bl2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul
// CHECK: %{{.*}} = or <2 x i64> [[VAL1]], [[VAL2]]
  ul |= ul2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul
// CHECK: %{{.*}} = or <2 x i64> [[VAL1]], [[VAL2]]
  ul |= bl2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl
// CHECK: %{{.*}} = or <2 x i64> [[VAL1]], [[VAL2]]
  bl |= bl2;
}

void test_xor (void)
{
// CHECK-LABEL: test_xor

// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc2
// CHECK: %{{.*}} = xor <16 x i8> [[VAL1]], [[VAL2]]
  sc = sc ^ sc2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc2
// CHECK: %{{.*}} = xor <16 x i8> [[VAL1]], [[VAL2]]
  sc = sc ^ bc2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc2
// CHECK: %{{.*}} = xor <16 x i8> [[VAL1]], [[VAL2]]
  sc = bc ^ sc2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc2
// CHECK: %{{.*}} = xor <16 x i8> [[VAL1]], [[VAL2]]
  uc = uc ^ uc2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc2
// CHECK: %{{.*}} = xor <16 x i8> [[VAL1]], [[VAL2]]
  uc = uc ^ bc2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc2
// CHECK: %{{.*}} = xor <16 x i8> [[VAL1]], [[VAL2]]
  uc = bc ^ uc2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc2
// CHECK: %{{.*}} = xor <16 x i8> [[VAL1]], [[VAL2]]
  bc = bc ^ bc2;

// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss2
// CHECK: %{{.*}} = xor <8 x i16> [[VAL1]], [[VAL2]]
  ss = ss ^ ss2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs2
// CHECK: %{{.*}} = xor <8 x i16> [[VAL1]], [[VAL2]]
  ss = ss ^ bs2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss2
// CHECK: %{{.*}} = xor <8 x i16> [[VAL1]], [[VAL2]]
  ss = bs ^ ss2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us2
// CHECK: %{{.*}} = xor <8 x i16> [[VAL1]], [[VAL2]]
  us = us ^ us2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs2
// CHECK: %{{.*}} = xor <8 x i16> [[VAL1]], [[VAL2]]
  us = us ^ bs2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us2
// CHECK: %{{.*}} = xor <8 x i16> [[VAL1]], [[VAL2]]
  us = bs ^ us2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs2
// CHECK: %{{.*}} = xor <8 x i16> [[VAL1]], [[VAL2]]
  bs = bs ^ bs2;

// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si2
// CHECK: %{{.*}} = xor <4 x i32> [[VAL1]], [[VAL2]]
  si = si ^ si2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi2
// CHECK: %{{.*}} = xor <4 x i32> [[VAL1]], [[VAL2]]
  si = si ^ bi2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si2
// CHECK: %{{.*}} = xor <4 x i32> [[VAL1]], [[VAL2]]
  si = bi ^ si2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui2
// CHECK: %{{.*}} = xor <4 x i32> [[VAL1]], [[VAL2]]
  ui = ui ^ ui2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi2
// CHECK: %{{.*}} = xor <4 x i32> [[VAL1]], [[VAL2]]
  ui = ui ^ bi2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui2
// CHECK: %{{.*}} = xor <4 x i32> [[VAL1]], [[VAL2]]
  ui = bi ^ ui2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi2
// CHECK: %{{.*}} = xor <4 x i32> [[VAL1]], [[VAL2]]
  bi = bi ^ bi2;

// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl2
// CHECK: %{{.*}} = xor <2 x i64> [[VAL1]], [[VAL2]]
  sl = sl ^ sl2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl2
// CHECK: %{{.*}} = xor <2 x i64> [[VAL1]], [[VAL2]]
  sl = sl ^ bl2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl2
// CHECK: %{{.*}} = xor <2 x i64> [[VAL1]], [[VAL2]]
  sl = bl ^ sl2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul2
// CHECK: %{{.*}} = xor <2 x i64> [[VAL1]], [[VAL2]]
  ul = ul ^ ul2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl2
// CHECK: %{{.*}} = xor <2 x i64> [[VAL1]], [[VAL2]]
  ul = ul ^ bl2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul2
// CHECK: %{{.*}} = xor <2 x i64> [[VAL1]], [[VAL2]]
  ul = bl ^ ul2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl2
// CHECK: %{{.*}} = xor <2 x i64> [[VAL1]], [[VAL2]]
  bl = bl ^ bl2;
}

void test_xor_assign (void)
{
// CHECK-LABEL: test_xor_assign

// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc
// CHECK: %{{.*}} = xor <16 x i8> [[VAL2]], [[VAL1]]
  sc ^= sc2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc
// CHECK: %{{.*}} = xor <16 x i8> [[VAL2]], [[VAL1]]
  sc ^= bc2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc
// CHECK: %{{.*}} = xor <16 x i8> [[VAL2]], [[VAL1]]
  uc ^= uc2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc
// CHECK: %{{.*}} = xor <16 x i8> [[VAL2]], [[VAL1]]
  uc ^= bc2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc
// CHECK: %{{.*}} = xor <16 x i8> [[VAL2]], [[VAL1]]
  bc ^= bc2;

// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss
// CHECK: %{{.*}} = xor <8 x i16> [[VAL2]], [[VAL1]]
  ss ^= ss2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss
// CHECK: %{{.*}} = xor <8 x i16> [[VAL2]], [[VAL1]]
  ss ^= bs2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us
// CHECK: %{{.*}} = xor <8 x i16> [[VAL2]], [[VAL1]]
  us ^= us2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us
// CHECK: %{{.*}} = xor <8 x i16> [[VAL2]], [[VAL1]]
  us ^= bs2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs
// CHECK: %{{.*}} = xor <8 x i16> [[VAL2]], [[VAL1]]
  bs ^= bs2;

// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si
// CHECK: %{{.*}} = xor <4 x i32> [[VAL2]], [[VAL1]]
  si ^= si2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si
// CHECK: %{{.*}} = xor <4 x i32> [[VAL2]], [[VAL1]]
  si ^= bi2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui
// CHECK: %{{.*}} = xor <4 x i32> [[VAL2]], [[VAL1]]
  ui ^= ui2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui
// CHECK: %{{.*}} = xor <4 x i32> [[VAL2]], [[VAL1]]
  ui ^= bi2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi
// CHECK: %{{.*}} = xor <4 x i32> [[VAL2]], [[VAL1]]
  bi ^= bi2;

// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl
// CHECK: %{{.*}} = xor <2 x i64> [[VAL2]], [[VAL1]]
  sl ^= sl2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl
// CHECK: %{{.*}} = xor <2 x i64> [[VAL2]], [[VAL1]]
  sl ^= bl2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul
// CHECK: %{{.*}} = xor <2 x i64> [[VAL2]], [[VAL1]]
  ul ^= ul2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul
// CHECK: %{{.*}} = xor <2 x i64> [[VAL2]], [[VAL1]]
  ul ^= bl2;
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl
// CHECK: %{{.*}} = xor <2 x i64> [[VAL2]], [[VAL1]]
  bl ^= bl2;
}

void test_sl (void)
{
// CHECK-LABEL: test_sl

// CHECK: [[VAL:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc
// CHECK: [[CNT:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc2
// CHECK: %{{.*}} = shl <16 x i8> [[VAL]], [[CNT]]
  sc = sc << sc2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc
// CHECK: [[CNT:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc2
// CHECK: %{{.*}} = shl <16 x i8> [[VAL]], [[CNT]]
  sc = sc << uc2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc
// CHECK: [[T1:%[^ ]+]] = load volatile i32, i32* @cnt
// CHECK: [[T2:%[^ ]+]] = insertelement <16 x i32> undef, i32 [[T1]], i32 0
// CHECK: [[T3:%[^ ]+]] = shufflevector <16 x i32> [[T2]], <16 x i32> undef, <16 x i32> zeroinitializer
// CHECK: [[CNT:%[^ ]+]] = trunc <16 x i32> [[T3]] to <16 x i8>
// CHECK: %{{.*}} = shl <16 x i8> [[VAL]], [[CNT]]
  sc = sc << cnt;
// CHECK: [[VAL:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc
// CHECK: %{{.*}} = shl <16 x i8> [[VAL]], <i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5>
  sc = sc << 5;
// CHECK: [[VAL:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc
// CHECK: [[CNT:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc2
// CHECK: %{{.*}} = shl <16 x i8> [[VAL]], [[CNT]]
  uc = uc << sc2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc
// CHECK: [[CNT:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc2
// CHECK: %{{.*}} = shl <16 x i8> [[VAL]], [[CNT]]
  uc = uc << uc2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc
// CHECK: [[T1:%[^ ]+]] = load volatile i32, i32* @cnt
// CHECK: [[T2:%[^ ]+]] = insertelement <16 x i32> undef, i32 [[T1]], i32 0
// CHECK: [[T3:%[^ ]+]] = shufflevector <16 x i32> [[T2]], <16 x i32> undef, <16 x i32> zeroinitializer
// CHECK: [[CNT:%[^ ]+]] = trunc <16 x i32> [[T3]] to <16 x i8>
// CHECK: %{{.*}} = shl <16 x i8> [[VAL]], [[CNT]]
  uc = uc << cnt;
// CHECK: [[VAL:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc
// CHECK: %{{.*}} = shl <16 x i8> [[VAL]], <i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5>
  uc = uc << 5;

// CHECK: [[VAL:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss
// CHECK: [[CNT:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss2
// CHECK: %{{.*}} = shl <8 x i16> [[VAL]], [[CNT]]
  ss = ss << ss2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss
// CHECK: [[CNT:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us2
// CHECK: %{{.*}} = shl <8 x i16> [[VAL]], [[CNT]]
  ss = ss << us2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss
// CHECK: [[T1:%[^ ]+]] = load volatile i32, i32* @cnt
// CHECK: [[T2:%[^ ]+]] = insertelement <8 x i32> undef, i32 [[T1]], i32 0
// CHECK: [[T3:%[^ ]+]] = shufflevector <8 x i32> [[T2]], <8 x i32> undef, <8 x i32> zeroinitializer
// CHECK: [[CNT:%[^ ]+]] = trunc <8 x i32> [[T3]] to <8 x i16>
// CHECK: %{{.*}} = shl <8 x i16> [[VAL]], [[CNT]]
  ss = ss << cnt;
// CHECK: [[VAL:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss
// CHECK: %{{.*}} = shl <8 x i16> [[VAL]], <i16 5, i16 5, i16 5, i16 5, i16 5, i16 5, i16 5, i16 5>
  ss = ss << 5;
// CHECK: [[VAL:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us
// CHECK: [[CNT:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss2
// CHECK: %{{.*}} = shl <8 x i16> [[VAL]], [[CNT]]
  us = us << ss2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us
// CHECK: [[CNT:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us2
// CHECK: %{{.*}} = shl <8 x i16> [[VAL]], [[CNT]]
  us = us << us2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us
// CHECK: [[T1:%[^ ]+]] = load volatile i32, i32* @cnt
// CHECK: [[T2:%[^ ]+]] = insertelement <8 x i32> undef, i32 [[T1]], i32 0
// CHECK: [[T3:%[^ ]+]] = shufflevector <8 x i32> [[T2]], <8 x i32> undef, <8 x i32> zeroinitializer
// CHECK: [[CNT:%[^ ]+]] = trunc <8 x i32> [[T3]] to <8 x i16>
// CHECK: %{{.*}} = shl <8 x i16> [[VAL]], [[CNT]]
  us = us << cnt;
// CHECK: [[VAL:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us
// CHECK: %{{.*}} = shl <8 x i16> [[VAL]], <i16 5, i16 5, i16 5, i16 5, i16 5, i16 5, i16 5, i16 5>
  us = us << 5;

// CHECK: [[VAL:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si
// CHECK: [[CNT:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si2
// CHECK: %{{.*}} = shl <4 x i32> [[VAL]], [[CNT]]
  si = si << si2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si
// CHECK: [[CNT:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui2
// CHECK: %{{.*}} = shl <4 x i32> [[VAL]], [[CNT]]
  si = si << ui2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si
// CHECK: [[T1:%[^ ]+]] = load volatile i32, i32* @cnt
// CHECK: [[T3:%[^ ]+]] = insertelement <4 x i32> undef, i32 [[T1]], i32 0
// CHECK: [[CNT:%[^ ]+]] = shufflevector <4 x i32> [[T3]], <4 x i32> undef, <4 x i32> zeroinitializer
// CHECK: %{{.*}} = shl <4 x i32> [[VAL]], [[CNT]]
  si = si << cnt;
// CHECK: [[VAL:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si
// CHECK: %{{.*}} = shl <4 x i32> [[VAL]], <i32 5, i32 5, i32 5, i32 5>
  si = si << 5;
// CHECK: [[VAL:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui
// CHECK: [[CNT:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si2
// CHECK: %{{.*}} = shl <4 x i32> [[VAL]], [[CNT]]
  ui = ui << si2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui
// CHECK: [[CNT:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui2
// CHECK: %{{.*}} = shl <4 x i32> [[VAL]], [[CNT]]
  ui = ui << ui2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui
// CHECK: [[T1:%[^ ]+]] = load volatile i32, i32* @cnt
// CHECK: [[T3:%[^ ]+]] = insertelement <4 x i32> undef, i32 [[T1]], i32 0
// CHECK: [[CNT:%[^ ]+]] = shufflevector <4 x i32> [[T3]], <4 x i32> undef, <4 x i32> zeroinitializer
// CHECK: %{{.*}} = shl <4 x i32> [[VAL]], [[CNT]]
  ui = ui << cnt;
// CHECK: [[VAL:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui
// CHECK: %{{.*}} = shl <4 x i32> [[VAL]], <i32 5, i32 5, i32 5, i32 5>
  ui = ui << 5;

// CHECK: [[VAL:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl
// CHECK: [[CNT:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl2
// CHECK: %{{.*}} = shl <2 x i64> [[VAL]], [[CNT]]
  sl = sl << sl2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl
// CHECK: [[CNT:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul2
// CHECK: %{{.*}} = shl <2 x i64> [[VAL]], [[CNT]]
  sl = sl << ul2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl
// CHECK: [[T1:%[^ ]+]] = load volatile i32, i32* @cnt
// CHECK: [[T2:%[^ ]+]] = insertelement <2 x i32> undef, i32 [[T1]], i32 0
// CHECK: [[T3:%[^ ]+]] = shufflevector <2 x i32> [[T2]], <2 x i32> undef, <2 x i32> zeroinitializer
// CHECK: [[CNT:%[^ ]+]] = zext <2 x i32> [[T3]] to <2 x i64>
// CHECK: %{{.*}} = shl <2 x i64> [[VAL]], [[CNT]]
  sl = sl << cnt;
// CHECK: [[VAL:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl
// CHECK: %{{.*}} = shl <2 x i64> [[VAL]], <i64 5, i64 5>
  sl = sl << 5;
// CHECK: [[VAL:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul
// CHECK: [[CNT:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl2
// CHECK: %{{.*}} = shl <2 x i64> [[VAL]], [[CNT]]
  ul = ul << sl2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul
// CHECK: [[CNT:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul2
// CHECK: %{{.*}} = shl <2 x i64> [[VAL]], [[CNT]]
  ul = ul << ul2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul
// CHECK: [[T1:%[^ ]+]] = load volatile i32, i32* @cnt
// CHECK: [[T2:%[^ ]+]] = insertelement <2 x i32> undef, i32 [[T1]], i32 0
// CHECK: [[T3:%[^ ]+]] = shufflevector <2 x i32> [[T2]], <2 x i32> undef, <2 x i32> zeroinitializer
// CHECK: [[CNT:%[^ ]+]] = zext <2 x i32> [[T3]] to <2 x i64>
// CHECK: %{{.*}} = shl <2 x i64> [[VAL]], [[CNT]]
  ul = ul << cnt;
// CHECK: [[VAL:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul
// CHECK: %{{.*}} = shl <2 x i64> [[VAL]], <i64 5, i64 5>
  ul = ul << 5;
}

void test_sl_assign (void)
{
// CHECK-LABEL: test_sl_assign

// CHECK: [[CNT:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc2
// CHECK: [[VAL:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc
// CHECK: %{{.*}} = shl <16 x i8> [[VAL]], [[CNT]]
  sc <<= sc2;
// CHECK: [[CNT:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc2
// CHECK: [[VAL:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc
// CHECK: %{{.*}} = shl <16 x i8> [[VAL]], [[CNT]]
  sc <<= uc2;
// CHECK: [[T1:%[^ ]+]] = load volatile i32, i32* @cnt
// CHECK: [[T2:%[^ ]+]] = insertelement <16 x i32> undef, i32 [[T1]], i32 0
// CHECK: [[T3:%[^ ]+]] = shufflevector <16 x i32> [[T2]], <16 x i32> undef, <16 x i32> zeroinitializer
// CHECK: [[VAL:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc
// CHECK: [[CNT:%[^ ]+]] = trunc <16 x i32> [[T3]] to <16 x i8>
// CHECK: %{{.*}} = shl <16 x i8> [[VAL]], [[CNT]]
  sc <<= cnt;
// CHECK: [[VAL:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc
// CHECK: %{{.*}} = shl <16 x i8> [[VAL]], <i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5>
  sc <<= 5;
// CHECK: [[CNT:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc2
// CHECK: [[VAL:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc
// CHECK: %{{.*}} = shl <16 x i8> [[VAL]], [[CNT]]
  uc <<= sc2;
// CHECK: [[CNT:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc2
// CHECK: [[VAL:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc
// CHECK: %{{.*}} = shl <16 x i8> [[VAL]], [[CNT]]
  uc <<= uc2;
// CHECK: [[T1:%[^ ]+]] = load volatile i32, i32* @cnt
// CHECK: [[T2:%[^ ]+]] = insertelement <16 x i32> undef, i32 [[T1]], i32 0
// CHECK: [[T3:%[^ ]+]] = shufflevector <16 x i32> [[T2]], <16 x i32> undef, <16 x i32> zeroinitializer
// CHECK: [[VAL:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc
// CHECK: [[CNT:%[^ ]+]] = trunc <16 x i32> [[T3]] to <16 x i8>
// CHECK: %{{.*}} = shl <16 x i8> [[VAL]], [[CNT]]
  uc <<= cnt;
// CHECK: [[VAL:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc
// CHECK: %{{.*}} = shl <16 x i8> [[VAL]], <i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5>
  uc <<= 5;

// CHECK: [[CNT:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss2
// CHECK: [[VAL:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss
// CHECK: %{{.*}} = shl <8 x i16> [[VAL]], [[CNT]]
  ss <<= ss2;
// CHECK: [[CNT:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us2
// CHECK: [[VAL:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss
// CHECK: %{{.*}} = shl <8 x i16> [[VAL]], [[CNT]]
  ss <<= us2;
// CHECK: [[T1:%[^ ]+]] = load volatile i32, i32* @cnt
// CHECK: [[T2:%[^ ]+]] = insertelement <8 x i32> undef, i32 [[T1]], i32 0
// CHECK: [[T3:%[^ ]+]] = shufflevector <8 x i32> [[T2]], <8 x i32> undef, <8 x i32> zeroinitializer
// CHECK: [[VAL:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss
// CHECK: [[CNT:%[^ ]+]] = trunc <8 x i32> [[T3]] to <8 x i16>
// CHECK: %{{.*}} = shl <8 x i16> [[VAL]], [[CNT]]
  ss <<= cnt;
// CHECK: [[VAL:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss
// CHECK: %{{.*}} = shl <8 x i16> [[VAL]], <i16 5, i16 5, i16 5, i16 5, i16 5, i16 5, i16 5, i16 5>
  ss <<= 5;
// CHECK: [[CNT:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss2
// CHECK: [[VAL:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us
// CHECK: %{{.*}} = shl <8 x i16> [[VAL]], [[CNT]]
  us <<= ss2;
// CHECK: [[CNT:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us2
// CHECK: [[VAL:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us
// CHECK: %{{.*}} = shl <8 x i16> [[VAL]], [[CNT]]
  us <<= us2;
// CHECK: [[T1:%[^ ]+]] = load volatile i32, i32* @cnt
// CHECK: [[T2:%[^ ]+]] = insertelement <8 x i32> undef, i32 [[T1]], i32 0
// CHECK: [[T3:%[^ ]+]] = shufflevector <8 x i32> [[T2]], <8 x i32> undef, <8 x i32> zeroinitializer
// CHECK: [[VAL:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us
// CHECK: [[CNT:%[^ ]+]] = trunc <8 x i32> [[T3]] to <8 x i16>
// CHECK: %{{.*}} = shl <8 x i16> [[VAL]], [[CNT]]
  us <<= cnt;
// CHECK: [[VAL:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us
// CHECK: %{{.*}} = shl <8 x i16> [[VAL]], <i16 5, i16 5, i16 5, i16 5, i16 5, i16 5, i16 5, i16 5>
  us <<= 5;

// CHECK: [[CNT:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si2
// CHECK: [[VAL:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si
// CHECK: %{{.*}} = shl <4 x i32> [[VAL]], [[CNT]]
  si <<= si2;
// CHECK: [[CNT:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui2
// CHECK: [[VAL:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si
// CHECK: %{{.*}} = shl <4 x i32> [[VAL]], [[CNT]]
  si <<= ui2;
// CHECK: [[T1:%[^ ]+]] = load volatile i32, i32* @cnt
// CHECK: [[T3:%[^ ]+]] = insertelement <4 x i32> undef, i32 [[T1]], i32 0
// CHECK: [[CNT:%[^ ]+]] = shufflevector <4 x i32> [[T3]], <4 x i32> undef, <4 x i32> zeroinitializer
// CHECK: [[VAL:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si
// CHECK: %{{.*}} = shl <4 x i32> [[VAL]], [[CNT]]
  si <<= cnt;
// CHECK: [[VAL:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si
// CHECK: %{{.*}} = shl <4 x i32> [[VAL]], <i32 5, i32 5, i32 5, i32 5>
  si <<= 5;
// CHECK: [[CNT:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si2
// CHECK: [[VAL:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui
// CHECK: %{{.*}} = shl <4 x i32> [[VAL]], [[CNT]]
  ui <<= si2;
// CHECK: [[CNT:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui2
// CHECK: [[VAL:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui
// CHECK: %{{.*}} = shl <4 x i32> [[VAL]], [[CNT]]
  ui <<= ui2;
// CHECK: [[T1:%[^ ]+]] = load volatile i32, i32* @cnt
// CHECK: [[T3:%[^ ]+]] = insertelement <4 x i32> undef, i32 [[T1]], i32 0
// CHECK: [[CNT:%[^ ]+]] = shufflevector <4 x i32> [[T3]], <4 x i32> undef, <4 x i32> zeroinitializer
// CHECK: [[VAL:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui
// CHECK: %{{.*}} = shl <4 x i32> [[VAL]], [[CNT]]
  ui <<= cnt;
// CHECK: [[VAL:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui
// CHECK: %{{.*}} = shl <4 x i32> [[VAL]], <i32 5, i32 5, i32 5, i32 5>
  ui <<= 5;

// CHECK: [[CNT:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl2
// CHECK: [[VAL:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl
// CHECK: %{{.*}} = shl <2 x i64> [[VAL]], [[CNT]]
  sl <<= sl2;
// CHECK: [[CNT:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul2
// CHECK: [[VAL:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl
// CHECK: %{{.*}} = shl <2 x i64> [[VAL]], [[CNT]]
  sl <<= ul2;
// CHECK: [[T1:%[^ ]+]] = load volatile i32, i32* @cnt
// CHECK: [[T2:%[^ ]+]] = insertelement <2 x i32> undef, i32 [[T1]], i32 0
// CHECK: [[T3:%[^ ]+]] = shufflevector <2 x i32> [[T2]], <2 x i32> undef, <2 x i32> zeroinitializer
// CHECK: [[VAL:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl
// CHECK: [[CNT:%[^ ]+]] = zext <2 x i32> [[T3]] to <2 x i64>
// CHECK: %{{.*}} = shl <2 x i64> [[VAL]], [[CNT]]
  sl <<= cnt;
// CHECK: [[VAL:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl
// CHECK: %{{.*}} = shl <2 x i64> [[VAL]], <i64 5, i64 5>
  sl <<= 5;
// CHECK: [[CNT:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl2
// CHECK: [[VAL:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul
// CHECK: %{{.*}} = shl <2 x i64> [[VAL]], [[CNT]]
  ul <<= sl2;
// CHECK: [[CNT:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul2
// CHECK: [[VAL:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul
// CHECK: %{{.*}} = shl <2 x i64> [[VAL]], [[CNT]]
  ul <<= ul2;
// CHECK: [[T1:%[^ ]+]] = load volatile i32, i32* @cnt
// CHECK: [[T2:%[^ ]+]] = insertelement <2 x i32> undef, i32 [[T1]], i32 0
// CHECK: [[T3:%[^ ]+]] = shufflevector <2 x i32> [[T2]], <2 x i32> undef, <2 x i32> zeroinitializer
// CHECK: [[VAL:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul
// CHECK: [[CNT:%[^ ]+]] = zext <2 x i32> [[T3]] to <2 x i64>
// CHECK: %{{.*}} = shl <2 x i64> [[VAL]], [[CNT]]
  ul <<= cnt;
// CHECK: [[VAL:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul
// CHECK: %{{.*}} = shl <2 x i64> [[VAL]], <i64 5, i64 5>
  ul <<= 5;
}

void test_sr (void)
{
// CHECK-LABEL: test_sr

// CHECK: [[VAL:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc
// CHECK: [[CNT:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc2
// CHECK: %{{.*}} = ashr <16 x i8> [[VAL]], [[CNT]]
  sc = sc >> sc2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc
// CHECK: [[CNT:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc2
// CHECK: %{{.*}} = ashr <16 x i8> [[VAL]], [[CNT]]
  sc = sc >> uc2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc
// CHECK: [[T1:%[^ ]+]] = load volatile i32, i32* @cnt
// CHECK: [[T2:%[^ ]+]] = insertelement <16 x i32> undef, i32 [[T1]], i32 0
// CHECK: [[T3:%[^ ]+]] = shufflevector <16 x i32> [[T2]], <16 x i32> undef, <16 x i32> zeroinitializer
// CHECK: [[CNT:%[^ ]+]] = trunc <16 x i32> [[T3]] to <16 x i8>
// CHECK: %{{.*}} = ashr <16 x i8> [[VAL]], [[CNT]]
  sc = sc >> cnt;
// CHECK: [[VAL:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc
// CHECK: %{{.*}} = ashr <16 x i8> [[VAL]], <i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5>
  sc = sc >> 5;
// CHECK: [[VAL:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc
// CHECK: [[CNT:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc2
// CHECK: %{{.*}} = lshr <16 x i8> [[VAL]], [[CNT]]
  uc = uc >> sc2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc
// CHECK: [[CNT:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc2
// CHECK: %{{.*}} = lshr <16 x i8> [[VAL]], [[CNT]]
  uc = uc >> uc2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc
// CHECK: [[T1:%[^ ]+]] = load volatile i32, i32* @cnt
// CHECK: [[T2:%[^ ]+]] = insertelement <16 x i32> undef, i32 [[T1]], i32 0
// CHECK: [[T3:%[^ ]+]] = shufflevector <16 x i32> [[T2]], <16 x i32> undef, <16 x i32> zeroinitializer
// CHECK: [[CNT:%[^ ]+]] = trunc <16 x i32> [[T3]] to <16 x i8>
// CHECK: %{{.*}} = lshr <16 x i8> [[VAL]], [[CNT]]
  uc = uc >> cnt;
// CHECK: [[VAL:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc
// CHECK: %{{.*}} = lshr <16 x i8> [[VAL]], <i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5>
  uc = uc >> 5;

// CHECK: [[VAL:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss
// CHECK: [[CNT:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss2
// CHECK: %{{.*}} = ashr <8 x i16> [[VAL]], [[CNT]]
  ss = ss >> ss2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss
// CHECK: [[CNT:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us2
// CHECK: %{{.*}} = ashr <8 x i16> [[VAL]], [[CNT]]
  ss = ss >> us2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss
// CHECK: [[T1:%[^ ]+]] = load volatile i32, i32* @cnt
// CHECK: [[T2:%[^ ]+]] = insertelement <8 x i32> undef, i32 [[T1]], i32 0
// CHECK: [[T3:%[^ ]+]] = shufflevector <8 x i32> [[T2]], <8 x i32> undef, <8 x i32> zeroinitializer
// CHECK: [[CNT:%[^ ]+]] = trunc <8 x i32> [[T3]] to <8 x i16>
// CHECK: %{{.*}} = ashr <8 x i16> [[VAL]], [[CNT]]
  ss = ss >> cnt;
// CHECK: [[VAL:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss
// CHECK: %{{.*}} = ashr <8 x i16> [[VAL]], <i16 5, i16 5, i16 5, i16 5, i16 5, i16 5, i16 5, i16 5>
  ss = ss >> 5;
// CHECK: [[VAL:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us
// CHECK: [[CNT:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss2
// CHECK: %{{.*}} = lshr <8 x i16> [[VAL]], [[CNT]]
  us = us >> ss2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us
// CHECK: [[CNT:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us2
// CHECK: %{{.*}} = lshr <8 x i16> [[VAL]], [[CNT]]
  us = us >> us2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us
// CHECK: [[T1:%[^ ]+]] = load volatile i32, i32* @cnt
// CHECK: [[T2:%[^ ]+]] = insertelement <8 x i32> undef, i32 [[T1]], i32 0
// CHECK: [[T3:%[^ ]+]] = shufflevector <8 x i32> [[T2]], <8 x i32> undef, <8 x i32> zeroinitializer
// CHECK: [[CNT:%[^ ]+]] = trunc <8 x i32> [[T3]] to <8 x i16>
// CHECK: %{{.*}} = lshr <8 x i16> [[VAL]], [[CNT]]
  us = us >> cnt;
// CHECK: [[VAL:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us
// CHECK: %{{.*}} = lshr <8 x i16> [[VAL]], <i16 5, i16 5, i16 5, i16 5, i16 5, i16 5, i16 5, i16 5>
  us = us >> 5;

// CHECK: [[VAL:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si
// CHECK: [[CNT:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si2
// CHECK: %{{.*}} = ashr <4 x i32> [[VAL]], [[CNT]]
  si = si >> si2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si
// CHECK: [[CNT:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui2
// CHECK: %{{.*}} = ashr <4 x i32> [[VAL]], [[CNT]]
  si = si >> ui2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si
// CHECK: [[T1:%[^ ]+]] = load volatile i32, i32* @cnt
// CHECK: [[T3:%[^ ]+]] = insertelement <4 x i32> undef, i32 [[T1]], i32 0
// CHECK: [[CNT:%[^ ]+]] = shufflevector <4 x i32> [[T3]], <4 x i32> undef, <4 x i32> zeroinitializer
// CHECK: %{{.*}} = ashr <4 x i32> [[VAL]], [[CNT]]
  si = si >> cnt;
// CHECK: [[VAL:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si
// CHECK: %{{.*}} = ashr <4 x i32> [[VAL]], <i32 5, i32 5, i32 5, i32 5>
  si = si >> 5;
// CHECK: [[VAL:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui
// CHECK: [[CNT:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si2
// CHECK: %{{.*}} = lshr <4 x i32> [[VAL]], [[CNT]]
  ui = ui >> si2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui
// CHECK: [[CNT:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui2
// CHECK: %{{.*}} = lshr <4 x i32> [[VAL]], [[CNT]]
  ui = ui >> ui2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui
// CHECK: [[T1:%[^ ]+]] = load volatile i32, i32* @cnt
// CHECK: [[T3:%[^ ]+]] = insertelement <4 x i32> undef, i32 [[T1]], i32 0
// CHECK: [[CNT:%[^ ]+]] = shufflevector <4 x i32> [[T3]], <4 x i32> undef, <4 x i32> zeroinitializer
// CHECK: %{{.*}} = lshr <4 x i32> [[VAL]], [[CNT]]
  ui = ui >> cnt;
// CHECK: [[VAL:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui
// CHECK: %{{.*}} = lshr <4 x i32> [[VAL]], <i32 5, i32 5, i32 5, i32 5>
  ui = ui >> 5;

// CHECK: [[VAL:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl
// CHECK: [[CNT:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl2
// CHECK: %{{.*}} = ashr <2 x i64> [[VAL]], [[CNT]]
  sl = sl >> sl2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl
// CHECK: [[CNT:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul2
// CHECK: %{{.*}} = ashr <2 x i64> [[VAL]], [[CNT]]
  sl = sl >> ul2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl
// CHECK: [[T1:%[^ ]+]] = load volatile i32, i32* @cnt
// CHECK: [[T2:%[^ ]+]] = insertelement <2 x i32> undef, i32 [[T1]], i32 0
// CHECK: [[T3:%[^ ]+]] = shufflevector <2 x i32> [[T2]], <2 x i32> undef, <2 x i32> zeroinitializer
// CHECK: [[CNT:%[^ ]+]] = zext <2 x i32> [[T3]] to <2 x i64>
// CHECK: %{{.*}} = ashr <2 x i64> [[VAL]], [[CNT]]
  sl = sl >> cnt;
// CHECK: [[VAL:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl
// CHECK: %{{.*}} = ashr <2 x i64> [[VAL]], <i64 5, i64 5>
  sl = sl >> 5;
// CHECK: [[VAL:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul
// CHECK: [[CNT:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl2
// CHECK: %{{.*}} = lshr <2 x i64> [[VAL]], [[CNT]]
  ul = ul >> sl2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul
// CHECK: [[CNT:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul2
// CHECK: %{{.*}} = lshr <2 x i64> [[VAL]], [[CNT]]
  ul = ul >> ul2;
// CHECK: [[VAL:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul
// CHECK: [[T1:%[^ ]+]] = load volatile i32, i32* @cnt
// CHECK: [[T2:%[^ ]+]] = insertelement <2 x i32> undef, i32 [[T1]], i32 0
// CHECK: [[T3:%[^ ]+]] = shufflevector <2 x i32> [[T2]], <2 x i32> undef, <2 x i32> zeroinitializer
// CHECK: [[CNT:%[^ ]+]] = zext <2 x i32> [[T3]] to <2 x i64>
// CHECK: %{{.*}} = lshr <2 x i64> [[VAL]], [[CNT]]
  ul = ul >> cnt;
// CHECK: [[VAL:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul
// CHECK: %{{.*}} = lshr <2 x i64> [[VAL]], <i64 5, i64 5>
  ul = ul >> 5;
}

void test_sr_assign (void)
{
// CHECK-LABEL: test_sr_assign

// CHECK: [[CNT:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc2
// CHECK: [[VAL:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc
// CHECK: %{{.*}} = ashr <16 x i8> [[VAL]], [[CNT]]
  sc >>= sc2;
// CHECK: [[CNT:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc2
// CHECK: [[VAL:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc
// CHECK: %{{.*}} = ashr <16 x i8> [[VAL]], [[CNT]]
  sc >>= uc2;
// CHECK: [[T1:%[^ ]+]] = load volatile i32, i32* @cnt
// CHECK: [[T2:%[^ ]+]] = insertelement <16 x i32> undef, i32 [[T1]], i32 0
// CHECK: [[T3:%[^ ]+]] = shufflevector <16 x i32> [[T2]], <16 x i32> undef, <16 x i32> zeroinitializer
// CHECK: [[VAL:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc
// CHECK: [[CNT:%[^ ]+]] = trunc <16 x i32> [[T3]] to <16 x i8>
// CHECK: %{{.*}} = ashr <16 x i8> [[VAL]], [[CNT]]
  sc >>= cnt;
// CHECK: [[VAL:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc
// CHECK: %{{.*}} = ashr <16 x i8> [[VAL]], <i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5>
  sc >>= 5;
// CHECK: [[CNT:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc2
// CHECK: [[VAL:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc
// CHECK: %{{.*}} = lshr <16 x i8> [[VAL]], [[CNT]]
  uc >>= sc2;
// CHECK: [[CNT:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc2
// CHECK: [[VAL:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc
// CHECK: %{{.*}} = lshr <16 x i8> [[VAL]], [[CNT]]
  uc >>= uc2;
// CHECK: [[T1:%[^ ]+]] = load volatile i32, i32* @cnt
// CHECK: [[T2:%[^ ]+]] = insertelement <16 x i32> undef, i32 [[T1]], i32 0
// CHECK: [[T3:%[^ ]+]] = shufflevector <16 x i32> [[T2]], <16 x i32> undef, <16 x i32> zeroinitializer
// CHECK: [[VAL:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc
// CHECK: [[CNT:%[^ ]+]] = trunc <16 x i32> [[T3]] to <16 x i8>
// CHECK: %{{.*}} = lshr <16 x i8> [[VAL]], [[CNT]]
  uc >>= cnt;
// CHECK: [[VAL:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc
// CHECK: %{{.*}} = lshr <16 x i8> [[VAL]], <i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5>
  uc >>= 5;

// CHECK: [[CNT:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss2
// CHECK: [[VAL:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss
// CHECK: %{{.*}} = ashr <8 x i16> [[VAL]], [[CNT]]
  ss >>= ss2;
// CHECK: [[CNT:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us2
// CHECK: [[VAL:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss
// CHECK: %{{.*}} = ashr <8 x i16> [[VAL]], [[CNT]]
  ss >>= us2;
// CHECK: [[T1:%[^ ]+]] = load volatile i32, i32* @cnt
// CHECK: [[T2:%[^ ]+]] = insertelement <8 x i32> undef, i32 [[T1]], i32 0
// CHECK: [[T3:%[^ ]+]] = shufflevector <8 x i32> [[T2]], <8 x i32> undef, <8 x i32> zeroinitializer
// CHECK: [[VAL:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss
// CHECK: [[CNT:%[^ ]+]] = trunc <8 x i32> [[T3]] to <8 x i16>
// CHECK: %{{.*}} = ashr <8 x i16> [[VAL]], [[CNT]]
  ss >>= cnt;
// CHECK: [[VAL:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss
// CHECK: %{{.*}} = ashr <8 x i16> [[VAL]], <i16 5, i16 5, i16 5, i16 5, i16 5, i16 5, i16 5, i16 5>
  ss >>= 5;
// CHECK: [[CNT:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss2
// CHECK: [[VAL:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us
// CHECK: %{{.*}} = lshr <8 x i16> [[VAL]], [[CNT]]
  us >>= ss2;
// CHECK: [[CNT:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us2
// CHECK: [[VAL:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us
// CHECK: %{{.*}} = lshr <8 x i16> [[VAL]], [[CNT]]
  us >>= us2;
// CHECK: [[T1:%[^ ]+]] = load volatile i32, i32* @cnt
// CHECK: [[T2:%[^ ]+]] = insertelement <8 x i32> undef, i32 [[T1]], i32 0
// CHECK: [[T3:%[^ ]+]] = shufflevector <8 x i32> [[T2]], <8 x i32> undef, <8 x i32> zeroinitializer
// CHECK: [[VAL:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us
// CHECK: [[CNT:%[^ ]+]] = trunc <8 x i32> [[T3]] to <8 x i16>
// CHECK: %{{.*}} = lshr <8 x i16> [[VAL]], [[CNT]]
  us >>= cnt;
// CHECK: [[VAL:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us
// CHECK: %{{.*}} = lshr <8 x i16> [[VAL]], <i16 5, i16 5, i16 5, i16 5, i16 5, i16 5, i16 5, i16 5>
  us >>= 5;

// CHECK: [[CNT:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si2
// CHECK: [[VAL:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si
// CHECK: %{{.*}} = ashr <4 x i32> [[VAL]], [[CNT]]
  si >>= si2;
// CHECK: [[CNT:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui2
// CHECK: [[VAL:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si
// CHECK: %{{.*}} = ashr <4 x i32> [[VAL]], [[CNT]]
  si >>= ui2;
// CHECK: [[T1:%[^ ]+]] = load volatile i32, i32* @cnt
// CHECK: [[T3:%[^ ]+]] = insertelement <4 x i32> undef, i32 [[T1]], i32 0
// CHECK: [[CNT:%[^ ]+]] = shufflevector <4 x i32> [[T3]], <4 x i32> undef, <4 x i32> zeroinitializer
// CHECK: [[VAL:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si
// CHECK: %{{.*}} = ashr <4 x i32> [[VAL]], [[CNT]]
  si >>= cnt;
// CHECK: [[VAL:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si
// CHECK: %{{.*}} = ashr <4 x i32> [[VAL]], <i32 5, i32 5, i32 5, i32 5>
  si >>= 5;
// CHECK: [[CNT:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si2
// CHECK: [[VAL:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui
// CHECK: %{{.*}} = lshr <4 x i32> [[VAL]], [[CNT]]
  ui >>= si2;
// CHECK: [[CNT:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui2
// CHECK: [[VAL:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui
// CHECK: %{{.*}} = lshr <4 x i32> [[VAL]], [[CNT]]
  ui >>= ui2;
// CHECK: [[T1:%[^ ]+]] = load volatile i32, i32* @cnt
// CHECK: [[T3:%[^ ]+]] = insertelement <4 x i32> undef, i32 [[T1]], i32 0
// CHECK: [[CNT:%[^ ]+]] = shufflevector <4 x i32> [[T3]], <4 x i32> undef, <4 x i32> zeroinitializer
// CHECK: [[VAL:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui
// CHECK: %{{.*}} = lshr <4 x i32> [[VAL]], [[CNT]]
  ui >>= cnt;
// CHECK: [[VAL:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui
// CHECK: %{{.*}} = lshr <4 x i32> [[VAL]], <i32 5, i32 5, i32 5, i32 5>
  ui >>= 5;

// CHECK: [[CNT:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl2
// CHECK: [[VAL:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl
// CHECK: %{{.*}} = ashr <2 x i64> [[VAL]], [[CNT]]
  sl >>= sl2;
// CHECK: [[CNT:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul2
// CHECK: [[VAL:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl
// CHECK: %{{.*}} = ashr <2 x i64> [[VAL]], [[CNT]]
  sl >>= ul2;
// CHECK: [[T1:%[^ ]+]] = load volatile i32, i32* @cnt
// CHECK: [[T2:%[^ ]+]] = insertelement <2 x i32> undef, i32 [[T1]], i32 0
// CHECK: [[T3:%[^ ]+]] = shufflevector <2 x i32> [[T2]], <2 x i32> undef, <2 x i32> zeroinitializer
// CHECK: [[VAL:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl
// CHECK: [[CNT:%[^ ]+]] = zext <2 x i32> [[T3]] to <2 x i64>
// CHECK: %{{.*}} = ashr <2 x i64> [[VAL]], [[CNT]]
  sl >>= cnt;
// CHECK: [[VAL:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl
// CHECK: %{{.*}} = ashr <2 x i64> [[VAL]], <i64 5, i64 5>
  sl >>= 5;
// CHECK: [[CNT:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl2
// CHECK: [[VAL:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul
// CHECK: %{{.*}} = lshr <2 x i64> [[VAL]], [[CNT]]
  ul >>= sl2;
// CHECK: [[CNT:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul2
// CHECK: [[VAL:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul
// CHECK: %{{.*}} = lshr <2 x i64> [[VAL]], [[CNT]]
  ul >>= ul2;
// CHECK: [[T1:%[^ ]+]] = load volatile i32, i32* @cnt
// CHECK: [[T2:%[^ ]+]] = insertelement <2 x i32> undef, i32 [[T1]], i32 0
// CHECK: [[T3:%[^ ]+]] = shufflevector <2 x i32> [[T2]], <2 x i32> undef, <2 x i32> zeroinitializer
// CHECK: [[VAL:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul
// CHECK: [[CNT:%[^ ]+]] = zext <2 x i32> [[T3]] to <2 x i64>
// CHECK: %{{.*}} = lshr <2 x i64> [[VAL]], [[CNT]]
  ul >>= cnt;
// CHECK: [[VAL:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul
// CHECK: %{{.*}} = lshr <2 x i64> [[VAL]], <i64 5, i64 5>
  ul >>= 5;
}


void test_cmpeq (void)
{
// CHECK-LABEL: test_cmpeq

// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc2
// CHECK: [[CMP:%[^ ]+]] = icmp eq <16 x i8> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <16 x i1> [[CMP]] to <16 x i8>
  bc = sc == sc2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc2
// CHECK: [[CMP:%[^ ]+]] = icmp eq <16 x i8> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <16 x i1> [[CMP]] to <16 x i8>
  bc = sc == bc2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc2
// CHECK: [[CMP:%[^ ]+]] = icmp eq <16 x i8> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <16 x i1> [[CMP]] to <16 x i8>
  bc = bc == sc2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc2
// CHECK: [[CMP:%[^ ]+]] = icmp eq <16 x i8> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <16 x i1> [[CMP]] to <16 x i8>
  bc = uc == uc2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc2
// CHECK: [[CMP:%[^ ]+]] = icmp eq <16 x i8> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <16 x i1> [[CMP]] to <16 x i8>
  bc = uc == bc2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc2
// CHECK: [[CMP:%[^ ]+]] = icmp eq <16 x i8> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <16 x i1> [[CMP]] to <16 x i8>
  bc = bc == uc2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc2
// CHECK: [[CMP:%[^ ]+]] = icmp eq <16 x i8> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <16 x i1> [[CMP]] to <16 x i8>
  bc = bc == bc2;

// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss2
// CHECK: [[CMP:%[^ ]+]] = icmp eq <8 x i16> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <8 x i1> [[CMP]] to <8 x i16>
  bs = ss == ss2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs2
// CHECK: [[CMP:%[^ ]+]] = icmp eq <8 x i16> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <8 x i1> [[CMP]] to <8 x i16>
  bs = ss == bs2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss2
// CHECK: [[CMP:%[^ ]+]] = icmp eq <8 x i16> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <8 x i1> [[CMP]] to <8 x i16>
  bs = bs == ss2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us2
// CHECK: [[CMP:%[^ ]+]] = icmp eq <8 x i16> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <8 x i1> [[CMP]] to <8 x i16>
  bs = us == us2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs2
// CHECK: [[CMP:%[^ ]+]] = icmp eq <8 x i16> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <8 x i1> [[CMP]] to <8 x i16>
  bs = us == bs2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us2
// CHECK: [[CMP:%[^ ]+]] = icmp eq <8 x i16> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <8 x i1> [[CMP]] to <8 x i16>
  bs = bs == us2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs2
// CHECK: [[CMP:%[^ ]+]] = icmp eq <8 x i16> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <8 x i1> [[CMP]] to <8 x i16>
  bs = bs == bs2;

// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si2
// CHECK: [[CMP:%[^ ]+]] = icmp eq <4 x i32> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <4 x i1> [[CMP]] to <4 x i32>
  bi = si == si2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi2
// CHECK: [[CMP:%[^ ]+]] = icmp eq <4 x i32> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <4 x i1> [[CMP]] to <4 x i32>
  bi = si == bi2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si2
// CHECK: [[CMP:%[^ ]+]] = icmp eq <4 x i32> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <4 x i1> [[CMP]] to <4 x i32>
  bi = bi == si2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui2
// CHECK: [[CMP:%[^ ]+]] = icmp eq <4 x i32> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <4 x i1> [[CMP]] to <4 x i32>
  bi = ui == ui2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi2
// CHECK: [[CMP:%[^ ]+]] = icmp eq <4 x i32> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <4 x i1> [[CMP]] to <4 x i32>
  bi = ui == bi2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui2
// CHECK: [[CMP:%[^ ]+]] = icmp eq <4 x i32> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <4 x i1> [[CMP]] to <4 x i32>
  bi = bi == ui2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi2
// CHECK: [[CMP:%[^ ]+]] = icmp eq <4 x i32> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <4 x i1> [[CMP]] to <4 x i32>
  bi = bi == bi2;

// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl2
// CHECK: [[CMP:%[^ ]+]] = icmp eq <2 x i64> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <2 x i1> [[CMP]] to <2 x i64>
  bl = sl == sl2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl2
// CHECK: [[CMP:%[^ ]+]] = icmp eq <2 x i64> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <2 x i1> [[CMP]] to <2 x i64>
  bl = sl == bl2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl2
// CHECK: [[CMP:%[^ ]+]] = icmp eq <2 x i64> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <2 x i1> [[CMP]] to <2 x i64>
  bl = bl == sl2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul2
// CHECK: [[CMP:%[^ ]+]] = icmp eq <2 x i64> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <2 x i1> [[CMP]] to <2 x i64>
  bl = ul == ul2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl2
// CHECK: [[CMP:%[^ ]+]] = icmp eq <2 x i64> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <2 x i1> [[CMP]] to <2 x i64>
  bl = ul == bl2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul2
// CHECK: [[CMP:%[^ ]+]] = icmp eq <2 x i64> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <2 x i1> [[CMP]] to <2 x i64>
  bl = bl == ul2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl2
// CHECK: [[CMP:%[^ ]+]] = icmp eq <2 x i64> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <2 x i1> [[CMP]] to <2 x i64>
  bl = bl == bl2;

// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x double>, <2 x double>* @fd
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x double>, <2 x double>* @fd2
// CHECK: [[CMP:%[^ ]+]] = fcmp oeq <2 x double> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <2 x i1> [[CMP]] to <2 x i64>
  bl = fd == fd2;
}

void test_cmpne (void)
{
// CHECK-LABEL: test_cmpne

// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc2
// CHECK: [[CMP:%[^ ]+]] = icmp ne <16 x i8> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <16 x i1> [[CMP]] to <16 x i8>
  bc = sc != sc2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc2
// CHECK: [[CMP:%[^ ]+]] = icmp ne <16 x i8> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <16 x i1> [[CMP]] to <16 x i8>
  bc = sc != bc2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc2
// CHECK: [[CMP:%[^ ]+]] = icmp ne <16 x i8> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <16 x i1> [[CMP]] to <16 x i8>
  bc = bc != sc2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc2
// CHECK: [[CMP:%[^ ]+]] = icmp ne <16 x i8> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <16 x i1> [[CMP]] to <16 x i8>
  bc = uc != uc2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc2
// CHECK: [[CMP:%[^ ]+]] = icmp ne <16 x i8> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <16 x i1> [[CMP]] to <16 x i8>
  bc = uc != bc2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc2
// CHECK: [[CMP:%[^ ]+]] = icmp ne <16 x i8> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <16 x i1> [[CMP]] to <16 x i8>
  bc = bc != uc2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc2
// CHECK: [[CMP:%[^ ]+]] = icmp ne <16 x i8> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <16 x i1> [[CMP]] to <16 x i8>
  bc = bc != bc2;

// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss2
// CHECK: [[CMP:%[^ ]+]] = icmp ne <8 x i16> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <8 x i1> [[CMP]] to <8 x i16>
  bs = ss != ss2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs2
// CHECK: [[CMP:%[^ ]+]] = icmp ne <8 x i16> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <8 x i1> [[CMP]] to <8 x i16>
  bs = ss != bs2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss2
// CHECK: [[CMP:%[^ ]+]] = icmp ne <8 x i16> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <8 x i1> [[CMP]] to <8 x i16>
  bs = bs != ss2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us2
// CHECK: [[CMP:%[^ ]+]] = icmp ne <8 x i16> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <8 x i1> [[CMP]] to <8 x i16>
  bs = us != us2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs2
// CHECK: [[CMP:%[^ ]+]] = icmp ne <8 x i16> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <8 x i1> [[CMP]] to <8 x i16>
  bs = us != bs2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us2
// CHECK: [[CMP:%[^ ]+]] = icmp ne <8 x i16> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <8 x i1> [[CMP]] to <8 x i16>
  bs = bs != us2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs2
// CHECK: [[CMP:%[^ ]+]] = icmp ne <8 x i16> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <8 x i1> [[CMP]] to <8 x i16>
  bs = bs != bs2;

// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si2
// CHECK: [[CMP:%[^ ]+]] = icmp ne <4 x i32> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <4 x i1> [[CMP]] to <4 x i32>
  bi = si != si2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi2
// CHECK: [[CMP:%[^ ]+]] = icmp ne <4 x i32> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <4 x i1> [[CMP]] to <4 x i32>
  bi = si != bi2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si2
// CHECK: [[CMP:%[^ ]+]] = icmp ne <4 x i32> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <4 x i1> [[CMP]] to <4 x i32>
  bi = bi != si2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui2
// CHECK: [[CMP:%[^ ]+]] = icmp ne <4 x i32> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <4 x i1> [[CMP]] to <4 x i32>
  bi = ui != ui2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi2
// CHECK: [[CMP:%[^ ]+]] = icmp ne <4 x i32> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <4 x i1> [[CMP]] to <4 x i32>
  bi = ui != bi2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui2
// CHECK: [[CMP:%[^ ]+]] = icmp ne <4 x i32> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <4 x i1> [[CMP]] to <4 x i32>
  bi = bi != ui2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi2
// CHECK: [[CMP:%[^ ]+]] = icmp ne <4 x i32> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <4 x i1> [[CMP]] to <4 x i32>
  bi = bi != bi2;

// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl2
// CHECK: [[CMP:%[^ ]+]] = icmp ne <2 x i64> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <2 x i1> [[CMP]] to <2 x i64>
  bl = sl != sl2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl2
// CHECK: [[CMP:%[^ ]+]] = icmp ne <2 x i64> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <2 x i1> [[CMP]] to <2 x i64>
  bl = sl != bl2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl2
// CHECK: [[CMP:%[^ ]+]] = icmp ne <2 x i64> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <2 x i1> [[CMP]] to <2 x i64>
  bl = bl != sl2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul2
// CHECK: [[CMP:%[^ ]+]] = icmp ne <2 x i64> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <2 x i1> [[CMP]] to <2 x i64>
  bl = ul != ul2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl2
// CHECK: [[CMP:%[^ ]+]] = icmp ne <2 x i64> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <2 x i1> [[CMP]] to <2 x i64>
  bl = ul != bl2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul2
// CHECK: [[CMP:%[^ ]+]] = icmp ne <2 x i64> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <2 x i1> [[CMP]] to <2 x i64>
  bl = bl != ul2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl2
// CHECK: [[CMP:%[^ ]+]] = icmp ne <2 x i64> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <2 x i1> [[CMP]] to <2 x i64>
  bl = bl != bl2;

// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x double>, <2 x double>* @fd
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x double>, <2 x double>* @fd2
// CHECK: [[CMP:%[^ ]+]] = fcmp une <2 x double> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <2 x i1> [[CMP]] to <2 x i64>
  bl = fd != fd2;
}

void test_cmpge (void)
{
// CHECK-LABEL: test_cmpge

// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc2
// CHECK: [[CMP:%[^ ]+]] = icmp sge <16 x i8> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <16 x i1> [[CMP]] to <16 x i8>
  bc = sc >= sc2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc2
// CHECK: [[CMP:%[^ ]+]] = icmp uge <16 x i8> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <16 x i1> [[CMP]] to <16 x i8>
  bc = uc >= uc2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc2
// CHECK: [[CMP:%[^ ]+]] = icmp uge <16 x i8> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <16 x i1> [[CMP]] to <16 x i8>
  bc = bc >= bc2;

// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss2
// CHECK: [[CMP:%[^ ]+]] = icmp sge <8 x i16> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <8 x i1> [[CMP]] to <8 x i16>
  bs = ss >= ss2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us2
// CHECK: [[CMP:%[^ ]+]] = icmp uge <8 x i16> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <8 x i1> [[CMP]] to <8 x i16>
  bs = us >= us2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs2
// CHECK: [[CMP:%[^ ]+]] = icmp uge <8 x i16> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <8 x i1> [[CMP]] to <8 x i16>
  bs = bs >= bs2;

// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si2
// CHECK: [[CMP:%[^ ]+]] = icmp sge <4 x i32> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <4 x i1> [[CMP]] to <4 x i32>
  bi = si >= si2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui2
// CHECK: [[CMP:%[^ ]+]] = icmp uge <4 x i32> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <4 x i1> [[CMP]] to <4 x i32>
  bi = ui >= ui2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi2
// CHECK: [[CMP:%[^ ]+]] = icmp uge <4 x i32> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <4 x i1> [[CMP]] to <4 x i32>
  bi = bi >= bi2;

// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl2
// CHECK: [[CMP:%[^ ]+]] = icmp sge <2 x i64> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <2 x i1> [[CMP]] to <2 x i64>
  bl = sl >= sl2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul2
// CHECK: [[CMP:%[^ ]+]] = icmp uge <2 x i64> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <2 x i1> [[CMP]] to <2 x i64>
  bl = ul >= ul2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl2
// CHECK: [[CMP:%[^ ]+]] = icmp uge <2 x i64> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <2 x i1> [[CMP]] to <2 x i64>
  bl = bl >= bl2;

// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x double>, <2 x double>* @fd
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x double>, <2 x double>* @fd2
// CHECK: [[CMP:%[^ ]+]] = fcmp oge <2 x double> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <2 x i1> [[CMP]] to <2 x i64>
  bl = fd >= fd2;
}

void test_cmpgt (void)
{
// CHECK-LABEL: test_cmpgt

// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc2
// CHECK: [[CMP:%[^ ]+]] = icmp sgt <16 x i8> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <16 x i1> [[CMP]] to <16 x i8>
  bc = sc > sc2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc2
// CHECK: [[CMP:%[^ ]+]] = icmp ugt <16 x i8> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <16 x i1> [[CMP]] to <16 x i8>
  bc = uc > uc2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc2
// CHECK: [[CMP:%[^ ]+]] = icmp ugt <16 x i8> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <16 x i1> [[CMP]] to <16 x i8>
  bc = bc > bc2;

// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss2
// CHECK: [[CMP:%[^ ]+]] = icmp sgt <8 x i16> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <8 x i1> [[CMP]] to <8 x i16>
  bs = ss > ss2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us2
// CHECK: [[CMP:%[^ ]+]] = icmp ugt <8 x i16> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <8 x i1> [[CMP]] to <8 x i16>
  bs = us > us2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs2
// CHECK: [[CMP:%[^ ]+]] = icmp ugt <8 x i16> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <8 x i1> [[CMP]] to <8 x i16>
  bs = bs > bs2;

// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si2
// CHECK: [[CMP:%[^ ]+]] = icmp sgt <4 x i32> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <4 x i1> [[CMP]] to <4 x i32>
  bi = si > si2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui2
// CHECK: [[CMP:%[^ ]+]] = icmp ugt <4 x i32> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <4 x i1> [[CMP]] to <4 x i32>
  bi = ui > ui2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi2
// CHECK: [[CMP:%[^ ]+]] = icmp ugt <4 x i32> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <4 x i1> [[CMP]] to <4 x i32>
  bi = bi > bi2;

// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl2
// CHECK: [[CMP:%[^ ]+]] = icmp sgt <2 x i64> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <2 x i1> [[CMP]] to <2 x i64>
  bl = sl > sl2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul2
// CHECK: [[CMP:%[^ ]+]] = icmp ugt <2 x i64> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <2 x i1> [[CMP]] to <2 x i64>
  bl = ul > ul2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl2
// CHECK: [[CMP:%[^ ]+]] = icmp ugt <2 x i64> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <2 x i1> [[CMP]] to <2 x i64>
  bl = bl > bl2;

// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x double>, <2 x double>* @fd
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x double>, <2 x double>* @fd2
// CHECK: [[CMP:%[^ ]+]] = fcmp ogt <2 x double> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <2 x i1> [[CMP]] to <2 x i64>
  bl = fd > fd2;
}

void test_cmple (void)
{
// CHECK-LABEL: test_cmple

// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc2
// CHECK: [[CMP:%[^ ]+]] = icmp sle <16 x i8> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <16 x i1> [[CMP]] to <16 x i8>
  bc = sc <= sc2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc2
// CHECK: [[CMP:%[^ ]+]] = icmp ule <16 x i8> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <16 x i1> [[CMP]] to <16 x i8>
  bc = uc <= uc2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc2
// CHECK: [[CMP:%[^ ]+]] = icmp ule <16 x i8> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <16 x i1> [[CMP]] to <16 x i8>
  bc = bc <= bc2;

// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss2
// CHECK: [[CMP:%[^ ]+]] = icmp sle <8 x i16> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <8 x i1> [[CMP]] to <8 x i16>
  bs = ss <= ss2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us2
// CHECK: [[CMP:%[^ ]+]] = icmp ule <8 x i16> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <8 x i1> [[CMP]] to <8 x i16>
  bs = us <= us2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs2
// CHECK: [[CMP:%[^ ]+]] = icmp ule <8 x i16> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <8 x i1> [[CMP]] to <8 x i16>
  bs = bs <= bs2;

// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si2
// CHECK: [[CMP:%[^ ]+]] = icmp sle <4 x i32> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <4 x i1> [[CMP]] to <4 x i32>
  bi = si <= si2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui2
// CHECK: [[CMP:%[^ ]+]] = icmp ule <4 x i32> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <4 x i1> [[CMP]] to <4 x i32>
  bi = ui <= ui2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi2
// CHECK: [[CMP:%[^ ]+]] = icmp ule <4 x i32> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <4 x i1> [[CMP]] to <4 x i32>
  bi = bi <= bi2;

// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl2
// CHECK: [[CMP:%[^ ]+]] = icmp sle <2 x i64> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <2 x i1> [[CMP]] to <2 x i64>
  bl = sl <= sl2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul2
// CHECK: [[CMP:%[^ ]+]] = icmp ule <2 x i64> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <2 x i1> [[CMP]] to <2 x i64>
  bl = ul <= ul2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl2
// CHECK: [[CMP:%[^ ]+]] = icmp ule <2 x i64> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <2 x i1> [[CMP]] to <2 x i64>
  bl = bl <= bl2;

// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x double>, <2 x double>* @fd
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x double>, <2 x double>* @fd2
// CHECK: [[CMP:%[^ ]+]] = fcmp ole <2 x double> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <2 x i1> [[CMP]] to <2 x i64>
  bl = fd <= fd2;
}

void test_cmplt (void)
{
// CHECK-LABEL: test_cmplt

// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @sc2
// CHECK: [[CMP:%[^ ]+]] = icmp slt <16 x i8> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <16 x i1> [[CMP]] to <16 x i8>
  bc = sc < sc2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @uc2
// CHECK: [[CMP:%[^ ]+]] = icmp ult <16 x i8> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <16 x i1> [[CMP]] to <16 x i8>
  bc = uc < uc2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc
// CHECK: [[VAL2:%[^ ]+]] = load volatile <16 x i8>, <16 x i8>* @bc2
// CHECK: [[CMP:%[^ ]+]] = icmp ult <16 x i8> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <16 x i1> [[CMP]] to <16 x i8>
  bc = bc < bc2;

// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @ss2
// CHECK: [[CMP:%[^ ]+]] = icmp slt <8 x i16> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <8 x i1> [[CMP]] to <8 x i16>
  bs = ss < ss2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @us2
// CHECK: [[CMP:%[^ ]+]] = icmp ult <8 x i16> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <8 x i1> [[CMP]] to <8 x i16>
  bs = us < us2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs
// CHECK: [[VAL2:%[^ ]+]] = load volatile <8 x i16>, <8 x i16>* @bs2
// CHECK: [[CMP:%[^ ]+]] = icmp ult <8 x i16> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <8 x i1> [[CMP]] to <8 x i16>
  bs = bs < bs2;

// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @si2
// CHECK: [[CMP:%[^ ]+]] = icmp slt <4 x i32> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <4 x i1> [[CMP]] to <4 x i32>
  bi = si < si2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @ui2
// CHECK: [[CMP:%[^ ]+]] = icmp ult <4 x i32> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <4 x i1> [[CMP]] to <4 x i32>
  bi = ui < ui2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x i32>, <4 x i32>* @bi2
// CHECK: [[CMP:%[^ ]+]] = icmp ult <4 x i32> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <4 x i1> [[CMP]] to <4 x i32>
  bi = bi < bi2;

// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @sl2
// CHECK: [[CMP:%[^ ]+]] = icmp slt <2 x i64> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <2 x i1> [[CMP]] to <2 x i64>
  bl = sl < sl2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @ul2
// CHECK: [[CMP:%[^ ]+]] = icmp ult <2 x i64> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <2 x i1> [[CMP]] to <2 x i64>
  bl = ul < ul2;
// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x i64>, <2 x i64>* @bl2
// CHECK: [[CMP:%[^ ]+]] = icmp ult <2 x i64> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <2 x i1> [[CMP]] to <2 x i64>
  bl = bl < bl2;

// CHECK: [[VAL1:%[^ ]+]] = load volatile <2 x double>, <2 x double>* @fd
// CHECK: [[VAL2:%[^ ]+]] = load volatile <2 x double>, <2 x double>* @fd2
// CHECK: [[CMP:%[^ ]+]] = fcmp olt <2 x double> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <2 x i1> [[CMP]] to <2 x i64>
  bl = fd < fd2;
}

