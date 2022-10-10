// RUN: %clang_cc1 -triple s390x-linux-gnu -target-cpu z13 -fzvector -emit-llvm -o - -W -Wall -Werror %s | opt -S -mem2reg | FileCheck %s

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

// CHECK-LABEL: define{{.*}} void @test_assign() #0 {
// CHECK:   [[TMP0:%.*]] = load volatile <16 x i8>, ptr @sc2, align 8
// CHECK:   store volatile <16 x i8> [[TMP0]], ptr @sc, align 8
// CHECK:   [[TMP1:%.*]] = load volatile <16 x i8>, ptr @uc2, align 8
// CHECK:   store volatile <16 x i8> [[TMP1]], ptr @uc, align 8
// CHECK:   [[TMP2:%.*]] = load volatile <8 x i16>, ptr @ss2, align 8
// CHECK:   store volatile <8 x i16> [[TMP2]], ptr @ss, align 8
// CHECK:   [[TMP3:%.*]] = load volatile <8 x i16>, ptr @us2, align 8
// CHECK:   store volatile <8 x i16> [[TMP3]], ptr @us, align 8
// CHECK:   [[TMP4:%.*]] = load volatile <4 x i32>, ptr @si2, align 8
// CHECK:   store volatile <4 x i32> [[TMP4]], ptr @si, align 8
// CHECK:   [[TMP5:%.*]] = load volatile <4 x i32>, ptr @ui2, align 8
// CHECK:   store volatile <4 x i32> [[TMP5]], ptr @ui, align 8
// CHECK:   [[TMP6:%.*]] = load volatile <2 x i64>, ptr @sl2, align 8
// CHECK:   store volatile <2 x i64> [[TMP6]], ptr @sl, align 8
// CHECK:   [[TMP7:%.*]] = load volatile <2 x i64>, ptr @ul2, align 8
// CHECK:   store volatile <2 x i64> [[TMP7]], ptr @ul, align 8
// CHECK:   [[TMP8:%.*]] = load volatile <2 x double>, ptr @fd2, align 8
// CHECK:   store volatile <2 x double> [[TMP8]], ptr @fd, align 8
// CHECK:   ret void
void test_assign(void) {

  sc = sc2;
  uc = uc2;

  ss = ss2;
  us = us2;

  si = si2;
  ui = ui2;

  sl = sl2;
  ul = ul2;

  fd = fd2;
}

// CHECK-LABEL: define{{.*}} void @test_pos() #0 {
// CHECK:   [[TMP0:%.*]] = load volatile <16 x i8>, ptr @sc2, align 8
// CHECK:   store volatile <16 x i8> [[TMP0]], ptr @sc, align 8
// CHECK:   [[TMP1:%.*]] = load volatile <16 x i8>, ptr @uc2, align 8
// CHECK:   store volatile <16 x i8> [[TMP1]], ptr @uc, align 8
// CHECK:   [[TMP2:%.*]] = load volatile <8 x i16>, ptr @ss2, align 8
// CHECK:   store volatile <8 x i16> [[TMP2]], ptr @ss, align 8
// CHECK:   [[TMP3:%.*]] = load volatile <8 x i16>, ptr @us2, align 8
// CHECK:   store volatile <8 x i16> [[TMP3]], ptr @us, align 8
// CHECK:   [[TMP4:%.*]] = load volatile <4 x i32>, ptr @si2, align 8
// CHECK:   store volatile <4 x i32> [[TMP4]], ptr @si, align 8
// CHECK:   [[TMP5:%.*]] = load volatile <4 x i32>, ptr @ui2, align 8
// CHECK:   store volatile <4 x i32> [[TMP5]], ptr @ui, align 8
// CHECK:   [[TMP6:%.*]] = load volatile <2 x i64>, ptr @sl2, align 8
// CHECK:   store volatile <2 x i64> [[TMP6]], ptr @sl, align 8
// CHECK:   [[TMP7:%.*]] = load volatile <2 x i64>, ptr @ul2, align 8
// CHECK:   store volatile <2 x i64> [[TMP7]], ptr @ul, align 8
// CHECK:   [[TMP8:%.*]] = load volatile <2 x double>, ptr @fd2, align 8
// CHECK:   store volatile <2 x double> [[TMP8]], ptr @fd, align 8
// CHECK:   ret void
void test_pos(void) {

  sc = +sc2;
  uc = +uc2;

  ss = +ss2;
  us = +us2;

  si = +si2;
  ui = +ui2;

  sl = +sl2;
  ul = +ul2;

  fd = +fd2;
}

// CHECK-LABEL: define{{.*}} void @test_neg() #0 {
// CHECK:   [[TMP0:%.*]] = load volatile <16 x i8>, ptr @sc2, align 8
// CHECK:   [[SUB:%.*]] = sub <16 x i8> zeroinitializer, [[TMP0]]
// CHECK:   store volatile <16 x i8> [[SUB]], ptr @sc, align 8
// CHECK:   [[TMP1:%.*]] = load volatile <8 x i16>, ptr @ss2, align 8
// CHECK:   [[SUB1:%.*]] = sub <8 x i16> zeroinitializer, [[TMP1]]
// CHECK:   store volatile <8 x i16> [[SUB1]], ptr @ss, align 8
// CHECK:   [[TMP2:%.*]] = load volatile <4 x i32>, ptr @si2, align 8
// CHECK:   [[SUB2:%.*]] = sub <4 x i32> zeroinitializer, [[TMP2]]
// CHECK:   store volatile <4 x i32> [[SUB2]], ptr @si, align 8
// CHECK:   [[TMP3:%.*]] = load volatile <2 x i64>, ptr @sl2, align 8
// CHECK:   [[SUB3:%.*]] = sub <2 x i64> zeroinitializer, [[TMP3]]
// CHECK:   store volatile <2 x i64> [[SUB3]], ptr @sl, align 8
// CHECK:   [[TMP4:%.*]] = load volatile <2 x double>, ptr @fd2, align 8
// CHECK:   [[SUB4:%.*]] = fneg <2 x double> [[TMP4]]
// CHECK:   store volatile <2 x double> [[SUB4]], ptr @fd, align 8
// CHECK:   ret void
void test_neg(void) {

  sc = -sc2;
  ss = -ss2;
  si = -si2;
  sl = -sl2;
  fd = -fd2;
}

// CHECK-LABEL: define{{.*}} void @test_preinc() #0 {
// CHECK:   [[TMP0:%.*]] = load volatile <16 x i8>, ptr @sc2, align 8
// CHECK:   [[INC:%.*]] = add <16 x i8> [[TMP0]], <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
// CHECK:   store volatile <16 x i8> [[INC]], ptr @sc2, align 8
// CHECK:   [[TMP1:%.*]] = load volatile <16 x i8>, ptr @uc2, align 8
// CHECK:   [[INC1:%.*]] = add <16 x i8> [[TMP1]], <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
// CHECK:   store volatile <16 x i8> [[INC1]], ptr @uc2, align 8
// CHECK:   [[TMP2:%.*]] = load volatile <8 x i16>, ptr @ss2, align 8
// CHECK:   [[INC2:%.*]] = add <8 x i16> [[TMP2]], <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
// CHECK:   store volatile <8 x i16> [[INC2]], ptr @ss2, align 8
// CHECK:   [[TMP3:%.*]] = load volatile <8 x i16>, ptr @us2, align 8
// CHECK:   [[INC3:%.*]] = add <8 x i16> [[TMP3]], <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
// CHECK:   store volatile <8 x i16> [[INC3]], ptr @us2, align 8
// CHECK:   [[TMP4:%.*]] = load volatile <4 x i32>, ptr @si2, align 8
// CHECK:   [[INC4:%.*]] = add <4 x i32> [[TMP4]], <i32 1, i32 1, i32 1, i32 1>
// CHECK:   store volatile <4 x i32> [[INC4]], ptr @si2, align 8
// CHECK:   [[TMP5:%.*]] = load volatile <4 x i32>, ptr @ui2, align 8
// CHECK:   [[INC5:%.*]] = add <4 x i32> [[TMP5]], <i32 1, i32 1, i32 1, i32 1>
// CHECK:   store volatile <4 x i32> [[INC5]], ptr @ui2, align 8
// CHECK:   [[TMP6:%.*]] = load volatile <2 x i64>, ptr @sl2, align 8
// CHECK:   [[INC6:%.*]] = add <2 x i64> [[TMP6]], <i64 1, i64 1>
// CHECK:   store volatile <2 x i64> [[INC6]], ptr @sl2, align 8
// CHECK:   [[TMP7:%.*]] = load volatile <2 x i64>, ptr @ul2, align 8
// CHECK:   [[INC7:%.*]] = add <2 x i64> [[TMP7]], <i64 1, i64 1>
// CHECK:   store volatile <2 x i64> [[INC7]], ptr @ul2, align 8
// CHECK:   [[TMP8:%.*]] = load volatile <2 x double>, ptr @fd2, align 8
// CHECK:   [[INC8:%.*]] = fadd <2 x double> [[TMP8]], <double 1.000000e+00, double 1.000000e+00>
// CHECK:   store volatile <2 x double> [[INC8]], ptr @fd2, align 8
// CHECK:   ret void
void test_preinc(void) {

  ++sc2;
  ++uc2;

  ++ss2;
  ++us2;

  ++si2;
  ++ui2;

  ++sl2;
  ++ul2;

  ++fd2;
}

// CHECK-LABEL: define{{.*}} void @test_postinc() #0 {
// CHECK:   [[TMP0:%.*]] = load volatile <16 x i8>, ptr @sc2, align 8
// CHECK:   [[INC:%.*]] = add <16 x i8> [[TMP0]], <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
// CHECK:   store volatile <16 x i8> [[INC]], ptr @sc2, align 8
// CHECK:   [[TMP1:%.*]] = load volatile <16 x i8>, ptr @uc2, align 8
// CHECK:   [[INC1:%.*]] = add <16 x i8> [[TMP1]], <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
// CHECK:   store volatile <16 x i8> [[INC1]], ptr @uc2, align 8
// CHECK:   [[TMP2:%.*]] = load volatile <8 x i16>, ptr @ss2, align 8
// CHECK:   [[INC2:%.*]] = add <8 x i16> [[TMP2]], <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
// CHECK:   store volatile <8 x i16> [[INC2]], ptr @ss2, align 8
// CHECK:   [[TMP3:%.*]] = load volatile <8 x i16>, ptr @us2, align 8
// CHECK:   [[INC3:%.*]] = add <8 x i16> [[TMP3]], <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
// CHECK:   store volatile <8 x i16> [[INC3]], ptr @us2, align 8
// CHECK:   [[TMP4:%.*]] = load volatile <4 x i32>, ptr @si2, align 8
// CHECK:   [[INC4:%.*]] = add <4 x i32> [[TMP4]], <i32 1, i32 1, i32 1, i32 1>
// CHECK:   store volatile <4 x i32> [[INC4]], ptr @si2, align 8
// CHECK:   [[TMP5:%.*]] = load volatile <4 x i32>, ptr @ui2, align 8
// CHECK:   [[INC5:%.*]] = add <4 x i32> [[TMP5]], <i32 1, i32 1, i32 1, i32 1>
// CHECK:   store volatile <4 x i32> [[INC5]], ptr @ui2, align 8
// CHECK:   [[TMP6:%.*]] = load volatile <2 x i64>, ptr @sl2, align 8
// CHECK:   [[INC6:%.*]] = add <2 x i64> [[TMP6]], <i64 1, i64 1>
// CHECK:   store volatile <2 x i64> [[INC6]], ptr @sl2, align 8
// CHECK:   [[TMP7:%.*]] = load volatile <2 x i64>, ptr @ul2, align 8
// CHECK:   [[INC7:%.*]] = add <2 x i64> [[TMP7]], <i64 1, i64 1>
// CHECK:   store volatile <2 x i64> [[INC7]], ptr @ul2, align 8
// CHECK:   [[TMP8:%.*]] = load volatile <2 x double>, ptr @fd2, align 8
// CHECK:   [[INC8:%.*]] = fadd <2 x double> [[TMP8]], <double 1.000000e+00, double 1.000000e+00>
// CHECK:   store volatile <2 x double> [[INC8]], ptr @fd2, align 8
// CHECK:   ret void
void test_postinc(void) {

  sc2++;
  uc2++;

  ss2++;
  us2++;

  si2++;
  ui2++;

  sl2++;
  ul2++;

  fd2++;
}

// CHECK-LABEL: define{{.*}} void @test_predec() #0 {
// CHECK:   [[TMP0:%.*]] = load volatile <16 x i8>, ptr @sc2, align 8
// CHECK:   [[DEC:%.*]] = add <16 x i8> [[TMP0]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
// CHECK:   store volatile <16 x i8> [[DEC]], ptr @sc2, align 8
// CHECK:   [[TMP1:%.*]] = load volatile <16 x i8>, ptr @uc2, align 8
// CHECK:   [[DEC1:%.*]] = add <16 x i8> [[TMP1]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
// CHECK:   store volatile <16 x i8> [[DEC1]], ptr @uc2, align 8
// CHECK:   [[TMP2:%.*]] = load volatile <8 x i16>, ptr @ss2, align 8
// CHECK:   [[DEC2:%.*]] = add <8 x i16> [[TMP2]], <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
// CHECK:   store volatile <8 x i16> [[DEC2]], ptr @ss2, align 8
// CHECK:   [[TMP3:%.*]] = load volatile <8 x i16>, ptr @us2, align 8
// CHECK:   [[DEC3:%.*]] = add <8 x i16> [[TMP3]], <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
// CHECK:   store volatile <8 x i16> [[DEC3]], ptr @us2, align 8
// CHECK:   [[TMP4:%.*]] = load volatile <4 x i32>, ptr @si2, align 8
// CHECK:   [[DEC4:%.*]] = add <4 x i32> [[TMP4]], <i32 -1, i32 -1, i32 -1, i32 -1>
// CHECK:   store volatile <4 x i32> [[DEC4]], ptr @si2, align 8
// CHECK:   [[TMP5:%.*]] = load volatile <4 x i32>, ptr @ui2, align 8
// CHECK:   [[DEC5:%.*]] = add <4 x i32> [[TMP5]], <i32 -1, i32 -1, i32 -1, i32 -1>
// CHECK:   store volatile <4 x i32> [[DEC5]], ptr @ui2, align 8
// CHECK:   [[TMP6:%.*]] = load volatile <2 x i64>, ptr @sl2, align 8
// CHECK:   [[DEC6:%.*]] = add <2 x i64> [[TMP6]], <i64 -1, i64 -1>
// CHECK:   store volatile <2 x i64> [[DEC6]], ptr @sl2, align 8
// CHECK:   [[TMP7:%.*]] = load volatile <2 x i64>, ptr @ul2, align 8
// CHECK:   [[DEC7:%.*]] = add <2 x i64> [[TMP7]], <i64 -1, i64 -1>
// CHECK:   store volatile <2 x i64> [[DEC7]], ptr @ul2, align 8
// CHECK:   [[TMP8:%.*]] = load volatile <2 x double>, ptr @fd2, align 8
// CHECK:   [[DEC8:%.*]] = fadd <2 x double> [[TMP8]], <double -1.000000e+00, double -1.000000e+00>
// CHECK:   store volatile <2 x double> [[DEC8]], ptr @fd2, align 8
// CHECK:   ret void
void test_predec(void) {

  --sc2;
  --uc2;

  --ss2;
  --us2;

  --si2;
  --ui2;

  --sl2;
  --ul2;

  --fd2;
}

// CHECK-LABEL: define{{.*}} void @test_postdec() #0 {
// CHECK:   [[TMP0:%.*]] = load volatile <16 x i8>, ptr @sc2, align 8
// CHECK:   [[DEC:%.*]] = add <16 x i8> [[TMP0]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
// CHECK:   store volatile <16 x i8> [[DEC]], ptr @sc2, align 8
// CHECK:   [[TMP1:%.*]] = load volatile <16 x i8>, ptr @uc2, align 8
// CHECK:   [[DEC1:%.*]] = add <16 x i8> [[TMP1]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
// CHECK:   store volatile <16 x i8> [[DEC1]], ptr @uc2, align 8
// CHECK:   [[TMP2:%.*]] = load volatile <8 x i16>, ptr @ss2, align 8
// CHECK:   [[DEC2:%.*]] = add <8 x i16> [[TMP2]], <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
// CHECK:   store volatile <8 x i16> [[DEC2]], ptr @ss2, align 8
// CHECK:   [[TMP3:%.*]] = load volatile <8 x i16>, ptr @us2, align 8
// CHECK:   [[DEC3:%.*]] = add <8 x i16> [[TMP3]], <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
// CHECK:   store volatile <8 x i16> [[DEC3]], ptr @us2, align 8
// CHECK:   [[TMP4:%.*]] = load volatile <4 x i32>, ptr @si2, align 8
// CHECK:   [[DEC4:%.*]] = add <4 x i32> [[TMP4]], <i32 -1, i32 -1, i32 -1, i32 -1>
// CHECK:   store volatile <4 x i32> [[DEC4]], ptr @si2, align 8
// CHECK:   [[TMP5:%.*]] = load volatile <4 x i32>, ptr @ui2, align 8
// CHECK:   [[DEC5:%.*]] = add <4 x i32> [[TMP5]], <i32 -1, i32 -1, i32 -1, i32 -1>
// CHECK:   store volatile <4 x i32> [[DEC5]], ptr @ui2, align 8
// CHECK:   [[TMP6:%.*]] = load volatile <2 x i64>, ptr @sl2, align 8
// CHECK:   [[DEC6:%.*]] = add <2 x i64> [[TMP6]], <i64 -1, i64 -1>
// CHECK:   store volatile <2 x i64> [[DEC6]], ptr @sl2, align 8
// CHECK:   [[TMP7:%.*]] = load volatile <2 x i64>, ptr @ul2, align 8
// CHECK:   [[DEC7:%.*]] = add <2 x i64> [[TMP7]], <i64 -1, i64 -1>
// CHECK:   store volatile <2 x i64> [[DEC7]], ptr @ul2, align 8
// CHECK:   [[TMP8:%.*]] = load volatile <2 x double>, ptr @fd2, align 8
// CHECK:   [[DEC8:%.*]] = fadd <2 x double> [[TMP8]], <double -1.000000e+00, double -1.000000e+00>
// CHECK:   store volatile <2 x double> [[DEC8]], ptr @fd2, align 8
// CHECK:   ret void
void test_postdec(void) {

  sc2--;
  uc2--;

  ss2--;
  us2--;

  si2--;
  ui2--;

  sl2--;
  ul2--;

  fd2--;
}

// CHECK-LABEL: define{{.*}} void @test_add() #0 {
// CHECK:   [[TMP0:%.*]] = load volatile <16 x i8>, ptr @sc, align 8
// CHECK:   [[TMP1:%.*]] = load volatile <16 x i8>, ptr @sc2, align 8
// CHECK:   [[ADD:%.*]] = add <16 x i8> [[TMP0]], [[TMP1]]
// CHECK:   store volatile <16 x i8> [[ADD]], ptr @sc, align 8
// CHECK:   [[TMP2:%.*]] = load volatile <16 x i8>, ptr @sc, align 8
// CHECK:   [[TMP3:%.*]] = load volatile <16 x i8>, ptr @bc2, align 8
// CHECK:   [[ADD1:%.*]] = add <16 x i8> [[TMP2]], [[TMP3]]
// CHECK:   store volatile <16 x i8> [[ADD1]], ptr @sc, align 8
// CHECK:   [[TMP4:%.*]] = load volatile <16 x i8>, ptr @bc, align 8
// CHECK:   [[TMP5:%.*]] = load volatile <16 x i8>, ptr @sc2, align 8
// CHECK:   [[ADD2:%.*]] = add <16 x i8> [[TMP4]], [[TMP5]]
// CHECK:   store volatile <16 x i8> [[ADD2]], ptr @sc, align 8
// CHECK:   [[TMP6:%.*]] = load volatile <16 x i8>, ptr @uc, align 8
// CHECK:   [[TMP7:%.*]] = load volatile <16 x i8>, ptr @uc2, align 8
// CHECK:   [[ADD3:%.*]] = add <16 x i8> [[TMP6]], [[TMP7]]
// CHECK:   store volatile <16 x i8> [[ADD3]], ptr @uc, align 8
// CHECK:   [[TMP8:%.*]] = load volatile <16 x i8>, ptr @uc, align 8
// CHECK:   [[TMP9:%.*]] = load volatile <16 x i8>, ptr @bc2, align 8
// CHECK:   [[ADD4:%.*]] = add <16 x i8> [[TMP8]], [[TMP9]]
// CHECK:   store volatile <16 x i8> [[ADD4]], ptr @uc, align 8
// CHECK:   [[TMP10:%.*]] = load volatile <16 x i8>, ptr @bc, align 8
// CHECK:   [[TMP11:%.*]] = load volatile <16 x i8>, ptr @uc2, align 8
// CHECK:   [[ADD5:%.*]] = add <16 x i8> [[TMP10]], [[TMP11]]
// CHECK:   store volatile <16 x i8> [[ADD5]], ptr @uc, align 8
// CHECK:   [[TMP12:%.*]] = load volatile <8 x i16>, ptr @ss, align 8
// CHECK:   [[TMP13:%.*]] = load volatile <8 x i16>, ptr @ss2, align 8
// CHECK:   [[ADD6:%.*]] = add <8 x i16> [[TMP12]], [[TMP13]]
// CHECK:   store volatile <8 x i16> [[ADD6]], ptr @ss, align 8
// CHECK:   [[TMP14:%.*]] = load volatile <8 x i16>, ptr @ss, align 8
// CHECK:   [[TMP15:%.*]] = load volatile <8 x i16>, ptr @bs2, align 8
// CHECK:   [[ADD7:%.*]] = add <8 x i16> [[TMP14]], [[TMP15]]
// CHECK:   store volatile <8 x i16> [[ADD7]], ptr @ss, align 8
// CHECK:   [[TMP16:%.*]] = load volatile <8 x i16>, ptr @bs, align 8
// CHECK:   [[TMP17:%.*]] = load volatile <8 x i16>, ptr @ss2, align 8
// CHECK:   [[ADD8:%.*]] = add <8 x i16> [[TMP16]], [[TMP17]]
// CHECK:   store volatile <8 x i16> [[ADD8]], ptr @ss, align 8
// CHECK:   [[TMP18:%.*]] = load volatile <8 x i16>, ptr @us, align 8
// CHECK:   [[TMP19:%.*]] = load volatile <8 x i16>, ptr @us2, align 8
// CHECK:   [[ADD9:%.*]] = add <8 x i16> [[TMP18]], [[TMP19]]
// CHECK:   store volatile <8 x i16> [[ADD9]], ptr @us, align 8
// CHECK:   [[TMP20:%.*]] = load volatile <8 x i16>, ptr @us, align 8
// CHECK:   [[TMP21:%.*]] = load volatile <8 x i16>, ptr @bs2, align 8
// CHECK:   [[ADD10:%.*]] = add <8 x i16> [[TMP20]], [[TMP21]]
// CHECK:   store volatile <8 x i16> [[ADD10]], ptr @us, align 8
// CHECK:   [[TMP22:%.*]] = load volatile <8 x i16>, ptr @bs, align 8
// CHECK:   [[TMP23:%.*]] = load volatile <8 x i16>, ptr @us2, align 8
// CHECK:   [[ADD11:%.*]] = add <8 x i16> [[TMP22]], [[TMP23]]
// CHECK:   store volatile <8 x i16> [[ADD11]], ptr @us, align 8
// CHECK:   [[TMP24:%.*]] = load volatile <4 x i32>, ptr @si, align 8
// CHECK:   [[TMP25:%.*]] = load volatile <4 x i32>, ptr @si2, align 8
// CHECK:   [[ADD12:%.*]] = add <4 x i32> [[TMP24]], [[TMP25]]
// CHECK:   store volatile <4 x i32> [[ADD12]], ptr @si, align 8
// CHECK:   [[TMP26:%.*]] = load volatile <4 x i32>, ptr @si, align 8
// CHECK:   [[TMP27:%.*]] = load volatile <4 x i32>, ptr @bi2, align 8
// CHECK:   [[ADD13:%.*]] = add <4 x i32> [[TMP26]], [[TMP27]]
// CHECK:   store volatile <4 x i32> [[ADD13]], ptr @si, align 8
// CHECK:   [[TMP28:%.*]] = load volatile <4 x i32>, ptr @bi, align 8
// CHECK:   [[TMP29:%.*]] = load volatile <4 x i32>, ptr @si2, align 8
// CHECK:   [[ADD14:%.*]] = add <4 x i32> [[TMP28]], [[TMP29]]
// CHECK:   store volatile <4 x i32> [[ADD14]], ptr @si, align 8
// CHECK:   [[TMP30:%.*]] = load volatile <4 x i32>, ptr @ui, align 8
// CHECK:   [[TMP31:%.*]] = load volatile <4 x i32>, ptr @ui2, align 8
// CHECK:   [[ADD15:%.*]] = add <4 x i32> [[TMP30]], [[TMP31]]
// CHECK:   store volatile <4 x i32> [[ADD15]], ptr @ui, align 8
// CHECK:   [[TMP32:%.*]] = load volatile <4 x i32>, ptr @ui, align 8
// CHECK:   [[TMP33:%.*]] = load volatile <4 x i32>, ptr @bi2, align 8
// CHECK:   [[ADD16:%.*]] = add <4 x i32> [[TMP32]], [[TMP33]]
// CHECK:   store volatile <4 x i32> [[ADD16]], ptr @ui, align 8
// CHECK:   [[TMP34:%.*]] = load volatile <4 x i32>, ptr @bi, align 8
// CHECK:   [[TMP35:%.*]] = load volatile <4 x i32>, ptr @ui2, align 8
// CHECK:   [[ADD17:%.*]] = add <4 x i32> [[TMP34]], [[TMP35]]
// CHECK:   store volatile <4 x i32> [[ADD17]], ptr @ui, align 8
// CHECK:   [[TMP36:%.*]] = load volatile <2 x i64>, ptr @sl, align 8
// CHECK:   [[TMP37:%.*]] = load volatile <2 x i64>, ptr @sl2, align 8
// CHECK:   [[ADD18:%.*]] = add <2 x i64> [[TMP36]], [[TMP37]]
// CHECK:   store volatile <2 x i64> [[ADD18]], ptr @sl, align 8
// CHECK:   [[TMP38:%.*]] = load volatile <2 x i64>, ptr @sl, align 8
// CHECK:   [[TMP39:%.*]] = load volatile <2 x i64>, ptr @bl2, align 8
// CHECK:   [[ADD19:%.*]] = add <2 x i64> [[TMP38]], [[TMP39]]
// CHECK:   store volatile <2 x i64> [[ADD19]], ptr @sl, align 8
// CHECK:   [[TMP40:%.*]] = load volatile <2 x i64>, ptr @bl, align 8
// CHECK:   [[TMP41:%.*]] = load volatile <2 x i64>, ptr @sl2, align 8
// CHECK:   [[ADD20:%.*]] = add <2 x i64> [[TMP40]], [[TMP41]]
// CHECK:   store volatile <2 x i64> [[ADD20]], ptr @sl, align 8
// CHECK:   [[TMP42:%.*]] = load volatile <2 x i64>, ptr @ul, align 8
// CHECK:   [[TMP43:%.*]] = load volatile <2 x i64>, ptr @ul2, align 8
// CHECK:   [[ADD21:%.*]] = add <2 x i64> [[TMP42]], [[TMP43]]
// CHECK:   store volatile <2 x i64> [[ADD21]], ptr @ul, align 8
// CHECK:   [[TMP44:%.*]] = load volatile <2 x i64>, ptr @ul, align 8
// CHECK:   [[TMP45:%.*]] = load volatile <2 x i64>, ptr @bl2, align 8
// CHECK:   [[ADD22:%.*]] = add <2 x i64> [[TMP44]], [[TMP45]]
// CHECK:   store volatile <2 x i64> [[ADD22]], ptr @ul, align 8
// CHECK:   [[TMP46:%.*]] = load volatile <2 x i64>, ptr @bl, align 8
// CHECK:   [[TMP47:%.*]] = load volatile <2 x i64>, ptr @ul2, align 8
// CHECK:   [[ADD23:%.*]] = add <2 x i64> [[TMP46]], [[TMP47]]
// CHECK:   store volatile <2 x i64> [[ADD23]], ptr @ul, align 8
// CHECK:   [[TMP48:%.*]] = load volatile <2 x double>, ptr @fd, align 8
// CHECK:   [[TMP49:%.*]] = load volatile <2 x double>, ptr @fd2, align 8
// CHECK:   [[ADD24:%.*]] = fadd <2 x double> [[TMP48]], [[TMP49]]
// CHECK:   store volatile <2 x double> [[ADD24]], ptr @fd, align 8
// CHECK:   ret void
void test_add(void) {

  sc = sc + sc2;
  sc = sc + bc2;
  sc = bc + sc2;
  uc = uc + uc2;
  uc = uc + bc2;
  uc = bc + uc2;

  ss = ss + ss2;
  ss = ss + bs2;
  ss = bs + ss2;
  us = us + us2;
  us = us + bs2;
  us = bs + us2;

  si = si + si2;
  si = si + bi2;
  si = bi + si2;
  ui = ui + ui2;
  ui = ui + bi2;
  ui = bi + ui2;

  sl = sl + sl2;
  sl = sl + bl2;
  sl = bl + sl2;
  ul = ul + ul2;
  ul = ul + bl2;
  ul = bl + ul2;

  fd = fd + fd2;
}

// CHECK-LABEL: define{{.*}} void @test_add_assign() #0 {
// CHECK:   [[TMP0:%.*]] = load volatile <16 x i8>, ptr @sc2, align 8
// CHECK:   [[TMP1:%.*]] = load volatile <16 x i8>, ptr @sc, align 8
// CHECK:   [[ADD:%.*]] = add <16 x i8> [[TMP1]], [[TMP0]]
// CHECK:   store volatile <16 x i8> [[ADD]], ptr @sc, align 8
// CHECK:   [[TMP2:%.*]] = load volatile <16 x i8>, ptr @bc2, align 8
// CHECK:   [[TMP3:%.*]] = load volatile <16 x i8>, ptr @sc, align 8
// CHECK:   [[ADD1:%.*]] = add <16 x i8> [[TMP3]], [[TMP2]]
// CHECK:   store volatile <16 x i8> [[ADD1]], ptr @sc, align 8
// CHECK:   [[TMP4:%.*]] = load volatile <16 x i8>, ptr @uc2, align 8
// CHECK:   [[TMP5:%.*]] = load volatile <16 x i8>, ptr @uc, align 8
// CHECK:   [[ADD2:%.*]] = add <16 x i8> [[TMP5]], [[TMP4]]
// CHECK:   store volatile <16 x i8> [[ADD2]], ptr @uc, align 8
// CHECK:   [[TMP6:%.*]] = load volatile <16 x i8>, ptr @bc2, align 8
// CHECK:   [[TMP7:%.*]] = load volatile <16 x i8>, ptr @uc, align 8
// CHECK:   [[ADD3:%.*]] = add <16 x i8> [[TMP7]], [[TMP6]]
// CHECK:   store volatile <16 x i8> [[ADD3]], ptr @uc, align 8
// CHECK:   [[TMP8:%.*]] = load volatile <8 x i16>, ptr @ss2, align 8
// CHECK:   [[TMP9:%.*]] = load volatile <8 x i16>, ptr @ss, align 8
// CHECK:   [[ADD4:%.*]] = add <8 x i16> [[TMP9]], [[TMP8]]
// CHECK:   store volatile <8 x i16> [[ADD4]], ptr @ss, align 8
// CHECK:   [[TMP10:%.*]] = load volatile <8 x i16>, ptr @bs2, align 8
// CHECK:   [[TMP11:%.*]] = load volatile <8 x i16>, ptr @ss, align 8
// CHECK:   [[ADD5:%.*]] = add <8 x i16> [[TMP11]], [[TMP10]]
// CHECK:   store volatile <8 x i16> [[ADD5]], ptr @ss, align 8
// CHECK:   [[TMP12:%.*]] = load volatile <8 x i16>, ptr @us2, align 8
// CHECK:   [[TMP13:%.*]] = load volatile <8 x i16>, ptr @us, align 8
// CHECK:   [[ADD6:%.*]] = add <8 x i16> [[TMP13]], [[TMP12]]
// CHECK:   store volatile <8 x i16> [[ADD6]], ptr @us, align 8
// CHECK:   [[TMP14:%.*]] = load volatile <8 x i16>, ptr @bs2, align 8
// CHECK:   [[TMP15:%.*]] = load volatile <8 x i16>, ptr @us, align 8
// CHECK:   [[ADD7:%.*]] = add <8 x i16> [[TMP15]], [[TMP14]]
// CHECK:   store volatile <8 x i16> [[ADD7]], ptr @us, align 8
// CHECK:   [[TMP16:%.*]] = load volatile <4 x i32>, ptr @si2, align 8
// CHECK:   [[TMP17:%.*]] = load volatile <4 x i32>, ptr @si, align 8
// CHECK:   [[ADD8:%.*]] = add <4 x i32> [[TMP17]], [[TMP16]]
// CHECK:   store volatile <4 x i32> [[ADD8]], ptr @si, align 8
// CHECK:   [[TMP18:%.*]] = load volatile <4 x i32>, ptr @bi2, align 8
// CHECK:   [[TMP19:%.*]] = load volatile <4 x i32>, ptr @si, align 8
// CHECK:   [[ADD9:%.*]] = add <4 x i32> [[TMP19]], [[TMP18]]
// CHECK:   store volatile <4 x i32> [[ADD9]], ptr @si, align 8
// CHECK:   [[TMP20:%.*]] = load volatile <4 x i32>, ptr @ui2, align 8
// CHECK:   [[TMP21:%.*]] = load volatile <4 x i32>, ptr @ui, align 8
// CHECK:   [[ADD10:%.*]] = add <4 x i32> [[TMP21]], [[TMP20]]
// CHECK:   store volatile <4 x i32> [[ADD10]], ptr @ui, align 8
// CHECK:   [[TMP22:%.*]] = load volatile <4 x i32>, ptr @bi2, align 8
// CHECK:   [[TMP23:%.*]] = load volatile <4 x i32>, ptr @ui, align 8
// CHECK:   [[ADD11:%.*]] = add <4 x i32> [[TMP23]], [[TMP22]]
// CHECK:   store volatile <4 x i32> [[ADD11]], ptr @ui, align 8
// CHECK:   [[TMP24:%.*]] = load volatile <2 x i64>, ptr @sl2, align 8
// CHECK:   [[TMP25:%.*]] = load volatile <2 x i64>, ptr @sl, align 8
// CHECK:   [[ADD12:%.*]] = add <2 x i64> [[TMP25]], [[TMP24]]
// CHECK:   store volatile <2 x i64> [[ADD12]], ptr @sl, align 8
// CHECK:   [[TMP26:%.*]] = load volatile <2 x i64>, ptr @bl2, align 8
// CHECK:   [[TMP27:%.*]] = load volatile <2 x i64>, ptr @sl, align 8
// CHECK:   [[ADD13:%.*]] = add <2 x i64> [[TMP27]], [[TMP26]]
// CHECK:   store volatile <2 x i64> [[ADD13]], ptr @sl, align 8
// CHECK:   [[TMP28:%.*]] = load volatile <2 x i64>, ptr @ul2, align 8
// CHECK:   [[TMP29:%.*]] = load volatile <2 x i64>, ptr @ul, align 8
// CHECK:   [[ADD14:%.*]] = add <2 x i64> [[TMP29]], [[TMP28]]
// CHECK:   store volatile <2 x i64> [[ADD14]], ptr @ul, align 8
// CHECK:   [[TMP30:%.*]] = load volatile <2 x i64>, ptr @bl2, align 8
// CHECK:   [[TMP31:%.*]] = load volatile <2 x i64>, ptr @ul, align 8
// CHECK:   [[ADD15:%.*]] = add <2 x i64> [[TMP31]], [[TMP30]]
// CHECK:   store volatile <2 x i64> [[ADD15]], ptr @ul, align 8
// CHECK:   [[TMP32:%.*]] = load volatile <2 x double>, ptr @fd2, align 8
// CHECK:   [[TMP33:%.*]] = load volatile <2 x double>, ptr @fd, align 8
// CHECK:   [[ADD16:%.*]] = fadd <2 x double> [[TMP33]], [[TMP32]]
// CHECK:   store volatile <2 x double> [[ADD16]], ptr @fd, align 8
// CHECK:   ret void
void test_add_assign(void) {

  sc += sc2;
  sc += bc2;
  uc += uc2;
  uc += bc2;

  ss += ss2;
  ss += bs2;
  us += us2;
  us += bs2;

  si += si2;
  si += bi2;
  ui += ui2;
  ui += bi2;

  sl += sl2;
  sl += bl2;
  ul += ul2;
  ul += bl2;

  fd += fd2;
}

// CHECK-LABEL: define{{.*}} void @test_sub() #0 {
// CHECK:   [[TMP0:%.*]] = load volatile <16 x i8>, ptr @sc, align 8
// CHECK:   [[TMP1:%.*]] = load volatile <16 x i8>, ptr @sc2, align 8
// CHECK:   [[SUB:%.*]] = sub <16 x i8> [[TMP0]], [[TMP1]]
// CHECK:   store volatile <16 x i8> [[SUB]], ptr @sc, align 8
// CHECK:   [[TMP2:%.*]] = load volatile <16 x i8>, ptr @sc, align 8
// CHECK:   [[TMP3:%.*]] = load volatile <16 x i8>, ptr @bc2, align 8
// CHECK:   [[SUB1:%.*]] = sub <16 x i8> [[TMP2]], [[TMP3]]
// CHECK:   store volatile <16 x i8> [[SUB1]], ptr @sc, align 8
// CHECK:   [[TMP4:%.*]] = load volatile <16 x i8>, ptr @bc, align 8
// CHECK:   [[TMP5:%.*]] = load volatile <16 x i8>, ptr @sc2, align 8
// CHECK:   [[SUB2:%.*]] = sub <16 x i8> [[TMP4]], [[TMP5]]
// CHECK:   store volatile <16 x i8> [[SUB2]], ptr @sc, align 8
// CHECK:   [[TMP6:%.*]] = load volatile <16 x i8>, ptr @uc, align 8
// CHECK:   [[TMP7:%.*]] = load volatile <16 x i8>, ptr @uc2, align 8
// CHECK:   [[SUB3:%.*]] = sub <16 x i8> [[TMP6]], [[TMP7]]
// CHECK:   store volatile <16 x i8> [[SUB3]], ptr @uc, align 8
// CHECK:   [[TMP8:%.*]] = load volatile <16 x i8>, ptr @uc, align 8
// CHECK:   [[TMP9:%.*]] = load volatile <16 x i8>, ptr @bc2, align 8
// CHECK:   [[SUB4:%.*]] = sub <16 x i8> [[TMP8]], [[TMP9]]
// CHECK:   store volatile <16 x i8> [[SUB4]], ptr @uc, align 8
// CHECK:   [[TMP10:%.*]] = load volatile <16 x i8>, ptr @bc, align 8
// CHECK:   [[TMP11:%.*]] = load volatile <16 x i8>, ptr @uc2, align 8
// CHECK:   [[SUB5:%.*]] = sub <16 x i8> [[TMP10]], [[TMP11]]
// CHECK:   store volatile <16 x i8> [[SUB5]], ptr @uc, align 8
// CHECK:   [[TMP12:%.*]] = load volatile <8 x i16>, ptr @ss, align 8
// CHECK:   [[TMP13:%.*]] = load volatile <8 x i16>, ptr @ss2, align 8
// CHECK:   [[SUB6:%.*]] = sub <8 x i16> [[TMP12]], [[TMP13]]
// CHECK:   store volatile <8 x i16> [[SUB6]], ptr @ss, align 8
// CHECK:   [[TMP14:%.*]] = load volatile <8 x i16>, ptr @ss, align 8
// CHECK:   [[TMP15:%.*]] = load volatile <8 x i16>, ptr @bs2, align 8
// CHECK:   [[SUB7:%.*]] = sub <8 x i16> [[TMP14]], [[TMP15]]
// CHECK:   store volatile <8 x i16> [[SUB7]], ptr @ss, align 8
// CHECK:   [[TMP16:%.*]] = load volatile <8 x i16>, ptr @bs, align 8
// CHECK:   [[TMP17:%.*]] = load volatile <8 x i16>, ptr @ss2, align 8
// CHECK:   [[SUB8:%.*]] = sub <8 x i16> [[TMP16]], [[TMP17]]
// CHECK:   store volatile <8 x i16> [[SUB8]], ptr @ss, align 8
// CHECK:   [[TMP18:%.*]] = load volatile <8 x i16>, ptr @us, align 8
// CHECK:   [[TMP19:%.*]] = load volatile <8 x i16>, ptr @us2, align 8
// CHECK:   [[SUB9:%.*]] = sub <8 x i16> [[TMP18]], [[TMP19]]
// CHECK:   store volatile <8 x i16> [[SUB9]], ptr @us, align 8
// CHECK:   [[TMP20:%.*]] = load volatile <8 x i16>, ptr @us, align 8
// CHECK:   [[TMP21:%.*]] = load volatile <8 x i16>, ptr @bs2, align 8
// CHECK:   [[SUB10:%.*]] = sub <8 x i16> [[TMP20]], [[TMP21]]
// CHECK:   store volatile <8 x i16> [[SUB10]], ptr @us, align 8
// CHECK:   [[TMP22:%.*]] = load volatile <8 x i16>, ptr @bs, align 8
// CHECK:   [[TMP23:%.*]] = load volatile <8 x i16>, ptr @us2, align 8
// CHECK:   [[SUB11:%.*]] = sub <8 x i16> [[TMP22]], [[TMP23]]
// CHECK:   store volatile <8 x i16> [[SUB11]], ptr @us, align 8
// CHECK:   [[TMP24:%.*]] = load volatile <4 x i32>, ptr @si, align 8
// CHECK:   [[TMP25:%.*]] = load volatile <4 x i32>, ptr @si2, align 8
// CHECK:   [[SUB12:%.*]] = sub <4 x i32> [[TMP24]], [[TMP25]]
// CHECK:   store volatile <4 x i32> [[SUB12]], ptr @si, align 8
// CHECK:   [[TMP26:%.*]] = load volatile <4 x i32>, ptr @si, align 8
// CHECK:   [[TMP27:%.*]] = load volatile <4 x i32>, ptr @bi2, align 8
// CHECK:   [[SUB13:%.*]] = sub <4 x i32> [[TMP26]], [[TMP27]]
// CHECK:   store volatile <4 x i32> [[SUB13]], ptr @si, align 8
// CHECK:   [[TMP28:%.*]] = load volatile <4 x i32>, ptr @bi, align 8
// CHECK:   [[TMP29:%.*]] = load volatile <4 x i32>, ptr @si2, align 8
// CHECK:   [[SUB14:%.*]] = sub <4 x i32> [[TMP28]], [[TMP29]]
// CHECK:   store volatile <4 x i32> [[SUB14]], ptr @si, align 8
// CHECK:   [[TMP30:%.*]] = load volatile <4 x i32>, ptr @ui, align 8
// CHECK:   [[TMP31:%.*]] = load volatile <4 x i32>, ptr @ui2, align 8
// CHECK:   [[SUB15:%.*]] = sub <4 x i32> [[TMP30]], [[TMP31]]
// CHECK:   store volatile <4 x i32> [[SUB15]], ptr @ui, align 8
// CHECK:   [[TMP32:%.*]] = load volatile <4 x i32>, ptr @ui, align 8
// CHECK:   [[TMP33:%.*]] = load volatile <4 x i32>, ptr @bi2, align 8
// CHECK:   [[SUB16:%.*]] = sub <4 x i32> [[TMP32]], [[TMP33]]
// CHECK:   store volatile <4 x i32> [[SUB16]], ptr @ui, align 8
// CHECK:   [[TMP34:%.*]] = load volatile <4 x i32>, ptr @bi, align 8
// CHECK:   [[TMP35:%.*]] = load volatile <4 x i32>, ptr @ui2, align 8
// CHECK:   [[SUB17:%.*]] = sub <4 x i32> [[TMP34]], [[TMP35]]
// CHECK:   store volatile <4 x i32> [[SUB17]], ptr @ui, align 8
// CHECK:   [[TMP36:%.*]] = load volatile <2 x i64>, ptr @sl, align 8
// CHECK:   [[TMP37:%.*]] = load volatile <2 x i64>, ptr @sl2, align 8
// CHECK:   [[SUB18:%.*]] = sub <2 x i64> [[TMP36]], [[TMP37]]
// CHECK:   store volatile <2 x i64> [[SUB18]], ptr @sl, align 8
// CHECK:   [[TMP38:%.*]] = load volatile <2 x i64>, ptr @sl, align 8
// CHECK:   [[TMP39:%.*]] = load volatile <2 x i64>, ptr @bl2, align 8
// CHECK:   [[SUB19:%.*]] = sub <2 x i64> [[TMP38]], [[TMP39]]
// CHECK:   store volatile <2 x i64> [[SUB19]], ptr @sl, align 8
// CHECK:   [[TMP40:%.*]] = load volatile <2 x i64>, ptr @bl, align 8
// CHECK:   [[TMP41:%.*]] = load volatile <2 x i64>, ptr @sl2, align 8
// CHECK:   [[SUB20:%.*]] = sub <2 x i64> [[TMP40]], [[TMP41]]
// CHECK:   store volatile <2 x i64> [[SUB20]], ptr @sl, align 8
// CHECK:   [[TMP42:%.*]] = load volatile <2 x i64>, ptr @ul, align 8
// CHECK:   [[TMP43:%.*]] = load volatile <2 x i64>, ptr @ul2, align 8
// CHECK:   [[SUB21:%.*]] = sub <2 x i64> [[TMP42]], [[TMP43]]
// CHECK:   store volatile <2 x i64> [[SUB21]], ptr @ul, align 8
// CHECK:   [[TMP44:%.*]] = load volatile <2 x i64>, ptr @ul, align 8
// CHECK:   [[TMP45:%.*]] = load volatile <2 x i64>, ptr @bl2, align 8
// CHECK:   [[SUB22:%.*]] = sub <2 x i64> [[TMP44]], [[TMP45]]
// CHECK:   store volatile <2 x i64> [[SUB22]], ptr @ul, align 8
// CHECK:   [[TMP46:%.*]] = load volatile <2 x i64>, ptr @bl, align 8
// CHECK:   [[TMP47:%.*]] = load volatile <2 x i64>, ptr @ul2, align 8
// CHECK:   [[SUB23:%.*]] = sub <2 x i64> [[TMP46]], [[TMP47]]
// CHECK:   store volatile <2 x i64> [[SUB23]], ptr @ul, align 8
// CHECK:   [[TMP48:%.*]] = load volatile <2 x double>, ptr @fd, align 8
// CHECK:   [[TMP49:%.*]] = load volatile <2 x double>, ptr @fd2, align 8
// CHECK:   [[SUB24:%.*]] = fsub <2 x double> [[TMP48]], [[TMP49]]
// CHECK:   store volatile <2 x double> [[SUB24]], ptr @fd, align 8
// CHECK:   ret void
void test_sub(void) {

  sc = sc - sc2;
  sc = sc - bc2;
  sc = bc - sc2;
  uc = uc - uc2;
  uc = uc - bc2;
  uc = bc - uc2;

  ss = ss - ss2;
  ss = ss - bs2;
  ss = bs - ss2;
  us = us - us2;
  us = us - bs2;
  us = bs - us2;

  si = si - si2;
  si = si - bi2;
  si = bi - si2;
  ui = ui - ui2;
  ui = ui - bi2;
  ui = bi - ui2;

  sl = sl - sl2;
  sl = sl - bl2;
  sl = bl - sl2;
  ul = ul - ul2;
  ul = ul - bl2;
  ul = bl - ul2;

  fd = fd - fd2;
}

// CHECK-LABEL: define{{.*}} void @test_sub_assign() #0 {
// CHECK:   [[TMP0:%.*]] = load volatile <16 x i8>, ptr @sc2, align 8
// CHECK:   [[TMP1:%.*]] = load volatile <16 x i8>, ptr @sc, align 8
// CHECK:   [[SUB:%.*]] = sub <16 x i8> [[TMP1]], [[TMP0]]
// CHECK:   store volatile <16 x i8> [[SUB]], ptr @sc, align 8
// CHECK:   [[TMP2:%.*]] = load volatile <16 x i8>, ptr @bc2, align 8
// CHECK:   [[TMP3:%.*]] = load volatile <16 x i8>, ptr @sc, align 8
// CHECK:   [[SUB1:%.*]] = sub <16 x i8> [[TMP3]], [[TMP2]]
// CHECK:   store volatile <16 x i8> [[SUB1]], ptr @sc, align 8
// CHECK:   [[TMP4:%.*]] = load volatile <16 x i8>, ptr @uc2, align 8
// CHECK:   [[TMP5:%.*]] = load volatile <16 x i8>, ptr @uc, align 8
// CHECK:   [[SUB2:%.*]] = sub <16 x i8> [[TMP5]], [[TMP4]]
// CHECK:   store volatile <16 x i8> [[SUB2]], ptr @uc, align 8
// CHECK:   [[TMP6:%.*]] = load volatile <16 x i8>, ptr @bc2, align 8
// CHECK:   [[TMP7:%.*]] = load volatile <16 x i8>, ptr @uc, align 8
// CHECK:   [[SUB3:%.*]] = sub <16 x i8> [[TMP7]], [[TMP6]]
// CHECK:   store volatile <16 x i8> [[SUB3]], ptr @uc, align 8
// CHECK:   [[TMP8:%.*]] = load volatile <8 x i16>, ptr @ss2, align 8
// CHECK:   [[TMP9:%.*]] = load volatile <8 x i16>, ptr @ss, align 8
// CHECK:   [[SUB4:%.*]] = sub <8 x i16> [[TMP9]], [[TMP8]]
// CHECK:   store volatile <8 x i16> [[SUB4]], ptr @ss, align 8
// CHECK:   [[TMP10:%.*]] = load volatile <8 x i16>, ptr @bs2, align 8
// CHECK:   [[TMP11:%.*]] = load volatile <8 x i16>, ptr @ss, align 8
// CHECK:   [[SUB5:%.*]] = sub <8 x i16> [[TMP11]], [[TMP10]]
// CHECK:   store volatile <8 x i16> [[SUB5]], ptr @ss, align 8
// CHECK:   [[TMP12:%.*]] = load volatile <8 x i16>, ptr @us2, align 8
// CHECK:   [[TMP13:%.*]] = load volatile <8 x i16>, ptr @us, align 8
// CHECK:   [[SUB6:%.*]] = sub <8 x i16> [[TMP13]], [[TMP12]]
// CHECK:   store volatile <8 x i16> [[SUB6]], ptr @us, align 8
// CHECK:   [[TMP14:%.*]] = load volatile <8 x i16>, ptr @bs2, align 8
// CHECK:   [[TMP15:%.*]] = load volatile <8 x i16>, ptr @us, align 8
// CHECK:   [[SUB7:%.*]] = sub <8 x i16> [[TMP15]], [[TMP14]]
// CHECK:   store volatile <8 x i16> [[SUB7]], ptr @us, align 8
// CHECK:   [[TMP16:%.*]] = load volatile <4 x i32>, ptr @si2, align 8
// CHECK:   [[TMP17:%.*]] = load volatile <4 x i32>, ptr @si, align 8
// CHECK:   [[SUB8:%.*]] = sub <4 x i32> [[TMP17]], [[TMP16]]
// CHECK:   store volatile <4 x i32> [[SUB8]], ptr @si, align 8
// CHECK:   [[TMP18:%.*]] = load volatile <4 x i32>, ptr @bi2, align 8
// CHECK:   [[TMP19:%.*]] = load volatile <4 x i32>, ptr @si, align 8
// CHECK:   [[SUB9:%.*]] = sub <4 x i32> [[TMP19]], [[TMP18]]
// CHECK:   store volatile <4 x i32> [[SUB9]], ptr @si, align 8
// CHECK:   [[TMP20:%.*]] = load volatile <4 x i32>, ptr @ui2, align 8
// CHECK:   [[TMP21:%.*]] = load volatile <4 x i32>, ptr @ui, align 8
// CHECK:   [[SUB10:%.*]] = sub <4 x i32> [[TMP21]], [[TMP20]]
// CHECK:   store volatile <4 x i32> [[SUB10]], ptr @ui, align 8
// CHECK:   [[TMP22:%.*]] = load volatile <4 x i32>, ptr @bi2, align 8
// CHECK:   [[TMP23:%.*]] = load volatile <4 x i32>, ptr @ui, align 8
// CHECK:   [[SUB11:%.*]] = sub <4 x i32> [[TMP23]], [[TMP22]]
// CHECK:   store volatile <4 x i32> [[SUB11]], ptr @ui, align 8
// CHECK:   [[TMP24:%.*]] = load volatile <2 x i64>, ptr @sl2, align 8
// CHECK:   [[TMP25:%.*]] = load volatile <2 x i64>, ptr @sl, align 8
// CHECK:   [[SUB12:%.*]] = sub <2 x i64> [[TMP25]], [[TMP24]]
// CHECK:   store volatile <2 x i64> [[SUB12]], ptr @sl, align 8
// CHECK:   [[TMP26:%.*]] = load volatile <2 x i64>, ptr @bl2, align 8
// CHECK:   [[TMP27:%.*]] = load volatile <2 x i64>, ptr @sl, align 8
// CHECK:   [[SUB13:%.*]] = sub <2 x i64> [[TMP27]], [[TMP26]]
// CHECK:   store volatile <2 x i64> [[SUB13]], ptr @sl, align 8
// CHECK:   [[TMP28:%.*]] = load volatile <2 x i64>, ptr @ul2, align 8
// CHECK:   [[TMP29:%.*]] = load volatile <2 x i64>, ptr @ul, align 8
// CHECK:   [[SUB14:%.*]] = sub <2 x i64> [[TMP29]], [[TMP28]]
// CHECK:   store volatile <2 x i64> [[SUB14]], ptr @ul, align 8
// CHECK:   [[TMP30:%.*]] = load volatile <2 x i64>, ptr @bl2, align 8
// CHECK:   [[TMP31:%.*]] = load volatile <2 x i64>, ptr @ul, align 8
// CHECK:   [[SUB15:%.*]] = sub <2 x i64> [[TMP31]], [[TMP30]]
// CHECK:   store volatile <2 x i64> [[SUB15]], ptr @ul, align 8
// CHECK:   [[TMP32:%.*]] = load volatile <2 x double>, ptr @fd2, align 8
// CHECK:   [[TMP33:%.*]] = load volatile <2 x double>, ptr @fd, align 8
// CHECK:   [[SUB16:%.*]] = fsub <2 x double> [[TMP33]], [[TMP32]]
// CHECK:   store volatile <2 x double> [[SUB16]], ptr @fd, align 8
// CHECK:   ret void
void test_sub_assign(void) {

  sc -= sc2;
  sc -= bc2;
  uc -= uc2;
  uc -= bc2;

  ss -= ss2;
  ss -= bs2;
  us -= us2;
  us -= bs2;

  si -= si2;
  si -= bi2;
  ui -= ui2;
  ui -= bi2;

  sl -= sl2;
  sl -= bl2;
  ul -= ul2;
  ul -= bl2;

  fd -= fd2;
}

// CHECK-LABEL: define{{.*}} void @test_mul() #0 {
// CHECK:   [[TMP0:%.*]] = load volatile <16 x i8>, ptr @sc, align 8
// CHECK:   [[TMP1:%.*]] = load volatile <16 x i8>, ptr @sc2, align 8
// CHECK:   [[MUL:%.*]] = mul <16 x i8> [[TMP0]], [[TMP1]]
// CHECK:   store volatile <16 x i8> [[MUL]], ptr @sc, align 8
// CHECK:   [[TMP2:%.*]] = load volatile <16 x i8>, ptr @uc, align 8
// CHECK:   [[TMP3:%.*]] = load volatile <16 x i8>, ptr @uc2, align 8
// CHECK:   [[MUL1:%.*]] = mul <16 x i8> [[TMP2]], [[TMP3]]
// CHECK:   store volatile <16 x i8> [[MUL1]], ptr @uc, align 8
// CHECK:   [[TMP4:%.*]] = load volatile <8 x i16>, ptr @ss, align 8
// CHECK:   [[TMP5:%.*]] = load volatile <8 x i16>, ptr @ss2, align 8
// CHECK:   [[MUL2:%.*]] = mul <8 x i16> [[TMP4]], [[TMP5]]
// CHECK:   store volatile <8 x i16> [[MUL2]], ptr @ss, align 8
// CHECK:   [[TMP6:%.*]] = load volatile <8 x i16>, ptr @us, align 8
// CHECK:   [[TMP7:%.*]] = load volatile <8 x i16>, ptr @us2, align 8
// CHECK:   [[MUL3:%.*]] = mul <8 x i16> [[TMP6]], [[TMP7]]
// CHECK:   store volatile <8 x i16> [[MUL3]], ptr @us, align 8
// CHECK:   [[TMP8:%.*]] = load volatile <4 x i32>, ptr @si, align 8
// CHECK:   [[TMP9:%.*]] = load volatile <4 x i32>, ptr @si2, align 8
// CHECK:   [[MUL4:%.*]] = mul <4 x i32> [[TMP8]], [[TMP9]]
// CHECK:   store volatile <4 x i32> [[MUL4]], ptr @si, align 8
// CHECK:   [[TMP10:%.*]] = load volatile <4 x i32>, ptr @ui, align 8
// CHECK:   [[TMP11:%.*]] = load volatile <4 x i32>, ptr @ui2, align 8
// CHECK:   [[MUL5:%.*]] = mul <4 x i32> [[TMP10]], [[TMP11]]
// CHECK:   store volatile <4 x i32> [[MUL5]], ptr @ui, align 8
// CHECK:   [[TMP12:%.*]] = load volatile <2 x i64>, ptr @sl, align 8
// CHECK:   [[TMP13:%.*]] = load volatile <2 x i64>, ptr @sl2, align 8
// CHECK:   [[MUL6:%.*]] = mul <2 x i64> [[TMP12]], [[TMP13]]
// CHECK:   store volatile <2 x i64> [[MUL6]], ptr @sl, align 8
// CHECK:   [[TMP14:%.*]] = load volatile <2 x i64>, ptr @ul, align 8
// CHECK:   [[TMP15:%.*]] = load volatile <2 x i64>, ptr @ul2, align 8
// CHECK:   [[MUL7:%.*]] = mul <2 x i64> [[TMP14]], [[TMP15]]
// CHECK:   store volatile <2 x i64> [[MUL7]], ptr @ul, align 8
// CHECK:   [[TMP16:%.*]] = load volatile <2 x double>, ptr @fd, align 8
// CHECK:   [[TMP17:%.*]] = load volatile <2 x double>, ptr @fd2, align 8
// CHECK:   [[MUL8:%.*]] = fmul <2 x double> [[TMP16]], [[TMP17]]
// CHECK:   store volatile <2 x double> [[MUL8]], ptr @fd, align 8
// CHECK:   ret void
void test_mul(void) {

  sc = sc * sc2;
  uc = uc * uc2;

  ss = ss * ss2;
  us = us * us2;

  si = si * si2;
  ui = ui * ui2;

  sl = sl * sl2;
  ul = ul * ul2;

  fd = fd * fd2;
}

// CHECK-LABEL: define{{.*}} void @test_mul_assign() #0 {
// CHECK:   [[TMP0:%.*]] = load volatile <16 x i8>, ptr @sc2, align 8
// CHECK:   [[TMP1:%.*]] = load volatile <16 x i8>, ptr @sc, align 8
// CHECK:   [[MUL:%.*]] = mul <16 x i8> [[TMP1]], [[TMP0]]
// CHECK:   store volatile <16 x i8> [[MUL]], ptr @sc, align 8
// CHECK:   [[TMP2:%.*]] = load volatile <16 x i8>, ptr @uc2, align 8
// CHECK:   [[TMP3:%.*]] = load volatile <16 x i8>, ptr @uc, align 8
// CHECK:   [[MUL1:%.*]] = mul <16 x i8> [[TMP3]], [[TMP2]]
// CHECK:   store volatile <16 x i8> [[MUL1]], ptr @uc, align 8
// CHECK:   [[TMP4:%.*]] = load volatile <8 x i16>, ptr @ss2, align 8
// CHECK:   [[TMP5:%.*]] = load volatile <8 x i16>, ptr @ss, align 8
// CHECK:   [[MUL2:%.*]] = mul <8 x i16> [[TMP5]], [[TMP4]]
// CHECK:   store volatile <8 x i16> [[MUL2]], ptr @ss, align 8
// CHECK:   [[TMP6:%.*]] = load volatile <8 x i16>, ptr @us2, align 8
// CHECK:   [[TMP7:%.*]] = load volatile <8 x i16>, ptr @us, align 8
// CHECK:   [[MUL3:%.*]] = mul <8 x i16> [[TMP7]], [[TMP6]]
// CHECK:   store volatile <8 x i16> [[MUL3]], ptr @us, align 8
// CHECK:   [[TMP8:%.*]] = load volatile <4 x i32>, ptr @si2, align 8
// CHECK:   [[TMP9:%.*]] = load volatile <4 x i32>, ptr @si, align 8
// CHECK:   [[MUL4:%.*]] = mul <4 x i32> [[TMP9]], [[TMP8]]
// CHECK:   store volatile <4 x i32> [[MUL4]], ptr @si, align 8
// CHECK:   [[TMP10:%.*]] = load volatile <4 x i32>, ptr @ui2, align 8
// CHECK:   [[TMP11:%.*]] = load volatile <4 x i32>, ptr @ui, align 8
// CHECK:   [[MUL5:%.*]] = mul <4 x i32> [[TMP11]], [[TMP10]]
// CHECK:   store volatile <4 x i32> [[MUL5]], ptr @ui, align 8
// CHECK:   [[TMP12:%.*]] = load volatile <2 x i64>, ptr @sl2, align 8
// CHECK:   [[TMP13:%.*]] = load volatile <2 x i64>, ptr @sl, align 8
// CHECK:   [[MUL6:%.*]] = mul <2 x i64> [[TMP13]], [[TMP12]]
// CHECK:   store volatile <2 x i64> [[MUL6]], ptr @sl, align 8
// CHECK:   [[TMP14:%.*]] = load volatile <2 x i64>, ptr @ul2, align 8
// CHECK:   [[TMP15:%.*]] = load volatile <2 x i64>, ptr @ul, align 8
// CHECK:   [[MUL7:%.*]] = mul <2 x i64> [[TMP15]], [[TMP14]]
// CHECK:   store volatile <2 x i64> [[MUL7]], ptr @ul, align 8
// CHECK:   [[TMP16:%.*]] = load volatile <2 x double>, ptr @fd2, align 8
// CHECK:   [[TMP17:%.*]] = load volatile <2 x double>, ptr @fd, align 8
// CHECK:   [[MUL8:%.*]] = fmul <2 x double> [[TMP17]], [[TMP16]]
// CHECK:   store volatile <2 x double> [[MUL8]], ptr @fd, align 8
// CHECK:   ret void
void test_mul_assign(void) {

  sc *= sc2;
  uc *= uc2;

  ss *= ss2;
  us *= us2;

  si *= si2;
  ui *= ui2;

  sl *= sl2;
  ul *= ul2;

  fd *= fd2;
}

// CHECK-LABEL: define{{.*}} void @test_div() #0 {
// CHECK:   [[TMP0:%.*]] = load volatile <16 x i8>, ptr @sc, align 8
// CHECK:   [[TMP1:%.*]] = load volatile <16 x i8>, ptr @sc2, align 8
// CHECK:   [[DIV:%.*]] = sdiv <16 x i8> [[TMP0]], [[TMP1]]
// CHECK:   store volatile <16 x i8> [[DIV]], ptr @sc, align 8
// CHECK:   [[TMP2:%.*]] = load volatile <16 x i8>, ptr @uc, align 8
// CHECK:   [[TMP3:%.*]] = load volatile <16 x i8>, ptr @uc2, align 8
// CHECK:   [[DIV1:%.*]] = udiv <16 x i8> [[TMP2]], [[TMP3]]
// CHECK:   store volatile <16 x i8> [[DIV1]], ptr @uc, align 8
// CHECK:   [[TMP4:%.*]] = load volatile <8 x i16>, ptr @ss, align 8
// CHECK:   [[TMP5:%.*]] = load volatile <8 x i16>, ptr @ss2, align 8
// CHECK:   [[DIV2:%.*]] = sdiv <8 x i16> [[TMP4]], [[TMP5]]
// CHECK:   store volatile <8 x i16> [[DIV2]], ptr @ss, align 8
// CHECK:   [[TMP6:%.*]] = load volatile <8 x i16>, ptr @us, align 8
// CHECK:   [[TMP7:%.*]] = load volatile <8 x i16>, ptr @us2, align 8
// CHECK:   [[DIV3:%.*]] = udiv <8 x i16> [[TMP6]], [[TMP7]]
// CHECK:   store volatile <8 x i16> [[DIV3]], ptr @us, align 8
// CHECK:   [[TMP8:%.*]] = load volatile <4 x i32>, ptr @si, align 8
// CHECK:   [[TMP9:%.*]] = load volatile <4 x i32>, ptr @si2, align 8
// CHECK:   [[DIV4:%.*]] = sdiv <4 x i32> [[TMP8]], [[TMP9]]
// CHECK:   store volatile <4 x i32> [[DIV4]], ptr @si, align 8
// CHECK:   [[TMP10:%.*]] = load volatile <4 x i32>, ptr @ui, align 8
// CHECK:   [[TMP11:%.*]] = load volatile <4 x i32>, ptr @ui2, align 8
// CHECK:   [[DIV5:%.*]] = udiv <4 x i32> [[TMP10]], [[TMP11]]
// CHECK:   store volatile <4 x i32> [[DIV5]], ptr @ui, align 8
// CHECK:   [[TMP12:%.*]] = load volatile <2 x i64>, ptr @sl, align 8
// CHECK:   [[TMP13:%.*]] = load volatile <2 x i64>, ptr @sl2, align 8
// CHECK:   [[DIV6:%.*]] = sdiv <2 x i64> [[TMP12]], [[TMP13]]
// CHECK:   store volatile <2 x i64> [[DIV6]], ptr @sl, align 8
// CHECK:   [[TMP14:%.*]] = load volatile <2 x i64>, ptr @ul, align 8
// CHECK:   [[TMP15:%.*]] = load volatile <2 x i64>, ptr @ul2, align 8
// CHECK:   [[DIV7:%.*]] = udiv <2 x i64> [[TMP14]], [[TMP15]]
// CHECK:   store volatile <2 x i64> [[DIV7]], ptr @ul, align 8
// CHECK:   [[TMP16:%.*]] = load volatile <2 x double>, ptr @fd, align 8
// CHECK:   [[TMP17:%.*]] = load volatile <2 x double>, ptr @fd2, align 8
// CHECK:   [[DIV8:%.*]] = fdiv <2 x double> [[TMP16]], [[TMP17]]
// CHECK:   store volatile <2 x double> [[DIV8]], ptr @fd, align 8
// CHECK:   ret void
void test_div(void) {

  sc = sc / sc2;
  uc = uc / uc2;

  ss = ss / ss2;
  us = us / us2;

  si = si / si2;
  ui = ui / ui2;

  sl = sl / sl2;
  ul = ul / ul2;

  fd = fd / fd2;
}

// CHECK-LABEL: define{{.*}} void @test_div_assign() #0 {
// CHECK:   [[TMP0:%.*]] = load volatile <16 x i8>, ptr @sc2, align 8
// CHECK:   [[TMP1:%.*]] = load volatile <16 x i8>, ptr @sc, align 8
// CHECK:   [[DIV:%.*]] = sdiv <16 x i8> [[TMP1]], [[TMP0]]
// CHECK:   store volatile <16 x i8> [[DIV]], ptr @sc, align 8
// CHECK:   [[TMP2:%.*]] = load volatile <16 x i8>, ptr @uc2, align 8
// CHECK:   [[TMP3:%.*]] = load volatile <16 x i8>, ptr @uc, align 8
// CHECK:   [[DIV1:%.*]] = udiv <16 x i8> [[TMP3]], [[TMP2]]
// CHECK:   store volatile <16 x i8> [[DIV1]], ptr @uc, align 8
// CHECK:   [[TMP4:%.*]] = load volatile <8 x i16>, ptr @ss2, align 8
// CHECK:   [[TMP5:%.*]] = load volatile <8 x i16>, ptr @ss, align 8
// CHECK:   [[DIV2:%.*]] = sdiv <8 x i16> [[TMP5]], [[TMP4]]
// CHECK:   store volatile <8 x i16> [[DIV2]], ptr @ss, align 8
// CHECK:   [[TMP6:%.*]] = load volatile <8 x i16>, ptr @us2, align 8
// CHECK:   [[TMP7:%.*]] = load volatile <8 x i16>, ptr @us, align 8
// CHECK:   [[DIV3:%.*]] = udiv <8 x i16> [[TMP7]], [[TMP6]]
// CHECK:   store volatile <8 x i16> [[DIV3]], ptr @us, align 8
// CHECK:   [[TMP8:%.*]] = load volatile <4 x i32>, ptr @si2, align 8
// CHECK:   [[TMP9:%.*]] = load volatile <4 x i32>, ptr @si, align 8
// CHECK:   [[DIV4:%.*]] = sdiv <4 x i32> [[TMP9]], [[TMP8]]
// CHECK:   store volatile <4 x i32> [[DIV4]], ptr @si, align 8
// CHECK:   [[TMP10:%.*]] = load volatile <4 x i32>, ptr @ui2, align 8
// CHECK:   [[TMP11:%.*]] = load volatile <4 x i32>, ptr @ui, align 8
// CHECK:   [[DIV5:%.*]] = udiv <4 x i32> [[TMP11]], [[TMP10]]
// CHECK:   store volatile <4 x i32> [[DIV5]], ptr @ui, align 8
// CHECK:   [[TMP12:%.*]] = load volatile <2 x i64>, ptr @sl2, align 8
// CHECK:   [[TMP13:%.*]] = load volatile <2 x i64>, ptr @sl, align 8
// CHECK:   [[DIV6:%.*]] = sdiv <2 x i64> [[TMP13]], [[TMP12]]
// CHECK:   store volatile <2 x i64> [[DIV6]], ptr @sl, align 8
// CHECK:   [[TMP14:%.*]] = load volatile <2 x i64>, ptr @ul2, align 8
// CHECK:   [[TMP15:%.*]] = load volatile <2 x i64>, ptr @ul, align 8
// CHECK:   [[DIV7:%.*]] = udiv <2 x i64> [[TMP15]], [[TMP14]]
// CHECK:   store volatile <2 x i64> [[DIV7]], ptr @ul, align 8
// CHECK:   [[TMP16:%.*]] = load volatile <2 x double>, ptr @fd2, align 8
// CHECK:   [[TMP17:%.*]] = load volatile <2 x double>, ptr @fd, align 8
// CHECK:   [[DIV8:%.*]] = fdiv <2 x double> [[TMP17]], [[TMP16]]
// CHECK:   store volatile <2 x double> [[DIV8]], ptr @fd, align 8
// CHECK:   ret void
void test_div_assign(void) {

  sc /= sc2;
  uc /= uc2;

  ss /= ss2;
  us /= us2;

  si /= si2;
  ui /= ui2;

  sl /= sl2;
  ul /= ul2;

  fd /= fd2;
}

// CHECK-LABEL: define{{.*}} void @test_rem() #0 {
// CHECK:   [[TMP0:%.*]] = load volatile <16 x i8>, ptr @sc, align 8
// CHECK:   [[TMP1:%.*]] = load volatile <16 x i8>, ptr @sc2, align 8
// CHECK:   [[REM:%.*]] = srem <16 x i8> [[TMP0]], [[TMP1]]
// CHECK:   store volatile <16 x i8> [[REM]], ptr @sc, align 8
// CHECK:   [[TMP2:%.*]] = load volatile <16 x i8>, ptr @uc, align 8
// CHECK:   [[TMP3:%.*]] = load volatile <16 x i8>, ptr @uc2, align 8
// CHECK:   [[REM1:%.*]] = urem <16 x i8> [[TMP2]], [[TMP3]]
// CHECK:   store volatile <16 x i8> [[REM1]], ptr @uc, align 8
// CHECK:   [[TMP4:%.*]] = load volatile <8 x i16>, ptr @ss, align 8
// CHECK:   [[TMP5:%.*]] = load volatile <8 x i16>, ptr @ss2, align 8
// CHECK:   [[REM2:%.*]] = srem <8 x i16> [[TMP4]], [[TMP5]]
// CHECK:   store volatile <8 x i16> [[REM2]], ptr @ss, align 8
// CHECK:   [[TMP6:%.*]] = load volatile <8 x i16>, ptr @us, align 8
// CHECK:   [[TMP7:%.*]] = load volatile <8 x i16>, ptr @us2, align 8
// CHECK:   [[REM3:%.*]] = urem <8 x i16> [[TMP6]], [[TMP7]]
// CHECK:   store volatile <8 x i16> [[REM3]], ptr @us, align 8
// CHECK:   [[TMP8:%.*]] = load volatile <4 x i32>, ptr @si, align 8
// CHECK:   [[TMP9:%.*]] = load volatile <4 x i32>, ptr @si2, align 8
// CHECK:   [[REM4:%.*]] = srem <4 x i32> [[TMP8]], [[TMP9]]
// CHECK:   store volatile <4 x i32> [[REM4]], ptr @si, align 8
// CHECK:   [[TMP10:%.*]] = load volatile <4 x i32>, ptr @ui, align 8
// CHECK:   [[TMP11:%.*]] = load volatile <4 x i32>, ptr @ui2, align 8
// CHECK:   [[REM5:%.*]] = urem <4 x i32> [[TMP10]], [[TMP11]]
// CHECK:   store volatile <4 x i32> [[REM5]], ptr @ui, align 8
// CHECK:   [[TMP12:%.*]] = load volatile <2 x i64>, ptr @sl, align 8
// CHECK:   [[TMP13:%.*]] = load volatile <2 x i64>, ptr @sl2, align 8
// CHECK:   [[REM6:%.*]] = srem <2 x i64> [[TMP12]], [[TMP13]]
// CHECK:   store volatile <2 x i64> [[REM6]], ptr @sl, align 8
// CHECK:   [[TMP14:%.*]] = load volatile <2 x i64>, ptr @ul, align 8
// CHECK:   [[TMP15:%.*]] = load volatile <2 x i64>, ptr @ul2, align 8
// CHECK:   [[REM7:%.*]] = urem <2 x i64> [[TMP14]], [[TMP15]]
// CHECK:   store volatile <2 x i64> [[REM7]], ptr @ul, align 8
// CHECK:   ret void
void test_rem(void) {

  sc = sc % sc2;
  uc = uc % uc2;

  ss = ss % ss2;
  us = us % us2;

  si = si % si2;
  ui = ui % ui2;

  sl = sl % sl2;
  ul = ul % ul2;
}

// CHECK-LABEL: define{{.*}} void @test_rem_assign() #0 {
// CHECK:   [[TMP0:%.*]] = load volatile <16 x i8>, ptr @sc2, align 8
// CHECK:   [[TMP1:%.*]] = load volatile <16 x i8>, ptr @sc, align 8
// CHECK:   [[REM:%.*]] = srem <16 x i8> [[TMP1]], [[TMP0]]
// CHECK:   store volatile <16 x i8> [[REM]], ptr @sc, align 8
// CHECK:   [[TMP2:%.*]] = load volatile <16 x i8>, ptr @uc2, align 8
// CHECK:   [[TMP3:%.*]] = load volatile <16 x i8>, ptr @uc, align 8
// CHECK:   [[REM1:%.*]] = urem <16 x i8> [[TMP3]], [[TMP2]]
// CHECK:   store volatile <16 x i8> [[REM1]], ptr @uc, align 8
// CHECK:   [[TMP4:%.*]] = load volatile <8 x i16>, ptr @ss2, align 8
// CHECK:   [[TMP5:%.*]] = load volatile <8 x i16>, ptr @ss, align 8
// CHECK:   [[REM2:%.*]] = srem <8 x i16> [[TMP5]], [[TMP4]]
// CHECK:   store volatile <8 x i16> [[REM2]], ptr @ss, align 8
// CHECK:   [[TMP6:%.*]] = load volatile <8 x i16>, ptr @us2, align 8
// CHECK:   [[TMP7:%.*]] = load volatile <8 x i16>, ptr @us, align 8
// CHECK:   [[REM3:%.*]] = urem <8 x i16> [[TMP7]], [[TMP6]]
// CHECK:   store volatile <8 x i16> [[REM3]], ptr @us, align 8
// CHECK:   [[TMP8:%.*]] = load volatile <4 x i32>, ptr @si2, align 8
// CHECK:   [[TMP9:%.*]] = load volatile <4 x i32>, ptr @si, align 8
// CHECK:   [[REM4:%.*]] = srem <4 x i32> [[TMP9]], [[TMP8]]
// CHECK:   store volatile <4 x i32> [[REM4]], ptr @si, align 8
// CHECK:   [[TMP10:%.*]] = load volatile <4 x i32>, ptr @ui2, align 8
// CHECK:   [[TMP11:%.*]] = load volatile <4 x i32>, ptr @ui, align 8
// CHECK:   [[REM5:%.*]] = urem <4 x i32> [[TMP11]], [[TMP10]]
// CHECK:   store volatile <4 x i32> [[REM5]], ptr @ui, align 8
// CHECK:   [[TMP12:%.*]] = load volatile <2 x i64>, ptr @sl2, align 8
// CHECK:   [[TMP13:%.*]] = load volatile <2 x i64>, ptr @sl, align 8
// CHECK:   [[REM6:%.*]] = srem <2 x i64> [[TMP13]], [[TMP12]]
// CHECK:   store volatile <2 x i64> [[REM6]], ptr @sl, align 8
// CHECK:   [[TMP14:%.*]] = load volatile <2 x i64>, ptr @ul2, align 8
// CHECK:   [[TMP15:%.*]] = load volatile <2 x i64>, ptr @ul, align 8
// CHECK:   [[REM7:%.*]] = urem <2 x i64> [[TMP15]], [[TMP14]]
// CHECK:   store volatile <2 x i64> [[REM7]], ptr @ul, align 8
// CHECK:   ret void
void test_rem_assign(void) {

  sc %= sc2;
  uc %= uc2;

  ss %= ss2;
  us %= us2;

  si %= si2;
  ui %= ui2;

  sl %= sl2;
  ul %= ul2;
}

// CHECK-LABEL: define{{.*}} void @test_not() #0 {
// CHECK:   [[TMP0:%.*]] = load volatile <16 x i8>, ptr @sc2, align 8
// CHECK:   [[NEG:%.*]] = xor <16 x i8> [[TMP0]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
// CHECK:   store volatile <16 x i8> [[NEG]], ptr @sc, align 8
// CHECK:   [[TMP1:%.*]] = load volatile <16 x i8>, ptr @uc2, align 8
// CHECK:   [[NEG1:%.*]] = xor <16 x i8> [[TMP1]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
// CHECK:   store volatile <16 x i8> [[NEG1]], ptr @uc, align 8
// CHECK:   [[TMP2:%.*]] = load volatile <16 x i8>, ptr @bc2, align 8
// CHECK:   [[NEG2:%.*]] = xor <16 x i8> [[TMP2]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
// CHECK:   store volatile <16 x i8> [[NEG2]], ptr @bc, align 8
// CHECK:   [[TMP3:%.*]] = load volatile <8 x i16>, ptr @ss2, align 8
// CHECK:   [[NEG3:%.*]] = xor <8 x i16> [[TMP3]], <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
// CHECK:   store volatile <8 x i16> [[NEG3]], ptr @ss, align 8
// CHECK:   [[TMP4:%.*]] = load volatile <8 x i16>, ptr @us2, align 8
// CHECK:   [[NEG4:%.*]] = xor <8 x i16> [[TMP4]], <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
// CHECK:   store volatile <8 x i16> [[NEG4]], ptr @us, align 8
// CHECK:   [[TMP5:%.*]] = load volatile <8 x i16>, ptr @bs2, align 8
// CHECK:   [[NEG5:%.*]] = xor <8 x i16> [[TMP5]], <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
// CHECK:   store volatile <8 x i16> [[NEG5]], ptr @bs, align 8
// CHECK:   [[TMP6:%.*]] = load volatile <4 x i32>, ptr @si2, align 8
// CHECK:   [[NEG6:%.*]] = xor <4 x i32> [[TMP6]], <i32 -1, i32 -1, i32 -1, i32 -1>
// CHECK:   store volatile <4 x i32> [[NEG6]], ptr @si, align 8
// CHECK:   [[TMP7:%.*]] = load volatile <4 x i32>, ptr @ui2, align 8
// CHECK:   [[NEG7:%.*]] = xor <4 x i32> [[TMP7]], <i32 -1, i32 -1, i32 -1, i32 -1>
// CHECK:   store volatile <4 x i32> [[NEG7]], ptr @ui, align 8
// CHECK:   [[TMP8:%.*]] = load volatile <4 x i32>, ptr @bi2, align 8
// CHECK:   [[NEG8:%.*]] = xor <4 x i32> [[TMP8]], <i32 -1, i32 -1, i32 -1, i32 -1>
// CHECK:   store volatile <4 x i32> [[NEG8]], ptr @bi, align 8
// CHECK:   [[TMP9:%.*]] = load volatile <2 x i64>, ptr @sl2, align 8
// CHECK:   [[NEG9:%.*]] = xor <2 x i64> [[TMP9]], <i64 -1, i64 -1>
// CHECK:   store volatile <2 x i64> [[NEG9]], ptr @sl, align 8
// CHECK:   [[TMP10:%.*]] = load volatile <2 x i64>, ptr @ul2, align 8
// CHECK:   [[NEG10:%.*]] = xor <2 x i64> [[TMP10]], <i64 -1, i64 -1>
// CHECK:   store volatile <2 x i64> [[NEG10]], ptr @ul, align 8
// CHECK:   [[TMP11:%.*]] = load volatile <2 x i64>, ptr @bl2, align 8
// CHECK:   [[NEG11:%.*]] = xor <2 x i64> [[TMP11]], <i64 -1, i64 -1>
// CHECK:   store volatile <2 x i64> [[NEG11]], ptr @bl, align 8
// CHECK:   ret void
void test_not(void) {

  sc = ~sc2;
  uc = ~uc2;
  bc = ~bc2;

  ss = ~ss2;
  us = ~us2;
  bs = ~bs2;

  si = ~si2;
  ui = ~ui2;
  bi = ~bi2;

  sl = ~sl2;
  ul = ~ul2;
  bl = ~bl2;
}

// CHECK-LABEL: define{{.*}} void @test_and() #0 {
// CHECK:   [[TMP0:%.*]] = load volatile <16 x i8>, ptr @sc, align 8
// CHECK:   [[TMP1:%.*]] = load volatile <16 x i8>, ptr @sc2, align 8
// CHECK:   [[AND:%.*]] = and <16 x i8> [[TMP0]], [[TMP1]]
// CHECK:   store volatile <16 x i8> [[AND]], ptr @sc, align 8
// CHECK:   [[TMP2:%.*]] = load volatile <16 x i8>, ptr @sc, align 8
// CHECK:   [[TMP3:%.*]] = load volatile <16 x i8>, ptr @bc2, align 8
// CHECK:   [[AND1:%.*]] = and <16 x i8> [[TMP2]], [[TMP3]]
// CHECK:   store volatile <16 x i8> [[AND1]], ptr @sc, align 8
// CHECK:   [[TMP4:%.*]] = load volatile <16 x i8>, ptr @bc, align 8
// CHECK:   [[TMP5:%.*]] = load volatile <16 x i8>, ptr @sc2, align 8
// CHECK:   [[AND2:%.*]] = and <16 x i8> [[TMP4]], [[TMP5]]
// CHECK:   store volatile <16 x i8> [[AND2]], ptr @sc, align 8
// CHECK:   [[TMP6:%.*]] = load volatile <16 x i8>, ptr @uc, align 8
// CHECK:   [[TMP7:%.*]] = load volatile <16 x i8>, ptr @uc2, align 8
// CHECK:   [[AND3:%.*]] = and <16 x i8> [[TMP6]], [[TMP7]]
// CHECK:   store volatile <16 x i8> [[AND3]], ptr @uc, align 8
// CHECK:   [[TMP8:%.*]] = load volatile <16 x i8>, ptr @uc, align 8
// CHECK:   [[TMP9:%.*]] = load volatile <16 x i8>, ptr @bc2, align 8
// CHECK:   [[AND4:%.*]] = and <16 x i8> [[TMP8]], [[TMP9]]
// CHECK:   store volatile <16 x i8> [[AND4]], ptr @uc, align 8
// CHECK:   [[TMP10:%.*]] = load volatile <16 x i8>, ptr @bc, align 8
// CHECK:   [[TMP11:%.*]] = load volatile <16 x i8>, ptr @uc2, align 8
// CHECK:   [[AND5:%.*]] = and <16 x i8> [[TMP10]], [[TMP11]]
// CHECK:   store volatile <16 x i8> [[AND5]], ptr @uc, align 8
// CHECK:   [[TMP12:%.*]] = load volatile <16 x i8>, ptr @bc, align 8
// CHECK:   [[TMP13:%.*]] = load volatile <16 x i8>, ptr @bc2, align 8
// CHECK:   [[AND6:%.*]] = and <16 x i8> [[TMP12]], [[TMP13]]
// CHECK:   store volatile <16 x i8> [[AND6]], ptr @bc, align 8
// CHECK:   [[TMP14:%.*]] = load volatile <8 x i16>, ptr @ss, align 8
// CHECK:   [[TMP15:%.*]] = load volatile <8 x i16>, ptr @ss2, align 8
// CHECK:   [[AND7:%.*]] = and <8 x i16> [[TMP14]], [[TMP15]]
// CHECK:   store volatile <8 x i16> [[AND7]], ptr @ss, align 8
// CHECK:   [[TMP16:%.*]] = load volatile <8 x i16>, ptr @ss, align 8
// CHECK:   [[TMP17:%.*]] = load volatile <8 x i16>, ptr @bs2, align 8
// CHECK:   [[AND8:%.*]] = and <8 x i16> [[TMP16]], [[TMP17]]
// CHECK:   store volatile <8 x i16> [[AND8]], ptr @ss, align 8
// CHECK:   [[TMP18:%.*]] = load volatile <8 x i16>, ptr @bs, align 8
// CHECK:   [[TMP19:%.*]] = load volatile <8 x i16>, ptr @ss2, align 8
// CHECK:   [[AND9:%.*]] = and <8 x i16> [[TMP18]], [[TMP19]]
// CHECK:   store volatile <8 x i16> [[AND9]], ptr @ss, align 8
// CHECK:   [[TMP20:%.*]] = load volatile <8 x i16>, ptr @us, align 8
// CHECK:   [[TMP21:%.*]] = load volatile <8 x i16>, ptr @us2, align 8
// CHECK:   [[AND10:%.*]] = and <8 x i16> [[TMP20]], [[TMP21]]
// CHECK:   store volatile <8 x i16> [[AND10]], ptr @us, align 8
// CHECK:   [[TMP22:%.*]] = load volatile <8 x i16>, ptr @us, align 8
// CHECK:   [[TMP23:%.*]] = load volatile <8 x i16>, ptr @bs2, align 8
// CHECK:   [[AND11:%.*]] = and <8 x i16> [[TMP22]], [[TMP23]]
// CHECK:   store volatile <8 x i16> [[AND11]], ptr @us, align 8
// CHECK:   [[TMP24:%.*]] = load volatile <8 x i16>, ptr @bs, align 8
// CHECK:   [[TMP25:%.*]] = load volatile <8 x i16>, ptr @us2, align 8
// CHECK:   [[AND12:%.*]] = and <8 x i16> [[TMP24]], [[TMP25]]
// CHECK:   store volatile <8 x i16> [[AND12]], ptr @us, align 8
// CHECK:   [[TMP26:%.*]] = load volatile <8 x i16>, ptr @bs, align 8
// CHECK:   [[TMP27:%.*]] = load volatile <8 x i16>, ptr @bs2, align 8
// CHECK:   [[AND13:%.*]] = and <8 x i16> [[TMP26]], [[TMP27]]
// CHECK:   store volatile <8 x i16> [[AND13]], ptr @bs, align 8
// CHECK:   [[TMP28:%.*]] = load volatile <4 x i32>, ptr @si, align 8
// CHECK:   [[TMP29:%.*]] = load volatile <4 x i32>, ptr @si2, align 8
// CHECK:   [[AND14:%.*]] = and <4 x i32> [[TMP28]], [[TMP29]]
// CHECK:   store volatile <4 x i32> [[AND14]], ptr @si, align 8
// CHECK:   [[TMP30:%.*]] = load volatile <4 x i32>, ptr @si, align 8
// CHECK:   [[TMP31:%.*]] = load volatile <4 x i32>, ptr @bi2, align 8
// CHECK:   [[AND15:%.*]] = and <4 x i32> [[TMP30]], [[TMP31]]
// CHECK:   store volatile <4 x i32> [[AND15]], ptr @si, align 8
// CHECK:   [[TMP32:%.*]] = load volatile <4 x i32>, ptr @bi, align 8
// CHECK:   [[TMP33:%.*]] = load volatile <4 x i32>, ptr @si2, align 8
// CHECK:   [[AND16:%.*]] = and <4 x i32> [[TMP32]], [[TMP33]]
// CHECK:   store volatile <4 x i32> [[AND16]], ptr @si, align 8
// CHECK:   [[TMP34:%.*]] = load volatile <4 x i32>, ptr @ui, align 8
// CHECK:   [[TMP35:%.*]] = load volatile <4 x i32>, ptr @ui2, align 8
// CHECK:   [[AND17:%.*]] = and <4 x i32> [[TMP34]], [[TMP35]]
// CHECK:   store volatile <4 x i32> [[AND17]], ptr @ui, align 8
// CHECK:   [[TMP36:%.*]] = load volatile <4 x i32>, ptr @ui, align 8
// CHECK:   [[TMP37:%.*]] = load volatile <4 x i32>, ptr @bi2, align 8
// CHECK:   [[AND18:%.*]] = and <4 x i32> [[TMP36]], [[TMP37]]
// CHECK:   store volatile <4 x i32> [[AND18]], ptr @ui, align 8
// CHECK:   [[TMP38:%.*]] = load volatile <4 x i32>, ptr @bi, align 8
// CHECK:   [[TMP39:%.*]] = load volatile <4 x i32>, ptr @ui2, align 8
// CHECK:   [[AND19:%.*]] = and <4 x i32> [[TMP38]], [[TMP39]]
// CHECK:   store volatile <4 x i32> [[AND19]], ptr @ui, align 8
// CHECK:   [[TMP40:%.*]] = load volatile <4 x i32>, ptr @bi, align 8
// CHECK:   [[TMP41:%.*]] = load volatile <4 x i32>, ptr @bi2, align 8
// CHECK:   [[AND20:%.*]] = and <4 x i32> [[TMP40]], [[TMP41]]
// CHECK:   store volatile <4 x i32> [[AND20]], ptr @bi, align 8
// CHECK:   [[TMP42:%.*]] = load volatile <2 x i64>, ptr @sl, align 8
// CHECK:   [[TMP43:%.*]] = load volatile <2 x i64>, ptr @sl2, align 8
// CHECK:   [[AND21:%.*]] = and <2 x i64> [[TMP42]], [[TMP43]]
// CHECK:   store volatile <2 x i64> [[AND21]], ptr @sl, align 8
// CHECK:   [[TMP44:%.*]] = load volatile <2 x i64>, ptr @sl, align 8
// CHECK:   [[TMP45:%.*]] = load volatile <2 x i64>, ptr @bl2, align 8
// CHECK:   [[AND22:%.*]] = and <2 x i64> [[TMP44]], [[TMP45]]
// CHECK:   store volatile <2 x i64> [[AND22]], ptr @sl, align 8
// CHECK:   [[TMP46:%.*]] = load volatile <2 x i64>, ptr @bl, align 8
// CHECK:   [[TMP47:%.*]] = load volatile <2 x i64>, ptr @sl2, align 8
// CHECK:   [[AND23:%.*]] = and <2 x i64> [[TMP46]], [[TMP47]]
// CHECK:   store volatile <2 x i64> [[AND23]], ptr @sl, align 8
// CHECK:   [[TMP48:%.*]] = load volatile <2 x i64>, ptr @ul, align 8
// CHECK:   [[TMP49:%.*]] = load volatile <2 x i64>, ptr @ul2, align 8
// CHECK:   [[AND24:%.*]] = and <2 x i64> [[TMP48]], [[TMP49]]
// CHECK:   store volatile <2 x i64> [[AND24]], ptr @ul, align 8
// CHECK:   [[TMP50:%.*]] = load volatile <2 x i64>, ptr @ul, align 8
// CHECK:   [[TMP51:%.*]] = load volatile <2 x i64>, ptr @bl2, align 8
// CHECK:   [[AND25:%.*]] = and <2 x i64> [[TMP50]], [[TMP51]]
// CHECK:   store volatile <2 x i64> [[AND25]], ptr @ul, align 8
// CHECK:   [[TMP52:%.*]] = load volatile <2 x i64>, ptr @bl, align 8
// CHECK:   [[TMP53:%.*]] = load volatile <2 x i64>, ptr @ul2, align 8
// CHECK:   [[AND26:%.*]] = and <2 x i64> [[TMP52]], [[TMP53]]
// CHECK:   store volatile <2 x i64> [[AND26]], ptr @ul, align 8
// CHECK:   [[TMP54:%.*]] = load volatile <2 x i64>, ptr @bl, align 8
// CHECK:   [[TMP55:%.*]] = load volatile <2 x i64>, ptr @bl2, align 8
// CHECK:   [[AND27:%.*]] = and <2 x i64> [[TMP54]], [[TMP55]]
// CHECK:   store volatile <2 x i64> [[AND27]], ptr @bl, align 8
// CHECK:   ret void
void test_and(void) {

  sc = sc & sc2;
  sc = sc & bc2;
  sc = bc & sc2;
  uc = uc & uc2;
  uc = uc & bc2;
  uc = bc & uc2;
  bc = bc & bc2;

  ss = ss & ss2;
  ss = ss & bs2;
  ss = bs & ss2;
  us = us & us2;
  us = us & bs2;
  us = bs & us2;
  bs = bs & bs2;

  si = si & si2;
  si = si & bi2;
  si = bi & si2;
  ui = ui & ui2;
  ui = ui & bi2;
  ui = bi & ui2;
  bi = bi & bi2;

  sl = sl & sl2;
  sl = sl & bl2;
  sl = bl & sl2;
  ul = ul & ul2;
  ul = ul & bl2;
  ul = bl & ul2;
  bl = bl & bl2;
}

// CHECK-LABEL: define{{.*}} void @test_and_assign() #0 {
// CHECK:   [[TMP0:%.*]] = load volatile <16 x i8>, ptr @sc2, align 8
// CHECK:   [[TMP1:%.*]] = load volatile <16 x i8>, ptr @sc, align 8
// CHECK:   [[AND:%.*]] = and <16 x i8> [[TMP1]], [[TMP0]]
// CHECK:   store volatile <16 x i8> [[AND]], ptr @sc, align 8
// CHECK:   [[TMP2:%.*]] = load volatile <16 x i8>, ptr @bc2, align 8
// CHECK:   [[TMP3:%.*]] = load volatile <16 x i8>, ptr @sc, align 8
// CHECK:   [[AND1:%.*]] = and <16 x i8> [[TMP3]], [[TMP2]]
// CHECK:   store volatile <16 x i8> [[AND1]], ptr @sc, align 8
// CHECK:   [[TMP4:%.*]] = load volatile <16 x i8>, ptr @uc2, align 8
// CHECK:   [[TMP5:%.*]] = load volatile <16 x i8>, ptr @uc, align 8
// CHECK:   [[AND2:%.*]] = and <16 x i8> [[TMP5]], [[TMP4]]
// CHECK:   store volatile <16 x i8> [[AND2]], ptr @uc, align 8
// CHECK:   [[TMP6:%.*]] = load volatile <16 x i8>, ptr @bc2, align 8
// CHECK:   [[TMP7:%.*]] = load volatile <16 x i8>, ptr @uc, align 8
// CHECK:   [[AND3:%.*]] = and <16 x i8> [[TMP7]], [[TMP6]]
// CHECK:   store volatile <16 x i8> [[AND3]], ptr @uc, align 8
// CHECK:   [[TMP8:%.*]] = load volatile <16 x i8>, ptr @bc2, align 8
// CHECK:   [[TMP9:%.*]] = load volatile <16 x i8>, ptr @bc, align 8
// CHECK:   [[AND4:%.*]] = and <16 x i8> [[TMP9]], [[TMP8]]
// CHECK:   store volatile <16 x i8> [[AND4]], ptr @bc, align 8
// CHECK:   [[TMP10:%.*]] = load volatile <8 x i16>, ptr @ss2, align 8
// CHECK:   [[TMP11:%.*]] = load volatile <8 x i16>, ptr @ss, align 8
// CHECK:   [[AND5:%.*]] = and <8 x i16> [[TMP11]], [[TMP10]]
// CHECK:   store volatile <8 x i16> [[AND5]], ptr @ss, align 8
// CHECK:   [[TMP12:%.*]] = load volatile <8 x i16>, ptr @bs2, align 8
// CHECK:   [[TMP13:%.*]] = load volatile <8 x i16>, ptr @ss, align 8
// CHECK:   [[AND6:%.*]] = and <8 x i16> [[TMP13]], [[TMP12]]
// CHECK:   store volatile <8 x i16> [[AND6]], ptr @ss, align 8
// CHECK:   [[TMP14:%.*]] = load volatile <8 x i16>, ptr @us2, align 8
// CHECK:   [[TMP15:%.*]] = load volatile <8 x i16>, ptr @us, align 8
// CHECK:   [[AND7:%.*]] = and <8 x i16> [[TMP15]], [[TMP14]]
// CHECK:   store volatile <8 x i16> [[AND7]], ptr @us, align 8
// CHECK:   [[TMP16:%.*]] = load volatile <8 x i16>, ptr @bs2, align 8
// CHECK:   [[TMP17:%.*]] = load volatile <8 x i16>, ptr @us, align 8
// CHECK:   [[AND8:%.*]] = and <8 x i16> [[TMP17]], [[TMP16]]
// CHECK:   store volatile <8 x i16> [[AND8]], ptr @us, align 8
// CHECK:   [[TMP18:%.*]] = load volatile <8 x i16>, ptr @bs2, align 8
// CHECK:   [[TMP19:%.*]] = load volatile <8 x i16>, ptr @bs, align 8
// CHECK:   [[AND9:%.*]] = and <8 x i16> [[TMP19]], [[TMP18]]
// CHECK:   store volatile <8 x i16> [[AND9]], ptr @bs, align 8
// CHECK:   [[TMP20:%.*]] = load volatile <4 x i32>, ptr @si2, align 8
// CHECK:   [[TMP21:%.*]] = load volatile <4 x i32>, ptr @si, align 8
// CHECK:   [[AND10:%.*]] = and <4 x i32> [[TMP21]], [[TMP20]]
// CHECK:   store volatile <4 x i32> [[AND10]], ptr @si, align 8
// CHECK:   [[TMP22:%.*]] = load volatile <4 x i32>, ptr @bi2, align 8
// CHECK:   [[TMP23:%.*]] = load volatile <4 x i32>, ptr @si, align 8
// CHECK:   [[AND11:%.*]] = and <4 x i32> [[TMP23]], [[TMP22]]
// CHECK:   store volatile <4 x i32> [[AND11]], ptr @si, align 8
// CHECK:   [[TMP24:%.*]] = load volatile <4 x i32>, ptr @ui2, align 8
// CHECK:   [[TMP25:%.*]] = load volatile <4 x i32>, ptr @ui, align 8
// CHECK:   [[AND12:%.*]] = and <4 x i32> [[TMP25]], [[TMP24]]
// CHECK:   store volatile <4 x i32> [[AND12]], ptr @ui, align 8
// CHECK:   [[TMP26:%.*]] = load volatile <4 x i32>, ptr @bi2, align 8
// CHECK:   [[TMP27:%.*]] = load volatile <4 x i32>, ptr @ui, align 8
// CHECK:   [[AND13:%.*]] = and <4 x i32> [[TMP27]], [[TMP26]]
// CHECK:   store volatile <4 x i32> [[AND13]], ptr @ui, align 8
// CHECK:   [[TMP28:%.*]] = load volatile <4 x i32>, ptr @bi2, align 8
// CHECK:   [[TMP29:%.*]] = load volatile <4 x i32>, ptr @bi, align 8
// CHECK:   [[AND14:%.*]] = and <4 x i32> [[TMP29]], [[TMP28]]
// CHECK:   store volatile <4 x i32> [[AND14]], ptr @bi, align 8
// CHECK:   [[TMP30:%.*]] = load volatile <2 x i64>, ptr @sl2, align 8
// CHECK:   [[TMP31:%.*]] = load volatile <2 x i64>, ptr @sl, align 8
// CHECK:   [[AND15:%.*]] = and <2 x i64> [[TMP31]], [[TMP30]]
// CHECK:   store volatile <2 x i64> [[AND15]], ptr @sl, align 8
// CHECK:   [[TMP32:%.*]] = load volatile <2 x i64>, ptr @bl2, align 8
// CHECK:   [[TMP33:%.*]] = load volatile <2 x i64>, ptr @sl, align 8
// CHECK:   [[AND16:%.*]] = and <2 x i64> [[TMP33]], [[TMP32]]
// CHECK:   store volatile <2 x i64> [[AND16]], ptr @sl, align 8
// CHECK:   [[TMP34:%.*]] = load volatile <2 x i64>, ptr @ul2, align 8
// CHECK:   [[TMP35:%.*]] = load volatile <2 x i64>, ptr @ul, align 8
// CHECK:   [[AND17:%.*]] = and <2 x i64> [[TMP35]], [[TMP34]]
// CHECK:   store volatile <2 x i64> [[AND17]], ptr @ul, align 8
// CHECK:   [[TMP36:%.*]] = load volatile <2 x i64>, ptr @bl2, align 8
// CHECK:   [[TMP37:%.*]] = load volatile <2 x i64>, ptr @ul, align 8
// CHECK:   [[AND18:%.*]] = and <2 x i64> [[TMP37]], [[TMP36]]
// CHECK:   store volatile <2 x i64> [[AND18]], ptr @ul, align 8
// CHECK:   [[TMP38:%.*]] = load volatile <2 x i64>, ptr @bl2, align 8
// CHECK:   [[TMP39:%.*]] = load volatile <2 x i64>, ptr @bl, align 8
// CHECK:   [[AND19:%.*]] = and <2 x i64> [[TMP39]], [[TMP38]]
// CHECK:   store volatile <2 x i64> [[AND19]], ptr @bl, align 8
// CHECK:   ret void
void test_and_assign(void) {

  sc &= sc2;
  sc &= bc2;
  uc &= uc2;
  uc &= bc2;
  bc &= bc2;

  ss &= ss2;
  ss &= bs2;
  us &= us2;
  us &= bs2;
  bs &= bs2;

  si &= si2;
  si &= bi2;
  ui &= ui2;
  ui &= bi2;
  bi &= bi2;

  sl &= sl2;
  sl &= bl2;
  ul &= ul2;
  ul &= bl2;
  bl &= bl2;
}

// CHECK-LABEL: define{{.*}} void @test_or() #0 {
// CHECK:   [[TMP0:%.*]] = load volatile <16 x i8>, ptr @sc, align 8
// CHECK:   [[TMP1:%.*]] = load volatile <16 x i8>, ptr @sc2, align 8
// CHECK:   [[OR:%.*]] = or <16 x i8> [[TMP0]], [[TMP1]]
// CHECK:   store volatile <16 x i8> [[OR]], ptr @sc, align 8
// CHECK:   [[TMP2:%.*]] = load volatile <16 x i8>, ptr @sc, align 8
// CHECK:   [[TMP3:%.*]] = load volatile <16 x i8>, ptr @bc2, align 8
// CHECK:   [[OR1:%.*]] = or <16 x i8> [[TMP2]], [[TMP3]]
// CHECK:   store volatile <16 x i8> [[OR1]], ptr @sc, align 8
// CHECK:   [[TMP4:%.*]] = load volatile <16 x i8>, ptr @bc, align 8
// CHECK:   [[TMP5:%.*]] = load volatile <16 x i8>, ptr @sc2, align 8
// CHECK:   [[OR2:%.*]] = or <16 x i8> [[TMP4]], [[TMP5]]
// CHECK:   store volatile <16 x i8> [[OR2]], ptr @sc, align 8
// CHECK:   [[TMP6:%.*]] = load volatile <16 x i8>, ptr @uc, align 8
// CHECK:   [[TMP7:%.*]] = load volatile <16 x i8>, ptr @uc2, align 8
// CHECK:   [[OR3:%.*]] = or <16 x i8> [[TMP6]], [[TMP7]]
// CHECK:   store volatile <16 x i8> [[OR3]], ptr @uc, align 8
// CHECK:   [[TMP8:%.*]] = load volatile <16 x i8>, ptr @uc, align 8
// CHECK:   [[TMP9:%.*]] = load volatile <16 x i8>, ptr @bc2, align 8
// CHECK:   [[OR4:%.*]] = or <16 x i8> [[TMP8]], [[TMP9]]
// CHECK:   store volatile <16 x i8> [[OR4]], ptr @uc, align 8
// CHECK:   [[TMP10:%.*]] = load volatile <16 x i8>, ptr @bc, align 8
// CHECK:   [[TMP11:%.*]] = load volatile <16 x i8>, ptr @uc2, align 8
// CHECK:   [[OR5:%.*]] = or <16 x i8> [[TMP10]], [[TMP11]]
// CHECK:   store volatile <16 x i8> [[OR5]], ptr @uc, align 8
// CHECK:   [[TMP12:%.*]] = load volatile <16 x i8>, ptr @bc, align 8
// CHECK:   [[TMP13:%.*]] = load volatile <16 x i8>, ptr @bc2, align 8
// CHECK:   [[OR6:%.*]] = or <16 x i8> [[TMP12]], [[TMP13]]
// CHECK:   store volatile <16 x i8> [[OR6]], ptr @bc, align 8
// CHECK:   [[TMP14:%.*]] = load volatile <8 x i16>, ptr @ss, align 8
// CHECK:   [[TMP15:%.*]] = load volatile <8 x i16>, ptr @ss2, align 8
// CHECK:   [[OR7:%.*]] = or <8 x i16> [[TMP14]], [[TMP15]]
// CHECK:   store volatile <8 x i16> [[OR7]], ptr @ss, align 8
// CHECK:   [[TMP16:%.*]] = load volatile <8 x i16>, ptr @ss, align 8
// CHECK:   [[TMP17:%.*]] = load volatile <8 x i16>, ptr @bs2, align 8
// CHECK:   [[OR8:%.*]] = or <8 x i16> [[TMP16]], [[TMP17]]
// CHECK:   store volatile <8 x i16> [[OR8]], ptr @ss, align 8
// CHECK:   [[TMP18:%.*]] = load volatile <8 x i16>, ptr @bs, align 8
// CHECK:   [[TMP19:%.*]] = load volatile <8 x i16>, ptr @ss2, align 8
// CHECK:   [[OR9:%.*]] = or <8 x i16> [[TMP18]], [[TMP19]]
// CHECK:   store volatile <8 x i16> [[OR9]], ptr @ss, align 8
// CHECK:   [[TMP20:%.*]] = load volatile <8 x i16>, ptr @us, align 8
// CHECK:   [[TMP21:%.*]] = load volatile <8 x i16>, ptr @us2, align 8
// CHECK:   [[OR10:%.*]] = or <8 x i16> [[TMP20]], [[TMP21]]
// CHECK:   store volatile <8 x i16> [[OR10]], ptr @us, align 8
// CHECK:   [[TMP22:%.*]] = load volatile <8 x i16>, ptr @us, align 8
// CHECK:   [[TMP23:%.*]] = load volatile <8 x i16>, ptr @bs2, align 8
// CHECK:   [[OR11:%.*]] = or <8 x i16> [[TMP22]], [[TMP23]]
// CHECK:   store volatile <8 x i16> [[OR11]], ptr @us, align 8
// CHECK:   [[TMP24:%.*]] = load volatile <8 x i16>, ptr @bs, align 8
// CHECK:   [[TMP25:%.*]] = load volatile <8 x i16>, ptr @us2, align 8
// CHECK:   [[OR12:%.*]] = or <8 x i16> [[TMP24]], [[TMP25]]
// CHECK:   store volatile <8 x i16> [[OR12]], ptr @us, align 8
// CHECK:   [[TMP26:%.*]] = load volatile <8 x i16>, ptr @bs, align 8
// CHECK:   [[TMP27:%.*]] = load volatile <8 x i16>, ptr @bs2, align 8
// CHECK:   [[OR13:%.*]] = or <8 x i16> [[TMP26]], [[TMP27]]
// CHECK:   store volatile <8 x i16> [[OR13]], ptr @bs, align 8
// CHECK:   [[TMP28:%.*]] = load volatile <4 x i32>, ptr @si, align 8
// CHECK:   [[TMP29:%.*]] = load volatile <4 x i32>, ptr @si2, align 8
// CHECK:   [[OR14:%.*]] = or <4 x i32> [[TMP28]], [[TMP29]]
// CHECK:   store volatile <4 x i32> [[OR14]], ptr @si, align 8
// CHECK:   [[TMP30:%.*]] = load volatile <4 x i32>, ptr @si, align 8
// CHECK:   [[TMP31:%.*]] = load volatile <4 x i32>, ptr @bi2, align 8
// CHECK:   [[OR15:%.*]] = or <4 x i32> [[TMP30]], [[TMP31]]
// CHECK:   store volatile <4 x i32> [[OR15]], ptr @si, align 8
// CHECK:   [[TMP32:%.*]] = load volatile <4 x i32>, ptr @bi, align 8
// CHECK:   [[TMP33:%.*]] = load volatile <4 x i32>, ptr @si2, align 8
// CHECK:   [[OR16:%.*]] = or <4 x i32> [[TMP32]], [[TMP33]]
// CHECK:   store volatile <4 x i32> [[OR16]], ptr @si, align 8
// CHECK:   [[TMP34:%.*]] = load volatile <4 x i32>, ptr @ui, align 8
// CHECK:   [[TMP35:%.*]] = load volatile <4 x i32>, ptr @ui2, align 8
// CHECK:   [[OR17:%.*]] = or <4 x i32> [[TMP34]], [[TMP35]]
// CHECK:   store volatile <4 x i32> [[OR17]], ptr @ui, align 8
// CHECK:   [[TMP36:%.*]] = load volatile <4 x i32>, ptr @ui, align 8
// CHECK:   [[TMP37:%.*]] = load volatile <4 x i32>, ptr @bi2, align 8
// CHECK:   [[OR18:%.*]] = or <4 x i32> [[TMP36]], [[TMP37]]
// CHECK:   store volatile <4 x i32> [[OR18]], ptr @ui, align 8
// CHECK:   [[TMP38:%.*]] = load volatile <4 x i32>, ptr @bi, align 8
// CHECK:   [[TMP39:%.*]] = load volatile <4 x i32>, ptr @ui2, align 8
// CHECK:   [[OR19:%.*]] = or <4 x i32> [[TMP38]], [[TMP39]]
// CHECK:   store volatile <4 x i32> [[OR19]], ptr @ui, align 8
// CHECK:   [[TMP40:%.*]] = load volatile <4 x i32>, ptr @bi, align 8
// CHECK:   [[TMP41:%.*]] = load volatile <4 x i32>, ptr @bi2, align 8
// CHECK:   [[OR20:%.*]] = or <4 x i32> [[TMP40]], [[TMP41]]
// CHECK:   store volatile <4 x i32> [[OR20]], ptr @bi, align 8
// CHECK:   [[TMP42:%.*]] = load volatile <2 x i64>, ptr @sl, align 8
// CHECK:   [[TMP43:%.*]] = load volatile <2 x i64>, ptr @sl2, align 8
// CHECK:   [[OR21:%.*]] = or <2 x i64> [[TMP42]], [[TMP43]]
// CHECK:   store volatile <2 x i64> [[OR21]], ptr @sl, align 8
// CHECK:   [[TMP44:%.*]] = load volatile <2 x i64>, ptr @sl, align 8
// CHECK:   [[TMP45:%.*]] = load volatile <2 x i64>, ptr @bl2, align 8
// CHECK:   [[OR22:%.*]] = or <2 x i64> [[TMP44]], [[TMP45]]
// CHECK:   store volatile <2 x i64> [[OR22]], ptr @sl, align 8
// CHECK:   [[TMP46:%.*]] = load volatile <2 x i64>, ptr @bl, align 8
// CHECK:   [[TMP47:%.*]] = load volatile <2 x i64>, ptr @sl2, align 8
// CHECK:   [[OR23:%.*]] = or <2 x i64> [[TMP46]], [[TMP47]]
// CHECK:   store volatile <2 x i64> [[OR23]], ptr @sl, align 8
// CHECK:   [[TMP48:%.*]] = load volatile <2 x i64>, ptr @ul, align 8
// CHECK:   [[TMP49:%.*]] = load volatile <2 x i64>, ptr @ul2, align 8
// CHECK:   [[OR24:%.*]] = or <2 x i64> [[TMP48]], [[TMP49]]
// CHECK:   store volatile <2 x i64> [[OR24]], ptr @ul, align 8
// CHECK:   [[TMP50:%.*]] = load volatile <2 x i64>, ptr @ul, align 8
// CHECK:   [[TMP51:%.*]] = load volatile <2 x i64>, ptr @bl2, align 8
// CHECK:   [[OR25:%.*]] = or <2 x i64> [[TMP50]], [[TMP51]]
// CHECK:   store volatile <2 x i64> [[OR25]], ptr @ul, align 8
// CHECK:   [[TMP52:%.*]] = load volatile <2 x i64>, ptr @bl, align 8
// CHECK:   [[TMP53:%.*]] = load volatile <2 x i64>, ptr @ul2, align 8
// CHECK:   [[OR26:%.*]] = or <2 x i64> [[TMP52]], [[TMP53]]
// CHECK:   store volatile <2 x i64> [[OR26]], ptr @ul, align 8
// CHECK:   [[TMP54:%.*]] = load volatile <2 x i64>, ptr @bl, align 8
// CHECK:   [[TMP55:%.*]] = load volatile <2 x i64>, ptr @bl2, align 8
// CHECK:   [[OR27:%.*]] = or <2 x i64> [[TMP54]], [[TMP55]]
// CHECK:   store volatile <2 x i64> [[OR27]], ptr @bl, align 8
// CHECK:   ret void
void test_or(void) {

  sc = sc | sc2;
  sc = sc | bc2;
  sc = bc | sc2;
  uc = uc | uc2;
  uc = uc | bc2;
  uc = bc | uc2;
  bc = bc | bc2;

  ss = ss | ss2;
  ss = ss | bs2;
  ss = bs | ss2;
  us = us | us2;
  us = us | bs2;
  us = bs | us2;
  bs = bs | bs2;

  si = si | si2;
  si = si | bi2;
  si = bi | si2;
  ui = ui | ui2;
  ui = ui | bi2;
  ui = bi | ui2;
  bi = bi | bi2;

  sl = sl | sl2;
  sl = sl | bl2;
  sl = bl | sl2;
  ul = ul | ul2;
  ul = ul | bl2;
  ul = bl | ul2;
  bl = bl | bl2;
}

// CHECK-LABEL: define{{.*}} void @test_or_assign() #0 {
// CHECK:   [[TMP0:%.*]] = load volatile <16 x i8>, ptr @sc2, align 8
// CHECK:   [[TMP1:%.*]] = load volatile <16 x i8>, ptr @sc, align 8
// CHECK:   [[OR:%.*]] = or <16 x i8> [[TMP1]], [[TMP0]]
// CHECK:   store volatile <16 x i8> [[OR]], ptr @sc, align 8
// CHECK:   [[TMP2:%.*]] = load volatile <16 x i8>, ptr @bc2, align 8
// CHECK:   [[TMP3:%.*]] = load volatile <16 x i8>, ptr @sc, align 8
// CHECK:   [[OR1:%.*]] = or <16 x i8> [[TMP3]], [[TMP2]]
// CHECK:   store volatile <16 x i8> [[OR1]], ptr @sc, align 8
// CHECK:   [[TMP4:%.*]] = load volatile <16 x i8>, ptr @uc2, align 8
// CHECK:   [[TMP5:%.*]] = load volatile <16 x i8>, ptr @uc, align 8
// CHECK:   [[OR2:%.*]] = or <16 x i8> [[TMP5]], [[TMP4]]
// CHECK:   store volatile <16 x i8> [[OR2]], ptr @uc, align 8
// CHECK:   [[TMP6:%.*]] = load volatile <16 x i8>, ptr @bc2, align 8
// CHECK:   [[TMP7:%.*]] = load volatile <16 x i8>, ptr @uc, align 8
// CHECK:   [[OR3:%.*]] = or <16 x i8> [[TMP7]], [[TMP6]]
// CHECK:   store volatile <16 x i8> [[OR3]], ptr @uc, align 8
// CHECK:   [[TMP8:%.*]] = load volatile <16 x i8>, ptr @bc2, align 8
// CHECK:   [[TMP9:%.*]] = load volatile <16 x i8>, ptr @bc, align 8
// CHECK:   [[OR4:%.*]] = or <16 x i8> [[TMP9]], [[TMP8]]
// CHECK:   store volatile <16 x i8> [[OR4]], ptr @bc, align 8
// CHECK:   [[TMP10:%.*]] = load volatile <8 x i16>, ptr @ss2, align 8
// CHECK:   [[TMP11:%.*]] = load volatile <8 x i16>, ptr @ss, align 8
// CHECK:   [[OR5:%.*]] = or <8 x i16> [[TMP11]], [[TMP10]]
// CHECK:   store volatile <8 x i16> [[OR5]], ptr @ss, align 8
// CHECK:   [[TMP12:%.*]] = load volatile <8 x i16>, ptr @bs2, align 8
// CHECK:   [[TMP13:%.*]] = load volatile <8 x i16>, ptr @ss, align 8
// CHECK:   [[OR6:%.*]] = or <8 x i16> [[TMP13]], [[TMP12]]
// CHECK:   store volatile <8 x i16> [[OR6]], ptr @ss, align 8
// CHECK:   [[TMP14:%.*]] = load volatile <8 x i16>, ptr @us2, align 8
// CHECK:   [[TMP15:%.*]] = load volatile <8 x i16>, ptr @us, align 8
// CHECK:   [[OR7:%.*]] = or <8 x i16> [[TMP15]], [[TMP14]]
// CHECK:   store volatile <8 x i16> [[OR7]], ptr @us, align 8
// CHECK:   [[TMP16:%.*]] = load volatile <8 x i16>, ptr @bs2, align 8
// CHECK:   [[TMP17:%.*]] = load volatile <8 x i16>, ptr @us, align 8
// CHECK:   [[OR8:%.*]] = or <8 x i16> [[TMP17]], [[TMP16]]
// CHECK:   store volatile <8 x i16> [[OR8]], ptr @us, align 8
// CHECK:   [[TMP18:%.*]] = load volatile <8 x i16>, ptr @bs2, align 8
// CHECK:   [[TMP19:%.*]] = load volatile <8 x i16>, ptr @bs, align 8
// CHECK:   [[OR9:%.*]] = or <8 x i16> [[TMP19]], [[TMP18]]
// CHECK:   store volatile <8 x i16> [[OR9]], ptr @bs, align 8
// CHECK:   [[TMP20:%.*]] = load volatile <4 x i32>, ptr @si2, align 8
// CHECK:   [[TMP21:%.*]] = load volatile <4 x i32>, ptr @si, align 8
// CHECK:   [[OR10:%.*]] = or <4 x i32> [[TMP21]], [[TMP20]]
// CHECK:   store volatile <4 x i32> [[OR10]], ptr @si, align 8
// CHECK:   [[TMP22:%.*]] = load volatile <4 x i32>, ptr @bi2, align 8
// CHECK:   [[TMP23:%.*]] = load volatile <4 x i32>, ptr @si, align 8
// CHECK:   [[OR11:%.*]] = or <4 x i32> [[TMP23]], [[TMP22]]
// CHECK:   store volatile <4 x i32> [[OR11]], ptr @si, align 8
// CHECK:   [[TMP24:%.*]] = load volatile <4 x i32>, ptr @ui2, align 8
// CHECK:   [[TMP25:%.*]] = load volatile <4 x i32>, ptr @ui, align 8
// CHECK:   [[OR12:%.*]] = or <4 x i32> [[TMP25]], [[TMP24]]
// CHECK:   store volatile <4 x i32> [[OR12]], ptr @ui, align 8
// CHECK:   [[TMP26:%.*]] = load volatile <4 x i32>, ptr @bi2, align 8
// CHECK:   [[TMP27:%.*]] = load volatile <4 x i32>, ptr @ui, align 8
// CHECK:   [[OR13:%.*]] = or <4 x i32> [[TMP27]], [[TMP26]]
// CHECK:   store volatile <4 x i32> [[OR13]], ptr @ui, align 8
// CHECK:   [[TMP28:%.*]] = load volatile <4 x i32>, ptr @bi2, align 8
// CHECK:   [[TMP29:%.*]] = load volatile <4 x i32>, ptr @bi, align 8
// CHECK:   [[OR14:%.*]] = or <4 x i32> [[TMP29]], [[TMP28]]
// CHECK:   store volatile <4 x i32> [[OR14]], ptr @bi, align 8
// CHECK:   [[TMP30:%.*]] = load volatile <2 x i64>, ptr @sl2, align 8
// CHECK:   [[TMP31:%.*]] = load volatile <2 x i64>, ptr @sl, align 8
// CHECK:   [[OR15:%.*]] = or <2 x i64> [[TMP31]], [[TMP30]]
// CHECK:   store volatile <2 x i64> [[OR15]], ptr @sl, align 8
// CHECK:   [[TMP32:%.*]] = load volatile <2 x i64>, ptr @bl2, align 8
// CHECK:   [[TMP33:%.*]] = load volatile <2 x i64>, ptr @sl, align 8
// CHECK:   [[OR16:%.*]] = or <2 x i64> [[TMP33]], [[TMP32]]
// CHECK:   store volatile <2 x i64> [[OR16]], ptr @sl, align 8
// CHECK:   [[TMP34:%.*]] = load volatile <2 x i64>, ptr @ul2, align 8
// CHECK:   [[TMP35:%.*]] = load volatile <2 x i64>, ptr @ul, align 8
// CHECK:   [[OR17:%.*]] = or <2 x i64> [[TMP35]], [[TMP34]]
// CHECK:   store volatile <2 x i64> [[OR17]], ptr @ul, align 8
// CHECK:   [[TMP36:%.*]] = load volatile <2 x i64>, ptr @bl2, align 8
// CHECK:   [[TMP37:%.*]] = load volatile <2 x i64>, ptr @ul, align 8
// CHECK:   [[OR18:%.*]] = or <2 x i64> [[TMP37]], [[TMP36]]
// CHECK:   store volatile <2 x i64> [[OR18]], ptr @ul, align 8
// CHECK:   [[TMP38:%.*]] = load volatile <2 x i64>, ptr @bl2, align 8
// CHECK:   [[TMP39:%.*]] = load volatile <2 x i64>, ptr @bl, align 8
// CHECK:   [[OR19:%.*]] = or <2 x i64> [[TMP39]], [[TMP38]]
// CHECK:   store volatile <2 x i64> [[OR19]], ptr @bl, align 8
// CHECK:   ret void
void test_or_assign(void) {

  sc |= sc2;
  sc |= bc2;
  uc |= uc2;
  uc |= bc2;
  bc |= bc2;

  ss |= ss2;
  ss |= bs2;
  us |= us2;
  us |= bs2;
  bs |= bs2;

  si |= si2;
  si |= bi2;
  ui |= ui2;
  ui |= bi2;
  bi |= bi2;

  sl |= sl2;
  sl |= bl2;
  ul |= ul2;
  ul |= bl2;
  bl |= bl2;
}

// CHECK-LABEL: define{{.*}} void @test_xor() #0 {
// CHECK:   [[TMP0:%.*]] = load volatile <16 x i8>, ptr @sc, align 8
// CHECK:   [[TMP1:%.*]] = load volatile <16 x i8>, ptr @sc2, align 8
// CHECK:   [[XOR:%.*]] = xor <16 x i8> [[TMP0]], [[TMP1]]
// CHECK:   store volatile <16 x i8> [[XOR]], ptr @sc, align 8
// CHECK:   [[TMP2:%.*]] = load volatile <16 x i8>, ptr @sc, align 8
// CHECK:   [[TMP3:%.*]] = load volatile <16 x i8>, ptr @bc2, align 8
// CHECK:   [[XOR1:%.*]] = xor <16 x i8> [[TMP2]], [[TMP3]]
// CHECK:   store volatile <16 x i8> [[XOR1]], ptr @sc, align 8
// CHECK:   [[TMP4:%.*]] = load volatile <16 x i8>, ptr @bc, align 8
// CHECK:   [[TMP5:%.*]] = load volatile <16 x i8>, ptr @sc2, align 8
// CHECK:   [[XOR2:%.*]] = xor <16 x i8> [[TMP4]], [[TMP5]]
// CHECK:   store volatile <16 x i8> [[XOR2]], ptr @sc, align 8
// CHECK:   [[TMP6:%.*]] = load volatile <16 x i8>, ptr @uc, align 8
// CHECK:   [[TMP7:%.*]] = load volatile <16 x i8>, ptr @uc2, align 8
// CHECK:   [[XOR3:%.*]] = xor <16 x i8> [[TMP6]], [[TMP7]]
// CHECK:   store volatile <16 x i8> [[XOR3]], ptr @uc, align 8
// CHECK:   [[TMP8:%.*]] = load volatile <16 x i8>, ptr @uc, align 8
// CHECK:   [[TMP9:%.*]] = load volatile <16 x i8>, ptr @bc2, align 8
// CHECK:   [[XOR4:%.*]] = xor <16 x i8> [[TMP8]], [[TMP9]]
// CHECK:   store volatile <16 x i8> [[XOR4]], ptr @uc, align 8
// CHECK:   [[TMP10:%.*]] = load volatile <16 x i8>, ptr @bc, align 8
// CHECK:   [[TMP11:%.*]] = load volatile <16 x i8>, ptr @uc2, align 8
// CHECK:   [[XOR5:%.*]] = xor <16 x i8> [[TMP10]], [[TMP11]]
// CHECK:   store volatile <16 x i8> [[XOR5]], ptr @uc, align 8
// CHECK:   [[TMP12:%.*]] = load volatile <16 x i8>, ptr @bc, align 8
// CHECK:   [[TMP13:%.*]] = load volatile <16 x i8>, ptr @bc2, align 8
// CHECK:   [[XOR6:%.*]] = xor <16 x i8> [[TMP12]], [[TMP13]]
// CHECK:   store volatile <16 x i8> [[XOR6]], ptr @bc, align 8
// CHECK:   [[TMP14:%.*]] = load volatile <8 x i16>, ptr @ss, align 8
// CHECK:   [[TMP15:%.*]] = load volatile <8 x i16>, ptr @ss2, align 8
// CHECK:   [[XOR7:%.*]] = xor <8 x i16> [[TMP14]], [[TMP15]]
// CHECK:   store volatile <8 x i16> [[XOR7]], ptr @ss, align 8
// CHECK:   [[TMP16:%.*]] = load volatile <8 x i16>, ptr @ss, align 8
// CHECK:   [[TMP17:%.*]] = load volatile <8 x i16>, ptr @bs2, align 8
// CHECK:   [[XOR8:%.*]] = xor <8 x i16> [[TMP16]], [[TMP17]]
// CHECK:   store volatile <8 x i16> [[XOR8]], ptr @ss, align 8
// CHECK:   [[TMP18:%.*]] = load volatile <8 x i16>, ptr @bs, align 8
// CHECK:   [[TMP19:%.*]] = load volatile <8 x i16>, ptr @ss2, align 8
// CHECK:   [[XOR9:%.*]] = xor <8 x i16> [[TMP18]], [[TMP19]]
// CHECK:   store volatile <8 x i16> [[XOR9]], ptr @ss, align 8
// CHECK:   [[TMP20:%.*]] = load volatile <8 x i16>, ptr @us, align 8
// CHECK:   [[TMP21:%.*]] = load volatile <8 x i16>, ptr @us2, align 8
// CHECK:   [[XOR10:%.*]] = xor <8 x i16> [[TMP20]], [[TMP21]]
// CHECK:   store volatile <8 x i16> [[XOR10]], ptr @us, align 8
// CHECK:   [[TMP22:%.*]] = load volatile <8 x i16>, ptr @us, align 8
// CHECK:   [[TMP23:%.*]] = load volatile <8 x i16>, ptr @bs2, align 8
// CHECK:   [[XOR11:%.*]] = xor <8 x i16> [[TMP22]], [[TMP23]]
// CHECK:   store volatile <8 x i16> [[XOR11]], ptr @us, align 8
// CHECK:   [[TMP24:%.*]] = load volatile <8 x i16>, ptr @bs, align 8
// CHECK:   [[TMP25:%.*]] = load volatile <8 x i16>, ptr @us2, align 8
// CHECK:   [[XOR12:%.*]] = xor <8 x i16> [[TMP24]], [[TMP25]]
// CHECK:   store volatile <8 x i16> [[XOR12]], ptr @us, align 8
// CHECK:   [[TMP26:%.*]] = load volatile <8 x i16>, ptr @bs, align 8
// CHECK:   [[TMP27:%.*]] = load volatile <8 x i16>, ptr @bs2, align 8
// CHECK:   [[XOR13:%.*]] = xor <8 x i16> [[TMP26]], [[TMP27]]
// CHECK:   store volatile <8 x i16> [[XOR13]], ptr @bs, align 8
// CHECK:   [[TMP28:%.*]] = load volatile <4 x i32>, ptr @si, align 8
// CHECK:   [[TMP29:%.*]] = load volatile <4 x i32>, ptr @si2, align 8
// CHECK:   [[XOR14:%.*]] = xor <4 x i32> [[TMP28]], [[TMP29]]
// CHECK:   store volatile <4 x i32> [[XOR14]], ptr @si, align 8
// CHECK:   [[TMP30:%.*]] = load volatile <4 x i32>, ptr @si, align 8
// CHECK:   [[TMP31:%.*]] = load volatile <4 x i32>, ptr @bi2, align 8
// CHECK:   [[XOR15:%.*]] = xor <4 x i32> [[TMP30]], [[TMP31]]
// CHECK:   store volatile <4 x i32> [[XOR15]], ptr @si, align 8
// CHECK:   [[TMP32:%.*]] = load volatile <4 x i32>, ptr @bi, align 8
// CHECK:   [[TMP33:%.*]] = load volatile <4 x i32>, ptr @si2, align 8
// CHECK:   [[XOR16:%.*]] = xor <4 x i32> [[TMP32]], [[TMP33]]
// CHECK:   store volatile <4 x i32> [[XOR16]], ptr @si, align 8
// CHECK:   [[TMP34:%.*]] = load volatile <4 x i32>, ptr @ui, align 8
// CHECK:   [[TMP35:%.*]] = load volatile <4 x i32>, ptr @ui2, align 8
// CHECK:   [[XOR17:%.*]] = xor <4 x i32> [[TMP34]], [[TMP35]]
// CHECK:   store volatile <4 x i32> [[XOR17]], ptr @ui, align 8
// CHECK:   [[TMP36:%.*]] = load volatile <4 x i32>, ptr @ui, align 8
// CHECK:   [[TMP37:%.*]] = load volatile <4 x i32>, ptr @bi2, align 8
// CHECK:   [[XOR18:%.*]] = xor <4 x i32> [[TMP36]], [[TMP37]]
// CHECK:   store volatile <4 x i32> [[XOR18]], ptr @ui, align 8
// CHECK:   [[TMP38:%.*]] = load volatile <4 x i32>, ptr @bi, align 8
// CHECK:   [[TMP39:%.*]] = load volatile <4 x i32>, ptr @ui2, align 8
// CHECK:   [[XOR19:%.*]] = xor <4 x i32> [[TMP38]], [[TMP39]]
// CHECK:   store volatile <4 x i32> [[XOR19]], ptr @ui, align 8
// CHECK:   [[TMP40:%.*]] = load volatile <4 x i32>, ptr @bi, align 8
// CHECK:   [[TMP41:%.*]] = load volatile <4 x i32>, ptr @bi2, align 8
// CHECK:   [[XOR20:%.*]] = xor <4 x i32> [[TMP40]], [[TMP41]]
// CHECK:   store volatile <4 x i32> [[XOR20]], ptr @bi, align 8
// CHECK:   [[TMP42:%.*]] = load volatile <2 x i64>, ptr @sl, align 8
// CHECK:   [[TMP43:%.*]] = load volatile <2 x i64>, ptr @sl2, align 8
// CHECK:   [[XOR21:%.*]] = xor <2 x i64> [[TMP42]], [[TMP43]]
// CHECK:   store volatile <2 x i64> [[XOR21]], ptr @sl, align 8
// CHECK:   [[TMP44:%.*]] = load volatile <2 x i64>, ptr @sl, align 8
// CHECK:   [[TMP45:%.*]] = load volatile <2 x i64>, ptr @bl2, align 8
// CHECK:   [[XOR22:%.*]] = xor <2 x i64> [[TMP44]], [[TMP45]]
// CHECK:   store volatile <2 x i64> [[XOR22]], ptr @sl, align 8
// CHECK:   [[TMP46:%.*]] = load volatile <2 x i64>, ptr @bl, align 8
// CHECK:   [[TMP47:%.*]] = load volatile <2 x i64>, ptr @sl2, align 8
// CHECK:   [[XOR23:%.*]] = xor <2 x i64> [[TMP46]], [[TMP47]]
// CHECK:   store volatile <2 x i64> [[XOR23]], ptr @sl, align 8
// CHECK:   [[TMP48:%.*]] = load volatile <2 x i64>, ptr @ul, align 8
// CHECK:   [[TMP49:%.*]] = load volatile <2 x i64>, ptr @ul2, align 8
// CHECK:   [[XOR24:%.*]] = xor <2 x i64> [[TMP48]], [[TMP49]]
// CHECK:   store volatile <2 x i64> [[XOR24]], ptr @ul, align 8
// CHECK:   [[TMP50:%.*]] = load volatile <2 x i64>, ptr @ul, align 8
// CHECK:   [[TMP51:%.*]] = load volatile <2 x i64>, ptr @bl2, align 8
// CHECK:   [[XOR25:%.*]] = xor <2 x i64> [[TMP50]], [[TMP51]]
// CHECK:   store volatile <2 x i64> [[XOR25]], ptr @ul, align 8
// CHECK:   [[TMP52:%.*]] = load volatile <2 x i64>, ptr @bl, align 8
// CHECK:   [[TMP53:%.*]] = load volatile <2 x i64>, ptr @ul2, align 8
// CHECK:   [[XOR26:%.*]] = xor <2 x i64> [[TMP52]], [[TMP53]]
// CHECK:   store volatile <2 x i64> [[XOR26]], ptr @ul, align 8
// CHECK:   [[TMP54:%.*]] = load volatile <2 x i64>, ptr @bl, align 8
// CHECK:   [[TMP55:%.*]] = load volatile <2 x i64>, ptr @bl2, align 8
// CHECK:   [[XOR27:%.*]] = xor <2 x i64> [[TMP54]], [[TMP55]]
// CHECK:   store volatile <2 x i64> [[XOR27]], ptr @bl, align 8
// CHECK:   ret void
void test_xor(void) {

  sc = sc ^ sc2;
  sc = sc ^ bc2;
  sc = bc ^ sc2;
  uc = uc ^ uc2;
  uc = uc ^ bc2;
  uc = bc ^ uc2;
  bc = bc ^ bc2;

  ss = ss ^ ss2;
  ss = ss ^ bs2;
  ss = bs ^ ss2;
  us = us ^ us2;
  us = us ^ bs2;
  us = bs ^ us2;
  bs = bs ^ bs2;

  si = si ^ si2;
  si = si ^ bi2;
  si = bi ^ si2;
  ui = ui ^ ui2;
  ui = ui ^ bi2;
  ui = bi ^ ui2;
  bi = bi ^ bi2;

  sl = sl ^ sl2;
  sl = sl ^ bl2;
  sl = bl ^ sl2;
  ul = ul ^ ul2;
  ul = ul ^ bl2;
  ul = bl ^ ul2;
  bl = bl ^ bl2;
}

// CHECK-LABEL: define{{.*}} void @test_xor_assign() #0 {
// CHECK:   [[TMP0:%.*]] = load volatile <16 x i8>, ptr @sc2, align 8
// CHECK:   [[TMP1:%.*]] = load volatile <16 x i8>, ptr @sc, align 8
// CHECK:   [[XOR:%.*]] = xor <16 x i8> [[TMP1]], [[TMP0]]
// CHECK:   store volatile <16 x i8> [[XOR]], ptr @sc, align 8
// CHECK:   [[TMP2:%.*]] = load volatile <16 x i8>, ptr @bc2, align 8
// CHECK:   [[TMP3:%.*]] = load volatile <16 x i8>, ptr @sc, align 8
// CHECK:   [[XOR1:%.*]] = xor <16 x i8> [[TMP3]], [[TMP2]]
// CHECK:   store volatile <16 x i8> [[XOR1]], ptr @sc, align 8
// CHECK:   [[TMP4:%.*]] = load volatile <16 x i8>, ptr @uc2, align 8
// CHECK:   [[TMP5:%.*]] = load volatile <16 x i8>, ptr @uc, align 8
// CHECK:   [[XOR2:%.*]] = xor <16 x i8> [[TMP5]], [[TMP4]]
// CHECK:   store volatile <16 x i8> [[XOR2]], ptr @uc, align 8
// CHECK:   [[TMP6:%.*]] = load volatile <16 x i8>, ptr @bc2, align 8
// CHECK:   [[TMP7:%.*]] = load volatile <16 x i8>, ptr @uc, align 8
// CHECK:   [[XOR3:%.*]] = xor <16 x i8> [[TMP7]], [[TMP6]]
// CHECK:   store volatile <16 x i8> [[XOR3]], ptr @uc, align 8
// CHECK:   [[TMP8:%.*]] = load volatile <16 x i8>, ptr @bc2, align 8
// CHECK:   [[TMP9:%.*]] = load volatile <16 x i8>, ptr @bc, align 8
// CHECK:   [[XOR4:%.*]] = xor <16 x i8> [[TMP9]], [[TMP8]]
// CHECK:   store volatile <16 x i8> [[XOR4]], ptr @bc, align 8
// CHECK:   [[TMP10:%.*]] = load volatile <8 x i16>, ptr @ss2, align 8
// CHECK:   [[TMP11:%.*]] = load volatile <8 x i16>, ptr @ss, align 8
// CHECK:   [[XOR5:%.*]] = xor <8 x i16> [[TMP11]], [[TMP10]]
// CHECK:   store volatile <8 x i16> [[XOR5]], ptr @ss, align 8
// CHECK:   [[TMP12:%.*]] = load volatile <8 x i16>, ptr @bs2, align 8
// CHECK:   [[TMP13:%.*]] = load volatile <8 x i16>, ptr @ss, align 8
// CHECK:   [[XOR6:%.*]] = xor <8 x i16> [[TMP13]], [[TMP12]]
// CHECK:   store volatile <8 x i16> [[XOR6]], ptr @ss, align 8
// CHECK:   [[TMP14:%.*]] = load volatile <8 x i16>, ptr @us2, align 8
// CHECK:   [[TMP15:%.*]] = load volatile <8 x i16>, ptr @us, align 8
// CHECK:   [[XOR7:%.*]] = xor <8 x i16> [[TMP15]], [[TMP14]]
// CHECK:   store volatile <8 x i16> [[XOR7]], ptr @us, align 8
// CHECK:   [[TMP16:%.*]] = load volatile <8 x i16>, ptr @bs2, align 8
// CHECK:   [[TMP17:%.*]] = load volatile <8 x i16>, ptr @us, align 8
// CHECK:   [[XOR8:%.*]] = xor <8 x i16> [[TMP17]], [[TMP16]]
// CHECK:   store volatile <8 x i16> [[XOR8]], ptr @us, align 8
// CHECK:   [[TMP18:%.*]] = load volatile <8 x i16>, ptr @bs2, align 8
// CHECK:   [[TMP19:%.*]] = load volatile <8 x i16>, ptr @bs, align 8
// CHECK:   [[XOR9:%.*]] = xor <8 x i16> [[TMP19]], [[TMP18]]
// CHECK:   store volatile <8 x i16> [[XOR9]], ptr @bs, align 8
// CHECK:   [[TMP20:%.*]] = load volatile <4 x i32>, ptr @si2, align 8
// CHECK:   [[TMP21:%.*]] = load volatile <4 x i32>, ptr @si, align 8
// CHECK:   [[XOR10:%.*]] = xor <4 x i32> [[TMP21]], [[TMP20]]
// CHECK:   store volatile <4 x i32> [[XOR10]], ptr @si, align 8
// CHECK:   [[TMP22:%.*]] = load volatile <4 x i32>, ptr @bi2, align 8
// CHECK:   [[TMP23:%.*]] = load volatile <4 x i32>, ptr @si, align 8
// CHECK:   [[XOR11:%.*]] = xor <4 x i32> [[TMP23]], [[TMP22]]
// CHECK:   store volatile <4 x i32> [[XOR11]], ptr @si, align 8
// CHECK:   [[TMP24:%.*]] = load volatile <4 x i32>, ptr @ui2, align 8
// CHECK:   [[TMP25:%.*]] = load volatile <4 x i32>, ptr @ui, align 8
// CHECK:   [[XOR12:%.*]] = xor <4 x i32> [[TMP25]], [[TMP24]]
// CHECK:   store volatile <4 x i32> [[XOR12]], ptr @ui, align 8
// CHECK:   [[TMP26:%.*]] = load volatile <4 x i32>, ptr @bi2, align 8
// CHECK:   [[TMP27:%.*]] = load volatile <4 x i32>, ptr @ui, align 8
// CHECK:   [[XOR13:%.*]] = xor <4 x i32> [[TMP27]], [[TMP26]]
// CHECK:   store volatile <4 x i32> [[XOR13]], ptr @ui, align 8
// CHECK:   [[TMP28:%.*]] = load volatile <4 x i32>, ptr @bi2, align 8
// CHECK:   [[TMP29:%.*]] = load volatile <4 x i32>, ptr @bi, align 8
// CHECK:   [[XOR14:%.*]] = xor <4 x i32> [[TMP29]], [[TMP28]]
// CHECK:   store volatile <4 x i32> [[XOR14]], ptr @bi, align 8
// CHECK:   [[TMP30:%.*]] = load volatile <2 x i64>, ptr @sl2, align 8
// CHECK:   [[TMP31:%.*]] = load volatile <2 x i64>, ptr @sl, align 8
// CHECK:   [[XOR15:%.*]] = xor <2 x i64> [[TMP31]], [[TMP30]]
// CHECK:   store volatile <2 x i64> [[XOR15]], ptr @sl, align 8
// CHECK:   [[TMP32:%.*]] = load volatile <2 x i64>, ptr @bl2, align 8
// CHECK:   [[TMP33:%.*]] = load volatile <2 x i64>, ptr @sl, align 8
// CHECK:   [[XOR16:%.*]] = xor <2 x i64> [[TMP33]], [[TMP32]]
// CHECK:   store volatile <2 x i64> [[XOR16]], ptr @sl, align 8
// CHECK:   [[TMP34:%.*]] = load volatile <2 x i64>, ptr @ul2, align 8
// CHECK:   [[TMP35:%.*]] = load volatile <2 x i64>, ptr @ul, align 8
// CHECK:   [[XOR17:%.*]] = xor <2 x i64> [[TMP35]], [[TMP34]]
// CHECK:   store volatile <2 x i64> [[XOR17]], ptr @ul, align 8
// CHECK:   [[TMP36:%.*]] = load volatile <2 x i64>, ptr @bl2, align 8
// CHECK:   [[TMP37:%.*]] = load volatile <2 x i64>, ptr @ul, align 8
// CHECK:   [[XOR18:%.*]] = xor <2 x i64> [[TMP37]], [[TMP36]]
// CHECK:   store volatile <2 x i64> [[XOR18]], ptr @ul, align 8
// CHECK:   [[TMP38:%.*]] = load volatile <2 x i64>, ptr @bl2, align 8
// CHECK:   [[TMP39:%.*]] = load volatile <2 x i64>, ptr @bl, align 8
// CHECK:   [[XOR19:%.*]] = xor <2 x i64> [[TMP39]], [[TMP38]]
// CHECK:   store volatile <2 x i64> [[XOR19]], ptr @bl, align 8
// CHECK:   ret void
void test_xor_assign(void) {

  sc ^= sc2;
  sc ^= bc2;
  uc ^= uc2;
  uc ^= bc2;
  bc ^= bc2;

  ss ^= ss2;
  ss ^= bs2;
  us ^= us2;
  us ^= bs2;
  bs ^= bs2;

  si ^= si2;
  si ^= bi2;
  ui ^= ui2;
  ui ^= bi2;
  bi ^= bi2;

  sl ^= sl2;
  sl ^= bl2;
  ul ^= ul2;
  ul ^= bl2;
  bl ^= bl2;
}

// CHECK-LABEL: define{{.*}} void @test_sl() #0 {
// CHECK:   [[TMP0:%.*]] = load volatile <16 x i8>, ptr @sc, align 8
// CHECK:   [[TMP1:%.*]] = load volatile <16 x i8>, ptr @sc2, align 8
// CHECK:   [[SHL:%.*]] = shl <16 x i8> [[TMP0]], [[TMP1]]
// CHECK:   store volatile <16 x i8> [[SHL]], ptr @sc, align 8
// CHECK:   [[TMP2:%.*]] = load volatile <16 x i8>, ptr @sc, align 8
// CHECK:   [[TMP3:%.*]] = load volatile <16 x i8>, ptr @uc2, align 8
// CHECK:   [[SHL1:%.*]] = shl <16 x i8> [[TMP2]], [[TMP3]]
// CHECK:   store volatile <16 x i8> [[SHL1]], ptr @sc, align 8
// CHECK:   [[TMP4:%.*]] = load volatile <16 x i8>, ptr @sc, align 8
// CHECK:   [[TMP5:%.*]] = load volatile i32, ptr @cnt, align 4
// CHECK:   [[SPLAT_SPLATINSERT:%.*]] = insertelement <16 x i32> poison, i32 [[TMP5]], i32 0
// CHECK:   [[SPLAT_SPLAT:%.*]] = shufflevector <16 x i32> [[SPLAT_SPLAT:%.*]]insert, <16 x i32> poison, <16 x i32> zeroinitializer
// CHECK:   [[SH_PROM:%.*]] = trunc <16 x i32> [[SPLAT_SPLAT]] to <16 x i8>
// CHECK:   [[SHL2:%.*]] = shl <16 x i8> [[TMP4]], [[SH_PROM]]
// CHECK:   store volatile <16 x i8> [[SHL2]], ptr @sc, align 8
// CHECK:   [[TMP6:%.*]] = load volatile <16 x i8>, ptr @sc, align 8
// CHECK:   [[SHL3:%.*]] = shl <16 x i8> [[TMP6]], <i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5>
// CHECK:   store volatile <16 x i8> [[SHL3]], ptr @sc, align 8
// CHECK:   [[TMP7:%.*]] = load volatile <16 x i8>, ptr @uc, align 8
// CHECK:   [[TMP8:%.*]] = load volatile <16 x i8>, ptr @sc2, align 8
// CHECK:   [[SHL4:%.*]] = shl <16 x i8> [[TMP7]], [[TMP8]]
// CHECK:   store volatile <16 x i8> [[SHL4]], ptr @uc, align 8
// CHECK:   [[TMP9:%.*]] = load volatile <16 x i8>, ptr @uc, align 8
// CHECK:   [[TMP10:%.*]] = load volatile <16 x i8>, ptr @uc2, align 8
// CHECK:   [[SHL5:%.*]] = shl <16 x i8> [[TMP9]], [[TMP10]]
// CHECK:   store volatile <16 x i8> [[SHL5]], ptr @uc, align 8
// CHECK:   [[TMP11:%.*]] = load volatile <16 x i8>, ptr @uc, align 8
// CHECK:   [[TMP12:%.*]] = load volatile i32, ptr @cnt, align 4
// CHECK:   [[SPLAT_SPLATINSERT6:%.*]] = insertelement <16 x i32> poison, i32 [[TMP12]], i32 0
// CHECK:   [[SPLAT_SPLAT7:%.*]] = shufflevector <16 x i32> [[SPLAT_SPLATINSERT6]], <16 x i32> poison, <16 x i32> zeroinitializer
// CHECK:   [[SH_PROM8:%.*]] = trunc <16 x i32> [[SPLAT_SPLAT7]] to <16 x i8>
// CHECK:   [[SHL9:%.*]] = shl <16 x i8> [[TMP11]], [[SH_PROM8]]
// CHECK:   store volatile <16 x i8> [[SHL9]], ptr @uc, align 8
// CHECK:   [[TMP13:%.*]] = load volatile <16 x i8>, ptr @uc, align 8
// CHECK:   [[SHL10:%.*]] = shl <16 x i8> [[TMP13]], <i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5>
// CHECK:   store volatile <16 x i8> [[SHL10]], ptr @uc, align 8
// CHECK:   [[TMP14:%.*]] = load volatile <8 x i16>, ptr @ss, align 8
// CHECK:   [[TMP15:%.*]] = load volatile <8 x i16>, ptr @ss2, align 8
// CHECK:   [[SHL11:%.*]] = shl <8 x i16> [[TMP14]], [[TMP15]]
// CHECK:   store volatile <8 x i16> [[SHL11]], ptr @ss, align 8
// CHECK:   [[TMP16:%.*]] = load volatile <8 x i16>, ptr @ss, align 8
// CHECK:   [[TMP17:%.*]] = load volatile <8 x i16>, ptr @us2, align 8
// CHECK:   [[SHL12:%.*]] = shl <8 x i16> [[TMP16]], [[TMP17]]
// CHECK:   store volatile <8 x i16> [[SHL12]], ptr @ss, align 8
// CHECK:   [[TMP18:%.*]] = load volatile <8 x i16>, ptr @ss, align 8
// CHECK:   [[TMP19:%.*]] = load volatile i32, ptr @cnt, align 4
// CHECK:   [[SPLAT_SPLATINSERT13:%.*]] = insertelement <8 x i32> poison, i32 [[TMP19]], i32 0
// CHECK:   [[SPLAT_SPLAT14:%.*]] = shufflevector <8 x i32> [[SPLAT_SPLATINSERT13]], <8 x i32> poison, <8 x i32> zeroinitializer
// CHECK:   [[SH_PROM15:%.*]] = trunc <8 x i32> [[SPLAT_SPLAT14]] to <8 x i16>
// CHECK:   [[SHL16:%.*]] = shl <8 x i16> [[TMP18]], [[SH_PROM15]]
// CHECK:   store volatile <8 x i16> [[SHL16]], ptr @ss, align 8
// CHECK:   [[TMP20:%.*]] = load volatile <8 x i16>, ptr @ss, align 8
// CHECK:   [[SHL17:%.*]] = shl <8 x i16> [[TMP20]], <i16 5, i16 5, i16 5, i16 5, i16 5, i16 5, i16 5, i16 5>
// CHECK:   store volatile <8 x i16> [[SHL17]], ptr @ss, align 8
// CHECK:   [[TMP21:%.*]] = load volatile <8 x i16>, ptr @us, align 8
// CHECK:   [[TMP22:%.*]] = load volatile <8 x i16>, ptr @ss2, align 8
// CHECK:   [[SHL18:%.*]] = shl <8 x i16> [[TMP21]], [[TMP22]]
// CHECK:   store volatile <8 x i16> [[SHL18]], ptr @us, align 8
// CHECK:   [[TMP23:%.*]] = load volatile <8 x i16>, ptr @us, align 8
// CHECK:   [[TMP24:%.*]] = load volatile <8 x i16>, ptr @us2, align 8
// CHECK:   [[SHL19:%.*]] = shl <8 x i16> [[TMP23]], [[TMP24]]
// CHECK:   store volatile <8 x i16> [[SHL19]], ptr @us, align 8
// CHECK:   [[TMP25:%.*]] = load volatile <8 x i16>, ptr @us, align 8
// CHECK:   [[TMP26:%.*]] = load volatile i32, ptr @cnt, align 4
// CHECK:   [[SPLAT_SPLATINSERT20:%.*]] = insertelement <8 x i32> poison, i32 [[TMP26]], i32 0
// CHECK:   [[SPLAT_SPLAT21:%.*]] = shufflevector <8 x i32> [[SPLAT_SPLATINSERT20]], <8 x i32> poison, <8 x i32> zeroinitializer
// CHECK:   [[SH_PROM22:%.*]] = trunc <8 x i32> [[SPLAT_SPLAT21]] to <8 x i16>
// CHECK:   [[SHL23:%.*]] = shl <8 x i16> [[TMP25]], [[SH_PROM22]]
// CHECK:   store volatile <8 x i16> [[SHL23]], ptr @us, align 8
// CHECK:   [[TMP27:%.*]] = load volatile <8 x i16>, ptr @us, align 8
// CHECK:   [[SHL24:%.*]] = shl <8 x i16> [[TMP27]], <i16 5, i16 5, i16 5, i16 5, i16 5, i16 5, i16 5, i16 5>
// CHECK:   store volatile <8 x i16> [[SHL24]], ptr @us, align 8
// CHECK:   [[TMP28:%.*]] = load volatile <4 x i32>, ptr @si, align 8
// CHECK:   [[TMP29:%.*]] = load volatile <4 x i32>, ptr @si2, align 8
// CHECK:   [[SHL25:%.*]] = shl <4 x i32> [[TMP28]], [[TMP29]]
// CHECK:   store volatile <4 x i32> [[SHL25]], ptr @si, align 8
// CHECK:   [[TMP30:%.*]] = load volatile <4 x i32>, ptr @si, align 8
// CHECK:   [[TMP31:%.*]] = load volatile <4 x i32>, ptr @ui2, align 8
// CHECK:   [[SHL26:%.*]] = shl <4 x i32> [[TMP30]], [[TMP31]]
// CHECK:   store volatile <4 x i32> [[SHL26]], ptr @si, align 8
// CHECK:   [[TMP32:%.*]] = load volatile <4 x i32>, ptr @si, align 8
// CHECK:   [[TMP33:%.*]] = load volatile i32, ptr @cnt, align 4
// CHECK:   [[SPLAT_SPLATINSERT27:%.*]] = insertelement <4 x i32> poison, i32 [[TMP33]], i32 0
// CHECK:   [[SPLAT_SPLAT28:%.*]] = shufflevector <4 x i32> [[SPLAT_SPLATINSERT27]], <4 x i32> poison, <4 x i32> zeroinitializer
// CHECK:   [[SHL29:%.*]] = shl <4 x i32> [[TMP32]], [[SPLAT_SPLAT28]]
// CHECK:   store volatile <4 x i32> [[SHL29]], ptr @si, align 8
// CHECK:   [[TMP34:%.*]] = load volatile <4 x i32>, ptr @si, align 8
// CHECK:   [[SHL30:%.*]] = shl <4 x i32> [[TMP34]], <i32 5, i32 5, i32 5, i32 5>
// CHECK:   store volatile <4 x i32> [[SHL30]], ptr @si, align 8
// CHECK:   [[TMP35:%.*]] = load volatile <4 x i32>, ptr @ui, align 8
// CHECK:   [[TMP36:%.*]] = load volatile <4 x i32>, ptr @si2, align 8
// CHECK:   [[SHL31:%.*]] = shl <4 x i32> [[TMP35]], [[TMP36]]
// CHECK:   store volatile <4 x i32> [[SHL31]], ptr @ui, align 8
// CHECK:   [[TMP37:%.*]] = load volatile <4 x i32>, ptr @ui, align 8
// CHECK:   [[TMP38:%.*]] = load volatile <4 x i32>, ptr @ui2, align 8
// CHECK:   [[SHL32:%.*]] = shl <4 x i32> [[TMP37]], [[TMP38]]
// CHECK:   store volatile <4 x i32> [[SHL32]], ptr @ui, align 8
// CHECK:   [[TMP39:%.*]] = load volatile <4 x i32>, ptr @ui, align 8
// CHECK:   [[TMP40:%.*]] = load volatile i32, ptr @cnt, align 4
// CHECK:   [[SPLAT_SPLATINSERT33:%.*]] = insertelement <4 x i32> poison, i32 [[TMP40]], i32 0
// CHECK:   [[SPLAT_SPLAT34:%.*]] = shufflevector <4 x i32> [[SPLAT_SPLATINSERT33]], <4 x i32> poison, <4 x i32> zeroinitializer
// CHECK:   [[SHL35:%.*]] = shl <4 x i32> [[TMP39]], [[SPLAT_SPLAT34]]
// CHECK:   store volatile <4 x i32> [[SHL35]], ptr @ui, align 8
// CHECK:   [[TMP41:%.*]] = load volatile <4 x i32>, ptr @ui, align 8
// CHECK:   [[SHL36:%.*]] = shl <4 x i32> [[TMP41]], <i32 5, i32 5, i32 5, i32 5>
// CHECK:   store volatile <4 x i32> [[SHL36]], ptr @ui, align 8
// CHECK:   [[TMP42:%.*]] = load volatile <2 x i64>, ptr @sl, align 8
// CHECK:   [[TMP43:%.*]] = load volatile <2 x i64>, ptr @sl2, align 8
// CHECK:   [[SHL37:%.*]] = shl <2 x i64> [[TMP42]], [[TMP43]]
// CHECK:   store volatile <2 x i64> [[SHL37]], ptr @sl, align 8
// CHECK:   [[TMP44:%.*]] = load volatile <2 x i64>, ptr @sl, align 8
// CHECK:   [[TMP45:%.*]] = load volatile <2 x i64>, ptr @ul2, align 8
// CHECK:   [[SHL38:%.*]] = shl <2 x i64> [[TMP44]], [[TMP45]]
// CHECK:   store volatile <2 x i64> [[SHL38]], ptr @sl, align 8
// CHECK:   [[TMP46:%.*]] = load volatile <2 x i64>, ptr @sl, align 8
// CHECK:   [[TMP47:%.*]] = load volatile i32, ptr @cnt, align 4
// CHECK:   [[SPLAT_SPLATINSERT39:%.*]] = insertelement <2 x i32> poison, i32 [[TMP47]], i32 0
// CHECK:   [[SPLAT_SPLAT40:%.*]] = shufflevector <2 x i32> [[SPLAT_SPLATINSERT39]], <2 x i32> poison, <2 x i32> zeroinitializer
// CHECK:   [[SH_PROM41:%.*]] = zext <2 x i32> [[SPLAT_SPLAT40]] to <2 x i64>
// CHECK:   [[SHL42:%.*]] = shl <2 x i64> [[TMP46]], [[SH_PROM41]]
// CHECK:   store volatile <2 x i64> [[SHL42]], ptr @sl, align 8
// CHECK:   [[TMP48:%.*]] = load volatile <2 x i64>, ptr @sl, align 8
// CHECK:   [[SHL43:%.*]] = shl <2 x i64> [[TMP48]], <i64 5, i64 5>
// CHECK:   store volatile <2 x i64> [[SHL43]], ptr @sl, align 8
// CHECK:   [[TMP49:%.*]] = load volatile <2 x i64>, ptr @ul, align 8
// CHECK:   [[TMP50:%.*]] = load volatile <2 x i64>, ptr @sl2, align 8
// CHECK:   [[SHL44:%.*]] = shl <2 x i64> [[TMP49]], [[TMP50]]
// CHECK:   store volatile <2 x i64> [[SHL44]], ptr @ul, align 8
// CHECK:   [[TMP51:%.*]] = load volatile <2 x i64>, ptr @ul, align 8
// CHECK:   [[TMP52:%.*]] = load volatile <2 x i64>, ptr @ul2, align 8
// CHECK:   [[SHL45:%.*]] = shl <2 x i64> [[TMP51]], [[TMP52]]
// CHECK:   store volatile <2 x i64> [[SHL45]], ptr @ul, align 8
// CHECK:   [[TMP53:%.*]] = load volatile <2 x i64>, ptr @ul, align 8
// CHECK:   [[TMP54:%.*]] = load volatile i32, ptr @cnt, align 4
// CHECK:   [[SPLAT_SPLATINSERT46:%.*]] = insertelement <2 x i32> poison, i32 [[TMP54]], i32 0
// CHECK:   [[SPLAT_SPLAT47:%.*]] = shufflevector <2 x i32> [[SPLAT_SPLATINSERT46]], <2 x i32> poison, <2 x i32> zeroinitializer
// CHECK:   [[SH_PROM48:%.*]] = zext <2 x i32> [[SPLAT_SPLAT47]] to <2 x i64>
// CHECK:   [[SHL49:%.*]] = shl <2 x i64> [[TMP53]], [[SH_PROM48]]
// CHECK:   store volatile <2 x i64> [[SHL49]], ptr @ul, align 8
// CHECK:   [[TMP55:%.*]] = load volatile <2 x i64>, ptr @ul, align 8
// CHECK:   [[SHL50:%.*]] = shl <2 x i64> [[TMP55]], <i64 5, i64 5>
// CHECK:   store volatile <2 x i64> [[SHL50]], ptr @ul, align 8
// CHECK:   ret void
void test_sl(void) {

  sc = sc << sc2;
  sc = sc << uc2;
  sc = sc << cnt;
  sc = sc << 5;
  uc = uc << sc2;
  uc = uc << uc2;
  uc = uc << cnt;
  uc = uc << 5;

  ss = ss << ss2;
  ss = ss << us2;
  ss = ss << cnt;
  ss = ss << 5;
  us = us << ss2;
  us = us << us2;
  us = us << cnt;
  us = us << 5;

  si = si << si2;
  si = si << ui2;
  si = si << cnt;
  si = si << 5;
  ui = ui << si2;
  ui = ui << ui2;
  ui = ui << cnt;
  ui = ui << 5;

  sl = sl << sl2;
  sl = sl << ul2;
  sl = sl << cnt;
  sl = sl << 5;
  ul = ul << sl2;
  ul = ul << ul2;
  ul = ul << cnt;
  ul = ul << 5;
}

// CHECK-LABEL: define{{.*}} void @test_sl_assign() #0 {
// CHECK:   [[TMP0:%.*]] = load volatile <16 x i8>, ptr @sc2, align 8
// CHECK:   [[TMP1:%.*]] = load volatile <16 x i8>, ptr @sc, align 8
// CHECK:   [[SHL:%.*]] = shl <16 x i8> [[TMP1]], [[TMP0]]
// CHECK:   store volatile <16 x i8> [[SHL]], ptr @sc, align 8
// CHECK:   [[TMP2:%.*]] = load volatile <16 x i8>, ptr @uc2, align 8
// CHECK:   [[TMP3:%.*]] = load volatile <16 x i8>, ptr @sc, align 8
// CHECK:   [[SHL1:%.*]] = shl <16 x i8> [[TMP3]], [[TMP2]]
// CHECK:   store volatile <16 x i8> [[SHL1]], ptr @sc, align 8
// CHECK:   [[TMP4:%.*]] = load volatile i32, ptr @cnt, align 4
// CHECK:   [[SPLAT_SPLATINSERT:%.*]] = insertelement <16 x i32> poison, i32 [[TMP4]], i32 0
// CHECK:   [[SPLAT_SPLAT:%.*]] = shufflevector <16 x i32> [[SPLAT_SPLAT:%.*]]insert, <16 x i32> poison, <16 x i32> zeroinitializer
// CHECK:   [[TMP5:%.*]] = load volatile <16 x i8>, ptr @sc, align 8
// CHECK:   [[SH_PROM:%.*]] = trunc <16 x i32> [[SPLAT_SPLAT]] to <16 x i8>
// CHECK:   [[SHL2:%.*]] = shl <16 x i8> [[TMP5]], [[SH_PROM]]
// CHECK:   store volatile <16 x i8> [[SHL2]], ptr @sc, align 8
// CHECK:   [[TMP6:%.*]] = load volatile <16 x i8>, ptr @sc, align 8
// CHECK:   [[SHL3:%.*]] = shl <16 x i8> [[TMP6]], <i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5>
// CHECK:   store volatile <16 x i8> [[SHL3]], ptr @sc, align 8
// CHECK:   [[TMP7:%.*]] = load volatile <16 x i8>, ptr @sc2, align 8
// CHECK:   [[TMP8:%.*]] = load volatile <16 x i8>, ptr @uc, align 8
// CHECK:   [[SHL4:%.*]] = shl <16 x i8> [[TMP8]], [[TMP7]]
// CHECK:   store volatile <16 x i8> [[SHL4]], ptr @uc, align 8
// CHECK:   [[TMP9:%.*]] = load volatile <16 x i8>, ptr @uc2, align 8
// CHECK:   [[TMP10:%.*]] = load volatile <16 x i8>, ptr @uc, align 8
// CHECK:   [[SHL5:%.*]] = shl <16 x i8> [[TMP10]], [[TMP9]]
// CHECK:   store volatile <16 x i8> [[SHL5]], ptr @uc, align 8
// CHECK:   [[TMP11:%.*]] = load volatile i32, ptr @cnt, align 4
// CHECK:   [[SPLAT_SPLATINSERT6:%.*]] = insertelement <16 x i32> poison, i32 [[TMP11]], i32 0
// CHECK:   [[SPLAT_SPLAT7:%.*]] = shufflevector <16 x i32> [[SPLAT_SPLATINSERT6]], <16 x i32> poison, <16 x i32> zeroinitializer
// CHECK:   [[TMP12:%.*]] = load volatile <16 x i8>, ptr @uc, align 8
// CHECK:   [[SH_PROM8:%.*]] = trunc <16 x i32> [[SPLAT_SPLAT7]] to <16 x i8>
// CHECK:   [[SHL9:%.*]] = shl <16 x i8> [[TMP12]], [[SH_PROM8]]
// CHECK:   store volatile <16 x i8> [[SHL9]], ptr @uc, align 8
// CHECK:   [[TMP13:%.*]] = load volatile <16 x i8>, ptr @uc, align 8
// CHECK:   [[SHL10:%.*]] = shl <16 x i8> [[TMP13]], <i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5>
// CHECK:   store volatile <16 x i8> [[SHL10]], ptr @uc, align 8
// CHECK:   [[TMP14:%.*]] = load volatile <8 x i16>, ptr @ss2, align 8
// CHECK:   [[TMP15:%.*]] = load volatile <8 x i16>, ptr @ss, align 8
// CHECK:   [[SHL11:%.*]] = shl <8 x i16> [[TMP15]], [[TMP14]]
// CHECK:   store volatile <8 x i16> [[SHL11]], ptr @ss, align 8
// CHECK:   [[TMP16:%.*]] = load volatile <8 x i16>, ptr @us2, align 8
// CHECK:   [[TMP17:%.*]] = load volatile <8 x i16>, ptr @ss, align 8
// CHECK:   [[SHL12:%.*]] = shl <8 x i16> [[TMP17]], [[TMP16]]
// CHECK:   store volatile <8 x i16> [[SHL12]], ptr @ss, align 8
// CHECK:   [[TMP18:%.*]] = load volatile i32, ptr @cnt, align 4
// CHECK:   [[SPLAT_SPLATINSERT13:%.*]] = insertelement <8 x i32> poison, i32 [[TMP18]], i32 0
// CHECK:   [[SPLAT_SPLAT14:%.*]] = shufflevector <8 x i32> [[SPLAT_SPLATINSERT13]], <8 x i32> poison, <8 x i32> zeroinitializer
// CHECK:   [[TMP19:%.*]] = load volatile <8 x i16>, ptr @ss, align 8
// CHECK:   [[SH_PROM15:%.*]] = trunc <8 x i32> [[SPLAT_SPLAT14]] to <8 x i16>
// CHECK:   [[SHL16:%.*]] = shl <8 x i16> [[TMP19]], [[SH_PROM15]]
// CHECK:   store volatile <8 x i16> [[SHL16]], ptr @ss, align 8
// CHECK:   [[TMP20:%.*]] = load volatile <8 x i16>, ptr @ss, align 8
// CHECK:   [[SHL17:%.*]] = shl <8 x i16> [[TMP20]], <i16 5, i16 5, i16 5, i16 5, i16 5, i16 5, i16 5, i16 5>
// CHECK:   store volatile <8 x i16> [[SHL17]], ptr @ss, align 8
// CHECK:   [[TMP21:%.*]] = load volatile <8 x i16>, ptr @ss2, align 8
// CHECK:   [[TMP22:%.*]] = load volatile <8 x i16>, ptr @us, align 8
// CHECK:   [[SHL18:%.*]] = shl <8 x i16> [[TMP22]], [[TMP21]]
// CHECK:   store volatile <8 x i16> [[SHL18]], ptr @us, align 8
// CHECK:   [[TMP23:%.*]] = load volatile <8 x i16>, ptr @us2, align 8
// CHECK:   [[TMP24:%.*]] = load volatile <8 x i16>, ptr @us, align 8
// CHECK:   [[SHL19:%.*]] = shl <8 x i16> [[TMP24]], [[TMP23]]
// CHECK:   store volatile <8 x i16> [[SHL19]], ptr @us, align 8
// CHECK:   [[TMP25:%.*]] = load volatile i32, ptr @cnt, align 4
// CHECK:   [[SPLAT_SPLATINSERT20:%.*]] = insertelement <8 x i32> poison, i32 [[TMP25]], i32 0
// CHECK:   [[SPLAT_SPLAT21:%.*]] = shufflevector <8 x i32> [[SPLAT_SPLATINSERT20]], <8 x i32> poison, <8 x i32> zeroinitializer
// CHECK:   [[TMP26:%.*]] = load volatile <8 x i16>, ptr @us, align 8
// CHECK:   [[SH_PROM22:%.*]] = trunc <8 x i32> [[SPLAT_SPLAT21]] to <8 x i16>
// CHECK:   [[SHL23:%.*]] = shl <8 x i16> [[TMP26]], [[SH_PROM22]]
// CHECK:   store volatile <8 x i16> [[SHL23]], ptr @us, align 8
// CHECK:   [[TMP27:%.*]] = load volatile <8 x i16>, ptr @us, align 8
// CHECK:   [[SHL24:%.*]] = shl <8 x i16> [[TMP27]], <i16 5, i16 5, i16 5, i16 5, i16 5, i16 5, i16 5, i16 5>
// CHECK:   store volatile <8 x i16> [[SHL24]], ptr @us, align 8
// CHECK:   [[TMP28:%.*]] = load volatile <4 x i32>, ptr @si2, align 8
// CHECK:   [[TMP29:%.*]] = load volatile <4 x i32>, ptr @si, align 8
// CHECK:   [[SHL25:%.*]] = shl <4 x i32> [[TMP29]], [[TMP28]]
// CHECK:   store volatile <4 x i32> [[SHL25]], ptr @si, align 8
// CHECK:   [[TMP30:%.*]] = load volatile <4 x i32>, ptr @ui2, align 8
// CHECK:   [[TMP31:%.*]] = load volatile <4 x i32>, ptr @si, align 8
// CHECK:   [[SHL26:%.*]] = shl <4 x i32> [[TMP31]], [[TMP30]]
// CHECK:   store volatile <4 x i32> [[SHL26]], ptr @si, align 8
// CHECK:   [[TMP32:%.*]] = load volatile i32, ptr @cnt, align 4
// CHECK:   [[SPLAT_SPLATINSERT27:%.*]] = insertelement <4 x i32> poison, i32 [[TMP32]], i32 0
// CHECK:   [[SPLAT_SPLAT28:%.*]] = shufflevector <4 x i32> [[SPLAT_SPLATINSERT27]], <4 x i32> poison, <4 x i32> zeroinitializer
// CHECK:   [[TMP33:%.*]] = load volatile <4 x i32>, ptr @si, align 8
// CHECK:   [[SHL29:%.*]] = shl <4 x i32> [[TMP33]], [[SPLAT_SPLAT28]]
// CHECK:   store volatile <4 x i32> [[SHL29]], ptr @si, align 8
// CHECK:   [[TMP34:%.*]] = load volatile <4 x i32>, ptr @si, align 8
// CHECK:   [[SHL30:%.*]] = shl <4 x i32> [[TMP34]], <i32 5, i32 5, i32 5, i32 5>
// CHECK:   store volatile <4 x i32> [[SHL30]], ptr @si, align 8
// CHECK:   [[TMP35:%.*]] = load volatile <4 x i32>, ptr @si2, align 8
// CHECK:   [[TMP36:%.*]] = load volatile <4 x i32>, ptr @ui, align 8
// CHECK:   [[SHL31:%.*]] = shl <4 x i32> [[TMP36]], [[TMP35]]
// CHECK:   store volatile <4 x i32> [[SHL31]], ptr @ui, align 8
// CHECK:   [[TMP37:%.*]] = load volatile <4 x i32>, ptr @ui2, align 8
// CHECK:   [[TMP38:%.*]] = load volatile <4 x i32>, ptr @ui, align 8
// CHECK:   [[SHL32:%.*]] = shl <4 x i32> [[TMP38]], [[TMP37]]
// CHECK:   store volatile <4 x i32> [[SHL32]], ptr @ui, align 8
// CHECK:   [[TMP39:%.*]] = load volatile i32, ptr @cnt, align 4
// CHECK:   [[SPLAT_SPLATINSERT33:%.*]] = insertelement <4 x i32> poison, i32 [[TMP39]], i32 0
// CHECK:   [[SPLAT_SPLAT34:%.*]] = shufflevector <4 x i32> [[SPLAT_SPLATINSERT33]], <4 x i32> poison, <4 x i32> zeroinitializer
// CHECK:   [[TMP40:%.*]] = load volatile <4 x i32>, ptr @ui, align 8
// CHECK:   [[SHL35:%.*]] = shl <4 x i32> [[TMP40]], [[SPLAT_SPLAT34]]
// CHECK:   store volatile <4 x i32> [[SHL35]], ptr @ui, align 8
// CHECK:   [[TMP41:%.*]] = load volatile <4 x i32>, ptr @ui, align 8
// CHECK:   [[SHL36:%.*]] = shl <4 x i32> [[TMP41]], <i32 5, i32 5, i32 5, i32 5>
// CHECK:   store volatile <4 x i32> [[SHL36]], ptr @ui, align 8
// CHECK:   [[TMP42:%.*]] = load volatile <2 x i64>, ptr @sl2, align 8
// CHECK:   [[TMP43:%.*]] = load volatile <2 x i64>, ptr @sl, align 8
// CHECK:   [[SHL37:%.*]] = shl <2 x i64> [[TMP43]], [[TMP42]]
// CHECK:   store volatile <2 x i64> [[SHL37]], ptr @sl, align 8
// CHECK:   [[TMP44:%.*]] = load volatile <2 x i64>, ptr @ul2, align 8
// CHECK:   [[TMP45:%.*]] = load volatile <2 x i64>, ptr @sl, align 8
// CHECK:   [[SHL38:%.*]] = shl <2 x i64> [[TMP45]], [[TMP44]]
// CHECK:   store volatile <2 x i64> [[SHL38]], ptr @sl, align 8
// CHECK:   [[TMP46:%.*]] = load volatile i32, ptr @cnt, align 4
// CHECK:   [[SPLAT_SPLATINSERT39:%.*]] = insertelement <2 x i32> poison, i32 [[TMP46]], i32 0
// CHECK:   [[SPLAT_SPLAT40:%.*]] = shufflevector <2 x i32> [[SPLAT_SPLATINSERT39]], <2 x i32> poison, <2 x i32> zeroinitializer
// CHECK:   [[TMP47:%.*]] = load volatile <2 x i64>, ptr @sl, align 8
// CHECK:   [[SH_PROM41:%.*]] = zext <2 x i32> [[SPLAT_SPLAT40]] to <2 x i64>
// CHECK:   [[SHL42:%.*]] = shl <2 x i64> [[TMP47]], [[SH_PROM41]]
// CHECK:   store volatile <2 x i64> [[SHL42]], ptr @sl, align 8
// CHECK:   [[TMP48:%.*]] = load volatile <2 x i64>, ptr @sl, align 8
// CHECK:   [[SHL43:%.*]] = shl <2 x i64> [[TMP48]], <i64 5, i64 5>
// CHECK:   store volatile <2 x i64> [[SHL43]], ptr @sl, align 8
// CHECK:   [[TMP49:%.*]] = load volatile <2 x i64>, ptr @sl2, align 8
// CHECK:   [[TMP50:%.*]] = load volatile <2 x i64>, ptr @ul, align 8
// CHECK:   [[SHL44:%.*]] = shl <2 x i64> [[TMP50]], [[TMP49]]
// CHECK:   store volatile <2 x i64> [[SHL44]], ptr @ul, align 8
// CHECK:   [[TMP51:%.*]] = load volatile <2 x i64>, ptr @ul2, align 8
// CHECK:   [[TMP52:%.*]] = load volatile <2 x i64>, ptr @ul, align 8
// CHECK:   [[SHL45:%.*]] = shl <2 x i64> [[TMP52]], [[TMP51]]
// CHECK:   store volatile <2 x i64> [[SHL45]], ptr @ul, align 8
// CHECK:   [[TMP53:%.*]] = load volatile i32, ptr @cnt, align 4
// CHECK:   [[SPLAT_SPLATINSERT46:%.*]] = insertelement <2 x i32> poison, i32 [[TMP53]], i32 0
// CHECK:   [[SPLAT_SPLAT47:%.*]] = shufflevector <2 x i32> [[SPLAT_SPLATINSERT46]], <2 x i32> poison, <2 x i32> zeroinitializer
// CHECK:   [[TMP54:%.*]] = load volatile <2 x i64>, ptr @ul, align 8
// CHECK:   [[SH_PROM48:%.*]] = zext <2 x i32> [[SPLAT_SPLAT47]] to <2 x i64>
// CHECK:   [[SHL49:%.*]] = shl <2 x i64> [[TMP54]], [[SH_PROM48]]
// CHECK:   store volatile <2 x i64> [[SHL49]], ptr @ul, align 8
// CHECK:   [[TMP55:%.*]] = load volatile <2 x i64>, ptr @ul, align 8
// CHECK:   [[SHL50:%.*]] = shl <2 x i64> [[TMP55]], <i64 5, i64 5>
// CHECK:   store volatile <2 x i64> [[SHL50]], ptr @ul, align 8
// CHECK:   ret void
void test_sl_assign(void) {

  sc <<= sc2;
  sc <<= uc2;
  sc <<= cnt;
  sc <<= 5;
  uc <<= sc2;
  uc <<= uc2;
  uc <<= cnt;
  uc <<= 5;

  ss <<= ss2;
  ss <<= us2;
  ss <<= cnt;
  ss <<= 5;
  us <<= ss2;
  us <<= us2;
  us <<= cnt;
  us <<= 5;

  si <<= si2;
  si <<= ui2;
  si <<= cnt;
  si <<= 5;
  ui <<= si2;
  ui <<= ui2;
  ui <<= cnt;
  ui <<= 5;

  sl <<= sl2;
  sl <<= ul2;
  sl <<= cnt;
  sl <<= 5;
  ul <<= sl2;
  ul <<= ul2;
  ul <<= cnt;
  ul <<= 5;
}

// CHECK-LABEL: define{{.*}} void @test_sr() #0 {
// CHECK:   [[TMP0:%.*]] = load volatile <16 x i8>, ptr @sc, align 8
// CHECK:   [[TMP1:%.*]] = load volatile <16 x i8>, ptr @sc2, align 8
// CHECK:   [[SHR:%.*]] = ashr <16 x i8> [[TMP0]], [[TMP1]]
// CHECK:   store volatile <16 x i8> [[SHR]], ptr @sc, align 8
// CHECK:   [[TMP2:%.*]] = load volatile <16 x i8>, ptr @sc, align 8
// CHECK:   [[TMP3:%.*]] = load volatile <16 x i8>, ptr @uc2, align 8
// CHECK:   [[SHR1:%.*]] = ashr <16 x i8> [[TMP2]], [[TMP3]]
// CHECK:   store volatile <16 x i8> [[SHR1]], ptr @sc, align 8
// CHECK:   [[TMP4:%.*]] = load volatile <16 x i8>, ptr @sc, align 8
// CHECK:   [[TMP5:%.*]] = load volatile i32, ptr @cnt, align 4
// CHECK:   [[SPLAT_SPLATINSERT:%.*]] = insertelement <16 x i32> poison, i32 [[TMP5]], i32 0
// CHECK:   [[SPLAT_SPLAT:%.*]] = shufflevector <16 x i32> [[SPLAT_SPLAT:%.*]]insert, <16 x i32> poison, <16 x i32> zeroinitializer
// CHECK:   [[SH_PROM:%.*]] = trunc <16 x i32> [[SPLAT_SPLAT]] to <16 x i8>
// CHECK:   [[SHR2:%.*]] = ashr <16 x i8> [[TMP4]], [[SH_PROM]]
// CHECK:   store volatile <16 x i8> [[SHR2]], ptr @sc, align 8
// CHECK:   [[TMP6:%.*]] = load volatile <16 x i8>, ptr @sc, align 8
// CHECK:   [[SHR3:%.*]] = ashr <16 x i8> [[TMP6]], <i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5>
// CHECK:   store volatile <16 x i8> [[SHR3]], ptr @sc, align 8
// CHECK:   [[TMP7:%.*]] = load volatile <16 x i8>, ptr @uc, align 8
// CHECK:   [[TMP8:%.*]] = load volatile <16 x i8>, ptr @sc2, align 8
// CHECK:   [[SHR4:%.*]] = lshr <16 x i8> [[TMP7]], [[TMP8]]
// CHECK:   store volatile <16 x i8> [[SHR4]], ptr @uc, align 8
// CHECK:   [[TMP9:%.*]] = load volatile <16 x i8>, ptr @uc, align 8
// CHECK:   [[TMP10:%.*]] = load volatile <16 x i8>, ptr @uc2, align 8
// CHECK:   [[SHR5:%.*]] = lshr <16 x i8> [[TMP9]], [[TMP10]]
// CHECK:   store volatile <16 x i8> [[SHR5]], ptr @uc, align 8
// CHECK:   [[TMP11:%.*]] = load volatile <16 x i8>, ptr @uc, align 8
// CHECK:   [[TMP12:%.*]] = load volatile i32, ptr @cnt, align 4
// CHECK:   [[SPLAT_SPLATINSERT6:%.*]] = insertelement <16 x i32> poison, i32 [[TMP12]], i32 0
// CHECK:   [[SPLAT_SPLAT7:%.*]] = shufflevector <16 x i32> [[SPLAT_SPLATINSERT6]], <16 x i32> poison, <16 x i32> zeroinitializer
// CHECK:   [[SH_PROM8:%.*]] = trunc <16 x i32> [[SPLAT_SPLAT7]] to <16 x i8>
// CHECK:   [[SHR9:%.*]] = lshr <16 x i8> [[TMP11]], [[SH_PROM8]]
// CHECK:   store volatile <16 x i8> [[SHR9]], ptr @uc, align 8
// CHECK:   [[TMP13:%.*]] = load volatile <16 x i8>, ptr @uc, align 8
// CHECK:   [[SHR10:%.*]] = lshr <16 x i8> [[TMP13]], <i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5>
// CHECK:   store volatile <16 x i8> [[SHR10]], ptr @uc, align 8
// CHECK:   [[TMP14:%.*]] = load volatile <8 x i16>, ptr @ss, align 8
// CHECK:   [[TMP15:%.*]] = load volatile <8 x i16>, ptr @ss2, align 8
// CHECK:   [[SHR11:%.*]] = ashr <8 x i16> [[TMP14]], [[TMP15]]
// CHECK:   store volatile <8 x i16> [[SHR11]], ptr @ss, align 8
// CHECK:   [[TMP16:%.*]] = load volatile <8 x i16>, ptr @ss, align 8
// CHECK:   [[TMP17:%.*]] = load volatile <8 x i16>, ptr @us2, align 8
// CHECK:   [[SHR12:%.*]] = ashr <8 x i16> [[TMP16]], [[TMP17]]
// CHECK:   store volatile <8 x i16> [[SHR12]], ptr @ss, align 8
// CHECK:   [[TMP18:%.*]] = load volatile <8 x i16>, ptr @ss, align 8
// CHECK:   [[TMP19:%.*]] = load volatile i32, ptr @cnt, align 4
// CHECK:   [[SPLAT_SPLATINSERT13:%.*]] = insertelement <8 x i32> poison, i32 [[TMP19]], i32 0
// CHECK:   [[SPLAT_SPLAT14:%.*]] = shufflevector <8 x i32> [[SPLAT_SPLATINSERT13]], <8 x i32> poison, <8 x i32> zeroinitializer
// CHECK:   [[SH_PROM15:%.*]] = trunc <8 x i32> [[SPLAT_SPLAT14]] to <8 x i16>
// CHECK:   [[SHR16:%.*]] = ashr <8 x i16> [[TMP18]], [[SH_PROM15]]
// CHECK:   store volatile <8 x i16> [[SHR16]], ptr @ss, align 8
// CHECK:   [[TMP20:%.*]] = load volatile <8 x i16>, ptr @ss, align 8
// CHECK:   [[SHR17:%.*]] = ashr <8 x i16> [[TMP20]], <i16 5, i16 5, i16 5, i16 5, i16 5, i16 5, i16 5, i16 5>
// CHECK:   store volatile <8 x i16> [[SHR17]], ptr @ss, align 8
// CHECK:   [[TMP21:%.*]] = load volatile <8 x i16>, ptr @us, align 8
// CHECK:   [[TMP22:%.*]] = load volatile <8 x i16>, ptr @ss2, align 8
// CHECK:   [[SHR18:%.*]] = lshr <8 x i16> [[TMP21]], [[TMP22]]
// CHECK:   store volatile <8 x i16> [[SHR18]], ptr @us, align 8
// CHECK:   [[TMP23:%.*]] = load volatile <8 x i16>, ptr @us, align 8
// CHECK:   [[TMP24:%.*]] = load volatile <8 x i16>, ptr @us2, align 8
// CHECK:   [[SHR19:%.*]] = lshr <8 x i16> [[TMP23]], [[TMP24]]
// CHECK:   store volatile <8 x i16> [[SHR19]], ptr @us, align 8
// CHECK:   [[TMP25:%.*]] = load volatile <8 x i16>, ptr @us, align 8
// CHECK:   [[TMP26:%.*]] = load volatile i32, ptr @cnt, align 4
// CHECK:   [[SPLAT_SPLATINSERT20:%.*]] = insertelement <8 x i32> poison, i32 [[TMP26]], i32 0
// CHECK:   [[SPLAT_SPLAT21:%.*]] = shufflevector <8 x i32> [[SPLAT_SPLATINSERT20]], <8 x i32> poison, <8 x i32> zeroinitializer
// CHECK:   [[SH_PROM22:%.*]] = trunc <8 x i32> [[SPLAT_SPLAT21]] to <8 x i16>
// CHECK:   [[SHR23:%.*]] = lshr <8 x i16> [[TMP25]], [[SH_PROM22]]
// CHECK:   store volatile <8 x i16> [[SHR23]], ptr @us, align 8
// CHECK:   [[TMP27:%.*]] = load volatile <8 x i16>, ptr @us, align 8
// CHECK:   [[SHR24:%.*]] = lshr <8 x i16> [[TMP27]], <i16 5, i16 5, i16 5, i16 5, i16 5, i16 5, i16 5, i16 5>
// CHECK:   store volatile <8 x i16> [[SHR24]], ptr @us, align 8
// CHECK:   [[TMP28:%.*]] = load volatile <4 x i32>, ptr @si, align 8
// CHECK:   [[TMP29:%.*]] = load volatile <4 x i32>, ptr @si2, align 8
// CHECK:   [[SHR25:%.*]] = ashr <4 x i32> [[TMP28]], [[TMP29]]
// CHECK:   store volatile <4 x i32> [[SHR25]], ptr @si, align 8
// CHECK:   [[TMP30:%.*]] = load volatile <4 x i32>, ptr @si, align 8
// CHECK:   [[TMP31:%.*]] = load volatile <4 x i32>, ptr @ui2, align 8
// CHECK:   [[SHR26:%.*]] = ashr <4 x i32> [[TMP30]], [[TMP31]]
// CHECK:   store volatile <4 x i32> [[SHR26]], ptr @si, align 8
// CHECK:   [[TMP32:%.*]] = load volatile <4 x i32>, ptr @si, align 8
// CHECK:   [[TMP33:%.*]] = load volatile i32, ptr @cnt, align 4
// CHECK:   [[SPLAT_SPLATINSERT27:%.*]] = insertelement <4 x i32> poison, i32 [[TMP33]], i32 0
// CHECK:   [[SPLAT_SPLAT28:%.*]] = shufflevector <4 x i32> [[SPLAT_SPLATINSERT27]], <4 x i32> poison, <4 x i32> zeroinitializer
// CHECK:   [[SHR29:%.*]] = ashr <4 x i32> [[TMP32]], [[SPLAT_SPLAT28]]
// CHECK:   store volatile <4 x i32> [[SHR29]], ptr @si, align 8
// CHECK:   [[TMP34:%.*]] = load volatile <4 x i32>, ptr @si, align 8
// CHECK:   [[SHR30:%.*]] = ashr <4 x i32> [[TMP34]], <i32 5, i32 5, i32 5, i32 5>
// CHECK:   store volatile <4 x i32> [[SHR30]], ptr @si, align 8
// CHECK:   [[TMP35:%.*]] = load volatile <4 x i32>, ptr @ui, align 8
// CHECK:   [[TMP36:%.*]] = load volatile <4 x i32>, ptr @si2, align 8
// CHECK:   [[SHR31:%.*]] = lshr <4 x i32> [[TMP35]], [[TMP36]]
// CHECK:   store volatile <4 x i32> [[SHR31]], ptr @ui, align 8
// CHECK:   [[TMP37:%.*]] = load volatile <4 x i32>, ptr @ui, align 8
// CHECK:   [[TMP38:%.*]] = load volatile <4 x i32>, ptr @ui2, align 8
// CHECK:   [[SHR32:%.*]] = lshr <4 x i32> [[TMP37]], [[TMP38]]
// CHECK:   store volatile <4 x i32> [[SHR32]], ptr @ui, align 8
// CHECK:   [[TMP39:%.*]] = load volatile <4 x i32>, ptr @ui, align 8
// CHECK:   [[TMP40:%.*]] = load volatile i32, ptr @cnt, align 4
// CHECK:   [[SPLAT_SPLATINSERT33:%.*]] = insertelement <4 x i32> poison, i32 [[TMP40]], i32 0
// CHECK:   [[SPLAT_SPLAT34:%.*]] = shufflevector <4 x i32> [[SPLAT_SPLATINSERT33]], <4 x i32> poison, <4 x i32> zeroinitializer
// CHECK:   [[SHR35:%.*]] = lshr <4 x i32> [[TMP39]], [[SPLAT_SPLAT34]]
// CHECK:   store volatile <4 x i32> [[SHR35]], ptr @ui, align 8
// CHECK:   [[TMP41:%.*]] = load volatile <4 x i32>, ptr @ui, align 8
// CHECK:   [[SHR36:%.*]] = lshr <4 x i32> [[TMP41]], <i32 5, i32 5, i32 5, i32 5>
// CHECK:   store volatile <4 x i32> [[SHR36]], ptr @ui, align 8
// CHECK:   [[TMP42:%.*]] = load volatile <2 x i64>, ptr @sl, align 8
// CHECK:   [[TMP43:%.*]] = load volatile <2 x i64>, ptr @sl2, align 8
// CHECK:   [[SHR37:%.*]] = ashr <2 x i64> [[TMP42]], [[TMP43]]
// CHECK:   store volatile <2 x i64> [[SHR37]], ptr @sl, align 8
// CHECK:   [[TMP44:%.*]] = load volatile <2 x i64>, ptr @sl, align 8
// CHECK:   [[TMP45:%.*]] = load volatile <2 x i64>, ptr @ul2, align 8
// CHECK:   [[SHR38:%.*]] = ashr <2 x i64> [[TMP44]], [[TMP45]]
// CHECK:   store volatile <2 x i64> [[SHR38]], ptr @sl, align 8
// CHECK:   [[TMP46:%.*]] = load volatile <2 x i64>, ptr @sl, align 8
// CHECK:   [[TMP47:%.*]] = load volatile i32, ptr @cnt, align 4
// CHECK:   [[SPLAT_SPLATINSERT39:%.*]] = insertelement <2 x i32> poison, i32 [[TMP47]], i32 0
// CHECK:   [[SPLAT_SPLAT40:%.*]] = shufflevector <2 x i32> [[SPLAT_SPLATINSERT39]], <2 x i32> poison, <2 x i32> zeroinitializer
// CHECK:   [[SH_PROM41:%.*]] = zext <2 x i32> [[SPLAT_SPLAT40]] to <2 x i64>
// CHECK:   [[SHR42:%.*]] = ashr <2 x i64> [[TMP46]], [[SH_PROM41]]
// CHECK:   store volatile <2 x i64> [[SHR42]], ptr @sl, align 8
// CHECK:   [[TMP48:%.*]] = load volatile <2 x i64>, ptr @sl, align 8
// CHECK:   [[SHR43:%.*]] = ashr <2 x i64> [[TMP48]], <i64 5, i64 5>
// CHECK:   store volatile <2 x i64> [[SHR43]], ptr @sl, align 8
// CHECK:   [[TMP49:%.*]] = load volatile <2 x i64>, ptr @ul, align 8
// CHECK:   [[TMP50:%.*]] = load volatile <2 x i64>, ptr @sl2, align 8
// CHECK:   [[SHR44:%.*]] = lshr <2 x i64> [[TMP49]], [[TMP50]]
// CHECK:   store volatile <2 x i64> [[SHR44]], ptr @ul, align 8
// CHECK:   [[TMP51:%.*]] = load volatile <2 x i64>, ptr @ul, align 8
// CHECK:   [[TMP52:%.*]] = load volatile <2 x i64>, ptr @ul2, align 8
// CHECK:   [[SHR45:%.*]] = lshr <2 x i64> [[TMP51]], [[TMP52]]
// CHECK:   store volatile <2 x i64> [[SHR45]], ptr @ul, align 8
// CHECK:   [[TMP53:%.*]] = load volatile <2 x i64>, ptr @ul, align 8
// CHECK:   [[TMP54:%.*]] = load volatile i32, ptr @cnt, align 4
// CHECK:   [[SPLAT_SPLATINSERT46:%.*]] = insertelement <2 x i32> poison, i32 [[TMP54]], i32 0
// CHECK:   [[SPLAT_SPLAT47:%.*]] = shufflevector <2 x i32> [[SPLAT_SPLATINSERT46]], <2 x i32> poison, <2 x i32> zeroinitializer
// CHECK:   [[SH_PROM48:%.*]] = zext <2 x i32> [[SPLAT_SPLAT47]] to <2 x i64>
// CHECK:   [[SHR49:%.*]] = lshr <2 x i64> [[TMP53]], [[SH_PROM48]]
// CHECK:   store volatile <2 x i64> [[SHR49]], ptr @ul, align 8
// CHECK:   [[TMP55:%.*]] = load volatile <2 x i64>, ptr @ul, align 8
// CHECK:   [[SHR50:%.*]] = lshr <2 x i64> [[TMP55]], <i64 5, i64 5>
// CHECK:   store volatile <2 x i64> [[SHR50]], ptr @ul, align 8
// CHECK:   ret void
void test_sr(void) {

  sc = sc >> sc2;
  sc = sc >> uc2;
  sc = sc >> cnt;
  sc = sc >> 5;
  uc = uc >> sc2;
  uc = uc >> uc2;
  uc = uc >> cnt;
  uc = uc >> 5;

  ss = ss >> ss2;
  ss = ss >> us2;
  ss = ss >> cnt;
  ss = ss >> 5;
  us = us >> ss2;
  us = us >> us2;
  us = us >> cnt;
  us = us >> 5;

  si = si >> si2;
  si = si >> ui2;
  si = si >> cnt;
  si = si >> 5;
  ui = ui >> si2;
  ui = ui >> ui2;
  ui = ui >> cnt;
  ui = ui >> 5;

  sl = sl >> sl2;
  sl = sl >> ul2;
  sl = sl >> cnt;
  sl = sl >> 5;
  ul = ul >> sl2;
  ul = ul >> ul2;
  ul = ul >> cnt;
  ul = ul >> 5;
}

// CHECK-LABEL: define{{.*}} void @test_sr_assign() #0 {
// CHECK:   [[TMP0:%.*]] = load volatile <16 x i8>, ptr @sc2, align 8
// CHECK:   [[TMP1:%.*]] = load volatile <16 x i8>, ptr @sc, align 8
// CHECK:   [[SHR:%.*]] = ashr <16 x i8> [[TMP1]], [[TMP0]]
// CHECK:   store volatile <16 x i8> [[SHR]], ptr @sc, align 8
// CHECK:   [[TMP2:%.*]] = load volatile <16 x i8>, ptr @uc2, align 8
// CHECK:   [[TMP3:%.*]] = load volatile <16 x i8>, ptr @sc, align 8
// CHECK:   [[SHR1:%.*]] = ashr <16 x i8> [[TMP3]], [[TMP2]]
// CHECK:   store volatile <16 x i8> [[SHR1]], ptr @sc, align 8
// CHECK:   [[TMP4:%.*]] = load volatile i32, ptr @cnt, align 4
// CHECK:   [[SPLAT_SPLATINSERT:%.*]] = insertelement <16 x i32> poison, i32 [[TMP4]], i32 0
// CHECK:   [[SPLAT_SPLAT:%.*]] = shufflevector <16 x i32> [[SPLAT_SPLAT:%.*]]insert, <16 x i32> poison, <16 x i32> zeroinitializer
// CHECK:   [[TMP5:%.*]] = load volatile <16 x i8>, ptr @sc, align 8
// CHECK:   [[SH_PROM:%.*]] = trunc <16 x i32> [[SPLAT_SPLAT]] to <16 x i8>
// CHECK:   [[SHR2:%.*]] = ashr <16 x i8> [[TMP5]], [[SH_PROM]]
// CHECK:   store volatile <16 x i8> [[SHR2]], ptr @sc, align 8
// CHECK:   [[TMP6:%.*]] = load volatile <16 x i8>, ptr @sc, align 8
// CHECK:   [[SHR3:%.*]] = ashr <16 x i8> [[TMP6]], <i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5>
// CHECK:   store volatile <16 x i8> [[SHR3]], ptr @sc, align 8
// CHECK:   [[TMP7:%.*]] = load volatile <16 x i8>, ptr @sc2, align 8
// CHECK:   [[TMP8:%.*]] = load volatile <16 x i8>, ptr @uc, align 8
// CHECK:   [[SHR4:%.*]] = lshr <16 x i8> [[TMP8]], [[TMP7]]
// CHECK:   store volatile <16 x i8> [[SHR4]], ptr @uc, align 8
// CHECK:   [[TMP9:%.*]] = load volatile <16 x i8>, ptr @uc2, align 8
// CHECK:   [[TMP10:%.*]] = load volatile <16 x i8>, ptr @uc, align 8
// CHECK:   [[SHR5:%.*]] = lshr <16 x i8> [[TMP10]], [[TMP9]]
// CHECK:   store volatile <16 x i8> [[SHR5]], ptr @uc, align 8
// CHECK:   [[TMP11:%.*]] = load volatile i32, ptr @cnt, align 4
// CHECK:   [[SPLAT_SPLATINSERT6:%.*]] = insertelement <16 x i32> poison, i32 [[TMP11]], i32 0
// CHECK:   [[SPLAT_SPLAT7:%.*]] = shufflevector <16 x i32> [[SPLAT_SPLATINSERT6]], <16 x i32> poison, <16 x i32> zeroinitializer
// CHECK:   [[TMP12:%.*]] = load volatile <16 x i8>, ptr @uc, align 8
// CHECK:   [[SH_PROM8:%.*]] = trunc <16 x i32> [[SPLAT_SPLAT7]] to <16 x i8>
// CHECK:   [[SHR9:%.*]] = lshr <16 x i8> [[TMP12]], [[SH_PROM8]]
// CHECK:   store volatile <16 x i8> [[SHR9]], ptr @uc, align 8
// CHECK:   [[TMP13:%.*]] = load volatile <16 x i8>, ptr @uc, align 8
// CHECK:   [[SHR10:%.*]] = lshr <16 x i8> [[TMP13]], <i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5>
// CHECK:   store volatile <16 x i8> [[SHR10]], ptr @uc, align 8
// CHECK:   [[TMP14:%.*]] = load volatile <8 x i16>, ptr @ss2, align 8
// CHECK:   [[TMP15:%.*]] = load volatile <8 x i16>, ptr @ss, align 8
// CHECK:   [[SHR11:%.*]] = ashr <8 x i16> [[TMP15]], [[TMP14]]
// CHECK:   store volatile <8 x i16> [[SHR11]], ptr @ss, align 8
// CHECK:   [[TMP16:%.*]] = load volatile <8 x i16>, ptr @us2, align 8
// CHECK:   [[TMP17:%.*]] = load volatile <8 x i16>, ptr @ss, align 8
// CHECK:   [[SHR12:%.*]] = ashr <8 x i16> [[TMP17]], [[TMP16]]
// CHECK:   store volatile <8 x i16> [[SHR12]], ptr @ss, align 8
// CHECK:   [[TMP18:%.*]] = load volatile i32, ptr @cnt, align 4
// CHECK:   [[SPLAT_SPLATINSERT13:%.*]] = insertelement <8 x i32> poison, i32 [[TMP18]], i32 0
// CHECK:   [[SPLAT_SPLAT14:%.*]] = shufflevector <8 x i32> [[SPLAT_SPLATINSERT13]], <8 x i32> poison, <8 x i32> zeroinitializer
// CHECK:   [[TMP19:%.*]] = load volatile <8 x i16>, ptr @ss, align 8
// CHECK:   [[SH_PROM15:%.*]] = trunc <8 x i32> [[SPLAT_SPLAT14]] to <8 x i16>
// CHECK:   [[SHR16:%.*]] = ashr <8 x i16> [[TMP19]], [[SH_PROM15]]
// CHECK:   store volatile <8 x i16> [[SHR16]], ptr @ss, align 8
// CHECK:   [[TMP20:%.*]] = load volatile <8 x i16>, ptr @ss, align 8
// CHECK:   [[SHR17:%.*]] = ashr <8 x i16> [[TMP20]], <i16 5, i16 5, i16 5, i16 5, i16 5, i16 5, i16 5, i16 5>
// CHECK:   store volatile <8 x i16> [[SHR17]], ptr @ss, align 8
// CHECK:   [[TMP21:%.*]] = load volatile <8 x i16>, ptr @ss2, align 8
// CHECK:   [[TMP22:%.*]] = load volatile <8 x i16>, ptr @us, align 8
// CHECK:   [[SHR18:%.*]] = lshr <8 x i16> [[TMP22]], [[TMP21]]
// CHECK:   store volatile <8 x i16> [[SHR18]], ptr @us, align 8
// CHECK:   [[TMP23:%.*]] = load volatile <8 x i16>, ptr @us2, align 8
// CHECK:   [[TMP24:%.*]] = load volatile <8 x i16>, ptr @us, align 8
// CHECK:   [[SHR19:%.*]] = lshr <8 x i16> [[TMP24]], [[TMP23]]
// CHECK:   store volatile <8 x i16> [[SHR19]], ptr @us, align 8
// CHECK:   [[TMP25:%.*]] = load volatile i32, ptr @cnt, align 4
// CHECK:   [[SPLAT_SPLATINSERT20:%.*]] = insertelement <8 x i32> poison, i32 [[TMP25]], i32 0
// CHECK:   [[SPLAT_SPLAT21:%.*]] = shufflevector <8 x i32> [[SPLAT_SPLATINSERT20]], <8 x i32> poison, <8 x i32> zeroinitializer
// CHECK:   [[TMP26:%.*]] = load volatile <8 x i16>, ptr @us, align 8
// CHECK:   [[SH_PROM22:%.*]] = trunc <8 x i32> [[SPLAT_SPLAT21]] to <8 x i16>
// CHECK:   [[SHR23:%.*]] = lshr <8 x i16> [[TMP26]], [[SH_PROM22]]
// CHECK:   store volatile <8 x i16> [[SHR23]], ptr @us, align 8
// CHECK:   [[TMP27:%.*]] = load volatile <8 x i16>, ptr @us, align 8
// CHECK:   [[SHR24:%.*]] = lshr <8 x i16> [[TMP27]], <i16 5, i16 5, i16 5, i16 5, i16 5, i16 5, i16 5, i16 5>
// CHECK:   store volatile <8 x i16> [[SHR24]], ptr @us, align 8
// CHECK:   [[TMP28:%.*]] = load volatile <4 x i32>, ptr @si2, align 8
// CHECK:   [[TMP29:%.*]] = load volatile <4 x i32>, ptr @si, align 8
// CHECK:   [[SHR25:%.*]] = ashr <4 x i32> [[TMP29]], [[TMP28]]
// CHECK:   store volatile <4 x i32> [[SHR25]], ptr @si, align 8
// CHECK:   [[TMP30:%.*]] = load volatile <4 x i32>, ptr @ui2, align 8
// CHECK:   [[TMP31:%.*]] = load volatile <4 x i32>, ptr @si, align 8
// CHECK:   [[SHR26:%.*]] = ashr <4 x i32> [[TMP31]], [[TMP30]]
// CHECK:   store volatile <4 x i32> [[SHR26]], ptr @si, align 8
// CHECK:   [[TMP32:%.*]] = load volatile i32, ptr @cnt, align 4
// CHECK:   [[SPLAT_SPLATINSERT27:%.*]] = insertelement <4 x i32> poison, i32 [[TMP32]], i32 0
// CHECK:   [[SPLAT_SPLAT28:%.*]] = shufflevector <4 x i32> [[SPLAT_SPLATINSERT27]], <4 x i32> poison, <4 x i32> zeroinitializer
// CHECK:   [[TMP33:%.*]] = load volatile <4 x i32>, ptr @si, align 8
// CHECK:   [[SHR29:%.*]] = ashr <4 x i32> [[TMP33]], [[SPLAT_SPLAT28]]
// CHECK:   store volatile <4 x i32> [[SHR29]], ptr @si, align 8
// CHECK:   [[TMP34:%.*]] = load volatile <4 x i32>, ptr @si, align 8
// CHECK:   [[SHR30:%.*]] = ashr <4 x i32> [[TMP34]], <i32 5, i32 5, i32 5, i32 5>
// CHECK:   store volatile <4 x i32> [[SHR30]], ptr @si, align 8
// CHECK:   [[TMP35:%.*]] = load volatile <4 x i32>, ptr @si2, align 8
// CHECK:   [[TMP36:%.*]] = load volatile <4 x i32>, ptr @ui, align 8
// CHECK:   [[SHR31:%.*]] = lshr <4 x i32> [[TMP36]], [[TMP35]]
// CHECK:   store volatile <4 x i32> [[SHR31]], ptr @ui, align 8
// CHECK:   [[TMP37:%.*]] = load volatile <4 x i32>, ptr @ui2, align 8
// CHECK:   [[TMP38:%.*]] = load volatile <4 x i32>, ptr @ui, align 8
// CHECK:   [[SHR32:%.*]] = lshr <4 x i32> [[TMP38]], [[TMP37]]
// CHECK:   store volatile <4 x i32> [[SHR32]], ptr @ui, align 8
// CHECK:   [[TMP39:%.*]] = load volatile i32, ptr @cnt, align 4
// CHECK:   [[SPLAT_SPLATINSERT33:%.*]] = insertelement <4 x i32> poison, i32 [[TMP39]], i32 0
// CHECK:   [[SPLAT_SPLAT34:%.*]] = shufflevector <4 x i32> [[SPLAT_SPLATINSERT33]], <4 x i32> poison, <4 x i32> zeroinitializer
// CHECK:   [[TMP40:%.*]] = load volatile <4 x i32>, ptr @ui, align 8
// CHECK:   [[SHR35:%.*]] = lshr <4 x i32> [[TMP40]], [[SPLAT_SPLAT34]]
// CHECK:   store volatile <4 x i32> [[SHR35]], ptr @ui, align 8
// CHECK:   [[TMP41:%.*]] = load volatile <4 x i32>, ptr @ui, align 8
// CHECK:   [[SHR36:%.*]] = lshr <4 x i32> [[TMP41]], <i32 5, i32 5, i32 5, i32 5>
// CHECK:   store volatile <4 x i32> [[SHR36]], ptr @ui, align 8
// CHECK:   [[TMP42:%.*]] = load volatile <2 x i64>, ptr @sl2, align 8
// CHECK:   [[TMP43:%.*]] = load volatile <2 x i64>, ptr @sl, align 8
// CHECK:   [[SHR37:%.*]] = ashr <2 x i64> [[TMP43]], [[TMP42]]
// CHECK:   store volatile <2 x i64> [[SHR37]], ptr @sl, align 8
// CHECK:   [[TMP44:%.*]] = load volatile <2 x i64>, ptr @ul2, align 8
// CHECK:   [[TMP45:%.*]] = load volatile <2 x i64>, ptr @sl, align 8
// CHECK:   [[SHR38:%.*]] = ashr <2 x i64> [[TMP45]], [[TMP44]]
// CHECK:   store volatile <2 x i64> [[SHR38]], ptr @sl, align 8
// CHECK:   [[TMP46:%.*]] = load volatile i32, ptr @cnt, align 4
// CHECK:   [[SPLAT_SPLATINSERT39:%.*]] = insertelement <2 x i32> poison, i32 [[TMP46]], i32 0
// CHECK:   [[SPLAT_SPLAT40:%.*]] = shufflevector <2 x i32> [[SPLAT_SPLATINSERT39]], <2 x i32> poison, <2 x i32> zeroinitializer
// CHECK:   [[TMP47:%.*]] = load volatile <2 x i64>, ptr @sl, align 8
// CHECK:   [[SH_PROM41:%.*]] = zext <2 x i32> [[SPLAT_SPLAT40]] to <2 x i64>
// CHECK:   [[SHR42:%.*]] = ashr <2 x i64> [[TMP47]], [[SH_PROM41]]
// CHECK:   store volatile <2 x i64> [[SHR42]], ptr @sl, align 8
// CHECK:   [[TMP48:%.*]] = load volatile <2 x i64>, ptr @sl, align 8
// CHECK:   [[SHR43:%.*]] = ashr <2 x i64> [[TMP48]], <i64 5, i64 5>
// CHECK:   store volatile <2 x i64> [[SHR43]], ptr @sl, align 8
// CHECK:   [[TMP49:%.*]] = load volatile <2 x i64>, ptr @sl2, align 8
// CHECK:   [[TMP50:%.*]] = load volatile <2 x i64>, ptr @ul, align 8
// CHECK:   [[SHR44:%.*]] = lshr <2 x i64> [[TMP50]], [[TMP49]]
// CHECK:   store volatile <2 x i64> [[SHR44]], ptr @ul, align 8
// CHECK:   [[TMP51:%.*]] = load volatile <2 x i64>, ptr @ul2, align 8
// CHECK:   [[TMP52:%.*]] = load volatile <2 x i64>, ptr @ul, align 8
// CHECK:   [[SHR45:%.*]] = lshr <2 x i64> [[TMP52]], [[TMP51]]
// CHECK:   store volatile <2 x i64> [[SHR45]], ptr @ul, align 8
// CHECK:   [[TMP53:%.*]] = load volatile i32, ptr @cnt, align 4
// CHECK:   [[SPLAT_SPLATINSERT46:%.*]] = insertelement <2 x i32> poison, i32 [[TMP53]], i32 0
// CHECK:   [[SPLAT_SPLAT47:%.*]] = shufflevector <2 x i32> [[SPLAT_SPLATINSERT46]], <2 x i32> poison, <2 x i32> zeroinitializer
// CHECK:   [[TMP54:%.*]] = load volatile <2 x i64>, ptr @ul, align 8
// CHECK:   [[SH_PROM48:%.*]] = zext <2 x i32> [[SPLAT_SPLAT47]] to <2 x i64>
// CHECK:   [[SHR49:%.*]] = lshr <2 x i64> [[TMP54]], [[SH_PROM48]]
// CHECK:   store volatile <2 x i64> [[SHR49]], ptr @ul, align 8
// CHECK:   [[TMP55:%.*]] = load volatile <2 x i64>, ptr @ul, align 8
// CHECK:   [[SHR50:%.*]] = lshr <2 x i64> [[TMP55]], <i64 5, i64 5>
// CHECK:   store volatile <2 x i64> [[SHR50]], ptr @ul, align 8
// CHECK:   ret void
void test_sr_assign(void) {

  sc >>= sc2;
  sc >>= uc2;
  sc >>= cnt;
  sc >>= 5;
  uc >>= sc2;
  uc >>= uc2;
  uc >>= cnt;
  uc >>= 5;

  ss >>= ss2;
  ss >>= us2;
  ss >>= cnt;
  ss >>= 5;
  us >>= ss2;
  us >>= us2;
  us >>= cnt;
  us >>= 5;

  si >>= si2;
  si >>= ui2;
  si >>= cnt;
  si >>= 5;
  ui >>= si2;
  ui >>= ui2;
  ui >>= cnt;
  ui >>= 5;

  sl >>= sl2;
  sl >>= ul2;
  sl >>= cnt;
  sl >>= 5;
  ul >>= sl2;
  ul >>= ul2;
  ul >>= cnt;
  ul >>= 5;
}


// CHECK-LABEL: define{{.*}} void @test_cmpeq() #0 {
// CHECK:   [[TMP0:%.*]] = load volatile <16 x i8>, ptr @sc, align 8
// CHECK:   [[TMP1:%.*]] = load volatile <16 x i8>, ptr @sc2, align 8
// CHECK:   [[CMP:%.*]] = icmp eq <16 x i8> [[TMP0]], [[TMP1]]
// CHECK:   [[SEXT:%.*]] = sext <16 x i1> [[CMP]] to <16 x i8>
// CHECK:   store volatile <16 x i8> [[SEXT]], ptr @bc, align 8
// CHECK:   [[TMP2:%.*]] = load volatile <16 x i8>, ptr @sc, align 8
// CHECK:   [[TMP3:%.*]] = load volatile <16 x i8>, ptr @bc2, align 8
// CHECK:   [[CMP1:%.*]] = icmp eq <16 x i8> [[TMP2]], [[TMP3]]
// CHECK:   [[SEXT2:%.*]] = sext <16 x i1> [[CMP1]] to <16 x i8>
// CHECK:   store volatile <16 x i8> [[SEXT2]], ptr @bc, align 8
// CHECK:   [[TMP4:%.*]] = load volatile <16 x i8>, ptr @bc, align 8
// CHECK:   [[TMP5:%.*]] = load volatile <16 x i8>, ptr @sc2, align 8
// CHECK:   [[CMP3:%.*]] = icmp eq <16 x i8> [[TMP4]], [[TMP5]]
// CHECK:   [[SEXT4:%.*]] = sext <16 x i1> [[CMP3]] to <16 x i8>
// CHECK:   store volatile <16 x i8> [[SEXT4]], ptr @bc, align 8
// CHECK:   [[TMP6:%.*]] = load volatile <16 x i8>, ptr @uc, align 8
// CHECK:   [[TMP7:%.*]] = load volatile <16 x i8>, ptr @uc2, align 8
// CHECK:   [[CMP5:%.*]] = icmp eq <16 x i8> [[TMP6]], [[TMP7]]
// CHECK:   [[SEXT6:%.*]] = sext <16 x i1> [[CMP5]] to <16 x i8>
// CHECK:   store volatile <16 x i8> [[SEXT6]], ptr @bc, align 8
// CHECK:   [[TMP8:%.*]] = load volatile <16 x i8>, ptr @uc, align 8
// CHECK:   [[TMP9:%.*]] = load volatile <16 x i8>, ptr @bc2, align 8
// CHECK:   [[CMP7:%.*]] = icmp eq <16 x i8> [[TMP8]], [[TMP9]]
// CHECK:   [[SEXT8:%.*]] = sext <16 x i1> [[CMP7]] to <16 x i8>
// CHECK:   store volatile <16 x i8> [[SEXT8]], ptr @bc, align 8
// CHECK:   [[TMP10:%.*]] = load volatile <16 x i8>, ptr @bc, align 8
// CHECK:   [[TMP11:%.*]] = load volatile <16 x i8>, ptr @uc2, align 8
// CHECK:   [[CMP9:%.*]] = icmp eq <16 x i8> [[TMP10]], [[TMP11]]
// CHECK:   [[SEXT10:%.*]] = sext <16 x i1> [[CMP9]] to <16 x i8>
// CHECK:   store volatile <16 x i8> [[SEXT10]], ptr @bc, align 8
// CHECK:   [[TMP12:%.*]] = load volatile <16 x i8>, ptr @bc, align 8
// CHECK:   [[TMP13:%.*]] = load volatile <16 x i8>, ptr @bc2, align 8
// CHECK:   [[CMP11:%.*]] = icmp eq <16 x i8> [[TMP12]], [[TMP13]]
// CHECK:   [[SEXT12:%.*]] = sext <16 x i1> [[CMP11]] to <16 x i8>
// CHECK:   store volatile <16 x i8> [[SEXT12]], ptr @bc, align 8
// CHECK:   [[TMP14:%.*]] = load volatile <8 x i16>, ptr @ss, align 8
// CHECK:   [[TMP15:%.*]] = load volatile <8 x i16>, ptr @ss2, align 8
// CHECK:   [[CMP13:%.*]] = icmp eq <8 x i16> [[TMP14]], [[TMP15]]
// CHECK:   [[SEXT14:%.*]] = sext <8 x i1> [[CMP13]] to <8 x i16>
// CHECK:   store volatile <8 x i16> [[SEXT14]], ptr @bs, align 8
// CHECK:   [[TMP16:%.*]] = load volatile <8 x i16>, ptr @ss, align 8
// CHECK:   [[TMP17:%.*]] = load volatile <8 x i16>, ptr @bs2, align 8
// CHECK:   [[CMP15:%.*]] = icmp eq <8 x i16> [[TMP16]], [[TMP17]]
// CHECK:   [[SEXT16:%.*]] = sext <8 x i1> [[CMP15]] to <8 x i16>
// CHECK:   store volatile <8 x i16> [[SEXT16]], ptr @bs, align 8
// CHECK:   [[TMP18:%.*]] = load volatile <8 x i16>, ptr @bs, align 8
// CHECK:   [[TMP19:%.*]] = load volatile <8 x i16>, ptr @ss2, align 8
// CHECK:   [[CMP17:%.*]] = icmp eq <8 x i16> [[TMP18]], [[TMP19]]
// CHECK:   [[SEXT18:%.*]] = sext <8 x i1> [[CMP17]] to <8 x i16>
// CHECK:   store volatile <8 x i16> [[SEXT18]], ptr @bs, align 8
// CHECK:   [[TMP20:%.*]] = load volatile <8 x i16>, ptr @us, align 8
// CHECK:   [[TMP21:%.*]] = load volatile <8 x i16>, ptr @us2, align 8
// CHECK:   [[CMP19:%.*]] = icmp eq <8 x i16> [[TMP20]], [[TMP21]]
// CHECK:   [[SEXT20:%.*]] = sext <8 x i1> [[CMP19]] to <8 x i16>
// CHECK:   store volatile <8 x i16> [[SEXT20]], ptr @bs, align 8
// CHECK:   [[TMP22:%.*]] = load volatile <8 x i16>, ptr @us, align 8
// CHECK:   [[TMP23:%.*]] = load volatile <8 x i16>, ptr @bs2, align 8
// CHECK:   [[CMP21:%.*]] = icmp eq <8 x i16> [[TMP22]], [[TMP23]]
// CHECK:   [[SEXT22:%.*]] = sext <8 x i1> [[CMP21]] to <8 x i16>
// CHECK:   store volatile <8 x i16> [[SEXT22]], ptr @bs, align 8
// CHECK:   [[TMP24:%.*]] = load volatile <8 x i16>, ptr @bs, align 8
// CHECK:   [[TMP25:%.*]] = load volatile <8 x i16>, ptr @us2, align 8
// CHECK:   [[CMP23:%.*]] = icmp eq <8 x i16> [[TMP24]], [[TMP25]]
// CHECK:   [[SEXT24:%.*]] = sext <8 x i1> [[CMP23]] to <8 x i16>
// CHECK:   store volatile <8 x i16> [[SEXT24]], ptr @bs, align 8
// CHECK:   [[TMP26:%.*]] = load volatile <8 x i16>, ptr @bs, align 8
// CHECK:   [[TMP27:%.*]] = load volatile <8 x i16>, ptr @bs2, align 8
// CHECK:   [[CMP25:%.*]] = icmp eq <8 x i16> [[TMP26]], [[TMP27]]
// CHECK:   [[SEXT26:%.*]] = sext <8 x i1> [[CMP25]] to <8 x i16>
// CHECK:   store volatile <8 x i16> [[SEXT26]], ptr @bs, align 8
// CHECK:   [[TMP28:%.*]] = load volatile <4 x i32>, ptr @si, align 8
// CHECK:   [[TMP29:%.*]] = load volatile <4 x i32>, ptr @si2, align 8
// CHECK:   [[CMP27:%.*]] = icmp eq <4 x i32> [[TMP28]], [[TMP29]]
// CHECK:   [[SEXT28:%.*]] = sext <4 x i1> [[CMP27]] to <4 x i32>
// CHECK:   store volatile <4 x i32> [[SEXT28]], ptr @bi, align 8
// CHECK:   [[TMP30:%.*]] = load volatile <4 x i32>, ptr @si, align 8
// CHECK:   [[TMP31:%.*]] = load volatile <4 x i32>, ptr @bi2, align 8
// CHECK:   [[CMP29:%.*]] = icmp eq <4 x i32> [[TMP30]], [[TMP31]]
// CHECK:   [[SEXT30:%.*]] = sext <4 x i1> [[CMP29]] to <4 x i32>
// CHECK:   store volatile <4 x i32> [[SEXT30]], ptr @bi, align 8
// CHECK:   [[TMP32:%.*]] = load volatile <4 x i32>, ptr @bi, align 8
// CHECK:   [[TMP33:%.*]] = load volatile <4 x i32>, ptr @si2, align 8
// CHECK:   [[CMP31:%.*]] = icmp eq <4 x i32> [[TMP32]], [[TMP33]]
// CHECK:   [[SEXT32:%.*]] = sext <4 x i1> [[CMP31]] to <4 x i32>
// CHECK:   store volatile <4 x i32> [[SEXT32]], ptr @bi, align 8
// CHECK:   [[TMP34:%.*]] = load volatile <4 x i32>, ptr @ui, align 8
// CHECK:   [[TMP35:%.*]] = load volatile <4 x i32>, ptr @ui2, align 8
// CHECK:   [[CMP33:%.*]] = icmp eq <4 x i32> [[TMP34]], [[TMP35]]
// CHECK:   [[SEXT34:%.*]] = sext <4 x i1> [[CMP33]] to <4 x i32>
// CHECK:   store volatile <4 x i32> [[SEXT34]], ptr @bi, align 8
// CHECK:   [[TMP36:%.*]] = load volatile <4 x i32>, ptr @ui, align 8
// CHECK:   [[TMP37:%.*]] = load volatile <4 x i32>, ptr @bi2, align 8
// CHECK:   [[CMP35:%.*]] = icmp eq <4 x i32> [[TMP36]], [[TMP37]]
// CHECK:   [[SEXT36:%.*]] = sext <4 x i1> [[CMP35]] to <4 x i32>
// CHECK:   store volatile <4 x i32> [[SEXT36]], ptr @bi, align 8
// CHECK:   [[TMP38:%.*]] = load volatile <4 x i32>, ptr @bi, align 8
// CHECK:   [[TMP39:%.*]] = load volatile <4 x i32>, ptr @ui2, align 8
// CHECK:   [[CMP37:%.*]] = icmp eq <4 x i32> [[TMP38]], [[TMP39]]
// CHECK:   [[SEXT38:%.*]] = sext <4 x i1> [[CMP37]] to <4 x i32>
// CHECK:   store volatile <4 x i32> [[SEXT38]], ptr @bi, align 8
// CHECK:   [[TMP40:%.*]] = load volatile <4 x i32>, ptr @bi, align 8
// CHECK:   [[TMP41:%.*]] = load volatile <4 x i32>, ptr @bi2, align 8
// CHECK:   [[CMP39:%.*]] = icmp eq <4 x i32> [[TMP40]], [[TMP41]]
// CHECK:   [[SEXT40:%.*]] = sext <4 x i1> [[CMP39]] to <4 x i32>
// CHECK:   store volatile <4 x i32> [[SEXT40]], ptr @bi, align 8
// CHECK:   [[TMP42:%.*]] = load volatile <2 x i64>, ptr @sl, align 8
// CHECK:   [[TMP43:%.*]] = load volatile <2 x i64>, ptr @sl2, align 8
// CHECK:   [[CMP41:%.*]] = icmp eq <2 x i64> [[TMP42]], [[TMP43]]
// CHECK:   [[SEXT42:%.*]] = sext <2 x i1> [[CMP41]] to <2 x i64>
// CHECK:   store volatile <2 x i64> [[SEXT42]], ptr @bl, align 8
// CHECK:   [[TMP44:%.*]] = load volatile <2 x i64>, ptr @sl, align 8
// CHECK:   [[TMP45:%.*]] = load volatile <2 x i64>, ptr @bl2, align 8
// CHECK:   [[CMP43:%.*]] = icmp eq <2 x i64> [[TMP44]], [[TMP45]]
// CHECK:   [[SEXT44:%.*]] = sext <2 x i1> [[CMP43]] to <2 x i64>
// CHECK:   store volatile <2 x i64> [[SEXT44]], ptr @bl, align 8
// CHECK:   [[TMP46:%.*]] = load volatile <2 x i64>, ptr @bl, align 8
// CHECK:   [[TMP47:%.*]] = load volatile <2 x i64>, ptr @sl2, align 8
// CHECK:   [[CMP45:%.*]] = icmp eq <2 x i64> [[TMP46]], [[TMP47]]
// CHECK:   [[SEXT46:%.*]] = sext <2 x i1> [[CMP45]] to <2 x i64>
// CHECK:   store volatile <2 x i64> [[SEXT46]], ptr @bl, align 8
// CHECK:   [[TMP48:%.*]] = load volatile <2 x i64>, ptr @ul, align 8
// CHECK:   [[TMP49:%.*]] = load volatile <2 x i64>, ptr @ul2, align 8
// CHECK:   [[CMP47:%.*]] = icmp eq <2 x i64> [[TMP48]], [[TMP49]]
// CHECK:   [[SEXT48:%.*]] = sext <2 x i1> [[CMP47]] to <2 x i64>
// CHECK:   store volatile <2 x i64> [[SEXT48]], ptr @bl, align 8
// CHECK:   [[TMP50:%.*]] = load volatile <2 x i64>, ptr @ul, align 8
// CHECK:   [[TMP51:%.*]] = load volatile <2 x i64>, ptr @bl2, align 8
// CHECK:   [[CMP49:%.*]] = icmp eq <2 x i64> [[TMP50]], [[TMP51]]
// CHECK:   [[SEXT50:%.*]] = sext <2 x i1> [[CMP49]] to <2 x i64>
// CHECK:   store volatile <2 x i64> [[SEXT50]], ptr @bl, align 8
// CHECK:   [[TMP52:%.*]] = load volatile <2 x i64>, ptr @bl, align 8
// CHECK:   [[TMP53:%.*]] = load volatile <2 x i64>, ptr @ul2, align 8
// CHECK:   [[CMP51:%.*]] = icmp eq <2 x i64> [[TMP52]], [[TMP53]]
// CHECK:   [[SEXT52:%.*]] = sext <2 x i1> [[CMP51]] to <2 x i64>
// CHECK:   store volatile <2 x i64> [[SEXT52]], ptr @bl, align 8
// CHECK:   [[TMP54:%.*]] = load volatile <2 x i64>, ptr @bl, align 8
// CHECK:   [[TMP55:%.*]] = load volatile <2 x i64>, ptr @bl2, align 8
// CHECK:   [[CMP53:%.*]] = icmp eq <2 x i64> [[TMP54]], [[TMP55]]
// CHECK:   [[SEXT54:%.*]] = sext <2 x i1> [[CMP53]] to <2 x i64>
// CHECK:   store volatile <2 x i64> [[SEXT54]], ptr @bl, align 8
// CHECK:   [[TMP56:%.*]] = load volatile <2 x double>, ptr @fd, align 8
// CHECK:   [[TMP57:%.*]] = load volatile <2 x double>, ptr @fd2, align 8
// CHECK:   [[CMP55:%.*]] = fcmp oeq <2 x double> [[TMP56]], [[TMP57]]
// CHECK:   [[SEXT56:%.*]] = sext <2 x i1> [[CMP55]] to <2 x i64>
// CHECK:   store volatile <2 x i64> [[SEXT56]], ptr @bl, align 8
// CHECK:   ret void
void test_cmpeq(void) {

  bc = sc == sc2;
  bc = sc == bc2;
  bc = bc == sc2;
  bc = uc == uc2;
  bc = uc == bc2;
  bc = bc == uc2;
  bc = bc == bc2;

  bs = ss == ss2;
  bs = ss == bs2;
  bs = bs == ss2;
  bs = us == us2;
  bs = us == bs2;
  bs = bs == us2;
  bs = bs == bs2;

  bi = si == si2;
  bi = si == bi2;
  bi = bi == si2;
  bi = ui == ui2;
  bi = ui == bi2;
  bi = bi == ui2;
  bi = bi == bi2;

  bl = sl == sl2;
  bl = sl == bl2;
  bl = bl == sl2;
  bl = ul == ul2;
  bl = ul == bl2;
  bl = bl == ul2;
  bl = bl == bl2;

  bl = fd == fd2;
}

// CHECK-LABEL: define{{.*}} void @test_cmpne() #0 {
// CHECK:   [[TMP0:%.*]] = load volatile <16 x i8>, ptr @sc, align 8
// CHECK:   [[TMP1:%.*]] = load volatile <16 x i8>, ptr @sc2, align 8
// CHECK:   [[CMP:%.*]] = icmp ne <16 x i8> [[TMP0]], [[TMP1]]
// CHECK:   [[SEXT:%.*]] = sext <16 x i1> [[CMP]] to <16 x i8>
// CHECK:   store volatile <16 x i8> [[SEXT]], ptr @bc, align 8
// CHECK:   [[TMP2:%.*]] = load volatile <16 x i8>, ptr @sc, align 8
// CHECK:   [[TMP3:%.*]] = load volatile <16 x i8>, ptr @bc2, align 8
// CHECK:   [[CMP1:%.*]] = icmp ne <16 x i8> [[TMP2]], [[TMP3]]
// CHECK:   [[SEXT2:%.*]] = sext <16 x i1> [[CMP1]] to <16 x i8>
// CHECK:   store volatile <16 x i8> [[SEXT2]], ptr @bc, align 8
// CHECK:   [[TMP4:%.*]] = load volatile <16 x i8>, ptr @bc, align 8
// CHECK:   [[TMP5:%.*]] = load volatile <16 x i8>, ptr @sc2, align 8
// CHECK:   [[CMP3:%.*]] = icmp ne <16 x i8> [[TMP4]], [[TMP5]]
// CHECK:   [[SEXT4:%.*]] = sext <16 x i1> [[CMP3]] to <16 x i8>
// CHECK:   store volatile <16 x i8> [[SEXT4]], ptr @bc, align 8
// CHECK:   [[TMP6:%.*]] = load volatile <16 x i8>, ptr @uc, align 8
// CHECK:   [[TMP7:%.*]] = load volatile <16 x i8>, ptr @uc2, align 8
// CHECK:   [[CMP5:%.*]] = icmp ne <16 x i8> [[TMP6]], [[TMP7]]
// CHECK:   [[SEXT6:%.*]] = sext <16 x i1> [[CMP5]] to <16 x i8>
// CHECK:   store volatile <16 x i8> [[SEXT6]], ptr @bc, align 8
// CHECK:   [[TMP8:%.*]] = load volatile <16 x i8>, ptr @uc, align 8
// CHECK:   [[TMP9:%.*]] = load volatile <16 x i8>, ptr @bc2, align 8
// CHECK:   [[CMP7:%.*]] = icmp ne <16 x i8> [[TMP8]], [[TMP9]]
// CHECK:   [[SEXT8:%.*]] = sext <16 x i1> [[CMP7]] to <16 x i8>
// CHECK:   store volatile <16 x i8> [[SEXT8]], ptr @bc, align 8
// CHECK:   [[TMP10:%.*]] = load volatile <16 x i8>, ptr @bc, align 8
// CHECK:   [[TMP11:%.*]] = load volatile <16 x i8>, ptr @uc2, align 8
// CHECK:   [[CMP9:%.*]] = icmp ne <16 x i8> [[TMP10]], [[TMP11]]
// CHECK:   [[SEXT10:%.*]] = sext <16 x i1> [[CMP9]] to <16 x i8>
// CHECK:   store volatile <16 x i8> [[SEXT10]], ptr @bc, align 8
// CHECK:   [[TMP12:%.*]] = load volatile <16 x i8>, ptr @bc, align 8
// CHECK:   [[TMP13:%.*]] = load volatile <16 x i8>, ptr @bc2, align 8
// CHECK:   [[CMP11:%.*]] = icmp ne <16 x i8> [[TMP12]], [[TMP13]]
// CHECK:   [[SEXT12:%.*]] = sext <16 x i1> [[CMP11]] to <16 x i8>
// CHECK:   store volatile <16 x i8> [[SEXT12]], ptr @bc, align 8
// CHECK:   [[TMP14:%.*]] = load volatile <8 x i16>, ptr @ss, align 8
// CHECK:   [[TMP15:%.*]] = load volatile <8 x i16>, ptr @ss2, align 8
// CHECK:   [[CMP13:%.*]] = icmp ne <8 x i16> [[TMP14]], [[TMP15]]
// CHECK:   [[SEXT14:%.*]] = sext <8 x i1> [[CMP13]] to <8 x i16>
// CHECK:   store volatile <8 x i16> [[SEXT14]], ptr @bs, align 8
// CHECK:   [[TMP16:%.*]] = load volatile <8 x i16>, ptr @ss, align 8
// CHECK:   [[TMP17:%.*]] = load volatile <8 x i16>, ptr @bs2, align 8
// CHECK:   [[CMP15:%.*]] = icmp ne <8 x i16> [[TMP16]], [[TMP17]]
// CHECK:   [[SEXT16:%.*]] = sext <8 x i1> [[CMP15]] to <8 x i16>
// CHECK:   store volatile <8 x i16> [[SEXT16]], ptr @bs, align 8
// CHECK:   [[TMP18:%.*]] = load volatile <8 x i16>, ptr @bs, align 8
// CHECK:   [[TMP19:%.*]] = load volatile <8 x i16>, ptr @ss2, align 8
// CHECK:   [[CMP17:%.*]] = icmp ne <8 x i16> [[TMP18]], [[TMP19]]
// CHECK:   [[SEXT18:%.*]] = sext <8 x i1> [[CMP17]] to <8 x i16>
// CHECK:   store volatile <8 x i16> [[SEXT18]], ptr @bs, align 8
// CHECK:   [[TMP20:%.*]] = load volatile <8 x i16>, ptr @us, align 8
// CHECK:   [[TMP21:%.*]] = load volatile <8 x i16>, ptr @us2, align 8
// CHECK:   [[CMP19:%.*]] = icmp ne <8 x i16> [[TMP20]], [[TMP21]]
// CHECK:   [[SEXT20:%.*]] = sext <8 x i1> [[CMP19]] to <8 x i16>
// CHECK:   store volatile <8 x i16> [[SEXT20]], ptr @bs, align 8
// CHECK:   [[TMP22:%.*]] = load volatile <8 x i16>, ptr @us, align 8
// CHECK:   [[TMP23:%.*]] = load volatile <8 x i16>, ptr @bs2, align 8
// CHECK:   [[CMP21:%.*]] = icmp ne <8 x i16> [[TMP22]], [[TMP23]]
// CHECK:   [[SEXT22:%.*]] = sext <8 x i1> [[CMP21]] to <8 x i16>
// CHECK:   store volatile <8 x i16> [[SEXT22]], ptr @bs, align 8
// CHECK:   [[TMP24:%.*]] = load volatile <8 x i16>, ptr @bs, align 8
// CHECK:   [[TMP25:%.*]] = load volatile <8 x i16>, ptr @us2, align 8
// CHECK:   [[CMP23:%.*]] = icmp ne <8 x i16> [[TMP24]], [[TMP25]]
// CHECK:   [[SEXT24:%.*]] = sext <8 x i1> [[CMP23]] to <8 x i16>
// CHECK:   store volatile <8 x i16> [[SEXT24]], ptr @bs, align 8
// CHECK:   [[TMP26:%.*]] = load volatile <8 x i16>, ptr @bs, align 8
// CHECK:   [[TMP27:%.*]] = load volatile <8 x i16>, ptr @bs2, align 8
// CHECK:   [[CMP25:%.*]] = icmp ne <8 x i16> [[TMP26]], [[TMP27]]
// CHECK:   [[SEXT26:%.*]] = sext <8 x i1> [[CMP25]] to <8 x i16>
// CHECK:   store volatile <8 x i16> [[SEXT26]], ptr @bs, align 8
// CHECK:   [[TMP28:%.*]] = load volatile <4 x i32>, ptr @si, align 8
// CHECK:   [[TMP29:%.*]] = load volatile <4 x i32>, ptr @si2, align 8
// CHECK:   [[CMP27:%.*]] = icmp ne <4 x i32> [[TMP28]], [[TMP29]]
// CHECK:   [[SEXT28:%.*]] = sext <4 x i1> [[CMP27]] to <4 x i32>
// CHECK:   store volatile <4 x i32> [[SEXT28]], ptr @bi, align 8
// CHECK:   [[TMP30:%.*]] = load volatile <4 x i32>, ptr @si, align 8
// CHECK:   [[TMP31:%.*]] = load volatile <4 x i32>, ptr @bi2, align 8
// CHECK:   [[CMP29:%.*]] = icmp ne <4 x i32> [[TMP30]], [[TMP31]]
// CHECK:   [[SEXT30:%.*]] = sext <4 x i1> [[CMP29]] to <4 x i32>
// CHECK:   store volatile <4 x i32> [[SEXT30]], ptr @bi, align 8
// CHECK:   [[TMP32:%.*]] = load volatile <4 x i32>, ptr @bi, align 8
// CHECK:   [[TMP33:%.*]] = load volatile <4 x i32>, ptr @si2, align 8
// CHECK:   [[CMP31:%.*]] = icmp ne <4 x i32> [[TMP32]], [[TMP33]]
// CHECK:   [[SEXT32:%.*]] = sext <4 x i1> [[CMP31]] to <4 x i32>
// CHECK:   store volatile <4 x i32> [[SEXT32]], ptr @bi, align 8
// CHECK:   [[TMP34:%.*]] = load volatile <4 x i32>, ptr @ui, align 8
// CHECK:   [[TMP35:%.*]] = load volatile <4 x i32>, ptr @ui2, align 8
// CHECK:   [[CMP33:%.*]] = icmp ne <4 x i32> [[TMP34]], [[TMP35]]
// CHECK:   [[SEXT34:%.*]] = sext <4 x i1> [[CMP33]] to <4 x i32>
// CHECK:   store volatile <4 x i32> [[SEXT34]], ptr @bi, align 8
// CHECK:   [[TMP36:%.*]] = load volatile <4 x i32>, ptr @ui, align 8
// CHECK:   [[TMP37:%.*]] = load volatile <4 x i32>, ptr @bi2, align 8
// CHECK:   [[CMP35:%.*]] = icmp ne <4 x i32> [[TMP36]], [[TMP37]]
// CHECK:   [[SEXT36:%.*]] = sext <4 x i1> [[CMP35]] to <4 x i32>
// CHECK:   store volatile <4 x i32> [[SEXT36]], ptr @bi, align 8
// CHECK:   [[TMP38:%.*]] = load volatile <4 x i32>, ptr @bi, align 8
// CHECK:   [[TMP39:%.*]] = load volatile <4 x i32>, ptr @ui2, align 8
// CHECK:   [[CMP37:%.*]] = icmp ne <4 x i32> [[TMP38]], [[TMP39]]
// CHECK:   [[SEXT38:%.*]] = sext <4 x i1> [[CMP37]] to <4 x i32>
// CHECK:   store volatile <4 x i32> [[SEXT38]], ptr @bi, align 8
// CHECK:   [[TMP40:%.*]] = load volatile <4 x i32>, ptr @bi, align 8
// CHECK:   [[TMP41:%.*]] = load volatile <4 x i32>, ptr @bi2, align 8
// CHECK:   [[CMP39:%.*]] = icmp ne <4 x i32> [[TMP40]], [[TMP41]]
// CHECK:   [[SEXT40:%.*]] = sext <4 x i1> [[CMP39]] to <4 x i32>
// CHECK:   store volatile <4 x i32> [[SEXT40]], ptr @bi, align 8
// CHECK:   [[TMP42:%.*]] = load volatile <2 x i64>, ptr @sl, align 8
// CHECK:   [[TMP43:%.*]] = load volatile <2 x i64>, ptr @sl2, align 8
// CHECK:   [[CMP41:%.*]] = icmp ne <2 x i64> [[TMP42]], [[TMP43]]
// CHECK:   [[SEXT42:%.*]] = sext <2 x i1> [[CMP41]] to <2 x i64>
// CHECK:   store volatile <2 x i64> [[SEXT42]], ptr @bl, align 8
// CHECK:   [[TMP44:%.*]] = load volatile <2 x i64>, ptr @sl, align 8
// CHECK:   [[TMP45:%.*]] = load volatile <2 x i64>, ptr @bl2, align 8
// CHECK:   [[CMP43:%.*]] = icmp ne <2 x i64> [[TMP44]], [[TMP45]]
// CHECK:   [[SEXT44:%.*]] = sext <2 x i1> [[CMP43]] to <2 x i64>
// CHECK:   store volatile <2 x i64> [[SEXT44]], ptr @bl, align 8
// CHECK:   [[TMP46:%.*]] = load volatile <2 x i64>, ptr @bl, align 8
// CHECK:   [[TMP47:%.*]] = load volatile <2 x i64>, ptr @sl2, align 8
// CHECK:   [[CMP45:%.*]] = icmp ne <2 x i64> [[TMP46]], [[TMP47]]
// CHECK:   [[SEXT46:%.*]] = sext <2 x i1> [[CMP45]] to <2 x i64>
// CHECK:   store volatile <2 x i64> [[SEXT46]], ptr @bl, align 8
// CHECK:   [[TMP48:%.*]] = load volatile <2 x i64>, ptr @ul, align 8
// CHECK:   [[TMP49:%.*]] = load volatile <2 x i64>, ptr @ul2, align 8
// CHECK:   [[CMP47:%.*]] = icmp ne <2 x i64> [[TMP48]], [[TMP49]]
// CHECK:   [[SEXT48:%.*]] = sext <2 x i1> [[CMP47]] to <2 x i64>
// CHECK:   store volatile <2 x i64> [[SEXT48]], ptr @bl, align 8
// CHECK:   [[TMP50:%.*]] = load volatile <2 x i64>, ptr @ul, align 8
// CHECK:   [[TMP51:%.*]] = load volatile <2 x i64>, ptr @bl2, align 8
// CHECK:   [[CMP49:%.*]] = icmp ne <2 x i64> [[TMP50]], [[TMP51]]
// CHECK:   [[SEXT50:%.*]] = sext <2 x i1> [[CMP49]] to <2 x i64>
// CHECK:   store volatile <2 x i64> [[SEXT50]], ptr @bl, align 8
// CHECK:   [[TMP52:%.*]] = load volatile <2 x i64>, ptr @bl, align 8
// CHECK:   [[TMP53:%.*]] = load volatile <2 x i64>, ptr @ul2, align 8
// CHECK:   [[CMP51:%.*]] = icmp ne <2 x i64> [[TMP52]], [[TMP53]]
// CHECK:   [[SEXT52:%.*]] = sext <2 x i1> [[CMP51]] to <2 x i64>
// CHECK:   store volatile <2 x i64> [[SEXT52]], ptr @bl, align 8
// CHECK:   [[TMP54:%.*]] = load volatile <2 x i64>, ptr @bl, align 8
// CHECK:   [[TMP55:%.*]] = load volatile <2 x i64>, ptr @bl2, align 8
// CHECK:   [[CMP53:%.*]] = icmp ne <2 x i64> [[TMP54]], [[TMP55]]
// CHECK:   [[SEXT54:%.*]] = sext <2 x i1> [[CMP53]] to <2 x i64>
// CHECK:   store volatile <2 x i64> [[SEXT54]], ptr @bl, align 8
// CHECK:   [[TMP56:%.*]] = load volatile <2 x double>, ptr @fd, align 8
// CHECK:   [[TMP57:%.*]] = load volatile <2 x double>, ptr @fd2, align 8
// CHECK:   [[CMP55:%.*]] = fcmp une <2 x double> [[TMP56]], [[TMP57]]
// CHECK:   [[SEXT56:%.*]] = sext <2 x i1> [[CMP55]] to <2 x i64>
// CHECK:   store volatile <2 x i64> [[SEXT56]], ptr @bl, align 8
// CHECK:   ret void
void test_cmpne(void) {

  bc = sc != sc2;
  bc = sc != bc2;
  bc = bc != sc2;
  bc = uc != uc2;
  bc = uc != bc2;
  bc = bc != uc2;
  bc = bc != bc2;

  bs = ss != ss2;
  bs = ss != bs2;
  bs = bs != ss2;
  bs = us != us2;
  bs = us != bs2;
  bs = bs != us2;
  bs = bs != bs2;

  bi = si != si2;
  bi = si != bi2;
  bi = bi != si2;
  bi = ui != ui2;
  bi = ui != bi2;
  bi = bi != ui2;
  bi = bi != bi2;

  bl = sl != sl2;
  bl = sl != bl2;
  bl = bl != sl2;
  bl = ul != ul2;
  bl = ul != bl2;
  bl = bl != ul2;
  bl = bl != bl2;

  bl = fd != fd2;
}

// CHECK-LABEL: define{{.*}} void @test_cmpge() #0 {
// CHECK:   [[TMP0:%.*]] = load volatile <16 x i8>, ptr @sc, align 8
// CHECK:   [[TMP1:%.*]] = load volatile <16 x i8>, ptr @sc2, align 8
// CHECK:   [[CMP:%.*]] = icmp sge <16 x i8> [[TMP0]], [[TMP1]]
// CHECK:   [[SEXT:%.*]] = sext <16 x i1> [[CMP]] to <16 x i8>
// CHECK:   store volatile <16 x i8> [[SEXT]], ptr @bc, align 8
// CHECK:   [[TMP2:%.*]] = load volatile <16 x i8>, ptr @uc, align 8
// CHECK:   [[TMP3:%.*]] = load volatile <16 x i8>, ptr @uc2, align 8
// CHECK:   [[CMP1:%.*]] = icmp uge <16 x i8> [[TMP2]], [[TMP3]]
// CHECK:   [[SEXT2:%.*]] = sext <16 x i1> [[CMP1]] to <16 x i8>
// CHECK:   store volatile <16 x i8> [[SEXT2]], ptr @bc, align 8
// CHECK:   [[TMP4:%.*]] = load volatile <16 x i8>, ptr @bc, align 8
// CHECK:   [[TMP5:%.*]] = load volatile <16 x i8>, ptr @bc2, align 8
// CHECK:   [[CMP3:%.*]] = icmp uge <16 x i8> [[TMP4]], [[TMP5]]
// CHECK:   [[SEXT4:%.*]] = sext <16 x i1> [[CMP3]] to <16 x i8>
// CHECK:   store volatile <16 x i8> [[SEXT4]], ptr @bc, align 8
// CHECK:   [[TMP6:%.*]] = load volatile <8 x i16>, ptr @ss, align 8
// CHECK:   [[TMP7:%.*]] = load volatile <8 x i16>, ptr @ss2, align 8
// CHECK:   [[CMP5:%.*]] = icmp sge <8 x i16> [[TMP6]], [[TMP7]]
// CHECK:   [[SEXT6:%.*]] = sext <8 x i1> [[CMP5]] to <8 x i16>
// CHECK:   store volatile <8 x i16> [[SEXT6]], ptr @bs, align 8
// CHECK:   [[TMP8:%.*]] = load volatile <8 x i16>, ptr @us, align 8
// CHECK:   [[TMP9:%.*]] = load volatile <8 x i16>, ptr @us2, align 8
// CHECK:   [[CMP7:%.*]] = icmp uge <8 x i16> [[TMP8]], [[TMP9]]
// CHECK:   [[SEXT8:%.*]] = sext <8 x i1> [[CMP7]] to <8 x i16>
// CHECK:   store volatile <8 x i16> [[SEXT8]], ptr @bs, align 8
// CHECK:   [[TMP10:%.*]] = load volatile <8 x i16>, ptr @bs, align 8
// CHECK:   [[TMP11:%.*]] = load volatile <8 x i16>, ptr @bs2, align 8
// CHECK:   [[CMP9:%.*]] = icmp uge <8 x i16> [[TMP10]], [[TMP11]]
// CHECK:   [[SEXT10:%.*]] = sext <8 x i1> [[CMP9]] to <8 x i16>
// CHECK:   store volatile <8 x i16> [[SEXT10]], ptr @bs, align 8
// CHECK:   [[TMP12:%.*]] = load volatile <4 x i32>, ptr @si, align 8
// CHECK:   [[TMP13:%.*]] = load volatile <4 x i32>, ptr @si2, align 8
// CHECK:   [[CMP11:%.*]] = icmp sge <4 x i32> [[TMP12]], [[TMP13]]
// CHECK:   [[SEXT12:%.*]] = sext <4 x i1> [[CMP11]] to <4 x i32>
// CHECK:   store volatile <4 x i32> [[SEXT12]], ptr @bi, align 8
// CHECK:   [[TMP14:%.*]] = load volatile <4 x i32>, ptr @ui, align 8
// CHECK:   [[TMP15:%.*]] = load volatile <4 x i32>, ptr @ui2, align 8
// CHECK:   [[CMP13:%.*]] = icmp uge <4 x i32> [[TMP14]], [[TMP15]]
// CHECK:   [[SEXT14:%.*]] = sext <4 x i1> [[CMP13]] to <4 x i32>
// CHECK:   store volatile <4 x i32> [[SEXT14]], ptr @bi, align 8
// CHECK:   [[TMP16:%.*]] = load volatile <4 x i32>, ptr @bi, align 8
// CHECK:   [[TMP17:%.*]] = load volatile <4 x i32>, ptr @bi2, align 8
// CHECK:   [[CMP15:%.*]] = icmp uge <4 x i32> [[TMP16]], [[TMP17]]
// CHECK:   [[SEXT16:%.*]] = sext <4 x i1> [[CMP15]] to <4 x i32>
// CHECK:   store volatile <4 x i32> [[SEXT16]], ptr @bi, align 8
// CHECK:   [[TMP18:%.*]] = load volatile <2 x i64>, ptr @sl, align 8
// CHECK:   [[TMP19:%.*]] = load volatile <2 x i64>, ptr @sl2, align 8
// CHECK:   [[CMP17:%.*]] = icmp sge <2 x i64> [[TMP18]], [[TMP19]]
// CHECK:   [[SEXT18:%.*]] = sext <2 x i1> [[CMP17]] to <2 x i64>
// CHECK:   store volatile <2 x i64> [[SEXT18]], ptr @bl, align 8
// CHECK:   [[TMP20:%.*]] = load volatile <2 x i64>, ptr @ul, align 8
// CHECK:   [[TMP21:%.*]] = load volatile <2 x i64>, ptr @ul2, align 8
// CHECK:   [[CMP19:%.*]] = icmp uge <2 x i64> [[TMP20]], [[TMP21]]
// CHECK:   [[SEXT20:%.*]] = sext <2 x i1> [[CMP19]] to <2 x i64>
// CHECK:   store volatile <2 x i64> [[SEXT20]], ptr @bl, align 8
// CHECK:   [[TMP22:%.*]] = load volatile <2 x i64>, ptr @bl, align 8
// CHECK:   [[TMP23:%.*]] = load volatile <2 x i64>, ptr @bl2, align 8
// CHECK:   [[CMP21:%.*]] = icmp uge <2 x i64> [[TMP22]], [[TMP23]]
// CHECK:   [[SEXT22:%.*]] = sext <2 x i1> [[CMP21]] to <2 x i64>
// CHECK:   store volatile <2 x i64> [[SEXT22]], ptr @bl, align 8
// CHECK:   [[TMP24:%.*]] = load volatile <2 x double>, ptr @fd, align 8
// CHECK:   [[TMP25:%.*]] = load volatile <2 x double>, ptr @fd2, align 8
// CHECK:   [[CMP23:%.*]] = fcmp oge <2 x double> [[TMP24]], [[TMP25]]
// CHECK:   [[SEXT24:%.*]] = sext <2 x i1> [[CMP23]] to <2 x i64>
// CHECK:   store volatile <2 x i64> [[SEXT24]], ptr @bl, align 8
// CHECK:   ret void
void test_cmpge(void) {

  bc = sc >= sc2;
  bc = uc >= uc2;
  bc = bc >= bc2;

  bs = ss >= ss2;
  bs = us >= us2;
  bs = bs >= bs2;

  bi = si >= si2;
  bi = ui >= ui2;
  bi = bi >= bi2;

  bl = sl >= sl2;
  bl = ul >= ul2;
  bl = bl >= bl2;

  bl = fd >= fd2;
}

// CHECK-LABEL: define{{.*}} void @test_cmpgt() #0 {
// CHECK:   [[TMP0:%.*]] = load volatile <16 x i8>, ptr @sc, align 8
// CHECK:   [[TMP1:%.*]] = load volatile <16 x i8>, ptr @sc2, align 8
// CHECK:   [[CMP:%.*]] = icmp sgt <16 x i8> [[TMP0]], [[TMP1]]
// CHECK:   [[SEXT:%.*]] = sext <16 x i1> [[CMP]] to <16 x i8>
// CHECK:   store volatile <16 x i8> [[SEXT]], ptr @bc, align 8
// CHECK:   [[TMP2:%.*]] = load volatile <16 x i8>, ptr @uc, align 8
// CHECK:   [[TMP3:%.*]] = load volatile <16 x i8>, ptr @uc2, align 8
// CHECK:   [[CMP1:%.*]] = icmp ugt <16 x i8> [[TMP2]], [[TMP3]]
// CHECK:   [[SEXT2:%.*]] = sext <16 x i1> [[CMP1]] to <16 x i8>
// CHECK:   store volatile <16 x i8> [[SEXT2]], ptr @bc, align 8
// CHECK:   [[TMP4:%.*]] = load volatile <16 x i8>, ptr @bc, align 8
// CHECK:   [[TMP5:%.*]] = load volatile <16 x i8>, ptr @bc2, align 8
// CHECK:   [[CMP3:%.*]] = icmp ugt <16 x i8> [[TMP4]], [[TMP5]]
// CHECK:   [[SEXT4:%.*]] = sext <16 x i1> [[CMP3]] to <16 x i8>
// CHECK:   store volatile <16 x i8> [[SEXT4]], ptr @bc, align 8
// CHECK:   [[TMP6:%.*]] = load volatile <8 x i16>, ptr @ss, align 8
// CHECK:   [[TMP7:%.*]] = load volatile <8 x i16>, ptr @ss2, align 8
// CHECK:   [[CMP5:%.*]] = icmp sgt <8 x i16> [[TMP6]], [[TMP7]]
// CHECK:   [[SEXT6:%.*]] = sext <8 x i1> [[CMP5]] to <8 x i16>
// CHECK:   store volatile <8 x i16> [[SEXT6]], ptr @bs, align 8
// CHECK:   [[TMP8:%.*]] = load volatile <8 x i16>, ptr @us, align 8
// CHECK:   [[TMP9:%.*]] = load volatile <8 x i16>, ptr @us2, align 8
// CHECK:   [[CMP7:%.*]] = icmp ugt <8 x i16> [[TMP8]], [[TMP9]]
// CHECK:   [[SEXT8:%.*]] = sext <8 x i1> [[CMP7]] to <8 x i16>
// CHECK:   store volatile <8 x i16> [[SEXT8]], ptr @bs, align 8
// CHECK:   [[TMP10:%.*]] = load volatile <8 x i16>, ptr @bs, align 8
// CHECK:   [[TMP11:%.*]] = load volatile <8 x i16>, ptr @bs2, align 8
// CHECK:   [[CMP9:%.*]] = icmp ugt <8 x i16> [[TMP10]], [[TMP11]]
// CHECK:   [[SEXT10:%.*]] = sext <8 x i1> [[CMP9]] to <8 x i16>
// CHECK:   store volatile <8 x i16> [[SEXT10]], ptr @bs, align 8
// CHECK:   [[TMP12:%.*]] = load volatile <4 x i32>, ptr @si, align 8
// CHECK:   [[TMP13:%.*]] = load volatile <4 x i32>, ptr @si2, align 8
// CHECK:   [[CMP11:%.*]] = icmp sgt <4 x i32> [[TMP12]], [[TMP13]]
// CHECK:   [[SEXT12:%.*]] = sext <4 x i1> [[CMP11]] to <4 x i32>
// CHECK:   store volatile <4 x i32> [[SEXT12]], ptr @bi, align 8
// CHECK:   [[TMP14:%.*]] = load volatile <4 x i32>, ptr @ui, align 8
// CHECK:   [[TMP15:%.*]] = load volatile <4 x i32>, ptr @ui2, align 8
// CHECK:   [[CMP13:%.*]] = icmp ugt <4 x i32> [[TMP14]], [[TMP15]]
// CHECK:   [[SEXT14:%.*]] = sext <4 x i1> [[CMP13]] to <4 x i32>
// CHECK:   store volatile <4 x i32> [[SEXT14]], ptr @bi, align 8
// CHECK:   [[TMP16:%.*]] = load volatile <4 x i32>, ptr @bi, align 8
// CHECK:   [[TMP17:%.*]] = load volatile <4 x i32>, ptr @bi2, align 8
// CHECK:   [[CMP15:%.*]] = icmp ugt <4 x i32> [[TMP16]], [[TMP17]]
// CHECK:   [[SEXT16:%.*]] = sext <4 x i1> [[CMP15]] to <4 x i32>
// CHECK:   store volatile <4 x i32> [[SEXT16]], ptr @bi, align 8
// CHECK:   [[TMP18:%.*]] = load volatile <2 x i64>, ptr @sl, align 8
// CHECK:   [[TMP19:%.*]] = load volatile <2 x i64>, ptr @sl2, align 8
// CHECK:   [[CMP17:%.*]] = icmp sgt <2 x i64> [[TMP18]], [[TMP19]]
// CHECK:   [[SEXT18:%.*]] = sext <2 x i1> [[CMP17]] to <2 x i64>
// CHECK:   store volatile <2 x i64> [[SEXT18]], ptr @bl, align 8
// CHECK:   [[TMP20:%.*]] = load volatile <2 x i64>, ptr @ul, align 8
// CHECK:   [[TMP21:%.*]] = load volatile <2 x i64>, ptr @ul2, align 8
// CHECK:   [[CMP19:%.*]] = icmp ugt <2 x i64> [[TMP20]], [[TMP21]]
// CHECK:   [[SEXT20:%.*]] = sext <2 x i1> [[CMP19]] to <2 x i64>
// CHECK:   store volatile <2 x i64> [[SEXT20]], ptr @bl, align 8
// CHECK:   [[TMP22:%.*]] = load volatile <2 x i64>, ptr @bl, align 8
// CHECK:   [[TMP23:%.*]] = load volatile <2 x i64>, ptr @bl2, align 8
// CHECK:   [[CMP21:%.*]] = icmp ugt <2 x i64> [[TMP22]], [[TMP23]]
// CHECK:   [[SEXT22:%.*]] = sext <2 x i1> [[CMP21]] to <2 x i64>
// CHECK:   store volatile <2 x i64> [[SEXT22]], ptr @bl, align 8
// CHECK:   [[TMP24:%.*]] = load volatile <2 x double>, ptr @fd, align 8
// CHECK:   [[TMP25:%.*]] = load volatile <2 x double>, ptr @fd2, align 8
// CHECK:   [[CMP23:%.*]] = fcmp ogt <2 x double> [[TMP24]], [[TMP25]]
// CHECK:   [[SEXT24:%.*]] = sext <2 x i1> [[CMP23]] to <2 x i64>
// CHECK:   store volatile <2 x i64> [[SEXT24]], ptr @bl, align 8
// CHECK:   ret void
void test_cmpgt(void) {

  bc = sc > sc2;
  bc = uc > uc2;
  bc = bc > bc2;

  bs = ss > ss2;
  bs = us > us2;
  bs = bs > bs2;

  bi = si > si2;
  bi = ui > ui2;
  bi = bi > bi2;

  bl = sl > sl2;
  bl = ul > ul2;
  bl = bl > bl2;

  bl = fd > fd2;
}

// CHECK-LABEL: define{{.*}} void @test_cmple() #0 {
// CHECK:   [[TMP0:%.*]] = load volatile <16 x i8>, ptr @sc, align 8
// CHECK:   [[TMP1:%.*]] = load volatile <16 x i8>, ptr @sc2, align 8
// CHECK:   [[CMP:%.*]] = icmp sle <16 x i8> [[TMP0]], [[TMP1]]
// CHECK:   [[SEXT:%.*]] = sext <16 x i1> [[CMP]] to <16 x i8>
// CHECK:   store volatile <16 x i8> [[SEXT]], ptr @bc, align 8
// CHECK:   [[TMP2:%.*]] = load volatile <16 x i8>, ptr @uc, align 8
// CHECK:   [[TMP3:%.*]] = load volatile <16 x i8>, ptr @uc2, align 8
// CHECK:   [[CMP1:%.*]] = icmp ule <16 x i8> [[TMP2]], [[TMP3]]
// CHECK:   [[SEXT2:%.*]] = sext <16 x i1> [[CMP1]] to <16 x i8>
// CHECK:   store volatile <16 x i8> [[SEXT2]], ptr @bc, align 8
// CHECK:   [[TMP4:%.*]] = load volatile <16 x i8>, ptr @bc, align 8
// CHECK:   [[TMP5:%.*]] = load volatile <16 x i8>, ptr @bc2, align 8
// CHECK:   [[CMP3:%.*]] = icmp ule <16 x i8> [[TMP4]], [[TMP5]]
// CHECK:   [[SEXT4:%.*]] = sext <16 x i1> [[CMP3]] to <16 x i8>
// CHECK:   store volatile <16 x i8> [[SEXT4]], ptr @bc, align 8
// CHECK:   [[TMP6:%.*]] = load volatile <8 x i16>, ptr @ss, align 8
// CHECK:   [[TMP7:%.*]] = load volatile <8 x i16>, ptr @ss2, align 8
// CHECK:   [[CMP5:%.*]] = icmp sle <8 x i16> [[TMP6]], [[TMP7]]
// CHECK:   [[SEXT6:%.*]] = sext <8 x i1> [[CMP5]] to <8 x i16>
// CHECK:   store volatile <8 x i16> [[SEXT6]], ptr @bs, align 8
// CHECK:   [[TMP8:%.*]] = load volatile <8 x i16>, ptr @us, align 8
// CHECK:   [[TMP9:%.*]] = load volatile <8 x i16>, ptr @us2, align 8
// CHECK:   [[CMP7:%.*]] = icmp ule <8 x i16> [[TMP8]], [[TMP9]]
// CHECK:   [[SEXT8:%.*]] = sext <8 x i1> [[CMP7]] to <8 x i16>
// CHECK:   store volatile <8 x i16> [[SEXT8]], ptr @bs, align 8
// CHECK:   [[TMP10:%.*]] = load volatile <8 x i16>, ptr @bs, align 8
// CHECK:   [[TMP11:%.*]] = load volatile <8 x i16>, ptr @bs2, align 8
// CHECK:   [[CMP9:%.*]] = icmp ule <8 x i16> [[TMP10]], [[TMP11]]
// CHECK:   [[SEXT10:%.*]] = sext <8 x i1> [[CMP9]] to <8 x i16>
// CHECK:   store volatile <8 x i16> [[SEXT10]], ptr @bs, align 8
// CHECK:   [[TMP12:%.*]] = load volatile <4 x i32>, ptr @si, align 8
// CHECK:   [[TMP13:%.*]] = load volatile <4 x i32>, ptr @si2, align 8
// CHECK:   [[CMP11:%.*]] = icmp sle <4 x i32> [[TMP12]], [[TMP13]]
// CHECK:   [[SEXT12:%.*]] = sext <4 x i1> [[CMP11]] to <4 x i32>
// CHECK:   store volatile <4 x i32> [[SEXT12]], ptr @bi, align 8
// CHECK:   [[TMP14:%.*]] = load volatile <4 x i32>, ptr @ui, align 8
// CHECK:   [[TMP15:%.*]] = load volatile <4 x i32>, ptr @ui2, align 8
// CHECK:   [[CMP13:%.*]] = icmp ule <4 x i32> [[TMP14]], [[TMP15]]
// CHECK:   [[SEXT14:%.*]] = sext <4 x i1> [[CMP13]] to <4 x i32>
// CHECK:   store volatile <4 x i32> [[SEXT14]], ptr @bi, align 8
// CHECK:   [[TMP16:%.*]] = load volatile <4 x i32>, ptr @bi, align 8
// CHECK:   [[TMP17:%.*]] = load volatile <4 x i32>, ptr @bi2, align 8
// CHECK:   [[CMP15:%.*]] = icmp ule <4 x i32> [[TMP16]], [[TMP17]]
// CHECK:   [[SEXT16:%.*]] = sext <4 x i1> [[CMP15]] to <4 x i32>
// CHECK:   store volatile <4 x i32> [[SEXT16]], ptr @bi, align 8
// CHECK:   [[TMP18:%.*]] = load volatile <2 x i64>, ptr @sl, align 8
// CHECK:   [[TMP19:%.*]] = load volatile <2 x i64>, ptr @sl2, align 8
// CHECK:   [[CMP17:%.*]] = icmp sle <2 x i64> [[TMP18]], [[TMP19]]
// CHECK:   [[SEXT18:%.*]] = sext <2 x i1> [[CMP17]] to <2 x i64>
// CHECK:   store volatile <2 x i64> [[SEXT18]], ptr @bl, align 8
// CHECK:   [[TMP20:%.*]] = load volatile <2 x i64>, ptr @ul, align 8
// CHECK:   [[TMP21:%.*]] = load volatile <2 x i64>, ptr @ul2, align 8
// CHECK:   [[CMP19:%.*]] = icmp ule <2 x i64> [[TMP20]], [[TMP21]]
// CHECK:   [[SEXT20:%.*]] = sext <2 x i1> [[CMP19]] to <2 x i64>
// CHECK:   store volatile <2 x i64> [[SEXT20]], ptr @bl, align 8
// CHECK:   [[TMP22:%.*]] = load volatile <2 x i64>, ptr @bl, align 8
// CHECK:   [[TMP23:%.*]] = load volatile <2 x i64>, ptr @bl2, align 8
// CHECK:   [[CMP21:%.*]] = icmp ule <2 x i64> [[TMP22]], [[TMP23]]
// CHECK:   [[SEXT22:%.*]] = sext <2 x i1> [[CMP21]] to <2 x i64>
// CHECK:   store volatile <2 x i64> [[SEXT22]], ptr @bl, align 8
// CHECK:   [[TMP24:%.*]] = load volatile <2 x double>, ptr @fd, align 8
// CHECK:   [[TMP25:%.*]] = load volatile <2 x double>, ptr @fd2, align 8
// CHECK:   [[CMP23:%.*]] = fcmp ole <2 x double> [[TMP24]], [[TMP25]]
// CHECK:   [[SEXT24:%.*]] = sext <2 x i1> [[CMP23]] to <2 x i64>
// CHECK:   store volatile <2 x i64> [[SEXT24]], ptr @bl, align 8
// CHECK:   ret void
void test_cmple(void) {

  bc = sc <= sc2;
  bc = uc <= uc2;
  bc = bc <= bc2;

  bs = ss <= ss2;
  bs = us <= us2;
  bs = bs <= bs2;

  bi = si <= si2;
  bi = ui <= ui2;
  bi = bi <= bi2;

  bl = sl <= sl2;
  bl = ul <= ul2;
  bl = bl <= bl2;

  bl = fd <= fd2;
}

// CHECK-LABEL: define{{.*}} void @test_cmplt() #0 {
// CHECK:   [[TMP0:%.*]] = load volatile <16 x i8>, ptr @sc, align 8
// CHECK:   [[TMP1:%.*]] = load volatile <16 x i8>, ptr @sc2, align 8
// CHECK:   [[CMP:%.*]] = icmp slt <16 x i8> [[TMP0]], [[TMP1]]
// CHECK:   [[SEXT:%.*]] = sext <16 x i1> [[CMP]] to <16 x i8>
// CHECK:   store volatile <16 x i8> [[SEXT]], ptr @bc, align 8
// CHECK:   [[TMP2:%.*]] = load volatile <16 x i8>, ptr @uc, align 8
// CHECK:   [[TMP3:%.*]] = load volatile <16 x i8>, ptr @uc2, align 8
// CHECK:   [[CMP1:%.*]] = icmp ult <16 x i8> [[TMP2]], [[TMP3]]
// CHECK:   [[SEXT2:%.*]] = sext <16 x i1> [[CMP1]] to <16 x i8>
// CHECK:   store volatile <16 x i8> [[SEXT2]], ptr @bc, align 8
// CHECK:   [[TMP4:%.*]] = load volatile <16 x i8>, ptr @bc, align 8
// CHECK:   [[TMP5:%.*]] = load volatile <16 x i8>, ptr @bc2, align 8
// CHECK:   [[CMP3:%.*]] = icmp ult <16 x i8> [[TMP4]], [[TMP5]]
// CHECK:   [[SEXT4:%.*]] = sext <16 x i1> [[CMP3]] to <16 x i8>
// CHECK:   store volatile <16 x i8> [[SEXT4]], ptr @bc, align 8
// CHECK:   [[TMP6:%.*]] = load volatile <8 x i16>, ptr @ss, align 8
// CHECK:   [[TMP7:%.*]] = load volatile <8 x i16>, ptr @ss2, align 8
// CHECK:   [[CMP5:%.*]] = icmp slt <8 x i16> [[TMP6]], [[TMP7]]
// CHECK:   [[SEXT6:%.*]] = sext <8 x i1> [[CMP5]] to <8 x i16>
// CHECK:   store volatile <8 x i16> [[SEXT6]], ptr @bs, align 8
// CHECK:   [[TMP8:%.*]] = load volatile <8 x i16>, ptr @us, align 8
// CHECK:   [[TMP9:%.*]] = load volatile <8 x i16>, ptr @us2, align 8
// CHECK:   [[CMP7:%.*]] = icmp ult <8 x i16> [[TMP8]], [[TMP9]]
// CHECK:   [[SEXT8:%.*]] = sext <8 x i1> [[CMP7]] to <8 x i16>
// CHECK:   store volatile <8 x i16> [[SEXT8]], ptr @bs, align 8
// CHECK:   [[TMP10:%.*]] = load volatile <8 x i16>, ptr @bs, align 8
// CHECK:   [[TMP11:%.*]] = load volatile <8 x i16>, ptr @bs2, align 8
// CHECK:   [[CMP9:%.*]] = icmp ult <8 x i16> [[TMP10]], [[TMP11]]
// CHECK:   [[SEXT10:%.*]] = sext <8 x i1> [[CMP9]] to <8 x i16>
// CHECK:   store volatile <8 x i16> [[SEXT10]], ptr @bs, align 8
// CHECK:   [[TMP12:%.*]] = load volatile <4 x i32>, ptr @si, align 8
// CHECK:   [[TMP13:%.*]] = load volatile <4 x i32>, ptr @si2, align 8
// CHECK:   [[CMP11:%.*]] = icmp slt <4 x i32> [[TMP12]], [[TMP13]]
// CHECK:   [[SEXT12:%.*]] = sext <4 x i1> [[CMP11]] to <4 x i32>
// CHECK:   store volatile <4 x i32> [[SEXT12]], ptr @bi, align 8
// CHECK:   [[TMP14:%.*]] = load volatile <4 x i32>, ptr @ui, align 8
// CHECK:   [[TMP15:%.*]] = load volatile <4 x i32>, ptr @ui2, align 8
// CHECK:   [[CMP13:%.*]] = icmp ult <4 x i32> [[TMP14]], [[TMP15]]
// CHECK:   [[SEXT14:%.*]] = sext <4 x i1> [[CMP13]] to <4 x i32>
// CHECK:   store volatile <4 x i32> [[SEXT14]], ptr @bi, align 8
// CHECK:   [[TMP16:%.*]] = load volatile <4 x i32>, ptr @bi, align 8
// CHECK:   [[TMP17:%.*]] = load volatile <4 x i32>, ptr @bi2, align 8
// CHECK:   [[CMP15:%.*]] = icmp ult <4 x i32> [[TMP16]], [[TMP17]]
// CHECK:   [[SEXT16:%.*]] = sext <4 x i1> [[CMP15]] to <4 x i32>
// CHECK:   store volatile <4 x i32> [[SEXT16]], ptr @bi, align 8
// CHECK:   [[TMP18:%.*]] = load volatile <2 x i64>, ptr @sl, align 8
// CHECK:   [[TMP19:%.*]] = load volatile <2 x i64>, ptr @sl2, align 8
// CHECK:   [[CMP17:%.*]] = icmp slt <2 x i64> [[TMP18]], [[TMP19]]
// CHECK:   [[SEXT18:%.*]] = sext <2 x i1> [[CMP17]] to <2 x i64>
// CHECK:   store volatile <2 x i64> [[SEXT18]], ptr @bl, align 8
// CHECK:   [[TMP20:%.*]] = load volatile <2 x i64>, ptr @ul, align 8
// CHECK:   [[TMP21:%.*]] = load volatile <2 x i64>, ptr @ul2, align 8
// CHECK:   [[CMP19:%.*]] = icmp ult <2 x i64> [[TMP20]], [[TMP21]]
// CHECK:   [[SEXT20:%.*]] = sext <2 x i1> [[CMP19]] to <2 x i64>
// CHECK:   store volatile <2 x i64> [[SEXT20]], ptr @bl, align 8
// CHECK:   [[TMP22:%.*]] = load volatile <2 x i64>, ptr @bl, align 8
// CHECK:   [[TMP23:%.*]] = load volatile <2 x i64>, ptr @bl2, align 8
// CHECK:   [[CMP21:%.*]] = icmp ult <2 x i64> [[TMP22]], [[TMP23]]
// CHECK:   [[SEXT22:%.*]] = sext <2 x i1> [[CMP21]] to <2 x i64>
// CHECK:   store volatile <2 x i64> [[SEXT22]], ptr @bl, align 8
// CHECK:   [[TMP24:%.*]] = load volatile <2 x double>, ptr @fd, align 8
// CHECK:   [[TMP25:%.*]] = load volatile <2 x double>, ptr @fd2, align 8
// CHECK:   [[CMP23:%.*]] = fcmp olt <2 x double> [[TMP24]], [[TMP25]]
// CHECK:   [[SEXT24:%.*]] = sext <2 x i1> [[CMP23]] to <2 x i64>
// CHECK:   store volatile <2 x i64> [[SEXT24]], ptr @bl, align 8
// CHECK:   ret void
void test_cmplt(void) {

  bc = sc < sc2;
  bc = uc < uc2;
  bc = bc < bc2;

  bs = ss < ss2;
  bs = us < us2;
  bs = bs < bs2;

  bi = si < si2;
  bi = ui < ui2;
  bi = bi < bi2;

  bl = sl < sl2;
  bl = ul < ul2;
  bl = bl < bl2;

  bl = fd < fd2;
}

