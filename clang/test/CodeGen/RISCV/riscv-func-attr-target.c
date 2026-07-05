// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +zifencei -target-feature +m \
// RUN:  -target-feature +a -target-feature +save-restore -target-feature -zbb \
// RUN:  -target-feature -relax -target-feature -zfa \
// RUN:  -emit-llvm %s -o - | FileCheck %s

#include <riscv_vector.h>

// CHECK-LABEL: define dso_local void @testDefault
// CHECK-SAME: () #0 {
void testDefault() {}
// CHECK-LABEL: define dso_local void @testMultiAttrStr
// CHECK-SAME: () #1 {
__attribute__((target("cpu=rocket-rv64;tune=generic-rv64;arch=+v"))) void
testMultiAttrStr() {}
// CHECK-LABEL: define dso_local void @testSingleExtension
// CHECK-SAME: () #2 {
__attribute__((target("arch=+zbb"))) void testSingleExtension() {}
// CHECK-LABEL: define dso_local void @testMultiExtension
// CHECK-SAME: () #3 {
__attribute__((target("arch=+zbb,+v,+zicond"))) void testMultiExtension() {}
// CHECK-LABEL: define dso_local void @testFullArch
// CHECK-SAME: () #4 {
__attribute__((target("arch=rv64gc_zbb"))) void testFullArch() {}
// CHECK-LABEL: define dso_local void @testFullArchButSmallThanCmdArch
// CHECK-SAME: () #5 {
__attribute__((target("arch=rv64im"))) void testFullArchButSmallThanCmdArch() {}
// CHECK-LABEL: define dso_local void @testAttrArchAndAttrCpu
// CHECK-SAME: () #6 {
__attribute__((target("cpu=sifive-u54;arch=+zbb"))) void
testAttrArchAndAttrCpu() {}
// CHECK-LABEL: define dso_local void @testAttrFullArchAndAttrCpu
// CHECK-SAME: () #7 {
__attribute__((target("cpu=sifive-u54;arch=rv64im"))) void
testAttrFullArchAndAttrCpu() {}
// CHECK-LABEL: define dso_local void @testAttrCpuOnly
// CHECK-SAME: () #8 {
__attribute__((target("cpu=sifive-u54"))) void testAttrCpuOnly() {}

__attribute__((target("arch=+zve32x")))
void test_builtin_w_zve32x() {
// CHECK-LABEL: test_builtin_w_zve32x
// CHECK-SAME: #9
  __riscv_vsetvl_e8m8(1);
}

__attribute__((target("arch=+zve32x")))
void test_rvv_i32_type_w_zve32x() {
// CHECK-LABEL: test_rvv_i32_type_w_zve32x
// CHECK-SAME: #9
  vint32m1_t v;
}

__attribute__((target("arch=+zve32f")))
void test_rvv_f32_type_w_zve32f() {
// CHECK-LABEL: test_rvv_f32_type_w_zve32f
// CHECK-SAME: #11
  vfloat32m1_t v;
}

__attribute__((target("arch=+zve64d")))
void test_rvv_f64_type_w_zve64d() {
// CHECK-LABEL: test_rvv_f64_type_w_zve64d
// CHECK-SAME: #12
  vfloat64m1_t v;
}

__attribute__((target("arch=+v")))
int test_vsetvl_e64m1(unsigned avl) {
// CHECK-LABEL: test_vsetvl_e64m1
// CHECK-SAME: #13
    return __riscv_vsetvl_e64m1(avl);
}

__attribute__((target("arch=+v")))
int test_vsetvlmax_e64m1() {
// CHECK-LABEL: test_vsetvlmax_e64m1
// CHECK-SAME: #13
    return __riscv_vsetvlmax_e64m1();
}

//.
// CHECK: attributes #0 = { {{.*}}"target-features"="+64bit,+a,+i,+m,+save-restore,+zaamo,+zalrsc,+zifencei,+zmmul,-relax,-zbb,-zfa" }
// CHECK: attributes #1 = { {{.*}}"target-cpu"="rocket-rv64" "target-features"="+64bit,+a,+d,+f,+i,+m,+save-restore,+v,+zaamo,+zalrsc,+zicsr,+zifencei,+zmmul,+zve32f,+zve32x,+zve64d,+zve64f,+zve64x,+zvl128b,+zvl32b,+zvl64b,-relax,-zbb,-zfa" "tune-cpu"="generic-rv64" }
// CHECK: attributes #2 = { {{.*}}"target-features"="+64bit,+a,+i,+m,+save-restore,+zaamo,+zalrsc,+zbb,+zifencei,+zmmul,-relax,-zfa" }
// CHECK: attributes #3 = { {{.*}}"target-features"="+64bit,+a,+d,+f,+i,+m,+save-restore,+v,+zaamo,+zalrsc,+zbb,+zicond,+zicsr,+zifencei,+zmmul,+zve32f,+zve32x,+zve64d,+zve64f,+zve64x,+zvl128b,+zvl32b,+zvl64b,-relax,-zfa" }
// Make sure we append negative features if we override the arch
// CHECK: attributes #4 = { {{.*}}"target-features"="+64bit,+a,+c,+d,+f,+i,+m,+save-restore,+zaamo,+zalrsc,+zbb,+zca,+zcd,+zicsr,+zifencei,+zmmul,{{(-[[:alnum:]-]+)(,-[[:alnum:]-]+)*}}" }
// CHECK: attributes #5 = { {{.*}}"target-features"="+64bit,+i,+m,+save-restore,+zmmul,{{(-[[:alnum:]-]+)(,-[[:alnum:]-]+)*}}" }
// CHECK: attributes #6 = { {{.*}}"target-cpu"="sifive-u54" "target-features"="+64bit,+a,+i,+m,+save-restore,+zaamo,+zalrsc,+zbb,+zifencei,+zmmul,-relax,-zfa" }
// CHECK: attributes #7 = { {{.*}}"target-cpu"="sifive-u54" "target-features"="+64bit,+i,+m,+save-restore,+zmmul,{{(-[[:alnum:]-]+)(,-[[:alnum:]-]+)*}}" }
// CHECK: attributes #8 = { {{.*}}"target-cpu"="sifive-u54" "target-features"="+64bit,+a,+c,+d,+f,+i,+m,+save-restore,+zaamo,+zalrsc,+zca,+zcd,+zicsr,+zifencei,+zmmul,{{(-[[:alnum:]-]+)(,-[[:alnum:]-]+)*}}" }
// CHECK: attributes #9 = { {{.*}}"target-features"="+64bit,+a,+i,+m,+save-restore,+zaamo,+zalrsc,+zicsr,+zifencei,+zmmul,+zve32x,+zvl32b,-relax,-zbb,-zfa" }
// CHECK: attributes #11 = { {{.*}}"target-features"="+64bit,+a,+f,+i,+m,+save-restore,+zaamo,+zalrsc,+zicsr,+zifencei,+zmmul,+zve32f,+zve32x,+zvl32b,-relax,-zbb,-zfa" }
// CHECK: attributes #12 = { {{.*}}"target-features"="+64bit,+a,+d,+f,+i,+m,+save-restore,+zaamo,+zalrsc,+zicsr,+zifencei,+zmmul,+zve32f,+zve32x,+zve64d,+zve64f,+zve64x,+zvl32b,+zvl64b,-relax,-zbb,-zfa" }
