// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +zifencei -target-feature +m -target-feature +a \
// RUN:  -emit-llvm %s -o - | FileCheck %s

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
// CHECK-LABEL: define dso_local void @testMultiArchSelectLast
// CHECK-SAME: () #4 {
__attribute__((target("arch=rv64gc;arch=rv64gc_zbb"))) void testMultiArchSelectLast() {}
// CHECK-LABEL: define dso_local void @testMultiCpuSelectLast
// CHECK-SAME: () #8 {
__attribute__((target("cpu=sifive-u74;cpu=sifive-u54"))) void testMultiCpuSelectLast() {}
// CHECK-LABEL: define dso_local void @testMultiTuneSelectLast
// CHECK-SAME: () #9 {
__attribute__((target("tune=sifive-u74;tune=sifive-u54"))) void testMultiTuneSelectLast() {}

//.
// CHECK: attributes #0 = { {{.*}}"target-features"="+64bit,+a,+m,+zifencei" }
// CHECK: attributes #1 = { {{.*}}"target-cpu"="rocket-rv64" "target-features"="+64bit,+a,+d,+f,+m,+v,+zicsr,+zifencei,+zve32f,+zve32x,+zve64d,+zve64f,+zve64x,+zvl128b,+zvl32b,+zvl64b" "tune-cpu"="generic-rv64" }
// CHECK: attributes #2 = { {{.*}}"target-features"="+64bit,+a,+m,+zbb,+zifencei" }
// CHECK: attributes #3 = { {{.*}}"target-features"="+64bit,+a,+d,+experimental-zicond,+f,+m,+v,+zbb,+zicsr,+zifencei,+zve32f,+zve32x,+zve64d,+zve64f,+zve64x,+zvl128b,+zvl32b,+zvl64b" }
// CHECK: attributes #4 = { {{.*}}"target-features"="+64bit,+a,+c,+d,+f,+m,+zbb,+zicsr,+zifencei" }
// CHECK: attributes #5 = { {{.*}}"target-features"="+64bit,+m" }
// CHECK: attributes #6 = { {{.*}}"target-cpu"="sifive-u54" "target-features"="+64bit,+a,+m,+zbb,+zifencei" }
// CHECK: attributes #7 = { {{.*}}"target-cpu"="sifive-u54" "target-features"="+64bit,+m" }
// CHECK: attributes #8 = { {{.*}}"target-cpu"="sifive-u54" "target-features"="+64bit,+a,+c,+d,+f,+m,+zicsr,+zifencei" }
// CHECK: attributes #9 = { {{.*}}"target-features"="+64bit,+a,+m,+zifencei" "tune-cpu"="sifive-u54" }
