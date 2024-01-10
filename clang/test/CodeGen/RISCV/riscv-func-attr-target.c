// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +zifencei -target-feature +m \
// RUN:  -target-feature +a -target-feature +save-restore -target-feature -zbb \
// RUN:  -target-feature -relax -target-feature -zfa \
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

//.
// CHECK: attributes #0 = { {{.*}}"target-features"="+64bit,+a,+m,+save-restore,+zifencei,-relax,-zbb,-zfa" }
// CHECK: attributes #1 = { {{.*}}"target-cpu"="rocket-rv64" "target-features"="+64bit,+a,+d,+f,+m,+save-restore,+v,+zicsr,+zifencei,+zve32f,+zve32x,+zve64d,+zve64f,+zve64x,+zvl128b,+zvl32b,+zvl64b,-relax,-zbb,-zfa" "tune-cpu"="generic-rv64" }
// CHECK: attributes #2 = { {{.*}}"target-features"="+64bit,+a,+m,+save-restore,+zbb,+zifencei,-relax,-zfa" }
// CHECK: attributes #3 = { {{.*}}"target-features"="+64bit,+a,+d,+experimental-zicond,+f,+m,+save-restore,+v,+zbb,+zicsr,+zifencei,+zve32f,+zve32x,+zve64d,+zve64f,+zve64x,+zvl128b,+zvl32b,+zvl64b,-relax,-zfa" }
// CHECK: attributes #4 = { {{.*}}"target-features"="+64bit,+a,+c,+d,+f,+m,+save-restore,+zbb,+zicsr,+zifencei,-relax" }
// CHECK: attributes #5 = { {{.*}}"target-features"="+64bit,+m,+save-restore,-relax" }
// CHECK: attributes #6 = { {{.*}}"target-cpu"="sifive-u54" "target-features"="+64bit,+a,+m,+save-restore,+zbb,+zifencei,-relax,-zfa" }
// CHECK: attributes #7 = { {{.*}}"target-cpu"="sifive-u54" "target-features"="+64bit,+m,+save-restore,-relax" }
// CHECK: attributes #8 = { {{.*}}"target-cpu"="sifive-u54" "target-features"="+64bit,+a,+c,+d,+f,+m,+save-restore,+zicsr,+zifencei,-relax" }
