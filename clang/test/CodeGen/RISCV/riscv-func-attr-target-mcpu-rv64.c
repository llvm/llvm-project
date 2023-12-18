// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-cpu sifive-x280 -target-feature +zifencei -target-feature +m \
// RUN:  -target-feature +a -target-feature +save-restore -target-feature -zbb \
// RUN:  -target-feature -relax -target-feature -zfa \
// RUN:  -emit-llvm %s -o - | FileCheck %s

// CHECK: define dso_local void @testDefault() #0
void testDefault() {}

// CHECK: define dso_local void @testFullArchOnly() #1
__attribute__((target("arch=rv64imac"))) void
testFullArchOnly() {}

// CHECK: define dso_local void @testFullArchAndCpu() #2
__attribute__((target("arch=rv64imac;cpu=sifive-u74"))) void
testFullArchAndCpu() {}

// CHECK: define dso_local void @testFullArchAndTune() #2
__attribute__((target("arch=rv64imac;tune=sifive-u74"))) void
testFullArchAndTune() {}

// CHECK: define dso_local void @testFullArchAndCpuAndTune() #2
__attribute__((target("arch=rv64imac;cpu=sifive-u54;tune=sifive-u74"))) void
testFullArchAndCpuAndTune() {}

// CHECK: define dso_local void @testAddExtOnly() #3
__attribute__((target("arch=+v"))) void
testAddExtOnly() {}

// CHECK: define dso_local void @testAddExtAndCpu() #4
__attribute__((target("arch=+v;cpu=sifive-u54"))) void
testAddExtAndCpu() {}

// CHECK: define dso_local void @testAddExtAndTune() #4
__attribute__((target("arch=+v;tune=sifive-u54"))) void
testAddExtAndTune() {}

// CHECK: define dso_local void @testAddExtAndCpuAndTune() #5
__attribute__((target("arch=+v;cpu=sifive-u54;tune=sifive-u74"))) void
testAddExtAndCpuAndTune() {}

// CHECK: define dso_local void @testCpuOnly() #6
__attribute__((target("cpu=sifive-u54"))) void
testCpuOnly() {}

// CHECK: define dso_local void @testCpuAndTune() #7
__attribute__((target("cpu=sifive-u54;tune=sifive-u74"))) void
testCpuAndTune() {}

// CHECK: define dso_local void @testTuneOnly() #8
__attribute__((target("tune=sifive-u74"))) void
testTuneOnly() {}

// .
// CHECK: attributes #0 = { {{.*}}"target-cpu"="sifive-x280" "target-features"="+64bit,+a,+m,+save-restore,+zifencei,-relax,-zbb,-zfa" }
// CHECK: attributes #1 = { {{.*}}"target-cpu"="generic-rv64" "target-features"="+64bit,+a,+c,+m,+save-restore,-relax,-zbb,-zfa" "tune-cpu"="sifive-x280" }
// CHECK: attributes #2 = { {{.*}}"target-cpu"="generic-rv64" "target-features"="+64bit,+a,+c,+m,+save-restore,-relax,-zbb,-zfa" "tune-cpu"="sifive-u74" }
// CHECK: attributes #3 = { {{.*}}"target-cpu"="sifive-x280" "target-features"="+64bit,+a,+d,+f,+m,+save-restore,+v,+zicsr,+zifencei,+zve32f,+zve32x,+zve64d,+zve64f,+zve64x,+zvl128b,+zvl32b,+zvl64b,-relax,-zbb,-zfa" }
// CHECK: attributes #4 = { {{.*}}"target-cpu"="sifive-x280" "target-features"="+64bit,+a,+d,+f,+m,+save-restore,+v,+zicsr,+zifencei,+zve32f,+zve32x,+zve64d,+zve64f,+zve64x,+zvl128b,+zvl32b,+zvl64b,-relax,-zbb,-zfa" "tune-cpu"="sifive-u54" }
// CHECK: attributes #5 = { {{.*}}"target-cpu"="sifive-x280" "target-features"="+64bit,+a,+d,+f,+m,+save-restore,+v,+zicsr,+zifencei,+zve32f,+zve32x,+zve64d,+zve64f,+zve64x,+zvl128b,+zvl32b,+zvl64b,-relax,-zbb,-zfa" "tune-cpu"="sifive-u74" }
// CHECK: attributes #6 = { {{.*}}"target-cpu"="sifive-u54" "target-features"="+64bit,+a,+c,+d,+f,+m,+save-restore,+zicsr,+zifencei,-relax,-zbb,-zfa" }
// CHECK: attributes #7 = { {{.*}}"target-cpu"="sifive-u54" "target-features"="+64bit,+a,+c,+d,+f,+m,+save-restore,+zicsr,+zifencei,-relax,-zbb,-zfa" "tune-cpu"="sifive-u74" }
// CHECK: attributes #8 = { {{.*}}"target-cpu"="sifive-x280" "target-features"="+64bit,+a,+m,+save-restore,+zifencei,-relax,-zbb,-zfa" "tune-cpu"="sifive-u74" }
