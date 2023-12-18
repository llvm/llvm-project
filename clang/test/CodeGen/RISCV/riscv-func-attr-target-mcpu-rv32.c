// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv32 -target-cpu sifive-e76 -target-feature +zifencei -target-feature +m \
// RUN:  -target-feature +a -target-feature +save-restore -target-feature -zbb \
// RUN:  -target-feature -relax -target-feature -zfa \
// RUN:  -emit-llvm %s -o - | FileCheck %s

// CHECK: define dso_local void @testDefault() #0
void testDefault() {}

// CHECK: define dso_local void @testFullArchOnly() #1
__attribute__((target("arch=rv32imac"))) void
testFullArchOnly() {}

// CHECK: define dso_local void @testFullArchAndCpu() #2
__attribute__((target("arch=rv32imac;cpu=sifive-e34"))) void
testFullArchAndCpu() {}

// CHECK: define dso_local void @testFullArchAndTune() #2
__attribute__((target("arch=rv32imac;tune=sifive-e34"))) void
testFullArchAndTune() {}

// CHECK: define dso_local void @testFullArchAndCpuAndTune() #2
__attribute__((target("arch=rv32imac;cpu=sifive-e31;tune=sifive-e34"))) void
testFullArchAndCpuAndTune() {}

// CHECK: define dso_local void @testAddExtOnly() #3
__attribute__((target("arch=+v"))) void
testAddExtOnly() {}

// CHECK: define dso_local void @testAddExtAndCpu() #4
__attribute__((target("arch=+v;cpu=sifive-e31"))) void
testAddExtAndCpu() {}

// CHECK: define dso_local void @testAddExtAndTune() #4
__attribute__((target("arch=+v;tune=sifive-e31"))) void
testAddExtAndTune() {}

// CHECK: define dso_local void @testAddExtAndCpuAndTune() #5
__attribute__((target("arch=+v;cpu=sifive-e31;tune=sifive-e34"))) void
testAddExtAndCpuAndTune() {}

// CHECK: define dso_local void @testCpuOnly() #6
__attribute__((target("cpu=sifive-e31"))) void
testCpuOnly() {}

// CHECK: define dso_local void @testCpuAndTune() #7
__attribute__((target("cpu=sifive-e31;tune=sifive-e34"))) void
testCpuAndTune() {}

// CHECK: define dso_local void @testTuneOnly() #8
__attribute__((target("tune=sifive-e34"))) void
testTuneOnly() {}

// .
// CHECK: attributes #0 = { {{.*}}"target-cpu"="sifive-e76" "target-features"="+32bit,+a,+m,+save-restore,+zifencei,-relax,-zbb,-zfa" }
// CHECK: attributes #1 = { {{.*}}"target-cpu"="generic-rv32" "target-features"="+32bit,+a,+c,+m,+save-restore,-relax,-zbb,-zfa" "tune-cpu"="sifive-e76" }
// CHECK: attributes #2 = { {{.*}}"target-cpu"="generic-rv32" "target-features"="+32bit,+a,+c,+m,+save-restore,-relax,-zbb,-zfa" "tune-cpu"="sifive-e34" }
// CHECK: attributes #3 = { {{.*}}"target-cpu"="sifive-e76" "target-features"="+32bit,+a,+d,+f,+m,+save-restore,+v,+zicsr,+zifencei,+zve32f,+zve32x,+zve64d,+zve64f,+zve64x,+zvl128b,+zvl32b,+zvl64b,-relax,-zbb,-zfa" }
// CHECK: attributes #4 = { {{.*}}"target-cpu"="sifive-e76" "target-features"="+32bit,+a,+d,+f,+m,+save-restore,+v,+zicsr,+zifencei,+zve32f,+zve32x,+zve64d,+zve64f,+zve64x,+zvl128b,+zvl32b,+zvl64b,-relax,-zbb,-zfa" "tune-cpu"="sifive-e31" }
// CHECK: attributes #5 = { {{.*}}"target-cpu"="sifive-e76" "target-features"="+32bit,+a,+d,+f,+m,+save-restore,+v,+zicsr,+zifencei,+zve32f,+zve32x,+zve64d,+zve64f,+zve64x,+zvl128b,+zvl32b,+zvl64b,-relax,-zbb,-zfa" "tune-cpu"="sifive-e34" }
// CHECK: attributes #6 = { {{.*}}"target-cpu"="sifive-e31" "target-features"="+32bit,+a,+c,+m,+save-restore,+zicsr,+zifencei,-relax,-zbb,-zfa" }
// CHECK: attributes #7 = { {{.*}}"target-cpu"="sifive-e31" "target-features"="+32bit,+a,+c,+m,+save-restore,+zicsr,+zifencei,-relax,-zbb,-zfa" "tune-cpu"="sifive-e34" }
// CHECK: attributes #8 = { {{.*}}"target-cpu"="sifive-e76" "target-features"="+32bit,+a,+m,+save-restore,+zifencei,-relax,-zbb,-zfa" "tune-cpu"="sifive-e34" }
