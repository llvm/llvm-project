// This is a regression test for ThinLTO indirect-call-promotion when candidate
// callees need to be imported from another IR module.  In the C++ test case,
// `main` calls `global_func` which is defined in another module. `global_func`
// has two indirect callees, one has external linkage and one has local linkage.
// All three functions should be imported into the IR module of main.

// What the test does:
// - Generate raw profiles from executables and convert it to indexed profiles.
//   During the conversion, a profiled callee address in raw profiles will be
//   converted to function hash in indexed profiles.
// - Run IRPGO profile use and ThinTLO prelink pipeline and get LLVM bitcodes
//   for both cpp files in the C++ test case.
// - Generate ThinLTO summary file with LLVM bitcodes, and run `function-import` pass.
// - Run `pgo-icall-prom` pass for the IR module which needs to import callees.

// REQUIRES: windows || linux || darwin

// The test failed on ppc when building the instrumented binary.
// ld.lld: error: /lib/../lib64/Scrt1.o: ABI version 1 is not supported
// UNSUPPORTED: ppc

// This test and IR test llvm/test/Transforms/PGOProfile/thinlto_indirect_call_promotion.ll
// are complementary to each other; a compiler-rt test has better test coverage
// on different platforms, and the IR test is less restrictive in terms of
// running environment and could be executed more widely.

// Use lld as linker for more robust test. We need to REQUIRE LLVMgold.so for
// LTO if default linker is GNU ld or gold anyway.
// REQUIRES: lld-available

// RUN: rm -rf %t && split-file %s %t && cd %t

// Do setup work for all below tests.
// Generate raw profiles from real programs and convert it into indexed profiles.
// Use clangxx_pgogen for IR level instrumentation for C++.
// RUN: %clangxx_pgogen -fuse-ld=lld -O2 lib.cpp main.cpp -o main
// RUN: env LLVM_PROFILE_FILE=main.profraw %run ./main
// RUN: llvm-profdata merge main.profraw -o main.profdata

// Use profile on lib and get bitcode. Explicitly skip ICP pass to test ICP happens as
// expected in the IR module that imports functions from lib.
// RUN: %clang -mllvm -disable-icp -fprofile-use=main.profdata -flto=thin -O2 -c lib.cpp -o lib.bc

// Use profile on main and get bitcode.
// RUN: %clang -fprofile-use=main.profdata -flto=thin -O2 -c main.cpp -o main.bc

// Run llvm-lto to get summary file.
// RUN: llvm-lto -thinlto -o summary main.bc lib.bc

// Test the imports of functions. Default import thresholds would work but do
// explicit override to be more futureproof. Note all functions have one basic
// block with a function-entry-count of one, so they are actually hot functions
// per default profile summary hotness cutoff.
// RUN: opt -passes=function-import -import-instr-limit=100 -import-cold-multiplier=1 -summary-file summary.thinlto.bc main.bc -o main.import.bc -print-imports 2>&1 | FileCheck %s --check-prefix=IMPORTS

// Test that both candidates are ICP'ed and there is no `!VP` in the IR.
// RUN: opt main.import.bc -icp-lto -passes=pgo-icall-prom -S -pass-remarks=pgo-icall-prom 2>&1 | FileCheck %s --check-prefixes=ICP-IR,ICP-REMARK --implicit-check-not="!VP"

// IMPORTS-DAG: main.cpp: Import {{.*}}callee1{{.*}}
// IMPORTS-DAG: main.cpp: Import {{.*}}callee0{{.*}}llvm.[[#]]
// IMPORTS-DAG: main.cpp: Import {{.*}}global_func{{.*}}

// PGOName-DAG: define {{.*}}callee1{{.*}} !prof ![[#]] {
// PGOName-DAG: define internal {{.*}}callee0{{.*}} !prof ![[#]] !PGOFuncName ![[#MD:]] {
// PGOName-DAG: ![[#MD]] = !{!"{{.*}}lib.cpp;{{.*}}callee0{{.*}}"}

// ICP-REMARK: Promote indirect call to {{.*}}callee0{{.*}}llvm.[[#]] with count 1 out of 1
// ICP-REMARK: Promote indirect call to {{.*}}callee1{{.*}} with count 1 out of 1

// ICP-IR: br i1 %[[#]], label %if.true.direct_targ, label %if.false.orig_indirect, !prof ![[#BRANCH_WEIGHT1:]]
// ICP-IR: br i1 %[[#]], label %if.true.direct_targ1, label %if.false.orig_indirect2, !prof ![[#BRANCH_WEIGHT1]]
// ICP-IR: ![[#BRANCH_WEIGHT1]] = !{!"branch_weights", i32 1, i32 0}

//--- lib.h
void global_func();

//--- lib.cpp
#include "lib.h"
static void callee0() {}
void callee1() {}
typedef void (*FPT)();
FPT calleeAddrs[] = {callee0, callee1};
// `global_func`` might call one of two indirect callees. callee0 has internal
// linkage and callee1 has external linkage.
void global_func() {
  FPT fp = calleeAddrs[0];
  fp();
  fp = calleeAddrs[1];
  fp();
}

//--- main.cpp
#include "lib.h"
int main() { global_func(); }
