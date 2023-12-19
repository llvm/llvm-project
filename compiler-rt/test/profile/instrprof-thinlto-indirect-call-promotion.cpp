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

// Use lld as linker for more robust test. We need to REQUIRE LLVMgold.so for
// LTO if default linker is GNU ld or gold anyway.
// REQUIRES: lld-available

// Test should fail where linkage-name and mangled-name diverges, see issue https://github.com/llvm/llvm-project/issues/74565).
// Currently, this name divergence happens on Mach-O object file format, or on
// many (but not all) 32-bit Windows systems.
//
// XFAIL: system-darwin
//
// Mark 32-bit Windows as UNSUPPORTED for now as opposed to XFAIL. This test
// should fail on many (but not all) 32-bit Windows systems and succeed on the
// rest. The flexibility in triple string parsing makes it tricky to capture
// both sets accurately. i[3-9]86 specifies arch as Triple::ArchType::x86, (win32|windows)
// specifies OS as Triple::OS::Win32
//
// UNSUPPORTED: target={{i.86.*windows.*}}

// RUN: rm -rf %t && split-file %s %t && cd %t

// Do setup work for all below tests.
// Generate raw profiles from real programs and convert it into indexed profiles.
// Use clangxx_pgogen for IR level instrumentation for C++.
// RUN: %clangxx_pgogen -fuse-ld=lld -O2 lib.cpp main.cpp -o main
// RUN: env LLVM_PROFILE_FILE=main.profraw %run ./main
// RUN: llvm-profdata merge main.profraw -o main.profdata

// Use profile on lib and get bitcode, test that local function callee0 has
// expected !PGOFuncName metadata and external function callee1 doesn't have
// !PGOFuncName metadata. Explicitly skip ICP pass to test ICP happens as
// expected in the IR module that imports functions from lib.
// RUN: %clang -mllvm -disable-icp -fprofile-use=main.profdata -flto=thin -O2 -c lib.cpp -o lib.bc
// RUN: llvm-dis lib.bc -o - | FileCheck %s --check-prefix=PGOName

// Use profile on main and get bitcode.
// RUN: %clang -fprofile-use=main.profdata -flto=thin -O2 -c main.cpp -o main.bc

// Run llvm-lto to get summary file.
// RUN: llvm-lto -thinlto -o summary main.bc lib.bc

// Test the imports of functions. Default import thresholds would work but do
// explicit override to be more futureproof. Note all functions have one basic
// block with a function-entry-count of one, so they are actually hot functions
// per default profile summary hotness cutoff.
// RUN: opt -passes=function-import -import-instr-limit=100 -import-cold-multiplier=1 -summary-file summary.thinlto.bc main.bc -o main.import.bc -print-imports 2>&1 | FileCheck %s --check-prefix=IMPORTS
// Test that '_Z11global_funcv' has indirect calls annotated with value profiles.
// RUN: llvm-dis main.import.bc -o - | FileCheck %s --check-prefix=IR

// Test that both candidates are ICP'ed and there is no `!VP` in the IR.
// RUN: opt main.import.bc -icp-lto -passes=pgo-icall-prom -S -pass-remarks=pgo-icall-prom 2>&1 | FileCheck %s --check-prefixes=ICP-IR,ICP-REMARK --implicit-check-not="!VP"

// IMPORTS: main.cpp: Import _Z7callee1v
// IMPORTS: main.cpp: Import _ZL7callee0v.llvm.[[#]]
// IMPORTS: main.cpp: Import _Z11global_funcv

// PGOName: define {{(dso_local )?}}void @_Z7callee1v() #[[#]] !prof ![[#]] {
// PGOName: define internal void @_ZL7callee0v() #[[#]] !prof ![[#]] !PGOFuncName ![[#MD:]] {
// PGOName: ![[#MD]] = !{!"{{.*}}lib.cpp;_ZL7callee0v"}

// IR-LABEL: define available_externally {{.*}} void @_Z11global_funcv() {{.*}} !prof ![[#]] {
// IR-NEXT: entry:
// IR-NEXT:  %0 = load ptr, ptr @calleeAddrs
// IR-NEXT:  tail call void %0(), !prof ![[#PROF1:]]
// IR-NEXT:  %1 = load ptr, ptr getelementptr inbounds ([2 x ptr], ptr @calleeAddrs,
// IR-NEXT:  tail call void %1(), !prof ![[#PROF2:]]

// The GUID of indirect callee is the MD5 hash of `/path/to/lib.cpp;_ZL7callee0v`
// that depends on the directory. Use [[#]] for its MD5 hash.
// Use {{.*}} for integer types so the test works on 32-bit and 64-bit systems.
// IR: ![[#PROF1]] = !{!"VP", i32 0, {{.*}} 1, {{.*}} [[#]], {{.*}} 1}
// IR: ![[#PROF2]] = !{!"VP", i32 0, {{.*}} 1, {{.*}} -3993653843325621743, {{.*}} 1}

// ICP-REMARK: Promote indirect call to _ZL7callee0v.llvm.[[#]] with count 1 out of 1
// ICP-REMARK: Promote indirect call to _Z7callee1v with count 1 out of 1

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
