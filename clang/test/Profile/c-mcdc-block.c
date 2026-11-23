// RUN: %clang_cc1 -triple %itanium_abi_triple -fblocks %s -o - -emit-llvm -fprofile-instrument=clang -fcoverage-mapping -fcoverage-mcdc -disable-llvm-passes | FileCheck %s -check-prefix=MCDC
// RUN: %clang_cc1 -triple %itanium_abi_triple -fblocks %s -o - -emit-llvm -fprofile-instrument=clang -fcoverage-mapping | FileCheck %s -check-prefix=NOMCDC

// Verify MC/DC coverage for block functions.

typedef int (^blk)(int, int);

blk make(void) {
  return ^(int a, int b) {
    return (a && b);
  };
}

// MCDC-LABEL: @__make_block_invoke(
// MCDC: call void @llvm.instrprof.mcdc.parameters(
// MCDC: call void @llvm.instrprof.mcdc.tvbitmap.update(

// NOMCDC-NOT: instrprof.mcdc
