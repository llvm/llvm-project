// Windows linkers can leave zero-filled gaps around COFF profile names blocks.
// A gap before the first block used to be read as an empty name. The runtime
// skips leading padding; the decoder skips padding after each block.
// Zeros inside an encoded block are data.
//
// Compile this file twice so main and helper each have a real names block.
// Move only helper's block from $M to $P: COFF sorts by subsection suffix,
// so the layout between the runtime's $A and $Z sentinels is deterministic:
//
//   $B: zeros | $M: main | $N: zeros | $P: helper | $Y: zeros
//
// Check both names and their distinct counts (main: 1, helper: 2) so the
// test fails if either block is lost or a count is attributed incorrectly.

// RUN: %clang_profgen -O0 -c %s -o %t.main.obj
// RUN: %clang_profgen -O0 -DHELPER -S -emit-llvm %s -o %t.helper.ll
// RUN: sed 's/\.lprfn[$]M/.lprfn$P/g' %t.helper.ll > %t.ordered.ll
// RUN: %clang -c %t.ordered.ll -o %t.helper.obj
// RUN: %clang_profgen %t.main.obj %t.helper.obj -o %t.exe
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t.exe
// RUN: llvm-profdata show --counts --function=main %t.profraw | FileCheck %s --check-prefix=MAIN
// RUN: llvm-profdata show --counts --function=helper %t.profraw | FileCheck %s --check-prefix=HELPER

#ifdef HELPER
void helper(void) {}
#else
void helper(void);
#  pragma section(".lprfn$B", read)
#  pragma section(".lprfn$N", read)
#  pragma section(".lprfn$Y", read)
__declspec(allocate(".lprfn$B")) const char Before[3] = {0};
__declspec(allocate(".lprfn$N")) const char Between[5] = {0};
__declspec(allocate(".lprfn$Y")) const char After[7] = {0};

int main(void) {
  helper();
  helper();
  return 0;
}
#endif

// MAIN-LABEL: main:
// MAIN:       Function count: 1{{$}}
// HELPER-LABEL: helper:
// HELPER:       Function count: 2{{$}}