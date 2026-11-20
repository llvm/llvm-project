// REQUIRES: target={{.*windows-msvc.*}}
//
// Keep Clang's actual records, references and relocations. Over-align both
// kinds of contribution so even a linker that packs them as tightly as it
// can must leave holes *between* records and null slots between references.
// The explicitly ordered contributions below also put padding at the ends.
// RUN: %clang_profgen -O0 -mllvm -enable-value-profiling -S -emit-llvm %s -o %t.ll
// RUN: sed -e '/section "[.]lprfd[$]0"/s/align [0-9][0-9]*/align 128/' \
// RUN:     -e '/section "[.]lprfp[$]M"/s/align [0-9][0-9]*/align 128/' %t.ll > %t.padded.ll
// RUN: FileCheck %s --check-prefix=LAYOUT < %t.padded.ll
// LAYOUT-DAG: section ".lprfd$0"{{.*}}align 128
// LAYOUT-DAG: section ".lprfd$0"{{.*}}align 128
// LAYOUT-DAG: section ".lprfp$M"{{.*}}align 128
// LAYOUT-DAG: section ".lprfp$M"{{.*}}align 128
// RUN: %clang -c %t.padded.ll -o %t.obj
//
// DEFINE: %{check_profile} = rm -rf %t.profiles && \
// DEFINE:   env LLVM_PROFILE_FILE=%t.profiles/%m.profraw %run %t.exe && \
// DEFINE:   llvm-profdata show --all-functions --counts --ic-targets %t.profiles/*.profraw > %t.out && \
// DEFINE:   FileCheck %s --check-prefix=CALLER < %t.out && FileCheck %s --check-prefix=CALLEE < %t.out
// RUN: %clang_profgen -fuse-ld=link %t.obj -o %t.exe && %{check_profile}
// RUN: %if lld-available %{ %clang_profgen -fuse-ld=lld %t.obj -o %t.exe && %{check_profile} %}

// These sort on either side of the indexed .lprfd$0 records, outside the
// legacy $A..$Z range. Neither contribution is a profile record.
#pragma section(".lprfd$.", read, write)
__attribute__((used)) __declspec(allocate(".lprfd$.")) char Before[13] = {0};
#pragma section(".lprfd$1", read, write)
__attribute__((used)) __declspec(allocate(".lprfd$1")) char After[19] = {0};

#pragma section(".lprfp$B", read)
__attribute__((used)) __declspec(allocate(".lprfp$B")) const void *const NullRef = 0;

int callee(int n) {
  if (n)
    return 1;
  return 0;
}

int (*volatile Target)(int) = callee;

int caller(void) {
  int sum = 0;
  for (int i = 0; i != 2; ++i)
    sum += Target(i);
  return sum;
}

void __llvm_profile_reset_counters(void);
int __llvm_profile_dump(void);

// The first dump creates the file; the next two merge into it. Keep all
// three in one process so indirect target addresses are unaffected by ASLR.
__attribute__((no_profile_instrument_function)) int main(void) {
  for (int i = 0; i != 3; ++i) {
    __llvm_profile_reset_counters();
    if (caller() != 1 || __llvm_profile_dump())
      return 1;
  }
  return 0;
}

// CALLER-LABEL: caller:
// CALLER:       Function count: 3{{$}}
// CALLER-NEXT:  Indirect Call Site Count: 1
// CALLER-NEXT:  Block counts: [6]
// CALLER-NEXT:  Indirect Target Results:
// CALLER-NEXT:  [ 0, callee, 6 ] (100.00%)
// CALLER:       Total functions: 2{{$}}

// CALLEE-LABEL: callee:
// CALLEE:       Function count: 6{{$}}
// CALLEE-NEXT:  Indirect Call Site Count: 0
// CALLEE-NEXT:  Block counts: [3]
