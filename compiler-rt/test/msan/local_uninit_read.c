// Tests the opt-in MSan local-uninitialized-read check
// (-fsanitize-memory-local-address-never-taken): flags reads of local
// scalar variables whose address is never taken, for values that are
// not always initialized before the read (ISO C 6.3.2.1p2 indeterminate
// value; the "address never taken" gate has been part of this rule
// since DR338/C11, not C23-specific).
//
// The canonical positive case below is a *dead load* (read, then
// discarded, no branch/return/store on the value) precisely because
// that pattern reaches none of MSan's other checked sinks on its own --
// unlike `return x;`, which vanilla -fsanitize=memory already flags via
// its own return-value shadow tracking, independent of this feature.
// Using `return x;` here would not actually test this code path; this
// was confirmed empirically before writing this test.
//
// Positive: dead load of a never-initialized scalar -> warns.
// RUN: %clang_msan -fsanitize-memory-local-address-never-taken -O0 -Wno-unused-value %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s --check-prefix=WARN
//
// Negative: same dead load, but flag not passed -> vanilla MSan misses it.
// RUN: %clang_msan -O0 -Wno-unused-value %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s --check-prefix=SILENT --allow-empty
//
// Negative: address taken -> not flagged even though never initialized.
// RUN: %clang_msan -fsanitize-memory-local-address-never-taken -O0 -Wno-unused-value -DADDR_TAKEN %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s --check-prefix=SILENT --allow-empty
//
// Negative: initialized via GCC asm output operand -> not flagged.
// RUN: %clang_msan -fsanitize-memory-local-address-never-taken -O0 -Wno-unused-value -DASM_INIT %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s --check-prefix=SILENT --allow-empty
//
// Negative: aggregate type (struct) -> out of scope (scalars only),
// not flagged either with or without the feature.
// RUN: %clang_msan -fsanitize-memory-local-address-never-taken -O0 -Wno-unused-value -DAGGREGATE %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s --check-prefix=SILENT --allow-empty
//
// Path-sensitive: conditionally-assigned scalar, dead load. Warns only
// when the uninitialized path is actually taken at runtime (no extra
// argv), silent when the initializing path was taken (extra argv
// present). This works via MSan's own runtime shadow propagation
// (genuinely path-sensitive via phi-node shadow merging at branch
// joins) -- UninitLocalVarVisitor's own AST-level candidate detection
// is a coarse whole-function existence pre-filter, not itself path
// sensitive; it does not erase a candidate merely because it is
// assigned somewhere in the function, and relies on MSan's shadow to
// correctly reflect whether that assignment actually dominated this
// particular execution.
//
// Note: deliberately not testing __builtin_unreachable() as a
// substitute for the "always initialized" path here. Reaching a branch
// marked unreachable at runtime is undefined behavior independent of
// this feature; any apparent pass/fail from doing so is a coincidence
// of code layout, not attributable to this check, and was confirmed
// unreliable before this test was written.
// RUN: %clang_msan -fsanitize-memory-local-address-never-taken -O0 -Wno-unused-value -DCONDITIONAL %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s --check-prefix=WARN
// RUN: %run %t extra_arg 2>&1 | FileCheck %s --check-prefix=SILENT --allow-empty
//
// Negative: C++ excluded entirely by the language gate
// (!CGF.getLangOpts().CPlusPlus in VisitDeclRefExpr).
// RUN: %clangxx_msan -fsanitize-memory-local-address-never-taken -O0 -Wno-unused-value -x c++ %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s --check-prefix=SILENT --allow-empty

#if defined(ADDR_TAKEN)
int main() {
  int x;
  int *p = &x;
  (void)p;
  x;
  return 0;
}
#elif defined(ASM_INIT)
int main() {
  int x;
#if defined(__x86_64__) || defined(__i386__)
  __asm__("movl $0, %0" : "=r"(x));
#else
  x = 0;
#endif
  x;
  return 0;
}
#elif defined(AGGREGATE)
struct S { int a; };
int main() {
  struct S s;
  s.a;
  return 0;
}
#elif defined(CONDITIONAL)
int main(int argc, char **argv) {
  int x;
  if (argc > 1)
    x = 5;
  x;
  return 0;
}
#else
int main() {
  int x;
  x;
  return 0;
}
#endif

// WARN: WARNING: MemorySanitizer: use-of-uninitialized-value
// SILENT-NOT: MemorySanitizer
