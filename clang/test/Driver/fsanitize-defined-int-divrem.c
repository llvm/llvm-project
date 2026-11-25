/// Under -fdefined-int-divrem, which is the default for OpenCL C and C++ for
/// OpenCL, integer division by zero and INT_MIN / -1 are not undefined, so
/// -fsanitize=undefined leaves out integer-divide-by-zero and
/// signed-integer-overflow-divide. Asking for them in any other way still
/// enables them, and recover and trap work as always.

/// -fsanitize=undefined, -fdefined-int-divrem in effect.
// RUN: %clang -### -c --target=x86_64-linux -x cl -fsanitize=undefined %s 2>&1 | FileCheck %s --check-prefix=DROP
// RUN: %clang -### -c --target=x86_64-linux -x cl-header -fsanitize=undefined %s 2>&1 | FileCheck %s --check-prefix=DROP
// RUN: %clang -### -c --target=x86_64-linux -x clcpp -fsanitize=undefined %s 2>&1 | FileCheck %s --check-prefix=DROP
// RUN: %clang -### -c --target=x86_64-linux -cl-std=CL3.0 -fsanitize=undefined %s 2>&1 | FileCheck %s --check-prefix=DROP
// RUN: %clang -### -c --target=x86_64-linux -fdefined-int-divrem -fsanitize=undefined %s 2>&1 | FileCheck %s --check-prefix=DROP
// RUN: %clang -### -c --target=x86_64-linux -fdefined-int-divrem -x c++ -fsanitize=undefined %s 2>&1 | FileCheck %s --check-prefix=DROP
// RUN: %clang -### -c --target=x86_64-linux -x cl -fno-defined-int-divrem -fdefined-int-divrem -fsanitize=undefined %s 2>&1 | FileCheck %s --check-prefix=DROP
// RUN: %clang -### -c --target=x86_64-linux -x cl -fsanitize=undefined-trap %s 2>&1 | FileCheck %s --check-prefix=DROP
/// cc1 gets -Xclang arguments after the driver's -f[no-]defined-int-divrem,
/// so they decide.
// RUN: %clang -### -c --target=x86_64-linux -fsanitize=undefined -Xclang -fdefined-int-divrem %s 2>&1 | FileCheck %s --check-prefix=DROP
// RUN: %clang -### -c --target=x86_64-linux -fsanitize=undefined -Xclang=-fdefined-int-divrem %s 2>&1 | FileCheck %s --check-prefix=DROP
// RUN: %clang -### -c --target=x86_64-linux -x cl -fno-defined-int-divrem -Xclang -fdefined-int-divrem -fsanitize=undefined %s 2>&1 | FileCheck %s --check-prefix=DROP
// RUN: %clang -### -c --target=x86_64-linux -Xclang -cl-std=CL3.0 -fsanitize=undefined %s 2>&1 | FileCheck %s --check-prefix=DROP
/// A request that a later -fno-sanitize= cancels does not count.
// RUN: %clang -### -c --target=x86_64-linux -x cl -fsanitize=integer-divide-by-zero,signed-integer-overflow -fno-sanitize=integer-divide-by-zero,signed-integer-overflow -fsanitize=undefined %s 2>&1 | FileCheck %s --check-prefix=DROP
/// Only the enabled set changes. The recover and merge lists still have the two
/// checks, which does nothing unless they are enabled.
// DROP: "-fsanitize={{[^"]*}},function,nonnull-attribute,{{[^"]*}},signed-integer-overflow-arith,unreachable,vla-bound"
// DROP: "-fsanitize-recover={{[^"]*}},function,integer-divide-by-zero,nonnull-attribute,{{[^"]*}},signed-integer-overflow-arith,signed-integer-overflow-divide,vla-bound"
// DROP: "-fsanitize-merge={{[^"]*}},function,integer-divide-by-zero,nonnull-attribute,{{[^"]*}},signed-integer-overflow-arith,signed-integer-overflow-divide,unreachable,vla-bound"

/// -fsanitize=undefined, -fdefined-int-divrem not in effect.
// RUN: %clang -### -c --target=x86_64-linux -fsanitize=undefined %s 2>&1 | FileCheck %s --check-prefix=KEEP
// RUN: %clang -### -c --target=x86_64-linux -x c++ -fsanitize=undefined %s 2>&1 | FileCheck %s --check-prefix=KEEP
// RUN: %clang -### -c --target=x86_64-linux -fdefined-int-divrem -fno-defined-int-divrem -fsanitize=undefined %s 2>&1 | FileCheck %s --check-prefix=KEEP
// RUN: %clang -### -c --target=x86_64-linux -x cl -fno-defined-int-divrem -fsanitize=undefined %s 2>&1 | FileCheck %s --check-prefix=KEEP
// RUN: %clang -### -c --target=x86_64-linux -x cl -Xclang -fno-defined-int-divrem -fsanitize=undefined %s 2>&1 | FileCheck %s --check-prefix=KEEP
// RUN: %clang -### -c --target=x86_64-linux -x cl -Xclang=-fno-defined-int-divrem -fsanitize=undefined %s 2>&1 | FileCheck %s --check-prefix=KEEP
// RUN: %clang -### -c --target=x86_64-linux -x cl -Xclang -fno-defined-int-divrem -fdefined-int-divrem -fsanitize=undefined %s 2>&1 | FileCheck %s --check-prefix=KEEP
/// -Xclang option values that happen to match flag names are not flags.
// RUN: %clang -### -c --target=x86_64-linux -Xclang -main-file-name -Xclang -fdefined-int-divrem -fsanitize=undefined %s 2>&1 | FileCheck %s --check-prefix=KEEP
// RUN: %clang -### -c --target=x86_64-linux -Xclang -main-file-name -Xclang -cl-std=CL3.0 -fsanitize=undefined %s 2>&1 | FileCheck %s --check-prefix=KEEP
/// Explicit requests, also in the same list as undefined.
// RUN: %clang -### -c --target=x86_64-linux -x cl -fsanitize=undefined,integer-divide-by-zero,signed-integer-overflow-divide %s 2>&1 | FileCheck %s --check-prefix=KEEP
// RUN: %clang -### -c --target=x86_64-linux -x cl -fsanitize=integer-divide-by-zero,signed-integer-overflow -fsanitize=undefined %s 2>&1 | FileCheck %s --check-prefix=KEEP
// KEEP: "-fsanitize={{[^"]*}},function,integer-divide-by-zero,nonnull-attribute,{{[^"]*}},signed-integer-overflow-arith,signed-integer-overflow-divide,unreachable,vla-bound"
// KEEP: "-fsanitize-recover={{[^"]*}},function,integer-divide-by-zero,nonnull-attribute,{{[^"]*}},signed-integer-overflow-arith,signed-integer-overflow-divide,vla-bound"

// RUN: %clang -### -c --target=x86_64-linux -x cl -fsanitize=undefined -fsanitize=integer %s 2>&1 | FileCheck %s --check-prefix=INTEGER
// INTEGER: "-fsanitize={{[^"]*}},function,integer-divide-by-zero,nonnull-attribute,{{[^"]*}},signed-integer-overflow-arith,signed-integer-overflow-divide,unreachable,{{[^"]*}}"

// RUN: %clang -### -c --target=x86_64-linux -x cl -fsanitize=undefined,integer-divide-by-zero %s 2>&1 | FileCheck %s --check-prefix=ZERO
// ZERO: "-fsanitize={{[^"]*}},function,integer-divide-by-zero,nonnull-attribute,{{[^"]*}},signed-integer-overflow-arith,unreachable,vla-bound"

// RUN: %clang -### -c --target=x86_64-linux -x cl -fsanitize=undefined -fsanitize=signed-integer-overflow %s 2>&1 | FileCheck %s --check-prefix=OVERFLOW
// OVERFLOW: "-fsanitize={{[^"]*}},function,nonnull-attribute,{{[^"]*}},signed-integer-overflow-arith,signed-integer-overflow-divide,unreachable,vla-bound"

/// The trap, merge, skip-hot-cutoff and annotate-debug-info lists are computed
/// as without -fdefined-int-divrem; an explicit check can trap.
// RUN: %clang -### -c --target=x86_64-linux -x cl -fsanitize=undefined,integer-divide-by-zero -fsanitize-trap=undefined -fsanitize-skip-hot-cutoff=undefined=0.5 -fsanitize-annotate-debug-info=undefined %s 2>&1 | FileCheck %s --check-prefix=TRAP
// TRAP:     "-fsanitize={{[^"]*}},function,integer-divide-by-zero,nonnull-attribute,{{[^"]*}},signed-integer-overflow-arith,unreachable,vla-bound"
// TRAP-NOT: -fsanitize-recover
// TRAP:     "-fsanitize-trap={{[^"]*}},function,integer-divide-by-zero,nonnull-attribute,{{[^"]*}},signed-integer-overflow-arith,signed-integer-overflow-divide,unreachable,vla-bound"
// TRAP:     "-fsanitize-merge={{[^"]*}},function,integer-divide-by-zero,nonnull-attribute,{{[^"]*}},signed-integer-overflow-arith,signed-integer-overflow-divide,unreachable,vla-bound"
// TRAP:     "-fsanitize-skip-hot-cutoff={{[^"]*}},function=0.5,integer-divide-by-zero=0.5,nonnull-attribute=0.5,{{[^"]*}},signed-integer-overflow-arith=0.5,signed-integer-overflow-divide=0.5,unreachable=0.5,vla-bound=0.5"
// TRAP:     "-fsanitize-annotate-debug-info={{[^"]*}},function,integer-divide-by-zero,nonnull-attribute,{{[^"]*}},signed-integer-overflow-arith,signed-integer-overflow-divide,unreachable,vla-bound"

/// A check requested only from cc1 (-Xclang -fsanitize=; cc1 adds up all
/// -fsanitize= arguments) recovers, traps or aborts like the driver spelling.
// RUN: %clang -### -c --target=x86_64-linux -x cl -fsanitize=undefined -Xclang -fsanitize=integer-divide-by-zero %s 2>&1 | FileCheck %s --check-prefix=XCLANG-RECOVER
// XCLANG-RECOVER:      "-fsanitize=integer-divide-by-zero"
// XCLANG-RECOVER-SAME: "-fsanitize={{[^"]*}},function,nonnull-attribute,{{[^"]*}},signed-integer-overflow-arith,unreachable,vla-bound"
// XCLANG-RECOVER-SAME: "-fsanitize-recover={{[^"]*}},function,integer-divide-by-zero,nonnull-attribute,{{[^"]*}}"
// RUN: %clang -### -c --target=x86_64-linux -x cl -fsanitize=undefined -fsanitize-trap=undefined -Xclang -fsanitize=integer-divide-by-zero %s 2>&1 | FileCheck %s --check-prefix=XCLANG-TRAP
// XCLANG-TRAP:      "-fsanitize=integer-divide-by-zero"
// XCLANG-TRAP-SAME: "-fsanitize={{[^"]*}},function,nonnull-attribute,{{[^"]*}},signed-integer-overflow-arith,unreachable,vla-bound"
// XCLANG-TRAP-NOT:  -fsanitize-recover
// XCLANG-TRAP-SAME: "-fsanitize-trap={{[^"]*}},function,integer-divide-by-zero,nonnull-attribute,{{[^"]*}}"
// RUN: %clang -### -c --target=x86_64-linux -x cl -fsanitize=undefined -fno-sanitize-recover=integer-divide-by-zero -Xclang -fsanitize=integer-divide-by-zero %s 2>&1 | FileCheck %s --check-prefix=XCLANG-FATAL
// XCLANG-FATAL:      "-fsanitize=integer-divide-by-zero"
// XCLANG-FATAL-SAME: "-fsanitize={{[^"]*}},function,nonnull-attribute,{{[^"]*}},signed-integer-overflow-arith,unreachable,vla-bound"
// XCLANG-FATAL-SAME: "-fsanitize-recover={{[^"]*}},function,nonnull-attribute,{{[^"]*}},signed-integer-overflow-arith,signed-integer-overflow-divide,vla-bound"

/// -fno-sanitize-recover keeps working as always.
// RUN: %clang -### -c --target=x86_64-linux -x cl -fsanitize=undefined,signed-integer-overflow-divide -fno-sanitize-recover=all %s 2>&1 | FileCheck %s --check-prefix=FATAL
// FATAL:     "-fsanitize={{[^"]*}},function,nonnull-attribute,{{[^"]*}},signed-integer-overflow-arith,signed-integer-overflow-divide,unreachable,vla-bound"
// FATAL-NOT: -fsanitize-recover

/// The decision is made per input.
// RUN: %clang -### -c --target=x86_64-linux -fsanitize=undefined -x c %s -x cl %s 2>&1 | FileCheck %s --check-prefix=MIXED
// MIXED:      "-cc1"
// MIXED-SAME: "-fsanitize={{[^"]*}},function,integer-divide-by-zero,nonnull-attribute,{{[^"]*}},signed-integer-overflow-arith,signed-integer-overflow-divide,unreachable,vla-bound"
// MIXED-SAME: "-x" "c"
// MIXED:      "-cc1"
// MIXED-SAME: "-fsanitize={{[^"]*}},function,nonnull-attribute,{{[^"]*}},signed-integer-overflow-arith,unreachable,vla-bound"
// MIXED-SAME: "-x" "cl"

/// When every other check of -fsanitize=undefined is disabled, dropping the two
/// division checks leaves no -fsanitize= flag on cc1.
// RUN: %clang -### -c --target=x86_64-linux -x cl -fsanitize=undefined -fno-sanitize=alignment,array-bounds,bool,builtin,enum,float-cast-overflow,function,nonnull-attribute,null,object-size,pointer-overflow,return,returns-nonnull-attribute,shift,signed-integer-overflow-arith,unreachable,vla-bound %s 2>&1 | FileCheck %s --check-prefix=EMPTY
// EMPTY:      "-cc1"
// EMPTY-NOT:  "-fsanitize="
// EMPTY-SAME: "-x" "cl"
