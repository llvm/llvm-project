// RUN: %clang -target x86_64-linux-gnu -fsanitize=address -fsanitize-coverage=0 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANITIZE-COVERAGE-0
// RUN: %clang -target x86_64-linux-gnu -fsanitize=address -fsanitize-coverage=edge -fsanitize-coverage=0 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANITIZE-COVERAGE-0
// RUN: %clang -target x86_64-linux-gnu -fsanitize=address %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANITIZE-COVERAGE-0
// CHECK-SANITIZE-COVERAGE-0-NOT: fsanitize-coverage-type
// CHECK-SANITIZE-COVERAGE-0: -fsanitize=address

// RUN: %clang -target x86_64-linux-gnu -fsanitize=address -fsanitize-coverage=func,trace-pc %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANITIZE-COVERAGE-FUNC
// RUN: %clang -target x86_64-linux-gnu -fsanitize=kernel-address -fsanitize-coverage=func,trace-pc %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANITIZE-COVERAGE-FUNC
// RUN: %clang -target x86_64-linux-gnu -fsanitize=hwaddress -fsanitize-coverage=func,trace-pc %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANITIZE-COVERAGE-FUNC
// RUN: %clang -target x86_64-linux-gnu -fsanitize=kernel-hwaddress -fsanitize-coverage=func,trace-pc %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANITIZE-COVERAGE-FUNC
// RUN: %clang -target x86_64-linux-gnu -fsanitize=memory -fsanitize-coverage=func,trace-pc %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANITIZE-COVERAGE-FUNC
// RUN: %clang -target x86_64-linux-gnu -fsanitize=kernel-memory -fsanitize-coverage=func,trace-pc %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANITIZE-COVERAGE-FUNC
// RUN: %clang -target x86_64-linux-gnu -fsanitize=leak -fsanitize-coverage=func,trace-pc %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANITIZE-COVERAGE-FUNC
// RUN: %clang -target x86_64-linux-gnu -fsanitize=undefined -fsanitize-coverage=func,trace-pc %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANITIZE-COVERAGE-FUNC
// RUN: %clang -target x86_64-linux-gnu -fsanitize=bounds -fsanitize-coverage=func,trace-pc %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANITIZE-COVERAGE-FUNC
// RUN: %clang -target x86_64-linux-gnu -fsanitize=bool -fsanitize-coverage=func,trace-pc %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANITIZE-COVERAGE-FUNC
// RUN: %clang -target x86_64-linux-gnu -fsanitize=dataflow -fsanitize-coverage=func,trace-pc %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANITIZE-COVERAGE-FUNC
// RUN: %clang -target x86_64-linux-gnu -fsanitize=thread -fsanitize-coverage=func,trace-pc %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANITIZE-COVERAGE-FUNC
// RUN: %clang -target %itanium_abi_triple -fsanitize=float-divide-by-zero -fsanitize-coverage=func,trace-pc %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANITIZE-COVERAGE-FUNC
// RUN: %clang -target x86_64-linux-gnu                     -fsanitize-coverage=func,trace-pc %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANITIZE-COVERAGE-FUNC
// CHECK-SANITIZE-COVERAGE-FUNC: fsanitize-coverage-type=1

// RUN: %clang -target x86_64-linux-gnu -fsanitize=address -fsanitize-coverage=bb %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANITIZE-COVERAGE-BB
// CHECK-SANITIZE-COVERAGE-BB: fsanitize-coverage-type=2

// RUN: %clang -target x86_64-linux-gnu -fsanitize=address -fsanitize-coverage=edge %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANITIZE-COVERAGE-EDGE
// CHECK-SANITIZE-COVERAGE-EDGE: fsanitize-coverage-type=3

// RUN: %clang -target x86_64-linux-gnu -fsanitize=address -fsanitize-coverage=edge,indirect-calls %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANITIZE-COVERAGE-FUNC_INDIR
// CHECK-SANITIZE-COVERAGE-FUNC_INDIR: fsanitize-coverage-type=3
// CHECK-SANITIZE-COVERAGE-FUNC_INDIR: fsanitize-coverage-indirect-calls

// RUN: %clang -target x86_64-linux-gnu -fsanitize=address -fsanitize-coverage=1 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANITIZE-COVERAGE-1
// CHECK-SANITIZE-COVERAGE-1: warning: argument '-fsanitize-coverage=1' is deprecated, use '-fsanitize-coverage=trace-pc-guard' instead

// RUN: %clang -target x86_64-linux-gnu -fsanitize=address -fsanitize-coverage=func %s -### 2>&1 | FileCheck %s --check-prefix=CHECK_FUNC_BB_EDGE_DEPRECATED
// RUN: %clang -target x86_64-linux-gnu -fsanitize=address -fsanitize-coverage=bb %s -### 2>&1 | FileCheck %s --check-prefix=CHECK_FUNC_BB_EDGE_DEPRECATED
// RUN: %clang -target x86_64-linux-gnu -fsanitize=address -fsanitize-coverage=edge %s -### 2>&1 | FileCheck %s --check-prefix=CHECK_FUNC_BB_EDGE_DEPRECATED
// CHECK_FUNC_BB_EDGE_DEPRECATED: warning: argument '-fsanitize-coverage=[func|bb|edge]' is deprecated, use '-fsanitize-coverage=[func|bb|edge],[trace-pc-guard|trace-pc]' instead

// RUN: %clang -target x86_64-linux-gnu -fsanitize=address -fsanitize-coverage=edge,indirect-calls,trace-pc,trace-cmp,trace-loads,trace-stores,trace-div,trace-gep %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANITIZE-COVERAGE-FEATURES
// CHECK-SANITIZE-COVERAGE-FEATURES: -fsanitize-coverage-type=3
// CHECK-SANITIZE-COVERAGE-FEATURES: -fsanitize-coverage-indirect-calls
// CHECK-SANITIZE-COVERAGE-FEATURES: -fsanitize-coverage-trace-cmp
// CHECK-SANITIZE-COVERAGE-FEATURES: -fsanitize-coverage-trace-div
// CHECK-SANITIZE-COVERAGE-FEATURES: -fsanitize-coverage-trace-gep
// CHECK-SANITIZE-COVERAGE-FEATURES: -fsanitize-coverage-trace-pc
// CHECK-SANITIZE-COVERAGE-FEATURES: -fsanitize-coverage-trace-loads
// CHECK-SANITIZE-COVERAGE-FEATURES: -fsanitize-coverage-trace-stores

// RUN: %clang -target x86_64-linux-gnu -fsanitize=address -fsanitize-coverage=func,edge,indirect-calls,trace-cmp -fno-sanitize-coverage=edge,indirect-calls %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-MASK
// CHECK-MASK: -fsanitize-coverage-type=1
// CHECK-MASK: -fsanitize-coverage-trace-cmp
// CHECK-MASK-NOT: -fsanitize-coverage-

// RUN: %clang -target x86_64-linux-gnu -fsanitize=address -fsanitize-coverage=foobar %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-INVALID-VALUE
// CHECK-INVALID-VALUE: error: unsupported argument 'foobar' to option 'fsanitize-coverage='

// RUN: %clang -target x86_64-linux-gnu -fsanitize=address -fsanitize-coverage=func -fsanitize-coverage=edge %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-INCOMPATIBLE
// CHECK-INCOMPATIBLE: error: invalid argument '-fsanitize-coverage=func' not allowed with '-fsanitize-coverage=edge'

// RUN: %clang -target x86_64-linux-gnu -fsanitize-coverage=8bit-counters %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-8BIT
// CHECK-8BIT: warning: argument '-fsanitize-coverage=8bit-counters' is deprecated, use '-fsanitize-coverage=trace-pc-guard' instead
// RUN: %clang -target x86_64-linux-gnu -fsanitize-coverage=trace-bb %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TRACE-BB
// CHECK-TRACE-BB: warning: argument '-fsanitize-coverage=trace-bb' is deprecated, use '-fsanitize-coverage=trace-pc-guard' instead

// RUN: %clang -target x86_64-linux-gnu -fsanitize-coverage=trace-pc %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TRACE_PC_EDGE
// RUN: %clang -target x86_64-linux-gnu -fsanitize-coverage=edge,trace-pc %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TRACE_PC_EDGE
// CHECK-TRACE_PC_EDGE: -fsanitize-coverage-type=3
// CHECK-TRACE_PC_EDGE: -fsanitize-coverage-trace-pc
// RUN: %clang -target x86_64-linux-gnu -fsanitize-coverage=func,trace-pc %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TRACE_PC_FUNC
// CHECK-TRACE_PC_FUNC: -fsanitize-coverage-type=1
// CHECK-TRACE_PC_FUNC: -fsanitize-coverage-trace-pc

// RUN: %clang -target x86_64-linux-gnu -fsanitize-coverage=trace-pc-guard %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TRACE_PC_GUARD_EDGE
// RUN: %clang -target x86_64-linux-gnu -fsanitize-coverage=edge,trace-pc-guard %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TRACE_PC_GUARD_EDGE
// CHECK-TRACE_PC_GUARD_EDGE: -fsanitize-coverage-type=3
// CHECK-TRACE_PC_GUARD_EDGE: -fsanitize-coverage-trace-pc-guard
// RUN: %clang -target x86_64-linux-gnu -fsanitize-coverage=func,trace-pc-guard %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TRACE_PC_GUARD_FUNC
// CHECK-TRACE_PC_GUARD_FUNC: -fsanitize-coverage-type=1
// CHECK-TRACE_PC_GUARD_FUNC: -fsanitize-coverage-trace-pc-guard

// RUN: %clang -target x86_64-linux-gnu -fsanitize-coverage=stack-depth %s \
// RUN:     -### 2>&1 | FileCheck %s --check-prefix=CHECK-STACK-DEPTH
// RUN: %clang -target x86_64-linux-gnu \
// RUN:     -fsanitize-coverage=trace-pc-guard,stack-depth %s -### 2>&1 | \
// RUN:     FileCheck %s --check-prefix=CHECK-STACK-DEPTH-PC-GUARD
// CHECK-STACK-DEPTH: -fsanitize-coverage-type=1
// CHECK-STACK-DEPTH: -fsanitize-coverage-stack-depth
// CHECK-STACK-DEPTH-PC-GUARD: -fsanitize-coverage-type=3
// CHECK-STACK-DEPTH-PC-GUARD: -fsanitize-coverage-trace-pc-guard
// CHECK-STACK-DEPTH-PC-GUARD: -fsanitize-coverage-stack-depth

// RUN: %clang -target x86_64-linux-gnu -fsanitize=address -fsanitize-coverage=trace-cmp,indirect-calls %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-TYPE-NECESSARY
// CHECK-NO-TYPE-NECESSARY-NOT: error:
// CHECK-NO-TYPE-NECESSARY: -fsanitize-coverage-indirect-calls
// CHECK-NO-TYPE-NECESSARY: -fsanitize-coverage-trace-cmp

// RUN: %clang -target x86_64-linux-gnu -fsanitize=address -fsanitize-coverage=func -fsanitize-coverage=trace-cmp %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-EXTEND-LEGACY
// CHECK-EXTEND-LEGACY: -fsanitize-coverage-type=1
// CHECK-EXTEND-LEGACY: -fsanitize-coverage-trace-cmp

// RUN: %clang -target x86_64-linux-gnu -fsanitize-coverage=no-prune,trace-pc %s -### 2>&1 | FileCheck %s --check-prefix=CHECK_NOPRUNE
// RUN: %clang -target x86_64-linux-gnu -fsanitize-coverage=no-prune,func,trace-pc-guard %s -### 2>&1 | FileCheck %s --check-prefix=CHECK_NOPRUNE
// CHECK_NOPRUNE: -fsanitize-coverage-no-prune

// RUN: %clang -target x86_64-linux-gnu -fsanitize-coverage=inline-8bit-counters %s -### 2>&1 | FileCheck %s --check-prefix=CHECK_INLINE8BIT
// RUN: %clang -target x86_64-linux-gnu -fsanitize-coverage=bb,inline-8bit-counters %s -### 2>&1 | FileCheck %s --check-prefix=CHECK_INLINE8BIT
// CHECK_INLINE8BIT-NOT: warning:
// CHECK_INLINE8BIT: -fsanitize-coverage-inline-8bit-counters

// RUN: %clang -target x86_64-linux-gnu -fsanitize-coverage=inline-8bit-counters,pc-table %s -### 2>&1 | FileCheck %s --check-prefix=CHECK_PC_TABLE_FOR_INLINE8BIT
// RUN: %clang -target x86_64-linux-gnu -fsanitize-coverage=trace-pc-guard,pc-table %s -### 2>&1 | FileCheck %s --check-prefix=CHECK_PC_TABLE_FOR_INLINE8BIT
// CHECK_PC_TABLE_FOR_INLINE8BIT: -fsanitize-coverage-pc-table

// RUN: %clang -target x86_64-linux-gnu -fsanitize-coverage=inline-bool-flag %s -### 2>&1 | FileCheck %s --check-prefix=CHECK_INLINE_BOOL_FLAG
// RUN: %clang -target x86_64-linux-gnu -fsanitize-coverage=bb,inline-bool-flag %s -### 2>&1 | FileCheck %s --check-prefix=CHECK_INLINE_BOOL_FLAG
// CHECK_INLINE_BOOL_FLAG-NOT: warning:
// CHECK_INLINE_BOOL_FLAG: -fsanitize-coverage-inline-bool-flag

// RUN: %clang -target x86_64-linux-gnu -fsanitize-coverage=inline-bool-flag,pc-table %s -### 2>&1 | FileCheck %s --check-prefix=CHECK_PC_TABLE_FOR_INLINEBOOL
// RUN: %clang -target x86_64-linux-gnu -fsanitize-coverage=trace-pc-guard,pc-table %s -### 2>&1 | FileCheck %s --check-prefix=CHECK_PC_TABLE_FOR_INLINEBOOL
// CHECK_PC_TABLE_FOR_INLINEBOOL: -fsanitize-coverage-pc-table

// RUN: %clang_cl --target=i386-pc-win32 -fsanitize=address -fsanitize-coverage=func,trace-pc-guard -c -### -- %s 2>&1 | FileCheck %s -check-prefix=CLANG-CL-COVERAGE
// CLANG-CL-COVERAGE-NOT: error:
// CLANG-CL-COVERAGE-NOT: warning:
// CLANG-CL-COVERAGE-NOT: argument unused
// CLANG-CL-COVERAGE-NOT: unknown argument
// CLANG-CL-COVERAGE: -fsanitize-coverage-type=1
// CLANG-CL-COVERAGE: -fsanitize=address

// RUN: %clang -target x86_64-linux-gnu -fsanitize=safe-stack -fsanitize-coverage=trace-pc-guard %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-VS-SAFESTACK
// CHECK-VS-SAFESTACK: -fsanitize-coverage-trace-pc-guard
// CHECK-VS-SAFESTACK: -fsanitize=safe-stack

// RUN: %clang -target x86_64-linux-gnu -fsanitize=safe-stack -fsanitize-coverage=trace-pc-guard -fno-sanitize=safe-stack %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-SAFESTACK
// CHECK-NO-SAFESTACK-NOT: error:
// CHECK-NO-SAFESTACK-NOT: warning:
// CHECK-NO-SAFESTACK-NOT: argument unused
// CHECK-NO-SAFESTACK-NOT: unknown argument
// CHECK-NO-SAFESTACK-NOT: -fsanitize=safe-stack
// CHECK-NO-SAFESTACK: -fsanitize-coverage-trace-pc-guard

// RUN: %clang -target x86_64-linux-gnu -fsanitize=shadow-call-stack -fsanitize-coverage=trace-pc-guard %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-VS-SHADOWCALLSTACK
// CHECK-VS-SHADOWCALLSTACK: -fsanitize-coverage-trace-pc-guard
// CHECK-VS-SHADOWCALLSTACK: -fsanitize=shadow-call-stack

// RUN: %clang -target x86_64-linux-gnu -fsanitize=shadow-call-stack -fsanitize-coverage=trace-pc-guard -fno-sanitize=shadow-call-stack %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-SAFESTACK
// CHECK-NO-SHADOWCALLSTACK-NOT: error:
// CHECK-NO-SHADOWCALLSTACK-NOT: warning:
// CHECK-NO-SHADOWCALLSTACK-NOT: argument unused
// CHECK-NO-SHADOWCALLSTACK-NOT: unknown argument
// CHECK-NO-SHADOWCALLSTACK-NOT: -fsanitize=shadow-call-stack
// CHECK-NO-SHADOWCALLSTACK: -fsanitize-coverage-trace-pc-guard
