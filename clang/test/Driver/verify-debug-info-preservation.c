// We support the CC1 options for testing whether each LLVM pass preserves
// original debug info.

// RUN: %clang -g -Xclang -fverify-debuginfo-preserve -### %s 2>&1 \
// RUN:     | FileCheck --check-prefix=VERIFYDIPRESERVE %s

// VERIFYDIPRESERVE: "-fverify-debuginfo-preserve"

// RUN: %clang -g -Xclang -fverify-debuginfo-preserve \
// RUN:     -Xclang -fverify-debuginfo-preserve-export=%t.json -### %s 2>&1 \
// RUN:     | FileCheck --check-prefix=VERIFYDIPRESERVE-JSON-EXPORT %s

// VERIFYDIPRESERVE-JSON-EXPORT: "-fverify-debuginfo-preserve"
// VERIFYDIPRESERVE-JSON-EXPORT: "-fverify-debuginfo-preserve-export={{.*}}"

// RUN: %clang -g -Xclang -fverify-debuginfo-preserve-export=%t.json %s -S -o /dev/null 2>&1 \
// RUN:     | FileCheck --check-prefix=WARN %s

// WARN: warning: ignoring -fverify-debuginfo-preserve-export
