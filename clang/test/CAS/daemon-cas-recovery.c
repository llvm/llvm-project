// REQUIRES: system-darwin, clang-cc1daemon

// RUN: rm -rf %t && mkdir -p %t

/// Construct a malformed CAS to recovery from.
// RUN: echo "abc" | llvm-cas --cas %t/cas --make-blob --data -
// RUN: rm %t/cas/v1.1/v9.data
// RUN: not llvm-cas --cas %t/cas --validate --check-hash

// RUN: env LLVM_CACHE_CAS_PATH=%t/cas LLVM_CAS_FORCE_VALIDATION=1 %clang-cache \
// RUN:   %clang -fsyntax-only -x c %s

int func(void);
