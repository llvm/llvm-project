// RUN: %clang_cc1 -fsyntax-only -Werror %s

// Check input kind association for cuh extension.

__attribute__((host, device)) void hd_fn() {}
