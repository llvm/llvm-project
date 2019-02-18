// RUN: not %clang_cc1 -fclang-abi-compat=banana %s -fsyntax-only 2>&1 | FileCheck --check-prefix=INVALID %s
// RUN: not %clang_cc1 -fclang-abi-compat=2.9 %s -fsyntax-only 2>&1 | FileCheck --check-prefix=INVALID %s
// RUN: not %clang_cc1 -fclang-abi-compat=42 %s -fsyntax-only 2>&1 | FileCheck --check-prefix=INVALID %s
// RUN: not %clang_cc1 -fclang-abi-compat=3.10 %s -fsyntax-only 2>&1 | FileCheck --check-prefix=INVALID %s
// RUN: not %clang_cc1 -fclang-abi-compat=4.1 %s -fsyntax-only 2>&1 | FileCheck --check-prefix=INVALID %s
// RUN: not %clang_cc1 -fclang-abi-compat=04 %s -fsyntax-only 2>&1 | FileCheck --check-prefix=INVALID %s
// RUN: not %clang_cc1 -fclang-abi-compat=4. %s -fsyntax-only 2>&1 | FileCheck --check-prefix=INVALID %s
// RUN: not %clang_cc1 -fclang-abi-compat=4.00 %s -fsyntax-only 2>&1 | FileCheck --check-prefix=INVALID %s
// INVALID: error: invalid value '{{.*}}' in '-fclang-abi-compat={{.*}}'
//
// RUN: %clang_cc1 -fclang-abi-compat=3.0 %s -fsyntax-only
// RUN: %clang_cc1 -fclang-abi-compat=3.9 %s -fsyntax-only
// RUN: %clang_cc1 -fclang-abi-compat=4 %s -fsyntax-only
// RUN: %clang_cc1 -fclang-abi-compat=4.0 %s -fsyntax-only
// RUN: %clang_cc1 -fclang-abi-compat=latest %s -fsyntax-only
