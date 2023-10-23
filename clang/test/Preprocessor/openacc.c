// RUN: %clang_cc1 -E -fopenacc %s | FileCheck %s --check-prefix=DEFAULT

// DEFAULT: OpenACC:1:
OpenACC:_OPENACC:
