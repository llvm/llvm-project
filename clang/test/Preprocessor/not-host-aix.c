// UNSUPPORTED: system-aix
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc-ibm-aix7.1.0.0 -fno-signed-char < /dev/null | FileCheck -match-full-lines -check-prefix PPC-AIX %s
// PPC-AIX-NOT:#define __HOS_AIX__ 1
