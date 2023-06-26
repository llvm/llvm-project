// Check the ABI version support.

// RUN: %clang -fptrauth-abi-version=5                              -arch arm64e -c %s -### 2>&1 | FileCheck %s --check-prefix ABIVERSION --check-prefix NOKERNELABIVERSION
// RUN: %clang -fptrauth-abi-version=5 -mkernel                     -arch arm64e -c %s -### 2>&1 | FileCheck %s --check-prefix ABIVERSION --check-prefix KERNELABIVERSION
// RUN: %clang -fptrauth-abi-version=5 -fapple-kext                 -arch arm64e -c %s -### 2>&1 | FileCheck %s --check-prefix ABIVERSION --check-prefix KERNELABIVERSION
// RUN: %clang -fptrauth-abi-version=5 -fptrauth-kernel-abi-version -arch arm64e -c %s -### 2>&1 | FileCheck %s --check-prefix ABIVERSION --check-prefix KERNELABIVERSION

// RUN: %clang -fno-ptrauth-abi-version                                        -arch arm64e -c %s -### 2>&1 | FileCheck %s --check-prefix NOABIVERSION --check-prefix NOKERNELABIVERSION
// RUN: %clang -fptrauth-abi-version=5         -fno-ptrauth-abi-version        -arch arm64e -c %s -### 2>&1 | FileCheck %s --check-prefix NOABIVERSION --check-prefix NOKERNELABIVERSION
// RUN: %clang -fno-ptrauth-abi-version        -fptrauth-abi-version=5         -arch arm64e -c %s -### 2>&1 | FileCheck %s --check-prefix ABIVERSION --check-prefix NOKERNELABIVERSION

// ABIVERSION: "-fptrauth-abi-version=5"
// ABIVERSION-DEFAULT: "-fptrauth-abi-version=0"
// NOABIVERSION-NOT: fptrauth-abi-version
// KERNELABIVERSION: "-fptrauth-kernel-abi-version"
// NOKERNELABIVERSION-NOT: fptrauth-kernel-abi-version
