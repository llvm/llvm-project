// Ensure that Clang sets some very basic target defines based on -aux-triple.

// RUN: %clang_cc1 -E -dM -ffreestanding < /dev/null \
// RUN:     -triple nvptx64-none-none \
// RUN:   | FileCheck -match-full-lines -check-prefixes NVPTX64,NONE %s
// RUN: %clang_cc1 -x c++ -E -dM -ffreestanding < /dev/null \
// RUN:     -triple nvptx64-none-none \
// RUN:   | FileCheck -match-full-lines -check-prefixes NVPTX64,NONE %s
// RUN: %clang_cc1 -x cuda -E -dM -ffreestanding < /dev/null \
// RUN:     -triple nvptx64-none-none \
// RUN:   | FileCheck -match-full-lines -check-prefixes NVPTX64,NONE %s

// CUDA:
// RUN: %clang_cc1 -x cuda -E -dM -ffreestanding < /dev/null \
// RUN:     -triple nvptx64-none-none -aux-triple powerpc64le-unknown-linux-gnu \
// RUN:   | FileCheck -match-full-lines %s \
// RUN:     -check-prefixes NVPTX64,PPC64,LINUX,LINUX-CPP
// RUN: %clang_cc1 -x cuda -E -dM -ffreestanding < /dev/null \
// RUN:     -triple nvptx64-none-none -aux-triple x86_64-unknown-linux-gnu \
// RUN:   | FileCheck -match-full-lines %s \
// RUN:     -check-prefixes NVPTX64,X86_64,LINUX,LINUX-CPP

// OpenMP:
// RUN: %clang_cc1 -E -dM -ffreestanding < /dev/null \
// RUN:     -fopenmp -fopenmp-is-device -triple nvptx64-none-none \
// RUN:     -aux-triple powerpc64le-unknown-linux-gnu \
// RUN:   | FileCheck -match-full-lines -check-prefixes NVPTX64,PPC64,LINUX %s
// RUN: %clang_cc1 -E -dM -ffreestanding < /dev/null \
// RUN:     -fopenmp -fopenmp-is-device -triple nvptx64-none-none \
// RUN:     -aux-triple x86_64-unknown-linux-gnu \
// RUN:   | FileCheck -match-full-lines -check-prefixes NVPTX64,X86_64,LINUX %s
// RUN: %clang_cc1 -x c++ -E -dM -ffreestanding < /dev/null \
// RUN:     -fopenmp -fopenmp-is-device -triple nvptx64-none-none \
// RUN:     -aux-triple powerpc64le-unknown-linux-gnu \
// RUN:   | FileCheck -match-full-lines %s \
// RUN:     -check-prefixes NVPTX64,PPC64,LINUX,LINUX-CPP
// RUN: %clang_cc1 -x c++ -E -dM -ffreestanding < /dev/null \
// RUN:     -fopenmp -fopenmp-is-device -triple nvptx64-none-none \
// RUN:     -aux-triple x86_64-unknown-linux-gnu \
// RUN:   | FileCheck -match-full-lines %s \
// RUN:     -check-prefixes NVPTX64,X86_64,LINUX,LINUX-CPP

// NONE-NOT:#define _GNU_SOURCE
// LINUX-CPP:#define _GNU_SOURCE 1

// NVPTX64:#define _LP64 1

// NONE-NOT:#define __ELF__
// LINUX:#define __ELF__ 1

// NVPTX64:#define __LP64__ 1
// NVPTX64:#define __NVPTX__ 1
// NVPTX64:#define __PTX__ 1

// NONE-NOT:#define __linux__
// LINUX:#define __linux__ 1

// NONE-NOT:#define __powerpc64__
// PPC64:#define __powerpc64__ 1

// NONE-NOT:#define __x86_64__
// X86_64:#define __x86_64__ 1
