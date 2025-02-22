#include "../Inputs/cuda.h"

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:            -x cuda -emit-cir -target-sdk-version=12.3 \
// RUN:            -fcuda-include-gpubinary fatbin.o\
// RUN:            %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR-HOST --input-file=%t.cir %s

// CIR-HOST: module @"{{.*}}" attributes{{.*}}cir.cu.binary_handle = #cir.cu.binary_handle<fatbin.o>{{.*}}
