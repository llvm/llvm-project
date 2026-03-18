// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: env LLVM_CACHE_CAS_PATH=%t/cas %clang-cache %clang_cl /Yc%t/cmake_pch.hh /Fp%t/cmake_pch.pch /FI%t/cmake_pch.hh /Fo%t/cmake_pch.cc.obj /Fd%t -Xclang -Rcompile-job-cache -c -- %t/cmake_pch.cc 2>&1 | FileCheck %s -check-prefix CHECK-CACHE-MISS
// RUN: env LLVM_CACHE_CAS_PATH=%t/cas %clang-cache %clang_cl /Yc%t/cmake_pch.hh /Fp%t/cmake_pch.pch /FI%t/cmake_pch.hh /Fo%t/cmake_pch.cc.obj /Fd%t -Xclang -Rcompile-job-cache -c -- %t/cmake_pch.cc 2>&1 | FileCheck %s -check-prefix CHECK-CACHE-HIT
// RUN: env LLVM_CACHE_CAS_PATH=%t/cas %clang-cache %clang_cl /Yu%t/cmake_pch.hh /Fp%t/cmake_pch.pch /FI%t/cmake_pch.hh /Fo%t/test.cc.obj /Fd%t -Xclang -Rcompile-job-cache -c -- %t/test.cc 2>&1 | FileCheck %s -check-prefix CHECK-CACHE-MISS

// CHECK-CACHE-HIT: remark: compile job cache hit
// CHECK-CACHE-MISS: remark: compile job cache miss

//--- test.cc
#include "header.h"

//--- header.h
#pragma once

//--- cmake_pch.cc
extern int _;

//--- cmake_pch.hh
#pragma clang system_header
#include "header.h"
