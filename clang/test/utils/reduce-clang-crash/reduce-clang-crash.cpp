// REQUIRES: x86-registered-target
// REQUIRES: reproducer-reduction
//
// RUN: rm -rf %t && mkdir -p %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: env LLVM_DISABLE_SYMBOLIZATION=0 %reduce-clang-crash --clang %clang --opt opt --llc llc --llvm-reduce llvm-reduce --llc-arg=-fast-isel-abort=3 --auto crash.sh reduce-clang-crash.cpp | FileCheck --check-prefix=BACKEND %s
// RUN: FileCheck --check-prefix=BACKEND-REDUCED %s < %t/reduced.ll
//
// BACKEND: Found Middle/Backend failure
// BACKEND-NEXT: Checking llc for failure
// BACKEND-NEXT: Found BackEnd Crash
// BACKEND-NEXT: Writing interestingness test...
// BACKEND-EMPTY:
// BACKEND-NEXT: Creating the interestingness test...
// BACKEND-NEXT: Starting llvm-reduce with llc test case
// BACKEND-EMPTY:
// BACKEND-NEXT: Running llvm-reduce tool...
// BACKEND-NEXT: Done Reducing IR file.
//
// BACKEND-REDUCED: define void @_Z3foov
//
// RUN: env LLVM_DISABLE_SYMBOLIZATION=0 %reduce-clang-crash --clang %clang --creduce %creduce --auto crash.sh reduce-clang-crash.cpp --n 1 | FileCheck --check-prefix=FALLBACK %s
// RUN: FileCheck --check-prefix=FALLBACK-REDUCED %s < %t/reduced.ll
//
// FALLBACK: Found Middle/Backend failure
// FALLBACK-NEXT: Checking llc for failure
// FALLBACK-NEXT: Checking opt for failure
// FALLBACK-NEXT: Check clang on IR file
// FALLBACK-NEXT: Found MiddleEnd Crash
// FALLBACK-NEXT: Writing interestingness test...
// FALLBACK-EMPTY:
// FALLBACK-NEXT: Creating the interestingness test...
// FALLBACK-NEXT: Starting llvm-reduce with clang test case
// FALLBACK-EMPTY:
// FALLBACK-NEXT: Running llvm-reduce tool...
// FALLBACK-NEXT: Done Reducing IR file.
//
// FALLBACK-REDUCED: define void @_Z3foov
//
// RUN: env LLVM_DISABLE_SYMBOLIZATION=0 %reduce-clang-crash --clang %clang --creduce %creduce --auto crash_pragma.sh crash_pragma.cpp --n 1 | FileCheck --check-prefix=PRAGMA %s
// RUN: FileCheck --check-prefix=PRAGMA-REDUCED %s < %t/crash_pragma.reduced.cpp
//
// PRAGMA: Found Frontend Crash
// PRAGMA: Starting reduction with creduce/cvise
//
// PRAGMA-REDUCED: #pragma clang __debug assert

//--- reduce-clang-crash.cpp
void foo() {
  volatile __int128 a = 1;
  volatile __int128 b = 2;
  volatile __int128 c = a * b;
}

//--- crash_pragma.cpp
#pragma clang __debug assert
void foo() {}

//--- crash.sh
clang -cc1 -triple x86_64-unknown-linux-gnu -O0 -emit-obj -mllvm -fast-isel-abort=3 -x c++ reduce-clang-crash.cpp

//--- crash_pragma.sh
clang -cc1 -triple x86_64-unknown-linux-gnu -O0 -x c++ crash_pragma.cpp
