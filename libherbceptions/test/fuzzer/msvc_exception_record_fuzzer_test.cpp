//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: windows
// RUN: %cxx %herbceptions_flags -I%herbceptions_include -I%herbceptions_src/src -L%herbceptions_lib -lherbceptions %herbceptions_src/fuzz/msvc_exception_record_fuzzer.cpp -o %t && echo "HE" | %t
int main(){return 0;}
