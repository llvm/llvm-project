// RUN: rm -rf %t.dir && mkdir -p %t.dir

// Check that use of profile data for PCH is ignored

// RUN: llvm-profdata merge -o %t.profdata %S/Inputs/pgo.profraw
// RUN: %clang -cc1depscan -fdepscan=inline -fdepscan-include-tree -o %t-pch.rsp -cc1-args -cc1 -triple x86_64-apple-macosx12.0.0 -emit-pch -O3 -Rcompile-job-cache \
// RUN:   -x c-header %s -o %t.h.pch -fcas-path %t.dir/cas -fprofile-instrument-use-path=%t.profdata
// RUN: %clang @%t-pch.rsp 2>&1 | FileCheck %s --check-prefix=CACHE-MISS
// RUN: FileCheck %s -check-prefix=PCHPROF -input-file %t-pch.rsp
// PCHPROF-NOT: -fprofile-instrument-use-path

// Update profdata file contents
// RUN: llvm-profdata merge -o %t.profdata %S/Inputs/pgo2.profraw

// Use the modified profdata file for the main file along with the PCH.
// RUN: %clang -cc1depscan -fdepscan=inline -fdepscan-include-tree -o %t.rsp -cc1-args -cc1 -triple x86_64-apple-macosx12.0.0 -emit-obj -O3 -Rcompile-job-cache \
// RUN:   -x c %s -o %t.o -fcas-path %t.dir/cas -fprofile-instrument-use-path=%t.profdata -include-pch %t.h.pch
// RUN: %clang @%t.rsp 2>&1 | FileCheck %s --check-prefix=CACHE-MISS
// RUN: FileCheck %s -check-prefix=TUPROF -input-file %t.rsp
// TUPROF: -fprofile-instrument-use-path

// Check that the modified profdata is ignored when re-scanning for the PCH.
// RUN: %clang -cc1depscan -fdepscan=inline -fdepscan-include-tree -o %t-pch2.rsp -cc1-args -cc1 -triple x86_64-apple-macosx12.0.0 -emit-pch -O3 -Rcompile-job-cache \
// RUN:   -x c-header %s -o %t.h.pch -fcas-path %t.dir/cas -fprofile-instrument-use-path=%t.profdata
// RUN: diff -u %t-pch.rsp %t-pch2.rsp

// CACHE-MISS: remark: compile job cache miss
// CACHE-HIT: remark: compile job cache hit
