// RUN: rm -rf %t.dir && mkdir -p %t.dir

/// Check use pgo profile.
// RUN: llvm-profdata merge -o %t.profdata %S/Inputs/pgo.profraw
// RUN: %clang -cc1depscan -fdepscan=inline -o %t.rsp -cc1-args -cc1 -triple x86_64-apple-macosx12.0.0 -emit-obj -O3 -Rcompile-job-cache \
// RUN:   -x c %s -o %t.o -fcas-path %t.dir/cas -fprofile-instrument-use=clang -fprofile-instrument-use-path=%t.profdata

/// Remove profile data to make sure the cc1 command is not reading from file system.
// RUN: rm %t.profdata
// RUN: %clang @%t.rsp 2>&1 | FileCheck %s --check-prefix=CACHE-MISS
// RUN: %clang @%t.rsp 2>&1 | FileCheck %s --check-prefix=CACHE-HIT

/// Check include tree.
// RUN: llvm-profdata merge -o %t.profdata %S/Inputs/pgo.profraw
// RUN: %clang -cc1depscan -fdepscan=inline -fdepscan-include-tree -o %t1.rsp -cc1-args -cc1 -triple x86_64-apple-macosx12.0.0 -emit-obj -O3 -Rcompile-job-cache \
// RUN:   -x c %s -o %t.o -fcas-path %t.dir/cas -fprofile-instrument-use=clang -fprofile-instrument-use-path=%t.profdata
// RUN: rm %t.profdata
// RUN: %clang @%t1.rsp 2>&1 | FileCheck %s --check-prefix=CACHE-MISS
// RUN: %clang @%t1.rsp 2>&1 | FileCheck %s --check-prefix=CACHE-HIT

/// Check change profile data will cause cache miss.
// RUN: llvm-profdata merge -o %t.profdata %S/Inputs/pgo2.profraw
// RUN: %clang -cc1depscan -fdepscan=inline -o %t2.rsp -cc1-args -cc1 -triple x86_64-apple-macosx12.0.0 -emit-obj -O3 -Rcompile-job-cache \
// RUN:   -x c %s -o %t.o -fcas-path %t.dir/cas -fprofile-instrument-use=clang -fprofile-instrument-use-path=%t.profdata
// RUN: not diff %t.rsp %t2.rsp
// RUN: %clang @%t2.rsp 2>&1 | FileCheck %s --check-prefix=CACHE-MISS

// RUN: %clang -cc1depscan -fdepscan=inline -fdepscan-include-tree -o %t3.rsp -cc1-args -cc1 -triple x86_64-apple-macosx12.0.0 -emit-obj -O3 -Rcompile-job-cache \
// RUN:   -x c %s -o %t.o -fcas-path %t.dir/cas -fprofile-instrument-use=clang -fprofile-instrument-use-path=%t.profdata
// RUN: not diff %t1.rsp %t3.rsp
// RUN: %clang @%t3.rsp 2>&1 | FileCheck %s --check-prefix=CACHE-MISS

// CACHE-MISS: remark: compile job cache miss
// CACHE-HIT: remark: compile job cache hit

/// Check remapping for profile.
// RUN: mkdir -p %t.dir/a && mkdir -p %t.dir/b
// RUN: cp %t.profdata %t.dir/a/a.profdata
// RUN: cp %t.profdata %t.dir/b/a.profdata
// RUN: %clang -cc1depscan -fdepscan=inline -o %t4.rsp -cc1-args -cc1 -triple x86_64-apple-macosx12.0.0 -emit-obj -O3 -Rcompile-job-cache -fdepscan-prefix-map %t.dir/a /^testdir  \
// RUN:   -x c %s -o %t.o -fcas-path %t.dir/cas -fprofile-instrument-use=clang -fprofile-instrument-use-path=%t.dir/a/a.profdata
// RUN: %clang -cc1depscan -fdepscan=inline -o %t5.rsp -cc1-args -cc1 -triple x86_64-apple-macosx12.0.0 -emit-obj -O3 -Rcompile-job-cache -fdepscan-prefix-map %t.dir/b /^testdir \
// RUN:   -x c %s -o %t.o -fcas-path %t.dir/cas -fprofile-instrument-use=clang -fprofile-instrument-use-path=%t.dir/b/a.profdata
// RUN: cat %t4.rsp | FileCheck %s --check-prefix=REMAP
// RUN: %clang @%t4.rsp 2>&1 | FileCheck %s --check-prefix=CACHE-MISS
// RUN: %clang @%t5.rsp 2>&1 | FileCheck %s --check-prefix=CACHE-HIT

// RUN: cat %t4.rsp | sed \
// RUN:   -e "s/^.*\"-fcas-fs\" \"//" \
// RUN:   -e "s/\" .*$//" > %t.dir/cache-key1
// RUN: cat %t5.rsp | sed \
// RUN:   -e "s/^.*\"-fcas-fs\" \"//" \
// RUN:   -e "s/\" .*$//" > %t.dir/cache-key2
// RUN: grep llvmcas %t.dir/cache-key1
// RUN: diff -u %t.dir/cache-key1 %t.dir/cache-key2

// RUN: %clang -cc1depscan -fdepscan=inline -fdepscan-include-tree -o %t4.inc.rsp -cc1-args -cc1 -triple x86_64-apple-macosx12.0.0 -emit-obj -O3 -Rcompile-job-cache -fdepscan-prefix-map %t.dir/a /^testdir \
// RUN:   -x c %s -o %t.o -fcas-path %t.dir/cas -fprofile-instrument-use=clang -fprofile-instrument-use-path=%t.dir/a/a.profdata
// RUN: %clang -cc1depscan -fdepscan=inline -fdepscan-include-tree -o %t5.inc.rsp -cc1-args -cc1 -triple x86_64-apple-macosx12.0.0 -emit-obj -O3 -Rcompile-job-cache -fdepscan-prefix-map %t.dir/b /^testdir \
// RUN:   -x c %s -o %t.o -fcas-path %t.dir/cas -fprofile-instrument-use=clang -fprofile-instrument-use-path=%t.dir/b/a.profdata
// RUN: cat %t4.inc.rsp | FileCheck %s --check-prefix=REMAP
// RUN: %clang @%t4.inc.rsp 2>&1 | FileCheck %s --check-prefix=CACHE-MISS
// RUN: %clang @%t5.inc.rsp 2>&1 | FileCheck %s --check-prefix=CACHE-HIT

// RUN: cat %t4.inc.rsp | sed \
// RUN:   -e "s/^.*\"-fcas-include-tree\" \"//" \
// RUN:   -e "s/\" .*$//" > %t.dir/inc-cache-key1
// RUN: cat %t5.inc.rsp | sed \
// RUN:   -e "s/^.*\"-fcas-include-tree\" \"//" \
// RUN:   -e "s/\" .*$//" > %t.dir/inc-cache-key2
// RUN: grep llvmcas %t.dir/inc-cache-key1
// RUN: diff -u %t.dir/inc-cache-key1 %t.dir/inc-cache-key2

// REMAP: -fprofile-instrument-use-path=/^testdir/a.profdata
