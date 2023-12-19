// Value profiling is currently not supported in lightweight mode.
// RUN: %clang_pgogen -o %t.normal -mllvm --disable-vp=true %S/../Inputs/instrprof-debug-info-correlate-main.cpp %S/../Inputs/instrprof-debug-info-correlate-foo.cpp
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t.normal
// RUN: llvm-profdata merge -o %t.normal.profdata %t.profraw

// RUN: %clang_pgogen -o %t.d4 -g -gdwarf-4 -mllvm --debug-info-correlate -mllvm --disable-vp=true %S/../Inputs/instrprof-debug-info-correlate-main.cpp %S/../Inputs/instrprof-debug-info-correlate-foo.cpp
// RUN: env LLVM_PROFILE_FILE=%t.d4.proflite %run %t.d4
// RUN: llvm-profdata merge -o %t.d4.profdata --debug-info=%t.d4 %t.d4.proflite

// RUN: diff <(llvm-profdata show --all-functions --counts %t.normal.profdata) <(llvm-profdata show --all-functions --counts %t.d4.profdata)

// RUN: %clang_pgogen -o %t -g -mllvm --debug-info-correlate -mllvm --disable-vp=true %S/../Inputs/instrprof-debug-info-correlate-main.cpp %S/../Inputs/instrprof-debug-info-correlate-foo.cpp
// RUN: env LLVM_PROFILE_FILE=%t.proflite %run %t
// RUN: llvm-profdata merge -o %t.profdata --debug-info=%t %t.proflite

// RUN: diff <(llvm-profdata show --all-functions --counts %t.normal.profdata) <(llvm-profdata show --all-functions --counts %t.profdata)

// RUN: %clang_pgogen -o %t.cov -g -mllvm --debug-info-correlate -mllvm -pgo-function-entry-coverage -mllvm --disable-vp=true %S/../Inputs/instrprof-debug-info-correlate-main.cpp %S/../Inputs/instrprof-debug-info-correlate-foo.cpp
// RUN: env LLVM_PROFILE_FILE=%t.cov.proflite %run %t.cov
// RUN: llvm-profdata merge -o %t.cov.profdata --debug-info=%t.cov %t.cov.proflite

// RUN: %clang_pgogen -o %t.cov.normal -mllvm --pgo-function-entry-coverage -mllvm --disable-vp=true %S/../Inputs/instrprof-debug-info-correlate-main.cpp %S/../Inputs/instrprof-debug-info-correlate-foo.cpp
// RUN: env LLVM_PROFILE_FILE=%t.cov.profraw %run %t.cov.normal
// RUN: llvm-profdata merge -o %t.cov.normal.profdata %t.cov.profraw

// RUN: diff <(llvm-profdata show --all-functions --counts %t.cov.normal.profdata) <(llvm-profdata show --all-functions --counts %t.cov.profdata)

// Test debug info correlate with build id.

// Both binaries are built with build id.
// RUN: %clang_pgogen -o %t -g -mllvm --debug-info-correlate -mllvm --disable-vp=true %S/../Inputs/instrprof-debug-info-correlate-foo.cpp -c -fpic -shared -Wl,--build-id -o %t-libfoo.so
// RUN: %clang_pgogen -o %t -g -mllvm --debug-info-correlate -mllvm --disable-vp=true %S/../Inputs/instrprof-debug-info-correlate-main.cpp %t-libfoo.so -Wl,--build-id -o %t
// RUN: env LLVM_PROFILE_FILE=%t.proflite %run %t
// RUN: llvm-profdata merge -o %t.profdata --debug-info=%t,%t-libfoo.so %t.proflite
// RUN: diff <(llvm-profdata show --all-functions --counts %t.normal.profdata) <(llvm-profdata show --all-functions --counts %t.profdata)

// One binary is built without build id.
// RUN: %clang_pgogen -o %t -g -mllvm --debug-info-correlate -mllvm --disable-vp=true %S/../Inputs/instrprof-debug-info-correlate-main.cpp %t-libfoo.so -o %t
// RUN: env LLVM_PROFILE_FILE=%t.proflite %run %t
// RUN: llvm-profdata merge -o %t.profdata --debug-info=%t,%t-libfoo.so %t.proflite
// RUN: diff <(llvm-profdata show --all-functions --counts %t.normal.profdata) <(llvm-profdata show --all-functions --counts %t.profdata)

// Warning about multiple correlate files have the same build id.
// RUN: llvm-profdata merge -o %t.profdata --debug-info=%t,%t,%t-libfoo.o %t.proflite 2>&1 >/dev/null | FileCheck %s --check-prefix=WARN
// WARN: Duplicate build id ({{.*}}) found for {{.*}} and {{.*}}

// Test debug info correlate with online merging.

// RUN: env LLVM_PROFILE_FILE=%t-1.profraw %run %t.normal
// RUN: env LLVM_PROFILE_FILE=%t-2.profraw %run %t.normal
// RUN: llvm-profdata merge -o %t.normal.profdata %t-1.profraw %t-2.profraw

// RUN: rm -rf %t.profdir && mkdir %t.profdir
// RUN: env LLVM_PROFILE_FILE=%t.profdir/%m.proflite %run %t
// RUN: env LLVM_PROFILE_FILE=%t.profdir/%m.proflite %run %t
// RUN: llvm-profdata merge -o %t.profdata --debug-info=%t %t.profdir/

// RUN: diff <(llvm-profdata show --all-functions --counts %t.normal.profdata) <(llvm-profdata show --all-functions --counts %t.profdata)

// RUN: rm -rf %t.profdir && mkdir %t.profdir
// RUN: env LLVM_PROFILE_FILE=%t.profdir/%m.cov.proflite %run %t.cov
// RUN: env LLVM_PROFILE_FILE=%t.profdir/%m.cov.proflite %run %t.cov
// RUN: llvm-profdata merge -o %t.cov.profdata --debug-info=%t.cov %t.profdir/

// RUN: diff <(llvm-profdata show --all-functions --counts %t.cov.normal.profdata) <(llvm-profdata show --all-functions --counts %t.cov.profdata)
