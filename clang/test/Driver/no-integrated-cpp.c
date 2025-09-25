// RUN: %clang -O2 %s -E -o %t.i
//
// RUN: %clang -O2 %s -c -o a.o -no-integrated-cpp -### 2>&1 | FileCheck %s --check-prefixes=SRC
// SRC: "-E"
// SRC-SAME: "-o" "[[PREPROC:.*.i]]"
// SRC-SAME: "-x" "c" "{{.*}}no-integrated-cpp.c"
//
// SRC-NEXT: "-emit-obj"
// SRC-SAME: "-o" "a.o"
// SRC-SAME: "-x" "cpp-output" "[[PREPROC]]"
//
// RUN: %clang -O2 %s -c -o a.o -no-integrated-cpp -save-temps -### 2>&1 | FileCheck %s --check-prefixes=SRC-SAVE
// SRC-SAVE: "-E"
// SRC-SAVE-SAME: "-o" "[[PREPROC:.*.i]]"
// SRC-SAVE-SAME: "-x" "c" "{{.*}}no-integrated-cpp.c"
//
// SRC-SAVE-NEXT: "-emit-llvm-bc"
// SRC-SAVE-SAME: "-o" "[[BITCODE:.*.bc]]"
// SRC-SAVE-SAME: "-x" "cpp-output" "[[PREPROC]]"
//
// SRC-SAVE-NEXT: "-S"
// SRC-SAVE-SAME: "-o" "[[ASM:.*.s]]"
// SRC-SAVE-SAME: "-x" "ir" "[[BITCODE]]"
//
// SRC-SAVE-NEXT: {{"-cc1as"|"[^"]*/as"}}
// SRC-SAVE-SAME: "-o" "a.o" "[[ASM]]"
//
// RUN: %clang -O2 %t.i  -c -o a.o -no-integrated-cpp -### 2>&1 | FileCheck %s --check-prefixes=PRE
// PRE-NOT: "-E"
// PRE: "-emit-obj"
// PRE-SAME: "-o" "a.o"
// PRE-SAME: "-x" "cpp-output" "{{.*}}no-integrated-cpp.c.tmp.i"
//
// RUN: %clang -O2 %t.i  -c -o a.o -no-integrated-cpp -save-temps -### 2>&1 | FileCheck %s --check-prefixes=PRE-SAVE
// PRE-SAVE-NOT: "-E"
// PRE-SAVE: "-emit-llvm-bc"
// PRE-SAVE-SAME: "-o" "[[BITCODE:.*.bc]]"
// PRE-SAVE-SAME: "-x" "cpp-output" "{{.*}}no-integrated-cpp.c.tmp.i"
//
// PRE-SAVE-NEXT: "-S"
// PRE-SAVE-SAME: "-o" "[[ASM:.*.s]]"
// PRE-SAVE-SAME: "-x" "ir" "[[BITCODE]]"
//
// PRE-SAVE-NEXT: {{"-cc1as"|"[^"]*/as"}}
// PRE-SAVE-SAME: "-o" "a.o" "[[ASM]]"
//
// RUN: %clang -O2 %s -c -emit-llvm -o a.bc -no-integrated-cpp -### 2>&1 | FileCheck %s --check-prefixes=LLVM
// LLVM: "-E"
// LLVM-SAME: "-o" "[[PREPROC:.*.i]]"
// LLVM-SAME: "-x" "c" "{{.*}}no-integrated-cpp.c"
//
// LLVM-NEXT: "-emit-llvm-bc"
// LLVM-SAME: "-o" "a.bc"
// LLVM-SAME: "-x" "cpp-output" "[[PREPROC]]"
//
// RUN: %clang -O2 %s -c -emit-llvm -o a.bc -no-integrated-cpp -save-temps -### 2>&1 | FileCheck %s --check-prefixes=LLVM-SAVE
// LLVM-SAVE: "-E"
// LLVM-SAVE-SAME: "-o" "[[PREPROC:.*.i]]"
// LLVM-SAVE-SAME: "-x" "c" "{{.*}}no-integrated-cpp.c"
//
// LLVM-SAVE-NEXT: "-emit-llvm-bc"
// LLVM-SAVE-SAME: "-o" "[[BITCODE:.*.bc]]"
// LLVM-SAVE-SAME: "-x" "cpp-output" "[[PREPROC]]"
//
// LLVM-SAVE-NEXT: "-emit-llvm-bc"
// LLVM-SAVE-SAME: "-o" "a.bc"
// LLVM-SAVE-SAME: "-x" "ir" "[[BITCODE]]"
//
// RUN: %clang -O2 %t.i -c -emit-llvm -o a.bc -no-integrated-cpp -### 2>&1 | FileCheck %s --check-prefixes=PRE-LLVM
// PRE-LLVM-NOT: "-E"
// PRE-LLVM: "-emit-llvm-bc"
// PRE-LLVM-SAME: "-o" "a.bc"
// PRE-LLVM-SAME: "-x" "cpp-output" "{{.*}}no-integrated-cpp.c.tmp.i"
//
// RUN: %clang -O2 %t.i -c -emit-llvm -o a.bc -no-integrated-cpp -save-temps -### 2>&1 | FileCheck %s --check-prefixes=PRE-LLVM-SAVE
// PRE-LLVM-SAVE-NOT: "-E"
// PRE-LLVM-SAVE: "-emit-llvm-bc"
// PRE-LLVM-SAVE-SAME: "-o" "[[BITCODE:.*.bc]]"
// PRE-LLVM-SAVE-SAME: "-x" "cpp-output" "{{.*}}no-integrated-cpp.c.tmp.i"

// PRE-LLVM-SAVE-NEXT: "-emit-llvm-bc"
// PRE-LLVM-SAVE-SAME: "-o" "a.bc"
// PRE-LLVM-SAVE-SAME: "-x" "ir" "[[BITCODE]]"
