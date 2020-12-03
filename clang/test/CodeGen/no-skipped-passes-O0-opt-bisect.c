// Test that no passes are skipped under NPM with -O0/opt-bisect
//
// RUN: %clang_cc1 -triple x86_64-linux-gnu -O0 -fexperimental-new-pass-manager %s -fdebug-pass-manager -emit-llvm -o /dev/null 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -O0 -fexperimental-new-pass-manager %s -fdebug-pass-manager -emit-llvm -o /dev/null -fcoroutines-ts 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -O0 -fexperimental-new-pass-manager %s -fdebug-pass-manager -emit-llvm -o /dev/null -fsanitize=address 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -O0 -fexperimental-new-pass-manager %s -fdebug-pass-manager -emit-llvm -o /dev/null -fsanitize=hwaddress 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -O0 -fexperimental-new-pass-manager %s -fdebug-pass-manager -emit-llvm -o /dev/null -fsanitize=memory 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -O0 -fexperimental-new-pass-manager %s -fdebug-pass-manager -emit-llvm -o /dev/null -fsanitize=thread 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -O0 -fexperimental-new-pass-manager %s -fdebug-pass-manager -emit-llvm -o /dev/null -fsanitize=local-bounds 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -O0 -fexperimental-new-pass-manager %s -fdebug-pass-manager -emit-llvm -o /dev/null -fsanitize=dataflow 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -O0 -fexperimental-new-pass-manager %s -fdebug-pass-manager -emit-llvm -o /dev/null -fsanitize-coverage-trace-pc-guard 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -O0 -fexperimental-new-pass-manager %s -fdebug-pass-manager -emit-llvm -o /dev/null -fmemory-profile 2>&1 | FileCheck %s

// RUN: %clang_cc1 -triple x86_64-linux-gnu -mllvm -opt-bisect-limit=0 -fexperimental-new-pass-manager %s -fdebug-pass-manager -emit-llvm -o /dev/null 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -mllvm -opt-bisect-limit=0 -fexperimental-new-pass-manager %s -fdebug-pass-manager -emit-llvm -o /dev/null -fcoroutines-ts 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -mllvm -opt-bisect-limit=0 -fexperimental-new-pass-manager %s -fdebug-pass-manager -emit-llvm -o /dev/null -fsanitize=address 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -mllvm -opt-bisect-limit=0 -fexperimental-new-pass-manager %s -fdebug-pass-manager -emit-llvm -o /dev/null -fsanitize=hwaddress 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -mllvm -opt-bisect-limit=0 -fexperimental-new-pass-manager %s -fdebug-pass-manager -emit-llvm -o /dev/null -fsanitize=memory 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -mllvm -opt-bisect-limit=0 -fexperimental-new-pass-manager %s -fdebug-pass-manager -emit-llvm -o /dev/null -fsanitize=thread 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -mllvm -opt-bisect-limit=0 -fexperimental-new-pass-manager %s -fdebug-pass-manager -emit-llvm -o /dev/null -fsanitize=local-bounds 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -mllvm -opt-bisect-limit=0 -fexperimental-new-pass-manager %s -fdebug-pass-manager -emit-llvm -o /dev/null -fsanitize=dataflow 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -mllvm -opt-bisect-limit=0 -fexperimental-new-pass-manager %s -fdebug-pass-manager -emit-llvm -o /dev/null -fsanitize-coverage-trace-pc-guard 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -mllvm -opt-bisect-limit=0 -fexperimental-new-pass-manager %s -fdebug-pass-manager -emit-llvm -o /dev/null -fmemory-profile 2>&1 | FileCheck %s

// CHECK-NOT: Skipping pass

int func(int a) { return a; }
