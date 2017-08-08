// REQUIRES: x86-registered-target

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm-bc -o /dev/null -fexperimental-new-pass-manager -fdebug-pass-manager -flto=full -O0 %s 2>&1 | FileCheck %s \
// RUN:   -check-prefix=CHECK-FULL-O0
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm-bc -o /dev/null -fexperimental-new-pass-manager -fdebug-pass-manager -flto=thin -O0 %s 2>&1 | FileCheck %s \
// RUN:   -check-prefix=CHECK-THIN-O0
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm-bc -o /dev/null -fexperimental-new-pass-manager -fdebug-pass-manager -flto=full -O1 %s 2>&1 | FileCheck %s \
// RUN:   -check-prefix=CHECK-FULL-OPTIMIZED
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm-bc -o /dev/null -fexperimental-new-pass-manager -fdebug-pass-manager -flto=thin -O1 %s 2>&1 | FileCheck %s \
// RUN:   -check-prefix=CHECK-THIN-OPTIMIZED
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm-bc -o /dev/null -fexperimental-new-pass-manager -fdebug-pass-manager -flto=full -O2 %s 2>&1 | FileCheck %s \
// RUN:   -check-prefix=CHECK-FULL-OPTIMIZED
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm-bc -o /dev/null -fexperimental-new-pass-manager -fdebug-pass-manager -flto=thin -O2 %s 2>&1 | FileCheck %s \
// RUN:   -check-prefix=CHECK-THIN-OPTIMIZED
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm-bc -o /dev/null -fexperimental-new-pass-manager -fdebug-pass-manager -flto=full -O3 %s 2>&1 | FileCheck %s \
// RUN:   -check-prefix=CHECK-FULL-OPTIMIZED
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm-bc -o /dev/null -fexperimental-new-pass-manager -fdebug-pass-manager -flto=thin -O3 %s 2>&1 | FileCheck %s \
// RUN:   -check-prefix=CHECK-THIN-OPTIMIZED
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm-bc -o /dev/null -fexperimental-new-pass-manager -fdebug-pass-manager -flto=full -Os %s 2>&1 | FileCheck %s \
// RUN:   -check-prefix=CHECK-FULL-OPTIMIZED
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm-bc -o /dev/null -fexperimental-new-pass-manager -fdebug-pass-manager -flto=thin -Os %s 2>&1 | FileCheck %s \
// RUN:   -check-prefix=CHECK-THIN-OPTIMIZED
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm-bc -o /dev/null -fexperimental-new-pass-manager -fdebug-pass-manager -flto=full -Oz %s 2>&1 | FileCheck %s \
// RUN:   -check-prefix=CHECK-FULL-OPTIMIZED
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm-bc -o /dev/null -fexperimental-new-pass-manager -fdebug-pass-manager -flto=thin -Oz %s 2>&1 | FileCheck %s \
// RUN:   -check-prefix=CHECK-THIN-OPTIMIZED

// CHECK-FULL-O0: Starting llvm::Module pass manager run.
// CHECK-FULL-O0: Running pass: AlwaysInlinerPass
// CHECK-FULL-O0-NEXT: Running pass: BitcodeWriterPass
// CHECK-FULL-O0: Finished llvm::Module pass manager run.

// CHECK-THIN-O0: Starting llvm::Module pass manager run.
// CHECK-THIN-O0: Running pass: AlwaysInlinerPass
// CHECK-THIN-O0-NEXT: Running pass: NameAnonGlobalPass
// CHECK-THIN-O0-NEXT: Running pass: ThinLTOBitcodeWriterPass
// CHECK-THIN-O0: Finished llvm::Module pass manager run.

// TODO: The LTO pre-link pipeline currently invokes
//       buildPerModuleDefaultPipeline(), which contains LoopVectorizePass.
//       This may change as the pipeline gets implemented.
// CHECK-FULL-OPTIMIZED: Starting llvm::Function pass manager run.
// CHECK-FULL-OPTIMIZED: Running pass: LoopVectorizePass
// CHECK-FULL-OPTIMIZED: Running pass: BitcodeWriterPass

// The ThinLTO pre-link pipeline shouldn't contain passes like
// LoopVectorizePass.
// CHECK-THIN-OPTIMIZED: Starting llvm::Function pass manager run.
// CHECK-THIN-OPTIMIZED-NOT: Running pass: LoopVectorizePass
// CHECK-THIN-OPTIMIZED: Running pass: NameAnonGlobalPass
// CHECK-THIN-OPTIMIZED: Running pass: ThinLTOBitcodeWriterPass

void Foo() {}
