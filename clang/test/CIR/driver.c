// RUN: %clang -target x86_64-unknown-linux-gnu -fclangir -fclangir-direct-lowering -S -Xclang -emit-cir %s -o %t1.cir
// RUN: FileCheck --input-file=%t1.cir %s -check-prefix=CIR
// RUN: %clang -target x86_64-unknown-linux-gnu -fclangir -fno-clangir-direct-lowering -S -Xclang -emit-cir %s -o %t2.cir
// RUN: FileCheck --input-file=%t2.cir %s -check-prefix=CIR
// RUN: %clang -target x86_64-unknown-linux-gnu -fclangir -fclangir-direct-lowering -S -emit-llvm %s -o %t1.ll
// RUN: FileCheck --input-file=%t1.ll %s -check-prefix=LLVM
// RUN: %clang -target x86_64-unknown-linux-gnu -fclangir -fno-clangir-direct-lowering -S -emit-llvm %s -o %t2.ll
// RUN: FileCheck --input-file=%t2.ll %s -check-prefix=LLVM
// RUN: %clang -target x86_64-unknown-linux-gnu -fclangir -c %s -o %t.o
// RUN: llvm-objdump -d %t.o | FileCheck %s -check-prefix=OBJ
// RUN: %clang -target x86_64-unknown-linux-gnu -fclangir -clangir-disable-passes -S -Xclang -emit-cir %s -o %t.cir
// RUN: %clang -target x86_64-unknown-linux-gnu -fclangir -clangir-disable-verifier -S -Xclang -emit-cir %s -o %t.cir
// RUN: %clang -target arm64-apple-macosx12.0.0 -fclangir -S -Xclang -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR

void foo(void) {}

//      CIR: module {{.*}} {
// CIR-NEXT:   cir.func @foo()
// CIR-NEXT:     cir.return
// CIR-NEXT:   }
// CIR-NEXT: }

//      LLVM: define void @foo()
// LLVM-NEXT:   ret void
// LLVM-NEXT: }

// OBJ: 0: c3 retq
