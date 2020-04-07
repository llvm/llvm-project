// RUN: %clang_cc1 -S -emit-llvm -fveclib=SVML -o - %s | FileCheck --check-prefixes=SVML %s
// RUN: %clang_cc1 -S -emit-llvm -fveclib=Accelerate -o - %s | FileCheck --check-prefixes=ACCELERATE %s
// RUN: %clang_cc1 -S -emit-llvm -fveclib=MASSV -o - %s | FileCheck --check-prefixes=MASSV %s
// RUN: %clang_cc1 -S -emit-llvm -fveclib=none -o - %s | FileCheck --check-prefixes=NOLIB %s
// RUN: %clang_cc1 -S -emit-llvm -o - %s | FileCheck --check-prefixes=NOLIB %s

int main() {
  return 0;
}

// SVML: "veclib"="SVML"
// ACCELERATE: "veclib"="Accelerate"
// MASSV: "veclib"="MASSV"
// NOLIB-NOT: "veclib"