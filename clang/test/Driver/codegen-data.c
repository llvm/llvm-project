// Verify only one of codegen-data flag is passed.
// RUN: not %clang -### -S --target=aarch64-linux-gnu -fcodegen-data-generate -fcodegen-data-use %s 2>&1 | FileCheck %s --check-prefix=CONFLICT
// RUN: not %clang -### -S --target=arm64-apple-darwin  -fcodegen-data-generate -fcodegen-data-use %s 2>&1 | FileCheck %s --check-prefix=CONFLICT
// CONFLICT: error: invalid argument '-fcodegen-data-generate' not allowed with '-fcodegen-data-use'

// Verify the codegen-data-generate (boolean) flag is passed to LLVM
// RUN: %clang -### -S --target=aarch64-linux-gnu -fcodegen-data-generate %s  2>&1| FileCheck %s --check-prefix=GENERATE
// RUN: %clang -### -S --target=arm64-apple-darwin -fcodegen-data-generate %s 2>&1| FileCheck %s --check-prefix=GENERATE
// GENERATE: "-mllvm" "-codegen-data-generate"

// Verify the codegen-data-use-path flag (with a default value) is passed to LLVM.
// RUN: %clang -### -S --target=aarch64-linux-gnu -fcodegen-data-use %s 2>&1| FileCheck %s --check-prefix=USE
// RUN: %clang -### -S --target=arm64-apple-darwin -fcodegen-data-use %s 2>&1| FileCheck %s --check-prefix=USE
// RUN: %clang -### -S --target=aarch64-linux-gnu -fcodegen-data-use=file %s 2>&1 | FileCheck %s --check-prefix=USE-FILE
// RUN: %clang -### -S --target=arm64-apple-darwin -fcodegen-data-use=file %s 2>&1 | FileCheck %s --check-prefix=USE-FILE
// USE: "-mllvm" "-codegen-data-use-path=default.cgdata"
// USE-FILE: "-mllvm" "-codegen-data-use-path=file"

// Verify the codegen-data-generate (boolean) flag with a LTO.
// RUN: %clang -### -flto --target=aarch64-linux-gnu -fcodegen-data-generate %s 2>&1 | FileCheck %s --check-prefix=GENERATE-LTO
// GENERATE-LTO: {{ld(.exe)?"}}
// GENERATE-LTO-SAME: "-plugin-opt=-codegen-data-generate"
// RUN: %clang -### -flto --target=arm64-apple-darwin -fcodegen-data-generate %s 2>&1 | FileCheck %s --check-prefix=GENERATE-LTO-DARWIN
// GENERATE-LTO-DARWIN: {{ld(.exe)?"}}
// GENERATE-LTO-DARWIN-SAME: "-mllvm" "-codegen-data-generate"

// Verify the codegen-data-use-path flag with a LTO is passed to LLVM.
// RUN: %clang -### -flto=thin --target=aarch64-linux-gnu -fcodegen-data-use %s 2>&1 | FileCheck %s --check-prefix=USE-LTO
// USE-LTO: {{ld(.exe)?"}}
// USE-LTO-SAME: "-plugin-opt=-codegen-data-use-path=default.cgdata"
// RUN: %clang -### -flto=thin --target=arm64-apple-darwin -fcodegen-data-use %s 2>&1 | FileCheck %s --check-prefix=USE-LTO-DARWIN
// USE-LTO-DARWIN: {{ld(.exe)?"}}
// USE-LTO-DARWIN-SAME: "-mllvm" "-codegen-data-use-path=default.cgdata"

// For now, LLD MachO supports for generating the codegen data at link time.
// RUN: %clang -### -fuse-ld=lld -B%S/Inputs/lld --target=arm64-apple-darwin -fcodegen-data-generate %s 2>&1 | FileCheck %s --check-prefix=GENERATE-LLD-DARWIN
// GENERATE-LLD-DARWIN: {{ld(.exe)?"}}
// GENERATE-LLD-DARWIN-SAME: "--codegen-data-generate-path=default.cgdata"
