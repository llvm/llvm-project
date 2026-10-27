// RUN: %clang_cc1 -triple x86_64-linux-gnu -fclangir -emit-cir %s \
// RUN:   -target-sdk-version=12.3 -o - | FileCheck %s --check-prefix=SDK
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fclangir -emit-cir %s -o - \
// RUN:   | FileCheck %s --check-prefix=NOSDK

// CIRGen records the platform SDK version on the ModuleOp as cir.sdk_version so
// post-CIRGen passes can gate on it without a live ASTContext; LLVM IR records
// the same fact as the "SDK Version" module flag. The version is a property of
// the target, not of a language mode, so it is emitted for every language - this
// test is plain C to pin that down. Nothing is emitted without
// -target-sdk-version, which leaves version-gated features disabled.

int x;

// SDK: cir.sdk_version = "12.3"

// NOSDK-NOT: cir.sdk_version
