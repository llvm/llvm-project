// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s | FileCheck %s

// CHECK: CIRGenModule::buildTopLevelDecl

void foo() {}
