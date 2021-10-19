// REQUIRES: xcore-registered-target

// RUN: %clang_cc1 -triple xcore-unknown-unknown -fno-signed-char -fno-common -emit-llvm -o - -x c++ %s | FileCheck %s

// CHECK: target triple = "xcore-unknown-unknown"


// C++ constants are not placed into the ".cp.rodata" section.
// CHECK: @cgx = external constant i32
extern const int cgx;
int fcgx() { return cgx;}
// CHECK: @g1 ={{.*}} global i32 0, align 4
int g1;
// CHECK: @cg1 ={{.*}} constant i32 0, align 4
extern const int cg1 = 0;

// Regression test for a bug in lib/CodeGen/CodeGenModule.cpp which called
// getLanguageLinkage() via a null 'VarDecl*'. This was an XCore specific
// conditional call to GV->setSection(".cp.rodata").
class C {
public:
  ~C(){};
};
C c;

// CHECK: "frame-pointer"="none"
