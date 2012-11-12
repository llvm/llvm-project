// RUN: %clang_cc1 -triple=i386-pc-solaris2.11 -w -emit-llvm %s -o - | FileCheck %s

#pragma redefine_extname fake real
#pragma redefine_extname name alias

extern int fake(void);

int name;

// __PRAGMA_REDEFINE_EXTNAME should be defined.  This will fail if it isn't...
int fish() { return fake() + __PRAGMA_REDEFINE_EXTNAME + name; }
// Check that the call to fake() is emitted as a call to real()
// CHECK:   call i32 @real()
// Check that this also works with variables names
// CHECK:   load i32* @alias
