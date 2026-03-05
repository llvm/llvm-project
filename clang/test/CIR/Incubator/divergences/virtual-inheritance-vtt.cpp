// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Virtual inheritance produces VTT (Virtual Table Table) with divergences:
// 1. Missing comdat on VTT, vtable, type info
// 2. Type info linkage: should be linkonce_odr constant with comdat, but is just constant
// 3. Missing unnamed_addr on vtables
// 4. Missing inrange annotations on GEP instructions in VTT
// 5. String constants missing null terminators
//
// CodeGen:
//   $_ZTT7Diamond = comdat any
//   $_ZTV7Diamond = comdat any
//   $_ZTI4Base = comdat any
//   @_ZTV7Diamond = linkonce_odr unnamed_addr constant {...}, comdat
//   @_ZTT7Diamond = linkonce_odr unnamed_addr constant [4 x ptr] [
//     ptr getelementptr inbounds inrange(-24, 0) ({...}, ptr @_ZTV7Diamond, ...)
//   ], comdat
//   @_ZTI4Base = linkonce_odr constant {...}, comdat
//   @_ZTS4Base = linkonce_odr constant [6 x i8] c"4Base\00", comdat
//
// CIR:
//   @_ZTV7Diamond = linkonce_odr global {...}  (no comdat, no unnamed_addr)
//   @_ZTT7Diamond = linkonce_odr global [4 x ptr] [
//     ptr getelementptr inbounds nuw (i8, ptr @_ZTV7Diamond, ...)  (no inrange)
//   ]  (no comdat, no unnamed_addr)
//   @_ZTI4Base = constant {...}  (no linkonce_odr, no comdat)
//   @_ZTS4Base = linkonce_odr global [5 x i8] c"4Base", comdat  (no \00)

// DIFF: -$_ZTV7Diamond = comdat any
// DIFF: -$_ZTT7Diamond = comdat any
// DIFF: -$_ZTI{{.*}} = comdat any
// DIFF: -@_ZTV7Diamond = linkonce_odr unnamed_addr constant
// DIFF: +@_ZTV7Diamond = linkonce_odr global
// DIFF: -@_ZTT7Diamond = linkonce_odr unnamed_addr constant
// DIFF: +@_ZTT7Diamond = linkonce_odr global
// DIFF: -@_ZTI{{.*}} = linkonce_odr constant
// DIFF: +@_ZTI{{.*}} = constant
// DIFF: inrange(-24, 0)
// DIFF: c"{{.*}}\00"
// DIFF: -c"{{.*}}\00"
// DIFF: +c"{{.*}}"

struct Base {
    int x = 10;
};

struct Derived1 : virtual Base {
    int y = 20;
};

struct Derived2 : virtual Base {
    int z = 30;
};

struct Diamond : Derived1, Derived2 {
    int w = 40;
};

int test() {
    Diamond d;
    return d.x + d.y + d.z + d.w;
}
