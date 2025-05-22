// REQUIRES: !system-windows

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 -emit-module-interface \
// RUN:     %t/Mod.cppm -o %t/Mod.pcm
//
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 %t/Mod.pcm \
// RUN:     -emit-llvm -o - | FileCheck %t/Mod.cppm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 -fmodule-file=Mod=%t/Mod.pcm \
// RUN:     %t/Use.cpp  -emit-llvm -o - | FileCheck %t/Use.cpp
//
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 -emit-module-interface \
// RUN:     %t/Mod.cppm -o %t/Mod.pcm -DKEY_FUNCTION_INLINE
//
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 %t/Mod.pcm \
// RUN:     -emit-llvm -o - | FileCheck %t/Mod.cppm -check-prefix=CHECK-INLINE
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 -fmodule-file=Mod=%t/Mod.pcm \
// RUN:     %t/Use.cpp  -emit-llvm -o - | FileCheck %t/Use.cpp -check-prefix=CHECK-INLINE
//
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 -emit-module-interface \
// RUN:     %t/M-A.cppm -o %t/M-A.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 -fmodule-file=M:A=%t/M-A.pcm \
// RUN:     %t/M-B.cppm  -emit-llvm -o - | FileCheck %t/M-B.cppm

//--- Mod.cppm
export module Mod;

export class Base {
public:
    virtual ~Base();
};
#ifdef KEY_FUNCTION_INLINE
inline
#endif
Base::~Base() {}

// CHECK: @_ZTVW3Mod4Base = unnamed_addr constant
// CHECK: @_ZTSW3Mod4Base = constant
// CHECK: @_ZTIW3Mod4Base = constant

// CHECK-INLINE: @_ZTVW3Mod4Base = linkonce_odr {{.*}}unnamed_addr constant
// CHECK-INLINE: @_ZTSW3Mod4Base = linkonce_odr {{.*}}constant
// CHECK-INLINE: @_ZTIW3Mod4Base = linkonce_odr {{.*}}constant

module :private;
int private_use() {
    Base base;
    return 43;
}

//--- Use.cpp
import Mod;
int use() {
    Base* base = new Base();
    return 43;
}

// CHECK-NOT: @_ZTSW3Mod4Base = constant
// CHECK-NOT: @_ZTIW3Mod4Base = constant
// CHECK: @_ZTVW3Mod4Base = external unnamed_addr

// CHECK-INLINE: @_ZTVW3Mod4Base = linkonce_odr {{.*}}unnamed_addr constant
// CHECK-INLINE: @_ZTSW3Mod4Base = linkonce_odr {{.*}}constant
// CHECK-INLINE: @_ZTIW3Mod4Base = linkonce_odr {{.*}}constant

// Check the case that the declaration of the key function comes from another
// module unit but the definition of the key function comes from the current
// module unit.

//--- M-A.cppm
export module M:A;
export class C {
public:
    virtual ~C();
};

int a_use() {
    C c;
    return 43;
}

//--- M-B.cppm
export module M:B;
import :A;

C::~C() {}

int b_use() {
    C c;
    return 43;
}

// CHECK: @_ZTVW1M1C = unnamed_addr constant
// CHECK: @_ZTSW1M1C = constant
// CHECK: @_ZTIW1M1C = constant
