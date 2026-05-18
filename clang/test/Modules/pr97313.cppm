// REQUIRES: !system-windows, !system-cygwin
//
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/Base.cppm \
// RUN:     -emit-module-interface -o %t/Base.pcm
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/Sub.cppm \
// RUN:     -emit-module-interface -o %t/Sub.pcm -fprebuilt-module-path=%t
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/Sub.pcm \
// RUN:     -emit-llvm -o %t/Sub.pcm -o - -fprebuilt-module-path=%t | \
// RUN:     FileCheck %t/Sub.cppm
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/main.cpp \
// RUN:     -emit-llvm -fprebuilt-module-path=%t -o - | FileCheck %t/main.cpp
//
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/Mod.cppm \
// RUN:     -emit-module-interface -o %t/Mod.pcm
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/Mod.pcm \
// RUN:     -emit-llvm -o - | FileCheck %t/Mod.cppm
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/Use.cpp \
// RUN:     -emit-llvm -fprebuilt-module-path=%t -o - | \
// RUN:     FileCheck %t/Use.cpp

//--- Base.cppm
export module Base;

export template <class>
class Base
{
public:
    constexpr Base();
    constexpr virtual ~Base();
};

template <class X>
constexpr Base<X>::Base() = default;

template <class X>
constexpr Base<X>::~Base() = default;

//--- Sub.cppm
export module Sub;
export import Base;

export class Sub : public Base<int>
{
};

// CHECK: @_ZTIW4Base4BaseIiE = {{.*}}linkonce_odr

//--- main.cpp
import Sub;

int main()
{
    Base<int> *b = new Sub();
    delete b;
}

// CHECK: @_ZTIW4Base4BaseIiE = {{.*}}linkonce_odr

//--- Mod.cppm
export module Mod;

export class NonTemplate {
public:
    virtual ~NonTemplate();
};

// CHECK: @_ZTIW3Mod11NonTemplate = {{.*}}constant

export template <class C>
class Template {
public:
    virtual ~Template();
};

export template<>
class Template<char> {
public:
    virtual ~Template();
};

// CHECK: @_ZTIW3Mod8TemplateIcE = {{.*}}constant

export template class Template<unsigned>;

// CHECK: @_ZTIW3Mod8TemplateIjE = {{.*}}weak_odr

export extern template class Template<double>;

auto v = new Template<signed int>();

// CHECK: @_ZTIW3Mod8TemplateIiE = {{.*}}linkonce_odr

//--- Use.cpp
import Mod;

auto v1 = new NonTemplate();
auto v2 = new Template<char>();
auto v3 = new Template<unsigned>();
auto v4 = new Template<double>();
auto v5 = new Template<signed int>();
auto v6 = new Template<NonTemplate>();

// CHECK: @_ZTVW3Mod11NonTemplate = {{.*}}external
// CHECK: @_ZTVW3Mod8TemplateIcE = {{.*}}external
// CHECK: @_ZTVW3Mod8TemplateIjE = {{.*}}weak_odr
// CHECK: @_ZTIW3Mod8TemplateIjE = {{.*}}weak_odr
// CHECK: @_ZTSW3Mod8TemplateIjE = {{.*}}weak_odr
// CHECK: @_ZTVW3Mod8TemplateIdE = {{.*}}external
// CHECK: @_ZTVW3Mod8TemplateIiE = {{.*}}linkonce_odr
// CHECK: @_ZTIW3Mod8TemplateIiE = {{.*}}linkonce_odr
// CHECK: @_ZTSW3Mod8TemplateIiE = {{.*}}linkonce_odr
// CHECK: @_ZTVW3Mod8TemplateIS_11NonTemplateE = {{.*}}linkonce_odr
// CHECK: @_ZTIW3Mod8TemplateIS_11NonTemplateE = {{.*}}linkonce_odr
// CHECK: @_ZTSW3Mod8TemplateIS_11NonTemplateE = {{.*}}linkonce_odr
