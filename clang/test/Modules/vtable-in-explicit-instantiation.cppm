// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 %t/a.cppm -triple %itanium_abi_triple -emit-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/a.cc -triple %itanium_abi_triple -fmodule-file=a=%t/a.pcm -emit-llvm -o - | FileCheck %t/a.cc
//
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -triple %itanium_abi_triple -emit-reduced-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/a.cc -triple %itanium_abi_triple -fmodule-file=a=%t/a.pcm -emit-llvm -o - | FileCheck %t/a.cc

//--- a.cppm
export module a;
class base {
public:
    ~base() = default;
    virtual void foo();
};

template <class T>
class a : public base {
public:
    virtual void foo() override;
};

extern template class a<int>;

//--- a.cc
module a;

template <class T>
void a<T>::foo() {}

template class a<int>;
// CHECK: _ZTVW1a1aIiE
