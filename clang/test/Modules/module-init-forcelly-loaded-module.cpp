// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: %clang_cc1 -fmodule-name=test -fno-cxx-modules -fmodule-map-file-home-is-cwd -xc++ -emit-module \
// RUN:     -triple %itanium_abi_triple \
// RUN:     -iquote . -fmodules -fno-implicit-modules %t/test.cppmap -o %t/test.pcm
// RUN: %clang_cc1 -fmodule-name=test2 -fno-cxx-modules -fmodule-map-file-home-is-cwd -xc++ -emit-module \
// RUN:     -iquote . -fmodules -fno-implicit-modules %t/test2.cppmap -o %t/test2.pcm \
// RUN:     -triple %itanium_abi_triple
// RUN: %clang_cc1 -fno-cxx-modules -fmodule-map-file-home-is-cwd -fmodules -fno-implicit-modules \
// RUN:     -fmodule-file=%t/test.pcm -fmodule-file=%t/test2.pcm %t/test.cc \
// RUN:     -triple %itanium_abi_triple \
// RUN:     -emit-llvm -o - | FileCheck %t/test.cc

//--- common.h
#ifndef COMMON_H
#define COMMON_H
extern "C" void exit(int);
namespace namespace_foo {
class FooBar {};
namespace namespace_bar {
class RegisterOnce;
template <const FooBar&>
struct FooBarRegisterer {
  static const RegisterOnce kRegisterOnce;
};
class RegisterOnce {};
void RegisterFooBarImpl();
template <const FooBar& t>
const RegisterOnce FooBarRegisterer<t>::kRegisterOnce =
    (RegisterFooBarImpl(), RegisterOnce());
template <FooBar& tag,
          const RegisterOnce& =
              FooBarRegisterer<tag>::kRegisterOnce>
FooBar CreateFooBarInternal ;
}  
template <FooBar& tag>
FooBar CreateFooBar(
    int , int ,
    int ) {
  return namespace_bar::CreateFooBarInternal<tag>;
}
FooBar kNullArgumentFooBar =
    CreateFooBar<kNullArgumentFooBar>(1, 0, 0);
namespace namespace_bar {
void RegisterFooBarImpl() {
  static bool donealready = false;
  if (donealready) 
    exit(1);
  donealready = true;
}
}
}
#endif

//--- test.cc
int main() {}

// Check that we won't have multiple initializer by not calling RegisterFooBarImpl twice
// CHECK: call {{.*}}@_ZN13namespace_foo13namespace_bar18RegisterFooBarImplEv
// CHECK-NOT: call {{.*}}@_ZN13namespace_foo13namespace_bar18RegisterFooBarImplEv

//--- test.cppmap
module "test" {
    header "test.h"
}

//--- test.h
#ifndef test
#include "common.h"
#endif  

//--- test2.cppmap
module "test2" {
    header "test2.h"
}

//--- test2.h
#ifndef test2
#include "common.h"
#endif
