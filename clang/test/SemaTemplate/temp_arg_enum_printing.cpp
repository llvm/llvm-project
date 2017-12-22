// RUN: %clang_cc1 -fsyntax-only -ast-print %s | FileCheck %s

namespace NamedEnumNS
{
  
enum NamedEnum
{
  Val0,
  Val1
};
  
template <NamedEnum E>
void foo();
  
void test() {
  // CHECK: template<> void foo<NamedEnumNS::Val0>()
  NamedEnumNS::foo<Val0>();
  // CHECK: template<> void foo<NamedEnumNS::Val1>()
  NamedEnumNS::foo<(NamedEnum)1>();
  // CHECK: template<> void foo<2>()
  NamedEnumNS::foo<(NamedEnum)2>();
}
  
} // NamedEnumNS
