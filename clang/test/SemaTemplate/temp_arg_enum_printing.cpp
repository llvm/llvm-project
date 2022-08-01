// RUN: %clang_cc1 -fsyntax-only -ast-print %s | FileCheck %s

namespace NamedEnumNS
{
  
enum class NamedEnum
{
  Val0,
  Val1
};
  
template <NamedEnum E>
void foo();
  
void test() {
  // CHECK: template<> void foo<NamedEnumNS::NamedEnum::Val0>()
  NamedEnumNS::foo<NamedEnum::Val0>();
  // CHECK: template<> void foo<NamedEnumNS::NamedEnum::Val1>()
  NamedEnumNS::foo<(NamedEnum)1>();
  // CHECK: template<> void foo<(NamedEnumNS::NamedEnum)2>()
  NamedEnumNS::foo<(NamedEnum)2>();
}
  
} // NamedEnumNS
