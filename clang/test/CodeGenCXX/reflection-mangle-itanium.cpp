// RUN: %clang_cc1 -std=c++26 -freflection -triple x86_64-unknown-linux-gnu -fblocks \
// RUN:   -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -std=c++26 -freflection -triple x86_64-unknown-linux-gnu -fblocks \
// RUN:   -fexperimental-new-constant-interpreter -emit-llvm -o - %s | FileCheck %s

using info = decltype(^^int);
using Type = int;

template <auto A>
void foo () {}

int main() {
  foo <info {}> ();
  // CHECK: @_Z3fooITnDaLDmnuEEvv
  foo <^^void> ();
  // CHECK: @_Z3fooITnDaLDmtyvEEvv
  foo <^^bool> ();
  // CHECK: @_Z3fooITnDaLDmtybEEvv
  foo <^^char> ();
  // CHECK: @_Z3fooITnDaLDmtycEEvv
  foo <^^signed char> ();
  // CHECK: @_Z3fooITnDaLDmtyaEEvv
  foo <^^unsigned char> ();
  // CHECK: @_Z3fooITnDaLDmtyhEEvv
  foo <^^short> ();
  // CHECK: @_Z3fooITnDaLDmtysEEvv
  foo <^^unsigned short> ();
  // CHECK: @_Z3fooITnDaLDmtytEEvv
  foo <^^int> ();
  // CHECK: @_Z3fooITnDaLDmtyiEEvv
  foo <^^unsigned int> ();
  // CHECK: @_Z3fooITnDaLDmtyjEEvv
  foo <^^long> ();
  // CHECK: @_Z3fooITnDaLDmtylEEvv
  foo <^^unsigned long> ();
  // CHECK: @_Z3fooITnDaLDmtymEEvv
  foo <^^long long> ();
  // CHECK: @_Z3fooITnDaLDmtyxEEvv
  foo <^^unsigned long long> ();
  // CHECK: @_Z3fooITnDaLDmtyyEEvv
  foo <^^float> ();
  // CHECK: @_Z3fooITnDaLDmtyfEEvv
  foo <^^double> ();
  // CHECK: @_Z3fooITnDaLDmtydEEvv
  foo <^^long double> ();
  // CHECK: @_Z3fooITnDaLDmtyeEEvv
  foo <^^__int128> ();
  // CHECK: @_Z3fooITnDaLDmtynEEvv
  foo <^^unsigned __int128> ();
  // CHECK: @_Z3fooITnDaLDmtyoEEvv
  foo <^^Type> ();
  // CHECK: @_Z3fooITnDaLDmta4TypeEEvv
  return 0;
}

struct Record {};
struct WithType { using type = int; };
using Alias = Record;
using Pointer = int *;
Pointer pointer;
template <auto R> void reflect() {}

template <auto R> struct Reflected {};
template <class T> Reflected<^^T *> dependent() { return {}; }

void test() {
  // CHECK: call void @_Z7reflectITnDaLDmtyP6RecordEEvv()
  reflect<^^Record *>();
  // CHECK: call void @_Z7reflectITnDaLDmtyA3_iEEvv()
  reflect<^^int[3]>();
  // CHECK: call void @_Z7reflectITnDaLDmtyFviEEEvv()
  reflect<^^void(int)>();
  // CHECK: call void @_Z7reflectITnDaLDmtyM6RecordiEEvv()
  reflect<^^int Record::*>();
  // CHECK: call void @_Z7reflectITnDaLDmta5AliasEEvv()
  reflect<^^Alias>();
  // CHECK: call void @_Z7reflectITnDaLDmtyDv4_iEEvv()
  reflect<^^int __attribute__((ext_vector_type(4)))>();
  // CHECK: call void @_Z7reflectITnDaLDmtyU13block_pointerFviEEEvv()
  reflect<^^void (^)(int)>();
  // CHECK: call void @_Z7reflectITnDaLDmtyDB17_EEvv()
  reflect<^^_BitInt(17)>();
  // CHECK: call {{.*}} @_Z9dependentIiE9ReflectedILDmtyPT_EEv()
  dependent<int>();
  // CHECK: call void @_Z7reflectITnDaLDmtyPiEEvv()
  reflect<^^decltype(pointer)>();
  // CHECK: call void @_Z7reflectITnDaLDmta7PointerEEvv()
  reflect<^^Pointer>();
  // CHECK: call void @_Z7reflectITnDaLDmtyiEEvv()
  reflect<^^typename WithType::type>();
}

namespace AliasTemplates {
template <class T = int> using A = T;
template <class T> using B = T;
template <unsigned N, unsigned M = N + 1> using Numbers = int;
template <class... Ts> using Pack = int;
template <class T, class... Ts> using MixedPack = int;
template <template <class> class... TTs> using Templates = int;
template <class T> using Ignored = int;
template <info R> using Reflection = int;
namespace Nested { template <class T> using Alias = T; }
template <class T> struct Owner { template <class U> using Alias = int; };
template <class T> Reflected<^^A<T>> dependent_alias() { return {}; }
// CHECK-LABEL: define {{.*}} @_ZN14AliasTemplates5callsEv()
void calls() {
  // CHECK: call void @_Z7reflectITnDaLDmtaN14AliasTemplates1AIiEEEEvv()
  reflect<^^A<>>();
  // CHECK: call void @_Z7reflectITnDaLDmtaN14AliasTemplates1BIiEEEEvv()
  reflect<^^B<int>>();
  // CHECK: call void @_Z7reflectITnDaLDmtaN14AliasTemplates7NumbersILj2ELj3EEEEEvv()
  reflect<^^Numbers<1 + 1>>();
  // CHECK: call void @_Z7reflectITnDaLDmtaN14AliasTemplates4PackIJifEEEEEvv()
  reflect<^^Pack<int, float>>();
  // CHECK: call void @_Z7reflectITnDaLDmtaN14AliasTemplates9MixedPackIiJfdEEEEEvv()
  reflect<^^MixedPack<int, float, double>>();
  // CHECK: call void @_Z7reflect{{.*}}9TemplatesIJ{{.*}}EEEEEvv()
  reflect<^^Templates<A, B>>();
  // CHECK: call void @_Z7reflectITnDaLDmtaN14AliasTemplates6Nested5AliasIiEEEEvv()
  reflect<^^Nested::Alias<int>>();
  // CHECK: call void @_Z7reflectITnDaLDmtaN14AliasTemplates5OwnerIiE5AliasIfEEEEvv()
  reflect<^^Owner<int>::Alias<float>>();
  // CHECK: call void @_Z7reflectITnDaLDmtaN14AliasTemplates7IgnoredIiEEEEvv()
  reflect<^^Ignored<int>>();
  // CHECK: call void @_Z7reflectITnDaLDmtaN14AliasTemplates7IgnoredIfEEEEvv()
  reflect<^^Ignored<float>>();
  // CHECK: call void @_Z7reflectITnDaLDmtaN14AliasTemplates10ReflectionILDmtaNS0_1AIiEEEEEEEvv()
  reflect<^^Reflection<^^A<int>>>();
  // CHECK: call {{.*}} @_ZN14AliasTemplates15dependent_aliasIiEE9ReflectedILDmtaNS_1AIT_EEEEv()
  dependent_alias<int>();
}
}

namespace Substituted {
template <info R> struct Owner {
  template <int N> static void f(int (&)[R == ^^int ? N : N + 1]) {}
};
// CHECK-LABEL: define {{.*}} @_ZN11Substituted5callsEv()
void calls() {
  int a[2];
  // CHECK: call void @_ZN11Substituted5Owner{{.*}}eqLDmtyiELDmtyiE
  Owner<^^int>::f<2>(a);
  // CHECK: call void @_ZN11Substituted5Owner{{.*}}eqLDmnuELDmtyiE
  Owner<info{}>::f<1>(a);
  // CHECK: call void @_ZN11Substituted5Owner{{.*}}eqLDmta4TypeELDmtyiE
  Owner<^^Type>::f<1>(a);
  // CHECK: call void @_ZN11Substituted5Owner{{.*}}eqLDmtaNS2_IiEEELDmtyiE
  Owner<^^AliasTemplates::Ignored<int>>::f<1>(a);
}
}
