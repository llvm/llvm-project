// RUN: %clang_cc1 -emit-llvm -triple %itanium_abi_triple %s -o - -debug-info-kind=standalone %std_cxx98-14 | FileCheck %s --check-prefixes=CHECK,PRE17
// RUN: %clang_cc1 -emit-llvm -triple %itanium_abi_triple %s -o - -debug-info-kind=standalone %std_cxx17- | FileCheck %s --check-prefixes=CHECK,CXX17
namespace __pointer_type_imp
{
  template <class _Tp, class _Dp, bool > struct __pointer_type1 {};

  // CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "__pointer_type1<C, default_delete<C>, false>",
  // CHECK-SAME:             templateParams: ![[PARAMS:[0-9]+]]
  // CHECK-SAME:             identifier: "_ZTSN18__pointer_type_imp15__pointer_type1I1C14default_deleteIS1_ELb0EEE"
  template <class _Tp, class _Dp> struct __pointer_type1<_Tp, _Dp, false>
  {
    typedef _Tp* type;
  };
}
template <class _Tp, class _Dp>
struct __pointer_type2
{
  // Test that the bool template type parameter is emitted.
  //
  // CHECK: ![[PARAMS]] = !{!{{.*}}, !{{.*}}, ![[FALSE:[0-9]+]]}
  // PRE17: ![[FALSE]] = !DITemplateValueParameter(type: !{{[0-9]+}}, value: i8 0)
  // CXX17: ![[FALSE]] = !DITemplateValueParameter(type: !{{[0-9]+}}, value: i1 false)
  typedef typename __pointer_type_imp::__pointer_type1<_Tp, _Dp, false>::type type;
};
template <class _Tp> struct default_delete {};
template <class _Tp, class _Dp = default_delete<_Tp> > class unique_ptr
{
  typedef typename __pointer_type2<_Tp, _Dp>::type pointer;
  unique_ptr(pointer __p, _Dp __d) {}
};
class C {
  unique_ptr<C> Ptr;
};
void foo(C &c) {
}
