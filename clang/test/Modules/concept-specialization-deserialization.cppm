// Reading an ImplicitConceptSpecializationDecl's trailing template arguments
// requires deserializing a FunctionDecl whose FunctionProtoType has a computed
// noexcept specification. Uniquing that type profiles the noexcept expression,
// which is a ConceptSpecializationExpr pointing back at the decl currently
// being read, so StmtProfiler walks the trailing array before
// setTemplateArguments() has written it.
//
// Before the fix that array was raw bump-allocator storage, and this compiled
// clean, crashed, or silently mis-keyed the type depending on what happened to
// be in the slab. It is now value-initialized, so the read is well defined.
//
// This test does not fail deterministically without the fix -- the window opens
// on every run, but whether the uninitialized bytes cause a crash does not. It
// is a regression test in the sense that the read is reachable here, so a
// sanitizer build catches it.

// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++23 -emit-module-interface %t/a.cppm -o %t/a.pcm
// RUN: %clang_cc1 -std=c++23 -fmodule-file=a=%t/a.pcm -fsyntax-only %t/use.cpp

//--- shared.h
namespace stdexec {
struct __cp {
  template <class _Tp> using __f = _Tp;
};
template <class> __cp __cpcvr;
template <class _Tp> using __copy_cvref_fn = decltype(__cpcvr<_Tp>);
template <class _Fun>
concept __callable = requires(_Fun __fun) { __fun; };
template <class _Fun>
concept __nothrow_callable = requires(_Fun __fun) { __fun; };
template <class... _As>
concept constructible_from = __is_constructible(_As...);
template <class>
concept move_constructible = constructible_from<>;
template <class _Ty>
concept __movable_value = move_constructible<_Ty>;
template <auto _Value> using __mtypeof = decltype(_Value);
struct __ignore {
  template <class _Ts> __ignore(_Ts);
};
template <class _Fn, class _Arg> using __mcall1 = _Fn::template __f<_Arg>;
template <class... _Args>
concept _Ok = (__is_same(_Args, int) && ...);
template <int> struct __i {
  template <template <class> class _Fn, class... _Args>
  using __g = _Fn<_Args...>;
  template <class _Fn, class... _Args> using __f = _Fn::template __f<_Args...>;
};
template <template <class> class _Fn, class... _Args>
using __minvoke_q = __i<_Ok<>>::__g<_Fn, _Args...>;
template <class _Fn, class... _Args>
using __minvoke = __i<_Ok<>>::__f<_Fn, _Args...>;
template <template <class> class _Fn> struct __qq {
  template <class... _Args> using __f = _Fn<_Args...>;
};
template <template <class> class _Tp, class... _Args>
concept __minvocable_q = requires { typename __minvoke_q<_Tp, _Args...>; };
template <class _Fun, class... _As>
using __call_result_t = decltype(_Fun()(_As()...));
struct set_value_t;
struct just_t;
extern just_t just;
namespace __tup {
template <class...> struct __tuple {};
template <class _CvRef> struct __impl {
  template <class... _Ts> using __tuple_t = __mcall1<_CvRef, __tuple<_Ts...>>;
  template <class... _Ts, __callable _Fn>
  auto operator()(_Fn, __tuple_t<_Ts...>)
      -> __call_result_t<_Fn, __mcall1<_CvRef, _Ts>...>;
};
template <class _Tuple> using __impl_t = __impl<__copy_cvref_fn<_Tuple>>;
struct __apply_t {
  // The noexcept specification here is the ConceptSpecializationExpr that
  // reaches back into the decl being deserialized.
  template <class _Fn, class _Tuple>
  auto operator()(_Fn, _Tuple) noexcept(__nothrow_callable<_Tuple>)
      -> __call_result_t<__impl_t<_Tuple>, _Fn, _Tuple>;
};
} // namespace __tup
using __tup::__tuple;
template <class _Fn, class _Tuple>
using __apply_result_t = __call_result_t<__tup::__apply_t, _Fn, _Tuple>;
template <class, class, class, class> using __gather_completions_t = int;
template <class _EnvProvider>
using __get_env_member_result_t = decltype(_EnvProvider().get_env());
struct get_env_t {
  template <class _EnvProvider>
  auto operator()(_EnvProvider) -> __get_env_member_result_t<_EnvProvider>;
};
template <class _EnvProvider>
concept __environment_provider =
    __minvocable_q<__call_result_t, get_env_t, _EnvProvider>;
template <class _Sender>
concept sender = __environment_provider<_Sender>;
template <class _Sender, class>
concept sender_in = sender<_Sender>;
template <auto> struct __sexpr;
template <class _Tag, class _Data> struct __desc {
  using __tag = _Tag;
  template <class _Fn> using __f = __minvoke<_Fn, _Tag, _Data>;
};
template <class _Descriptor> auto __descriptor_fn_v = _Descriptor{};
template <class> struct __sexpr_impl;
template <class _Tag, class _Data>
using __sexpr_t = __sexpr<__descriptor_fn_v<__desc<_Tag, _Data>>>;
template <auto _DescriptorFn> struct __sexpr {
  using __desc_t = decltype(_DescriptorFn);
  using __base_t = __minvoke<__desc_t, __qq<__tuple>>;
  using __get_attrs_t =
      __mtypeof<__sexpr_impl<typename __desc_t::__tag>::__get_attrs>;
  using __attrs_t = __apply_result_t<__get_attrs_t, __base_t>;
  auto get_env() -> __attrs_t;
};
struct __make_sexpr_t {
  template <class _Data> auto operator()(_Data) {
    return __sexpr_t<just_t, _Data>{};
  }
};
template <class> __make_sexpr_t __make_sexpr;
template <class _Sender, class _SetSig>
concept sender_of =
    sender_in<_Sender, __gather_completions_t<_SetSig, _Sender, int, int>>;
struct just_t {
  template <__movable_value... _Ts> auto operator()(_Ts... __ts) {
    return __make_sexpr<just_t>(__tuple{__ts...});
  }
};
template <> struct __sexpr_impl<just_t> {
  static void __get_attrs(__ignore, __ignore);
};
using __force = __call_result_t<just_t>;
} // namespace stdexec

//--- a.cppm
module;
#include "shared.h"
export module a;

export template <stdexec::sender_of<stdexec::set_value_t()> T> void ta(T) {}
export using stdexec::just;
export using stdexec::set_value_t;

//--- use.cpp
import a;

// Calling ta() forces constraint satisfaction, which triggers the lazy
// specialization lookup that deserializes the ImplicitConceptSpecializationDecl.
void f() { ta(just()); }
