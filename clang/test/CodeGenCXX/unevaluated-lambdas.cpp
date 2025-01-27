// RUN: %clang_cc1 -std=c++2b -emit-llvm %s -o - | FileCheck %s -dump-input=always

namespace GH82926 {

template<class Tp>
using simd_vector = Tp;

template<class VecT>
using simd_vector_underlying_type_t
    = decltype([]<class Tp>(simd_vector<Tp>) {}(VecT {}), 1);

template<class VecT>
void temp() {
  // CHECK: call void @_ZZN7GH829264tempIcEEvvENKUliE_clEi
  [](simd_vector_underlying_type_t<VecT>) {}(42);
}

void call() {
  temp<simd_vector<char>>(); 
}

} // namespace GH82926

namespace GH111058 {

// FIXME: This still crashes because the unevaluated lambda as an argument
// is also supposed to skipping codegen in Sema::InstantiateFunctionDefinition().
// auto eat(auto) {}

void foo() {
	// [] -> decltype(eat([] {})) {};

  // CHECK: call void @"_ZZN8GH1110583fooEvENK3$_0clEv"
  [] -> decltype([](auto){}(1)) {}();
}

} // namespace GH111058
