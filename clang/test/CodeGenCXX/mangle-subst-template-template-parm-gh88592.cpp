// RUN: %clang_cc1 -std=c++20 -fclang-abi-compat=latest -emit-llvm -triple x86_64-linux-gnu -o - %s | FileCheck %s
// RUN: %clang_cc1 -std=c++20 -fclang-abi-compat=23 -emit-llvm -triple x86_64-linux-gnu -o - %s | FileCheck --check-prefix=ABICOMPAT23 %s

namespace GH88592 {
  template <bool> struct A;
  template <typename> struct B { static constexpr bool value = false; };

  template <class T, template <class> class TT>
  using C = A<TT<T>::value>;

  // The alias substitutes B for TT without B having been mangled earlier, so
  // the substituted template template parameter can't be a substitution and
  // must be mangled as the replacement template itself.
  template <class T, C<T, B> * = nullptr> void f() {}
  template void f<int>();
  // CHECK: @_ZN7GH885921fIiTnPNS_1AIXsr1BIT_EE5valueEEELPNS1_ILb0EEE0EEEvv
  // Clang 23 and before mangled nothing for the template name, producing an
  // ill-formed symbol.
  // ABICOMPAT23: @_ZN7GH885921fIiTnPNS_1AIXsrIT_EE5valueEEELPNS1_ILb0EEE0EEEvv
} // namespace GH88592
