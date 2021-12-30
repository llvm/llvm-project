// Fake standard library with uglified names.
// Parameters (including template params) get ugliness stripped.
namespace std {

template <typename _Tp>
class __vector_base {};

template <typename _Tp>
class vector : private __vector_base<_Tp> {
public:
  _Tp &at(unsigned __index) const;
  int __stays_ugly();
};

} // namespace std

int x = std::vector<int>{}.at(42);
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:17:14 %s -o - | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: COMPLETION: __vector_base : __vector_base<<#typename Tp#>>
// CHECK-CC1: COMPLETION: vector : vector<<#typename Tp#>>
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:17:28 %s -o - | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: COMPLETION: __stays_ugly : [#int#]__stays_ugly()
// CHECK-CC2: COMPLETION: at : [#int &#]at(<#unsigned int index#>)[# const#]
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:17:31 %s -o - | FileCheck -check-prefix=CHECK-CC3 %s
// CHECK-CC3: OVERLOAD: [#int &#]at(<#unsigned int index#>)
