// Header used to build a precompiled header that contains compiler errors.
//
// The unresolved #include makes this a "PCH built with compiler errors"
// (clangd always builds preambles with -fallow-pch-with-compiler-errors, and an
// unresolved include is a routine state in interactive editing). The class
// template below is a reduction of libstdc++ 15's std::expected, whose private
// member variable template __cons_from_expected has a partial specialization.
// See GH202956.

#include "this_header_does_not_exist.h"

namespace GH202956 {

template <typename T> struct type_identity { using type = T; };
template <typename T> using type_identity_t = typename type_identity<T>::type;

template <typename E> class box { E e; };

template <typename T, typename E> class wrapper {
  // A member variable template whose last template parameter has a default
  // argument that is a dependent alias-template specialization, plus a partial
  // specialization that fixes that parameter. This mirrors libstdc++'s
  // __cons_from_expected<_Up, _Gr, _Unex = unexpected<_Er>, = remove_cv_t<_Tp>>.
  template <typename U, typename G, typename Unex = box<E>,
            typename = type_identity_t<T>>
  static constexpr bool cons_from = true;
  template <typename U, typename G, typename Unex>
  static constexpr bool cons_from<U, G, Unex, bool> = false;
};

} // namespace GH202956
