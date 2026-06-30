#include "this_header_does_not_exist.h"

namespace GH202956 {

template <typename T> struct type_identity { using type = T; };
template <typename T> using type_identity_t = typename type_identity<T>::type;

template <typename E> class box { E e; };

template <typename T, typename E> class wrapper {
  template <typename U, typename G, typename Unex = box<E>,
            typename = type_identity_t<T>>
  static constexpr bool cons_from = true;
  template <typename U, typename G, typename Unex>
  static constexpr bool cons_from<U, G, Unex, bool> = false;
};

} // namespace GH202956
