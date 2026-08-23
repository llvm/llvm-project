

#include <__type_traits/is_unqualified.h>

#include "test_macros.h"
#include <type_traits>
#include <bit>

namespace support {

#if TEST_STD_VER >= 26

// TODO: use _ prefix
template <typename _Tp>
concept EnabledForHash = requires(_Tp t) {
  { std::bool_constant<std::__is_unqualified_v<_Tp>>() } -> std::same_as<std::true_type>;
};

template <typename _Tp>
concept DisabledForHash = not EnabledForHash<_Tp>;

// TODO: document the constraints of using this at runtime OR make it consteval only
template <typename _Tp>
struct constexpr_hash;

template <DisabledForHash _Tp>
struct constexpr_hash<_Tp> {
  constexpr_hash()                                   = delete;
  constexpr_hash(const constexpr_hash&)            = delete;
  constexpr_hash& operator=(const constexpr_hash&) = delete;
};

template <EnabledForHash _Tp>
struct constexpr_hash<_Tp> {
  [[__nodiscard__]] constexpr _LIBCPP_HIDE_FROM_ABI size_t operator()(const _Tp& __v) const noexcept {
    if constexpr (std::is_same_v<_Tp, nullptr_t>) {
      return 662607004ull;
    } else if constexpr (std::is_integral_v<_Tp>) {
      if constexpr (sizeof(_Tp) <= sizeof(size_t)) {
        return static_cast<size_t>(__v);
      } else {
        constexpr size_t multiple = sizeof(_Tp) / sizeof(size_t);
        char region[multiple];

        // TODO: 0, 1, 2, 3, 4
        return region[multiple - 1]; // TODO: hash-ing
      }
    } else if constexpr (std::is_floating_point_v<_Tp>) {
    if (__v == 0.0f)
      return 0;
    // return __scalar_hash<_Tp>::operator()(__v);

        constexpr size_t multiple = sizeof(_Tp) / sizeof(char);
        std::array<char, multiple> region = std::bit_cast<std::array<char, multiple>>(__v);

        // TODO: pack indexing?
        size_t res = 0;
        for(size_t i = 0; i < multiple; ++i) {
            res ^= region[i];
        }
        return res;

        // TODO: 0, 1, 2, 3, 4
        // return region[multiple - 1]; // TODO: hash-ing
        // return 1234ULL;

    }
    __builtin_unreachable(); // todo: revisit
  }

  constexpr_hash() noexcept                          = default;
  constexpr_hash& operator=(const constexpr_hash&) = default;
};

} // namespace support

#endif
