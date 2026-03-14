//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <experimental/simd>
//
// [simd.class]
// simd() noexcept = default;

#include "../test_utils.h"
#include <experimental/simd>

namespace ex = std::experimental::parallelism_v2;

// See https://www.open-std.org/jtc1/sc22/WG21/docs/papers/2019/n4808.pdf
// Default initialization performs no initialization of the elements; value-initialization initializes each element with T().
// Thus, default initialization leaves the elements in an indeterminate state.
template <class T, std::size_t>
struct CheckSimdDefaultCtor {
  template <class SimdAbi>
  void operator()() {
    static_assert(std::is_nothrow_default_constructible_v<ex::simd<T, SimdAbi>>);
    ex::simd<T, SimdAbi> pure_simd;
    // trash value in default ctor
    static_assert(pure_simd.size() > 0);
  }
};

template <class T, std::size_t>
struct CheckSimdDefaultCopyCtor {
  template <class SimdAbi>
  void operator()() {
    ex::simd<T, SimdAbi> pure_simd([](T i) { return i; });
    constexpr std::size_t array_size = ex::simd_size_v<T, SimdAbi>;
    std::array<T, array_size> expected_value;
    for (size_t i = 0; i < array_size; ++i)
      expected_value[i] = pure_simd[i];

    static_assert(std::is_nothrow_copy_constructible_v<ex::simd<T, SimdAbi>>);
    ex::simd<T, SimdAbi> from_copy_ctor(pure_simd);
    assert_simd_values_equal<array_size>(from_copy_ctor, expected_value);
  }
};

template <class T, std::size_t>
struct CheckSimdDefaultMoveCtor {
  template <class SimdAbi>
  void operator()() {
    ex::simd<T, SimdAbi> pure_simd([](T i) { return i; });
    constexpr std::size_t array_size = ex::simd_size_v<T, SimdAbi>;
    std::array<T, array_size> expected_value;
    for (size_t i = 0; i < array_size; ++i)
      expected_value[i] = pure_simd[i];

    static_assert(std::is_nothrow_move_constructible_v<ex::simd<T, SimdAbi>>);
    ex::simd<T, SimdAbi> from_move_ctor(std::move(pure_simd));
    assert_simd_values_equal<array_size>(from_move_ctor, expected_value);
  }
};

template <class T, std::size_t>
struct CheckSimdDefaultCopyAssignment {
  template <class SimdAbi>
  void operator()() {
    ex::simd<T, SimdAbi> pure_simd([](T i) { return i; });
    constexpr std::size_t array_size = ex::simd_size_v<T, SimdAbi>;
    std::array<T, array_size> expected_value;
    for (size_t i = 0; i < array_size; ++i)
      expected_value[i] = pure_simd[i];

    static_assert(std::is_nothrow_copy_assignable_v<ex::simd<T, SimdAbi>>);
    ex::simd<T, SimdAbi> from_copy_assignment;
    from_copy_assignment = pure_simd;
    assert_simd_values_equal<array_size>(from_copy_assignment, expected_value);
  }
};

template <class T, std::size_t>
struct CheckSimdDefaultMoveAssignment {
  template <class SimdAbi>
  void operator()() {
    ex::simd<T, SimdAbi> pure_simd([](T i) { return i; });
    constexpr std::size_t array_size = ex::simd_size_v<T, SimdAbi>;
    std::array<T, array_size> expected_value;
    for (size_t i = 0; i < array_size; ++i)
      expected_value[i] = pure_simd[i];

    static_assert(std::is_nothrow_move_assignable_v<ex::simd<T, SimdAbi>>);
    ex::simd<T, SimdAbi> from_move_assignment;
    from_move_assignment = std::move(pure_simd);
    assert_simd_values_equal<array_size>(from_move_assignment, expected_value);
  }
};

int main(int, char**) {
  test_all_simd_abi<CheckSimdDefaultCtor>();
  test_all_simd_abi<CheckSimdDefaultCopyCtor>();
  test_all_simd_abi<CheckSimdDefaultMoveCtor>();
  test_all_simd_abi<CheckSimdDefaultCopyAssignment>();
  test_all_simd_abi<CheckSimdDefaultMoveAssignment>();
  return 0;
}
