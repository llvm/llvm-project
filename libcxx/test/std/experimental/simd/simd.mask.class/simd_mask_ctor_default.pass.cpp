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
// [simd.mask.class]
// simd_mask() noexcept = default;

#include "../test_utils.h"
#include <experimental/simd>

namespace ex = std::experimental::parallelism_v2;

// See https://www.open-std.org/jtc1/sc22/WG21/docs/papers/2019/n4808.pdf
// Default initialization performs no initialization of the elements; value-initialization initializes each element with T().
// Thus, default initialization leaves the elements in an indeterminate state.
template <class T, std::size_t>
struct CheckSimdMaskDefaultCtor {
  template <class SimdAbi>
  void operator()() {
    ex::simd_mask<T, SimdAbi> pure_mask;
    // trash value in default ctor
    static_assert(pure_mask.size() > 0);
  }
};

template <class T, std::size_t>
struct CheckSimdMaskDefaultCopyCtor {
  template <class SimdAbi>
  void operator()() {
    ex::simd_mask<T, SimdAbi> pure_mask;
    constexpr std::size_t array_size = ex::simd_size_v<T, SimdAbi>;
    std::array<bool, array_size> expected_value;
    for (size_t i = 0; i < array_size; ++i) {
      if (i % 2 == 0)
        pure_mask[i] = true;
      else
        pure_mask[i] = false;
      expected_value[i] = pure_mask[i];
    }

    ex::simd_mask<T, SimdAbi> from_copy_ctor(pure_mask);
    assert_simd_mask_values_equal<array_size>(from_copy_ctor, expected_value);
  }
};

template <class T, std::size_t>
struct CheckSimdMaskDefaultMoveCtor {
  template <class SimdAbi>
  void operator()() {
    ex::simd_mask<T, SimdAbi> pure_mask;
    constexpr std::size_t array_size = ex::simd_size_v<T, SimdAbi>;
    std::array<bool, array_size> expected_value;
    for (size_t i = 0; i < array_size; ++i) {
      if (i % 2 == 0)
        pure_mask[i] = true;
      else
        pure_mask[i] = false;
      expected_value[i] = pure_mask[i];
    }

    ex::simd_mask<T, SimdAbi> from_move_ctor(std::move(pure_mask));
    assert_simd_mask_values_equal<array_size>(from_move_ctor, expected_value);
  }
};

template <class T, std::size_t>
struct CheckSimdMaskDefaultCopyAssignment {
  template <class SimdAbi>
  void operator()() {
    ex::simd_mask<T, SimdAbi> pure_mask;
    constexpr std::size_t array_size = ex::simd_size_v<T, SimdAbi>;
    std::array<bool, array_size> expected_value;
    for (size_t i = 0; i < array_size; ++i) {
      if (i % 2 == 0)
        pure_mask[i] = true;
      else
        pure_mask[i] = false;
      expected_value[i] = pure_mask[i];
    }

    ex::simd_mask<T, SimdAbi> from_copy_assignment;
    from_copy_assignment = pure_mask;
    assert_simd_mask_values_equal<array_size>(from_copy_assignment, expected_value);
  }
};

template <class T, std::size_t>
struct CheckSimdMaskDefaultMoveAssignment {
  template <class SimdAbi>
  void operator()() {
    ex::simd_mask<T, SimdAbi> pure_mask;
    constexpr std::size_t array_size = ex::simd_size_v<T, SimdAbi>;
    std::array<bool, array_size> expected_value;
    for (size_t i = 0; i < array_size; ++i) {
      if (i % 2 == 0)
        pure_mask[i] = true;
      else
        pure_mask[i] = false;
      expected_value[i] = pure_mask[i];
    }

    ex::simd_mask<T, SimdAbi> from_move_assignment;
    from_move_assignment = std::move(pure_mask);
    assert_simd_mask_values_equal<array_size>(from_move_assignment, expected_value);
  }
};

int main(int, char**) {
  test_all_simd_abi<CheckSimdMaskDefaultCtor>();
  test_all_simd_abi<CheckSimdMaskDefaultCopyCtor>();
  test_all_simd_abi<CheckSimdMaskDefaultMoveCtor>();
  test_all_simd_abi<CheckSimdMaskDefaultCopyAssignment>();
  test_all_simd_abi<CheckSimdMaskDefaultMoveAssignment>();
  return 0;
}
