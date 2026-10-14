// RUN: %clang_cc1 -std=c++20 -ffreestanding -fexperimental-new-constant-interpreter -triple x86_64-unknown-unknown -target-feature +avx512fp16 -verify %s

#include <immintrin.h>
#include "../CodeGen/X86/builtin_test_helpers.h"

constexpr int ROUND_CUR_DIRECTION = 4;
constexpr int ROUND_NO_EXC = 8;
constexpr int ROUND_CUR_DIRECTION_NO_EXC = 12;

namespace Test_mm_min_sh_round_mask_invalid_rounding {
constexpr __m128h a = (__m128h)(__v8hf){10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
constexpr __m128h b = (__m128h)(__v8hf){100.0f, 200.0f, 300.0f, 400.0f, 500.0f, 600.0f, 700.0f, 800.0f};
constexpr __m128h src = (__m128h)(__v8hf){1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
constexpr __m128h result = (__m128h)__builtin_ia32_minsh_round_mask((__v8hf)a, (__v8hf)b, (__v8hf)src, 1, ROUND_NO_EXC); // expected-error {{must be initialized by a constant expression}}
}

namespace Test_mm_max_sh_round_mask_invalid_rounding_8 {
constexpr __m128h a = (__m128h)(__v8hf){10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
constexpr __m128h b = (__m128h)(__v8hf){100.0f, 200.0f, 300.0f, 400.0f, 500.0f, 600.0f, 700.0f, 800.0f};
constexpr __m128h src = (__m128h)(__v8hf){1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
constexpr __m128h result = (__m128h)__builtin_ia32_maxsh_round_mask((__v8hf)a, (__v8hf)b, (__v8hf)src, 1, ROUND_NO_EXC); // expected-error {{must be initialized by a constant expression}}
}

namespace Test_mm_max_sh_round_mask_invalid_rounding_12 {
constexpr __m128h a = (__m128h)(__v8hf){10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
constexpr __m128h b = (__m128h)(__v8hf){100.0f, 200.0f, 300.0f, 400.0f, 500.0f, 600.0f, 700.0f, 800.0f};
constexpr __m128h src = (__m128h)(__v8hf){1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
constexpr __m128h result = (__m128h)__builtin_ia32_maxsh_round_mask((__v8hf)a, (__v8hf)b, (__v8hf)src, 1, ROUND_CUR_DIRECTION_NO_EXC); // expected-error {{must be initialized by a constant expression}}
}

namespace Test_mm_min_sh_round_mask_valid_rounding {
constexpr __m128h a = (__m128h)(__v8hf){10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
constexpr __m128h b = (__m128h)(__v8hf){100.0f, 200.0f, 300.0f, 400.0f, 500.0f, 600.0f, 700.0f, 800.0f};
constexpr __m128h src = (__m128h)(__v8hf){1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
constexpr __m128h result = (__m128h)__builtin_ia32_minsh_round_mask((__v8hf)a, (__v8hf)b, (__v8hf)src, 1, ROUND_CUR_DIRECTION);
static_assert(match_m128h(result, 10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f));
}

namespace Test_mm_max_sh_round_mask_valid_rounding {
constexpr __m128h a = (__m128h)(__v8hf){10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
constexpr __m128h b = (__m128h)(__v8hf){100.0f, 200.0f, 300.0f, 400.0f, 500.0f, 600.0f, 700.0f, 800.0f};
constexpr __m128h src = (__m128h)(__v8hf){1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
constexpr __m128h result = (__m128h)__builtin_ia32_maxsh_round_mask((__v8hf)a, (__v8hf)b, (__v8hf)src, 1, ROUND_CUR_DIRECTION);
static_assert(match_m128h(result, 100.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f));
}

namespace Test_mm_min_sh_round_mask_nan {
constexpr __m128h a = (__m128h)(__v8hf){__builtin_nanf16(""), 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
constexpr __m128h b = (__m128h)(__v8hf){100.0f, 200.0f, 300.0f, 400.0f, 500.0f, 600.0f, 700.0f, 800.0f};
constexpr __m128h src = (__m128h)(__v8hf){1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
constexpr __m128h result = (__m128h)__builtin_ia32_minsh_round_mask((__v8hf)a, (__v8hf)b, (__v8hf)src, 1, ROUND_CUR_DIRECTION); // expected-error {{must be initialized by a constant expression}}
}

namespace Test_mm_min_sh_round_mask_pos_infinity {
constexpr __m128h a = (__m128h)(__v8hf){__builtin_inff16(), 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
constexpr __m128h b = (__m128h)(__v8hf){100.0f, 200.0f, 300.0f, 400.0f, 500.0f, 600.0f, 700.0f, 800.0f};
constexpr __m128h src = (__m128h)(__v8hf){1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
constexpr __m128h result = (__m128h)__builtin_ia32_minsh_round_mask((__v8hf)a, (__v8hf)b, (__v8hf)src, 1, ROUND_CUR_DIRECTION); // expected-error {{must be initialized by a constant expression}}
}

namespace Test_mm_min_sh_round_mask_neg_infinity {
constexpr __m128h a = (__m128h)(__v8hf){-__builtin_inff16(), 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
constexpr __m128h b = (__m128h)(__v8hf){100.0f, 200.0f, 300.0f, 400.0f, 500.0f, 600.0f, 700.0f, 800.0f};
constexpr __m128h src = (__m128h)(__v8hf){1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
constexpr __m128h result = (__m128h)__builtin_ia32_minsh_round_mask((__v8hf)a, (__v8hf)b, (__v8hf)src, 1, ROUND_CUR_DIRECTION); // expected-error {{must be initialized by a constant expression}}
}

namespace Test_mm_min_sh_round_mask_denormal {
constexpr _Float16 denormal = 0x1.0p-15f16;
constexpr __m128h a = (__m128h)(__v8hf){denormal, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
constexpr __m128h b = (__m128h)(__v8hf){100.0f, 200.0f, 300.0f, 400.0f, 500.0f, 600.0f, 700.0f, 800.0f};
constexpr __m128h src = (__m128h)(__v8hf){1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
constexpr __m128h result = (__m128h)__builtin_ia32_minsh_round_mask((__v8hf)a, (__v8hf)b, (__v8hf)src, 1, ROUND_CUR_DIRECTION); // expected-error {{must be initialized by a constant expression}}
}

namespace Test_mm_min_sh_valid {
constexpr __m128h a = (__m128h)(__v8hf){10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
constexpr __m128h b = (__m128h)(__v8hf){100.0f, 200.0f, 300.0f, 400.0f, 500.0f, 600.0f, 700.0f, 800.0f};
constexpr __m128h result = _mm_min_sh(a, b);
static_assert(match_m128h(result, 10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f));
}

namespace Test_mm_max_sh_valid {
constexpr __m128h a = (__m128h)(__v8hf){10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
constexpr __m128h b = (__m128h)(__v8hf){100.0f, 200.0f, 300.0f, 400.0f, 500.0f, 600.0f, 700.0f, 800.0f};
constexpr __m128h result = _mm_max_sh(a, b);
static_assert(match_m128h(result, 100.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f));
}

namespace Test_mm_min_sh_nan {
constexpr __m128h a = (__m128h)(__v8hf){__builtin_nanf16(""), 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
constexpr __m128h b = (__m128h)(__v8hf){100.0f, 200.0f, 300.0f, 400.0f, 500.0f, 600.0f, 700.0f, 800.0f};
constexpr __m128h result = _mm_min_sh(a, b); // expected-error {{must be initialized by a constant expression}}
}

namespace Test_mm_min_sh_pos_infinity {
constexpr __m128h a = (__m128h)(__v8hf){__builtin_inff16(), 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
constexpr __m128h b = (__m128h)(__v8hf){100.0f, 200.0f, 300.0f, 400.0f, 500.0f, 600.0f, 700.0f, 800.0f};
constexpr __m128h result = _mm_min_sh(a, b); // expected-error {{must be initialized by a constant expression}}
}

namespace Test_mm_min_sh_neg_infinity {
constexpr __m128h a = (__m128h)(__v8hf){-__builtin_inff16(), 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
constexpr __m128h b = (__m128h)(__v8hf){100.0f, 200.0f, 300.0f, 400.0f, 500.0f, 600.0f, 700.0f, 800.0f};
constexpr __m128h result = _mm_min_sh(a, b); // expected-error {{must be initialized by a constant expression}}
}

namespace Test_mm_min_sh_denormal {
constexpr _Float16 denormal = 0x1.0p-15f16;
constexpr __m128h a = (__m128h)(__v8hf){denormal, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
constexpr __m128h b = (__m128h)(__v8hf){100.0f, 200.0f, 300.0f, 400.0f, 500.0f, 600.0f, 700.0f, 800.0f};
constexpr __m128h result = _mm_min_sh(a, b); // expected-error {{must be initialized by a constant expression}}
}
