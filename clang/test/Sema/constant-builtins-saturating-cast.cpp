// RUN: %clang_cc1 -std=c++20 -fsyntax-only %s
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -fexperimental-new-constant-interpreter %s

typedef int int4 __attribute__((ext_vector_type(4)));
typedef signed char char4 __attribute__((ext_vector_type(4)));
typedef bool bool4 __attribute__((ext_vector_type(4)));

// signed_to_narrower_signed
static_assert(__builtin_elementwise_saturating_cast(-200, signed char) == -128);
// signed_to_wider_signed
static_assert(__builtin_elementwise_saturating_cast(static_cast<short>(-200),
                                                int) == -200);
// signed_to_equal_signed
static_assert(__builtin_elementwise_saturating_cast(-200, int) == -200);
// unsigned_to_narrower_unsigned
static_assert(__builtin_elementwise_saturating_cast(300u, unsigned short) == 300);
// unsigned_to_wider_unsigned
static_assert(__builtin_elementwise_saturating_cast(static_cast<unsigned short>(300),
                                                unsigned int) == 300u);
// unsigned_to_equal_unsigned
static_assert(__builtin_elementwise_saturating_cast(300u, unsigned int) == 300u);
// signed_to_narrower_unsigned
static_assert(__builtin_elementwise_saturating_cast(-1, unsigned short) == 0);
// signed_to_wider_unsigned
static_assert(__builtin_elementwise_saturating_cast(static_cast<short>(-1),
                                                unsigned int) == 0);
// signed_to_equal_unsigned
static_assert(__builtin_elementwise_saturating_cast(-1, unsigned) == 0);
// unsigned_to_narrower_signed
static_assert(__builtin_elementwise_saturating_cast(300u, signed char) == 127);
// unsigned_to_wider_signed
static_assert(__builtin_elementwise_saturating_cast(static_cast<unsigned short>(300),
                                                int) == 300);
// unsigned_to_equal_signed
static_assert(__builtin_elementwise_saturating_cast(300u, int) == 300);
static_assert(__builtin_elementwise_saturating_cast(true, int) == 1);
static_assert(__builtin_elementwise_saturating_cast(2, bool) == true);

static_assert(__builtin_elementwise_saturating_cast(-1, unsigned char) == 0);
static_assert(__builtin_elementwise_saturating_cast(300, unsigned char) == 255);

constexpr char4 from_scalar_type =
    __builtin_elementwise_saturating_cast((int4){-200, -1, 0, 300}, signed char);
constexpr char4 from_vector_type =
    __builtin_elementwise_saturating_cast((int4){-200, -1, 0, 300}, char4);

static_assert(from_scalar_type[0] == -128 && from_scalar_type[1] == -1 &&
              from_scalar_type[2] == 0 && from_scalar_type[3] == 127);
static_assert(from_vector_type[0] == -128 && from_vector_type[1] == -1 &&
              from_vector_type[2] == 0 && from_vector_type[3] == 127);

constexpr bool4 from_bool_vector = __builtin_elementwise_saturating_cast(
    (int4){0, -1, 2, 0}, bool4);
static_assert(!from_bool_vector[0] && !from_bool_vector[1] &&
              from_bool_vector[2] && !from_bool_vector[3]);
