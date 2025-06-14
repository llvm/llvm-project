// RUN: %check_clang_tidy -check-suffix=CONFIG %s performance-noexcept-move-constructor,performance-noexcept-destructor %t -- \
// RUN: -config="{CheckOptions: {performance-noexcept-move-constructor.AllowFalseEvaluated: true}}" \
// RUN: -- -fexceptions

namespace std
{
  template <typename T>
  struct is_nothrow_move_constructible
  {
    static constexpr bool value = __is_nothrow_constructible(T, __add_rvalue_reference(T));
  };
} // namespace std

struct ThrowOnAnything {
  ThrowOnAnything() noexcept(false);
  ThrowOnAnything(ThrowOnAnything&&) noexcept(false);
  ThrowOnAnything& operator=(ThrowOnAnything &&) noexcept(false);
  ~ThrowOnAnything() noexcept(false);
};

struct C_1 {
    static constexpr bool kFalse = false;
    C_1(C_1&&) noexcept(kFalse) = default;
    C_1 &operator=(C_1 &&) noexcept(kFalse);
};

struct C_2 {
    static constexpr bool kEval = std::is_nothrow_move_constructible<ThrowOnAnything>::value;
    static_assert(!kEval); // kEval == false;

    C_2(C_2&&) noexcept(kEval) = default;
    C_2 &operator=(C_2 &&) noexcept(kEval);

    ThrowOnAnything field;
};

struct C_3 {
    static constexpr bool kEval = std::is_nothrow_move_constructible<ThrowOnAnything>::value;
    static_assert(!kEval); // kEval == false;

    C_3(C_3&&) noexcept(kEval) = default;
    ~C_3() noexcept(kEval) = default;
    // CHECK-MESSAGES-CONFIG: :[[@LINE-1]]:21: warning: noexcept specifier on the destructor evaluates to 'false'

    ThrowOnAnything field;
};
