// RUN: %check_clang_tidy %s performance-noexcept-move-constructor %t -- -- -fexceptions

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
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:3: warning: move constructors should be marked noexcept
  // CHECK-MESSAGES-CONFIG-NOT: :[[@LINE-2]]:3: warning: move constructors should be marked noexcept
  ThrowOnAnything& operator=(ThrowOnAnything &&) noexcept(false);
  ~ThrowOnAnything() noexcept(false);
};

struct C_1 {
    static constexpr bool kFalse = false;
    C_1(C_1&&) noexcept(kFalse) = default;
    // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: noexcept specifier on the move constructor evaluates to 'false' [performance-noexcept-move-constructor]
    // CHECK-MESSAGES-CONFIG-NOT: :[[@LINE-2]]:25: warning: noexcept specifier on the move constructor evaluates to 'false' [performance-noexcept-move-constructor]

    C_1 &operator=(C_1 &&) noexcept(kFalse);
    // CHECK-MESSAGES: :[[@LINE-1]]:37: warning: noexcept specifier on the move assignment operator evaluates to 'false' [performance-noexcept-move-constructor]
    // CHECK-MESSAGES-CONFIG-NOT: :[[@LINE-2]]:37: warning: noexcept specifier on the move assignment operator evaluates to 'false' [performance-noexcept-move-constructor]
};

struct C_2 {
    static constexpr bool kEval = std::is_nothrow_move_constructible<ThrowOnAnything>::value;
    static_assert(!kEval); // kEval == false;

    C_2(C_2&&) noexcept(kEval) = default;
    // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: noexcept specifier on the move constructor evaluates to 'false' [performance-noexcept-move-constructor]
    // CHECK-MESSAGES-CONFIG-NOT: :[[@LINE-2]]:25: warning: noexcept specifier on the move constructor evaluates to 'false' [performance-noexcept-move-constructor]

    C_2 &operator=(C_2 &&) noexcept(kEval);
    // CHECK-MESSAGES: :[[@LINE-1]]:37: warning: noexcept specifier on the move assignment operator evaluates to 'false' [performance-noexcept-move-constructor]
    // CHECK-MESSAGES-CONFIG-NOT: :[[@LINE-2]]:37: warning: noexcept specifier on the move assignment operator evaluates to 'false' [performance-noexcept-move-constructor]

    ThrowOnAnything field;
};

struct C_3 {
    static constexpr bool kEval = std::is_nothrow_move_constructible<ThrowOnAnything>::value;
    static_assert(!kEval); // kEval == false;

    C_3(C_3&&) noexcept(kEval) = default;
    // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: noexcept specifier on the move constructor evaluates to 'false' [performance-noexcept-move-constructor]
    // CHECK-MESSAGES-CONFIG-NOT: :[[@LINE-2]]:25: warning: noexcept specifier on the move constructor evaluates to 'false' [performance-noexcept-move-constructor]

    ~C_3() noexcept(kEval) = default;
    // CHECK-MESSAGES-CONFIG: :[[@LINE-1]]:21: warning: noexcept specifier on the destructor evaluates to 'false'

    ThrowOnAnything field;
};
