// RUN: %check_clang_tidy -std=c++17-or-later %s bugprone-optional-value-conversion %t -- --fix-notes
// RUN: %check_clang_tidy -check-suffix=CUSTOM -std=c++17-or-later %s bugprone-optional-value-conversion %t -- \
// RUN: -config="{CheckOptions: {bugprone-optional-value-conversion.OptionalTypes: 'CustomOptional', \
// RUN:                          bugprone-optional-value-conversion.ValueMethods: '::Read$;::Ooo$'}}" --fix-notes

namespace std {
  template<typename T>
  struct optional
  {
    constexpr optional() noexcept;
    constexpr optional(T&&) noexcept;
    constexpr optional(const T&) noexcept;
    template<typename U>
    constexpr optional(U&&) noexcept;
    const T& operator*() const;
    T* operator->();
    const T* operator->() const;
    T& operator*();
    const T& value() const;
    T& value();
    const T& get() const;
    T& get();
    T value_or(T) const;
  };

  template <class T>
  T&& move(T &x) {
    return static_cast<T&&>(x);
  }
}

namespace boost {
  template<typename T>
  struct optional {
    constexpr optional() noexcept;
    constexpr optional(const T&) noexcept;
    const T& operator*() const;
    const T& get() const;
  };
}

namespace absl {
  template<typename T>
  struct optional {
    constexpr optional() noexcept;
    constexpr optional(const T&) noexcept;
    const T& operator*() const;
    const T& value() const;
  };
}

template<typename T>
struct CustomOptional {
  CustomOptional();
  CustomOptional(const T&);
  const T& Read() const;
  T& operator*();
  T& Ooo();
};

void takeOptionalValue(std::optional<int>);
void takeOptionalRef(const std::optional<int>&);
void takeOptionalRRef(std::optional<int>&&);
void takeOtherOptional(std::optional<long>);
void takeBOptionalValue(boost::optional<int>);
void takeAOptionalValue(absl::optional<int>);

void incorrect(std::optional<int> param)
{
  std::optional<int>* ptr = &param;
  takeOptionalValue(**ptr);
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: conversion from 'std::optional<int>' into 'int' and back into 'std::optional<int>', remove potentially error-prone optional dereference [bugprone-optional-value-conversion]
  // CHECK-FIXES: takeOptionalValue(*ptr);
  takeOptionalValue(*param);
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: conversion from 'std::optional<int>' into 'int' and back into 'std::optional<int>', remove potentially error-prone optional dereference [bugprone-optional-value-conversion]
  // CHECK-FIXES: takeOptionalValue(param);
  takeOptionalValue(param.value());
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: conversion from 'std::optional<int>' into 'int' and back into 'std::optional<int>', remove potentially error-prone optional dereference [bugprone-optional-value-conversion]
  // CHECK-FIXES: takeOptionalValue(param);
  takeOptionalValue(ptr->value());
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: conversion from 'std::optional<int>' into 'int' and back into 'std::optional<int>', remove potentially error-prone optional dereference [bugprone-optional-value-conversion]
  // CHECK-FIXES: takeOptionalValue(*ptr);
  takeOptionalValue(param.operator*());
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: conversion from 'std::optional<int>' into 'int' and back into 'std::optional<int>', remove potentially error-prone optional dereference [bugprone-optional-value-conversion]
  // CHECK-FIXES: takeOptionalValue(param);
  takeOptionalValue(ptr->operator*());
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: conversion from 'std::optional<int>' into 'int' and back into 'std::optional<int>', remove potentially error-prone optional dereference [bugprone-optional-value-conversion]
  // CHECK-FIXES: takeOptionalValue(*ptr);
  takeOptionalRef(*param);
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: conversion from 'std::optional<int>' into 'int' and back into 'std::optional<int>', remove potentially error-prone optional dereference [bugprone-optional-value-conversion]
  // CHECK-FIXES: takeOptionalRef(param);
  takeOptionalRef(param.value());
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: conversion from 'std::optional<int>' into 'int' and back into 'std::optional<int>', remove potentially error-prone optional dereference [bugprone-optional-value-conversion]
  // CHECK-FIXES: takeOptionalRef(param);
  takeOptionalRef(ptr->value());
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: conversion from 'std::optional<int>' into 'int' and back into 'std::optional<int>', remove potentially error-prone optional dereference [bugprone-optional-value-conversion]
  // CHECK-FIXES: takeOptionalRef(*ptr);
  takeOptionalRef(param.operator*());
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: conversion from 'std::optional<int>' into 'int' and back into 'std::optional<int>', remove potentially error-prone optional dereference [bugprone-optional-value-conversion]
  // CHECK-FIXES: takeOptionalRef(param);
  takeOptionalRef(ptr->operator*());
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: conversion from 'std::optional<int>' into 'int' and back into 'std::optional<int>', remove potentially error-prone optional dereference [bugprone-optional-value-conversion]
  // CHECK-FIXES: takeOptionalRef(*ptr);
  std::optional<int> p = *param;
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: conversion from 'std::optional<int>' into 'int' and back into 'std::optional<int>', remove potentially error-prone optional dereference [bugprone-optional-value-conversion]
  // CHECK-FIXES: std::optional<int> p = param;

  takeOptionalValue(std::move(**ptr));
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: conversion from 'std::optional<int>' into 'int' and back into 'std::optional<int>', remove potentially error-prone optional dereference [bugprone-optional-value-conversion]
  // CHECK-FIXES: takeOptionalValue(std::move(*ptr));
  takeOptionalValue(std::move(*param));
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: conversion from 'std::optional<int>' into 'int' and back into 'std::optional<int>', remove potentially error-prone optional dereference [bugprone-optional-value-conversion]
  // CHECK-FIXES: takeOptionalValue(std::move(param));
  takeOptionalValue(std::move(param.value()));
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: conversion from 'std::optional<int>' into 'int' and back into 'std::optional<int>', remove potentially error-prone optional dereference [bugprone-optional-value-conversion]
  // CHECK-FIXES: takeOptionalValue(std::move(param));
  takeOptionalValue(std::move(ptr->value()));
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: conversion from 'std::optional<int>' into 'int' and back into 'std::optional<int>', remove potentially error-prone optional dereference [bugprone-optional-value-conversion]
  // CHECK-FIXES: takeOptionalValue(std::move(*ptr));
  takeOptionalValue(std::move(param.operator*()));
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: conversion from 'std::optional<int>' into 'int' and back into 'std::optional<int>', remove potentially error-prone optional dereference [bugprone-optional-value-conversion]
  // CHECK-FIXES: takeOptionalValue(std::move(param));
  takeOptionalRef(std::move(*param));
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: conversion from 'std::optional<int>' into 'int' and back into 'std::optional<int>', remove potentially error-prone optional dereference [bugprone-optional-value-conversion]
  // CHECK-FIXES: takeOptionalRef(std::move(param));
  takeOptionalRef(std::move(param.value()));
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: conversion from 'std::optional<int>' into 'int' and back into 'std::optional<int>', remove potentially error-prone optional dereference [bugprone-optional-value-conversion]
  // CHECK-FIXES: takeOptionalRef(std::move(param));
  takeOptionalRef(std::move(ptr->value()));
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: conversion from 'std::optional<int>' into 'int' and back into 'std::optional<int>', remove potentially error-prone optional dereference [bugprone-optional-value-conversion]
  // CHECK-FIXES: takeOptionalRef(std::move(*ptr));
  takeOptionalRef(std::move(param.operator*()));
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: conversion from 'std::optional<int>' into 'int' and back into 'std::optional<int>', remove potentially error-prone optional dereference [bugprone-optional-value-conversion]
  // CHECK-FIXES: takeOptionalRef(std::move(param));
  takeOptionalRRef(std::move(*param));
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: conversion from 'std::optional<int>' into 'int' and back into 'std::optional<int>', remove potentially error-prone optional dereference [bugprone-optional-value-conversion]
  // CHECK-FIXES: takeOptionalRRef(std::move(param));
  takeOptionalRRef(std::move(param.value()));
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: conversion from 'std::optional<int>' into 'int' and back into 'std::optional<int>', remove potentially error-prone optional dereference [bugprone-optional-value-conversion]
  // CHECK-FIXES: takeOptionalRRef(std::move(param));
  takeOptionalRRef(std::move(ptr->value()));
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: conversion from 'std::optional<int>' into 'int' and back into 'std::optional<int>', remove potentially error-prone optional dereference [bugprone-optional-value-conversion]
  // CHECK-FIXES: takeOptionalRRef(std::move(*ptr));
  takeOptionalRRef(std::move(param.operator*()));
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: conversion from 'std::optional<int>' into 'int' and back into 'std::optional<int>', remove potentially error-prone optional dereference [bugprone-optional-value-conversion]
  // CHECK-FIXES: takeOptionalRRef(std::move(param));
  takeOptionalRRef(std::move(ptr->operator*()));
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: conversion from 'std::optional<int>' into 'int' and back into 'std::optional<int>', remove potentially error-prone optional dereference [bugprone-optional-value-conversion]
  // CHECK-FIXES: takeOptionalRRef(std::move(*ptr));
  std::optional<int> p2 = std::move(*param);
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: conversion from 'std::optional<int>' into 'int' and back into 'std::optional<int>', remove potentially error-prone optional dereference [bugprone-optional-value-conversion]
  // CHECK-FIXES: std::optional<int> p2 = std::move(param);

  std::optional<std::optional<int>> opt;
  takeOptionalValue(opt->value());
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: conversion from 'std::optional<int>' into 'int' and back into 'std::optional<int>', remove potentially error-prone optional dereference [bugprone-optional-value-conversion]
  // CHECK-FIXES: takeOptionalValue(*opt);
  takeOptionalValue(opt->operator*());
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: conversion from 'std::optional<int>' into 'int' and back into 'std::optional<int>', remove potentially error-prone optional dereference [bugprone-optional-value-conversion]
  // CHECK-FIXES: takeOptionalValue(*opt);

  boost::optional<int> bopt;
  takeBOptionalValue(*bopt);
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: conversion from 'boost::optional<int>' into 'int' and back into 'boost::optional<int>', remove potentially error-prone optional dereference [bugprone-optional-value-conversion]
  // CHECK-FIXES: takeBOptionalValue(bopt);
  takeBOptionalValue(bopt.get());
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: conversion from 'boost::optional<int>' into 'int' and back into 'boost::optional<int>', remove potentially error-prone optional dereference [bugprone-optional-value-conversion]
  // CHECK-FIXES: takeBOptionalValue(bopt);

  absl::optional<int> aopt;
  takeAOptionalValue(*aopt);
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: conversion from 'absl::optional<int>' into 'int' and back into 'absl::optional<int>', remove potentially error-prone optional dereference [bugprone-optional-value-conversion]
  // CHECK-FIXES: takeAOptionalValue(aopt);
  takeAOptionalValue(aopt.value());
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: conversion from 'absl::optional<int>' into 'int' and back into 'absl::optional<int>', remove potentially error-prone optional dereference [bugprone-optional-value-conversion]
  // CHECK-FIXES: takeAOptionalValue(aopt);
}

void takeCustom(const CustomOptional<int>&);

void testCustom(CustomOptional<int> param) {
  takeCustom(*param);
  // CHECK-MESSAGES-CUSTOM: :[[@LINE-1]]:14: warning: conversion from 'CustomOptional<int>' into 'int' and back into 'CustomOptional<int>', remove potentially error-prone optional dereference [bugprone-optional-value-conversion]
  // CHECK-FIXES-CUSTOM: takeCustom(param);
  takeCustom(param.Read());
  // CHECK-MESSAGES-CUSTOM: :[[@LINE-1]]:14: warning: conversion from 'CustomOptional<int>' into 'int' and back into 'CustomOptional<int>', remove potentially error-prone optional dereference [bugprone-optional-value-conversion]
  // CHECK-FIXES-CUSTOM: takeCustom(param);
  takeCustom(param.Ooo());
  // CHECK-MESSAGES-CUSTOM: :[[@LINE-1]]:14: warning: conversion from 'CustomOptional<int>' into 'int' and back into 'CustomOptional<int>', remove potentially error-prone optional dereference [bugprone-optional-value-conversion]
  // CHECK-FIXES-CUSTOM: takeCustom(param);
}

void correct(std::optional<int> param)
{
  takeOtherOptional(*param);
  takeOtherOptional(param.value());
  takeOtherOptional(param.value_or(5U));
  takeOtherOptional(param.operator*());

  std::optional<long> p = *param;
  takeOptionalValue(param.value_or(5U));
  takeOptionalRef(param.value_or(5U));

  std::optional<int>* ptr = &param;
  takeOtherOptional(**ptr);
  takeOtherOptional(ptr->value());
  takeOtherOptional(ptr->value_or(5U));
  takeOtherOptional(ptr->operator*());

  std::optional<long>* p2 = &p;
  takeOptionalValue(p2->value_or(5U));
  takeOptionalRef(p2->value_or(5U));
}
