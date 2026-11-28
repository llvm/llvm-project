// RUN: %clang_cc1 -fsyntax-only -verify %std_cxx11- %s
// RUN: %clang_cc1 -fsyntax-only -verify %std_cxx11- -fms-compatibility %s
// expected-no-diagnostics

// When a constructor is called as a user-defined conversion, copying its
// argument into its parameter must not look for further user-defined
// conversions if a constructor can take the argument without one. In each case
// below, looking for one checks a constraint that depends on the conversion
// being performed.

template <class T> T &&declval();

template <bool, class = void> struct enable_if {};
template <class T> struct enable_if<true, T> {
  typedef T type;
};
template <class, class> struct is_same {
  static const bool value = false;
};
template <class T> struct is_same<T, T> {
  static const bool value = true;
};

struct ConvertibleFromAny {
  template <class V> ConvertibleFromAny(V);
};

namespace converting_constructor {
void Overload(ConvertibleFromAny);

struct Variant {
  template <class T, class = decltype(Overload(declval<T>()))>
  Variant(T &&) = delete;
};

struct Matcher {
  Matcher() = default;
  Matcher(Variant);
  Matcher(const Matcher &) = default;
};

void test(Matcher m) {
  Matcher copied = m;
  Matcher moved = static_cast<Matcher &&>(m);
}
} // namespace converting_constructor

namespace reference_to_converted_temporary {
void ByConstReference(const ConvertibleFromAny &);
void ByRvalueReference(ConvertibleFromAny &&);

struct Variant {
  template <class T, class = decltype(ByConstReference(declval<T>())),
            class = decltype(ByRvalueReference(declval<T>()))>
  Variant(T &&) = delete;
};

struct Matcher {
  Matcher() = default;
  Matcher(Variant);
  Matcher(const Matcher &) = default;
};

void test(Matcher m) {
  Matcher copied = m;
  Matcher moved = static_cast<Matcher &&>(m);
}
} // namespace reference_to_converted_temporary

namespace argument_of_derived_class {
struct Matcher;
struct ConvertibleFromMatcher {
  ConvertibleFromMatcher(Matcher);
};
void Overload(ConvertibleFromMatcher);

struct Variant {
  template <class T, class = decltype(Overload(declval<T>()))>
  Variant(T &&) = delete;
};

struct Matcher {
  Matcher() = default;
  Matcher(Variant);
  Matcher(const Matcher &) = default;
};
struct DerivedMatcher : Matcher {};

void test(DerivedMatcher d) {
  Matcher copied = d;
  Matcher moved = static_cast<DerivedMatcher &&>(d);
}
} // namespace argument_of_derived_class

namespace base_class_parameter_of_selected_constructor {
void Overload(ConvertibleFromAny);

struct Variant {
  template <class T, class = decltype(Overload(declval<T>()))>
  Variant(T &&) = delete;
};

struct Base {
  Base() = default;
  Base(Variant);
  Base(const Base &) = default;
};
struct Derived : Base {
  Derived() = default;
  Derived(Derived &);
  Derived(Base);
};

void test(const Derived &d) { Overload(d); }
} // namespace base_class_parameter_of_selected_constructor

namespace conversion_function {
void Overload(ConvertibleFromAny);

struct Variant {};

struct Matcher {
  Matcher() = default;
  Matcher(Variant);
  Matcher(const Matcher &) = default;

  template <class T,
            class = decltype(Overload(declval<
                typename enable_if<is_same<T, Variant>::value, Matcher &>::type>()))>
  operator T();
};

void test(Matcher m) {
  Matcher copied = m;
  Matcher moved = static_cast<Matcher &&>(m);
}
} // namespace conversion_function

namespace conversion_function_to_base_class {
struct Base {
  Base() = default;
  Base(const Base &) = default;
};
struct ConvertibleFromBase {
  ConvertibleFromBase(Base);
};
void Overload(ConvertibleFromBase);

struct Derived : Base {
  template <class T,
            class = decltype(Overload(declval<
                typename enable_if<is_same<T, Base>::value, Derived &>::type>()))>
  operator T();
};

void test(Derived d) {
  Overload(d);
  Base copied = d;
}
} // namespace conversion_function_to_base_class

#if __cplusplus >= 202002L
namespace requires_clause {
void Overload(ConvertibleFromAny);

struct Variant {
  template <class T>
    requires requires(T &&t) { Overload(static_cast<T &&>(t)); }
  Variant(T &&) = delete;
};

struct Matcher {
  Matcher() = default;
  Matcher(Variant);
  Matcher(const Matcher &) = default;
};

void test(Matcher m) {
  Matcher copied = m;
  Matcher moved = static_cast<Matcher &&>(m);
}
} // namespace requires_clause
#endif
