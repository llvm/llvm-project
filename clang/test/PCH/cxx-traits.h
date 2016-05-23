// Header for PCH test cxx-traits.cpp

namespace n {

template<typename _Tp>
struct __is_pod { // expected-warning {{keyword '__is_pod' will be made available as an identifier for the remainder of the translation unit}}
  enum { __value };
};

template<typename _Tp>
struct __is_empty { // expected-warning {{keyword '__is_empty' will be made available as an identifier for the remainder of the translation unit}}
  enum { __value };
};

template<typename T, typename ...Args>
struct is_trivially_constructible {
  static const bool value = __is_trivially_constructible(T, Args...);
};

struct __is_abstract {};  // expected-warning {{made available}}
struct __is_arithmetic {};  // expected-warning {{made available}}
struct __is_array {};  // expected-warning {{made available}}
struct __is_assignable {};  // expected-warning {{made available}}
struct __is_base_of {};  // expected-warning {{made available}}
struct __is_class {};  // expected-warning {{made available}}
struct __is_complete_type {};  // expected-warning {{made available}}
struct __is_compound {};  // expected-warning {{made available}}
struct __is_const {};  // expected-warning {{made available}}
struct __is_constructible {};  // expected-warning {{made available}}
struct __is_convertible {};  // expected-warning {{made available}}
struct __is_convertible_to {};  // expected-warning {{made available}}
struct __is_destructible {};  // expected-warning {{made available}}
struct __is_enum {};  // expected-warning {{made available}}
struct __is_floating_point {};  // expected-warning {{made available}}
struct __is_final {};  // expected-warning {{made available}}
struct __is_function {};  // expected-warning {{made available}}
struct __is_fundamental {};  // expected-warning {{made available}}
struct __is_integral {};  // expected-warning {{made available}}
struct __is_interface_class {};  // expected-warning {{made available}}
struct __is_literal {};  // expected-warning {{made available}}
struct __is_lvalue_expr {};  // expected-warning {{made available}}
struct __is_lvalue_reference {};  // expected-warning {{made available}}
struct __is_member_function_pointer {};  // expected-warning {{made available}}
struct __is_member_object_pointer {};  // expected-warning {{made available}}
struct __is_member_pointer {};  // expected-warning {{made available}}
struct __is_nothrow_assignable {};  // expected-warning {{made available}}
struct __is_nothrow_constructible {};  // expected-warning {{made available}}
struct __is_nothrow_destructible {};  // expected-warning {{made available}}
struct __is_object {};  // expected-warning {{made available}}
struct __is_pointer {};  // expected-warning {{made available}}
struct __is_polymorphic {};  // expected-warning {{made available}}
struct __is_reference {};  // expected-warning {{made available}}
struct __is_rvalue_expr {};  // expected-warning {{made available}}
struct __is_rvalue_reference {};  // expected-warning {{made available}}
struct __is_same {};  // expected-warning {{made available}}
struct __is_scalar {};  // expected-warning {{made available}}
struct __is_sealed {};  // expected-warning {{made available}}
struct __is_signed {};  // expected-warning {{made available}}
struct __is_standard_layout {};  // expected-warning {{made available}}
struct __is_trivial {};  // expected-warning {{made available}}
struct __is_trivially_assignable {};  // expected-warning {{made available}}
struct __is_trivially_constructible {};  // expected-warning {{made available}}
struct __is_trivially_copyable {};  // expected-warning {{made available}}
struct __is_union {};  // expected-warning {{made available}}
struct __is_unsigned {};  // expected-warning {{made available}}
struct __is_void {};  // expected-warning {{made available}}
struct __is_volatile {};  // expected-warning {{made available}}


}
