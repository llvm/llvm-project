#pragma clang system_header

namespace std {

template<class T, T v>
struct integral_constant {
    static constexpr T value = v;
    typedef T value_type;
    typedef integral_constant type;
    constexpr operator value_type() const noexcept { return value; }
};

template <bool B>
using bool_constant = integral_constant<bool, B>;
using true_type = bool_constant<true>;
using false_type = bool_constant<false>;

template<class T>
struct is_error_code_enum : false_type {};

template<class T>
void swap(T &a, T &b);
}

