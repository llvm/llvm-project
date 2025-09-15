// RUN: %check_clang_tidy -std=c++11 %s cppcoreguidelines-rvalue-reference-param-not-moved %t -- \
// RUN: -config="{CheckOptions: {cppcoreguidelines-rvalue-reference-param-not-moved.AllowPartialMove: true, cppcoreguidelines-rvalue-reference-param-not-moved.IgnoreUnnamedParams: true, cppcoreguidelines-rvalue-reference-param-not-moved.IgnoreNonDeducedTemplateTypes: true, cppcoreguidelines-rvalue-reference-param-not-moved.MoveFunction: custom_move}}" -- -fno-delayed-template-parsing

// NOLINTBEGIN
namespace std {
template <typename>
struct remove_reference;

template <typename _Tp> struct remove_reference { typedef _Tp type; };
template <typename _Tp> struct remove_reference<_Tp&> { typedef _Tp type; };
template <typename _Tp> struct remove_reference<_Tp&&> { typedef _Tp type; };

template <typename _Tp>
constexpr typename std::remove_reference<_Tp>::type &&move(_Tp &&__t) noexcept;

template <typename _Tp>
constexpr _Tp &&
forward(typename remove_reference<_Tp>::type &__t) noexcept;

}
// NOLINTEND


struct Obj {
  Obj();
  Obj(const Obj&);
  Obj& operator=(const Obj&);
  Obj(Obj&&);
  Obj& operator=(Obj&&);
  void member() const;
};

template<class T>
constexpr typename std::remove_reference<T>::type&& custom_move(T&& x) noexcept
{
    return static_cast<typename std::remove_reference<T>::type&&>(x);
}

void move_with_std(Obj&& o) {
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: rvalue reference parameter 'o' is never moved from inside the function body [cppcoreguidelines-rvalue-reference-param-not-moved]
  Obj other{std::move(o)};
}

void move_with_custom(Obj&& o) {
  Obj other{custom_move(o)};
}

