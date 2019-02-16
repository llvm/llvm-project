// RUN: %check_clang_tidy %s bugprone-forwarding-reference-overload %t -- -- -std=c++14

namespace std {
template <bool B, class T = void> struct enable_if { typedef T type; };

template <class T> struct enable_if<true, T> { typedef T type; };

template <bool B, class T = void>
using enable_if_t = typename enable_if<B, T>::type;

template <class T> struct enable_if_nice { typedef T type; };
} // namespace std

namespace foo {
template <class T> struct enable_if { typedef T type; };
} // namespace foo

template <typename T> constexpr bool just_true = true;

class Test1 {
public:
  template <typename T> Test1(T &&n);
  // CHECK-NOTES: [[@LINE-1]]:25: warning: constructor accepting a forwarding reference can hide the copy and move constructors [bugprone-forwarding-reference-overload]
  // CHECK-NOTES: 48:3: note: copy constructor declared here
  // CHECK-NOTES: 49:3: note: copy constructor declared here
  // CHECK-NOTES: 50:3: note: move constructor declared here

  template <typename T> Test1(T &&n, int i = 5, ...);
  // CHECK-NOTES: :[[@LINE-1]]:25: warning: constructor accepting a forwarding reference can hide the copy and move constructors
  // CHECK-NOTES: 48:3: note: copy constructor declared here
  // CHECK-NOTES: 49:3: note: copy constructor declared here
  // CHECK-NOTES: 50:3: note: move constructor declared here

  template <typename T, typename U = typename std::enable_if_nice<T>::type>
  Test1(T &&n);
  // CHECK-NOTES: :[[@LINE-1]]:3: warning: constructor accepting a forwarding reference can hide the copy and move constructors
  // CHECK-NOTES: 48:3: note: copy constructor declared here
  // CHECK-NOTES: 49:3: note: copy constructor declared here
  // CHECK-NOTES: 50:3: note: move constructor declared here

  template <typename T>
  Test1(T &&n, typename foo::enable_if<long>::type i = 5, ...);
  // CHECK-NOTES: :[[@LINE-1]]:3: warning: constructor accepting a forwarding reference can hide the copy and move constructors
  // CHECK-NOTES: 48:3: note: copy constructor declared here
  // CHECK-NOTES: 49:3: note: copy constructor declared here
  // CHECK-NOTES: 50:3: note: move constructor declared here

  Test1(const Test1 &other) {}
  Test1(Test1 &other) {}
  Test1(Test1 &&other) {}
};

template <typename U> class Test2 {
public:
  // Two parameters without default value, can't act as copy / move constructor.
  template <typename T, class V> Test2(T &&n, V &&m, int i = 5, ...);

  // Guarded with enable_if.
  template <typename T>
  Test2(T &&n, int i = 5,
        std::enable_if_t<sizeof(int) < sizeof(long), int> a = 5, ...);

  // Guarded with enable_if.
  template <typename T, typename X = typename std::enable_if<
                            sizeof(int) < sizeof(long), double>::type &>
  Test2(T &&n);

  // Guarded with enable_if.
  template <typename T>
  Test2(T &&n, typename std::enable_if<just_true<T>>::type **a = nullptr);

  // Guarded with enable_if.
  template <typename T, typename X = std::enable_if_t<just_true<T>> *&&>
  Test2(T &&n, double d = 0.0);

  // Not a forwarding reference parameter.
  template <typename T> Test2(const T &&n);

  // Not a forwarding reference parameter.
  Test2(int &&x);

  // Two parameters without default value, can't act as copy / move constructor.
  template <typename T> Test2(T &&n, int x);

  // Not a forwarding reference parameter.
  template <typename T> Test2(U &&n);
};

// The copy and move constructors are both disabled.
class Test3 {
public:
  template <typename T> Test3(T &&n);

  template <typename T> Test3(T &&n, int I = 5, ...);

  Test3(const Test3 &rhs) = delete;

private:
  Test3(Test3 &&rhs);
};

// Both the copy and the (compiler generated) move constructors can be hidden.
class Test4 {
public:
  template <typename T> Test4(T &&n);
  // CHECK-NOTES: :[[@LINE-1]]:25: warning: constructor accepting a forwarding reference can hide the copy and move constructors

  Test4(const Test4 &rhs);
  // CHECK-NOTES: :[[@LINE-1]]:3: note: copy constructor declared here
};

// Nothing can be hidden, the copy constructor is implicitly deleted.
class Test5 {
public:
  template <typename T> Test5(T &&n);

  Test5(Test5 &&rhs) = delete;
};

// Only the move constructor can be hidden.
class Test6 {
public:
  template <typename T> Test6(T &&n);
  // CHECK-NOTES: :[[@LINE-1]]:25: warning: constructor accepting a forwarding reference can hide the move constructor

  Test6(Test6 &&rhs);
  // CHECK-NOTES: :[[@LINE-1]]:3: note: move constructor declared here
private:
  Test6(const Test6 &rhs);
};

// Do not dereference a null BaseType.
template <class _Callable> class result_of;
template <class _Fp, class ..._Args> class result_of<_Fp(_Args...)> { };
template <class _Tp> using result_of_t = typename result_of<_Tp>::type;

template <class... _Types> struct __overload;
template <class _Tp, class... _Types>
struct __overload<_Tp, _Types...> : __overload<_Types...> {
  using __overload<_Types...>::operator();
};

template <class _Tp, class... _Types>
using __best_match_t = typename result_of_t<__overload<_Types...>(_Tp&&)>::type;

template <class... _Types>
class variant {
public:
  template <class _Arg, class _Tp = __best_match_t<_Arg, _Types...> >
  constexpr variant(_Arg&& __arg) {}
  // CHECK-NOTES: :[[@LINE-1]]:13: warning: constructor accepting a forwarding reference can hide the copy and move constructors
};
