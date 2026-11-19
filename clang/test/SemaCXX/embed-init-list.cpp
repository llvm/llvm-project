// RUN: %clang_cc1 -fsyntax-only -verify -Wno-c++26-extensions %s

namespace std {
typedef decltype(sizeof(int)) size_t;

template <class _E> class initializer_list {
  const _E *__begin_;
  size_t __size_;

  constexpr initializer_list(const _E *__b, size_t __s)
      : __begin_(__b), __size_(__s) {}

public:
  constexpr initializer_list() : __begin_(nullptr), __size_(0) {}
  constexpr size_t size() const { return __size_; }
};
} // namespace std

template <typename T> struct S {
  S(std::initializer_list<T>);
};

template <> struct S<char> {
  S(std::initializer_list<char>);
};

struct S1 {
  S<char> data;
  int a;
};

template <typename _Tp, std::size_t _Nm> void to_array(_Tp (&&__a)[_Nm]) {}


template<typename T>
void tfn(T) {}

void tests() {

  S<char>{{
#embed __FILE__
  }};

  S1 ss{std::initializer_list<char>{
#embed __FILE__
  }};

  S sss = {
#embed __FILE__
  };

  std::initializer_list<int> il{
#embed __FILE__
  };

  static constexpr auto initializer_list = std::initializer_list<char>{
#embed __FILE__
      , '\0'};

  static constexpr auto intinitializer_list = std::initializer_list<int>{
#embed __FILE__
      , '\0'};

  to_array({
#embed __FILE__
  });

  tfn<std::initializer_list<int>>({
#embed __FILE__
  });
}

namespace GH222796 {
struct X {
  template <typename... T>
  constexpr X(T... t) : sum((0 + ... + int(t))), size(sizeof...(T)) {}
  int sum;
  int size;
};

constexpr X copy_list = {
#embed __FILE__ limit(2)
};
static_assert(copy_list.size == 2, "");
static_assert(copy_list.sum == '/' + '/', "");

constexpr X direct_list{
#embed __FILE__ limit(3)
};
static_assert(direct_list.size == 3, "");
static_assert(direct_list.sum == '/' + '/' + ' ', "");

constexpr X functional_cast() {
  return X{
#embed __FILE__ limit(2)
  };
}
static_assert(functional_cast().size == 2, "");

constexpr X returned() {
  return {
#embed __FILE__ limit(2)
  };
}
static_assert(returned().size == 2, "");

constexpr int arg(X x) { return x.size; }
constexpr int arg(int) { return -1; }
static_assert(arg({
#embed __FILE__ limit(2)
}) == 2, "");

struct Member {
  X m{
#embed __FILE__ limit(2)
  };
};
constexpr Member member{};
static_assert(member.m.size == 2, "");

constexpr X empty{
#embed __FILE__ limit(0)
};
static_assert(empty.size == 0, "");

struct Two {
  Two(int, int);
};
void two() {
  Two t = {
#embed __FILE__ limit(2)
  };
  Two *p = new Two{
#embed __FILE__ limit(2)
  };
  delete p;
  (void)Two{
#embed __FILE__ limit(2)
  };
}

template <class... T> struct Deduced {
  Deduced(T...);
};
Deduced deduced{
#embed __FILE__ limit(2)
};
static_assert(__is_same(decltype(deduced), Deduced<int, int>), "");

template <class T, unsigned N> constexpr unsigned bound(const T (&)[N]) {
  return N;
}
static_assert(bound({
#embed __FILE__ limit(3)
}) == 3, "");

template <class T> constexpr std::size_t il_size(std::initializer_list<T> l) {
  return l.size();
}
static_assert(il_size({
#embed __FILE__ limit(3)
}) == 3, "");

constexpr std::initializer_list<int> il = {
#embed __FILE__ limit(3)
};
static_assert(il.size() == 3, "");

constexpr const int &one_ref = {
#embed __FILE__ limit(1)
};
static_assert(one_ref == '/', "");

const int &ref = { // expected-error {{excess elements in scalar initializer}}
#embed __FILE__ limit(2)
};

enum E : int { E1 };
constexpr E one_enum{
#embed __FILE__ limit(1)
};
static_assert(one_enum == '/', "");

const E e{ // expected-error {{excess elements in scalar initializer}}
#embed __FILE__ limit(2)
};

int scalar = { // expected-error {{excess elements in scalar initializer}}
#embed __FILE__ limit(2)
};

constexpr _Complex double complex = {
#embed __FILE__ limit(2)
};
static_assert(__real__ complex == '/' && __imag__ complex == '/', "");

auto deduced_auto{ // expected-error {{excess elements in scalar initializer}}
#embed __FILE__ limit(2)
};

void deduced_auto_expr() {
  (void)new auto{ // expected-error {{excess elements in scalar initializer}}
#embed __FILE__ limit(2)
  };
  (void)auto{ // expected-warning {{'auto' as a functional-style cast is a C++23 extension}} \
                 expected-error {{excess elements in scalar initializer}}
#embed __FILE__ limit(2)
  };
}

void scalar_arg(int); // expected-note {{candidate function not viable: cannot convert initializer list argument to 'int'}}
void call_scalar_arg() {
  scalar_arg({ // expected-error {{no matching function for call to 'scalar_arg'}}
#embed __FILE__ limit(2)
  });
}

_BitInt(7) fits[] = {
#embed __FILE__ limit(3)
};
_BitInt(7) narrowed[] = { // expected-error@+1 {{constant expression evaluates to 82 which cannot be narrowed to type '_BitInt(7)'}}
#embed __FILE__ limit(4)
};
std::initializer_list<_BitInt(7)> narrowed_il = { // expected-error@+1 {{constant expression evaluates to 82 which cannot be narrowed to type '_BitInt(7)'}}
#embed __FILE__ limit(4)
};
}
