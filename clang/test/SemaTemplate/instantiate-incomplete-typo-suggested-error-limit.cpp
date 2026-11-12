// RUN: not %clang -fsyntax-only -std=c++11 -ferror-limit=1 %s 2>&1 | FileCheck %s

// Test case for PR35682.
// The issue be caused by the typo correction that changes String to the
// incomplete type string. The example is based on the std::pair code and
// reduced to a minimal test case. When using std::pair the issue can only be
// reproduced when using the -stdlib=libc++ compiler option.

template <class T> class allocator;

template <class charT> struct char_traits;

template <class CharT, class Traits = char_traits<CharT>,
          class Allocator = allocator<CharT>>
class basic_string;
typedef basic_string<char, char_traits<char>, allocator<char>> string;

template <bool, class Tp = void> struct enable_if {};
template <class Tp> struct enable_if<true, Tp> { typedef Tp type; };

template <class Tp, Tp v> struct integral_constant {
  static constexpr const Tp value = v;
  typedef Tp value_type;
  typedef integral_constant type;

  constexpr operator value_type() const noexcept { return value; }
  constexpr value_type operator()() const noexcept { return value; }
};

template <class Tp, Tp v> constexpr const Tp integral_constant<Tp, v>::value;

using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;

template <class Tp, class Up> struct is_same : public false_type {};
template <class Tp> struct is_same<Tp, Tp> : public true_type {};

template <class T> struct single {
  typedef T first_type;

  T first;

  struct CheckArgs {
    template <class U1> static constexpr bool enable_implicit() {
      return is_same<first_type, U1>::value;
    }
  };

  template <class U1,
            typename enable_if<CheckArgs::template enable_implicit<U1>(),
                               bool>::type = false>
  single(U1 &&u1);
};

using SetKeyType = String;
single<SetKeyType> v;

// CHECK: error: unknown type name 'String'; did you mean 'string'?
// CHECK: fatal error: too many errors emitted, stopping now [-ferror-limit=]
// CHECK-NOT: Assertion{{.*}}failed
