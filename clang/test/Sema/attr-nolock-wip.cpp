// RUN: %clang_cc1 -fsyntax-only -fblocks -verify %s

#if !__has_attribute(clang_nolock)
#error "the 'nolock' attribute is not available"
#endif

// ============================================================================

void f2(int);
void f2(int) [[clang::nolock]]; // expected-note {{previous declaration is here}}
void f2(int); // expected-warning {{attribute 'nolock' on function does not match previous declaration}}

// ============================================================================

#if 0
// https://github.com/llvm/llvm-project/pull/84983#issuecomment-1994978033

// the bug where AttributedType sugar gets lost on lambdas (when the "inferred" return type gets
// converted to a concrete one) happens here and the nolock(false) attribute is lost from h.

template <class T>
void f(T a) [[clang::nolock]] { a(); }

void m()
{
	auto g = []() [[clang::nolock]] {
	};
	
	auto h = []() [[clang::nolock(false)]] {
	};

	f(g);
	f(h);
}
#endif

// ============================================================================

#if 0
// some messing around with type traits
template <class _Tp, _Tp __v>
struct integral_constant
{
  static constexpr const _Tp      value = __v;
  typedef _Tp               value_type;
  typedef integral_constant type;
  constexpr operator value_type() const noexcept {return value;}
  constexpr value_type operator ()() const noexcept {return value;}
};

template <class _Tp, _Tp __v>
const _Tp integral_constant<_Tp, __v>::value;

typedef integral_constant<bool, true>  true_type;
typedef integral_constant<bool, false> false_type;

template <typename T>
struct is_nolock;

void g() [[clang::nolock]];
void h() [[clang::nolock(false)]];
#endif

