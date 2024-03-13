// RUN: %clang_cc1 -fsyntax-only -fblocks -verify %s
// R UN: %clang_cc1 -fsyntax-only -fblocks -verify -x c -std=c23 %s

#if !__has_attribute(clang_nolock)
#error "the 'nolock' attribute is not available"
#endif

// ============================================================================

#if 0 // C function type problems
void unannotated();
void nolock() [[clang::nolock]];
void noalloc() [[clang::noalloc]];


void callthis(void (*fp)());


void type_conversions()
{
// 	callthis(nolock);

	// It's fine to remove a performance constraint.
	void (*fp_plain)();

// 	fp_plain = unannotated;
	fp_plain = nolock;
// 	fp_plain = noalloc;
}
#endif

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
