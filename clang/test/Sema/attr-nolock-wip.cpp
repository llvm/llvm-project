// RUN: %clang_cc1 -fsyntax-only -fblocks -verify %s

#if !__has_attribute(clang_nolock)
#error "the 'nolock' attribute is not available"
#endif


// The diagnostic for inference not following a non-inline method should override
// the one for a virtual method.

struct HasVirtual {
	virtual ~HasVirtual() = default;
	virtual void method();
};

void nl999(HasVirtual& x) [[clang::nolock]] {
	x.method();
}



#if 0
	using nl_sugar = int (*)(int) [[clang::nolock]];

	void receives_fp_nl(nl_sugar fp) {
	}
	
	int callback(int) noexcept [[clang::nolock]];

void type_conversions_2()
{
	auto receives_fp = [](void (*fp)()) {
	};
	
	//auto receives_fp_nl = [](void (*fp)() [[clang::nolock]]) {
	//};

	auto ne = +[]() noexcept {};
	auto nl = +[]() [[clang::nolock]] {};
	//auto nl_ne = +[](int x) noexcept [[clang::nolock]] -> int  { return x; };
	
	receives_fp(ne);
	receives_fp(nl);
// 	receives_fp(nl_ne);
	
	receives_fp_nl(callback);
}
#endif

#if 0
struct S {
	void foo();
	// void foo() noexcept; // error, redeclaration
	// void foo() [[clang::nolock]]; // error, redeclaration
	
	using FP = void (*)();
	using FPNE = void (*)() noexcept;
	using FPNL = void (*)() [[clang::nolock]];
	
	void bar(FP x);
	void bar(FPNE x);	// This is a distinct overload
	void bar(FPNL x);	// This is a distinct overload
};
#endif

// ============================================================================

#if 0
#define RT_UNSAFE_BEGIN(reason)                                   \
	_Pragma("clang diagnostic push")                                 \
	_Pragma("clang diagnostic ignored \"-Wunknown-warning-option\"") \
	_Pragma("clang diagnostic ignored \"-Wfunction-effects\"")

#define RT_UNSAFE_END \
	_Pragma("clang diagnostic pop")

#define RT_UNSAFE(...)  \
	RT_UNSAFE_BEGIN("") \
	__VA_ARGS__            \
	RT_UNSAFE_END
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

