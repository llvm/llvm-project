// RUN: %clang_cc1 -fsyntax-only -fblocks -fcxx-exceptions -std=c++20 -verify -Wfunction-effects %s
// These are in a separate file because errors (e.g. incompatible attributes) currently prevent
// the FXAnalysis pass from running at all.

// This diagnostic is re-enabled and exercised in isolation later in this file.
#pragma clang diagnostic ignored "-Wperf-constraint-implies-noexcept"

// --- CONSTRAINTS ---

void nb1() [[clang::nonblocking]]
{
	int *pInt = new int; // expected-warning {{function with 'nonblocking' attribute must not allocate or deallocate memory}}
	delete pInt; // expected-warning {{function with 'nonblocking' attribute must not allocate or deallocate memory}}
}

void nb2() [[clang::nonblocking]]
{
	static int global; // expected-warning {{function with 'nonblocking' attribute must not have static local variables}}
}

void nb3() [[clang::nonblocking]]
{
	try {
		throw 42; // expected-warning {{function with 'nonblocking' attribute must not throw or catch exceptions}}
	}
	catch (...) { // expected-warning {{function with 'nonblocking' attribute must not throw or catch exceptions}}
	}
}

void nb4_inline() {}
void nb4_not_inline(); // expected-note {{declaration cannot be inferred 'nonblocking' because it has no definition in this translation unit}}

void nb4() [[clang::nonblocking]]
{
	nb4_inline(); // OK
	nb4_not_inline(); // expected-warning {{function with 'nonblocking' attribute must not call non-'nonblocking' function}}
}


struct HasVirtual {
	virtual void unsafe(); // expected-note {{virtual method cannot be inferred 'nonblocking'}}
};

void nb5() [[clang::nonblocking]]
{
 	HasVirtual hv;
 	hv.unsafe(); // expected-warning {{function with 'nonblocking' attribute must not call non-'nonblocking' function}}
}

void nb6_unsafe(); // expected-note {{declaration cannot be inferred 'nonblocking' because it has no definition in this translation unit}}
void nb6_transitively_unsafe()
{
	nb6_unsafe(); // expected-note {{function cannot be inferred 'nonblocking' because it calls non-'nonblocking' function}}
}

void nb6() [[clang::nonblocking]]
{
	nb6_transitively_unsafe(); // expected-warning {{function with 'nonblocking' attribute must not call non-'nonblocking' function}}
}

thread_local int tl_var{ 42 };

bool tl_test() [[clang::nonblocking]]
{
	return tl_var > 0; // expected-warning {{function with 'nonblocking' attribute must not use thread-local variables}}
}

void nb7()
{
	// Make sure we verify blocks
	auto blk = ^() [[clang::nonblocking]] {
		throw 42; // expected-warning {{block with 'nonblocking' attribute must not throw or catch exceptions}}
	};
}

void nb8()
{
	// Make sure we verify lambdas
	auto lambda = []() [[clang::nonblocking]] {
		throw 42; // expected-warning {{lambda with 'nonblocking' attribute must not throw or catch exceptions}}
	};
}

void nb8a() [[clang::nonblocking]]
{
	// A blocking lambda shouldn't make the outer function unsafe.
	auto unsafeLambda = []() {
		throw 42;
	};
}

void nb8b() [[clang::nonblocking]]
{
	// An unsafe lambda capture makes the outer function unsafe.
	auto unsafeCapture = [foo = new int]() { // expected-warning {{function with 'nonblocking' attribute must not allocate or deallocate memory}}
		delete foo;
	};
}

void nb8c()
{
	// An unsafe lambda capture does not make the lambda unsafe.
	auto unsafeCapture = [foo = new int]() [[clang::nonblocking]] {
	};
}

// Make sure template expansions are found and verified.
	template <typename T>
	struct Adder {
		static T add_explicit(T x, T y) [[clang::nonblocking]]
		{
			return x + y; // expected-warning {{function with 'nonblocking' attribute must not call non-'nonblocking' function}}
		}
		static T add_implicit(T x, T y)
		{
			return x + y; // expected-note {{function cannot be inferred 'nonblocking' because it calls non-'nonblocking' function}}
		}
	};

	struct Stringy {
		friend Stringy operator+(const Stringy& x, const Stringy& y)
		{
			// Do something inferably unsafe
			auto* z = new char[42]; // expected-note {{function cannot be inferred 'nonblocking' because it allocates or deallocates memory}}
			return {};
		}
	};

	struct Stringy2 {
		friend Stringy2 operator+(const Stringy2& x, const Stringy2& y)
		{
			// Do something inferably unsafe
			throw 42; // expected-note {{function cannot be inferred 'nonblocking' because it throws or catches exceptions}}
		}
	};

void nb9() [[clang::nonblocking]]
{
	Adder<int>::add_explicit(1, 2);
	Adder<int>::add_implicit(1, 2);

	Adder<Stringy>::add_explicit({}, {}); // expected-note {{in template expansion here}}
	Adder<Stringy2>::add_implicit({}, {}); // expected-warning {{function with 'nonblocking' attribute must not call non-'nonblocking' function}} \
		expected-note {{in template expansion here}}
}

// Make sure we verify lambdas produced from template expansions.
struct HasTemplatedLambda {
	void (*fptr)() [[clang::nonblocking]];

	template <typename C>
	HasTemplatedLambda(const C&)
		: fptr{ []() [[clang::nonblocking]] {
			auto* y = new int; // expected-warning {{lambda with 'nonblocking' attribute must not allocate or deallocate memory}}
		} }
	{}
};

void nb9a()
{
	HasTemplatedLambda bad(42);
}

// Templated function and lambda.
template <typename T>
void TemplatedFunc(T x) [[clang::nonblocking]] {
	auto* ptr = new T; // expected-warning {{function with 'nonblocking' attribute must not allocate or deallocate memory}}
}

void nb9b() [[clang::nonblocking]] {
	TemplatedFunc(42); // expected-note {{in template expansion here}}

	auto foo = [](auto x) [[clang::nonblocking]] {
		auto* ptr = new int; // expected-warning {{lambda with 'nonblocking' attribute must not allocate or deallocate memory}}
		return x;
	};

	// Note that foo() won't be validated unless instantiated.
	foo(42);
}

void nb10(
	void (*fp1)(), // expected-note {{function pointer cannot be inferred 'nonblocking'}}
	void (*fp2)() [[clang::nonblocking]]
	) [[clang::nonblocking]]
{
	fp1(); // expected-warning {{function with 'nonblocking' attribute must not call non-'nonblocking' function}}
	fp2();

	// When there's a cast, there's a separate diagnostic.
	static_cast<void (*)()>(fp1)(); // expected-warning {{function with 'nonblocking' attribute must not call non-'nonblocking' expression}}
}

// Expression involving indirection
int nb10a() [[clang::nonblocking]];
int nb10b() [[clang::nonblocking]];
int blocking();

int nb10c(bool x) [[clang::nonblocking]]
{
	int y = (x ? nb10a : blocking)(); // expected-warning {{attribute 'nonblocking' should not be added via type conversion}}
	return (x ? nb10a : nb10b)(); // No diagnostic.
}

// Interactions with nonblocking(false)
void nb11_no_inference_1() [[clang::nonblocking(false)]] // expected-note {{function does not permit inference of 'nonblocking'}}
{
}
void nb11_no_inference_2() [[clang::nonblocking(false)]]; // expected-note {{function does not permit inference of 'nonblocking'}}

template <bool V>
struct ComputedNB {
	void method() [[clang::nonblocking(V)]]; // expected-note {{function does not permit inference of 'nonblocking' because it is declared 'blocking'}}
};

void nb11() [[clang::nonblocking]]
{
	nb11_no_inference_1(); // expected-warning {{function with 'nonblocking' attribute must not call non-'nonblocking' function}}
	nb11_no_inference_2(); // expected-warning {{function with 'nonblocking' attribute must not call non-'nonblocking' function}}

	ComputedNB<true> CNB_true;
	CNB_true.method();
	
	ComputedNB<false> CNB_false;
	CNB_false.method(); // expected-warning {{function with 'nonblocking' attribute must not call non-'nonblocking' function}}
}

// Verify that when attached to a redeclaration, the attribute successfully attaches.
void nb12() {
	static int x; // expected-warning {{function with 'nonblocking' attribute must not have static local variables}}
}
void nb12() [[clang::nonblocking]];
void nb13() [[clang::nonblocking]] { nb12(); }

// C++ member function pointers
struct PTMFTester {
	typedef void (PTMFTester::*ConvertFunction)() [[clang::nonblocking]];

	void convert() [[clang::nonblocking]];

	ConvertFunction mConvertFunc;
};

void PTMFTester::convert() [[clang::nonblocking]]
{
	(this->*mConvertFunc)();
}

// Block variables
void nb17(void (^blk)() [[clang::nonblocking]]) [[clang::nonblocking]] {
	blk();
}

// References to blocks
void nb18(void (^block)() [[clang::nonblocking]]) [[clang::nonblocking]]
{
	auto &ref = block;
	ref();
}

// Builtin functions
void nb19() [[clang::nonblocking]] {
	__builtin_assume(1);
	void *ptr = __builtin_malloc(1); // expected-warning {{function with 'nonblocking' attribute must not call non-'nonblocking' function '__builtin_malloc'}}
	__builtin_free(ptr); // expected-warning {{function with 'nonblocking' attribute must not call non-'nonblocking' function '__builtin_free'}}
	
	void *p2 = __builtin_operator_new(1); // expected-warning {{function with 'nonblocking' attribute must not call non-'nonblocking' function '__builtin_operator_new'}}
	__builtin_operator_delete(p2); // expected-warning {{function with 'nonblocking' attribute must not call non-'nonblocking' function '__builtin_operator_delete'}}
}

// Function try-block
void catches() try {} catch (...) {} // expected-note {{function cannot be inferred 'nonblocking' because it throws or catches exceptions}}

void nb20() [[clang::nonblocking]] {
	catches(); // expected-warning {{function with 'nonblocking' attribute must not call non-'nonblocking' function 'catches'}}
}

struct S {
    int x;
    S(int x) try : x(x) {} catch (...) {} // expected-note {{constructor cannot be inferred 'nonblocking' because it throws or catches exceptions}}
    S(double) : x((throw 3, 3)) {} // expected-note {{member initializer cannot be inferred 'nonblocking' because it throws or catches exceptions}} \
                                      expected-note {{in constructor here}}
};

int badi(); // expected-note {{declaration cannot be inferred 'nonblocking' because it has no definition in this translation unit}} \
            // expected-note {{declaration cannot be inferred 'nonblocking' because it has no definition in this translation unit}}

struct A {                // expected-note {{in implicit constructor here}}
    int x = (throw 3, 3); // expected-note {{member initializer cannot be inferred 'nonblocking' because it throws or catches exceptions}}
};

struct B {
    int y = badi(); // expected-note {{member initializer cannot be inferred 'nonblocking' because it calls non-'nonblocking' function 'badi'}}
};

void f() [[clang::nonblocking]] {
    S s1(3);   // expected-warning {{function with 'nonblocking' attribute must not call non-'nonblocking' constructor 'S::S'}}
    S s2(3.0); // expected-warning {{function with 'nonblocking' attribute must not call non-'nonblocking' constructor 'S::S'}}
    A a;       // expected-warning {{function with 'nonblocking' attribute must not call non-'nonblocking' constructor 'A::A'}}
    B b;       // expected-warning {{function with 'nonblocking' attribute must not call non-'nonblocking' constructor 'B::B'}}
}

struct T {
	int x = badi();               // expected-warning {{member initializer of constructor with 'nonblocking' attribute must not call non-'nonblocking' function 'badi'}}
	T() [[clang::nonblocking]] {} // expected-note {{in constructor here}}
	T(int x) [[clang::nonblocking]] : x(x) {} // OK
};

// Default arguments
int badForDefaultArg(); // expected-note {{declaration cannot be inferred 'nonblocking' because it has no definition in this translation unit}} \
                           expected-note {{declaration cannot be inferred 'nonblocking' because it has no definition in this translation unit}} \
						   expected-note {{declaration cannot be inferred 'nonblocking' because it has no definition in this translation unit}}

void hasDefaultArg(int param = badForDefaultArg()) { // expected-warning {{function with 'nonblocking' attribute must not call non-'nonblocking' function 'badForDefaultArg'}} \
                                                        expected-note {{function cannot be inferred 'nonblocking' because it calls non-'nonblocking' function 'badForDefaultArg'}}
}

void nb21() [[clang::nonblocking]] {
	hasDefaultArg(); // expected-note {{in evaluating default argument here}} \
	                    expected-warning {{function with 'nonblocking' attribute must not call non-'nonblocking' function 'hasDefaultArg'}}
}

void nb22(int param = badForDefaultArg()) [[clang::nonblocking]] { // expected-warning {{function with 'nonblocking' attribute must not call non-'nonblocking' function 'badForDefaultArg'}}
}

// Verify traversal of implicit code paths - constructors and destructors.
struct Unsafe {
  static void problem1();   // expected-note {{declaration cannot be inferred 'nonblocking' because it has no definition in this translation unit}}
  static void problem2();   // expected-note {{declaration cannot be inferred 'nonblocking' because it has no definition in this translation unit}}

  Unsafe() { problem1(); }  // expected-note {{constructor cannot be inferred 'nonblocking' because it calls non-'nonblocking' function 'Unsafe::problem1'}}
  ~Unsafe() { problem2(); } // expected-note {{destructor cannot be inferred 'nonblocking' because it calls non-'nonblocking' function 'Unsafe::problem2'}}

  Unsafe(int x); // expected-note {{declaration cannot be inferred 'nonblocking' because it has no definition in this translation unit}} expected-note {{declaration cannot be inferred 'nonblocking' because it has no definition in this translation unit}}

  // Delegating initializer.
  Unsafe(float y) [[clang::nonblocking]] : Unsafe(int(y)) {} // expected-warning {{constructor with 'nonblocking' attribute must not call non-'nonblocking' constructor 'Unsafe::Unsafe'}}
};

struct DerivedFromUnsafe : public Unsafe {
  DerivedFromUnsafe() [[clang::nonblocking]] {} // expected-warning {{constructor with 'nonblocking' attribute must not call non-'nonblocking' constructor 'Unsafe::Unsafe'}}
  DerivedFromUnsafe(int x) [[clang::nonblocking]] : Unsafe(x) {} // expected-warning {{constructor with 'nonblocking' attribute must not call non-'nonblocking' constructor 'Unsafe::Unsafe'}}
  ~DerivedFromUnsafe() [[clang::nonblocking]] {} // expected-warning {{destructor with 'nonblocking' attribute must not call non-'nonblocking' destructor 'Unsafe::~Unsafe'}}
};

// Don't try to follow a deleted destructor, as with std::optional<T>.
struct HasDtor {
	~HasDtor() {}
};

template <typename T>
struct Optional {
	union {
		char __null_state_;
		T __val_;
	};
	bool engaged = false;

	~Optional() {
		if (engaged)
			__val_.~T();
	}
};

void nb_opt() [[clang::nonblocking]] {
	Optional<HasDtor> x;
}

// Virtual inheritance
struct VBase {
  int *Ptr;

  VBase() { Ptr = new int; }       // expected-note {{constructor cannot be inferred 'nonblocking' because it allocates or deallocates memory}}
  virtual ~VBase() { delete Ptr; } // expected-note {{virtual method cannot be inferred 'nonblocking'}}
};

struct VDerived : virtual VBase {
  VDerived() [[clang::nonblocking]] {} // expected-warning {{constructor with 'nonblocking' attribute must not call non-'nonblocking' constructor 'VBase::VBase'}}

  ~VDerived() [[clang::nonblocking]] {} // expected-warning {{destructor with 'nonblocking' attribute must not call non-'nonblocking' destructor 'VBase::~VBase'}}
};

// Contexts where there is no function call, no diagnostic.
bool bad();

template <bool>
requires requires { bad(); }
void g() [[clang::nonblocking]] {}

void g() [[clang::nonblocking]] {
    decltype(bad()) a; // doesn't generate a call so, OK
    [[maybe_unused]] auto b = noexcept(bad());
    [[maybe_unused]] auto c = sizeof(bad());
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wassume"
    [[assume(bad())]]; // never evaluated, but maybe still semantically questionable?
#pragma clang diagnostic pop
}

// Make sure we are skipping concept requirements -- they can trigger an unexpected
// warning involving use of a function pointer (e.g. std::reverse_iterator::operator==
struct HasFoo { int foo() const { return 0; } };

template <class A, class B>
inline bool compare(const A& a, const B& b)
	requires requires { 
		a.foo();
	}
{
	return a.foo() == b.foo();
}

void nb25() [[clang::nonblocking]] {
	HasFoo a, b;
	compare(a, b);
}

// If the callee is both noreturn and noexcept, it presumably terminates.
// Ignore it for the purposes of effect analysis.
[[noreturn]] void abort_wrapper() noexcept;

void nb26() [[clang::nonblocking]] {
	abort_wrapper(); // no diagnostic
}

// --- Make sure we don't traverse requires and noexcept clauses. ---

// Apparently some requires clauses are able to be collapsed into a constant before the nonblocking
// analysis sees any function calls. This example (extracted from a real-world case where
// `operator&&` in <valarray>, preceding the inclusion of <expected>) is sufficiently complex
// to look like it contains function calls. There may be simpler examples.

namespace ExpectedTest {

template <class _Tp>
inline constexpr bool is_copy_constructible_v = __is_constructible(_Tp, _Tp&);

template <bool, class _Tp = void>
struct enable_if {};
template <class _Tp>
struct enable_if<true, _Tp> {
  typedef _Tp type;
};

template <bool _Bp, class _Tp = void>
using enable_if_t = typename enable_if<_Bp, _Tp>::type;

// Doesn't seem to matter whether the enable_if is true or false.
template <class E1, class E2, enable_if_t<is_copy_constructible_v<E1>> = 0>
inline bool operator&&(const E1& x, const E2& y);

template <class _Tp, class _Err>
class expected {
public:
  constexpr expected()
    {}

  // This is a deliberate corruption of the real implementation for simplicity.
  constexpr expected(const expected&)
    requires(is_copy_constructible_v<_Tp> && is_copy_constructible_v<_Err>)
  = default;
};

void test() [[clang::nonblocking]]
{
	expected<int, int> a;
	auto b = a;            // Copy constructor.
}

} // namespace ExpectedTest

// Make sure a function call in a noexcept() clause is ignored.
constexpr bool foo() [[clang::nonblocking(false)]] { return true; }
void nb27() noexcept(foo()) [[clang::nonblocking]] {}

// Make sure that simple type traits don't cause violations.
void nb28() [[clang::nonblocking]] {
	bool x = __is_constructible(int, const int&);
}

// --- nonblocking implies noexcept ---
#pragma clang diagnostic warning "-Wperf-constraint-implies-noexcept"

void needs_noexcept() [[clang::nonblocking]] // expected-warning {{function with 'nonblocking' attribute should be declared noexcept}}
{
	auto lambda = []() [[clang::nonblocking]] {}; // expected-warning {{lambda with 'nonblocking' attribute should be declared noexcept}}
}
