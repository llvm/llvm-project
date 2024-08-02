// RUN: %clang_cc1 -fsyntax-only -fblocks -fcxx-exceptions -verify %s
// These are in a separate file because errors (e.g. incompatible attributes) currently prevent
// the FXAnalysis pass from running at all.

// This diagnostic is re-enabled and exercised in isolation later in this file.
#pragma clang diagnostic ignored "-Wperf-constraint-implies-noexcept"

// --- CONSTRAINTS ---

void nb1() [[clang::nonblocking]]
{
	int *pInt = new int; // expected-warning {{'nonblocking' function must not allocate or deallocate memory}}
	delete pInt; // expected-warning {{'nonblocking' function must not allocate or deallocate memory}}
}

void nb2() [[clang::nonblocking]]
{
	static int global; // expected-warning {{'nonblocking' function must not have static locals}}
}

void nb3() [[clang::nonblocking]]
{
	try {
		throw 42; // expected-warning {{'nonblocking' function must not throw or catch exceptions}}
	}
	catch (...) { // expected-warning {{'nonblocking' function must not throw or catch exceptions}}
	}
}

void nb4_inline() {}
void nb4_not_inline(); // expected-note {{function cannot be inferred 'nonblocking' because it has no definition in this translation unit}}

void nb4() [[clang::nonblocking]]
{
	nb4_inline(); // OK
	nb4_not_inline(); // expected-warning {{'nonblocking' function must not call non-'nonblocking' function}}
}


struct HasVirtual {
	virtual void unsafe(); // expected-note {{virtual method cannot be inferred 'nonblocking'}}
};

void nb5() [[clang::nonblocking]]
{
 	HasVirtual hv;
 	hv.unsafe(); // expected-warning {{'nonblocking' function must not call non-'nonblocking' function}}
}

void nb6_unsafe(); // expected-note {{function cannot be inferred 'nonblocking' because it has no definition in this translation unit}}
void nb6_transitively_unsafe()
{
	nb6_unsafe(); // expected-note {{function cannot be inferred 'nonblocking' because it calls non-'nonblocking' function}}
}

void nb6() [[clang::nonblocking]]
{
	nb6_transitively_unsafe(); // expected-warning {{'nonblocking' function must not call non-'nonblocking' function}}
}

thread_local int tl_var{ 42 };

bool tl_test() [[clang::nonblocking]]
{
	return tl_var > 0; // expected-warning {{'nonblocking' function must not use thread-local variables}}
}

void nb7()
{
	// Make sure we verify blocks
	auto blk = ^() [[clang::nonblocking]] {
		throw 42; // expected-warning {{'nonblocking' function must not throw or catch exceptions}}
	};
}

void nb8()
{
	// Make sure we verify lambdas
	auto lambda = []() [[clang::nonblocking]] {
		throw 42; // expected-warning {{'nonblocking' function must not throw or catch exceptions}}
	};
}

// Make sure template expansions are found and verified.
	template <typename T>
	struct Adder {
		static T add_explicit(T x, T y) [[clang::nonblocking]]
		{
			return x + y; // expected-warning {{'nonblocking' function must not call non-'nonblocking' function}}
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
			auto* z = new char[42]; // expected-note {{function cannot be inferred 'nonblocking' because it allocates/deallocates memory}}
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
	Adder<Stringy2>::add_implicit({}, {}); // expected-warning {{'nonblocking' function must not call non-'nonblocking' function}} \
		expected-note {{in template expansion here}}
}

void nb10(
	void (*fp1)(), // expected-note {{function pointer cannot be inferred 'nonblocking'}}
	void (*fp2)() [[clang::nonblocking]]
	) [[clang::nonblocking]]
{
	fp1(); // expected-warning {{'nonblocking' function must not call non-'nonblocking' function}}
	fp2();
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
	nb11_no_inference_1(); // expected-warning {{'nonblocking' function must not call non-'nonblocking' function}}
	nb11_no_inference_2(); // expected-warning {{'nonblocking' function must not call non-'nonblocking' function}}

	ComputedNB<true> CNB_true;
	CNB_true.method();
	
	ComputedNB<false> CNB_false;
	CNB_false.method(); // expected-warning {{'nonblocking' function must not call non-'nonblocking' function}}
}

// Verify that when attached to a redeclaration, the attribute successfully attaches.
void nb12() {
	static int x; // expected-warning {{'nonblocking' function must not have static locals}}
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

// Verify traversal of implicit code paths - constructors and destructors.
struct Unsafe {
  static void problem1(); // expected-note {{function cannot be inferred 'nonblocking' because it has no definition in this translation unit}}
  static void problem2(); // expected-note {{function cannot be inferred 'nonblocking' because it has no definition in this translation unit}}

  Unsafe() { problem1(); } // expected-note {{function cannot be inferred 'nonblocking' because it calls non-'nonblocking' function 'Unsafe::problem1'}}
  ~Unsafe() { problem2(); } // expected-note {{function cannot be inferred 'nonblocking' because it calls non-'nonblocking' function 'Unsafe::problem2'}}
};

struct DerivedFromUnsafe : public Unsafe {
  DerivedFromUnsafe() [[clang::nonblocking]] {} // expected-warning {{'nonblocking' function must not call non-'nonblocking' function 'Unsafe::Unsafe'}}
  ~DerivedFromUnsafe() [[clang::nonblocking]] {} // expected-warning {{'nonblocking' function must not call non-'nonblocking' function 'Unsafe::~Unsafe'}}
};

// --- nonblocking implies noexcept ---
#pragma clang diagnostic warning "-Wperf-constraint-implies-noexcept"

void nb19() [[clang::nonblocking]] // expected-warning {{'nonblocking' function should be declared noexcept}}
{
}
