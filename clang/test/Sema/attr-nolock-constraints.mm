// RUN: %clang_cc1 -fsyntax-only -fblocks -fcxx-exceptions -verify %s
// These are in a separate file because errors (e.g. incompatible attributes) currently prevent
// the AnalysisBasedWarnings pass from running at all.

#if !__has_attribute(clang_nolock)
#error "the 'nolock' attribute is not available"
#endif

// --- CONSTRAINTS ---

void nl1() [[clang::nolock]]
{
	auto* pInt = new int; // expected-warning {{'nolock' function must not allocate or deallocate memory}}
}

void nl2() [[clang::nolock]]
{
	static int global; // expected-warning {{'nolock' function must not have static locals}}
}

void nl3() [[clang::nolock]]
{
	try {
		throw 42; // expected-warning {{'nolock' function must not throw or catch exceptions}}
	}
	catch (...) { // expected-warning {{'nolock' function must not throw or catch exceptions}}
	}
}

void nl4_inline() {}
void nl4_not_inline(); // expected-note {{function cannot be inferred 'nolock' because it has no definition in this translation unit}}

void nl4() [[clang::nolock]]
{
	nl4_inline(); // OK
	nl4_not_inline(); // expected-warning {{'nolock' function must not call non-'nolock' function}}
}


struct HasVirtual {
	virtual void unsafe(); // expected-note {{virtual method cannot be inferred 'nolock'}}
};

void nl5() [[clang::nolock]]
{
 	HasVirtual hv;
 	hv.unsafe(); // expected-warning {{'nolock' function must not call non-'nolock' function}}
}

void nl6_unsafe(); // expected-note {{function cannot be inferred 'nolock' because it has no definition in this translation unit}}
void nl6_transitively_unsafe()
{
	nl6_unsafe(); // expected-note {{function cannot be inferred 'nolock' because it calls non-'nolock' function}}
}

void nl6() [[clang::nolock]]
{
	nl6_transitively_unsafe(); // expected-warning {{'nolock' function must not call non-'nolock' function}}
}

thread_local int tl_var{ 42 };

bool tl_test() [[clang::nolock]]
{
	return tl_var > 0; // expected-warning {{'nolock' function must not use thread-local variables}}
}

void nl7()
{
	// Make sure we verify blocks
	auto blk = ^() [[clang::nolock]] {
		throw 42; // expected-warning {{'nolock' function must not throw or catch exceptions}}
	};
}

void nl8()
{
	// Make sure we verify lambdas
	auto lambda = []() [[clang::nolock]] {
		throw 42; // expected-warning {{'nolock' function must not throw or catch exceptions}}
	};
}

// Make sure template expansions are found and verified.
	template <typename T>
	struct Adder {
		static T add_explicit(T x, T y) [[clang::nolock]]
		{
			return x + y; // expected-warning {{'nolock' function must not call non-'nolock' function}}
		}
		static T add_implicit(T x, T y)
		{
			return x + y; // expected-note {{function cannot be inferred 'nolock' because it calls non-'nolock' function}}
		}
	};

	struct Stringy {
		friend Stringy operator+(const Stringy& x, const Stringy& y)
		{
			// Do something inferably unsafe
			auto* z = new char[42]; // expected-note {{function cannot be inferred 'nolock' because it allocates/deallocates memory}}
			return {};
		}
	};

	struct Stringy2 {
		friend Stringy2 operator+(const Stringy2& x, const Stringy2& y)
		{
			// Do something inferably unsafe
			throw 42; // expected-note {{function cannot be inferred 'nolock' because it throws or catches exceptions}}
		}
	};

void nl9() [[clang::nolock]]
{
	Adder<int>::add_explicit(1, 2);
	Adder<int>::add_implicit(1, 2);

	Adder<Stringy>::add_explicit({}, {}); // expected-note {{in template expansion here}}
	Adder<Stringy2>::add_implicit({}, {}); // expected-warning {{'nolock' function must not call non-'nolock' function}} \
		expected-note {{in template expansion here}}
}

void nl10(
	void (*fp1)(), // expected-note {{function pointer cannot be inferred 'nolock'}}
	void (*fp2)() [[clang::nolock]]
	) [[clang::nolock]]
{
	fp1(); // expected-warning {{'nolock' function must not call non-'nolock' function}}
	fp2();
}

// Interactions with nolock(false)
void nl11_no_inference() [[clang::nolock(false)]] // expected-note {{function does not permit inference of 'nolock'}}
{
}

void nl11() [[clang::nolock]]
{
	nl11_no_inference(); // expected-warning {{'nolock' function must not call non-'nolock' function}}
}

// Verify that when attached to a redeclaration, the attribute successfully attaches.
void nl12() {
	static int x; // expected-warning {{'nolock' function must not have static locals}}
}
void nl12() [[clang::nolock]];
void nl13() [[clang::nolock]] { nl12(); }

// Objective-C
@interface OCClass
- (void)method;
@end

void nl14(OCClass *oc) [[clang::nolock]] {
	[oc method]; // expected-warning {{'nolock' function must not access an ObjC method or property}}
}
void nl15(OCClass *oc) {
	[oc method]; // expected-note {{function cannot be inferred 'nolock' because it accesses an ObjC method or property}}
}
void nl16(OCClass *oc) [[clang::nolock]] {
	nl15(oc); // expected-warning {{'nolock' function must not call non-'nolock' function 'nl15'}}
}

// C++ member function pointers
struct PTMFTester {
	typedef void (PTMFTester::*ConvertFunction)() [[clang::nolock]];

	void convert() [[clang::nolock]];

	ConvertFunction mConvertFunc;
};

void PTMFTester::convert() [[clang::nolock]]
{
	(this->*mConvertFunc)();
}

// Block variables
void nl17(void (^blk)() [[clang::nolock]]) [[clang::nolock]] {
	blk();
}

// References to blocks
void nl18(void (^block)() [[clang::nolock]]) [[clang::nolock]]
{
	auto &ref = block;
	ref();
}

