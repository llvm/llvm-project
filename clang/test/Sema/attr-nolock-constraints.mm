// RUN: %clang_cc1 -fsyntax-only -fblocks -fcxx-exceptions -verify %s
// These are in a separate file because errors (e.g. incompatible attributes) currently prevent
// the AnalysisBasedWarnings pass from running at all.

#if !__has_attribute(clang_nonblocking)
#error "the 'nonblocking' attribute is not available"
#endif

// --- CONSTRAINTS ---

void nl1() [[clang::nonblocking]]
{
	auto* pInt = new int; // expected-warning {{'nonblocking' function must not allocate or deallocate memory}}
}

void nl2() [[clang::nonblocking]]
{
	static int global; // expected-warning {{'nonblocking' function must not have static locals}}
}

void nl3() [[clang::nonblocking]]
{
	try {
		throw 42; // expected-warning {{'nonblocking' function must not throw or catch exceptions}}
	}
	catch (...) { // expected-warning {{'nonblocking' function must not throw or catch exceptions}}
	}
}

void nl4_inline() {}
void nl4_not_inline(); // expected-note {{function cannot be inferred 'nonblocking' because it has no definition in this translation unit}}

void nl4() [[clang::nonblocking]]
{
	nl4_inline(); // OK
	nl4_not_inline(); // expected-warning {{'nonblocking' function must not call non-'nonblocking' function}}
}


struct HasVirtual {
	virtual void unsafe(); // expected-note {{virtual method cannot be inferred 'nonblocking'}}
};

void nl5() [[clang::nonblocking]]
{
 	HasVirtual hv;
 	hv.unsafe(); // expected-warning {{'nonblocking' function must not call non-'nonblocking' function}}
}

void nl6_unsafe(); // expected-note {{function cannot be inferred 'nonblocking' because it has no definition in this translation unit}}
void nl6_transitively_unsafe()
{
	nl6_unsafe(); // expected-note {{function cannot be inferred 'nonblocking' because it calls non-'nonblocking' function}}
}

void nl6() [[clang::nonblocking]]
{
	nl6_transitively_unsafe(); // expected-warning {{'nonblocking' function must not call non-'nonblocking' function}}
}

thread_local int tl_var{ 42 };

bool tl_test() [[clang::nonblocking]]
{
	return tl_var > 0; // expected-warning {{'nonblocking' function must not use thread-local variables}}
}

void nl7()
{
	// Make sure we verify blocks
	auto blk = ^() [[clang::nonblocking]] {
		throw 42; // expected-warning {{'nonblocking' function must not throw or catch exceptions}}
	};
}

void nl8()
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

void nl9() [[clang::nonblocking]]
{
	Adder<int>::add_explicit(1, 2);
	Adder<int>::add_implicit(1, 2);

	Adder<Stringy>::add_explicit({}, {}); // expected-note {{in template expansion here}}
	Adder<Stringy2>::add_implicit({}, {}); // expected-warning {{'nonblocking' function must not call non-'nonblocking' function}} \
		expected-note {{in template expansion here}}
}

void nl10(
	void (*fp1)(), // expected-note {{function pointer cannot be inferred 'nonblocking'}}
	void (*fp2)() [[clang::nonblocking]]
	) [[clang::nonblocking]]
{
	fp1(); // expected-warning {{'nonblocking' function must not call non-'nonblocking' function}}
	fp2();
}

// Interactions with nonblocking(false)
void nl11_no_inference() [[clang::nonblocking(false)]] // expected-note {{function does not permit inference of 'nonblocking'}}
{
}

void nl11() [[clang::nonblocking]]
{
	nl11_no_inference(); // expected-warning {{'nonblocking' function must not call non-'nonblocking' function}}
}

// Verify that when attached to a redeclaration, the attribute successfully attaches.
void nl12() {
	static int x; // expected-warning {{'nonblocking' function must not have static locals}}
}
void nl12() [[clang::nonblocking]];
void nl13() [[clang::nonblocking]] { nl12(); }

// Objective-C
@interface OCClass
- (void)method;
@end

void nl14(OCClass *oc) [[clang::nonblocking]] {
	[oc method]; // expected-warning {{'nonblocking' function must not access an ObjC method or property}}
}
void nl15(OCClass *oc) {
	[oc method]; // expected-note {{function cannot be inferred 'nonblocking' because it accesses an ObjC method or property}}
}
void nl16(OCClass *oc) [[clang::nonblocking]] {
	nl15(oc); // expected-warning {{'nonblocking' function must not call non-'nonblocking' function 'nl15'}}
}

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
void nl17(void (^blk)() [[clang::nonblocking]]) [[clang::nonblocking]] {
	blk();
}

// References to blocks
void nl18(void (^block)() [[clang::nonblocking]]) [[clang::nonblocking]]
{
	auto &ref = block;
	ref();
}

