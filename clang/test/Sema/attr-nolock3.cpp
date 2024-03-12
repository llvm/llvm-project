// RUN: %clang_cc1 -fsyntax-only -fblocks -fcxx-exceptions -verify %s

#if !__has_attribute(clang_nolock)
#error "the 'nolock' attribute is not available"
#endif

// --- CONSTRAINTS ---

void nl1() [[clang::nolock]]
{
	auto* pInt = new int; // expected-warning {{'nolock' function 'nl1' must not allocate or deallocate memory}}
}

void nl2() [[clang::nolock]]
{
	static int global; // expected-warning {{'nolock' function 'nl2' must not have static locals}}
}

void nl3() [[clang::nolock]]
{
	try {
		throw 42; // expected-warning {{'nolock' function 'nl3' must not throw or catch exceptions}}
	}
	catch (...) { // expected-warning {{'nolock' function 'nl3' must not throw or catch exceptions}}
	}
}

void nl4_inline() {}
void nl4_not_inline(); // expected-note {{'nl4_not_inline' cannot be inferred 'nolock' because it has no definition in this translation unit}}

void nl4() [[clang::nolock]]
{
	nl4_inline(); // OK
	nl4_not_inline(); // expected-warning {{'nolock' function 'nl4' must not call non-'nolock' function 'nl4_not_inline'}}
}


struct HasVirtual {
	virtual void unsafe(); // expected-note {{'HasVirtual::unsafe' cannot be inferred 'nolock' because it is virtual}}
};

void nl5() [[clang::nolock]]
{
 	HasVirtual hv;
 	hv.unsafe(); // expected-warning {{'nolock' function 'nl5' must not call non-'nolock' function 'HasVirtual::unsafe'}}
}

void nl6_unsafe(); // expected-note {{'nl6_unsafe' cannot be inferred 'nolock' because it has no definition in this translation unit}}
void nl6_transitively_unsafe()
{
	nl6_unsafe(); // expected-note {{'nl6_transitively_unsafe' cannot be inferred 'nolock' because it calls non-'nolock' function 'nl6_unsafe'}}
}

void nl6() [[clang::nolock]]
{
	nl6_transitively_unsafe(); // expected-warning {{'nolock' function 'nl6' must not call non-'nolock' function 'nl6_transitively_unsafe'}}
}

thread_local int tl_var{ 42 };

bool tl_test() [[clang::nolock]]
{
	return tl_var > 0; // expected-warning {{'nolock' function 'tl_test' must not use thread-local variables}}
}

void nl7()
{
	// Make sure we verify blocks
	auto blk = ^() [[clang::nolock]] {
		throw 42; // expected-warning {{'nolock' function '(block 0)' must not throw or catch exceptions}}
	};
}

void nl8()
{
	// Make sure we verify lambdas
	auto lambda = []() [[clang::nolock]] {
		throw 42; // expected-warning {{'nolock' function 'nl8()::(anonymous class)::operator()' must not throw or catch exceptions}}
	};
}

// Make sure template expansions are found and verified.
	template <typename T>
	struct Adder {
		static T add_explicit(T x, T y) [[clang::nolock]]
		{
			return x + y; // expected-warning {{'nolock' function 'Adder<Stringy>::add_explicit' must not call non-'nolock' function 'operator+'}}
		}
		static T add_implicit(T x, T y)
		{
			return x + y; // expected-note {{'Adder<Stringy2>::add_implicit' cannot be inferred 'nolock' because it calls non-'nolock' function 'operator+'}}
		}
	};

	struct Stringy {
		friend Stringy operator+(const Stringy& x, const Stringy& y)
		{
			// Do something inferably unsafe
			auto* z = new char[42]; // expected-note {{'operator+' cannot be inferred 'nolock' because it allocates/deallocates memory}}
			return {};
		}
	};

	struct Stringy2 {
		friend Stringy2 operator+(const Stringy2& x, const Stringy2& y)
		{
			// Do something inferably unsafe
			throw 42; // expected-note {{'operator+' cannot be inferred 'nolock' because it throws or catches exceptions}}
		}
	};

void nl9() [[clang::nolock]]
{
	Adder<int>::add_explicit(1, 2);
	Adder<int>::add_implicit(1, 2);

	Adder<Stringy>::add_explicit({}, {}); // expected-note {{in template expansion here}}
	Adder<Stringy2>::add_implicit({}, {}); // expected-warning {{'nolock' function 'nl9' must not call non-'nolock' function 'Adder<Stringy2>::add_implicit'}} \
		expected-note {{in template expansion here}}
}

void nl10(
	void (*fp1)(), // expected-note {{'fp1' cannot be inferred 'nolock' because it is a function pointer}}
	void (*fp2)() [[clang::nolock]]
	) [[clang::nolock]]
{
	fp1(); // expected-warning {{'nolock' function 'nl10' must not call non-'nolock' function 'fp1'}}
	fp2();
}

// Interactions with nolock(false)
void nl11_no_inference() [[clang::nolock(false)]] // expected-note {{'nl11_no_inference' does not permit inference of 'nolock'}}
{
}

void nl11() [[clang::nolock]]
{
	nl11_no_inference(); // expected-warning {{'nolock' function 'nl11' must not call non-'nolock' function 'nl11_no_inference'}}
}
