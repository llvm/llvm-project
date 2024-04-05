// RUN: %clang_cc1 -fsyntax-only -fblocks -fcxx-exceptions -verify %s
// RUN: %clang_cc1 -fsyntax-only -fblocks -verify -x c -std=c23 %s

#if !__has_attribute(clang_nonblocking)
#error "the 'nonblocking' attribute is not available"
#endif

// --- ATTRIBUTE SYNTAX: SUBJECTS ---

int nl_var [[clang::nonblocking]]; // expected-warning {{'nonblocking' only applies to function types; type here is 'int'}}
struct nl_struct {} [[clang::nonblocking]]; // expected-warning {{attribute 'nonblocking' is ignored, place it after "struct" to apply attribute to type declaration}}
struct [[clang::nonblocking]] nl_struct2 {}; // expected-error {{'nonblocking' attribute cannot be applied to a declaration}}

// --- ATTRIBUTE SYNTAX: COMBINATIONS ---
// Check invalid combinations of nonblocking/nonallocating attributes

void nl_true_false_1() [[clang::nonblocking(true)]] [[clang::blocking]]; // expected-error {{nonblocking(true) and blocking attributes are not compatible}}
void nl_true_false_2() [[clang::blocking]] [[clang::nonblocking(true)]]; // expected-error {{nonblocking(true) and blocking attributes are not compatible}}

void na_true_false_1() [[clang::nonallocating(true)]] [[clang::allocating]]; // expected-error {{nonallocating(true) and allocating attributes are not compatible}}
void na_true_false_2() [[clang::allocating]] [[clang::nonallocating(true)]]; // expected-error {{nonallocating(true) and allocating attributes are not compatible}}

void nl_true_na_true_1() [[clang::nonblocking]] [[clang::nonallocating]];
void nl_true_na_true_2() [[clang::nonallocating]] [[clang::nonblocking]];

void nl_true_na_false_1() [[clang::nonblocking]] [[clang::allocating]]; // expected-error {{nonblocking(true) and allocating attributes are not compatible}}
void nl_true_na_false_2() [[clang::allocating]] [[clang::nonblocking]]; // expected-error {{nonblocking(true) and allocating attributes are not compatible}}

void nl_false_na_true_1() [[clang::blocking]] [[clang::nonallocating]];
void nl_false_na_true_2() [[clang::nonallocating]] [[clang::blocking]];

void nl_false_na_false_1() [[clang::blocking]] [[clang::allocating]];
void nl_false_na_false_2() [[clang::allocating]] [[clang::blocking]];

// --- TYPE CONVERSIONS ---

void unannotated();
void nonblocking() [[clang::nonblocking]];
void nonallocating() [[clang::nonallocating]];
void type_conversions()
{
	// It's fine to remove a performance constraint.
	void (*fp_plain)();

	fp_plain = unannotated;
	fp_plain = nonblocking;
	fp_plain = nonallocating;

	// Adding/spoofing nonblocking is unsafe.
	void (*fp_nonblocking)() [[clang::nonblocking]];
	fp_nonblocking = nonblocking;
	fp_nonblocking = unannotated; // expected-warning {{attribute 'nonblocking' should not be added via type conversion}}
	fp_nonblocking = nonallocating; // expected-warning {{attribute 'nonblocking' should not be added via type conversion}}

	// Adding/spoofing nonallocating is unsafe.
	void (*fp_nonallocating)() [[clang::nonallocating]];
	fp_nonallocating = nonallocating;
	fp_nonallocating = nonblocking; // no warning because nonblocking includes nonallocating fp_nonallocating = unannotated;
	fp_nonallocating = unannotated; // expected-warning {{attribute 'nonallocating' should not be added via type conversion}}
}

#ifdef __cplusplus
// There was a bug: noexcept and nonblocking could be individually removed in conversion, but not both	
void type_conversions_2()
{
	auto receives_fp = [](void (*fp)()) {
	};
	
	auto ne = +[]() noexcept {};
	auto nl = +[]() [[clang::nonblocking]] {};
	auto nl_ne = +[]() noexcept [[clang::nonblocking]] {};
	
	receives_fp(ne);
	receives_fp(nl);
	receives_fp(nl_ne);
}
#endif

// --- VIRTUAL METHODS ---
// Attributes propagate to overridden methods, so no diagnostics.
// Check this in the syntax tests too.
#ifdef __cplusplus
struct Base {
	virtual void f1();
	virtual void nonblocking() noexcept [[clang::nonblocking]];
	virtual void nonallocating() noexcept [[clang::nonallocating]];
};

struct Derived : public Base {
	void f1() [[clang::nonblocking]] override;
	void nonblocking() noexcept override;
	void nonallocating() noexcept override;
};
#endif // __cplusplus

// --- REDECLARATIONS ---

#ifdef __cplusplus
// In C++, the third declaration gets seen as a redeclaration of the second.
void f2();
void f2() [[clang::nonblocking]]; // expected-note {{previous declaration is here}}
void f2(); // expected-warning {{attribute 'nonblocking' on function does not match previous declaration}}
#else
// In C, the third declaration is redeclaration of the first (?).
void f2();
void f2() [[clang::nonblocking]];
void f2();
#endif
// Note: we verify that the attribute is actually seen during the constraints tests.

// --- OVERLOADS ---
#ifdef __cplusplus
struct S {
	void foo(); // expected-note {{previous declaration is here}}
	void foo(); // expected-error {{class member cannot be redeclared}}
};
#endif // __cplusplus
