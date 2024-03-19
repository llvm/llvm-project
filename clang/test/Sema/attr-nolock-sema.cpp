// RUN: %clang_cc1 -fsyntax-only -fblocks -fcxx-exceptions -verify %s
// RUN: %clang_cc1 -fsyntax-only -fblocks -verify -x c -std=c23 %s

#if !__has_attribute(clang_nolock)
#error "the 'nolock' attribute is not available"
#endif

// --- ATTRIBUTE SYNTAX: SUBJECTS ---

int nl_var [[clang::nolock]]; // expected-warning {{'nolock' only applies to function types; type here is 'int'}}
struct nl_struct {} [[clang::nolock]]; // expected-warning {{attribute 'nolock' is ignored, place it after "struct" to apply attribute to type declaration}}
struct [[clang::nolock]] nl_struct2 {}; // expected-error {{'nolock' attribute cannot be applied to a declaration}}

// --- ATTRIBUTE SYNTAX: COMBINATIONS ---
// Check invalid combinations of nolock/noalloc attributes

void nl_true_false_1() [[clang::nolock(true)]] [[clang::nolock(false)]]; // expected-error {{nolock(true) and nolock(false) attributes are not compatible}}
void nl_true_false_2() [[clang::nolock(false)]] [[clang::nolock(true)]]; // expected-error {{nolock(true) and nolock(false) attributes are not compatible}}

void na_true_false_1() [[clang::noalloc(true)]] [[clang::noalloc(false)]]; // expected-error {{noalloc(true) and noalloc(false) attributes are not compatible}}
void na_true_false_2() [[clang::noalloc(false)]] [[clang::noalloc(true)]]; // expected-error {{noalloc(true) and noalloc(false) attributes are not compatible}}

void nl_true_na_true_1() [[clang::nolock]] [[clang::noalloc]];
void nl_true_na_true_2() [[clang::noalloc]] [[clang::nolock]];

void nl_true_na_false_1() [[clang::nolock]] [[clang::noalloc(false)]]; // expected-error {{nolock(true) and noalloc(false) attributes are not compatible}}
void nl_true_na_false_2() [[clang::noalloc(false)]] [[clang::nolock]]; // expected-error {{nolock(true) and noalloc(false) attributes are not compatible}}

void nl_false_na_true_1() [[clang::nolock(false)]] [[clang::noalloc]];
void nl_false_na_true_2() [[clang::noalloc]] [[clang::nolock(false)]];

void nl_false_na_false_1() [[clang::nolock(false)]] [[clang::noalloc(false)]];
void nl_false_na_false_2() [[clang::noalloc(false)]] [[clang::nolock(false)]];

// --- TYPE CONVERSIONS ---

void unannotated();
void nolock() [[clang::nolock]];
void noalloc() [[clang::noalloc]];
void type_conversions()
{
	// It's fine to remove a performance constraint.
	void (*fp_plain)();

	fp_plain = unannotated;
	fp_plain = nolock;
	fp_plain = noalloc;

	// Adding/spoofing nolock is unsafe.
	void (*fp_nolock)() [[clang::nolock]];
	fp_nolock = nolock;
	fp_nolock = unannotated; // expected-warning {{attribute 'nolock' should not be added via type conversion}}
	fp_nolock = noalloc; // expected-warning {{attribute 'nolock' should not be added via type conversion}}

	// Adding/spoofing noalloc is unsafe.
	void (*fp_noalloc)() [[clang::noalloc]];
	fp_noalloc = noalloc;
	fp_noalloc = nolock; // no warning because nolock includes noalloc fp_noalloc = unannotated;
	fp_noalloc = unannotated; // expected-warning {{attribute 'noalloc' should not be added via type conversion}}
}

// --- VIRTUAL METHODS ---
#ifdef __cplusplus
struct Base {
	virtual void f1();
	virtual void nolock() noexcept [[clang::nolock]]; // expected-note {{overridden virtual function is here}}
	virtual void noalloc() noexcept [[clang::noalloc]]; // expected-note {{overridden virtual function is here}}
};

struct Derived : public Base {
	void f1() [[clang::nolock]] override;
	void nolock() noexcept override; // expected-warning {{attribute 'nolock' on overriding function does not match base version}}
	void noalloc() noexcept override; // expected-warning {{attribute 'noalloc' on overriding function does not match base version}}
};
#endif // __cplusplus

// --- REDECLARATIONS ---

#ifdef __cplusplus
// In C++, the third declaration gets seen as a redeclaration of the second.
void f2();
void f2() [[clang::nolock]]; // expected-note {{previous declaration is here}}
void f2(); // expected-warning {{attribute 'nolock' on function does not match previous declaration}}
#else
// In C, the third declaration is redeclaration of the first (?).
void f2();
void f2() [[clang::nolock]];
void f2();
#endif
// Note: we verify that the attribute is actually seen during the constraints tests.


// Ensure that the redeclaration's attribute is seen and diagnosed correctly.

// void f2() {
// 	static int x;
// }
// void f2() [[clang::nolock]];
// 
// void f3() [[clang::nolock]] {
// 	f2();
// }

#if 0
int f2();
// redeclaration with a stronger constraint is OK.
int f2() [[clang::nolock]]; // e xpected-note {{previous declaration is here}}
int f2() { return 42; } // e xpected-warning {{attribute 'nolock' on function does not match previous declaration}}

int f3();
// redeclaration with a stronger constraint is OK.
int f3() [[clang::noalloc]]; // e xpected-note {{previous declaration is here}}
int f3() { return 42; } // e xpected-warning {{attribute 'noalloc' on function does not match previous declaration}}
#endif
