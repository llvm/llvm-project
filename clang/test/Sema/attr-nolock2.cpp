// RUN: %clang_cc1 -fsyntax-only -fblocks -verify %s
// R UN: %clang_cc1 -fsyntax-only -fblocks -verify -x c -std=c2x %s

// TODO: There's a problem with diagnosing type conversions in plain C.

#pragma clang diagnostic error "-Wstrict-prototypes"

#if !__has_attribute(clang_nolock)
#error "the 'nolock' attribute is not available"
#endif

#if 1 // TEMP_DISABLE

// --- ATTRIBUTE SYNTAX: COMBINATIONS ---
// Check invalid combinations of nolock/noalloc attributes
void nl_true_false_1(void) [[clang::nolock(true)]] [[clang::nolock(false)]]; // expected-error {{nolock(true) and nolock(false) attributes are not compatible}}
void nl_true_false_2(void) [[clang::nolock(false)]] [[clang::nolock(true)]]; // expected-error {{nolock(true) and nolock(false) attributes are not compatible}}

void na_true_false_1(void) [[clang::noalloc(true)]] [[clang::noalloc(false)]]; // expected-error {{noalloc(true) and noalloc(false) attributes are not compatible}}
void na_true_false_2(void) [[clang::noalloc(false)]] [[clang::noalloc(true)]]; // expected-error {{noalloc(true) and noalloc(false) attributes are not compatible}}

void nl_true_na_true_1(void) [[clang::nolock]] [[clang::noalloc]];
void nl_true_na_true_2(void) [[clang::noalloc]] [[clang::nolock]];

void nl_true_na_false_1(void) [[clang::nolock]] [[clang::noalloc(false)]]; // expected-error {{nolock(true) and noalloc(false) attributes are not compatible}}
void nl_true_na_false_2(void) [[clang::noalloc(false)]] [[clang::nolock]]; // expected-error {{nolock(true) and noalloc(false) attributes are not compatible}}

void nl_false_na_true_1(void) [[clang::nolock(false)]] [[clang::noalloc]];
void nl_false_na_true_2(void) [[clang::noalloc]] [[clang::nolock(false)]];

void nl_false_na_false_1(void) [[clang::nolock(false)]] [[clang::noalloc(false)]];
void nl_false_na_false_2(void) [[clang::noalloc(false)]] [[clang::nolock(false)]];

// --- TYPE CONVERSIONS ---

void unannotated(void);
void nolock(void) [[clang::nolock]];
void noalloc(void) [[clang::noalloc]];
void type_conversions(void)
{
	// It's fine to remove a performance constraint.
	void (*fp_plain)(void);

	fp_plain = unannotated;
	fp_plain = nolock;
	fp_plain = noalloc;

	// Adding/spoofing nolock is unsafe.
	void (*fp_nolock)(void) [[clang::nolock]];
	fp_nolock = nolock;
	fp_nolock = unannotated; // expected-warning {{attribute 'nolock' should not be added via type conversion}}
	fp_nolock = noalloc; // expected-warning {{attribute 'nolock' should not be added via type conversion}}

	// Adding/spoofing noalloc is unsafe.
	void (*fp_noalloc)(void) [[clang::noalloc]];
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

int f2(void);
// redeclaration with a stronger constraint is OK.
int f2(void) [[clang::nolock]]; // expected-note {{previous declaration is here}}
int f2(void) { return 42; } // expected-warning {{attribute 'nolock' on function does not match previous declaration}}

int f3(void);
// redeclaration with a stronger constraint is OK.
int f3(void) [[clang::noalloc]]; // expected-note {{previous declaration is here}}
int f3(void) { return 42; } // expected-warning {{attribute 'noalloc' on function does not match previous declaration}}

#endif // TEMP_DISABLE
