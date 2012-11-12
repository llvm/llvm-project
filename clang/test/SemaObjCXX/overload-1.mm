// RUN: %clang_cc1 -fsyntax-only -verify %s

@protocol Proto1 @end

@protocol Proto2 @end

void f(id<Proto1> *) { }		// expected-note {{previous definition is here}}

void f(id<Proto1, Proto2> *) { }	// expected-error {{conflicting types for 'f'}}

void f(Class<Proto1> *) { }		// expected-note {{previous definition is here}}

void f(Class<Proto1, Proto2> *) { }	// expected-error {{conflicting types for 'f'}}

@interface I @end

void f(I<Proto1> *) { }		// expected-note {{previous definition is here}}

void f(I<Proto1, Proto2> *) { }		// expected-error {{conflicting types for 'f'}}

@interface I1 @end

void f1(I<Proto1> *) { }

void f1(I1<Proto1, Proto2> *) { }
