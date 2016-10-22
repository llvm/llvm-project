// RUN: %clang_cc1 -std=c++1z -verify %s

void f() noexcept;
void (&r)() = f;
void (&s)() noexcept = r; // expected-error {{cannot bind}}

void (&cond1)() noexcept = true ? r : f; // expected-error {{cannot bind}}
void (&cond2)() noexcept = true ? f : r; // expected-error {{cannot bind}}
// FIXME: Strictly, the rules in p4 don't allow this, because the operand types
// are not of the same type other than cv-qualifiers, but we consider that to
// be a defect, and instead allow reference-compatible types here.
void (&cond3)() = true ? r : f;
void (&cond4)() = true ? f : r;
