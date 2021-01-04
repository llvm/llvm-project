// RUN: %clang_cc1 -fsyntax-only -fcxx-exceptions -Wreturn-std-move -Wreturn-std-move-in-c++11 -std=c++14 -verify %s
// RUN: %clang_cc1 -fsyntax-only -fcxx-exceptions -Wreturn-std-move -Wreturn-std-move-in-c++11 -std=c++14 -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

// definitions for std::move
namespace std {
inline namespace foo {
template <class T> struct remove_reference { typedef T type; };
template <class T> struct remove_reference<T&> { typedef T type; };
template <class T> struct remove_reference<T&&> { typedef T type; };

template <class T> typename remove_reference<T>::type &&move(T &&t);
} // namespace foo
} // namespace std

struct Instrument {
    Instrument() {}
    Instrument(Instrument&&) { /* MOVE */ }
    Instrument(const Instrument&) { /* COPY */ }
};
struct ConvertFromBase { Instrument i; };
struct ConvertFromDerived { Instrument i; };
struct Base {
    Instrument i;
    operator ConvertFromBase() const& { return ConvertFromBase{i}; }
    operator ConvertFromBase() && { return ConvertFromBase{std::move(i)}; }
};
struct Derived : public Base {
    operator ConvertFromDerived() const& { return ConvertFromDerived{i}; }
    operator ConvertFromDerived() && { return ConvertFromDerived{std::move(i)}; }
};
struct ConstructFromBase {
    Instrument i;
    ConstructFromBase(const Base& b): i(b.i) {}
    ConstructFromBase(Base&& b): i(std::move(b.i)) {}
};
struct ConstructFromDerived {
    Instrument i;
    ConstructFromDerived(const Derived& d): i(d.i) {}
    ConstructFromDerived(Derived&& d): i(std::move(d.i)) {}
};

struct TrivialInstrument {
    int i = 42;
};
struct ConvertFromTrivialBase { TrivialInstrument i; };
struct ConvertFromTrivialDerived { TrivialInstrument i; };
struct TrivialBase {
    TrivialInstrument i;
    operator ConvertFromTrivialBase() const& { return ConvertFromTrivialBase{i}; }
    operator ConvertFromTrivialBase() && { return ConvertFromTrivialBase{std::move(i)}; }
};
struct TrivialDerived : public TrivialBase {
    operator ConvertFromTrivialDerived() const& { return ConvertFromTrivialDerived{i}; }
    operator ConvertFromTrivialDerived() && { return ConvertFromTrivialDerived{std::move(i)}; }
};
struct ConstructFromTrivialBase {
    TrivialInstrument i;
    ConstructFromTrivialBase(const TrivialBase& b): i(b.i) {}
    ConstructFromTrivialBase(TrivialBase&& b): i(std::move(b.i)) {}
};
struct ConstructFromTrivialDerived {
    TrivialInstrument i;
    ConstructFromTrivialDerived(const TrivialDerived& d): i(d.i) {}
    ConstructFromTrivialDerived(TrivialDerived&& d): i(std::move(d.i)) {}
};

Derived test1() {
    Derived d1;
    return d1;  // ok
}
Base test2() {
    Derived d2;
    return d2;  // e1
    // expected-warning@-1{{will be copied despite being returned by name}}
    // expected-note@-2{{to avoid copying}}
    // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:12-[[@LINE-3]]:14}:"std::move(d2)"
}
ConstructFromDerived test3() {
    Derived d3;
    return d3;  // e2-cxx11
    // expected-warning@-1{{would have been copied despite being returned by name}}
    // expected-note@-2{{to avoid copying on older compilers}}
    // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:12-[[@LINE-3]]:14}:"std::move(d3)"
}
ConstructFromBase test4() {
    Derived d4;
    return d4;  // e3
    // expected-warning@-1{{will be copied despite being returned by name}}
    // expected-note@-2{{to avoid copying}}
    // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:12-[[@LINE-3]]:14}:"std::move(d4)"
}
ConvertFromDerived test5() {
    Derived d5;
    return d5;  // e4
    // expected-warning@-1{{will be copied despite being returned by name}}
    // expected-note@-2{{to avoid copying}}
    // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:12-[[@LINE-3]]:14}:"std::move(d5)"
}
ConvertFromBase test6() {
    Derived d6;
    return d6;  // e5
    // expected-warning@-1{{will be copied despite being returned by name}}
    // expected-note@-2{{to avoid copying}}
    // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:12-[[@LINE-3]]:14}:"std::move(d6)"
}

// These test cases should not produce the warning.
Derived ok1() { Derived d; return d; }
Base ok2() { Derived d; return static_cast<Derived&&>(d); }
ConstructFromDerived ok3() { Derived d; return static_cast<Derived&&>(d); }
ConstructFromBase ok4() { Derived d; return static_cast<Derived&&>(d); }
ConvertFromDerived ok5() { Derived d; return static_cast<Derived&&>(d); }
ConvertFromBase ok6() { Derived d; return static_cast<Derived&&>(d); }

// If the target is an lvalue reference, assume it's not safe to move from.
Derived ok_plvalue1(Derived& d) { return d; }
Base ok_plvalue2(Derived& d) { return d; }
ConstructFromDerived ok_plvalue3(const Derived& d) { return d; }
ConstructFromBase ok_plvalue4(Derived& d) { return d; }
ConvertFromDerived ok_plvalue5(Derived& d) { return d; }
ConvertFromBase ok_plvalue6(Derived& d) { return d; }

Derived ok_lvalue1(Derived *p) { Derived& d = *p; return d; }
Base ok_lvalue2(Derived *p) { Derived& d = *p; return d; }
ConstructFromDerived ok_lvalue3(Derived *p) { const Derived& d = *p; return d; }
ConstructFromBase ok_lvalue4(Derived *p) { Derived& d = *p; return d; }
ConvertFromDerived ok_lvalue5(Derived *p) { Derived& d = *p; return d; }
ConvertFromBase ok_lvalue6(Derived *p) { Derived& d = *p; return d; }

// If the target is a global, assume it's not safe to move from.
static Derived global_d;
Derived ok_global1() { return global_d; }
Base ok_global2() { return global_d; }
ConstructFromDerived ok_global3() { return global_d; }
ConstructFromBase ok_global4() { return global_d; }
ConvertFromDerived ok_global5() { return global_d; }
ConvertFromBase ok_global6() { return global_d; }

// If the target's copy constructor is trivial, assume the programmer doesn't care.
TrivialDerived ok_trivial1(TrivialDerived d) { return d; }
TrivialBase ok_trivial2(TrivialDerived d) { return d; }
ConstructFromTrivialDerived ok_trivial3(TrivialDerived d) { return d; }
ConstructFromTrivialBase ok_trivial4(TrivialDerived d) { return d; }
ConvertFromTrivialDerived ok_trivial5(TrivialDerived d) { return d; }
ConvertFromTrivialBase ok_trivial6(TrivialDerived d) { return d; }

// If the target is a parameter, do apply the diagnostic.
Derived testParam1(Derived d) { return d; }
Base testParam2(Derived d) {
    return d;  // e6
    // expected-warning@-1{{will be copied despite being returned by name}}
    // expected-note@-2{{to avoid copying}}
    // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:12-[[@LINE-3]]:13}:"std::move(d)"
}
ConstructFromDerived testParam3(Derived d) {
    return d;  // e7-cxx11
    // expected-warning@-1{{would have been copied despite being returned by name}}
    // expected-note@-2{{to avoid copying on older compilers}}
    // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:12-[[@LINE-3]]:13}:"std::move(d)"
}
ConstructFromBase testParam4(Derived d) {
    return d;  // e8
    // expected-warning@-1{{will be copied despite being returned by name}}
    // expected-note@-2{{to avoid copying}}
    // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:12-[[@LINE-3]]:13}:"std::move(d)"
}
ConvertFromDerived testParam5(Derived d) {
    return d;  // e9
    // expected-warning@-1{{will be copied despite being returned by name}}
    // expected-note@-2{{to avoid copying}}
    // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:12-[[@LINE-3]]:13}:"std::move(d)"
}
ConvertFromBase testParam6(Derived d) {
    return d;  // e10
    // expected-warning@-1{{will be copied despite being returned by name}}
    // expected-note@-2{{to avoid copying}}
    // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:12-[[@LINE-3]]:13}:"std::move(d)"
}

// If the target is an rvalue reference parameter, do apply the diagnostic.
Derived testRParam1(Derived&& d) {
    return d;  // e11
    // expected-warning@-1{{will be copied despite being returned by name}}
    // expected-note@-2{{to avoid copying}}
    // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:12-[[@LINE-3]]:13}:"std::move(d)"
}
Base testRParam2(Derived&& d) {
    return d;  // e12
    // expected-warning@-1{{will be copied despite being returned by name}}
    // expected-note@-2{{to avoid copying}}
    // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:12-[[@LINE-3]]:13}:"std::move(d)"
}
ConstructFromDerived testRParam3(Derived&& d) {
    return d;  // e13
    // expected-warning@-1{{will be copied despite being returned by name}}
    // expected-note@-2{{to avoid copying}}
    // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:12-[[@LINE-3]]:13}:"std::move(d)"
}
ConstructFromBase testRParam4(Derived&& d) {
    return d;  // e14
    // expected-warning@-1{{will be copied despite being returned by name}}
    // expected-note@-2{{to avoid copying}}
    // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:12-[[@LINE-3]]:13}:"std::move(d)"
}
ConvertFromDerived testRParam5(Derived&& d) {
    return d;  // e15
    // expected-warning@-1{{will be copied despite being returned by name}}
    // expected-note@-2{{to avoid copying}}
    // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:12-[[@LINE-3]]:13}:"std::move(d)"
}
ConvertFromBase testRParam6(Derived&& d) {
    return d;  // e16
    // expected-warning@-1{{will be copied despite being returned by name}}
    // expected-note@-2{{to avoid copying}}
    // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:12-[[@LINE-3]]:13}:"std::move(d)"
}

// But if the return type is a reference type, then moving would be wrong.
Derived& testRetRef1(Derived&& d) { return d; }
Base& testRetRef2(Derived&& d) { return d; }
auto&& testRetRef3(Derived&& d) { return d; }
decltype(auto) testRetRef4(Derived&& d) { return (d); }

// As long as we're checking parentheses, make sure parentheses don't disable the warning.
Base testParens1() {
    Derived d;
    return (d);  // e17
    // expected-warning@-1{{will be copied despite being returned by name}}
    // expected-note@-2{{to avoid copying}}
    // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:12-[[@LINE-3]]:15}:"std::move(d)"
}
ConstructFromDerived testParens2() {
    Derived d;
    return (d);  // e18-cxx11
    // expected-warning@-1{{would have been copied despite being returned by name}}
    // expected-note@-2{{to avoid copying}}
    // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:12-[[@LINE-3]]:15}:"std::move(d)"
}


// If the target is a catch-handler parameter, do apply the diagnostic.
void throw_derived();
Derived testEParam1() {
    try { throw_derived(); } catch (Derived d) { return d; }  // e19
    // expected-warning@-1{{will be copied despite being returned by name}}
    // expected-note@-2{{to avoid copying}}
    // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:57-[[@LINE-3]]:58}:"std::move(d)"
    __builtin_unreachable();
}
Base testEParam2() {
    try { throw_derived(); } catch (Derived d) { return d; }  // e20
    // expected-warning@-1{{will be copied despite being returned by name}}
    // expected-note@-2{{to avoid copying}}
    // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:57-[[@LINE-3]]:58}:"std::move(d)"
    __builtin_unreachable();
}
ConstructFromDerived testEParam3() {
    try { throw_derived(); } catch (Derived d) { return d; }  // e21
    // expected-warning@-1{{will be copied despite being returned by name}}
    // expected-note@-2{{to avoid copying}}
    // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:57-[[@LINE-3]]:58}:"std::move(d)"
    __builtin_unreachable();
}
ConstructFromBase testEParam4() {
    try { throw_derived(); } catch (Derived d) { return d; }  // e22
    // expected-warning@-1{{will be copied despite being returned by name}}
    // expected-note@-2{{to avoid copying}}
    // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:57-[[@LINE-3]]:58}:"std::move(d)"
    __builtin_unreachable();
}
ConvertFromDerived testEParam5() {
    try { throw_derived(); } catch (Derived d) { return d; }  // e23
    // expected-warning@-1{{will be copied despite being returned by name}}
    // expected-note@-2{{to avoid copying}}
    // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:57-[[@LINE-3]]:58}:"std::move(d)"
    __builtin_unreachable();
}
ConvertFromBase testEParam6() {
    try { throw_derived(); } catch (Derived d) { return d; }  // e24
    // expected-warning@-1{{will be copied despite being returned by name}}
    // expected-note@-2{{to avoid copying}}
    // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:57-[[@LINE-3]]:58}:"std::move(d)"
    __builtin_unreachable();
}

// If the exception variable is an lvalue reference, we cannot be sure
// that we own it; it is extremely contrived, but possible, for this to
// be a reference to an exception object that was thrown via
// `std::rethrow_exception(xp)` in Thread A, and meanwhile somebody else
// has got a copy of `xp` in Thread B, so that moving out of this object
// in Thread A would be observable (and racy) with respect to Thread B.
// Therefore assume it's not safe to move from.
Derived ok_REParam1() { try { throw_derived(); } catch (Derived& d) { return d; } __builtin_unreachable(); }
Base ok_REParam2() { try { throw_derived(); } catch (Derived& d) { return d; } __builtin_unreachable(); }
ConstructFromDerived ok_REParam3() { try { throw_derived(); } catch (Derived& d) { return d; } __builtin_unreachable(); }
ConstructFromBase ok_REParam4() { try { throw_derived(); } catch (Derived& d) { return d; } __builtin_unreachable(); }
ConvertFromDerived ok_REParam5() { try { throw_derived(); } catch (Derived& d) { return d; } __builtin_unreachable(); }
ConvertFromBase ok_REParam6() { try { throw_derived(); } catch (Derived& d) { return d; } __builtin_unreachable(); }

Derived ok_CEParam1() { try { throw_derived(); } catch (const Derived& d) { return d; } __builtin_unreachable(); }
Base ok_CEParam2() { try { throw_derived(); } catch (const Derived& d) { return d; } __builtin_unreachable(); }
ConstructFromDerived ok_CEParam3() { try { throw_derived(); } catch (const Derived& d) { return d; } __builtin_unreachable(); }
ConstructFromBase ok_CEParam4() { try { throw_derived(); } catch (const Derived& d) { return d; } __builtin_unreachable(); }
ConvertFromDerived ok_CEParam5() { try { throw_derived(); } catch (const Derived& d) { return d; } __builtin_unreachable(); }
ConvertFromBase ok_CEParam6() { try { throw_derived(); } catch (const Derived& d) { return d; } __builtin_unreachable(); }

// If rvalue overload resolution would find a copy constructor anyway,
// or if the copy constructor actually selected is trivial, then don't warn.
struct TriviallyCopyable {};
struct OnlyCopyable {
    OnlyCopyable() = default;
    OnlyCopyable(const OnlyCopyable&) {}
};

TriviallyCopyable ok_copy1() { TriviallyCopyable c; return c; }
OnlyCopyable ok_copy2() { OnlyCopyable c; return c; }
TriviallyCopyable ok_copyparam1(TriviallyCopyable c) { return c; }
OnlyCopyable ok_copyparam2(OnlyCopyable c) { return c; }

void test_throw1(Derived&& d) {
    throw d;  // e25
    // expected-warning@-1{{will be copied despite being thrown by name}}
    // expected-note@-2{{to avoid copying}}
    // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:11-[[@LINE-3]]:12}:"std::move(d)"
}

void ok_throw1() {
  Derived d;
  throw d;
}
void ok_throw2(Derived d) { throw d; }
void ok_throw3(Derived &d) { throw d; }
void ok_throw4(Derived d) { throw std::move(d); }
void ok_throw5(Derived &d) { throw std::move(d); }
void ok_throw6(Derived &d) { throw static_cast<Derived &&>(d); }
void ok_throw7(TriviallyCopyable d) { throw d; }
void ok_throw8(OnlyCopyable d) { throw d; }

namespace test_delete {
struct Base {
  Base();
  Base(Base &&) = delete;
  Base(Base const &);
};

struct Derived : public Base {};

Base test_ok() {
  Derived d;
  return d;
}
} // namespace test_delete
