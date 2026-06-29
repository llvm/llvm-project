// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 %s
// expected-no-diagnostics

// This is a regression test that verifies handling a mix of using-declarations
// from dependent and non-dependent base classes does not cause name lookup
// to crash when a dependent entity cannot be converted to a TemplateDecl.

namespace GH174951 {

template<typename A>
class X {
public:
    template<typename T>
    void execute(int a) {}
};

class Y {
public:
    void execute(int a) {}
};

template<typename A>
class Exec : public X<A>, public Y {
public:
    using X<A>::execute;
    using Y::execute;
    
    void validate() {
        // In C++20 the 'execute' here is followed by '<'.
        // The lookup result will include an UnresolvedUsingValueDecl (from X<A>)
        // and a UsingShadowDecl (from Y).
        execute<int>(42); 
        execute(42);
    }
};

} // namespace GH174951
