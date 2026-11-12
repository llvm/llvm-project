// RUN: %clang_cc1 -fsyntax-only -verify -std=c++17 %s

struct Weak {
    [[gnu::weak]]void weak_method();
};
static_assert([](){ return &Weak::weak_method != nullptr; }()); // expected-error {{static assertion expression is not an integral constant expression}} \
                                                                // expected-note {{comparison against pointer to weak member 'Weak::weak_method' can only be performed at runtime}} \
                                                                // expected-note {{in call to}}
