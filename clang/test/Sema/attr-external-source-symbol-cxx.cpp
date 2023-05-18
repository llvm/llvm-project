// RUN: %clang_cc1 -fsyntax-only -fblocks -verify %s

template<class T>
class Class {
public:
    [[clang::external_source_symbol(language="Swift", defined_in="module", USR="test", generated_declaration)]]
    void testExternalSourceSymbol();

    // expected-error@+1 {{expected string literal for USR in 'external_source_symbol' attribute}}
    [[clang::external_source_symbol(language="Swift", defined_in="module", USR=T, generated_declaration)]]
    void testExternalSourceSymbol2();
};

template<class T>
void Class<T>::testExternalSourceSymbol() {
}
