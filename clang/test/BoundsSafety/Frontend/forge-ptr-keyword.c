
// RUN: %clang_cc1 -x c -verify=c_lang %s
// RUN: %clang_cc1 -x c++ -verify=cpp_lang %s

void Test1() {
    __builtin_unsafe_forge_bidi_indexable(0, sizeof(int));
    // c_lang-error@-1 {{use of unknown builtin '__builtin_unsafe_forge_bidi_indexable'}}
    // cpp_lang-error@-2 {{use of undeclared identifier '__builtin_unsafe_forge_bidi_indexable'}}
}
