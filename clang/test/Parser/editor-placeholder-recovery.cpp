// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c++11 -fallow-editor-placeholders -DSUPPRESS -verify %s

struct Struct {
public:
    void method(Struct &x);
};

struct <#struct name#> {
  int <#field-name#>;
#ifndef SUPPRESS
  // expected-error@-3 {{editor placeholder in source file}}
  // expected-error@-3 {{editor placeholder in source file}}
#endif
};

typename <#typename#>::<#name#>;
decltype(<#expression#>) foobar;
typedef <#type#> <#name#>;
#ifndef SUPPRESS
  // expected-error@-4 2 {{editor placeholder in source file}}
  // expected-error@-4 {{editor placeholder in source file}}
  // expected-error@-4 2 {{editor placeholder in source file}}
#endif

namespace <#identifier#> {
  <#declarations#>
#ifndef SUPPRESS
  // expected-error@-3 {{editor placeholder in source file}}
  // expected-error@-3 {{editor placeholder in source file}}
#endif

}

using <#qualifier#>::<#name#>;
#ifndef SUPPRESS
  // expected-error@-2 2 {{editor placeholder in source file}}
#endif

void avoidPlaceholderErrors(Struct &obj) {
    static_cast< <#type#> >(<#expression#>);
    while (<#condition#>) {
        <#statements#>
    }
    obj.method(<#Struct &x#>);
#ifndef SUPPRESS
  // expected-error@-6 2 {{editor placeholder in source file}}
  // expected-error@-6 {{editor placeholder in source file}}
  // expected-error@-6 {{editor placeholder in source file}}
  // expected-error@-5 {{editor placeholder in source file}}
#endif
    switch (<#expression#>) {
        case <#constant#>:
            <#statements#>
#ifndef SUPPRESS
  // expected-error@-4 {{editor placeholder in source file}}
  // expected-error@-4 {{editor placeholder in source file}}
  // expected-error@-4 {{editor placeholder in source file}}
#endif
            break;

        default:
            break;
    }
}

void Struct::method(<#Struct &x#>, noSupressionHere) { // expected-error {{unknown type name 'noSupressionHere'}}
#ifndef SUPPRESS
  // expected-error@-2 {{editor placeholder in source file}}
#endif
}

void handleTrigraph() {
  <??=placeholder#> // expected-error {{expected expression}} expected-error {{expected expression}} expected-warning {{trigraph converted to '#' character}}
}
