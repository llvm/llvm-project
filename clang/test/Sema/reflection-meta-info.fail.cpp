// RUN: %clang_cc1 %s -std=c++26 -freflection -fsyntax-only -verify
// RUN: %clang_cc1 %s -std=c++26 -freflection -fexperimental-new-constant-interpreter -fsyntax-only -verify

using info = decltype(^^int);

struct X { int a; }; // expected-note {{candidate constructor (the implicit copy constructor) not viable: no known conversion from 'std::meta::info' to 'const X' for 1st argument}} \
                     // expected-note {{candidate constructor (the implicit move constructor) not viable: no known conversion from 'std::meta::info' to 'X' for 1st argument}} \
                     // expected-note {{candidate constructor (the implicit default constructor) not viable: requires 0 arguments, but 1 was provided}}
template <typename T = X, auto ptm = &X::a>
constexpr auto ptmOp = ((T)(^^int)).*ptm; // expected-error {{no matching conversion for C-style cast from 'std::meta::info' to 'X'}}
constexpr auto var = ptmOp<>; // expected-note {{in instantiation of variable template specialization 'ptmOp' requested here}}

static_assert(^^int == ^^float); // expected-error {{static assertion failed due to requirement '^^int == ^^float'}}
static_assert(info{} == ^^int); // expected-error {{static assertion failed due to requirement 'std::meta::info{} == ^^int'}}

consteval void test()
{
    (^^char)++; // expected-error {{cannot increment value of type 'std::meta::info'}}
    (^^short)++; // expected-error {{cannot increment value of type 'std::meta::info'}}
    (^^int)++; // expected-error {{cannot increment value of type 'std::meta::info'}}
    (^^char)--; // expected-error {{cannot decrement value of type 'std::meta::info'}}
    (^^short)--; // expected-error {{cannot decrement value of type 'std::meta::info'}}
    (^^int)--; // expected-error {{cannot decrement value of type 'std::meta::info'}}

    ++(^^char); // expected-error {{cannot increment value of type 'std::meta::info'}}
    ++(^^short); // expected-error {{cannot increment value of type 'std::meta::info'}}
    ++(^^int); // expected-error {{cannot increment value of type 'std::meta::info'}}
    --(^^char); // expected-error {{cannot decrement value of type 'std::meta::info'}}
    --(^^short); // expected-error {{cannot decrement value of type 'std::meta::info'}}
    --(^^int); // expected-error {{cannot decrement value of type 'std::meta::info'}}

    ~(^^int);  // expected-error {{invalid argument type 'std::meta::info' to unary expression}}
    !(^^float);  // expected-error {{invalid argument type 'std::meta::info' to unary expression}}

    (^^int) + (^^int); // expected-error {{invalid operands to binary expression ('std::meta::info' and 'std::meta::info')}}
    (^^double) - (^^float); // expected-error {{invalid operands to binary expression ('std::meta::info' and 'std::meta::info')}}
    (^^char) * (^^short); // expected-error {{invalid operands to binary expression ('std::meta::info' and 'std::meta::info')}}
    (^^long) / (^^long); // expected-error {{invalid operands to binary expression ('std::meta::info' and 'std::meta::info')}}

    (^^int) & (^^int); // expected-error {{invalid operands to binary expression ('std::meta::info' and 'std::meta::info')}}
    (^^int) | (^^int); // expected-error {{invalid operands to binary expression ('std::meta::info' and 'std::meta::info')}}
    (^^int) ^ (^^int); // expected-error {{invalid operands to binary expression ('std::meta::info' and 'std::meta::info')}}

    (^^int) < (^^int); // expected-error {{invalid operands to binary expression ('std::meta::info' and 'std::meta::info')}}
    (^^int) > (^^int); // expected-error {{invalid operands to binary expression ('std::meta::info' and 'std::meta::info')}}
    (^^int) <= (^^int); // expected-error {{invalid operands to binary expression ('std::meta::info' and 'std::meta::info')}}
    (^^int) >= (^^int); // expected-error {{invalid operands to binary expression ('std::meta::info' and 'std::meta::info')}}
    (^^int) <=>(^^int); // expected-error {{invalid operands to binary expression ('std::meta::info' and 'std::meta::info')}}

    // expected-error@+2 {{value of type 'std::meta::info' is not contextually convertible to 'bool'}}
    // expected-error@+1 {{invalid operands to binary expression ('std::meta::info' and 'std::meta::info')}}
    (^^int) && (^^int);
    // expected-error@+2 {{value of type 'std::meta::info' is not contextually convertible to 'bool'}}
    // expected-error@+1 {{invalid operands to binary expression ('std::meta::info' and 'std::meta::info')}}
    (^^int) || (^^int);
}
