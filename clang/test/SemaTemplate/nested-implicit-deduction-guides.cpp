// RUN: %clang_cc1 -std=c++20 -verify %s

template<class T> struct S {
    template<class U> struct N {
        N(T) {}
        N(T, U) {}
        template<class V> N(V, U) {}
    };
};

S<int>::N x{"a", 1};
using T = decltype(x);
using T = S<int>::N<int>;

template<class X> struct default_ftd_argument {
    template<class Y> struct B {
        template<class W = X, class Z = Y, class V = Z, int I = 0> B(Y);
    };
};

default_ftd_argument<int>::B default_arg("a");
using DefaultArg = decltype(default_arg);
using DefaultArg = default_ftd_argument<int>::B<const char *>;

template<bool> struct test;
template<class X> struct non_type_param {
    template<class Y> struct B {
        B(Y);
        template<class Z, test<Z::value> = 0> B(Z);
    };
};

non_type_param<int>::B ntp = 5;
using NonTypeParam = decltype(ntp);
using NonTypeParam = non_type_param<int>::B<int>;

template<typename A, typename T>
concept True = true;

template<typename T>
concept False = false;

template<class X> struct concepts {
    template<class Y> struct B {
        template<class K = X, True<K> Z> B(Y, Z);
    };
};

concepts<int>::B cc(1, 3);
using Concepts = decltype(cc);
using Concepts = concepts<int>::B<int>;

template<class X> struct requires_clause {
    template<class Y> struct B {
        template<class Z> requires true
            B(Y, Z);
    };
};

requires_clause<int>::B req(1, 2);
using RC = decltype(req);
using RC = requires_clause<int>::B<int>;

template<typename X> struct nested_init_list {
    template<True<X> Y>
    struct B {
        X x;
        Y y;
    };

    template<False F>
    struct concept_fail { // #INIT_LIST_INNER_INVALID
        X x;
        F f;
    };
};

nested_init_list<int>::B nil {1, 2};
using NIL = decltype(nil);
using NIL = nested_init_list<int>::B<int>;

// expected-error@+1 {{no viable constructor or deduction guide for deduction of template arguments of 'concept_fail'}}
nested_init_list<int>::concept_fail nil_invalid{1, ""};
// expected-note@#INIT_LIST_INNER_INVALID {{candidate template ignored: substitution failure [with F = const char *]: constraints not satisfied for class template 'concept_fail' [with F = const char *]}}
// expected-note@#INIT_LIST_INNER_INVALID {{candidate function template not viable: requires 1 argument, but 2 were provided}}
// expected-note@#INIT_LIST_INNER_INVALID {{candidate function template not viable: requires 0 arguments, but 2 were provided}}
