// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++26 -triple x86_64-gnu-linux

template <bool Leak>
struct RAIIBase {
    constexpr RAIIBase(const char* in) {
        s = __builtin_strlen(in);
        d = new char[s + 1]; // expected-note 4{{allocation performed here was not deallocated}}
        for(int i = 0 ; i < s; i++)
            d[i] = in[i];
    }
    int s;
    char* d;
    constexpr unsigned long size() const {
        return s;
    }
    constexpr const char* data() const {
        return d;
    }
    constexpr ~RAIIBase() {
        if constexpr(!Leak)
            delete[] d;
    }
};

using RAII = RAIIBase<false>;
using RAIILeak = RAIIBase<true>;

void test_leaks(int i) {
    asm((RAII("nop")) : (RAII("+ir")) (i) : (RAII("g")) (i) : (RAII("memory")));
    asm((RAIILeak("nop"))); // expected-error {{the expression in this asm operand must be produced by a constant expression}}
    asm((RAII("nop"))
        : (RAIILeak("+ir")) (i) // expected-error {{the expression in this asm operand must be produced by a constant expression}}
        ::
    );
    asm((RAII("nop"))
        : (RAII("+ir")) (i)
        : (RAIILeak("g")) (i) // expected-error {{the expression in this asm operand must be produced by a constant expression}}
        :
    );
    asm((RAII("nop"))
        : (RAII("+ir")) (i)
        : (RAII("g")) (i)
        : (RAIILeak("memory")) // expected-error {{the expression in this asm operand must be produced by a constant expression}}
    );
}

struct NotAString{};
struct MessageInvalidSize {
    constexpr unsigned long size(int) const; // expected-note {{'size' declared here}}
    constexpr const char* data() const;
};
struct MessageInvalidData {
    constexpr unsigned long size() const;
    constexpr const char* data(int) const; // expected-note {{'data' declared here}}
};


struct WMessage {
    constexpr unsigned long long size() const {return 0;};
    constexpr const wchar_t* data() const {return L"";}
};

struct string_view {
  int S;
  const char* D;
  constexpr string_view() : S(0), D(0){}
  constexpr string_view(const char* Str) : S(__builtin_strlen(Str)), D(Str) {}
  constexpr string_view(int Size, const char* Str) : S(Size), D(Str) {}
  constexpr int size() const {
      return S;
  }
  constexpr const char* data() const {
      return D;
  }
};


void f() {
    asm(("")); // expected-error {{the expression in this asm operand must be a string literal or an object with 'data()' and 'size()' member functions}}
    asm((NotAString{})); // expected-error {{the string object in this asm operand is missing 'data()' and 'size()' member functions}}
    asm((MessageInvalidData{})); // expected-error {{the expression in this asm operand must have a 'data()' member function returning an object convertible to 'const char *'}} \
                                 // expected-error {{too few arguments to function call, expected 1, have 0}}
    asm((MessageInvalidSize{})); // expected-error {{the expression in this asm operand must have a 'size()' member function returning an object convertible to 'std::size_t'}} \
                                 // expected-error {{too few arguments to function call, expected 1, have 0}}

    asm((WMessage{})); // expected-error {{value of type 'const wchar_t *' is not implicitly convertible to 'const char *'}} \
                       // expected-error {{the expression in this asm operand must have a 'data()' member function returning an object convertible to 'const char *'}}
}

template <typename... U>
void test_packs() {
    asm((U{})); // expected-error {{expression contains unexpanded parameter pack 'U'}}
    asm("" : (U{})); // expected-error {{expression contains unexpanded parameter pack 'U'}}
    asm("" :: (U{})); // expected-error {{expression contains unexpanded parameter pack 'U'}}
    asm("" ::: (U{})); // expected-error {{expression contains unexpanded parameter pack 'U'}}
}

template <typename T>
void test_dependent1(int i) {
    asm((T{})); // #err-int
    asm("" : (T{"+g"})(i)); // #err-int2
    asm("" :: (T{"g"})(i)); // #err-int3
    asm("" ::: (T{"memory"})); // #err-int4
}

template void test_dependent1<int>(int);
// expected-note@-1 {{in instantiation of function template specialization}}
// expected-error@#err-int {{the expression in this asm operand must be a string literal or an object with 'data()' and 'size()' member functions}}
// expected-error@#err-int2 {{cannot initialize a value of type 'int' with an lvalue of type 'const char[3]'}}
// expected-error@#err-int3 {{cannot initialize a value of type 'int' with an lvalue of type 'const char[2]'}}
// expected-error@#err-int4 {{cannot initialize a value of type 'int' with an lvalue of type 'const char[7]'}}

template void test_dependent1<string_view>(int);


template <typename T>
void test_dependent2(int i) {
    asm("" : (T{"g"})(i)); // #err-invalid1
    asm("" :: (T{"+g"})(i)); // #err-invalid2
    asm("" ::: (T{"foo"})); // #err-invalid3
}
template void test_dependent2<string_view>(int);
// expected-note@-1 {{in instantiation of function template specialization}}
// expected-error@#err-invalid1 {{invalid output constraint 'g' in asm}}
// expected-error@#err-invalid2 {{invalid input constraint '+g' in asm}}
// expected-error@#err-invalid3 {{unknown register name 'foo' in asm}}

