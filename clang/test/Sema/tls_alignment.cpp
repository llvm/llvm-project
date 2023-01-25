// TLS variable cannot be aligned to more than 32 bytes on PS4.

// RUN: %clang_cc1 -triple x86_64-scei-ps4 -std=c++17 -fsyntax-only -verify %s


// A non-aligned type.
struct non_aligned_struct {
    int some_data[16]; // 64 bytes of stuff, non aligned.
};

// An aligned type.
struct __attribute__(( aligned(64) )) aligned_struct {
    int some_data[12]; // 48 bytes of stuff, aligned to 64.
};

// A type with an aligned field.
struct  struct_with_aligned_field {
    int some_aligned_data[12] __attribute__(( aligned(64) )); // 48 bytes of stuff, aligned to 64.
};

// A templated type
template <typename>
struct templated_struct {};
// expected-note@-1{{candidate template ignored: couldn't infer template argument ''}}
// expected-note@-2{{candidate function template not viable: requires 1 argument, but 0 were provided}}

// A typedef of the aligned struct.
typedef aligned_struct another_aligned_struct;

// A typedef to redefine a non-aligned struct as aligned.
typedef __attribute__(( aligned(64) )) non_aligned_struct yet_another_aligned_struct;

// Non aligned variable doesn't cause an error.
__thread non_aligned_struct foo;

// Variable aligned because of its type should cause an error.
__thread aligned_struct                    bar; // expected-error{{alignment (64) of thread-local variable}}

// Variable explicitly aligned in the declaration should cause an error.
__thread non_aligned_struct                bar2 __attribute__(( aligned(64) )); // expected-error{{alignment (64) of thread-local variable}}

// Variable aligned because of one of its fields should cause an error.
__thread struct_with_aligned_field         bar3; // expected-error{{alignment (64) of thread-local variable}}

// Variable aligned because of typedef, first case.
__thread another_aligned_struct            bar4; // expected-error{{alignment (64) of thread-local variable}}

// Variable aligned because of typedef, second case.
__thread yet_another_aligned_struct        bar5; // expected-error{{alignment (64) of thread-local variable}}

// No crash for undeduced type.
__thread templated_struct                  bar6; // expected-error{{no viable constructor or deduction guide for deduction of template arguments of 'templated_struct'}}

int baz ()
{
    return foo.some_data[0] + bar.some_data[1] + bar2.some_data[2] +
           bar3.some_aligned_data[3] + bar4.some_data[4] +
           bar5.some_data[5];
}

template<class T> struct templated_tls {
    static __thread T t;
    T other_t __attribute__(( aligned(64) ));
};
 __thread templated_tls<int> blah; // expected-error{{alignment (64) of thread-local variable}}

template <int N>
struct S {
  struct alignas(64) B {};
  struct alignas(N) C {};
  static inline void f() {
    thread_local B b; // expected-error{{alignment (64) of thread-local variable}}
    thread_local C c; // expected-error{{alignment (64) of thread-local variable}}
  }
  template<int J> static inline thread_local int b alignas(J) = J; // expected-error{{alignment (64) of thread-local variable}}
  static int __thread __attribute__((aligned(N))) x; // expected-error{{alignment (64) of thread-local variable}}
};

int blag() {
    // Verify alignment check where the alignment is a template parameter.
    // The check is only performed during instantiation.
    S<64> s_instance; // expected-note{{in instantiation of template class 'S<64>' requested here}}

    // Verify alignment for dependent local variables.
    S<64>::f(); // expected-note{{in instantiation of member function 'S<64>::f' requested here}}

    // Verify alignment check where a dependent type is involved.
    // The check is (correctly) not performed on "t", but the check still is
    // performed on the structure as a whole once it has been instantiated.
    return blah.other_t * 2 + S<64>::b<64>; // expected-note{{in instantiation of static data member 'S<64>::b' requested here}}
}
