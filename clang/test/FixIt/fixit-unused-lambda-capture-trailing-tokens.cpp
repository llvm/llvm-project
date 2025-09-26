// RUN: cp %s %t
// RUN: %clang_cc1 -x c++ -Wunused-lambda-capture -Wno-unused-value -std=c++1z -fixit %t
// RUN: grep -v CHECK %t | FileCheck %s
// RUN: %clang_cc1 -x c++ -Wunused-lambda-capture -Wno-unused-value -std=c++1z -fsyntax-only %s 2>&1 | FileCheck --match-full-lines --strict-whitespace --check-prefix POINTER %s


#define MACRO_CAPTURE(...) __VA_ARGS__
#define M_A a
#define M_B b

int main() {
    int a = 0, b = 0, c = 0, d = 0;
    auto F0 = [a, &b]() mutable {
    // CHECK: auto F0 = [a]()
         // POINTER:{{^.*}}|     auto F0 = [a, &b]() mutable {
    // POINTER-NEXT:{{^.*}}|                 ~~~^

        a++;
    };

    auto F1 = [&a, &b]() {
    // CHECK: auto F1 = []() {
         // POINTER:{{^.*}}|     auto F1 = [&a, &b]() {
    // POINTER-NEXT:{{^.*}}|                ~^~
         // POINTER:{{^.*}}|     auto F1 = [&a, &b]() {
    // POINTER-NEXT:{{^.*}}|                  ~~~^
    };

    auto F2 = [&a, b]() {
    // CHECK: auto F2 = []() {
         // POINTER:{{^.*}}|     auto F2 = [&a, b]() {
    // POINTER-NEXT:{{^.*}}|                ~^~
         // POINTER:{{^.*}}|     auto F2 = [&a, b]() {
    // POINTER-NEXT:{{^.*}}|                  ~~^
    };

    auto F3 = [&a,
         &b]() {
    // CHECK: auto F3 = []() {
         // POINTER:{{^.*}}|     auto F3 = [&a,
    // POINTER-NEXT:{{^.*}}|                ~^~
    // POINTER-NEXT:{{^.*}}|          &b]() {
         // POINTER:{{^.*}}|     auto F3 = [&a,
    // POINTER-NEXT:{{^.*}}|                  ~
    // POINTER-NEXT:{{^.*}}|          &b]() {
    // POINTER-NEXT:{{^.*}}|          ~^

    };

    auto F4 = [&a
        , &b]() {
    // CHECK: auto F4 = []() {
         // POINTER:{{^.*}}|     auto F4 = [&a
    // POINTER-NEXT:{{^.*}}|                ~^
    // POINTER-NEXT:{{^.*}}|         , &b]() {
    // POINTER-NEXT:{{^.*}}|         ~
         // POINTER:{{^.*}}|     auto F4 = [&a
    // POINTER-NEXT:{{^.*}}|                  {{$}}
    // POINTER-NEXT:{{^.*}}|         , &b]() {
    // POINTER-NEXT:{{^.*}}|         ~~~^
    };
    auto F5 = [&a ,&b]()  {
    // CHECK: auto F5 = []() {
         // POINTER:{{^.*}}|     auto F5 = [&a ,&b]()  {
    // POINTER-NEXT:{{^.*}}|                ~^~~
         // POINTER:{{^.*}}|     auto F5 = [&a ,&b]()  {
    // POINTER-NEXT:{{^.*}}|                   ~~^
    };

    auto F0a = [a, &b]() mutable {
    // CHECK: auto F0a = [a]() mutable {
         // POINTER:{{^.*}}|     auto F0a = [a, &b]() mutable {
    // POINTER-NEXT:{{^.*}}|                  ~~~^
        a++;
    };

    auto F1a = [&a, &b]() {
    // CHECK: auto F1a = [&a]() {
         // POINTER:{{^.*}}|     auto F1a = [&a, &b]() {
    // POINTER-NEXT:{{^.*}}|                   ~~~^
        a++;
    };

    auto F2a = [&a, b]() {
    // CHECK: auto F2a = [&a]() {
         // POINTER:{{^.*}}|     auto F2a = [&a, b]() {
    // POINTER-NEXT:{{^.*}}|                   ~~^
        a++;
    };

    auto F3a = [&a,
         &b]() {
    // CHECK: auto F3a = [&a]() {
         // POINTER:{{^.*}}|     auto F3a = [&a,
    // POINTER-NEXT:{{^.*}}|                   ~
    // POINTER-NEXT:{{^.*}}|          &b]() {
    // POINTER-NEXT:{{^.*}}|          ~^
        a++;
    };

    auto F4a = [&a
        , &b]() {
    // CHECK: auto F4a = [&a]() {
         // POINTER:{{^.*}}|     auto F4a = [&a
    // POINTER-NEXT:{{^.*}}|                   {{$}}
    // POINTER-NEXT:{{^.*}}|         , &b]() {
    // POINTER-NEXT:{{^.*}}|         ~~~^
        a++;
    };

    auto F5a = [&a ,&b]() {
    // CHECK: auto F5a = [&a]() {
         // POINTER:{{^.*}}|     auto F5a = [&a ,&b]() {
    // POINTER-NEXT:{{^.*}}|                    ~~^
        a++;
    };
    auto F0b = [a, &b]() mutable {
    // CHECK: auto F0b = [ &b]() mutable
         // POINTER:{{^.*}}|     auto F0b = [a, &b]() mutable {
    // POINTER-NEXT:{{^.*}}|                 ^~

        b++;
    };

    auto F1b = [&a, &b]() {
    // CHECK: auto F1b = [ &b]() {
         // POINTER:{{^.*}}|     auto F1b = [&a, &b]() {
    // POINTER-NEXT:{{^.*}}|                 ~^~
        b++;
    };

    auto F2b = [&a, b]() mutable {
    // CHECK: auto F2b = [ b]() mutable {
         // POINTER:{{^.*}}|     auto F2b = [&a, b]() mutable {
    // POINTER-NEXT:{{^.*}}|                 ~^~
        b++;
    };

    auto F3b = [&a,
         &b]() {
    // CHECK: auto F3b = [ &b]() {
         // POINTER:{{^.*}}|     auto F3b = [&a,
    // POINTER-NEXT:{{^.*}}|                 ~^~
    // POINTER-NEXT:{{^.*}}|          &b]() {
        b++;
    };

    auto F4b = [&a
        , &b]() {
    // CHECK: auto F4b = [ &b]() {
         // POINTER:{{^.*}}|     auto F4b = [&a
    // POINTER-NEXT:{{^.*}}|                 ~^
    // POINTER-NEXT:{{^.*}}|         , &b]() {
    // POINTER-NEXT:{{^.*}}|         ~

        b++;
    };
    auto F5b = [&a ,&b]() {
    // CHECK: auto F5b = [&b]() {
         // POINTER:{{^.*}}|     auto F5b = [&a ,&b]() {
    // POINTER-NEXT:{{^.*}}|                 ~^~~
        b++;
    };

    auto F6 = [&a, &b, &c]() {
    // CHECK: auto F6 = [&a, &b]() {
         // POINTER:{{^.*}}|     auto F6 = [&a, &b, &c]() {
    // POINTER-NEXT:{{^.*}}|                      ~~~^
        a++;
        b++;
    };
    auto F7 = [&a, &b, &c]() {
    // CHECK: auto F7 = [&a, &c]() {
         // POINTER:{{^.*}}|     auto F7 = [&a, &b, &c]() {
    // POINTER-NEXT:{{^.*}}|                  ~~~^
        a++;
        c++;
    };
    auto F8 = [&a, &b, &c]() {
    // CHECK: auto F8 = [ &b, &c]() {
         // POINTER:{{^.*}}|     auto F8 = [&a, &b, &c]() {
    // POINTER-NEXT:{{^.*}}|                ~^~
        b++;
        c++;
    };
    auto F9 = [&a, &b    , &c]() {
    // CHECK: auto F9 = [&a   , &c]() {
         // POINTER:{{^.*}}|     auto F9 = [&a, &b    , &c]() {
    // POINTER-NEXT:{{^.*}}|                  ~~~^
        a++;
        c++;
    };
    auto F10 = [&a,
         &b, &c]() {
    // CHECK: auto F10 = [&a, &c]() {
         // POINTER:{{^.*}}|     auto F10 = [&a,
    // POINTER-NEXT:{{^.*}}|                   ~
    // POINTER-NEXT:{{^.*}}|          &b, &c]() {
    // POINTER-NEXT:{{^.*}}|          ~^
        a++;
        c++;
    };
    auto F11 = [&a,  &b  ,
         &c]() {
    // CHECK: auto F11 = [&a ,
    // CHECK-NEXT:      &c]() {
         // POINTER:{{^.*}}|     auto F11 = [&a,  &b  ,
    // POINTER-NEXT:{{^.*}}|                   ~~~~^
        a++;
        c++;
    };
    auto F12 = [a = 0,  b  ,
         c]() mutable {
    // CHECK: auto F12 = [ b  ,
    // CHECK-NEXT:     c]() mutable {
         // POINTER:{{^.*}}|     auto F12 = [a = 0,  b  ,
    // POINTER-NEXT:{{^.*}}|                 ^~~~~~
        b++;
        c++;
    };
    auto F13 = [a,  b = 0 ,
         c]() mutable {
    // CHECK: auto F13 = [a ,
    // CHECK-NEXT:     c]() mutable {
         // POINTER:{{^.*}}|     auto F13 = [a,  b = 0 ,
    // POINTER-NEXT:{{^.*}}|                  ~~~^~~~~
        a++;
        c++;
    };
    auto F14 = [a,  b ,
         c
        = 0]() mutable {
    // CHECK: auto F14 = [a,  b]() mutable {
         // POINTER:{{^.*}}|     auto F14 = [a,  b ,
    // POINTER-NEXT:{{^.*}}|                       ~
    // POINTER-NEXT:{{^.*}}|          c
    // POINTER-NEXT:{{^.*}}|          ^
    // POINTER-NEXT:{{^.*}}|         = 0]() mutable {
    // POINTER-NEXT:{{^.*}}|         ~~~
        a++;
        b++;
    };

    // We want to remove everything including the comment
    // as well as the comma following the capture of `a`
    auto F15 = [&a /* comment */, &b]() {
    // CHECK: auto F15 = [ &b]() {
         // POINTER:{{^.*}}|     auto F15 = [&a /* comment */, &b]() {
    // POINTER-NEXT:{{^.*}}|                 ~^~~~~~~~~~~~~~~~
        b++;
    };

    // The comment preceding the first capture remains. This is more
    // by design of the fixit logic than anything else, but we should
    // consider the preceding comment might actually be a comment for
    // the entire capture set, so maybe we do want it to hang around
    auto F16 = [/* comment */ &a , &b]() {
    // CHECK: auto F16 = [/* comment */ &b]() {
         // POINTER:{{^.*}}|     auto F16 = [/* comment */ &a , &b]() {
    // POINTER-NEXT:{{^.*}}|                               ~^~~
        b++;
    };

    auto F16b = [&a ,    /* comment */ &b]() {
    // CHECK: auto F16b = [ /* comment */ &b]() {
         // POINTER:{{^.*}}|     auto F16b = [&a ,    /* comment */ &b]() {
    // POINTER-NEXT:{{^.*}}|                  ~^~~
        b++;
    };

    auto F17 = [&a /* comment */, &b]() {
    // CHECK: auto F17 = [&a]() {
         // POINTER:{{^.*}}|     auto F17 = [&a /* comment */, &b]() {
    // POINTER-NEXT:{{^.*}}|                    ~~~~~~~~~~~~~~~~^

        a++;
    };

    auto F18 = [&a , /* comment */ &b]() {
    // CHECK: auto F18 = [&a]() {
         // POINTER:{{^.*}}|     auto F18 = [&a , /* comment */ &b]() {
    // POINTER-NEXT:{{^.*}}|                    ~~~~~~~~~~~~~~~~~^

        a++;
    };

    auto F19 = [&a /* comment */, &b /* comment */]() {
    // CHECK: auto F19 = [&a /* comment */]() {
         // POINTER:{{^.*}}|     auto F19 = [&a /* comment */, &b /* comment */]() {
    // POINTER-NEXT:{{^.*}}|                    ~~~~~~~~~~~~~~~~^

        a++;
    };

    auto F20 = [MACRO_CAPTURE(&a, &b)]() {
    // CHECK: auto F20 = [MACRO_CAPTURE(&a, &b)]() {
         // POINTER:{{^.*}}|     auto F20 = [MACRO_CAPTURE(&a, &b)]() {
    // POINTER-NEXT:{{^.*}}|                               ~^
          // POINTER:{{^.*}}|     auto F20 = [MACRO_CAPTURE(&a, &b)]() {
    // POINTER-NEXT:{{^.*}}|                                   ~^
    };

    auto F21 = [MACRO_CAPTURE(&a), &b]() {
    // CHECK: auto F21 = []() {
         // POINTER:{{^.*}}|     auto F21 = [MACRO_CAPTURE(&a), &b]() {
    // POINTER-NEXT:{{^.*}}|                 ~~~~~~~~~~~~~~~^~~
         // POINTER:{{^.*}}|     auto F21 = [MACRO_CAPTURE(&a), &b]() {
    // POINTER-NEXT:{{^.*}}|                                  ~~~^
    };

    auto F22 = [MACRO_CAPTURE(&a,) &b]() {
    // CHECK: auto F22 = [MACRO_CAPTURE(&a,) &b]() {
         // POINTER:{{^.*}}|     auto F22 = [MACRO_CAPTURE(&a,) &b]() {
    // POINTER-NEXT:{{^.*}}|                               ~^
         // POINTER:{{^.*}}|     auto F22 = [MACRO_CAPTURE(&a,) &b]() {
    // POINTER-NEXT:{{^.*}}|                                    ~^
    };
    auto F23 = [&a MACRO_CAPTURE(, &b)]() {
    // CHECK: auto F23 = [&a]() {
         // POINTER:{{^.*}}|     auto F23 = [&a MACRO_CAPTURE(, &b)]() {
    // POINTER-NEXT:{{^.*}}|                 ~^
         // POINTER:{{^.*}}|     auto F23 = [&a MACRO_CAPTURE(, &b)]() {
    // POINTER-NEXT:{{^.*}}|                    ~~~~~~~~~~~~~~~~~^~
    };
    auto F24 = [&a, MACRO_CAPTURE(&b)]() {
    // CHECK: auto F24 = []() {
         // POINTER:{{^.*}}|     auto F24 = [&a, MACRO_CAPTURE(&b)]() {
    // POINTER-NEXT:{{^.*}}|                 ~^~
         // POINTER:{{^.*}}|     auto F24 = [&a, MACRO_CAPTURE(&b)]() {
    // POINTER-NEXT:{{^.*}}|                   ~~~~~~~~~~~~~~~~~^~
    };

    auto F20a = [MACRO_CAPTURE(&a, &b)]() {
    // CHECK: auto F20a = [MACRO_CAPTURE(&a, &b)]() {
         // POINTER:{{^.*}}|     auto F20a = [MACRO_CAPTURE(&a, &b)]() {
    // POINTER-NEXT:{{^.*}}|                                    ~^
      a++;
    };

    auto F21a = [MACRO_CAPTURE(&a), &b]() {
    // CHECK: auto F21a = [MACRO_CAPTURE(&a)]() {
         // POINTER:{{^.*}}|     auto F21a = [MACRO_CAPTURE(&a), &b]() {
    // POINTER-NEXT:{{^.*}}|                                   ~~~^
      a++;
    };

    auto F22a = [MACRO_CAPTURE(&a,) &b]() {
    // CHECK: auto F22a = [MACRO_CAPTURE(&a,) &b]() {
         // POINTER:{{^.*}}|     auto F22a = [MACRO_CAPTURE(&a,) &b]() {
    // POINTER-NEXT:{{^.*}}|                                     ~^
      a++;
    };
    auto F23a = [&a MACRO_CAPTURE(, &b)]() {
    // CHECK: auto F23a = [&a]() {
         // POINTER:{{^.*}}|     auto F23a = [&a MACRO_CAPTURE(, &b)]() {
    // POINTER-NEXT:{{^.*}}|                     ~~~~~~~~~~~~~~~~~^~
      a++;
    };
    auto F24a = [&a, MACRO_CAPTURE(&b)]() {
    // CHECK: auto F24a = [&a]() {
         // POINTER:{{^.*}}|     auto F24a = [&a, MACRO_CAPTURE(&b)]() {
    // POINTER-NEXT:{{^.*}}|                    ~~~~~~~~~~~~~~~~~^~
      a++;
    };
    auto F20b = [MACRO_CAPTURE(&a, &b)]() {
    // CHECK: auto F20b = [MACRO_CAPTURE(&a, &b)]() {
         // POINTER:{{^.*}}|     auto F20b = [MACRO_CAPTURE(&a, &b)]() {
    // POINTER-NEXT:{{^.*}}|                                ~^
      b++;
    };

    auto F21b = [MACRO_CAPTURE(&a), &b]() {
    // CHECK: auto F21b = [ &b]() {
         // POINTER:{{^.*}}|     auto F21b = [MACRO_CAPTURE(&a), &b]() {
    // POINTER-NEXT:{{^.*}}|                  ~~~~~~~~~~~~~~~^~~
      b++;
    };

    auto F22b = [MACRO_CAPTURE(&a,) &b]() {
    // CHECK: auto F22b = [MACRO_CAPTURE(&a,) &b]() {
         // POINTER:{{^.*}}|     auto F22b = [MACRO_CAPTURE(&a,) &b]() {
    // POINTER-NEXT:{{^.*}}|                                ~^
      b++;
    };
    auto F23b = [&a MACRO_CAPTURE(, &b)]() {
    // CHECK: auto F23b = [&a MACRO_CAPTURE(, &b)]() {
         // POINTER:{{^.*}}|     auto F23b = [&a MACRO_CAPTURE(, &b)]() {
    // POINTER-NEXT:{{^.*}}|                  ~^
      b++;
    };
    auto F24b = [&a, MACRO_CAPTURE(&b)]() {
    // CHECK: auto F24b = [ MACRO_CAPTURE(&b)]() {
         // POINTER:{{^.*}}|     auto F24b = [&a, MACRO_CAPTURE(&b)]() {
    // POINTER-NEXT:{{^.*}}|                  ~^~
      b++;
    };

    auto F25ma = [&M_A, &b]() {
    // CHECK:     auto F25ma = []() {
         // POINTER:{{^.*}}|     auto F25ma = [&M_A, &b]() {
    // POINTER-NEXT:{{^.*}}|                   ~^~~~
         // POINTER:{{^.*}}|     auto F25ma = [&M_A, &b]() {
    // POINTER-NEXT:{{^.*}}|                       ~~~^
    };
    auto F25mb = [&a, &M_B]() {
    // CHECK: auto F25mb = []() {
         // POINTER:{{^.*}}|     auto F25mb = [&a, &M_B]() {
    // POINTER-NEXT:{{^.*}}|                   ~^~
         // POINTER:{{^.*}}|     auto F25mb = [&a, &M_B]() {
    // POINTER-NEXT:{{^.*}}|                     ~~~^~~
    };

    auto F25mab = [&M_A, &b]() {
    // CHECK: auto F25mab = [ &b]() {
         // POINTER:{{^.*}}|     auto F25mab = [&M_A, &b]() {
    // POINTER-NEXT:{{^.*}}|                    ~^~~~
        b++;
    };
    auto F25amb = [&a, &M_B]() {
    //CHECK: auto F25amb = []() {
         // POINTER:{{^.*}}|     auto F25amb = [&a, &M_B]() {
    // POINTER-NEXT:{{^.*}}|                    ~^~
         // POINTER:{{^.*}}|     auto F25amb = [&a, &M_B]() {
    // POINTER-NEXT:{{^.*}}|                      ~~~^~~
    };

    auto F26 = [&a, &b, &c, &d]() {
    // CHECK: auto F26 = [&a, &b]() {
         // POINTER:{{^.*}}|     auto F26 = [&a, &b, &c, &d]() {
    // POINTER-NEXT:{{^.*}}|                       ~~~^
         // POINTER:{{^.*}}|     auto F26 = [&a, &b, &c, &d]() {
    // POINTER-NEXT:{{^.*}}|                           ~~~^

        (void)a;
        (void)b;
    };

    auto F27 = [&a, &b, &c, &d]() {
    // CHECK: auto F27 = [&a, &c]() {
         // POINTER:{{^.*}}|     auto F27 = [&a, &b, &c, &d]() {
    // POINTER-NEXT:{{^.*}}|                   ~~~^
         // POINTER:{{^.*}}|     auto F27 = [&a, &b, &c, &d]() {
    // POINTER-NEXT:{{^.*}}|                           ~~~^

        (void)a;
        (void)c;
    };

    auto F28 = [&a, &b, &c, &d]() {
        // CHECK: auto F28 = [&a, &d]() {
         // POINTER:{{^.*}}|     auto F28 = [&a, &b, &c, &d]() {
    // POINTER-NEXT:{{^.*}}|                   ~~~^
         // POINTER:{{^.*}}|     auto F28 = [&a, &b, &c, &d]() {
    // POINTER-NEXT:{{^.*}}|                       ~~~^

        (void)a;
        (void)d;
    };

    auto F29 = [&a, &b, &c, &d]() {
    // CHECK: auto F29 = [ &b, &c]() {
         // POINTER:{{^.*}}|     auto F29 = [&a, &b, &c, &d]() {
    // POINTER-NEXT:{{^.*}}|                 ~^~
         // POINTER:{{^.*}}|     auto F29 = [&a, &b, &c, &d]() {
    // POINTER-NEXT:{{^.*}}|                           ~~~^

        (void)b;
        (void)c;
    };

    auto F30 = [&a, &b, &c, &d]() {
    // CHECK: auto F30 = [ &b, &d]() {
         // POINTER:{{^.*}}|     auto F30 = [&a, &b, &c, &d]() {
    // POINTER-NEXT:{{^.*}}|                 ~^~
         // POINTER:{{^.*}}|     auto F30 = [&a, &b, &c, &d]() {
    // POINTER-NEXT:{{^.*}}|                       ~~~^

        (void)b;
        (void)d;
    };

    auto F31 = [&a, &b, &c, &d]() {
    // CHECK: auto F31 = [ &c, &d]() {
         // POINTER:{{^.*}}|     auto F31 = [&a, &b, &c, &d]() {
    // POINTER-NEXT:{{^.*}}|                 ~^~
         // POINTER:{{^.*}}|     auto F31 = [&a, &b, &c, &d]() {
    // POINTER-NEXT:{{^.*}}|                     ~^~
        (void)c;
        (void)d;
    };
}
