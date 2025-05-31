// RUN: cp %s %t
// RUN: %clang_cc1 -x c++ -Wunused-lambda-capture -Wno-unused-value -std=c++1z -fixit %t
// RUN: grep -v CHECK %t | FileCheck %s


#define MACRO_CAPTURE(...) __VA_ARGS__
int main() {
    int a = 0, b = 0, c = 0;
    auto F0 = [a, &b]() mutable {
    // CHECK: auto F0 = [a]()
        a++;
    };

    auto F1 = [&a, &b]() {
    // CHECK: auto F1 = []() {
    };

    auto F2 = [&a, b]() {
    // CHECK: auto F2 = []() {
    };

    auto F3 = [&a,
         &b]() {
    // CHECK: auto F3 = []() {
    };

    auto F4 = [&a
        , &b]() {
    // CHECK: auto F4 = []() {
    };
    auto F5 = [&a ,&b]()  {
    // CHECK: auto F5 = []() {
    };

    auto F0a = [a, &b]() mutable {
    // CHECK: auto F0a = [a]() mutable {
        a++;
    };

    auto F1a = [&a, &b]() {
    // CHECK: auto F1a = [&a]() {
        a++;
    };

    auto F2a = [&a, b]() {
    // CHECK: auto F2a = [&a]() {
        a++;
    };

    auto F3a = [&a,
         &b]() {
    // CHECK: auto F3a = [&a]() {
        a++;
    };

    auto F4a = [&a
        , &b]() {
    // CHECK: auto F4a = [&a]() {
        a++;
    };

    auto F5a = [&a ,&b]() {
    // CHECK: auto F5a = [&a]() {
        a++;
    };
    auto F0b = [a, &b]() mutable {
    // CHECK: auto F0b = [ &b]() mutable
        b++;
    };

    auto F1b = [&a, &b]() {
    // CHECK: auto F1b = [ &b]() {
        b++;
    };

    auto F2b = [&a, b]() mutable {
    // CHECK: auto F2b = [ b]() mutable {
        b++;
    };

    auto F3b = [&a,
         &b]() {
    // CHECK: auto F3b = [ &b]() {
        b++;
    };

    auto F4b = [&a
        , &b]() {
    // CHECK: auto F4b = [ &b]() {
        b++;
    };
    auto F5b = [&a ,&b]() {
    // CHECK: auto F5b = [&b]() {
        b++;
    };

    auto F6 = [&a, &b, &c]() {
    // CHECK: auto F6 = [&a, &b]() {
        a++;
        b++;
    };
    auto F7 = [&a, &b, &c]() {
    // CHECK: auto F7 = [&a, &c]() {
        a++;
        c++;
    };
    auto F8 = [&a, &b, &c]() {
    // CHECK: auto F8 = [ &b, &c]() {
        b++;
        c++;
    };
    auto F9 = [&a, &b    , &c]() {
    // CHECK: auto F9 = [&a   , &c]() {
        a++;
        c++;
    };
    auto F10 = [&a,
         &b, &c]() {
    // CHECK: auto F10 = [&a, &c]() {
        a++;
        c++;
    };
    auto F11 = [&a,  &b  ,
         &c]() {
    // CHECK: auto F11 = [&a ,
    // CHECK-NEXT:      &c]() {
        a++;
        c++;
    };
    auto F12 = [a = 0,  b  ,
         c]() mutable {
    // CHECK: auto F12 = [ b  ,
    // CHECK-NEXT:     c]() mutable {
        b++;
        c++;
    };
    auto F13 = [a,  b = 0 ,
         c]() mutable {
    // CHECK: auto F13 = [a ,
    // CHECK-NEXT:     c]() mutable {
        a++;
        c++;
    };
    auto F14 = [a,  b ,
         c
        = 0]() mutable {
    // CHECK: auto F14 = [a,  b]() mutable {
        a++;
        b++;
    };

    // We want to remove everything including the comment
    // as well as the comma following the capture of `a`
    auto F15 = [&a /* comment */, &b]() {
    // CHECK: auto F15 = [ &b]() {
        b++;
    };

    // The comment preceding the first capture remains. This is more
    // by design of the fixit logic than anything else, but we should
    // consider the preceding comment might actually be a comment for
    // the entire capture set, so maybe we do want it to hang around
    auto F16 = [/* comment */ &a , &b]() {
    // CHECK: auto F16 = [/* comment */ &b]() {
        b++;
    };

    auto F16b = [&a ,    /* comment */ &b]() {
    // CHECK: auto F16b = [ /* comment */ &b]() {
        b++;
    };

    auto F17 = [&a /* comment */, &b]() {
    // CHECK: auto F17 = [&a]() {
        a++;
    };

    auto F18 = [&a , /* comment */ &b]() {
    // CHECK: auto F18 = [&a]() {
        a++;
    };
    
    auto F19 = [&a /* comment */, &b /* comment */]() {
    // CHECK: auto F19 = [&a /* comment */]() {
        a++;
    };

    auto F20 = [MACRO_CAPTURE(&a, &b)]() {
    // CHECK: auto F20 = [MACRO_CAPTURE(&a, &b)]() {
    };

    auto F21 = [MACRO_CAPTURE(&a), &b]() {
    // CHECK: auto F21 = []() {
    };

    auto F22 = [MACRO_CAPTURE(&a,) &b]() {
    // CHECK: auto F22 = [MACRO_CAPTURE(&a,) &b]() {
    };
    auto F23 = [&a MACRO_CAPTURE(, &b)]() {
    // CHECK: auto F23 = []() {
    };
    auto F24 = [&a, MACRO_CAPTURE(&b)]() {
    // CHECK: auto F24 = []() {
    };

    auto F20a = [MACRO_CAPTURE(&a, &b)]() {
    // CHECK: auto F20a = [MACRO_CAPTURE(&a, &b)]() {
      a++;
    };

    auto F21a = [MACRO_CAPTURE(&a), &b]() {
    // CHECK: auto F21a = [MACRO_CAPTURE(&a)]() {
      a++;
    };

    auto F22a = [MACRO_CAPTURE(&a,) &b]() {
    // Cauto F22a = [MACRO_CAPTURE(&a,) &b]() {
      a++;
    };
    auto F23a = [&a MACRO_CAPTURE(, &b)]() {
    // CHECK: auto F23a = [&a]() {
      a++;
    };
    auto F24a = [&a, MACRO_CAPTURE(&b)]() {
    // CHECK: auto F24a = [&a]() {
      a++;
    };
    auto F20b = [MACRO_CAPTURE(&a, &b)]() {
    // CHECK: auto F20b = [MACRO_CAPTURE(&a, &b)]() {
      b++;
    };

    auto F21b = [MACRO_CAPTURE(&a), &b]() {
    // CHECK: auto F21b = [ &b]() {
      b++;
    };

    auto F22b = [MACRO_CAPTURE(&a,) &b]() {
    // CHECK: auto F22b = [MACRO_CAPTURE(&a,) &b]() {
      b++;
    };
    auto F23b = [&a MACRO_CAPTURE(, &b)]() {
    // CHECK: auto F23b = [(, &b)]() {
      b++;
    };
    auto F24b = [&a, MACRO_CAPTURE(&b)]() {
    // CHECK: auto F24b = [ MACRO_CAPTURE(&b)]() {
      b++;
    };
}
