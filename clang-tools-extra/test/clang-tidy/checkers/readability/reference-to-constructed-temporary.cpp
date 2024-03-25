// RUN: %check_clang_tidy -std=c++11-or-later %s readability-reference-to-constructed-temporary %t

struct WithConstructor
{
    WithConstructor(int, int);
};

struct WithoutConstructor
{
    int a, b;
};

void test()
{
// CHECK-MESSAGES: :[[@LINE+1]]:27: warning: reference variable 'tmp1' extends the lifetime of a just-constructed temporary object 'const WithConstructor', consider changing reference to value [readability-reference-to-constructed-temporary]
   const WithConstructor& tmp1{1,2};

// CHECK-MESSAGES: :[[@LINE+1]]:30: warning: reference variable 'tmp2' extends the lifetime of a just-constructed temporary object 'const WithoutConstructor', consider changing reference to value [readability-reference-to-constructed-temporary]
   const WithoutConstructor& tmp2{1,2};


// CHECK-MESSAGES: :[[@LINE+1]]:22: warning: reference variable 'tmp3' extends the lifetime of a just-constructed temporary object 'WithConstructor', consider changing reference to value [readability-reference-to-constructed-temporary]
   WithConstructor&& tmp3{1,2};

// CHECK-MESSAGES: :[[@LINE+1]]:25: warning: reference variable 'tmp4' extends the lifetime of a just-constructed temporary object 'WithoutConstructor', consider changing reference to value [readability-reference-to-constructed-temporary]
   WithoutConstructor&& tmp4{1,2};

   WithConstructor tmp5{1,2};
   WithoutConstructor tmp6{1,2};
}
