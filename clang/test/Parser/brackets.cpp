// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: cp %s %t
// RUN: not %clang_cc1 -fixit %t -x c++ -DFIXIT
// RUN: %clang_cc1 -fsyntax-only %t -x c++ -DFIXIT
// RUN: not %clang_cc1 -fsyntax-only -fdiagnostics-parseable-fixits -fno-diagnostics-show-line-numbers %s 2>&1 | FileCheck %s -strict-whitespace

void test1() {
  int a[] = {0,1,1,2,3};
  int []b = {0,1,4,9,16};
  // expected-error@-1{{brackets are not allowed here; to declare an array, place the brackets after the name}}
  // CHECK: {{^}}  int []b = {0,1,4,9,16};
  // CHECK: {{^}}      ~~ ^
  // CHECK: {{^}}         []
  // CHECK: fix-it:{{.*}}:{[[@LINE-5]]:7-[[@LINE-5]]:9}:""
  // CHECK: fix-it:{{.*}}:{[[@LINE-6]]:10-[[@LINE-6]]:10}:"[]"

  int c = a[0];
  int d = b[0];  // No undeclared identifier error here.

  int *e = a;
  int *f = b;  // No undeclared identifier error here.

  int[1] g[2];
  // expected-error@-1{{brackets are not allowed here; to declare an array, place the brackets after the name}}
  // CHECK: {{^}}  int[1] g[2];
  // CHECK: {{^}}     ~~~     ^
  // CHECK: {{^}}             [1]
  // CHECK: fix-it:{{.*}}:{[[@LINE-5]]:6-[[@LINE-5]]:9}:""
  // CHECK: fix-it:{{.*}}:{[[@LINE-6]]:14-[[@LINE-6]]:14}:"[1]"
}

void test2() {
  int [3] (*a) = 0;
  // expected-error@-1{{brackets are not allowed here; to declare an array, place the brackets after the name}}
  // CHECK: {{^}}  int [3] (*a) = 0;
  // CHECK: {{^}}      ~~~     ^
  // CHECK: {{^}}              [3]
  // CHECK: fix-it:{{.*}}:{[[@LINE-5]]:7-[[@LINE-5]]:11}:""
  // CHECK: fix-it:{{.*}}:{[[@LINE-6]]:15-[[@LINE-6]]:15}:"[3]"

#ifndef FIXIT
  // Make sure a is corrected to be like type y, instead of like type z.
  int (*b)[3] = a;
  int (*c[3]) = a;  // expected-error{{}}
#endif
}

struct A {
  static int [1][1]x;
  // expected-error@-1{{brackets are not allowed here; to declare an array, place the brackets after the name}}
  // CHECK: {{^}}  static int [1][1]x;
  // CHECK: {{^}}             ~~~~~~ ^
  // CHECK: {{^}}                    [1][1]
  // CHECK: fix-it:{{.*}}:{[[@LINE-5]]:14-[[@LINE-5]]:20}:""
  // CHECK: fix-it:{{.*}}:{[[@LINE-6]]:21-[[@LINE-6]]:21}:"[1][1]"
};

int [1][1]A::x = { {42} };
// expected-error@-1{{brackets are not allowed here; to declare an array, place the brackets after the name}}
// CHECK: {{^}}int [1][1]A::x = { {42} };
// CHECK: {{^}}    ~~~~~~    ^
// CHECK: {{^}}              [1][1]
// CHECK: fix-it:{{.*}}:{[[@LINE-5]]:5-[[@LINE-5]]:11}:""
// CHECK: fix-it:{{.*}}:{[[@LINE-6]]:15-[[@LINE-6]]:15}:"[1][1]"

struct B { static int (*x)[5]; };
int [5] *B::x = 0;
// expected-error@-1{{brackets are not allowed here; to declare an array, place the brackets after the name}}
// CHECK: {{^}}int [5] *B::x = 0;
// CHECK: {{^}}    ~~~      ^
// CHECK: {{^}}        (    )[5]
// CHECK: fix-it:{{.*}}:{[[@LINE-5]]:5-[[@LINE-5]]:9}:""
// CHECK: fix-it:{{.*}}:{[[@LINE-6]]:9-[[@LINE-6]]:9}:"("
// CHECK: fix-it:{{.*}}:{[[@LINE-7]]:14-[[@LINE-7]]:14}:")[5]"

void test3() {
  int [3] *a;
  // expected-error@-1{{brackets are not allowed here; to declare an array, place the brackets after the name}}
  // CHECK: {{^}}  int [3] *a;
  // CHECK: {{^}}      ~~~   ^
  // CHECK: {{^}}          ( )[3]
  // CHECK: fix-it:{{.*}}:{[[@LINE-5]]:7-[[@LINE-5]]:11}:""
  // CHECK: fix-it:{{.*}}:{[[@LINE-6]]:11-[[@LINE-6]]:11}:"("
  // CHECK: fix-it:{{.*}}:{[[@LINE-7]]:13-[[@LINE-7]]:13}:")[3]"

  int (*b)[3] = a;  // no error
}

void test4() {
  int [2] a;
  // expected-error@-1{{brackets are not allowed here; to declare an array, place the brackets after the name}}
  // CHECK: {{^}}  int [2] a;
  // CHECK: {{^}}      ~~~  ^
  // CHECK: {{^}}           [2]
  // CHECK: fix-it:{{.*}}:{[[@LINE-5]]:7-[[@LINE-5]]:11}:""
  // CHECK: fix-it:{{.*}}:{[[@LINE-6]]:12-[[@LINE-6]]:12}:"[2]"

  int [2] &b = a;
  // expected-error@-1{{brackets are not allowed here; to declare an array, place the brackets after the name}}
  // CHECK: {{^}}  int [2] &b = a;
  // CHECK: {{^}}      ~~~   ^
  // CHECK: {{^}}          ( )[2]
  // CHECK: fix-it:{{.*}}:{[[@LINE-5]]:7-[[@LINE-5]]:11}:""
  // CHECK: fix-it:{{.*}}:{[[@LINE-6]]:11-[[@LINE-6]]:11}:"("
  // CHECK: fix-it:{{.*}}:{[[@LINE-7]]:13-[[@LINE-7]]:13}:")[2]"

}

namespace test5 {
#ifndef FIXIT
int [][][];
// expected-error@-1{{expected unqualified-id}}
// CHECK: {{^}}int [][][];
// CHECK: {{^}}    ^

struct C {
  int [];
  // expected-error@-1{{expected member name or ';' after declaration specifiers}}
  // CHECK: {{^}}  int [];
  // CHECK: {{^}}  ~~~ ^
};

#endif
}

namespace test6 {
struct A {
  static int arr[3];
};
int [3] ::test6::A::arr = {1,2,3};
// expected-error@-1{{brackets are not allowed here; to declare an array, place the brackets after the name}}
// CHECK: {{^}}int [3] ::test6::A::arr = {1,2,3};
// CHECK: {{^}}    ~~~                ^
// CHECK: {{^}}                       [3]
// CHECK: fix-it:{{.*}}:{[[@LINE-5]]:5-[[@LINE-5]]:9}:""
// CHECK: fix-it:{{.*}}:{[[@LINE-6]]:24-[[@LINE-6]]:24}:"[3]"

}

namespace test7 {
class A{};
void test() {
  int [3] A::*a;
  // expected-error@-1{{brackets are not allowed here; to declare an array, place the brackets after the name}}
  // CHECK: {{^}}  int [3] A::*a;
  // CHECK: {{^}}      ~~~      ^
  // CHECK: {{^}}          (    )[3]
  // CHECK: fix-it:{{.*}}:{[[@LINE-5]]:7-[[@LINE-5]]:11}:""
  // CHECK: fix-it:{{.*}}:{[[@LINE-6]]:11-[[@LINE-6]]:11}:"("
  // CHECK: fix-it:{{.*}}:{[[@LINE-7]]:16-[[@LINE-7]]:16}:")[3]"
}
}

namespace test8 {
struct A {
  static const char f[];
};
const char[] A::f = "f";
// expected-error@-1{{brackets are not allowed here; to declare an array, place the brackets after the name}}
}

namespace gh147333 {
    template<class T, char fmt>
    constexpr inline auto& to_print_fmt = "";
    template<> constexpr inline char[] to_print_fmt<unsigned, 'x'> = "0x%x";
    // expected-error@-1{{brackets are not allowed here; to declare an array, place the brackets after the name}}

#ifndef FIXIT
    // Further related test cases.

    int[1] operator+();
    // expected-error@-1{{brackets are not allowed here; to declare an array, place the brackets after the name}}
    // expected-error@-2{{function cannot return array type}}
    
    int[1] operator ""_x(unsigned long long);
    // expected-error@-1{{brackets are not allowed here; to declare an array, place the brackets after the name}}
    // expected-error@-2{{function cannot return array type}}
       
    struct A {
        int[1] operator int();
        // expected-error@-1{{brackets are not allowed here; to declare an array, place the brackets after the name}}
        // TODO: The following is too noisy and redundant.
        // expected-error@-3{{conversion function cannot have a return type}}
        // expected-error@-4{{cannot specify any part of a return type in the declaration of a conversion function}}
        // expected-error@-5{{conversion function cannot convert to an array type}}

        int[1] A();
        // expected-error@-1{{brackets are not allowed here; to declare an array, place the brackets after the name}}
        // TODO: The following is too noisy and redundant.
        // expected-error@-3{{function cannot return array type}}
        // expected-error@-4{{constructor cannot have a return type}}
        
        int[1] ~A();
        // expected-error@-1{{brackets are not allowed here; to declare an array, place the brackets after the name}}
        // TODO: This isn't helpful.
        // expected-error@-3{{array has incomplete element type 'void'}}
    };
    
    template<typename T>
    struct B {
        int[1] B<T>();
        // expected-error@-1{{brackets are not allowed here; to declare an array, place the brackets after the name}}
        // TODO: The following is too noisy and redundant.
        // expected-error@-3{{function cannot return array type}}
        // expected-error@-4{{constructor cannot have a return type}}
    };
#endif
}

// CHECK: 32 errors generated.
