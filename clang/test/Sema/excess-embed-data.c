// RUN: %clang_cc1 -std=c23 -fsyntax-only -verify=c %s
// RUN: %clang_cc1 -fsyntax-only -verify=cpp -x c++ -Wno-c23-extensions %s


struct S {
  int arr[3];
};

struct S1 {
  struct S s;
};

void cases(int x) {
  int a[8] = {x, x, x, x, x, x,
#embed __FILE__
    // c-warning@-1{{excess elements in array initializer}}
    // cpp-error@-2{{excess elements in array initializer}}
};
  int b[8] = {
#embed __FILE__
    // c-warning@-1{{excess elements in array initializer}}
    // cpp-error@-2{{excess elements in array initializer}}
};
  int c[3000] = {x, x, x, x, x, x,
#embed __FILE__
  };
 char d[3] = {
#embed __FILE__
    // c-warning@-1{{initializer-string for char array is too long}}
    // cpp-error@-2{{initializer-string for char array is too long}}
  };

char e[3000] = { 1,
#embed __FILE__
};

struct S s = {
#embed __FILE__
    // c-warning@-1{{excess elements in struct initializer}}
    // cpp-error@-2{{excess elements in struct initializer}}
  , x
};

struct S1 s1 = {
#embed __FILE__
    // c-warning@-1{{excess elements in struct initializer}}
    // cpp-error@-2{{excess elements in struct initializer}}
  , x
};
}
