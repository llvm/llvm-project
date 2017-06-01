// RUN: %clang_cc1 -fsyntax-only -verify -std=c++1y -triple x86_64-linux-gnu %s

// If there is a preceding declaration of the entity *in the same scope* in
// which the bound was specified, an omitted array bound is taken to be the
// same as in that earlier declaration

// rdar://13535367
namespace test0 {
  extern "C" int array[];
  void declare() { extern int array[100]; }
  int value1 = sizeof(array); // expected-error {{invalid application of 'sizeof' to an incomplete type 'int []'}}
  extern "C" int array[];
  int value2 = sizeof(array); // expected-error {{invalid application of 'sizeof' to an incomplete type 'int []'}}
}

namespace test1 {
  extern "C" int array[];
  void test() {
    { extern int array[100]; }
    extern int array[];
    int x = sizeof(array); // expected-error {{invalid application of 'sizeof' to an incomplete type 'int []'}}
  }
}

namespace test2 {
  void declare() { extern int array[100]; }
  extern int array[];
  int value = sizeof(array); // expected-error {{invalid application of 'sizeof' to an incomplete type 'int []'}}
}

namespace test3 {
  void test() {
    { extern int array[100]; }
    extern int array[];
    int x = sizeof(array); // expected-error {{invalid application of 'sizeof' to an incomplete type 'int []'}}
  }
}

namespace test4 {
  extern int array[];
  void test() {
    extern int array[100];
    int x = sizeof(array);
  }
  int y = sizeof(array); // expected-error {{invalid application of 'sizeof' to an incomplete type 'int []'}}
}

namespace test5 {
  void test() {
    extern int array[100];
    extern int array[];
    int x = sizeof(array);
  }
}

namespace test6 {
  void test() {
    extern int array[100];
    {
      extern int array[];
      int x = sizeof(array); // expected-error {{invalid application of 'sizeof' to an incomplete type 'int []'}}
    }
    int y = sizeof(array);
    extern int array[];
    int z = sizeof(array);
  }
}

namespace test7 {
  extern int array[100];
  void test() {
    extern int array[];
    int x = sizeof(array); // expected-error {{invalid application of 'sizeof' to an incomplete type 'int []'}}
  }
  int y = sizeof(array);
  extern int array[];
  int z = sizeof(array);
}

namespace test8 {
  extern int array[];
  void test() {
    extern int array[100];
    int x = sizeof(array);
  }
  int y = sizeof(array); // expected-error {{invalid application of 'sizeof' to an incomplete type 'int []'}}
  extern int array[];
  int z = sizeof(array); // expected-error {{invalid application of 'sizeof' to an incomplete type 'int []'}}
}

namespace dependent {
  template<typename T> void f() {
    extern int arr1[];
    extern T arr1;
    extern T arr2;
    extern int arr2[];
    static_assert(sizeof(arr1) == 12, "");
    static_assert(sizeof(arr2) == 12, "");

    // Use a failing test to ensure the type isn't considered dependent.
    static_assert(sizeof(arr2) == 13, ""); // expected-error {{failed}}
  }

  void g() { f<int[3]>(); } // expected-note {{in instantiation of}}

  template<typename T> void h1() {
    extern T arr3;
    {
      int arr3;
      {
        extern int arr3[];
        // Detected in template definition.
        (void)sizeof(arr3); // expected-error {{incomplete}}
      }
    }
  }

  template<typename T> void h2() {
    extern int arr4[3];
    {
      int arr4;
      {
        extern T arr4;
        // Detected in template instantiation.
        (void)sizeof(arr4); // expected-error {{incomplete}}
      }
    }
  }

  void i() {
    h1<int[3]>();
    h2<int[]>(); // expected-note {{in instantiation of}}
  }

  int arr5[3];
  template<typename T> void j() {
    extern T arr5;
    extern T arr6;
    (void)sizeof(arr5); // expected-error {{incomplete}}
    (void)sizeof(arr6); // expected-error {{incomplete}}
  }
  int arr6[3];

  void k() { j<int[]>(); } // expected-note {{in instantiation of}}

  template<typename T, typename U> void l() {
    extern T arrX; // expected-note {{previous}}
    extern U arrX; // expected-error {{different type: 'int [4]' vs 'int [3]'}}
    (void)sizeof(arrX); // expected-error {{incomplete}}
  }

  void m() {
    l<int[], int[3]>(); // ok
    l<int[3], int[]>(); // ok
    l<int[3], int[3]>(); // ok
    l<int[3], int[4]>(); // expected-note {{in instantiation of}}
    l<int[], int[]>(); // expected-note {{in instantiation of}}
  }

  template<typename T> void n() {
    extern T n_var; // expected-error {{redeclaration of 'n_var' with a different type: 'double' vs 'int'}} expected-note {{previous}}
    extern T n_fn(); // expected-error {{functions that differ only in their return type cannot be overloaded}} expected-note {{previous}}
  }
  template void n<int>();
  template void n<double>(); // expected-note {{in instantiation of}}

  template<typename T> void o() {
    extern T o_var; // expected-note {{previous}}
    extern T o_fn(); // expected-note {{previous}}
  }
  template void o<int>();
  float o_var; // expected-error {{redefinition of 'o_var' with a different type: 'float' vs 'int'}}
  float o_fn(); // expected-error {{functions that differ only in their return type cannot be overloaded}}

  int p_var;
  int p_fn();
  template<typename T> void p() {
    extern T p_var;
    extern T p_fn();
  }
}

namespace use_outside_ns {
  namespace A {
    extern int a[3];
    extern int b[];
    extern int c[3];
    void f() {
      extern int a[];
      extern int b[3];
    }
    template<typename T> void x() {
      extern T c;
      extern T d;
    }
    extern int d[3];
    template void x<int[]>();
  }
  int w = sizeof(A::a);
  int x = sizeof(A::b); // expected-error {{incomplete}}
  int y = sizeof(A::c);
  int z = sizeof(A::d);
  namespace A {
    int g() { return sizeof(a); }
    int h() { return sizeof(b); } // expected-error {{incomplete}}
    int i() { return sizeof(c); }
    int j() { return sizeof(d); }
  }
}

extern int arr[];
void f1() { extern int arr[2]; } // expected-note {{previous}}
void f2() { extern int arr[3]; } // expected-error {{different type: 'int [3]' vs 'int [2]'}}
