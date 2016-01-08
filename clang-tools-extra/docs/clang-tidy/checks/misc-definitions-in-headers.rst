misc-definitions-in-headers
===========================

Finds non-extern non-inline function and variable definitions in header files, which can lead to potential ODR violations.

.. code:: c++
   // Foo.h
   int a = 1; // Warning.
   extern int d; // OK: extern variable.

   namespace N {
     int e = 2; // Warning.
   }

   // Internal linkage variable definitions are ignored for now.
   // Although these might also cause ODR violations, we can be less certain and
   // should try to keep the false-positive rate down.
   static int b = 1;
   const int c = 1;

   // Warning.
   int g() {
     return 1;
   }

   // OK: inline function definition.
   inline int e() {
     return 1;
   }

   class A {
    public:
     int f1() { return 1; } // OK: inline member function definition.
     int f2();
   };

   int A::f2() { return 1; } // Warning.
