// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-output text \
// RUN:   -verify %s

void test_no_overflow_note(int a, int b)
{
   int res;

   if (__builtin_add_overflow(a, b, &res)) // expected-note {{Assuming no overflow}}
                                           // expected-note@-1 {{Taking false branch}}
     return;

   if (res) { // expected-note {{Assuming 'res' is not equal to 0}}
              // expected-note@-1 {{Taking true branch}}
     int *ptr = 0; // expected-note {{'ptr' initialized to a null pointer value}}
     int var = *(int *) ptr; //expected-warning {{Dereference of null pointer}}
                             //expected-note@-1 {{Dereference of null pointer}}
   }
}

void test_overflow_note(int a, int b)
{
   int res; // expected-note{{'res' declared without an initial value}}

   if (__builtin_add_overflow(a, b, &res)) { // expected-note {{Assuming overflow}}
                                             // expected-note@-1 {{Taking true branch}}
     int var = res; // expected-warning{{Assigned value is garbage or undefined}}
                    // expected-note@-1 {{Assigned value is garbage or undefined}}
     return;
   }
}
