// RUN: %clang_analyze_cc1 -x c++ -analyzer-checker=core -analyzer-output=text -verify %s

typedef unsigned char uint8_t;

#define UINT8_MAX 255
#define TCP_MAXWIN 65535

uint8_t get_uint8_max() {
  uint8_t rcvscale = UINT8_MAX; // expected-note{{'rcvscale' initialized to 255}}
  return rcvscale; // expected-note{{Returning the value 255 (loaded from 'rcvscale')}}
}

void shift_by_undefined_value() {
  uint8_t shift_amount = get_uint8_max(); // expected-note{{'shift_amount' initialized to 255}}
  // expected-note@-1{{Calling 'get_uint8_max'}}
  // expected-note@-2{{Returning from 'get_uint8_max'}}
  (void)(TCP_MAXWIN << shift_amount); // expected-warning{{Left shift by '255' overflows the capacity of 'int'}}
  // expected-note@-1{{The result of left shift is undefined because the right operand '255' is not smaller than 32, the capacity of 'int'}}
}

namespace array_index_tracking {
void consume(int);

int getIndex(int x) {
  int a;
  if (x > 0) // expected-note {{Assuming 'x' is > 0}}
             // expected-note@-1 {{Taking true branch}}
    a = 3; // expected-note {{The value 3 is assigned to 'a'}}
  else
    a = 2;
  return a; // expected-note {{Returning the value 3 (loaded from 'a')}}
}

int getInt();

void testArrayIndexTracking() {
  int arr[10];

  for (int i = 0; i < 3; ++i)
    // expected-note@-1 3{{Loop condition is true.  Entering loop body}}
    // expected-note@-2 {{Loop condition is false. Execution continues on line 43}}
    arr[i] = 0;
  int x = getInt();
  int n = getIndex(x); // expected-note {{Calling 'getIndex'}}
                       // expected-note@-1 {{Returning from 'getIndex'}}
                       // expected-note@-2 {{'n' initialized to 3}}
  consume(arr[n]);
  // expected-note@-1 {{1st function call argument is an uninitialized value}}
  // expected-warning@-2{{1st function call argument is an uninitialized value}}
}
} // end of namespace array_index_tracking

namespace multi_array_index_tracking {
void consume(int);

int getIndex(int x) {
  int a;
  if (x > 0) // expected-note {{Assuming 'x' is > 0}}
             // expected-note@-1 {{Taking true branch}}
    a = 3; // expected-note {{The value 3 is assigned to 'a'}}
  else
    a = 2;
  return a; // expected-note {{Returning the value 3 (loaded from 'a')}}
}

int getInt();

void testArrayIndexTracking() {
  int arr[2][10];

  for (int i = 0; i < 3; ++i)
    // expected-note@-1 3{{Loop condition is true.  Entering loop body}}
    // expected-note@-2 {{Loop condition is false. Execution continues on line 75}}
    arr[1][i] = 0;
  int x = getInt();
  int n = getIndex(x); // expected-note {{Calling 'getIndex'}}
                       // expected-note@-1 {{Returning from 'getIndex'}}
                       // expected-note@-2 {{'n' initialized to 3}}
  consume(arr[1][n]);
  // expected-note@-1 {{1st function call argument is an uninitialized value}}
  // expected-warning@-2{{1st function call argument is an uninitialized value}}
}
} // end of namespace mulit_array_index_tracking
