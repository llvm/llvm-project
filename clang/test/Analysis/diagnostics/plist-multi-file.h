void foo(int *ptr) {
  *ptr = 1; // expected-warning{{Dereference of null pointer (loaded from variable 'ptr')}}
}
