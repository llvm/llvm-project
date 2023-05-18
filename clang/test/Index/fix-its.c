// RUN: c-index-test -test-load-source all -fspell-checking %s 2> %t  
// RUN: FileCheck %s < %t
struct X {
  int wibble;
};

#define MACRO(X) X

void f(struct X *x) {
  // CHECK: error: no member named 'wobble' in 'struct X'; did you mean 'wibble'?
  // CHECK: FIX-IT: Replace [13:12 - 13:18] with "wibble"
  // CHECK: note: 'wibble' declared here
  MACRO(x->wobble = 17);
  // CHECK: error: no member named 'wabble' in 'struct X'; did you mean 'wibble'?
  // CHECK: FIX-IT: Replace [17:6 - 17:12] with "wibble"
  // CHECK: note: 'wibble' declared here
  x->wabble = 17;
}

int printf(const char *restrict, ...);

void f2() {
  unsigned long index;
  // CHECK: warning: format specifies type 'int' but the argument has type 'unsigned long'
  // CHECK: FIX-IT: Replace [26:17 - 26:19] with "%lu"
  MACRO(printf("%d", index));
}
