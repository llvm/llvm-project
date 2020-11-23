// RUN: %clang_cc1 -triple x86_64-windows -fborland-extensions -DBORLAND -fsyntax-only -verify -fblocks %s
// RUN: %clang_cc1 -triple x86_64-windows -fms-extensions -fsyntax-only -verify -fblocks %s

#define JOIN2(x,y) x ## y
#define JOIN(x,y) JOIN2(x,y)
#define TEST2(name) JOIN(name,__LINE__)
#define TEST TEST2(test)
typedef int DWORD;

#pragma sysheader begin

struct EXCEPTION_INFO{};

unsigned long __exception_code();
#ifdef BORLAND
struct EXCEPTION_INFO* __exception_info();
#endif
int __abnormal_termination();

#define GetExceptionCode __exception_code
#define GetExceptionInformation __exception_info
#define AbnormalTermination __abnormal_termination

#pragma sysheader end

DWORD FilterExpression(int); // expected-note{{declared here}}
DWORD FilterExceptionInformation(struct EXCEPTION_INFO*);

const char * NotFilterExpression();

void TEST() {
  __try {
    __try {
      __try {
      }
      __finally{
      }
    }
    __finally{
    }
  }
  __finally{
  }
}

void TEST() {
  __try {

  }
}  // expected-error{{expected '__except' or '__finally' block}}

void TEST() {
  __except (FilterExpression()) { // expected-warning{{implicit declaration of function '__except' is invalid in C99}} \
    // expected-error{{too few arguments to function call, expected 1, have 0}} \
    // expected-error{{expected ';' after expression}}
  }
}

void TEST() {
  __finally { } // expected-error{{}}
}

void TEST() {
  __try{
    int try_scope = 0;
  } // TODO: expected expression is an extra error
  __except( try_scope ? 1 : -1 ) // expected-error{{undeclared identifier 'try_scope'}} expected-error{{expected expression}}
  {}
}

void TEST() {
  __try {

  }
  // TODO: Why are there two errors?
  __except( ) { // expected-error{{expected expression}} expected-error{{expected expression}}
  }
}

void TEST() {
  __try {

  }
  __except ( FilterExpression(GetExceptionCode()) ) {

  }

  __try {

  }
  __except( FilterExpression(__exception_code()) ) {

  }

  __try {

  }
  __except( FilterExceptionInformation(__exception_info()) ) {

  }

  __try {

  }
  __except(FilterExceptionInformation( GetExceptionInformation() ) ) {

  }
}

void TEST() {
  __try {

  }
  __except ( NotFilterExpression() ) { // expected-error{{filter expression has non-integral type 'const char *'}}

  }
}

void TEST() {
  int function_scope = 0;
  __try {
    int try_scope = 0;
  }
  __except ( FilterExpression(GetExceptionCode()) ) {
    (void)function_scope;
    (void)try_scope; // expected-error{{undeclared identifier}}
  }
}

void TEST() {
  int function_scope = 0;
  __try {
    int try_scope = 0;
  }
  __finally {
    (void)function_scope;
    (void)try_scope; // expected-error{{undeclared identifier}}
  }
}

void TEST() {
  int function_scope = 0;
  __try {

  }
  __except( function_scope ? 1 : -1 ) {}
}

#ifdef BORLAND
void TEST() {
  (void)__abnormal_termination(); // expected-error{{only allowed in __finally block}}
  (void)AbnormalTermination();  // expected-error{{only allowed in __finally block}}

  __try {
    (void)AbnormalTermination;  // expected-error{{only allowed in __finally block}}
    (void)__abnormal_termination; // expected-error{{only allowed in __finally block}}
  }
  __except( 1 ) {
    (void)AbnormalTermination;  // expected-error{{only allowed in __finally block}}
    (void)__abnormal_termination; // expected-error{{only allowed in __finally block}}
  }

  __try {
  }
  __finally {
    AbnormalTermination();
    __abnormal_termination();
  }
}
#endif

void TEST() {
  (void)__exception_info();       // expected-error{{only allowed in __except filter expression}}
  (void)GetExceptionInformation(); // expected-error{{only allowed in __except filter expression}}
}

void TEST() {
#ifndef BORLAND
  (void)__exception_code;     // expected-error{{builtin functions must be directly called}}
#endif
  (void)__exception_code();     // expected-error{{only allowed in __except block or filter expression}}
  (void)GetExceptionCode();     // expected-error{{only allowed in __except block or filter expression}}
}

void TEST() {
  __try {
  } __except(1) {
    GetExceptionCode(); // valid
    GetExceptionInformation(); // expected-error{{only allowed in __except filter expression}}
  }
}

void test_seh_leave_stmt() {
  __leave; // expected-error{{'__leave' statement not in __try block}}

  __try {
    __leave;
    __leave 4; // expected-error{{expected ';' after __leave statement}}
  } __except(1) {
    __leave; // expected-error{{'__leave' statement not in __try block}}
  }

  __try {
    __leave;
  } __finally {
    __leave; // expected-error{{'__leave' statement not in __try block}}
  }
  __leave; // expected-error{{'__leave' statement not in __try block}}
}

void test_jump_out_of___finally() {
  while(1) {
    __try {
    } __finally {
      continue; // expected-warning{{jump out of __finally block has undefined behavior}}
    }
  }
  __try {
  } __finally {
    while (1) {
      continue;
    }
  }

  // Check that a deep __finally containing a block with a shallow continue
  // doesn't trigger the warning.
  while(1) {{{{
    __try {
    } __finally {
      ^{
        while(1)
          continue;
      }();
    }
  }}}}

  while(1) {
    __try {
    } __finally {
      break; // expected-warning{{jump out of __finally block has undefined behavior}}
    }
  }
  switch(1) {
  case 1:
    __try {
    } __finally {
      break; // expected-warning{{jump out of __finally block has undefined behavior}}
    }
  }
  __try {
  } __finally {
    while (1) {
      break;
    }
  }

  __try {
    __try {
    } __finally {
      __leave; // expected-warning{{jump out of __finally block has undefined behavior}}
    }
  } __finally {
  }
  __try {
  } __finally {
    __try {
      __leave;
    } __finally {
    }
  }

  __try {
  } __finally {
    return; // expected-warning{{jump out of __finally block has undefined behavior}}
  }

  __try {
  } __finally {
    ^{
      return;
    }();
  }
}

void test_typo_in_except() {
  __try {
  } __except(undeclared_identifier) { // expected-error {{use of undeclared identifier 'undeclared_identifier'}} expected-error {{expected expression}}
  }
}
