// RUN: %clang_cc1 -fborland-extensions -fsyntax-only -verify %s

#define JOIN2(x,y) x ## y
#define JOIN(x,y) JOIN2(x,y)
#define TEST2(name) JOIN(name,__LINE__)
#define TEST TEST2(test)
typedef int DWORD;

#pragma sysheader begin

struct EXCEPTION_INFO{};

int __exception_code();
struct EXCEPTION_INFO* __exception_info();
void __abnormal_termination();

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
  __except ( FilterExpression() ) { // expected-warning{{implicit declaration of function '__except' is invalid in C99}} \
    // expected-error{{too few arguments to function call, expected 1, have 0}}

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
  __except ( NotFilterExpression() ) { // expected-error{{filter expression type should be an integral value not 'const char *'}}

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

void TEST() {
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

void TEST() {
  (void)__exception_code;       // expected-error{{only allowed in __except block}}
  (void)__exception_info;       // expected-error{{only allowed in __except filter expression}}
  (void)__abnormal_termination; // expected-error{{only allowed in __finally block}}

  (void)GetExceptionCode();     // expected-error{{only allowed in __except block}}
  (void)GetExceptionInformation(); // expected-error{{only allowed in __except filter expression}}
  (void)AbnormalTermination();  // expected-error{{only allowed in __finally block}}
}
