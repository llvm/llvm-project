// RUN: %clang_cc1 -triple s390x-ibm-zos %s -fsyntax-only -verify

typedef int _Export ty; //expected-error {{cannot be '_Export` qualified}}
ty x;
int f(int _Export argument); //expected-error {{cannot be '_Export` qualified}}
static int _Export file_scope_static; //expected-error {{cannot be '_Export` qualified}}
struct S {
  int _Export nonstaticdatamember; //expected-error {{cannot be '_Export` qualified}}
};
void g() {
  int _Export automatic; //expected-error {{cannot be '_Export` qualified}}
}

static void _Export static_func() { //expected-error {{cannot be '_Export` qualified}}
}

void _Export h() {
  static_func();
}

void j() {
  static int _Export sl = 0; //expected-error {{cannot be '_Export` qualified}}
}

int _Export file_scope;
