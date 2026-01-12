// RUN: %clang_cc1 -triple s390x-ibm-zos %s -fsyntax-only -verify

typedef int _Export ty; //expected-error {{needs to have external linkage to be '_Export` qualified}}
ty typedef_var;
int f(int _Export argument); //expected-error {{needs to have external linkage to be '_Export` qualified}}
static int _Export file_scope_static; //expected-error {{needs to have external linkage to be '_Export` qualified}}
struct S {
  int _Export nonstaticdatamember; //expected-error {{needs to have external linkage to be '_Export` qualified}}
};
void g() {
  int _Export automatic; //expected-error {{needs to have external linkage to be '_Export` qualified}}
}

static void _Export static_func() { //expected-error {{needs to have external linkage to be '_Export` qualified}}
}

void _Export h() {
  static_func();
}

void j() {
  static int _Export sl = 0; //expected-error {{needs to have external linkage to be '_Export` qualified}}
}

int _Export file_scope;

struct _Export SE {
};

struct ST {
  void _Export f();
  virtual void _Export v_();
  static int _Export i;
};

namespace {
  int _Export anon_var; //expected-error {{needs to have external linkage to be '_Export` qualified}}
  extern "C" int _Export anon_C_var;
  void _Export anon_f() {} //expected-error {{needs to have external linkage to be '_Export` qualified}}
  extern "C" void _Export anon_C_f() {}
  struct anon_S {
    static int _Export anon_static_data_member; //expected-error {{needs to have external linkage to be '_Export` qualified}}
  };
}
