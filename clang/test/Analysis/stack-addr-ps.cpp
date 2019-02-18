// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-store=region -verify %s -Wno-undefined-bool-conversion

typedef __INTPTR_TYPE__ intptr_t;

const int& g() {
  int s;
  return s; // expected-warning{{Address of stack memory associated with local variable 's' returned}} expected-warning{{reference to stack memory associated with local variable 's' returned}}
}

const int& g2() {
  int s1;
  int &s2 = s1; // expected-note {{binding reference variable 's2' here}}
  return s2; // expected-warning{{Address of stack memory associated with local variable 's1' returned}} expected-warning {{reference to stack memory associated with local variable 's1' returned}}
}

const int& g3() {
  int s1;
  int &s2 = s1; // expected-note {{binding reference variable 's2' here}}
  int &s3 = s2; // expected-note {{binding reference variable 's3' here}}
  return s3; // expected-warning{{Address of stack memory associated with local variable 's1' returned}} expected-warning {{reference to stack memory associated with local variable 's1' returned}}
}

void g4() {
  static const int &x = 3; // no warning
}

int get_value();

const int &get_reference1() { return get_value(); } // expected-warning{{Address of stack memory associated with temporary object of type 'int' returned}} expected-warning {{returning reference to local temporary}}

const int &get_reference2() {
  const int &x = get_value(); // expected-note {{binding reference variable 'x' here}}
  return x; // expected-warning{{Address of stack memory associated with temporary object of type 'int' returned}} expected-warning {{returning reference to local temporary}}
}

const int &get_reference3() {
  const int &x1 = get_value(); // expected-note {{binding reference variable 'x1' here}}
  const int &x2 = x1; // expected-note {{binding reference variable 'x2' here}}
  return x2; // expected-warning{{Address of stack memory associated with temporary object of type 'int' returned}} expected-warning {{returning reference to local temporary}}
}

int global_var;
int *f1() {
  int &y = global_var;
  return &y;
}

int *f2() {
  int x1;
  int &x2 = x1; // expected-note {{binding reference variable 'x2' here}}
  return &x2; // expected-warning{{Address of stack memory associated with local variable 'x1' returned}} expected-warning {{address of stack memory associated with local variable 'x1' returned}}
}

int *f3() {
  int x1;
  int *const &x2 = &x1; // expected-note {{binding reference variable 'x2' here}}
  return x2; // expected-warning {{address of stack memory associated with local variable 'x1' returned}} expected-warning {{Address of stack memory associated with local variable 'x1' returned to caller}}
}

const int *f4() {
  const int &x1 = get_value(); // expected-note {{binding reference variable 'x1' here}}
  const int &x2 = x1; // expected-note {{binding reference variable 'x2' here}}
  return &x2; // expected-warning{{Address of stack memory associated with temporary object of type 'int' returned}} expected-warning {{returning address of local temporary}}
}

struct S {
  int x;
};

int *mf() {
  S s1;
  S &s2 = s1; // expected-note {{binding reference variable 's2' here}}
  int &x = s2.x; // expected-note {{binding reference variable 'x' here}}
  return &x; // expected-warning{{Address of stack memory associated with local variable 's1' returned}} expected-warning {{address of stack memory associated with local variable 's1' returned}}
}

void *lf() {
    label:
    void *const &x = &&label; // expected-note {{binding reference variable 'x' here}}
    return x; // expected-warning {{returning address of label, which is local}}
}

template <typename T>
struct TS {
  int *get();
  int *m() {
    int *&x = get();
    return x;
  }
};

// rdar://11345441
int* f5() {
  int& i = i; // expected-warning {{Assigned value is garbage or undefined}} expected-warning{{reference 'i' is not yet bound to a value when used within its own initialization}}
  return &i;
}

void *radar13226577() {
    void *p = &p;
    return p; // expected-warning {{stack memory associated with local variable 'p' returned to caller}}
}

namespace rdar13296133 {
  class ConvertsToBool {
  public:
    operator bool() const { return this; }
  };

  class ConvertsToIntptr {
  public:
    operator intptr_t() const { return reinterpret_cast<intptr_t>(this); }
  };

  class ConvertsToPointer {
  public:
    operator const void *() const { return this; }
  };

  intptr_t returnAsNonLoc() {
    ConvertsToIntptr obj;
    return obj; // expected-warning{{Address of stack memory associated with local variable 'obj' returned to caller}}
  }

  bool returnAsBool() {
    ConvertsToBool obj;
    return obj; // no-warning
  }

  intptr_t returnAsNonLocViaPointer() {
    ConvertsToPointer obj;
    return reinterpret_cast<intptr_t>(static_cast<const void *>(obj)); // expected-warning{{Address of stack memory associated with local variable 'obj' returned to caller}}
  }

  bool returnAsBoolViaPointer() {
    ConvertsToPointer obj;
    return obj; // no-warning
  }
}

