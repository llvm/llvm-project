// RUN: %clang_cc1 -std=c++1z -fcilkplus -fsyntax-only -verify %s

class Bar {
  int val[4] = {0,0,0,0};
public:
  Bar();
  ~Bar();
  Bar(const Bar &that);
  Bar(Bar &&that);
  Bar &operator=(Bar that);
  friend void swap(Bar &left, Bar &right);

  const int &getVal(int i) const { return val[i]; }
  void incVal(int i) { val[i]++; }
  
};

int bar(int n);

int x = _Cilk_spawn 0; // expected-error{{'_Cilk_spawn' cannot be used outside a function}}

int illegal_spawn_uses(int n) {
  // FIXME: Fails an assertion during codegen.
  // int x = _Cilk_spawn 0;

  Bar Arrb[4] = _Cilk_spawn { Bar(), Bar(), Bar(), Bar() }; // expected-error{{expected expression}}

  if (int i = _Cilk_spawn bar(n)) // expected-error{{'_Cilk_spawn' not allowed in this scope}}
    bar(i);

  if ((_Cilk_spawn bar(n))) // expected-error{{'_Cilk_spawn' not allowed in this scope}}
    bar(n);

  for (int i = _Cilk_spawn bar(n); i < n; ++i) { // expected-error{{'_Cilk_spawn' not allowed in this scope}}
    bar(i);
  }

  for (int i = 0; i < n; ++i)
    _Cilk_spawn break; // expected-error{{'break' statement not in loop or switch statement}}

  for (int i = 0; i < n; ++i) {
    _Cilk_spawn break; // expected-error{{'break' statement not in loop or switch statement}}
  }

  return _Cilk_spawn bar(n); // expected-warning{{no parallelism from a '_Cilk_spawn' in a return statement}}
}

void bad_jumps_spawn(int n) {
 label3: bar(n);
  goto label2; // expected-error{{cannot jump from this goto statement to its label}}

  _Cilk_spawn { // expected-note{{jump bypasses '_Cilk_spawn'}}
  label1: bar(n);
  label2: bar(n);
    goto label1;
    goto label3; // expected-error{{cannot jump out of '_Cilk_spawn' statement}}
  };

  _Cilk_spawn goto label1; // expected-error{{use of undeclared label}}
}


void bad_jumps_cilk_for(int n) {
 label3: bar(n);
  goto label2; // expected-error{{cannot jump from this goto statement to its label}}

  _Cilk_for(int i = 0; i < n; ++i) { // expected-note{{jump bypasses '_Cilk_for'}} expected-note{{jump bypasses variable initialization}}
  label2: bar(i);
    goto label3; // expected-error{{cannot jump out of '_Cilk_for' statement}}
  }
}
