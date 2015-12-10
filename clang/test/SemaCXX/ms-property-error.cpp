// RUN: %clang_cc1 -verify -fms-compatibility %s -fsyntax-only -o -

class S {
public:
  __declspec(property(get=GetX,put=PutX)) int x[];
  int GetX(int i, int j) { return i+j; } // expected-note {{'GetX' declared here}}
  void PutX(int i, int j, int k) { j = i = k; } // expected-note {{'PutX' declared here}}
};

char *ptr;
template <typename T>
class St {
public:
  __declspec(property(get=GetX,put=PutX)) T x[];
  T GetX(T i, T j) { return i+j; } // expected-note 3 {{'GetX' declared here}}
  T PutX(T i, T j, T k) { return j = i = k; }  // expected-note 2 {{'PutX' declared here}}
  ~St() {
    x[1] = 0; // expected-error {{too few arguments to function call, expected 3, have 2}}
    x[2][3] = 4;
    ++x[2][3];
    x[1][2] = x[3][4][5]; // expected-error {{too many arguments to function call, expected 2, have 3}}
    ptr = x[1][2] = x[3][4]; // expected-error {{assigning to 'char *' from incompatible type 'int'}}
  }
};

// CHECK-LABEL: main
int main(int argc, char **argv) {
  S *p1 = 0;
  St<float> *p2 = 0;
  St<int> a; // expected-note {{in instantiation of member function 'St<int>::~St' requested here}}
  int j = (p1->x)[223][11][2]; // expected-error {{too many arguments to function call, expected 2, have 3}}
  (p1->x[23]) = argc; // expected-error {{too few arguments to function call, expected 3, have 2}}
  float j1 = (p2->x); // expected-error {{too few arguments to function call, expected 2, have 0}}
  ((p2->x)[23])[1][2] = *argv; // expected-error {{too many arguments to function call, expected 3, have 4}}
  argv = p2->x[11][22] = argc; // expected-error {{assigning to 'char **' from incompatible type 'float'}}
  return ++(((p2->x)[23])); // expected-error {{too few arguments to function call, expected 2, have 1}}
}
