// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-output=text -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-output=plist-multi-file  %s -o %t.plist
// RUN: cat %t.plist | %diff_plist %S/Inputs/expected-plists/undef-value-param.c.plist -

void foo_irrelevant(int c) {
    if (c)
        return;
    c++;
    return;
}
void foo(int c, int *x) {
    if (c)
           //expected-note@-1{{Assuming 'c' is not equal to 0}}
           //expected-note@-2{{Taking true branch}}
           return; // expected-note{{Returning without writing to '*x'}}
    *x = 5;
}

int use(int c) {
    int xx; //expected-note {{'xx' declared without an initial value}}
    int *y = &xx;
    foo (c, y);
                //expected-note@-1{{Calling 'foo'}}
                //expected-note@-2{{Returning from 'foo'}}
    foo_irrelevant(c);
    return xx+3; //expected-warning{{The left operand of '+' is a garbage value}}
                 //expected-note@-1{{The left operand of '+' is a garbage value}}
}

void initArray(int x, double XYZ[3]) {
    if (x <= 0) //expected-note {{Taking true branch}}
                //expected-note@-1 {{Assuming 'x' is <= 0}}
        return;
    XYZ[0] = 1;
    XYZ[1] = 1;
    XYZ[2] = 1;
}
int testPassingParentRegionArray(int x) {
    double XYZ[3];
    initArray(x, XYZ); //expected-note {{Calling 'initArray'}}
                       //expected-note@-1 {{Returning from 'initArray'}}
    return 1 * XYZ[1]; //expected-warning {{The right operand of '*' is a garbage value}}
                       //expected-note@-1 {{The right operand of '*' is a garbage value}}
}

double *getValidPtr();
struct WithFields {
  double *f1;
};
void initStruct(int x, struct WithFields *X) {
  if (x <= 0) //expected-note {{Taking true branch}}
              //expected-note@-1 {{Assuming 'x' is <= 0}}

    return; //expected-note{{Returning without writing to 'X->f1'}}
  X->f1 = getValidPtr();
}
double testPassingParentRegionStruct(int x) {
  struct WithFields st;
  st.f1 = 0; // expected-note {{Null pointer value stored to 'st.f1'}}
  initStruct(x, &st); //expected-note {{Calling 'initStruct'}}
                      //expected-note@-1 {{Returning from 'initStruct'}}
  return (*st.f1); //expected-warning {{Dereference of null pointer}}
                   //expected-note@-1{{Dereference of null pointer (loaded from field 'f1')}}
}

