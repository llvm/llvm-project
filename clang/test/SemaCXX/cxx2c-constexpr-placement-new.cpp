// RUN: %clang_cc1 -std=c++2c -verify %s


namespace std {
  using size_t = decltype(sizeof(0));
}

void *operator new(std::size_t, void *p) { return p; }
void* operator new[] (std::size_t, void* p) {return p;}


consteval int ok() {
    int i;
    new (&i) int(0);
    new (&i) int[1]{1};
    new (static_cast<void*>(&i)) int(0);
    return 0;
}

consteval int conversion() {
    int i;
    new (static_cast<void*>(&i)) float(0);
    // expected-note@-1 {{placement new would change type of storage from 'int' to 'float'}}
    return 0;
}

consteval int indeterminate() {
    int * indeterminate;
    new (indeterminate) int(0);
    // expected-note@-1 {{read of uninitialized object is not allowed in a constant expression}}
    return 0;
}

consteval int array1() {
    int i[2];
    new (&i) int[]{1,2};
    new (&i) int[]{1};
    new (&i) int(0);
    new (static_cast<void*>(&i)) int[]{1,2};
    new (static_cast<void*>(&i)) int[]{1};
    return 0;
}

consteval int array2() {
    int i[1];
    new (&i) int[2];
    //expected-note@-1 {{placement new would change type of storage from 'int[1]' to 'int[2]'}}
    return 0;
}

struct S{
    int* i;
    constexpr S() : i(new int(42)) {} // #no-deallocation
    constexpr ~S() {delete i;}
};

consteval void alloc() {
    S* s = new S();
    s->~S();
    new (s) S();
    delete s;
}


consteval void alloc_err() {
    S* s = new S();
    new (s) S();
    delete s;
}



int a = ok();
int b = conversion(); // expected-error {{call to consteval function 'conversion' is not a constant expression}} \
                      // expected-note {{in call to 'conversion()'}}
int c = indeterminate(); // expected-error {{call to consteval function 'indeterminate' is not a constant expression}} \
                         // expected-note {{in call to 'indeterminate()'}}
int d = array1();
int e = array2(); // expected-error {{call to consteval function 'array2' is not a constant expression}} \
                  // expected-note {{in call to 'array2()'}}
int alloc1 = (alloc(), 0);
int alloc2 = (alloc_err(), 0); // expected-error {{call to consteval function 'alloc_err' is not a constant expression}}
                               // expected-note@#no-deallocation {{allocation performed here was not deallocated}}

constexpr int *intptr() {
  return new int;
}

constexpr bool yay() {
  int *ptr = new (intptr()) int(42);
  bool ret = *ptr == 42;
  delete ptr;
  return ret;
}
static_assert(yay());

constexpr bool blah() {
  int *ptr = new (intptr()) int[3]{ 1, 2, 3 }; // expected-note {{placement new would change type of storage from 'int' to 'int[3]'}}
  bool ret = ptr[0] == 1 && ptr[1] == 2 && ptr[2] == 3;
  delete [] ptr;
  return ret;
}
static_assert(blah()); // expected-error {{not an integral constant expression}} \
                       // expected-note {{in call to 'blah()'}}

constexpr int *get_indeterminate() {
  int *evil;
  return evil; // expected-note {{read of uninitialized object is not allowed in a constant expression}}
}

constexpr bool bleh() {
  int *ptr = new (get_indeterminate()) int;  // expected-note {{in call to 'get_indeterminate()'}}
  return true;
}
static_assert(bleh()); // expected-error {{not an integral constant expression}} \
                        // expected-note {{in call to 'bleh()'}}
