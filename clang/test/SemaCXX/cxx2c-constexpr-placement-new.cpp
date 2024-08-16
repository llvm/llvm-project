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

consteval int conversion() { // expected-error {{consteval function never produces a constant expression}}
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

consteval int array2() { // expected-error {{consteval function never produces a constant expression}}
    int i[1];
    new (&i) int[2];
    //expected-note@-1 {{placement new would change type of storage from 'int[1]' to 'int[2]'}}
    return 0;
}

struct S{
    int* i;
    constexpr S() : i(new int(42)) {} // expected-note {{allocation performed here was not deallocated}}
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
int c = indeterminate(); // expected-error {{call to consteval function 'placement_new::indeterminate' is not a constant expression}} \
             // expected-note {{in call to 'indeterminate()'}}
int d = array1();
int alloc1 = (alloc(), 0);
int alloc2 = (alloc_err(), 0); // expected-error {{call to consteval function 'placement_new::alloc_err' is not a constant expression}}
