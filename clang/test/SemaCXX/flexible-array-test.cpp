// RUN: %clang_cc1 -fsyntax-only -verify %s 
// pr7029

template <class Key, class T> struct QMap
{
  void insert(const Key &, const T &);
  T v;
};


template <class Key, class T>
void QMap<Key, T>::insert(const Key &, const T &avalue)
{
  v = avalue;
}

struct Rec {
  union { // expected-warning-re {{variable sized type '{{.*}}' not at the end of a struct or class is a GNU extension}}
    int u0[];
  };
  int x;
} rec;

struct inotify_event
{
  int wd;
 
  // clang doesn't like '[]': 
  // cannot initialize a parameter of type 'void *' with an rvalue of type 'char (*)[]'
  char name [];	
};


void foo()
{
    inotify_event event;
    inotify_event* ptr = &event;
    inotify_event event1 = *ptr;
    *ptr = event;
    QMap<int, inotify_event> eventForId;
    eventForId.insert(ptr->wd, *ptr);
}

struct S {
  virtual void foo();
};

struct X {
   int blah;
   S strings[];
};

S a, b = a;
S f(X &x) {
  a = b;
  return x.strings[0];
}

class A {
  int s;
  char c[];
};

union B {
  int s;
  char c[];
};

class C {
  char c[]; // expected-error {{flexible array member 'c' with type 'char []' is not at the end of class}}
  int s; // expected-note {{next field declaration is here}}
};

namespace rdar9065507 {

struct StorageBase {
  long ref_count;
  unsigned size;
  unsigned capacity;
};

struct Storage : StorageBase {
  int data[];
};

struct VirtStorage : virtual StorageBase {
  int data[]; // expected-error {{flexible array member 'data' not allowed in struct which has a virtual base class}}
};

}

struct NonTrivDtor { ~NonTrivDtor(); };
// FIXME: It's not clear whether we should disallow examples like this. GCC accepts.
struct FlexNonTrivDtor {
  int n;
  NonTrivDtor ntd[]; // expected-error {{flexible array member 'ntd' of type 'NonTrivDtor []' with non-trivial destruction}}
  ~FlexNonTrivDtor() {
    for (int i = n; i != 0; --i)
      ntd[i-1].~NonTrivDtor();
  }
};
