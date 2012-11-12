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
   S strings[];	// expected-error {{flexible array member 'strings' of non-POD element type 'S []'}}
};

class A {
  int s;
  char c[];
};

union B {
  int s;
  char c[];
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

}
