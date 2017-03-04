// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-config suppress-null-return-paths=false -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core -verify -DSUPPRESSED=1 %s

namespace rdar12676053 {
  // Delta-reduced from a preprocessed file.
  template<class T>
  class RefCount {
    T *ref;
  public:
    T *operator->() const {
      return ref ? ref : 0;
    }
  };

  class string {};

  class ParserInputState {
  public:
    string filename;
  };

  class Parser {
    void setFilename(const string& f)  {
      inputState->filename = f;
#ifndef SUPPRESSED
// expected-warning@-2 {{Called C++ object pointer is null}}
#endif
    }
  protected:
    RefCount<ParserInputState> inputState;
  };
}


// This is the standard placement new.
inline void* operator new(__typeof__(sizeof(int)), void* __p) throw()
{
  return __p;
}

extern bool coin();

class SomeClass {
public:
  void doSomething();
};

namespace References {
  class Map {
    int *&getNewBox();
    int *firstBox;

  public:
    int *&getValue(int key) {
      if (coin()) {
        return firstBox;
      } else {
        int *&newBox = getNewBox();
        newBox = 0;
        return newBox;
      }
    }

    int *&getValueIndirectly(int key) {
      int *&valueBox = getValue(key);
      return valueBox;
    }
  };

  void testMap(Map &m, int i) {
    *m.getValue(i) = 1;
#ifndef SUPPRESSED
    // expected-warning@-2 {{Dereference of null pointer}}
#endif

    *m.getValueIndirectly(i) = 1;
#ifndef SUPPRESSED
    // expected-warning@-2 {{Dereference of null pointer}}
#endif

    int *&box = m.getValue(i);
    extern int *getPointer();
    box = getPointer();
    *box = 1; // no-warning

    int *&box2 = m.getValue(i);
    box2 = 0;
    *box2 = 1; // expected-warning {{Dereference of null pointer}}
  }

  SomeClass *&getSomeClass() {
    if (coin()) {
      extern SomeClass *&opaqueClass();
      return opaqueClass();
    } else {
      static SomeClass *sharedClass;
      sharedClass = 0;
      return sharedClass;
    }
  }

  void testClass() {
    getSomeClass()->doSomething();
#ifndef SUPPRESSED
    // expected-warning@-2 {{Called C++ object pointer is null}}
#endif

    // Separate the lvalue-to-rvalue conversion from the subsequent dereference.
    SomeClass *object = getSomeClass();
    object->doSomething();
#ifndef SUPPRESSED
    // expected-warning@-2 {{Called C++ object pointer is null}}
#endif
  }

  SomeClass *getNull() {
    return 0;
  }

  SomeClass &returnNullReference() {
    SomeClass *x = getNull();
    return *x;
#ifndef SUPPRESSED
    // expected-warning@-2 {{Returning null reference}}
#endif
  }
}

class X{
public:
	void get();
};

X *getNull() {
	return 0;
}

void deref1(X *const &p) {
	return p->get();
	#ifndef SUPPRESSED
	  // expected-warning@-2 {{Called C++ object pointer is null}}
	#endif
}

void test1() {
	return deref1(getNull());
}

void deref2(X *p3) {
	p3->get();
	#ifndef SUPPRESSED
	  // expected-warning@-2 {{Called C++ object pointer is null}}
	#endif
}

void pass2(X *const &p2) {
	deref2(p2);
}

void test2() {
	pass2(getNull());
}

void deref3(X *const &p2) {
	X *p3 = p2;
	p3->get();
	#ifndef SUPPRESSED
	  // expected-warning@-2 {{Called C++ object pointer is null}}
	#endif
}

void test3() {
	deref3(getNull());
}


namespace Cleanups {
  class NonTrivial {
  public:
    ~NonTrivial();

    SomeClass *getNull() {
      return 0;
    }
  };

  void testImmediate() {
    NonTrivial().getNull()->doSomething();
#ifndef SUPPRESSED
    // expected-warning@-2 {{Called C++ object pointer is null}}
#endif
  }

  void testAssignment() {
    SomeClass *ptr = NonTrivial().getNull();
    ptr->doSomething();
#ifndef SUPPRESSED
    // expected-warning@-2 {{Called C++ object pointer is null}}
#endif
  }

  void testArgumentHelper(SomeClass *arg) {
    arg->doSomething();
#ifndef SUPPRESSED
    // expected-warning@-2 {{Called C++ object pointer is null}}
#endif
  }

  void testArgument() {
    testArgumentHelper(NonTrivial().getNull());
  }
}
