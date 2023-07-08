// RUN: %clang_cc1 -x objective-c++ -fcxx-exceptions -fsyntax-only -Werror -verify -Wno-objc-root-class %s

@interface MyClass
- (void)someMethod;
@end

struct BadReturn {
  BadReturn(MyClass * myObject);
  int bar(MyClass * myObject);
  void MemFunc(MyClass * myObject);
  int i;
  MyClass *CObj;
};

@implementation MyClass
- (void)someMethod {
    [self privateMethod];  // clang already does not warn here
}

int BadReturn::bar(MyClass * myObject) {
    [myObject privateMethod];
    return 0;
}

BadReturn::BadReturn(MyClass * myObject) try : CObj(myObject) {
} catch(...) {
  try {
    [myObject privateMethod];
    [myObject privateMethod1];
    getMe = bar(myObject); // expected-error {{cannot refer to a non-static member from the handler of a constructor function try block}}
    [CObj privateMethod1]; // expected-error {{cannot refer to a non-static member from the handler of a constructor function try block}}
  } catch(int ei) {
    i = ei; // expected-error {{cannot refer to a non-static member from the handler of a constructor function try block}}
  } catch(...) {
    {
      i = 0; // expected-error {{cannot refer to a non-static member from the handler of a constructor function try block}}
    }
  }
}

void BadReturn::MemFunc(MyClass * myObject) try {
} catch(...) {
  try {
    [myObject privateMethod];
    [myObject privateMethod1];
    getMe = bar(myObject);
    [CObj privateMethod1];
  } catch(int ei) {
    i = ei;
  } catch(...) {
    {
      i = 0;
    }
  }
}

- (void)privateMethod { }

- (void)privateMethod1 {
  getMe = getMe+1;
}

static int getMe;

@end
