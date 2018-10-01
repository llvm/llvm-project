// RUN: %clang_analyze_cc1 -analyze -analyzer-checker=core,osx.cocoa.RetainCount -analyzer-config osx.cocoa.RetainCount:CheckOSObject=true -analyzer-output=text -verify %s

struct OSObject {
  virtual void retain();
  virtual void release();

  virtual ~OSObject(){}
};

struct OSArray : public OSObject {
  unsigned int getCount();

  static OSArray *withCapacity(unsigned int capacity);
};

void use_after_release() {
  OSArray *arr = OSArray::withCapacity(10); // expected-note{{Call to function 'withCapacity' returns an OSObject of type struct OSArray * with a +1 retain count}}
  arr->release(); // expected-note{{Object released}}
  arr->getCount(); // expected-warning{{Reference-counted object is used after it is released}}
                   // expected-note@-1{{Reference-counted object is used after it is released}}
}

void potential_leak() {
  OSArray *arr = OSArray::withCapacity(10); // expected-note{{Call to function 'withCapacity' returns an OSObject of type struct OSArray * with a +1 retain count}}
  arr->retain(); // expected-note{{Reference count incremented. The object now has a +2 retain count}}
  arr->release(); // expected-note{{Reference count decremented. The object now has a +1 retain count}}
  arr->getCount();
} // expected-warning{{Potential leak of an object stored into 'arr'}}
  // expected-note@-1{{Object leaked: object allocated and stored into 'arr' is not referenced later in this execution path and has a retain count of +1}}

void proper_cleanup() {
  OSArray *arr = OSArray::withCapacity(10); // +1
  arr->retain(); // +2
  arr->release(); // +1
  arr->getCount();
  arr->release(); // 0
}

struct ArrayOwner {
  OSArray *arr;

  OSArray *getArray() {
    return arr;
  }

  OSArray *createArray() {
    return OSArray::withCapacity(10);
  }

  OSArray *createArraySourceUnknown();

  OSArray *getArraySourceUnknown();
};

//unsigned int leak_on_create_no_release(ArrayOwner *owner) {
  //OSArray *myArray = 

//}

unsigned int no_warning_on_getter(ArrayOwner *owner) {
  OSArray *arr = owner->getArray();
  return arr->getCount();
}

unsigned int warn_on_overrelease(ArrayOwner *owner) {
  OSArray *arr = owner->getArray(); // expected-note{{function call returns an OSObject of type struct OSArray * with a +0 retain count}}
  arr->release(); // expected-warning{{Incorrect decrement of the reference count of an object that is not owned at this point by the caller}}
                  // expected-note@-1{{Incorrect decrement of the reference count of an object that is not owned at this point by the caller}}
  return arr->getCount();
}

unsigned int nowarn_on_release_of_created(ArrayOwner *owner) {
  OSArray *arr = owner->createArray();
  unsigned int out = arr->getCount();
  arr->release();
  return out;
}

unsigned int nowarn_on_release_of_created_source_unknown(ArrayOwner *owner) {
  OSArray *arr = owner->createArraySourceUnknown();
  unsigned int out = arr->getCount();
  arr->release();
  return out;
}

unsigned int no_warn_ok_release(ArrayOwner *owner) {
  OSArray *arr = owner->getArray(); // +0
  arr->retain(); // +1
  arr->release(); // +0
  return arr->getCount(); // no-warning
}

unsigned int warn_on_overrelease_with_unknown_source(ArrayOwner *owner) {
  OSArray *arr = owner->getArraySourceUnknown(); // expected-note{{function call returns an OSObject of type struct OSArray * with a +0 retain count}}
  arr->release(); // expected-warning{{Incorrect decrement of the reference count of an object that is not owned at this point by the caller}}
                  // expected-note@-1{{Incorrect decrement of the reference count of an object that is not owned at this point by the caller}}
  return arr->getCount();
}

unsigned int ok_release_with_unknown_source(ArrayOwner *owner) {
  OSArray *arr = owner->getArraySourceUnknown(); // +0
  arr->retain(); // +1
  arr->release(); // +0
  return arr->getCount();
}
