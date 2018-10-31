// RUN: %clang_analyze_cc1 -analyze -analyzer-checker=core,osx.cocoa.RetainCount -analyzer-config osx.cocoa.RetainCount:CheckOSObject=true -analyzer-output=text -verify %s

struct OSMetaClass;

#define TRUSTED __attribute__((annotate("rc_ownership_trusted_implementation")))
#define OS_CONSUME TRUSTED __attribute__((annotate("rc_ownership_consumed")))
#define OS_RETURNS_RETAINED TRUSTED __attribute__((annotate("rc_ownership_returns_retained")))
#define OS_RETURNS_NOT_RETAINED TRUSTED __attribute__((annotate("rc_ownership_returns_not_retained")))

#define OSTypeID(type)   (type::metaClass)

#define OSDynamicCast(type, inst)   \
    ((type *) OSMetaClassBase::safeMetaCast((inst), OSTypeID(type)))

struct OSObject {
  virtual void retain();
  virtual void release() {};
  virtual ~OSObject(){}

  static OSObject *generateObject(int);

  static const OSMetaClass * const metaClass;
};

struct OSArray : public OSObject {
  unsigned int getCount();

  static OSArray *withCapacity(unsigned int capacity);
  static void consumeArray(OS_CONSUME OSArray * array);

  static OSArray* consumeArrayHasCode(OS_CONSUME OSArray * array) {
    return nullptr;
  }

  static OS_RETURNS_NOT_RETAINED OSArray *MaskedGetter();
  static OS_RETURNS_RETAINED OSArray *getOoopsActuallyCreate();


  static const OSMetaClass * const metaClass;
};

struct OtherStruct {
  static void doNothingToArray(OSArray *array);
};

struct OSMetaClassBase {
  static OSObject *safeMetaCast(const OSObject *inst, const OSMetaClass *meta);
};

void check_no_invalidation() {
  OSArray *arr = OSArray::withCapacity(10); // expected-note{{Call to function 'withCapacity' returns an OSObject of type struct OSArray * with a +1 retain count}}
  OtherStruct::doNothingToArray(arr);
} // expected-warning{{Potential leak of an object stored into 'arr'}}
  // expected-note@-1{{Object leaked}}

void check_rc_consumed() {
  OSArray *arr = OSArray::withCapacity(10);
  OSArray::consumeArray(arr);
}

void check_rc_consume_temporary() {
  OSArray::consumeArray(OSArray::withCapacity(10));
}

void check_rc_getter() {
  OSArray *arr = OSArray::MaskedGetter();
  (void)arr;
}

void check_rc_create() {
  OSArray *arr = OSArray::getOoopsActuallyCreate();
  arr->release();
}


void check_dynamic_cast() {
  OSArray *arr = OSDynamicCast(OSArray, OSObject::generateObject(1));
  arr->release();
}

void check_dynamic_cast_null_check() {
  OSArray *arr = OSDynamicCast(OSArray, OSObject::generateObject(1));
  if (!arr)
    return;
  arr->release();
}

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
