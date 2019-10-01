// RUN: %clang_analyze_cc1 -fblocks -analyze -analyzer-output=text\
// RUN:                    -analyzer-checker=core,osx -verify %s

#include "os_object_base.h"
#include "os_smart_ptr.h"

struct OSIterator : public OSObject {
  static const OSMetaClass * const metaClass;
};

struct OSArray : public OSObject {
  unsigned int getCount();

  OSIterator * getIterator();

  OSObject *identity() override;

  virtual OSObject *generateObject(OSObject *input);

  virtual void consumeReference(OS_CONSUME OSArray *other);

  void putIntoArray(OSArray *array) OS_CONSUMES_THIS;

  template <typename T>
  void putIntoT(T *owner) OS_CONSUMES_THIS;

  static OSArray *generateArrayHasCode() {
    return new OSArray;
  }

  static OSArray *withCapacity(unsigned int capacity);
  static void consumeArray(OS_CONSUME OSArray * array);

  static OSArray* consumeArrayHasCode(OS_CONSUME OSArray * array) { // expected-note{{Parameter 'array' starts at +1, as it is marked as consuming}}
    return nullptr; // expected-warning{{Potential leak of an object of type 'OSArray'}}
// expected-note@-1{{Object leaked: allocated object of type 'OSArray' is not referenced later in this execution path and has a retain count of +1}}
  }


  static OS_RETURNS_NOT_RETAINED OSArray *MaskedGetter();
  static OS_RETURNS_RETAINED OSArray *getOoopsActuallyCreate();

  static const OSMetaClass * const metaClass;
};

struct MyArray : public OSArray {
  void consumeReference(OSArray *other) override;

  OSObject *identity() override;

  OSObject *generateObject(OSObject *input) override;
};

struct OtherStruct {
  static void doNothingToArray(OSArray *array);
  OtherStruct(OSArray *arr);
};

bool test_meta_cast_no_leak(OSMetaClassBase *arg) {
  return arg && arg->metaCast("blah") != nullptr;
}

static void consumedMismatch(OS_CONSUME OSObject *a,
                             OSObject *b) { // expected-note{{Parameter 'b' starts at +0}}
  a->release();
  b->retain(); // expected-note{{Reference count incremented. The object now has a +1 retain count}}
} // expected-warning{{Potential leak of an object of type 'OSObject'}}
// expected-note@-1{{Object leaked: allocated object of type 'OSObject' is not referenced later in this execution path and has a retain count of +1}}

void escape(void *);
void escape_with_source(void *p) {}
bool coin();

typedef int kern_return_t;
typedef kern_return_t IOReturn;
typedef kern_return_t OSReturn;
#define kOSReturnSuccess  0
#define kIOReturnSuccess 0

bool write_into_out_param_on_success(OS_RETURNS_RETAINED OSObject **obj);

void use_out_param() {
  OSObject *obj;
  if (write_into_out_param_on_success(&obj)) {
    obj->release();
  }
}

void use_out_param_leak() {
  OSObject *obj;
  write_into_out_param_on_success(&obj); // expected-note-re{{Call to function 'write_into_out_param_on_success' writes an OSObject of type 'OSObject' with a +1 retain count into an out parameter 'obj' (assuming the call returns non-zero){{$}}}}
} // expected-warning{{Potential leak of an object stored into 'obj'}}
 // expected-note@-1{{Object leaked: object allocated and stored into 'obj' is not referenced later in this execution path and has a retain count of +1}}

bool write_into_out_param_on_failure(OS_RETURNS_RETAINED_ON_ZERO OSObject **obj);

void use_out_param_leak2() {
  OSObject *obj;
  write_into_out_param_on_failure(&obj); // expected-note-re{{Call to function 'write_into_out_param_on_failure' writes an OSObject of type 'OSObject' with a +1 retain count into an out parameter 'obj' (assuming the call returns zero){{$}}}}
} // expected-warning{{Potential leak of an object stored into 'obj'}}
 // expected-note@-1{{Object leaked: object allocated and stored into 'obj' is not referenced later in this execution path and has a retain count of +1}}

void use_out_param_on_failure() {
  OSObject *obj;
  if (!write_into_out_param_on_failure(&obj)) {
    obj->release();
  }
}

IOReturn write_into_out_param_on_nonzero(OS_RETURNS_RETAINED_ON_NONZERO OSObject **obj);

void use_out_param_on_nonzero() {
  OSObject *obj;
  if (write_into_out_param_on_nonzero(&obj) != kIOReturnSuccess) {
    obj->release();
  }
}

bool write_into_two_out_params(OS_RETURNS_RETAINED OSObject **a,
                               OS_RETURNS_RETAINED OSObject **b);

void use_write_into_two_out_params() {
  OSObject *obj1;
  OSObject *obj2;
  if (write_into_two_out_params(&obj1, &obj2)) {
    obj1->release();
    obj2->release();
  }
}

void use_write_two_out_params_leak() {
  OSObject *obj1;
  OSObject *obj2;
  write_into_two_out_params(&obj1, &obj2); // expected-note-re{{Call to function 'write_into_two_out_params' writes an OSObject of type 'OSObject' with a +1 retain count into an out parameter 'a' (assuming the call returns non-zero){{$}}}}
                                           // expected-note-re@-1{{Call to function 'write_into_two_out_params' writes an OSObject of type 'OSObject' with a +1 retain count into an out parameter 'b' (assuming the call returns non-zero){{$}}}}
} // expected-warning{{Potential leak of an object stored into 'obj1'}}
  // expected-warning@-1{{Potential leak of an object stored into 'obj2'}}
  // expected-note@-2{{Object leaked: object allocated and stored into 'obj1' is not referenced later in this execution path and has a retain count of +1}}
  // expected-note@-3{{Object leaked: object allocated and stored into 'obj2' is not referenced later in this execution path and has a retain count of +1}}

void always_write_into_two_out_params(OS_RETURNS_RETAINED OSObject **a,
                                      OS_RETURNS_RETAINED OSObject **b);

void use_always_write_into_two_out_params() {
  OSObject *obj1;
  OSObject *obj2;
  always_write_into_two_out_params(&obj1, &obj2);
  obj1->release();
  obj2->release();
}

void use_always_write_into_two_out_params_leak() {
  OSObject *obj1;
  OSObject *obj2;
  always_write_into_two_out_params(&obj1, &obj2); // expected-note-re{{Call to function 'always_write_into_two_out_params' writes an OSObject of type 'OSObject' with a +1 retain count into an out parameter 'a'{{$}}}}
                                                  // expected-note-re@-1{{Call to function 'always_write_into_two_out_params' writes an OSObject of type 'OSObject' with a +1 retain count into an out parameter 'b'{{$}}}}
} // expected-warning{{Potential leak of an object stored into 'obj1'}}
  // expected-warning@-1{{Potential leak of an object stored into 'obj2'}}
  // expected-note@-2{{Object leaked: object allocated and stored into 'obj1' is not referenced later in this execution path and has a retain count of +1}}
  // expected-note@-3{{Object leaked: object allocated and stored into 'obj2' is not referenced later in this execution path and has a retain count of +1}}

char *write_into_out_param_on_nonnull(OS_RETURNS_RETAINED OSObject **obj);

void use_out_param_osreturn_on_nonnull() {
  OSObject *obj;
  if (write_into_out_param_on_nonnull(&obj)) {
    obj->release();
  }
}

void use_out_param_leak_osreturn_on_nonnull() {
  OSObject *obj;
  write_into_out_param_on_nonnull(&obj); // expected-note-re{{Call to function 'write_into_out_param_on_nonnull' writes an OSObject of type 'OSObject' with a +1 retain count into an out parameter 'obj' (assuming the call returns non-zero){{$}}}}
} // expected-warning{{Potential leak of an object stored into 'obj'}}
  // expected-note@-1{{Object leaked: object allocated and stored into 'obj' is not referenced later in this execution path and has a retain count of +1}}

bool write_optional_out_param(OS_RETURNS_RETAINED OSObject **obj=nullptr);

void use_optional_out_param() {
  if (write_optional_out_param()) {};
}

OSReturn write_into_out_param_on_os_success(OS_RETURNS_RETAINED OSObject **obj);

void write_into_non_retained_out_param(OS_RETURNS_NOT_RETAINED OSObject **obj);

void use_write_into_non_retained_out_param() {
  OSObject *obj;
  write_into_non_retained_out_param(&obj);
}

void use_write_into_non_retained_out_param_uaf() {
  OSObject *obj;
  write_into_non_retained_out_param(&obj); // expected-note-re{{Call to function 'write_into_non_retained_out_param' writes an OSObject of type 'OSObject' with a +0 retain count into an out parameter 'obj'{{$}}}}
  obj->release(); // expected-warning{{Incorrect decrement of the reference count of an object that is not owned at this point by the caller}}
                  // expected-note@-1{{Incorrect decrement of the reference count of an object that is not owned at this point by the caller}}
}

void always_write_into_out_param(OS_RETURNS_RETAINED OSObject **obj);

void pass_through_out_param(OSObject **obj) {
  always_write_into_out_param(obj);
}

void always_write_into_out_param_has_source(OS_RETURNS_RETAINED OSObject **obj) {
  *obj = new OSObject; // expected-note{{Operator 'new' returns an OSObject of type 'OSObject' with a +1 retain count}}
}

void use_always_write_into_out_param_has_source_leak() {
  OSObject *obj;
  always_write_into_out_param_has_source(&obj); // expected-note{{Calling 'always_write_into_out_param_has_source'}}
                                                // expected-note@-1{{Returning from 'always_write_into_out_param_has_source'}}
} // expected-warning{{Potential leak of an object stored into 'obj'}}
  // expected-note@-1{{Object leaked: object allocated and stored into 'obj' is not referenced later in this execution path and has a retain count of +1}}

void use_void_out_param_osreturn() {
  OSObject *obj;
  always_write_into_out_param(&obj);
  obj->release();
}

void use_void_out_param_osreturn_leak() {
  OSObject *obj;
  always_write_into_out_param(&obj); // expected-note-re{{Call to function 'always_write_into_out_param' writes an OSObject of type 'OSObject' with a +1 retain count into an out parameter 'obj'{{$}}}}
} // expected-warning{{Potential leak of an object stored into 'obj'}}
  // expected-note@-1{{Object leaked: object allocated and stored into 'obj' is not referenced later in this execution path and has a retain count of +1}}

void use_out_param_osreturn() {
  OSObject *obj;
  if (write_into_out_param_on_os_success(&obj) == kOSReturnSuccess) {
    obj->release();
  }
}

void use_out_param_leak_osreturn() {
  OSObject *obj;
  write_into_out_param_on_os_success(&obj); // expected-note-re{{Call to function 'write_into_out_param_on_os_success' writes an OSObject of type 'OSObject' with a +1 retain count into an out parameter 'obj' (assuming the call returns zero){{$}}}}
} // expected-warning{{Potential leak of an object stored into 'obj'}}
  // expected-note@-1{{Object leaked: object allocated and stored into 'obj' is not referenced later in this execution path and has a retain count of +1}}

void cleanup(OSObject **obj);

void test_cleanup_escaping() {
  __attribute__((cleanup(cleanup))) OSObject *obj;
  always_write_into_out_param(&obj); // no-warning, the value has escaped.
}

struct StructWithField {
  OSObject *obj;

  void initViaOutParamCall() { // no warning on writing into fields
    always_write_into_out_param(&obj);
  }

};

bool os_consume_violation_two_args(OS_CONSUME OSObject *obj, bool extra) {
  if (coin()) { // expected-note{{Assuming the condition is false}}
                // expected-note@-1{{Taking false branch}}
    escape(obj);
    return true;
  }
  return false; // expected-note{{Parameter 'obj' is marked as consuming, but the function did not consume the reference}}
}

bool os_consume_violation(OS_CONSUME OSObject *obj) {
  if (coin()) { // expected-note{{Assuming the condition is false}}
                // expected-note@-1{{Taking false branch}}
    escape(obj);
    return true;
  }
  return false; // expected-note{{Parameter 'obj' is marked as consuming, but the function did not consume the reference}}
}

void os_consume_ok(OS_CONSUME OSObject *obj) {
  escape(obj);
}

void use_os_consume_violation() {
  OSObject *obj = new OSObject; // expected-note{{Operator 'new' returns an OSObject of type 'OSObject' with a +1 retain count}}
  os_consume_violation(obj); // expected-note{{Calling 'os_consume_violation'}}
                             // expected-note@-1{{Returning from 'os_consume_violation'}}
} // expected-note{{Object leaked: object allocated and stored into 'obj' is not referenced later in this execution path and has a retain count of +1}}
  // expected-warning@-1{{Potential leak of an object stored into 'obj'}}

void use_os_consume_violation_two_args() {
  OSObject *obj = new OSObject; // expected-note{{Operator 'new' returns an OSObject of type 'OSObject' with a +1 retain count}}
  os_consume_violation_two_args(obj, coin()); // expected-note{{Calling 'os_consume_violation_two_args'}}
                             // expected-note@-1{{Returning from 'os_consume_violation_two_args'}}
} // expected-note{{Object leaked: object allocated and stored into 'obj' is not referenced later in this execution path and has a retain count of +1}}
  // expected-warning@-1{{Potential leak of an object stored into 'obj'}}

void use_os_consume_ok() {
  OSObject *obj = new OSObject;
  os_consume_ok(obj);
}

void test_escaping_into_voidstar() {
  OSObject *obj = new OSObject;
  escape(obj);
}

void test_escape_has_source() {
  OSObject *obj = new OSObject;
  if (obj)
    escape_with_source(obj);
  return;
}

void test_no_infinite_check_recursion(MyArray *arr) {
  OSObject *input = new OSObject;
  OSObject *o = arr->generateObject(input);
  o->release();
  input->release();
}


void check_param_attribute_propagation(MyArray *parent) {
  OSArray *arr = new OSArray;
  parent->consumeReference(arr);
}

unsigned int check_attribute_propagation(OSArray *arr) {
  OSObject *other = arr->identity();
  OSArray *casted = OSDynamicCast(OSArray, other);
  if (casted)
    return casted->getCount();
  return 0;
}

unsigned int check_attribute_indirect_propagation(MyArray *arr) {
  OSObject *other = arr->identity();
  OSArray *casted = OSDynamicCast(OSArray, other);
  if (casted)
    return casted->getCount();
  return 0;
}

void check_consumes_this(OSArray *owner) {
  OSArray *arr = new OSArray;
  arr->putIntoArray(owner);
}

void check_consumes_this_with_template(OSArray *owner) {
  OSArray *arr = new OSArray;
  arr->putIntoT(owner);
}

void check_free_no_error() {
  OSArray *arr = OSArray::withCapacity(10);
  arr->retain();
  arr->retain();
  arr->retain();
  arr->free();
}

void check_free_use_after_free() {
  OSArray *arr = OSArray::withCapacity(10); // expected-note{{Call to method 'OSArray::withCapacity' returns an OSObject of type 'OSArray' with a +1 retain count}}
  arr->retain(); // expected-note{{Reference count incremented. The object now has a +2 retain count}}
  arr->free(); // expected-note{{Object released}}
  arr->retain(); // expected-warning{{Reference-counted object is used after it is released}}
                 // expected-note@-1{{Reference-counted object is used after it is released}}
}

unsigned int check_leak_explicit_new() {
  OSArray *arr = new OSArray; // expected-note{{Operator 'new' returns an OSObject of type 'OSArray' with a +1 retain count}}
  return arr->getCount(); // expected-note{{Object leaked: object allocated and stored into 'arr' is not referenced later in this execution path and has a retain count of +1}}
                          // expected-warning@-1{{Potential leak of an object stored into 'arr'}}
}

unsigned int check_leak_factory() {
  OSArray *arr = OSArray::withCapacity(10); // expected-note{{Call to method 'OSArray::withCapacity' returns an OSObject of type 'OSArray' with a +1 retain count}}
  return arr->getCount(); // expected-note{{Object leaked: object allocated and stored into 'arr' is not referenced later in this execution path and has a retain count of +1}}
                          // expected-warning@-1{{Potential leak of an object stored into 'arr'}}
}

void check_get_object() {
  OSObject::getObject();
}

void check_Get_object() {
  OSObject::GetObject();
}

void check_custom_iterator_rule(OSArray *arr) {
  OSIterator *it = arr->getIterator();
  it->release();
}

void check_iterator_leak(OSArray *arr) {
  arr->getIterator(); // expected-note{{Call to method 'OSArray::getIterator' returns an OSObject of type 'OSIterator' with a +1 retain count}}
} // expected-note{{Object leaked: allocated object of type 'OSIterator' is not referenced later}}
  // expected-warning@-1{{Potential leak of an object of type 'OSIterator}}'

void check_no_invalidation() {
  OSArray *arr = OSArray::withCapacity(10); // expected-note{{Call to method 'OSArray::withCapacity' returns an OSObject of type 'OSArray' with a +1 retain count}}
  OtherStruct::doNothingToArray(arr);
} // expected-warning{{Potential leak of an object stored into 'arr'}}
  // expected-note@-1{{Object leaked}}

void check_no_invalidation_other_struct() {
  OSArray *arr = OSArray::withCapacity(10); // expected-note{{Call to method 'OSArray::withCapacity' returns an OSObject of type 'OSArray' with a +1 retain count}}
  OtherStruct other(arr); // expected-warning{{Potential leak}}
                          // expected-note@-1{{Object leaked}}
}

struct ArrayOwner : public OSObject {
  OSArray *arr;
  ArrayOwner(OSArray *arr) : arr(arr) {}

  static ArrayOwner* create(OSArray *arr) {
    return new ArrayOwner(arr);
  }

  OSArray *getArray() {
    return arr;
  }

  OSArray *createArray() {
    return OSArray::withCapacity(10);
  }

  OSArray *createArraySourceUnknown();

  OSArray *getArraySourceUnknown();
};

OSArray *generateArray() {
  return OSArray::withCapacity(10); // expected-note{{Call to method 'OSArray::withCapacity' returns an OSObject of type 'OSArray' with a +1 retain count}}
                                    // expected-note@-1{{Call to method 'OSArray::withCapacity' returns an OSObject of type 'OSArray' with a +1 retain count}}
}

unsigned int check_leak_good_error_message() {
  unsigned int out;
  {
    OSArray *leaked = generateArray(); // expected-note{{Calling 'generateArray'}}
                                       // expected-note@-1{{Returning from 'generateArray'}}
    out = leaked->getCount(); // expected-warning{{Potential leak of an object stored into 'leaked'}}
                              // expected-note@-1{{Object leaked: object allocated and stored into 'leaked' is not referenced later in this execution path and has a retain count of +1}}
  }
  return out;
}

unsigned int check_leak_msg_temporary() {
  return generateArray()->getCount(); // expected-warning{{Potential leak of an object}}
                                      // expected-note@-1{{Calling 'generateArray'}}
                                      // expected-note@-2{{Returning from 'generateArray'}}
                                      // expected-note@-3{{Object leaked: allocated object of type 'OSArray' is not referenced later in this execution path and has a retain count of +1}}
}

void check_confusing_getters() {
  OSArray *arr = OSArray::withCapacity(10);

  ArrayOwner *AO = ArrayOwner::create(arr);
  AO->getArray();

  AO->release();
  arr->release();
}

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

unsigned int check_dynamic_cast_no_null_on_orig(OSObject *obj) {
  OSArray *arr = OSDynamicCast(OSArray, obj);
  if (arr) {
    return arr->getCount();
  } else {

    // The fact that dynamic cast has failed should not imply that
    // the input object was null.
    return obj->foo(); // no-warning
  }
}

void check_dynamic_cast_null_branch(OSObject *obj) {
  OSArray *arr1 = OSArray::withCapacity(10); // expected-note{{Call to method 'OSArray::withCapacity' returns an OSObject}}
  OSArray *arr = OSDynamicCast(OSArray, obj); // expected-note{{Assuming dynamic cast returns null due to type mismatch}}
  if (!arr) // expected-note{{'arr' is null}}
            // expected-note@-1{{Taking true branch}}
    return; // expected-warning{{Potential leak of an object stored into 'arr1'}}
            // expected-note@-1{{Object leaked}}
  arr1->release();
}

void check_dynamic_cast_null_check() {
  OSArray *arr = OSDynamicCast(OSArray, OSObject::generateObject(1)); // expected-note{{Call to method 'OSObject::generateObject' returns an OSObject}}
    // expected-warning@-1{{Potential leak of an object}}
    // expected-note@-2{{Object leaked}}
    // expected-note@-3{{Assuming dynamic cast returns null due to type mismatch}}
  if (!arr)
    return;
  arr->release();
}

void use_after_release() {
  OSArray *arr = OSArray::withCapacity(10); // expected-note{{Call to method 'OSArray::withCapacity' returns an OSObject of type 'OSArray' with a +1 retain count}}
  arr->release(); // expected-note{{Object released}}
  arr->getCount(); // expected-warning{{Reference-counted object is used after it is released}}
                   // expected-note@-1{{Reference-counted object is used after it is released}}
}

void potential_leak() {
  OSArray *arr = OSArray::withCapacity(10); // expected-note{{Call to method 'OSArray::withCapacity' returns an OSObject of type 'OSArray' with a +1 retain count}}
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

unsigned int no_warning_on_getter(ArrayOwner *owner) {
  OSArray *arr = owner->getArray();
  return arr->getCount();
}

unsigned int warn_on_overrelease(ArrayOwner *owner) {
  // FIXME: summaries are not applied in case the source of the getter/setter
  // is known.
  // rdar://45681203
  OSArray *arr = owner->getArray();
  arr->release();
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
  OSArray *arr = owner->getArraySourceUnknown(); // expected-note{{Call to method 'ArrayOwner::getArraySourceUnknown' returns an OSObject of type 'OSArray' with a +0 retain count}}
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

OSObject *getObject();
typedef bool (^Blk)(OSObject *);

void test_escape_to_unknown_block(Blk blk) {
  blk(getObject()); // no-crash
}

using OSObjectPtr = os::smart_ptr<OSObject>;

void test_smart_ptr_uaf() {
  OSObject *obj = new OSObject; // expected-note{{Operator 'new' returns an OSObject of type 'OSObject' with a +1 retain count}}
  {
    OSObjectPtr p(obj); // expected-note{{Calling constructor for 'smart_ptr<OSObject>'}}
   // expected-note@-1{{Returning from constructor for 'smart_ptr<OSObject>'}}
    // expected-note@os_smart_ptr.h:13{{Field 'pointer' is non-null}}
    // expected-note@os_smart_ptr.h:13{{Taking true branch}}
    // expected-note@os_smart_ptr.h:14{{Calling 'smart_ptr::_retain'}}
    // expected-note@os_smart_ptr.h:71{{Reference count incremented. The object now has a +2 retain count}}
    // expected-note@os_smart_ptr.h:14{{Returning from 'smart_ptr::_retain'}}
  } // expected-note{{Calling '~smart_ptr'}}
  // expected-note@os_smart_ptr.h:35{{Field 'pointer' is non-null}}
  // expected-note@os_smart_ptr.h:35{{Taking true branch}}
  // expected-note@os_smart_ptr.h:36{{Calling 'smart_ptr::_release'}}
  // expected-note@os_smart_ptr.h:76{{Reference count decremented. The object now has a +1 retain count}}
  // expected-note@os_smart_ptr.h:36{{Returning from 'smart_ptr::_release'}}
 // expected-note@-6{{Returning from '~smart_ptr'}}
  obj->release(); // expected-note{{Object released}}
  obj->release(); // expected-warning{{Reference-counted object is used after it is released}}
// expected-note@-1{{Reference-counted object is used after it is released}}
}

void test_smart_ptr_leak() {
  OSObject *obj = new OSObject; // expected-note{{Operator 'new' returns an OSObject of type 'OSObject' with a +1 retain count}}
  {
    OSObjectPtr p(obj); // expected-note{{Calling constructor for 'smart_ptr<OSObject>'}}
   // expected-note@-1{{Returning from constructor for 'smart_ptr<OSObject>'}}
    // expected-note@os_smart_ptr.h:13{{Field 'pointer' is non-null}}
    // expected-note@os_smart_ptr.h:13{{Taking true branch}}
    // expected-note@os_smart_ptr.h:14{{Calling 'smart_ptr::_retain'}}
    // expected-note@os_smart_ptr.h:71{{Reference count incremented. The object now has a +2 retain count}}
    // expected-note@os_smart_ptr.h:14{{Returning from 'smart_ptr::_retain'}}
  } // expected-note{{Calling '~smart_ptr'}}
  // expected-note@os_smart_ptr.h:35{{Field 'pointer' is non-null}}
  // expected-note@os_smart_ptr.h:35{{Taking true branch}}
  // expected-note@os_smart_ptr.h:36{{Calling 'smart_ptr::_release'}}
  // expected-note@os_smart_ptr.h:76{{Reference count decremented. The object now has a +1 retain count}}
  // expected-note@os_smart_ptr.h:36{{Returning from 'smart_ptr::_release'}}
 // expected-note@-6{{Returning from '~smart_ptr'}}
} // expected-warning{{Potential leak of an object stored into 'obj'}}
// expected-note@-1{{Object leaked: object allocated and stored into 'obj' is not referenced later in this execution path and has a retain count of +1}}

void test_smart_ptr_no_leak() {
  OSObject *obj = new OSObject;
  {
    OSObjectPtr p(obj);
  }
  obj->release();
}

OSObject *getRuleViolation() {
  return new OSObject; // expected-warning{{Potential leak of an object of type 'OSObject'}}
// expected-note@-1{{Operator 'new' returns an OSObject of type 'OSObject' with a +1 retain count}}
// expected-note@-2{{Object leaked: allocated object of type 'OSObject' is returned from a function whose name ('getRuleViolation') starts with 'get'}}
}

OSObject *createRuleViolation(OSObject *param) { // expected-note{{Parameter 'param' starts at +0}}
  return param; // expected-warning{{Object with a +0 retain count returned to caller where a +1 (owning) retain count is expected}}
  // expected-note@-1{{Object with a +0 retain count returned to caller where a +1 (owning) retain count is expected}}
}

void test_ostypealloc_correct_diagnostic_name() {
  OSArray *arr = OSTypeAlloc(OSArray); // expected-note{{Call to method 'OSMetaClass::alloc' returns an OSObject of type 'OSArray' with a +1 retain count}}
  arr->retain(); // expected-note{{Reference count incremented. The object now has a +2 retain count}}
  arr->release(); // expected-note{{Reference count decremented. The object now has a +1 retain count}}
} // expected-note{{Object leaked: object allocated and stored into 'arr' is not referenced later in this execution path and has a retain count of +1}}
  // expected-warning@-1{{Potential leak of an object stored into 'arr'}}

void escape_elsewhere(OSObject *obj);

void test_free_on_escaped_object_diagnostics() {
  OSObject *obj = new OSObject; // expected-note{{Operator 'new' returns an OSObject of type 'OSObject' with a +1 retain count}}
  escape_elsewhere(obj); // expected-note{{Object is now not exclusively owned}}
  obj->free(); // expected-note{{'free' called on an object that may be referenced elsewhere}}
  // expected-warning@-1{{'free' called on an object that may be referenced elsewhere}}
}

void test_tagged_retain_no_leak() {
  OSObject *obj = new OSObject;
  obj->taggedRelease();
}

void test_tagged_retain_no_uaf() {
  OSObject *obj = new OSObject;
  obj->taggedRetain();
  obj->release();
  obj->release();
}

class IOService {
public:
  OSObject *somethingMatching(OSObject *table = 0);
};

OSObject *testSuppressionForMethodsEndingWithMatching(IOService *svc,
                                                      OSObject *table = 0) {
  // This probably just passes table through. We should probably not make
  // ptr1 definitely equal to table, but we should not warn about leaks.
  OSObject *ptr1 = svc->somethingMatching(table); // no-warning

  // FIXME: This, however, should follow the Create Rule regardless.
  // We should warn about the leak here.
  OSObject *ptr2 = svc->somethingMatching(); // no-warning

  if (!table)
    table = OSTypeAlloc(OSArray);

  // This function itself ends with "Matching"! Do not warn when we're
  // returning from it at +0.
  return table; // no-warning
}

namespace weird_result {
struct WeirdResult {
  int x, y, z;
};

WeirdResult outParamWithWeirdResult(OS_RETURNS_RETAINED_ON_ZERO OSObject **obj);

WeirdResult testOutParamWithWeirdResult() {
  OSObject *obj;
  return outParamWithWeirdResult(&obj); // no-warning
}
} // namespace weird_result
