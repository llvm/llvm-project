// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection \
// RUN:   -verify %s

void clang_analyzer_checkInlined(bool);
void clang_analyzer_eval(bool);

typedef void (*Callback)(unsigned);
typedef void (&CallbackRef)(unsigned);

static int Storage;

static void pointerTarget(unsigned Value) {
  int *Ptr = nullptr;

  if (Value == 1)
    Ptr = &Storage;

  clang_analyzer_checkInlined(Value == 1); // expected-warning{{TRUE}}
  *Ptr = 0;
}

static void referenceTarget(unsigned Value) {
  int *Ptr = nullptr;

  if (Value == 1)
    Ptr = &Storage;

  clang_analyzer_checkInlined(Value == 1); // expected-warning{{TRUE}}
  *Ptr = 0;
}

static Callback const ConstPointer = pointerTarget;
static Callback const AddressPointer = &pointerTarget;
static Callback const CastPointer = (Callback)pointerTarget;
static Callback MutablePointer = pointerTarget;
static CallbackRef Reference = referenceTarget;

extern CallbackRef ExternalReference;

void testPointers(unsigned Value) {
  if (Value != 1)
    return;

  clang_analyzer_eval(ConstPointer == pointerTarget); // expected-warning{{TRUE}}
  ConstPointer(Value);
  clang_analyzer_eval(AddressPointer == pointerTarget); // expected-warning{{TRUE}}
  clang_analyzer_eval(CastPointer == pointerTarget); // expected-warning{{TRUE}}
  clang_analyzer_eval(MutablePointer == pointerTarget); // expected-warning{{UNKNOWN}}
}

void testReference(unsigned Value) {
  if (Value != 1)
    return;

  clang_analyzer_eval(Reference == referenceTarget); // expected-warning{{TRUE}}
  Reference(Value);
}

void testExternalReference() {
  clang_analyzer_eval(ExternalReference == referenceTarget); // expected-warning{{UNKNOWN}}

  Callback Before = ExternalReference;
  clang_analyzer_eval(ExternalReference == Before); // expected-warning{{TRUE}}
}
