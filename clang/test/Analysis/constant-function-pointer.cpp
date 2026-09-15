// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection \
// RUN:   -verify %s

template <class T>
void clang_analyzer_dump(T);
void clang_analyzer_eval(bool);

using Callback = void (*)(const char *);
using CallbackRef = void (&)(const char *);

static int Storage;

static void pointerTarget(const char *Value) {
  int *Ptr = nullptr;

  if (Value)
    Ptr = &Storage;

  clang_analyzer_dump(Value); // expected-warning{{"pointer"}}
  *Ptr = 0; // no-warning: Ptr is never null here.
}

static void referenceTarget(const char *Value) {
  int *Ptr = nullptr;

  if (Value)
    Ptr = &Storage;

  clang_analyzer_dump(Value); // expected-warning{{"reference"}}
  *Ptr = 0; // no-warning: Ptr is never null here.
}

static Callback const ConstPointer = pointerTarget;
static Callback const AddressPointer = &pointerTarget;
static Callback const CastPointer = (Callback)pointerTarget;
static Callback MutablePointer = pointerTarget;
static CallbackRef Reference = referenceTarget;

extern CallbackRef ExternalReference;

void testPointers(unsigned Value) {
  clang_analyzer_eval(ConstPointer == pointerTarget); // expected-warning{{TRUE}}
  ConstPointer("pointer");
  clang_analyzer_eval(AddressPointer == pointerTarget); // expected-warning{{TRUE}}
  clang_analyzer_eval(CastPointer == pointerTarget); // expected-warning{{TRUE}}
  clang_analyzer_eval(MutablePointer == pointerTarget); // expected-warning{{UNKNOWN}}
}

void testReference(unsigned Value) {
  clang_analyzer_eval(Reference == referenceTarget); // expected-warning{{TRUE}}
  Reference("reference");
}

void testExternalReference() {
  clang_analyzer_eval(ExternalReference == referenceTarget); // expected-warning{{UNKNOWN}}

  Callback Before = ExternalReference;
  clang_analyzer_eval(ExternalReference == Before); // expected-warning{{TRUE}}
}

void myGlobalFn();
static const bool Truthy = &myGlobalFn;

// Verify that a function pointer converted to bool
// is modeled as `true`, not as FunctionCodeRegion.
void testBoolInitializer() {
  clang_analyzer_dump(Truthy); // expected-warning{{1 U1b}}
}
