// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyze-function=test_function_pointer_forms -verify=forms %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyze-function=test_mutable_pointer -verify=mutable %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyze-function=test_no_false_positive -verify=fp %s

void clang_analyzer_checkInlined(int);
void clang_analyzer_eval(int);

typedef void (*callback)(unsigned);

static void decay_target(unsigned value) {
  clang_analyzer_checkInlined(value == 1); // forms-warning{{TRUE}}
}

static callback const decayCallback = decay_target;

static void address_target(unsigned value) {
  clang_analyzer_checkInlined(value == 2); // forms-warning{{TRUE}}
}

static callback const addressCallback = &address_target;

static void cast_target(unsigned value) {
  clang_analyzer_checkInlined(value == 3); // forms-warning{{TRUE}}
}

static callback const castCallback = (callback)cast_target;

void test_function_pointer_forms(void) {
  clang_analyzer_eval(decayCallback == decay_target); // forms-warning{{TRUE}}
  decay_target(1);

  clang_analyzer_eval(addressCallback == address_target); // forms-warning{{TRUE}}
  address_target(2);

  clang_analyzer_eval(castCallback == cast_target); // forms-warning{{TRUE}}
  castCallback(3);
}

static callback mutableCallback = decay_target;

void test_mutable_pointer(void) {
  clang_analyzer_eval(mutableCallback == decay_target); // mutable-warning{{UNKNOWN}}
}

struct St {
  int f;
};

static struct St sts[4];

static void helper(unsigned id) {
  struct St *s = 0;

  if (id < 1)
    return;
  if (id < 5)
    s = &sts[id-1];
  else if (id == 5)
    return;

  s->f = 60;
}

static void notify(unsigned id) {
  clang_analyzer_checkInlined(id >= 1 && id <= 4); // fp-warning{{TRUE}}
  helper(id);
}

static callback const callbackFn = notify;

void test_no_false_positive(unsigned id) {
  if (id < 1 || id > 4)
    return;

  clang_analyzer_eval(callbackFn == notify); // fp-warning{{TRUE}}
  callbackFn(id);
}
