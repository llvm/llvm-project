// RUN: %clang_cc1 -fsyntax-only -verify -std=c++14 %s

typedef __SIZE_TYPE__ size_t;

namespace basic {
// Ensuring that __bos can be used in constexpr functions without anything
// sketchy going on...
constexpr int bos0() {
  int k = 5;
  char cs[10] = {};
  return __builtin_object_size(&cs[k], 0);
}

constexpr int bos1() {
  int k = 5;
  char cs[10] = {};
  return __builtin_object_size(&cs[k], 1);
}

constexpr int bos2() {
  int k = 5;
  char cs[10] = {};
  return __builtin_object_size(&cs[k], 2);
}

constexpr int bos3() {
  int k = 5;
  char cs[10] = {};
  return __builtin_object_size(&cs[k], 3);
}

static_assert(bos0() == sizeof(char) * 5, "");
static_assert(bos1() == sizeof(char) * 5, "");
static_assert(bos2() == sizeof(char) * 5, "");
static_assert(bos3() == sizeof(char) * 5, "");
}

namespace in_enable_if {
// The code that prompted these changes was __bos in enable_if

void copy5CharsInto(char *buf) // expected-note{{candidate}}
    __attribute__((enable_if(__builtin_object_size(buf, 0) != -1 &&
                                 __builtin_object_size(buf, 0) > 5,
                             "")));

// We use different EvalModes for __bos with type 0 versus 1. Ensure 1 works,
// too...
void copy5CharsIntoStrict(char *buf) // expected-note{{candidate}}
    __attribute__((enable_if(__builtin_object_size(buf, 1) != -1 &&
                                 __builtin_object_size(buf, 1) > 5,
                             "")));

struct LargeStruct {
  int pad;
  char buf[6];
  int pad2;
};

struct SmallStruct {
  int pad;
  char buf[5];
  int pad2;
};

void noWriteToBuf() {
  char buf[6];
  copy5CharsInto(buf);

  LargeStruct large;
  copy5CharsIntoStrict(large.buf);
}

void initTheBuf() {
  char buf[6] = {};
  copy5CharsInto(buf);

  LargeStruct large = {0, {}, 0};
  copy5CharsIntoStrict(large.buf);
}

int getI();
void initTheBufWithALoop() {
  char buf[6] = {};
  for (unsigned I = getI(); I != sizeof(buf); ++I)
    buf[I] = I;
  copy5CharsInto(buf);

  LargeStruct large;
  for (unsigned I = getI(); I != sizeof(buf); ++I)
    large.buf[I] = I;
  copy5CharsIntoStrict(large.buf);
}

void tooSmallBuf() {
  char buf[5];
  copy5CharsInto(buf); // expected-error{{no matching function for call}}

  SmallStruct small;
  copy5CharsIntoStrict(small.buf); // expected-error{{no matching function for call}}
}
}

namespace InvalidBase {
  // Ensure this doesn't crash.
  struct S { const char *name; };
  S invalid_base();
  constexpr size_t bos_name = __builtin_object_size(invalid_base().name, 1);
}
