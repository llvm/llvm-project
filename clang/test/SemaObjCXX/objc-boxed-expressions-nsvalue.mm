// RUN: %clang_cc1  -fsyntax-only -triple x86_64-apple-macosx10.9 -verify %s

#define BOXABLE __attribute__((objc_boxable))

typedef struct BOXABLE _NSPoint {
  int dummy;
} NSPoint;

typedef struct BOXABLE _NSSize {
  int dummy;
} NSSize;

typedef struct BOXABLE _NSRect {
  int dummy;
} NSRect;

typedef struct BOXABLE _CGPoint {
  int dummy;
} CGPoint;

typedef struct BOXABLE _CGSize {
  int dummy;
} CGSize;

typedef struct BOXABLE _CGRect {
  int dummy;
} CGRect;

typedef struct BOXABLE _NSRange {
  int dummy;
} NSRange;

typedef struct BOXABLE _NSEdgeInsets {
  int dummy;
} NSEdgeInsets;

typedef struct BOXABLE _NSEdgeInsets NSEdgeInsets;

typedef struct _SomeStruct {
  double d;
} SomeStruct;

struct BOXABLE NonTriviallyCopyable {
  double d;
  NonTriviallyCopyable() {}
  NonTriviallyCopyable(const NonTriviallyCopyable &obj) {}
};

void checkNSValueDiagnostic() {
  NSRect rect;
  id value = @(rect); // expected-error{{definition of class NSValue must be available to use Objective-C boxed expressions}}
}

@interface NSValue
+ (NSValue *)valueWithBytes:(const void *)value objCType:(const char *)type;
@end

int main() {
  NSPoint ns_point;
  id ns_point_value = @(ns_point);

  NSSize ns_size;
  id ns_size_value = @(ns_size);

  NSRect ns_rect;
  id ns_rect_value = @(ns_rect);

  CGPoint cg_point;
  id cg_point_value = @(cg_point);

  CGSize cg_size;
  id cg_size_value = @(cg_size);

  CGRect cg_rect;
  id cg_rect_value = @(cg_rect);

  NSRange ns_range;
  id ns_range_value = @(ns_range);

  NSEdgeInsets edge_insets;
  id edge_insets_object = @(edge_insets);

  SomeStruct s;
  id err = @(s); // expected-error{{illegal type 'SomeStruct' (aka '_SomeStruct') used in a boxed expression}}

  NonTriviallyCopyable ntc;
  id ntcErr = @(ntc); // expected-error{{non-trivially copyable type 'NonTriviallyCopyable' cannot be used in a boxed expression}}
}

CGRect getRect() {
  CGRect r;
  return r;
}

SomeStruct getSomeStruct() {
  SomeStruct s;
  return s;
}

void rvalue() {
  id rv_rect = @(getRect());
  id rv_some_struct = @(getSomeStruct()); // expected-error {{illegal type 'SomeStruct' (aka '_SomeStruct') used in a boxed expression}}
}

template <class T> id box(T value) { return @(value); } // expected-error{{non-trivially copyable type 'NonTriviallyCopyable' cannot be used in a boxed expression}}
void test_template_1(NSRect rect, NonTriviallyCopyable ntc) {
 id x = box(rect);
 id y = box(ntc);  // expected-note{{in instantiation of function template specialization 'box<NonTriviallyCopyable>' requested here}}
}

template <unsigned i> id boxRect(NSRect rect) { return @(rect); }
template <unsigned i> id boxNTC(NonTriviallyCopyable ntc) { return @(ntc); }  // expected-error{{non-trivially copyable type 'NonTriviallyCopyable' cannot be used in a boxed expression}}
void test_template_2(NSRect rect, NonTriviallyCopyable ntc) {
 id x = boxRect<0>(rect);
 id y = boxNTC<0>(ntc);
}


