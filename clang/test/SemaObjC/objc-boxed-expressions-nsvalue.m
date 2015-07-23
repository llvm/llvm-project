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

struct _NSEdgeInsets {
  int dummy;
};

typedef struct BOXABLE _NSEdgeInsets NSEdgeInsets;

typedef struct _SomeStruct {
  double d;
} SomeStruct;

typedef union BOXABLE _BoxableUnion {
  int dummy;
} BoxableUnion;

void checkNSValueDiagnostic() {
  NSRect rect;
  id value = @(rect); // expected-error{{NSValue must be available to use Objective-C boxed expressions}}
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

  BoxableUnion boxable_union;
  id boxed_union = @(boxable_union);

  SomeStruct s;
  id err = @(s); // expected-error{{illegal type 'SomeStruct' (aka 'struct _SomeStruct') used in a boxed expression}}
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
  id rv_some_struct = @(getSomeStruct()); // expected-error {{illegal type 'SomeStruct' (aka 'struct _SomeStruct') used in a boxed expression}}
}
