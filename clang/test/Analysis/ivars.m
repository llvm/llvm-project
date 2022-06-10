// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -analyzer-store=region -fblocks -verify -Wno-objc-root-class -analyzer-config eagerly-assume=false %s

void clang_analyzer_eval(int);

@interface Root {
@public
  int uniqueID;
}

- (void)refreshID;
@end

void testInvalidation(Root *obj) {
  int savedID = obj->uniqueID;
  clang_analyzer_eval(savedID == obj->uniqueID); // expected-warning{{TRUE}}

  [obj refreshID];
  clang_analyzer_eval(savedID == obj->uniqueID); // expected-warning{{UNKNOWN}}
}


@interface Child : Root
@end

@implementation Child
- (void)testSuperInvalidation {
  int savedID = self->uniqueID;
  clang_analyzer_eval(savedID == self->uniqueID); // expected-warning{{TRUE}}

  [super refreshID];
  clang_analyzer_eval(savedID == self->uniqueID); // expected-warning{{UNKNOWN}}
}
@end


@interface ManyIvars {
  struct S { int a, b; } s;
  int c;
  int d;
}
@end

struct S makeS(void);

@implementation ManyIvars

- (void)testMultipleIvarInvalidation:(int)useConstraints {
  if (useConstraints) {
    if (s.a != 1) return;
    if (s.b != 2) return;
    if (c != 3) return;
    if (d != 4) return;
    return;
  } else {
    s.a = 1;
    s.b = 2;
    c = 3;
    d = 4;
  }

  clang_analyzer_eval(s.a == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(s.b == 2); // expected-warning{{TRUE}}
  clang_analyzer_eval(c == 3); // expected-warning{{TRUE}}
  clang_analyzer_eval(d == 4); // expected-warning{{TRUE}}

  d = 0;

  clang_analyzer_eval(s.a == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(s.b == 2); // expected-warning{{TRUE}}
  clang_analyzer_eval(c == 3); // expected-warning{{TRUE}}
  clang_analyzer_eval(d == 0); // expected-warning{{TRUE}}

  d = 4;
  s = makeS();

  clang_analyzer_eval(s.a == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(s.b == 2); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(c == 3); // expected-warning{{TRUE}}
  clang_analyzer_eval(d == 4); // expected-warning{{TRUE}}

  s.a = 1;

  clang_analyzer_eval(s.a == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(s.b == 2); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(c == 3); // expected-warning{{TRUE}}
  clang_analyzer_eval(d == 4); // expected-warning{{TRUE}}
}

+ (void)testMultipleIvarInvalidation:(int)useConstraints
                           forObject:(ManyIvars *)obj {
  if (useConstraints) {
    if (obj->s.a != 1) return;
    if (obj->s.b != 2) return;
    if (obj->c != 3) return;
    if (obj->d != 4) return;
    return;
  } else {
    obj->s.a = 1;
    obj->s.b = 2;
    obj->c = 3;
    obj->d = 4;
  }

  clang_analyzer_eval(obj->s.a == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(obj->s.b == 2); // expected-warning{{TRUE}}
  clang_analyzer_eval(obj->c == 3); // expected-warning{{TRUE}}
  clang_analyzer_eval(obj->d == 4); // expected-warning{{TRUE}}

  obj->d = 0;

  clang_analyzer_eval(obj->s.a == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(obj->s.b == 2); // expected-warning{{TRUE}}
  clang_analyzer_eval(obj->c == 3); // expected-warning{{TRUE}}
  clang_analyzer_eval(obj->d == 0); // expected-warning{{TRUE}}

  obj->d = 4;
  obj->s = makeS();

  clang_analyzer_eval(obj->s.a == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(obj->s.b == 2); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(obj->c == 3); // expected-warning{{TRUE}}
  clang_analyzer_eval(obj->d == 4); // expected-warning{{TRUE}}

  obj->s.a = 1;

  clang_analyzer_eval(obj->s.a == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(obj->s.b == 2); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(obj->c == 3); // expected-warning{{TRUE}}
  clang_analyzer_eval(obj->d == 4); // expected-warning{{TRUE}}
}

@end


int testNull(Root *obj) {
  if (obj) return 0;

  int *x = &obj->uniqueID;
  return *x; // expected-warning{{Dereference of null pointer (loaded from variable 'x')}}
}
