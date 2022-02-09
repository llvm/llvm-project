// RUN: %clang_cc1 -fsyntax-only -verify -fblocks %s

void takevoidptr(void*);


@interface Foo
- iMethod;
+ cMethod;
@end

@interface A
+ superClassMethod;
- (void)instanceMethod;
@end

@interface B : A
- (void)instanceMethod;
+ classMethod;
@end

@implementation B

- (void)instanceMethod {
  [super iMethod]; // expected-warning{{'A' may not respond to 'iMethod'}}
  
  // Use of super in a block is ok and does codegen to the right thing.
  // rdar://7852959
  takevoidptr(^{
    [super instanceMethod];
  });
}

+ classMethod {
  [super cMethod]; // expected-warning{{method '+cMethod' not found (return type defaults to 'id')}}
  
  id X[] = { [ super superClassMethod] };
  id Y[] = {
    [ super.superClassMethod iMethod],
    super.superClassMethod,
    (id)super.superClassMethod  // not a cast of super: rdar://7853261
  };
  return 0;
}
@end

@interface XX
- m;
@end

void f(id super) {
  [super m];
}
void f0(int super) {
  [super m]; // expected-warning{{receiver type 'int' is not 'id'}}
}
void f1(id puper) {  // expected-note {{'puper' declared here}}
  [super m]; // expected-error{{use of undeclared identifier 'super'}}
}

// radar 7400691
typedef Foo super;

typedef Foo FooTD;

void test(void) {
  [FooTD cMethod];
  [super cMethod];
}

struct SomeStruct {
  int X;
};

int test2(void) {
  struct SomeStruct super = { 0 };
  return super.X;
}

int test3(void) {
  id super = 0;
  [(B*)super instanceMethod];
  int *s1 = (int*)super;
  
  id X[] = { [ super superClassMethod] };
  return 0;
}
