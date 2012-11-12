// RUN: %clang_cc1  -Wstrict-selector-match -fsyntax-only -verify %s

@interface Foo
-(int) method; // expected-note {{using}}
@end

@interface Bar
-(float) method;	// expected-note {{also found}}
@end

int main() { [(id)0 method]; } // expected-warning {{multiple methods named 'method' found}}

@interface Object @end

@interface Class1
- (void)setWindow:(Object *)wdw;	// expected-note {{using}}
@end

@interface Class2
- (void)setWindow:(Class1 *)window;	// expected-note {{also found}}
@end

id foo(void) {
  Object *obj = 0;
  id obj2 = obj;
  [obj setWindow:0]; 	// expected-warning {{Object' may not respond to 'setWindow:'}}
  [obj2 setWindow:0]; // expected-warning {{multiple methods named 'setWindow:' found}}
  return obj;
}

@protocol MyObject
- (id)initWithData:(Object *)data;	// expected-note {{using}} \
					// expected-note {{passing argument to parameter 'data' here}}
@end

@protocol SomeOther
- (id)initWithData:(int)data;	// expected-note {{also found}}
@end

@protocol MyCoding
- (id)initWithData:(id<MyObject, MyCoding>)data;	// expected-note {{also found}}
@end

@interface NTGridDataObject: Object <MyCoding>
{
    Object<MyCoding> *_data;
}
+ (NTGridDataObject*)dataObject:(id<MyObject, MyCoding>)data;
@end

@implementation NTGridDataObject
- (id)initWithData:(id<MyObject, MyCoding>)data {
  return data;
}
+ (NTGridDataObject*)dataObject:(id<MyObject, MyCoding>)data
{
    NTGridDataObject *result = [(id)0 initWithData:data]; // expected-warning {{multiple methods named 'initWithData:' found}} \
    expected-warning {{sending 'id<MyObject,MyCoding>' to parameter of incompatible type 'Object *'}}
    return result;
}
@end

@interface Base
- (unsigned)port;
@end

@interface Derived: Base
- (Object *)port;
+ (Protocol *)port;
@end

void foo1(void) {
  [(Class)0 port]; // OK - gcc issues warning but there is only one Class method so no ambiguity to warn
}

