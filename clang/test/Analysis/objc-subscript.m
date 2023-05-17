// RUN: %clang_analyze_cc1 -analyzer-checker=core,osx.cocoa.RetainCount -verify -Wno-objc-root-class %s

typedef signed char BOOL;
typedef unsigned int NSUInteger;

@interface NSObject
+(id)alloc;
-(id)init;
-(id)autorelease;
-(id)copy;
-(id)retain;
@end

@interface Subscriptable : NSObject
- (void)setObject:(id)obj atIndexedSubscript:(NSUInteger)index;
- (id)objectAtIndexedSubscript:(NSUInteger)index;

- (void)setObject:(id)obj forKeyedSubscript:(id)key;
- (id)objectForKeyedSubscript:(id)key;
@end

@interface Test : Subscriptable
@end

@implementation Test

// <rdar://problem/6946338> for subscripting
- (id)storeDoesNotRetain {
  Test *cell = [[[Test alloc] init] autorelease];

  NSObject *string1 = [[NSObject alloc] init]; // expected-warning {{Potential leak}}
  cell[0] = string1;
  cell[self] = string1;
  cell[string1] = self;

  return cell;
}

// <rdar://problem/8824416> for subscripting
- (id)getDoesNotRetain:(BOOL)keyed {
  if (keyed)
    return [self[self] autorelease]; // expected-warning{{Object autoreleased too many times}}
  else
    return [self[0] autorelease]; // expected-warning{{Object autoreleased too many times}}
}

// <rdar://problem/9241180> for subscripting
- (id)testUninitializedObject:(BOOL)keyed {
  Test *o;
  if (keyed) {
    if (o[self]) // expected-warning {{Subscript access on an uninitialized object pointer}}
      return o; // no-warning (sink)
  } else {
    if (o[0]) // expected-warning {{Subscript access on an uninitialized object pointer}}
      return o; // no-warning (sink)
  }
  return self;
}

- (void)testUninitializedArgument:(id)input testCase:(unsigned)testCase {
  NSUInteger i;
  id o;

  switch (testCase) {
  case 0:
    self[0] = o; // expected-warning {{Argument for subscript setter is an uninitialized value}}
    break;
  case 1:
    self[i] = input; // expected-warning {{Subscript index is an uninitialized value}}
    break;
  case 2:
    (void)self[i]; // expected-warning {{Subscript index is an uninitialized value}}
    break;
  case 3:
    self[input] = o; // expected-warning {{Argument for subscript setter is an uninitialized value}}
    break;
  case 4:
    self[o] = input; // expected-warning {{Subscript index is an uninitialized value}}
    break;
  case 5:
    (void)self[o]; // expected-warning {{Subscript index is an uninitialized value}}
    break;
  default:
    break;
  }

}

@end
