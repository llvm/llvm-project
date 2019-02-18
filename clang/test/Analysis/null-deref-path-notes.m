// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-store=region -analyzer-output=text -fblocks -verify -Wno-objc-root-class %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-store=region -analyzer-output=plist-multi-file -fblocks -Wno-objc-root-class %s -o %t
// RUN: cat %t | %diff_plist %S/Inputs/expected-plists/null-deref-path-notes.m.plist -

@interface Root {
@public
  int uniqueID;
}
- (id)initWithID:(int)newID;
- (void)refreshID;
@end

int testNull(Root *obj) {
  if (obj) return 0;
  // expected-note@-1 {{Assuming 'obj' is nil}}
  // expected-note@-2 {{Taking false branch}}

  int *x = &obj->uniqueID; // expected-note{{'x' initialized to a null pointer value}}
  return *x; // expected-warning{{Dereference of null pointer (loaded from variable 'x')}} expected-note{{Dereference of null pointer (loaded from variable 'x')}}
}


@interface Subclass : Root
@end

@implementation Subclass
- (id)initWithID:(int)newID {
  self = [super initWithID:newID]; // expected-note{{Value assigned to 'self'}}
  if (self) return self;
  // expected-note@-1 {{Assuming 'self' is nil}}
  // expected-note@-2 {{Taking false branch}}

  uniqueID = newID; // expected-warning{{Access to instance variable 'uniqueID' results in a dereference of a null pointer (loaded from variable 'self')}} expected-note{{Access to instance variable 'uniqueID' results in a dereference of a null pointer (loaded from variable 'self')}}
  return self;
}

@end

void repeatedStores(int coin) {
  int *p = 0;
  if (coin) {
    // expected-note@-1 {{Assuming 'coin' is 0}}
    // expected-note@-2 {{Taking false branch}}
    extern int *getPointer();
    p = getPointer();
  } else {
    p = 0; // expected-note {{Null pointer value stored to 'p'}}
  }

  *p = 1; // expected-warning{{Dereference of null pointer}} expected-note{{Dereference of null pointer}}
}

@interface WithArrayPtr
- (void) useArray;
@end

@implementation WithArrayPtr {
@public int *p;
}
- (void)useArray {
  p[1] = 2; // expected-warning{{Array access (via ivar 'p') results in a null pointer dereference}}
            // expected-note@-1{{Array access (via ivar 'p') results in a null pointer dereference}}
}
@end

void testWithArrayPtr(WithArrayPtr *w) {
  w->p = 0; // expected-note{{Null pointer value stored to 'p'}}
  [w useArray]; // expected-note{{Calling 'useArray'}}
}

