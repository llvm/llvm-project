// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s

@interface A
-(float) x;	// expected-note {{declared here}}
@property int x; // expected-warning {{type of property 'x' does not match type of accessor 'x'}}
@end

@interface A (Cat)
@property int moo;	// expected-note {{previous definition is here}}
@end

@implementation A (Cat)
-(int) moo {
  return 0;
}
-(void) setMoo: (float) x { // expected-warning {{conflicting parameter types in implementation of 'setMoo:': 'int' vs 'float'}}
}
@end


typedef int T[2];
typedef void (F)(void);

@interface C
@property(assign) T p2;  // expected-error {{property cannot have array or function type 'T'}}

@property(assign) F f2; // expected-error {{property cannot have array or function type 'F'}}

@end


@class SSyncSet;

@interface SPeer
  @property(nonatomic,readonly,retain) SSyncSet* syncSet;
@end

@class SSyncSet_iDisk;

@interface SPeer_iDisk_remote1 : SPeer
- (SSyncSet_iDisk*) syncSet; // expected-note {{declared here}}
@end

@interface SPeer_iDisk_local
- (SSyncSet_iDisk*) syncSet;
@end

@interface SSyncSet
@end

@interface SSyncSet_iDisk
@property(nonatomic,readonly,retain) SPeer_iDisk_local* localPeer;
@end

@interface SPeer_iDisk_remote1 (protected)
@end

@implementation SPeer_iDisk_remote1 (protected)
- (id) preferredSource1
{
  return self.syncSet.localPeer; // expected-warning {{type of property 'syncSet' does not match type of accessor 'syncSet'}}
}
@end

@interface NSArray @end

@interface NSMutableArray : NSArray
@end

@interface Class1 
{
 NSMutableArray* pieces;
 NSArray* first;
}

@property (readonly) NSArray* pieces; // expected-warning {{type of property 'pieces' does not match type of accessor 'pieces'}}
@property (readonly) NSMutableArray* first;

- (NSMutableArray*) pieces; // expected-note 2 {{declared here}}
- (NSArray*) first;
@end

@interface Class2  {
 Class1* container;
}

@end

@implementation Class2

- (id) lastPiece
{
 return container.pieces; // expected-warning {{type of property 'pieces' does not match type of accessor 'pieces'}}
}

- (id)firstPeice
{
  return container.first;
}
@end

