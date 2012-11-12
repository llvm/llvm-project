// RUN: %clang_cc1 -analyze -analyzer-checker=alpha.core -analyzer-checker=deadcode.DeadStores -verify %s
//
// This test exercises the live variables analysis (LiveVariables.cpp).
// The case originally identified a non-termination bug.
//
typedef signed char BOOL;
typedef unsigned int NSUInteger;
typedef struct _NSZone NSZone;
@protocol NSObject  - (BOOL)isEqual:(id)object; @end
@interface NSObject <NSObject> {} @end
extern id NSAllocateObject(Class aClass, NSUInteger extraBytes, NSZone *zone);
@class NSArray;
@class NSMutableArray, NSIndexSet, NSView, NSPredicate, NSString, NSViewAnimation, NSTimer; // expected-note{{forward declaration of class here}}
@interface FooBazController : NSObject {}
@end
typedef struct {} TazVersion;
@class TazNode;
@interface TazGuttenberg : NSObject {} typedef NSUInteger BugsBunnyType; @end
@interface FooBaz : NSObject {}
@property (nonatomic) BugsBunnyType matchType;
@property (nonatomic, retain) NSArray *papyrus; @end
@implementation FooBazController
- (NSArray *)excitingStuff:(FooBaz *)options {
  BugsBunnyType matchType = options.matchType;
  NSPredicate *isSearchablePredicate = [NSPredicate predicateWithFormat:@"isSearchable == YES"]; // expected-warning{{receiver 'NSPredicate' is a forward class and corresponding}} // expected-warning{{return type defaults to 'id'}}
  for (TazGuttenberg *Guttenberg in options.papyrus) {
    NSArray *GuttenbergNodes = [Guttenberg nodes]; // expected-warning{{return type defaults to 'id'}}
    NSArray *searchableNodes = [GuttenbergNodes filteredArrayUsingPredicate:isSearchablePredicate]; // expected-warning{{return type defaults to 'id'}}
    for (TazNode *node in searchableNodes) {
      switch (matchType) {
        default: break;
      }
    }
  }
  while (1) {}
}
@end
