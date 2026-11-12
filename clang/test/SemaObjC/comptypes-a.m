// RUN: %clang_cc1 -fsyntax-only -Wmethod-signatures -verify -pedantic -Wno-objc-root-class %s
typedef signed char BOOL;
typedef int NSInteger;

@class NSString;

@protocol PBXCompletionItem
- (NSString *) name;
- (NSInteger)priority;
@end

extern NSInteger codeAssistantCaseCompareItems(id a, id b, void *context);

NSInteger codeAssistantCaseCompareItems(id<PBXCompletionItem> a, id<PBXCompletionItem> b, void *context)
{
  return 0;
}

@interface TedWantsToVerifyObjCDoesTheRightThing

- compareThis:(int)a withThat:(id)b;  // expected-note {{previous definition is here}} \
				      // expected-note {{previous definition is here}}

@end

@implementation TedWantsToVerifyObjCDoesTheRightThing

- compareThis:(id<PBXCompletionItem>)
    a // expected-warning {{conflicting parameter types in implementation of 'compareThis:withThat:': 'int' vs 'id<PBXCompletionItem>'}}
     withThat:(id<PBXCompletionItem>)b { // expected-warning {{conflicting parameter types in implementation of 'compareThis:withThat:': 'id' vs 'id<PBXCompletionItem>'}}
  return self;
}

@end
