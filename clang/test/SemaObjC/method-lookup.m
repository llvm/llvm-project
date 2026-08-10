// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s
typedef signed char BOOL;
typedef int NSInteger;

@protocol NSObject
- (BOOL)isEqual:(id)object;
- (BOOL)respondsToSelector:(SEL)s;
@end

@interface NSObject <NSObject> {}
@end

@class NSString, NSData, NSMutableData, NSMutableDictionary, NSMutableArray;

@protocol PBXCompletionItem
- (NSString *) name;
- (NSInteger)priority;
- setPriority:(NSInteger)p;
@end

@implementation PBXCodeAssistant // expected-warning{{cannot find interface declaration for 'PBXCodeAssistant'}}
static NSMutableArray * recentCompletions = ((void *)0);
+ (float) factorForRecentCompletion:(NSString *) completion
{
    for (NSObject<PBXCompletionItem> * item in [self completionItems]) // expected-warning{{method '+completionItems' not found (return type defaults to 'id')}}
    {
        if ([item respondsToSelector:@selector(setPriority:)])
        {
            [(id)item setPriority:[item priority] / [PBXCodeAssistant factorForRecentCompletion:[item name]]];
        }
    }
    return 0;
}
@end

