// RUN: %clang_cc1 -pedantic -fsyntax-only -verify -Wno-objc-root-class %s
typedef signed char BOOL;

@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;

@protocol NSObject
- (BOOL)isEqual:(id)object;
@end

@protocol NSCoding
- (void)encodeWithCoder:(NSCoder *)aCoder;
@end

@interface NSObject <NSObject> {}
@end

typedef float CGFloat;

@interface NSResponder : NSObject <NSCoding> {}
@end

@protocol XCSelectionSource;

@interface XCSelection : NSResponder {}
- (NSObject <XCSelectionSource> *) source;
@end

extern NSString * const XCActiveSelectionLevel;

@interface XCActionManager : NSResponder {}
+defaultActionManager;
-selectionAtLevel:(NSString *const)s;
@end

@implementation XDMenuItemsManager // expected-warning {{cannot find interface declaration for 'XDMenuItemsManager'}}
+ (void)initialize {
  id<XCSelectionSource, NSObject> source = 
    [[[XCActionManager defaultActionManager] selectionAtLevel:XCActiveSelectionLevel] source];
}
@end

@protocol NSTextStorageDelegate;
@class NSNotification;

@interface NSTextStorage : NSObject

- (void)setDelegate:(id <NSTextStorageDelegate>)delegate; // expected-note{{passing argument to parameter 'delegate' here}}
- (id <NSTextStorageDelegate>)delegate;

@end

@protocol NSTextStorageDelegate <NSObject>
@optional

- (void)textStorageWillProcessEditing:(NSNotification *)notification;
- (void)textStorageDidProcessEditing:(NSNotification *)notification;

@end

@interface SKTText : NSObject {
    @private


    NSTextStorage *_contents;
}
@end

@implementation SKTText


- (NSTextStorage *)contents {
 [_contents setDelegate:self]; // expected-warning {{sending 'SKTText *' to parameter of incompatible type 'id<NSTextStorageDelegate>'}}
 return 0;
}

@end
