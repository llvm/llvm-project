// RUN: rm -rf %t
// RUN: %clang_cc1 -objcmt-migrate-instancetype -mt-migrate-directory %t %s -x objective-c -fobjc-runtime-has-weak -fobjc-arc -triple x86_64-apple-darwin11
// RUN: c-arcmt-test -mt-migrate-directory %t | arcmt-test -verify-transformed-files %s.result
// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -fsyntax-only -x objective-c -fobjc-runtime-has-weak -fobjc-arc %s.result

typedef unsigned int NSUInteger;
typedef int NSInteger;
typedef char BOOL;
@class NSData, NSError, NSProtocolChecker, NSObject;
@class NSPortNameServer, NSTimeZone;

@interface NSMutableString
@end

@interface NSString @end

@class NSString, NSURL;
@interface NSString (NSStringDeprecated)
+ (id)stringWithContentsOfFile:(NSString *)path __attribute__((availability(macosx,introduced=10.0 ,message="" )));
+ (id)stringWithContentsOfURL:(NSURL *)url __attribute__((availability(macosx,introduced=10.0 ,message="" )));
+ (id)stringWithCString:(const char *)bytes length:(NSUInteger)length __attribute__((availability(macosx,introduced=10.0 ,message="" )));
+ (id)stringWithCString:(const char *)bytes __attribute__((availability(macosx,introduced=10.0 ,message="" )));
@end


typedef enum NSURLBookmarkResolutionOptions {
                Bookmark
} NSURLBookmarkResolutionOptions;

@interface NSURL
+ (id)URLWithString:(NSString *)URLString;
+ (id)URLWithString:(NSString *)URLString relativeToURL:(NSURL *)baseURL;
+ (id)URLByResolvingBookmarkData:(NSData *)bookmarkData options:(NSURLBookmarkResolutionOptions)options relativeToURL:(NSURL *)relativeURL bookmarkDataIsStale:(BOOL *)isStale error:(NSError **)error __attribute__((availability(macosx,introduced=10.6)));
@end

@class NSDictionary;
@interface NSError
+ (id)errorWithDomain:(NSString *)domain code:(NSInteger)code userInfo:(NSDictionary *)dict;
@end


@interface NSMutableString (NSMutableStringExtensionMethods)
+ (id)stringWithCapacity:(NSUInteger)capacity;
@end

@interface NSMutableData
+ (id)dataWithCapacity:(NSUInteger)aNumItems;
+ (id)dataWithLength:(NSUInteger)length;
@end

@interface NSMutableDictionary @end

@interface NSMutableDictionary (NSSharedKeySetDictionary)
+ (id )dictionaryWithSharedKeySet:(id)keyset __attribute__((availability(macosx,introduced=10.8)));
@end

@interface NSProtocolChecker
+ (id)protocolCheckerWithTarget:(NSObject *)anObject protocol:(Protocol *)aProtocol;
@end

@interface NSConnection
+ (id)connectionWithRegisteredName:(NSString *)name host:(NSString *)hostName;
+ (id)connectionWithRegisteredName:(NSString *)name host:(NSString *)hostName usingNameServer:(NSPortNameServer *)server;
@end

@interface NSDate
+ (id)dateWithString:(NSString *)aString __attribute__((availability(macosx,introduced=10.4)));
@end

@interface NSCalendarDate : NSDate
+ (id)calendarDate __attribute__((availability(macosx,introduced=10.4)));
+ (id)dateWithString:(NSString *)description calendarFormat:(NSString *)format locale:(id)locale __attribute__((availability(macosx,introduced=10.4)));
+ (id)dateWithString:(NSString *)description calendarFormat:(NSString *)format __attribute__((availability(macosx,introduced=10.4)));
+ (id)dateWithYear:(NSInteger)year month:(NSUInteger)month day:(NSUInteger)day hour:(NSUInteger)hour minute:(NSUInteger)minute second:(NSUInteger)second timeZone:(NSTimeZone *)aTimeZone __attribute__((availability(macosx,introduced=10.4)));
@end

@interface NSUserDefaults
+ (id) standardUserDefaults;
@end

@interface NSNotificationCenter
+ (id) defaultCenter;
+  sharedCenter;
@end

@interface UIApplication
+ (id)sharedApplication;
+ defaultApplication;
@end

//===----------------------------------------------------------------------===//
// Method name that has a null IdentifierInfo* for its first selector slot.
// This test just makes sure that we handle it.
//===----------------------------------------------------------------------===//
@interface TestNullIdentifier
@end

@implementation TestNullIdentifier
+ (id):(int)x, ... {
  return 0;
}
@end

