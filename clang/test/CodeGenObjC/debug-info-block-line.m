// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -emit-llvm -fblocks -fobjc-arc -debug-info-kind=limited -triple x86_64-apple-darwin10 %s -o - | FileCheck %s

// rdar://11562117
typedef unsigned int NSUInteger;
typedef long NSInteger;
typedef signed char BOOL;

#define nil ((void*) 0)
#define YES             ((BOOL)1)
#define NO              ((BOOL)0)

@interface NSObject
- (id)init;
@end

@interface NSError : NSObject
@end

@interface NSString : NSObject
@end

@interface NSString (NSStringExtensionMethods)
- (void)enumerateLinesUsingBlock:(void (^)(NSString *line, BOOL *stop))block;
@end

@interface NSData : NSObject
@end

@interface NSData (ASBase64)
- (NSString *)encodedString:(NSInteger)position;
- (NSData *)compressedData;
@end

typedef void (^TDataCompletionBlock)(NSData *data, NSError *error);
@interface TMap : NSObject
- (NSString *)identifier;
- (NSString *)name;
+ (TMap *)mapForID:(NSString *)identifier;
- (void)dataWithCompletionBlock:(TDataCompletionBlock)block;
@end

typedef enum : NSUInteger {
    TOK                = 100,
    TError = 125,
} TResponseCode;

@interface TConnection : NSObject
- (void)sendString:(NSString *)string;
- (void)sendFormat:(NSString *)format, ...;
- (void)sendResponseCode:(TResponseCode)responseCode dataFollows:(BOOL)flag
                         format:(NSString *)format, ...;
@end

@interface TServer : NSObject
@end

@implementation TServer
- (void)serverConnection:(TConnection *)connection getCommand:(NSString *)str
{
    NSString    *mapID = nil;
    TMap       *map = [TMap mapForID:mapID];
// Make sure we do not map code generated for the block to the above line.
// CHECK: define internal void @"__39-[TServer serverConnection:getCommand:]_block_invoke"
// CHECK: call void @llvm.objc.storeStrong(i8** [[ZERO:%.*]], i8* [[ONE:%.*]]) [[NUW:#[0-9]+]]
// CHECK: call void @llvm.objc.storeStrong(i8** [[TWO:%.*]], i8* [[THREE:%.*]]) [[NUW]]
// CHECK: call {{.*}}@objc_msgSend{{.*}}, !dbg ![[LINE_ABOVE:[0-9]+]]
// CHECK: getelementptr
// CHECK-NOT: !dbg, ![[LINE_ABOVE]]
// CHECK: bitcast %5** [[TMP:%.*]] to i8**
// CHECK-NOT: !dbg, ![[LINE_ABOVE]]
// CHECK: call void @llvm.objc.storeStrong(i8** [[VAL1:%.*]], i8* null) [[NUW]]
// CHECK-NEXT: bitcast %4** [[TMP:%.*]] to i8**
// CHECK-NEXT: call void @llvm.objc.storeStrong(i8** [[VAL2:%.*]], i8* null) [[NUW]]
// CHECK-NEXT: ret
// CHECK: attributes [[NUW]] = { nounwind }
    [map dataWithCompletionBlock:^(NSData *data, NSError *error) {
        if (data) {
            NSString    *encoded = [[data compressedData] encodedString:18];
            [connection sendResponseCode:TOK dataFollows:YES
                format:@"Sending \"%@\" (%@)", [map name], [map identifier]];
            [encoded enumerateLinesUsingBlock:^(NSString *line, BOOL *stop) {
                [connection sendFormat:@"%@\r\n", line];
            }];
            [connection sendString:@".\r\n"];
        } else {
            [connection sendResponseCode:TError dataFollows:NO
                format:@"Failed \"%@\" (%@)", [map name], [map identifier]];
        }
    }];
}
@end
