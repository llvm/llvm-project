// RUN: %clang_cc1 %s -emit-llvm -triple x86_64-apple-darwin -o - | FileCheck %s

@interface NSDictionary @end
@interface NSMutableDictionary : NSDictionary@end@interface CalDAVAddManagedAttachmentsTaskGroup {
    NSMutableDictionary *_filenamesToServerLocation; 
}
- (NSDictionary *)filenamesToServerLocation;
@property (readwrite, retain) NSMutableDictionary *filenamesToServerLocation;
@end 

@implementation CalDAVAddManagedAttachmentsTaskGroup
@synthesize filenamesToServerLocation=_filenamesToServerLocation;
@end

// CHECK:  [[CALL:%.*]] = tail call ptr @objc_getProperty
// CHECK:  ret ptr [[CALL:%.*]]
