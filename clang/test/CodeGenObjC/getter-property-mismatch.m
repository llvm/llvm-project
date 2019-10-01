// RUN: %clang_cc1 %s -emit-llvm -triple x86_64-apple-darwin -o - | FileCheck %s
// rdar://11323676

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

// CHECK:  [[CALL:%.*]] = tail call i8* @objc_getProperty
// CHECK:  [[ONE:%.*]] = bitcast i8* [[CALL:%.*]] to [[T1:%.*]]*
// CHECK:  ret [[T1]]* [[ONE]]
