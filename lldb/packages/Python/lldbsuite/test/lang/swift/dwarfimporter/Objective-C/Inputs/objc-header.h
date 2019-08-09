/* -*- ObjC -*- */
@import Foundation;

@protocol WithName
- (NSString *)name;
@end

@protocol ObjCProtocol <WithName>
@end

@interface ObjCClass : NSObject
- (instancetype)init;
@end

// FIXME: id<ObjCProtocol> doesn't anchor the type since Clang doesn't
// yet describe protocol conformances in DWARF.
id<ObjCProtocol> getProto();
