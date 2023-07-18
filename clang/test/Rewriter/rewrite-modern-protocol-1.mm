// RUN: %clang_cc1 -E %s -o %t.mm
// RUN: %clang_cc1 -x objective-c++ -fblocks -fms-extensions -rewrite-objc %t.mm -o - | FileCheck %s
// rdar://11248048

@protocol NSCopying @end

@interface INTF<NSCopying>
@end

@implementation INTF @end

// CHECK: static struct _protocol_t _OBJC_PROTOCOL_NSCopying
// CHECK: static struct _protocol_t *_OBJC_LABEL_PROTOCOL_$_NSCopying = &_OBJC_PROTOCOL_NSCopying;

