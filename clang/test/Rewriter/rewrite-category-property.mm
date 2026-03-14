// RUN: %clang_cc1 -x objective-c++ -fms-extensions -rewrite-objc -fobjc-runtime=macosx-fragile-10.5 %s -o %t-rw.cpp
// RUN: FileCheck -check-prefix CHECK-LP --input-file=%t-rw.cpp %s

@class Y, Z;

@interface A
@property (readonly) Y *y;
@end

@interface A (cat)
@property (readonly) Z *z;
@end

// CHECK-LP: // @property (readonly) Z *z;
