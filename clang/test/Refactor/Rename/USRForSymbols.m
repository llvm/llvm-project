@interface I1

@property int p1;

@end

@implementation I1

@end

// CHECK: 3 symbols
// CHECK: 'c:objc(cs)I1(py)p1'
// CHECK: 'c:objc(cs)I1(im)p1'
// CHECK: 'c:objc(cs)I1(im)setP1:'

// RUN: clang-refactor-test rename-initiate -at=%s:3:15 -new-name=foo -dump-symbols %s | FileCheck %s
