// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s
@interface A
@end

@implementation A
- (id):(int)x :(int)y {
    int z;
    // <rdar://problem/8939352>
    if (self = [self :x :y]) {} // expected-warning{{using the result of an assignment as a condition without parentheses}} \
    // expected-note{{use '==' to turn this assignment into an equality comparison}} \
    // expected-note{{place parentheses around the assignment to silence this warning}}
    return self;
}
@end
