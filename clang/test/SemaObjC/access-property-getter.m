// RUN: %clang_cc1 -verify %s

@protocol Protocol
- (oneway void) method;
@end

void accessMethodViaPropertySyntaxAndTriggerWarning(id<Protocol> object) {
    object.method; // expected-warning {{property access result unused - getters should not be used for side effects}}
}

// rdar://19137815
#pragma clang diagnostic ignored "-Wunused-getter-return-value"

void accessMethodViaPropertySyntaxWhenWarningIsIgnoredDoesNotTriggerWarning(id<Protocol> object) {
    object.method;
}

