// RUN: %clang_cc1 -triple x86_64-apple-macosx10.11 -Wunguarded-availability -fdiagnostics-parseable-fixits -fsyntax-only -verify %s

// Testing that even for source code using '_' as a delimiter in availability version tuple '.' is actually used in diagnostic output as a delimiter.

@interface foo
- (void) method_bar __attribute__((availability(macosx, introduced = 10_12))); // expected-note {{'method_bar' has been marked as being introduced in macOS 10.12 here, but the deployment target is macOS 10.11}}
@end

int main() {
    [foo method_bar]; // \
    // expected-warning {{'method_bar' is only available on macOS 10.12 or newer}} \
    // expected-note {{enclose 'method_bar' in an @available check to silence this warning}} \
    // CHECK: "fix-it:.*if (@available(macOS 10.12, *))"
    return 0;
}
