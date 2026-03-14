// RUN: %clang_cc1 -Wmethod-signatures -fsyntax-only -verify %s

@protocol NSWindowDelegate @end

@protocol IBStringsTableWindowDelegate <NSWindowDelegate>
@end

@interface NSWindow
- (void)setDelegate:(id <NSWindowDelegate>)anObject; // expected-note {{previous definition is here}}
- (id <IBStringsTableWindowDelegate>) delegate; // expected-note {{previous definition is here}}
@end


@interface IBStringsTableWindow : NSWindow {}
@end

@implementation IBStringsTableWindow
- (void)setDelegate:(id <IBStringsTableWindowDelegate>)delegate { // expected-warning {{conflicting parameter types in implementation of 'setDelegate:'}}
}
- (id <NSWindowDelegate>)delegate { // expected-warning {{conflicting return type in implementation of 'delegate':}}
        return 0;
}
@end
