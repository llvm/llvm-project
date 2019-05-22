// Test that nullability attributes still get merged even though they are
// wrapped with a MacroQualifiedType. This should just compile with no errors.
// RUN: %clang_cc1 %s -Wno-objc-root-class -fsyntax-only -verify
// expected-no-diagnostics
#define UI_APPEARANCE_SELECTOR __attribute__((annotate("ui_appearance_selector")))

@class UIColor;

@interface Test
@property(null_resettable, nonatomic, strong) UIColor *onTintColor UI_APPEARANCE_SELECTOR;
@end

@implementation Test
- (void)setOnTintColor:(nullable UIColor *)onTintColor {
}

@end
