// RUN: %clang_cc1 -fblocks -x objective-c-header -emit-pch -o %t.pch %S/Inputs/localization-pch.h

// RUN: %clang_cc1 -analyze -fblocks -analyzer-store=region  -analyzer-checker=optin.osx.cocoa.localizability.NonLocalizedStringChecker -analyzer-checker=optin.osx.cocoa.localizability.EmptyLocalizationContextChecker -include-pch %t.pch -verify  -analyzer-config AggressiveReport=true %s

// These declarations were reduced using Delta-Debugging from Foundation.h
// on Mac OS X.

#define nil ((id)0)
#define NSLocalizedString(key, comment)                                        \
  [[NSBundle mainBundle] localizedStringForKey:(key) value:@"" table:nil]
#define NSLocalizedStringFromTable(key, tbl, comment)                          \
  [[NSBundle mainBundle] localizedStringForKey:(key) value:@"" table:(tbl)]
#define NSLocalizedStringFromTableInBundle(key, tbl, bundle, comment)          \
  [bundle localizedStringForKey:(key) value:@"" table:(tbl)]
#define NSLocalizedStringWithDefaultValue(key, tbl, bundle, val, comment)      \
  [bundle localizedStringForKey:(key) value:(val) table:(tbl)]
#define CGFLOAT_TYPE double
typedef CGFLOAT_TYPE CGFloat;
struct CGPoint {
  CGFloat x;
  CGFloat y;
};
typedef struct CGPoint CGPoint;
@interface NSObject
+ (id)alloc;
- (id)init;
@end
@class NSDictionary;
@interface NSString : NSObject
- (void)drawAtPoint:(CGPoint)point withAttributes:(NSDictionary *)attrs;
+ (instancetype)localizedStringWithFormat:(NSString *)format, ...;
@end
@interface NSBundle : NSObject
+ (NSBundle *)mainBundle;
- (NSString *)localizedStringForKey:(NSString *)key
                              value:(NSString *)value
                              table:(NSString *)tableName;
@end
@protocol UIAccessibility 
- (void)accessibilitySetIdentification:(NSString *)ident;
- (void)setAccessibilityLabel:(NSString *)label;
@end
@interface UILabel : NSObject <UIAccessibility>
@property(nullable, nonatomic, copy) NSString *text;
@end
@interface TestObject : NSObject
@property(strong) NSString *text;
@end
@interface NSView : NSObject
@property (strong) NSString *toolTip;
@end
@interface NSViewSubclass : NSView
@end

@interface LocalizationTestSuite : NSObject
NSString *ForceLocalized(NSString *str)
    __attribute__((annotate("returns_localized_nsstring")));
CGPoint CGPointMake(CGFloat x, CGFloat y);
int random();
// This next one is a made up API
NSString *CFNumberFormatterCreateStringWithNumber(float x);
+ (NSString *)forceLocalized:(NSString *)str
    __attribute__((annotate("returns_localized_nsstring")));
@end

// Test cases begin here
@implementation LocalizationTestSuite

// A C-Funtion that returns a localized string because it has the
// "returns_localized_nsstring" annotation
NSString *ForceLocalized(NSString *str) { return str; }
// An ObjC method that returns a localized string because it has the
// "returns_localized_nsstring" annotation
+ (NSString *)forceLocalized:(NSString *)str {
  return str;
}

// An ObjC method that returns a localized string
+ (NSString *)unLocalizedStringMethod {
  return @"UnlocalizedString";
}

- (void)testLocalizationErrorDetectedOnPathway {
  UILabel *testLabel = [[UILabel alloc] init];
  NSString *bar = NSLocalizedString(@"Hello", @"Comment");

  if (random()) {
    bar = @"Unlocalized string";
  }

  [testLabel setText:bar]; // expected-warning {{User-facing text should use localized string macro}}
}

- (void)testLocalizationErrorDetectedOnNSString {
  NSString *bar = NSLocalizedString(@"Hello", @"Comment");

  if (random()) {
    bar = @"Unlocalized string";
  }

  [bar drawAtPoint:CGPointMake(0, 0) withAttributes:nil]; // expected-warning {{User-facing text should use localized string macro}}
}

- (void)testNoLocalizationErrorDetectedFromCFunction {
  UILabel *testLabel = [[UILabel alloc] init];
  NSString *bar = CFNumberFormatterCreateStringWithNumber(1);

  [testLabel setText:bar]; // no-warning
}

- (void)testAnnotationAddsLocalizedStateForCFunction {
  UILabel *testLabel = [[UILabel alloc] init];
  NSString *bar = NSLocalizedString(@"Hello", @"Comment");

  if (random()) {
    bar = @"Unlocalized string";
  }

  [testLabel setText:ForceLocalized(bar)]; // no-warning
}

- (void)testAnnotationAddsLocalizedStateForObjCMethod {
  UILabel *testLabel = [[UILabel alloc] init];
  NSString *bar = NSLocalizedString(@"Hello", @"Comment");

  if (random()) {
    bar = @"Unlocalized string";
  }

  [testLabel setText:[LocalizationTestSuite forceLocalized:bar]]; // no-warning
}

// An empty string literal @"" should not raise an error
- (void)testEmptyStringLiteralHasLocalizedState {
  UILabel *testLabel = [[UILabel alloc] init];
  NSString *bar = @"";

  [testLabel setText:bar]; // no-warning
}

// An empty string literal @"" inline should not raise an error
- (void)testInlineEmptyStringLiteralHasLocalizedState {
  UILabel *testLabel = [[UILabel alloc] init];
  [testLabel setText:@""]; // no-warning
}

// An string literal @"Hello" inline should raise an error
- (void)testInlineStringLiteralHasLocalizedState {
  UILabel *testLabel = [[UILabel alloc] init];
  [testLabel setText:@"Hello"]; // expected-warning {{User-facing text should use localized string macro}}
}

// A nil string should not raise an error
- (void)testNilStringIsNotMarkedAsUnlocalized {
  UILabel *testLabel = [[UILabel alloc] init];
  [testLabel setText:nil]; // no-warning
}

// A method that takes in a localized string and returns a string
// most likely that string is localized.
- (void)testLocalizedStringArgument {
  UILabel *testLabel = [[UILabel alloc] init];
  NSString *localizedString = NSLocalizedString(@"Hello", @"Comment");

  NSString *combinedString =
      [NSString localizedStringWithFormat:@"%@", localizedString];

  [testLabel setText:combinedString]; // no-warning
}

// A String passed in as a an parameter should not be considered
// unlocalized
- (void)testLocalizedStringAsArgument:(NSString *)argumentString {
  UILabel *testLabel = [[UILabel alloc] init];

  [testLabel setText:argumentString]; // no-warning
}

// The warning is expected to be seen in localizedStringAsArgument: body
- (void)testLocalizedStringAsArgumentOtherMethod:(NSString *)argumentString {
  [self localizedStringAsArgument:@"UnlocalizedString"];
}

// A String passed into another method that calls a method that
// requires a localized string should give an error
- (void)localizedStringAsArgument:(NSString *)argumentString {
  UILabel *testLabel = [[UILabel alloc] init];

  [testLabel setText:argumentString]; // expected-warning {{User-facing text should use localized string macro}}
}

// [LocalizationTestSuite unLocalizedStringMethod] returns an unlocalized string
// so we expect an error. Unfrtunately, it probably doesn't make a difference
// what [LocalizationTestSuite unLocalizedStringMethod] returns since all
// string values returned are marked as Unlocalized in aggressive reporting.
- (void)testUnLocalizedStringMethod {
  UILabel *testLabel = [[UILabel alloc] init];
  NSString *bar = NSLocalizedString(@"Hello", @"Comment");

  [testLabel setText:[LocalizationTestSuite unLocalizedStringMethod]]; // expected-warning {{User-facing text should use localized string macro}}
}

// This is the reverse situation: accessibilitySetIdentification: doesn't care
// about localization so we don't expect a warning
- (void)testMethodNotInRequiresLocalizedStringMethods {
  UILabel *testLabel = [[UILabel alloc] init];

  [testLabel accessibilitySetIdentification:@"UnlocalizedString"]; // no-warning
}

// An NSView subclass should raise a warning for methods in NSView that 
// require localized strings 
- (void)testRequiresLocalizationMethodFromSuperclass {
  NSViewSubclass *s = [[NSViewSubclass alloc] init];
  NSString *bar = @"UnlocalizedString";

  [s setToolTip:bar]; // expected-warning {{User-facing text should use localized string macro}}
}

- (void)testRequiresLocalizationMethodFromProtocol {
  UILabel *testLabel = [[UILabel alloc] init];

  [testLabel setAccessibilityLabel:@"UnlocalizedString"]; // expected-warning {{User-facing text should use localized string macro}}
}

// EmptyLocalizationContextChecker tests
#define HOM(s) YOLOC(s)
#define YOLOC(x) NSLocalizedString(x, nil)

- (void)testNilLocalizationContext {
  NSString *string = NSLocalizedString(@"LocalizedString", nil); // expected-warning {{Localized string macro should include a non-empty comment for translators}}
  NSString *string2 = NSLocalizedString(@"LocalizedString", nil); // expected-warning {{Localized string macro should include a non-empty comment for translators}}
  NSString *string3 = NSLocalizedString(@"LocalizedString", nil); // expected-warning {{Localized string macro should include a non-empty comment for translators}}
}

- (void)testEmptyLocalizationContext {
  NSString *string = NSLocalizedString(@"LocalizedString", @""); // expected-warning {{Localized string macro should include a non-empty comment for translators}}
  NSString *string2 = NSLocalizedString(@"LocalizedString", @" "); // expected-warning {{Localized string macro should include a non-empty comment for translators}}
  NSString *string3 = NSLocalizedString(@"LocalizedString", @"	 "); // expected-warning {{Localized string macro should include a non-empty comment for translators}}
}

- (void)testNSLocalizedStringVariants {
  NSString *string = NSLocalizedStringFromTable(@"LocalizedString", nil, @""); // expected-warning {{Localized string macro should include a non-empty comment for translators}}
  NSString *string2 = NSLocalizedStringFromTableInBundle(@"LocalizedString", nil, [[NSBundle alloc] init],@""); // expected-warning {{Localized string macro should include a non-empty comment for translators}}
  NSString *string3 = NSLocalizedStringWithDefaultValue(@"LocalizedString", nil, [[NSBundle alloc] init], nil,@""); // expected-warning {{Localized string macro should include a non-empty comment for translators}}
}

- (void)testMacroExpansionNilString {
  NSString *string = YOLOC(@"Hello"); // expected-warning {{Localized string macro should include a non-empty comment for translators}}
  NSString *string2 = HOM(@"Hello");  // expected-warning {{Localized string macro should include a non-empty comment for translators}}
  NSString *string3 = NSLocalizedString((0 ? @"Critical" : @"Current"),nil); // expected-warning {{Localized string macro should include a non-empty comment for translators}}
}

- (void)testMacroExpansionDefinedInPCH {
  NSString *string = MyLocalizedStringInPCH(@"Hello"); // expected-warning {{Localized string macro should include a non-empty comment for translators}}
}

#define KCLocalizedString(x,comment) NSLocalizedString(x, comment)
#define POSSIBLE_FALSE_POSITIVE(s,other) KCLocalizedString(s,@"Comment")

- (void)testNoWarningForNilCommentPassedIntoOtherMacro {
  NSString *string = KCLocalizedString(@"Hello",@""); // no-warning
  NSString *string2 = KCLocalizedString(@"Hello",nil); // no-warning
  NSString *string3 = KCLocalizedString(@"Hello",@"Comment"); // no-warning
}

- (void)testPossibleFalsePositiveSituationAbove {
  NSString *string = POSSIBLE_FALSE_POSITIVE(@"Hello", nil); // no-warning
  NSString *string2 = POSSIBLE_FALSE_POSITIVE(@"Hello", @"Hello"); // no-warning
}

@end
