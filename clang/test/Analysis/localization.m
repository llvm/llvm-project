// RUN: %clang_cc1 -analyze -fblocks -analyzer-store=region -analyzer-checker=optin.osx.cocoa.localizability.NonLocalizedStringChecker -analyzer-checker=alpha.osx.cocoa.localizability.PluralMisuseChecker -verify  %s

// The larger set of tests in located in localization.m. These are tests
// specific for non-aggressive reporting.

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
@interface NSObject
+ (id)alloc;
- (id)init;
@end
@interface NSString : NSObject
- (NSString *)stringByAppendingFormat:(NSString *)format, ...;
+ (instancetype)stringWithFormat:(NSString *)format, ...;
@end
@interface NSBundle : NSObject
+ (NSBundle *)mainBundle;
- (NSString *)localizedStringForKey:(NSString *)key
                              value:(NSString *)value
                              table:(NSString *)tableName;
@end
@interface UILabel : NSObject
@property(nullable, nonatomic, copy) NSString *text;
@end
@interface TestObject : NSObject
@property(strong) NSString *text;
@end

@interface LocalizationTestSuite : NSObject
int random();
@property (assign) int unreadArticlesCount;
@end
#define MCLocalizedString(s) NSLocalizedString(s,nil);
// Test cases begin here
@implementation LocalizationTestSuite

NSString *KHLocalizedString(NSString* key, NSString* comment) {
    return NSLocalizedString(key, comment);
}

// An object passed in as an parameter's string member
// should not be considered unlocalized
- (void)testObjectAsArgument:(TestObject *)argumentObject {
  UILabel *testLabel = [[UILabel alloc] init];

  [testLabel setText:[argumentObject text]]; // no-warning
  [testLabel setText:argumentObject.text];   // no-warning
}

- (void)testLocalizationErrorDetectedOnPathway {
  UILabel *testLabel = [[UILabel alloc] init];
  NSString *bar = NSLocalizedString(@"Hello", @"Comment");

  if (random()) {
    bar = @"Unlocalized string";
  }

  [testLabel setText:bar]; // expected-warning {{User-facing text should use localized string macro}}
}

- (void)testOneCharacterStringsDoNotGiveAWarning {
  UILabel *testLabel = [[UILabel alloc] init];
  NSString *bar = NSLocalizedString(@"Hello", @"Comment");

  if (random()) {
    bar = @"-";
  }

  [testLabel setText:bar]; // no-warning
}

- (void)testOneCharacterUTFStringsDoNotGiveAWarning {
  UILabel *testLabel = [[UILabel alloc] init];
  NSString *bar = NSLocalizedString(@"Hello", @"Comment");

  if (random()) {
    bar = @"\u2014";
  }

  [testLabel setText:bar]; // no-warning
}


// Suppress diagnostic about user-facing string constants when the method name
// contains the term "Debug".
- (void)debugScreen:(UILabel *)label {
  label.text = @"Unlocalized";
}

// Plural Misuse Checker Tests
// These tests are modeled off incorrect uses of the many-one pattern
// from real projects. 

- (NSString *)test1:(int)plural {
    if (plural) {
        return MCLocalizedString(@"TYPE_PLURAL"); // expected-warning {{Plural cases are not supported accross all languages. Use a .stringsdict file}}
    }
    return MCLocalizedString(@"TYPE");
}

- (NSString *)test2:(int)numOfReminders {
    if (numOfReminders > 0) {
        return [NSString stringWithFormat:@"%@, %@", @"Test", (numOfReminders != 1) ? [NSString stringWithFormat:NSLocalizedString(@"%@ Reminders", @"Plural count of reminders"), numOfReminders] : [NSString stringWithFormat:NSLocalizedString(@"1 reminder", @"One reminder")]]; // expected-warning {{Plural cases are not supported accross all languages. Use a .stringsdict file}} expected-warning {{Plural cases are not supported accross all languages. Use a .stringsdict file}}
    } 
    return nil;
}

- (void)test3 {
    NSString *count;
    if (self.unreadArticlesCount > 1)
    {
        count = [count stringByAppendingFormat:@"%@", KHLocalizedString(@"New Stories", @"Plural count for new stories")]; // expected-warning {{Plural cases are not supported accross all languages. Use a .stringsdict file}}
    } else {
        count = [count stringByAppendingFormat:@"%@",  KHLocalizedString(@"New Story", @"One new story")]; // expected-warning {{Plural cases are not supported accross all languages. Use a .stringsdict file}}
    }
}

- (NSString *)test4:(int)count {
    if ( count == 1 )
    {
        return [NSString stringWithFormat:KHLocalizedString(@"value.singular",nil), count]; // expected-warning {{Plural cases are not supported accross all languages. Use a .stringsdict file}}
    } else {
        return [NSString stringWithFormat:KHLocalizedString(@"value.plural",nil), count]; // expected-warning {{Plural cases are not supported accross all languages. Use a .stringsdict file}}
    }
}

- (NSString *)test5:(int)count {
	int test = count == 1;
    if (test)
    {
        return [NSString stringWithFormat:KHLocalizedString(@"value.singular",nil), count]; // expected-warning {{Plural cases are not supported accross all languages. Use a .stringsdict file}}
    } else {
        return [NSString stringWithFormat:KHLocalizedString(@"value.plural",nil), count]; // expected-warning {{Plural cases are not supported accross all languages. Use a .stringsdict file}}
    }
}

// This tests the heuristic that the direct parent IfStmt must match the isCheckingPlurality confition to avoid false positives generated from complex code (generally the pattern we're looking for is simple If-Else)

- (NSString *)test6:(int)sectionIndex {
	int someOtherVariable = 0;
    if (sectionIndex == 1)
    {
		// Do some other crazy stuff
		if (someOtherVariable)
        	return KHLocalizedString(@"OK",nil); // no-warning
    } else {
        return KHLocalizedString(@"value.plural",nil); // expected-warning {{Plural cases are not supported accross all languages. Use a .stringsdict file}}
    }
	return nil;
}

// False positives that we are not accounting for involve matching the heuristic
// of having 1 or 2 in the RHS of a BinaryOperator and having a localized string 
// in the body of the IfStmt. This is seen a lot when checking for the section
// indexpath of something like a UITableView

// - (NSString *)testNotAccountedFor:(int)sectionIndex {
//     if (sectionIndex == 1)
//     {
//         return KHLocalizedString(@"1",nil); // false-positive
//     } else if (sectionIndex == 2) {
//     	return KHLocalizedString(@"2",nil); // false-positive
//     } else if (sectionIndex == 3) {
// 		return KHLocalizedString(@"3",nil); // no-false-positive
// 	}
// }

// Potential test-cases to support in the future

// - (NSString *)test7:(int)count {
//     BOOL plural = count != 1;
//     return KHLocalizedString(plural ? @"PluralString" : @"SingularString", @"");
// }
//
// - (NSString *)test8:(BOOL)plural {
//     return KHLocalizedString(([NSString stringWithFormat:@"RELATIVE_DATE_%@_%@", ((1 == 1) ? @"FUTURE" : @"PAST"), plural ? @"PLURAL" : @"SINGULAR"]));
// }
//
//
//
// - (void)test9:(int)numberOfTimesEarned {
//     NSString* localizedDescriptionKey;
//     if (numberOfTimesEarned == 1) {
//         localizedDescriptionKey = @"SINGULAR_%@";
//     } else {
//         localizedDescriptionKey = @"PLURAL_%@_%@";
//     }
//     NSLocalizedString(localizedDescriptionKey, nil);
// }
//
// - (NSString *)test10 {
//     NSInteger count = self.problems.count;
//     NSString *title = [NSString stringWithFormat:@"%ld Problems", (long) count];
//     if (count < 2) {
//         if (count == 0) {
//             title = [NSString stringWithFormat:@"No Problems Found"];
//         } else {
//             title = [NSString stringWithFormat:@"%ld Problem", (long) count];
//         }
//     }
//     return title;
// }

@end


// Suppress diagnostic about user-facing string constants when the class name
// contains "Debug"
@interface MyDebugView : NSObject
@end

@implementation MyDebugView
- (void)setupScreen:(UILabel *)label {
  label.text = @"Unlocalized";
}
@end
