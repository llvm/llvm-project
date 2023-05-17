// RUN: %check_clang_tidy %s objc-nsdate-formatter %t
@interface NSObject
+ (instancetype)alloc;
- (instancetype)init;
@end

@interface TestClass : NSObject
+ (void)testMethod1;
+ (void)testMethod2;
+ (void)testMethod3;
+ (void)testAnotherClass;
@end

@interface NSString : NSObject
@end

void NSLog(NSString *format, ...);

@interface NSDate : NSObject
@end

@interface NSDateFormatter : NSObject
@property(copy) NSString *dateFormat;
- (NSString *)stringFromDate:(NSDate *)date;
@end

@interface AnotherClass : NSObject
@property(copy) NSString *dateFormat;
@end

@interface NSDateComponents : NSObject
@property long year;
@property long month;
@property long day;
@end

@interface NSCalendar : NSObject
@property(class, readonly, copy) NSCalendar *currentCalendar;
- (nullable NSDate *)dateFromComponents:(NSDateComponents *)Comps;
@end

@implementation TestClass
+ (void)testMethod1 {
  // Reproducing incorrect behavior from Radar
  NSDateFormatter *formatter = [[NSDateFormatter alloc] init];
  [formatter setDateFormat:@"YYYY_MM_dd_HH_mm_ss_SSS_ZZZ"];
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use of week year (Y) with month (M); did you mean to use calendar year (y) instead? [objc-nsdate-formatter]
  NSDateComponents *comps = [[NSDateComponents alloc] init];
  [comps setDay:29];
  [comps setMonth:12];
  [comps setYear:2014];
  NSDate *date_radar = [[NSCalendar currentCalendar] dateFromComponents:comps];
  NSLog(@"YYYY_MM_dd_HH_mm_ss_SSS_ZZZ %@", [formatter stringFromDate:date_radar]);

  // Radar correct behavior
  [formatter setDateFormat:@"yyyy_MM_dd_HH_mm_ss_SSS_ZZZ"];
  NSLog(@"yyyy_MM_dd_HH_mm_ss_SSS_ZZZ %@", [formatter stringFromDate:date_radar]);

  // Radar correct behavior - week year
  [formatter setDateFormat:@"YYYY_ww_dd_HH_mm_ss_SSS_ZZZ"];
  NSLog(@"YYYY_ww_dd_HH_mm_ss_SSS_ZZZ %@", [formatter stringFromDate:date_radar]);

  // Radar incorrect behavior
  [formatter setDateFormat:@"yyyy_ww_dd_HH_mm_ss_SSS_ZZZ"];
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use of calendar year (y) with week of the year (w); did you mean to use week-year (Y) instead? [objc-nsdate-formatter]
  NSLog(@"yyyy_ww_dd_HH_mm_ss_SSS_ZZZ %@", [formatter stringFromDate:date_radar]);

  NSLog(@"==========================================");
  // Correct
  [formatter setDateFormat:@"yyyy_MM"];
  NSLog(@"yyyy_MM %@", [formatter stringFromDate:date_radar]);

  // Correct
  [formatter setDateFormat:@"yyyy_dd"];
  NSLog(@"yyyy_dd %@", [formatter stringFromDate:date_radar]);

  // Correct
  [formatter setDateFormat:@"yyyy_DD"];
  NSLog(@"yyyy_DD %@", [formatter stringFromDate:date_radar]);

  // Incorrect
  [formatter setDateFormat:@"yyyy_ww"];
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use of calendar year (y) with week of the year (w); did you mean to use week-year (Y) instead? [objc-nsdate-formatter]
  NSLog(@"yyyy_ww %@", [formatter stringFromDate:date_radar]);

  // Incorrect
  [formatter setDateFormat:@"yyyy_WW"];
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: Week of Month (W) used without the month (M); did you forget M in the format string? [objc-nsdate-formatter]
  NSLog(@"yyyy_WW %@", [formatter stringFromDate:date_radar]);

  NSLog(@"==========================================");
  // Correct
  [formatter setDateFormat:@"yyyy_MM_dd"];
  NSLog(@"yyyy_MM_dd %@", [formatter stringFromDate:date_radar]);

  // Potentially Incorrect
  [formatter setDateFormat:@"yyyy_MM_DD"];
  NSLog(@"yyyy_MM_DD %@", [formatter stringFromDate:date_radar]);

  // Incorrect
  [formatter setDateFormat:@"yyyy_MM_ww"];
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use of calendar year (y) with week of the year (w); did you mean to use week-year (Y) instead? [objc-nsdate-formatter]
  NSLog(@"yyyy_MM_ww %@", [formatter stringFromDate:date_radar]);

  NSLog(@"=======WEEK YEAR==========");
  // Incorrect
  [formatter setDateFormat:@"YYYY_MM"];
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use of week year (Y) with month (M); did you mean to use calendar year (y) instead? [objc-nsdate-formatter]
  NSLog(@"YYYY_MM %@", [formatter stringFromDate:date_radar]);

  // Correct
  [formatter setDateFormat:@"YYYY_ww"];
  NSLog(@"YYYY_ww %@", [formatter stringFromDate:date_radar]);

  // Incorrect
  [formatter setDateFormat:@"YYYY_WW"];
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: Week of Month (W) used without the month (M); did you forget M in the format string? [objc-nsdate-formatter]
  // CHECK-MESSAGES: :[[@LINE-2]]:28: warning: use of week year (Y) with week of the month (W); did you mean to use calendar year (y) instead? [objc-nsdate-formatter]
  NSLog(@"YYYY_WW %@", [formatter stringFromDate:date_radar]);

  // Correct
  [formatter setDateFormat:@"YYYY_dd"];
  NSLog(@"YYYY_dd %@", [formatter stringFromDate:date_radar]);

  // Incorrect
  [formatter setDateFormat:@"YYYY_DD"];
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use of week year (Y) with day of the year (D); did you mean to use calendar year (y) instead? [objc-nsdate-formatter]
  NSLog(@"YYYY_DD %@", [formatter stringFromDate:date_radar]);

  NSLog(@"==========================================");
  // Potentially Incorrect
  [formatter setDateFormat:@"YYYY_ww_dd"];
  NSLog(@"YYYY ww dd %@", [formatter stringFromDate:date_radar]);

  // Incorrect
  [formatter setDateFormat:@"YYYY_ww_DD"];
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use of week year (Y) with day of the year (D); did you mean to use calendar year (y) instead? [objc-nsdate-formatter]
  NSLog(@"YYYY_ww_DD %@", [formatter stringFromDate:date_radar]);
}

+ (void)testMethod2 {
  NSDateFormatter *formatter = [[NSDateFormatter alloc] init];
  NSDateComponents *comps = [[NSDateComponents alloc] init];
  [comps setDay:29];
  [comps setMonth:12];
  [comps setYear:2014];
  NSDate *date_radar = [[NSCalendar currentCalendar] dateFromComponents:comps];

  // Test 1 : incorrect
  [formatter setDateFormat:@"yyyy_QQ_MM_ww_dd_EE"];
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use of calendar year (y) with week of the year (w); did you mean to use week-year (Y) instead? [objc-nsdate-formatter]
  NSLog(@"yyyy_QQ_MM_ww_dd_EE %@", [formatter stringFromDate:date_radar]);

  // Test 2 : incorrect
  [formatter setDateFormat:@"yyyy_QQ_MM_ww_dd_ee"];
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use of calendar year (y) with week of the year (w); did you mean to use week-year (Y) instead? [objc-nsdate-formatter]
  NSLog(@"yyyy_QQ_MM_ww_dd_ee %@", [formatter stringFromDate:date_radar]);

  // Test 3 : incorrect
  [formatter setDateFormat:@"yyyy_QQ_MM_ww_DD_EE"];
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use of calendar year (y) with week of the year (w); did you mean to use week-year (Y) instead? [objc-nsdate-formatter]
  NSLog(@"yyyy_QQ_MM_ww_DD_EE %@", [formatter stringFromDate:date_radar]);

  // Test 4 : incorrect
  [formatter setDateFormat:@"yyyy_QQ_MM_ww_DD_ee"];
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use of calendar year (y) with week of the year (w); did you mean to use week-year (Y) instead? [objc-nsdate-formatter]
  NSLog(@"yyyy_QQ_MM_ww_DD_ee %@", [formatter stringFromDate:date_radar]);

  // Test 5 : incorrect
  [formatter setDateFormat:@"yyyy_QQ_MM_ww_F_EE"];
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use of calendar year (y) with week of the year (w); did you mean to use week-year (Y) instead? [objc-nsdate-formatter]
  NSLog(@"yyyy_QQ_MM_ww_F_EE %@", [formatter stringFromDate:date_radar]);

  // Test 6 : incorrect
  [formatter setDateFormat:@"yyyy_QQ_MM_ww_F_ee"];
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use of calendar year (y) with week of the year (w); did you mean to use week-year (Y) instead? [objc-nsdate-formatter]
  NSLog(@"yyyy_QQ_MM_ww_F_ee %@", [formatter stringFromDate:date_radar]);

  // Test 7 : correct
  [formatter setDateFormat:@"yyyy_QQ_MM_WW_dd_EE"];
  NSLog(@"yyyy_QQ_MM_WW_dd_EE %@", [formatter stringFromDate:date_radar]);

  // Test 8 : correct
  [formatter setDateFormat:@"yyyy_QQ_MM_WW_dd_ee"];
  NSLog(@"yyyy_QQ_MM_WW_dd_ee %@", [formatter stringFromDate:date_radar]);

  // Test 9 : correct
  [formatter setDateFormat:@"yyyy_QQ_MM_WW_DD_EE"];
  NSLog(@"yyyy_QQ_MM_WW_DD_EE %@", [formatter stringFromDate:date_radar]);

  // Test 10 : correct
  [formatter setDateFormat:@"yyyy_QQ_MM_WW_DD_ee"];
  NSLog(@"yyyy_QQ_MM_WW_DD_ee %@", [formatter stringFromDate:date_radar]);

  // Test 11 : correct
  [formatter setDateFormat:@"yyyy_QQ_MM_WW_F_EE"];
  NSLog(@"yyyy_QQ_MM_WW_F_EE %@", [formatter stringFromDate:date_radar]);

  // Test 12 : correct
  [formatter setDateFormat:@"yyyy_QQ_MM_WW_F_ee"];
  NSLog(@"yyyy_QQ_MM_WW_F_ee %@", [formatter stringFromDate:date_radar]);

  // Test 13 : incorrect
  [formatter setDateFormat:@"YYYY_QQ_MM_ww_dd_EE"];
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use of week year (Y) with month (M); did you mean to use calendar year (y) instead? [objc-nsdate-formatter]
  // CHECK-MESSAGES: :[[@LINE-2]]:28: warning: use of week year (Y) with quarter number (Q); did you mean to use calendar year (y) instead? [objc-nsdate-formatter]
  NSLog(@"YYYY_QQ_MM_ww_dd_EE %@", [formatter stringFromDate:date_radar]);

  // Test 14 : incorrect
  [formatter setDateFormat:@"YYYY_QQ_MM_ww_dd_ee"];
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use of week year (Y) with month (M); did you mean to use calendar year (y) instead? [objc-nsdate-formatter]
  // CHECK-MESSAGES: :[[@LINE-2]]:28: warning: use of week year (Y) with quarter number (Q); did you mean to use calendar year (y) instead? [objc-nsdate-formatter]
  NSLog(@"YYYY_QQ_MM_ww_dd_ee %@", [formatter stringFromDate:date_radar]);

  // Test 15 : incorrect
  [formatter setDateFormat:@"YYYY_QQ_MM_ww_DD_EE"];
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use of week year (Y) with day of the year (D); did you mean to use calendar year (y) instead? [objc-nsdate-formatter]
  // CHECK-MESSAGES: :[[@LINE-2]]:28: warning: use of week year (Y) with month (M); did you mean to use calendar year (y) instead? [objc-nsdate-formatter]
  // CHECK-MESSAGES: :[[@LINE-3]]:28: warning: use of week year (Y) with quarter number (Q); did you mean to use calendar year (y) instead? [objc-nsdate-formatter]
  NSLog(@"YYYY_QQ_MM_ww_DD_EE %@", [formatter stringFromDate:date_radar]);

  // Test 16 : incorrect
  [formatter setDateFormat:@"YYYY_QQ_MM_ww_DD_ee"];
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use of week year (Y) with day of the year (D); did you mean to use calendar year (y) instead? [objc-nsdate-formatter]
  // CHECK-MESSAGES: :[[@LINE-2]]:28: warning: use of week year (Y) with month (M); did you mean to use calendar year (y) instead? [objc-nsdate-formatter]
  // CHECK-MESSAGES: :[[@LINE-3]]:28: warning: use of week year (Y) with quarter number (Q); did you mean to use calendar year (y) instead? [objc-nsdate-formatter]
  NSLog(@"YYYY_QQ_MM_ww_DD_ee %@", [formatter stringFromDate:date_radar]);

  // Test 17 : incorrect
  [formatter setDateFormat:@"YYYY_QQ_MM_ww_F_EE"];
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use of week year (Y) with day of the week in month (F); did you mean to use calendar year (y) instead? [objc-nsdate-formatter]
  // CHECK-MESSAGES: :[[@LINE-2]]:28: warning: use of week year (Y) with month (M); did you mean to use calendar year (y) instead? [objc-nsdate-formatter]
  // CHECK-MESSAGES: :[[@LINE-3]]:28: warning: use of week year (Y) with quarter number (Q); did you mean to use calendar year (y) instead? [objc-nsdate-formatter]
  NSLog(@"YYYY_QQ_MM_ww_F_EE %@", [formatter stringFromDate:date_radar]);

  // Test 18 : incorrect
  [formatter setDateFormat:@"YYYY_QQ_MM_ww_F_ee"];
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use of week year (Y) with day of the week in month (F); did you mean to use calendar year (y) instead? [objc-nsdate-formatter]
  // CHECK-MESSAGES: :[[@LINE-2]]:28: warning: use of week year (Y) with month (M); did you mean to use calendar year (y) instead? [objc-nsdate-formatter]
  // CHECK-MESSAGES: :[[@LINE-3]]:28: warning: use of week year (Y) with quarter number (Q); did you mean to use calendar year (y) instead? [objc-nsdate-formatter]
  NSLog(@"YYYY_QQ_MM_ww_F_ee %@", [formatter stringFromDate:date_radar]);

  // Test 19 : incorrect
  [formatter setDateFormat:@"YYYY_QQ_MM_WW_dd_EE"];
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use of week year (Y) with month (M); did you mean to use calendar year (y) instead? [objc-nsdate-formatter]
  // CHECK-MESSAGES: :[[@LINE-2]]:28: warning: use of week year (Y) with quarter number (Q); did you mean to use calendar year (y) instead? [objc-nsdate-formatter]
  // CHECK-MESSAGES: :[[@LINE-3]]:28: warning: use of week year (Y) with week of the month (W); did you mean to use calendar year (y) instead? [objc-nsdate-formatter]
  NSLog(@"YYYY_QQ_MM_WW_dd_EE %@", [formatter stringFromDate:date_radar]);

  // Test 20 : incorrect
  [formatter setDateFormat:@"YYYY_QQ_MM_WW_dd_ee"];
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use of week year (Y) with month (M); did you mean to use calendar year (y) instead? [objc-nsdate-formatter]
  // CHECK-MESSAGES: :[[@LINE-2]]:28: warning: use of week year (Y) with quarter number (Q); did you mean to use calendar year (y) instead? [objc-nsdate-formatter]
  // CHECK-MESSAGES: :[[@LINE-3]]:28: warning: use of week year (Y) with week of the month (W); did you mean to use calendar year (y) instead? [objc-nsdate-formatter]
  NSLog(@"YYYY_QQ_MM_WW_dd_ee %@", [formatter stringFromDate:date_radar]);

  // Test 21 : incorrect
  [formatter setDateFormat:@"YYYY_QQ_MM_WW_DD_EE"];
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use of week year (Y) with day of the year (D); did you mean to use calendar year (y) instead? [objc-nsdate-formatter]
  // CHECK-MESSAGES: :[[@LINE-2]]:28: warning: use of week year (Y) with month (M); did you mean to use calendar year (y) instead? [objc-nsdate-formatter]
  // CHECK-MESSAGES: :[[@LINE-3]]:28: warning: use of week year (Y) with quarter number (Q); did you mean to use calendar year (y) instead? [objc-nsdate-formatter]
  // CHECK-MESSAGES: :[[@LINE-4]]:28: warning: use of week year (Y) with week of the month (W); did you mean to use calendar year (y) instead? [objc-nsdate-formatter]
  NSLog(@"YYYY_QQ_MM_WW_DD_EE %@", [formatter stringFromDate:date_radar]);

  // Test 22 : incorrect
  [formatter setDateFormat:@"YYYY_QQ_MM_WW_DD_ee"];
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use of week year (Y) with day of the year (D); did you mean to use calendar year (y) instead? [objc-nsdate-formatter]
  // CHECK-MESSAGES: :[[@LINE-2]]:28: warning: use of week year (Y) with month (M); did you mean to use calendar year (y) instead? [objc-nsdate-formatter]
  // CHECK-MESSAGES: :[[@LINE-3]]:28: warning: use of week year (Y) with quarter number (Q); did you mean to use calendar year (y) instead? [objc-nsdate-formatter]
  // CHECK-MESSAGES: :[[@LINE-4]]:28: warning: use of week year (Y) with week of the month (W); did you mean to use calendar year (y) instead? [objc-nsdate-formatter]
  NSLog(@"YYYY_QQ_MM_WW_DD_ee %@", [formatter stringFromDate:date_radar]);

  // Test 23 : incorrect
  [formatter setDateFormat:@"YYYY_QQ_MM_WW_F_EE"];
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use of week year (Y) with day of the week in month (F); did you mean to use calendar year (y) instead? [objc-nsdate-formatter]
  // CHECK-MESSAGES: :[[@LINE-2]]:28: warning: use of week year (Y) with month (M); did you mean to use calendar year (y) instead? [objc-nsdate-formatter]
  // CHECK-MESSAGES: :[[@LINE-3]]:28: warning: use of week year (Y) with quarter number (Q); did you mean to use calendar year (y) instead? [objc-nsdate-formatter]
  // CHECK-MESSAGES: :[[@LINE-4]]:28: warning: use of week year (Y) with week of the month (W); did you mean to use calendar year (y) instead? [objc-nsdate-formatter]
  NSLog(@"YYYY_QQ_MM_WW_F_EE %@", [formatter stringFromDate:date_radar]);

  // Test 24 : incorrect
  [formatter setDateFormat:@"YYYY_QQ_MM_WW_F_ee"];
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use of week year (Y) with day of the week in month (F); did you mean to use calendar year (y) instead? [objc-nsdate-formatter]
  // CHECK-MESSAGES: :[[@LINE-2]]:28: warning: use of week year (Y) with month (M); did you mean to use calendar year (y) instead? [objc-nsdate-formatter]
  // CHECK-MESSAGES: :[[@LINE-3]]:28: warning: use of week year (Y) with quarter number (Q); did you mean to use calendar year (y) instead? [objc-nsdate-formatter]
  // CHECK-MESSAGES: :[[@LINE-4]]:28: warning: use of week year (Y) with week of the month (W); did you mean to use calendar year (y) instead? [objc-nsdate-formatter]
  NSLog(@"YYYY_QQ_MM_WW_F_ee %@", [formatter stringFromDate:date_radar]);
}

+ (void)testMethod3 {
  NSDateFormatter *Formatter = [[NSDateFormatter alloc] init];
  NSDateComponents *Comps = [[NSDateComponents alloc] init];
  [Comps setDay:29];
  [Comps setMonth:12];
  [Comps setYear:2014];
  NSDate *DateRadar = [[NSCalendar currentCalendar] dateFromComponents:Comps];

  // Incorrect : has reserved and invalid chars
  [Formatter setDateFormat:@"Rashmi"];
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: invalid date format specifier [objc-nsdate-formatter]
  NSLog(@"Rashmi %@", [Formatter stringFromDate:DateRadar]);

  // Correct
  [Formatter setDateFormat:@"AMy"];
  NSLog(@"AMy %@", [Formatter stringFromDate:DateRadar]);
}

+ (void)testAnotherClass {
    AnotherClass *Formatter = [[AnotherClass alloc] init];
    [Formatter setDateFormat:@"RandomString"];
    [Formatter setDateFormat:@"YYYY_QQ_MM_WW_dd_EE"];
}
@end
