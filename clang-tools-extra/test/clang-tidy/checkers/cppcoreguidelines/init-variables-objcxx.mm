// RUN: %check_clang_tidy %s cppcoreguidelines-init-variables %t -- -- -fno-objc-arc

@interface NSObject
@end

@interface NSString : NSObject
@end

@interface NSURL : NSObject
- (NSString *)absoluteString;
@end

@interface NSArray : NSObject
@end

void objc_for_in_no_false_positive(NSArray *urls) {
  for (NSURL *url in urls) {
    {
      // 'url' should NOT be flagged - it is a for-in loop variable.
      NSString *urlString = [url absoluteString];
    }
  }
}

void objc_for_in_body_uninit(NSArray *urls) {
  for (NSURL *url in urls) {
    (void)url;

    // CHECK-MESSAGES: :[[@LINE+2]]:15: warning: variable 'str' is not initialized [cppcoreguidelines-init-variables]
    // CHECK-FIXES: NSString *str = nullptr;
    NSString *str;
    (void)str;
  }
}
