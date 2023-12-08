// RUN: %check_clang_tidy %s objc-avoid-nserror-init %t
@interface NSError
+ (instancetype)alloc;
- (instancetype)init;
@end

@implementation foo
- (void)bar {
    NSError *error = [[NSError alloc] init];
    // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: use errorWithDomain:code:userInfo: or initWithDomain:code:userInfo: to create a new NSError [objc-avoid-nserror-init]
}
@end
