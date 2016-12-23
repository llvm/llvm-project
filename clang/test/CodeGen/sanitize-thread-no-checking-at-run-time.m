// RUN: %clang_cc1 -triple x86_64-apple-darwin -x objective-c++ -emit-llvm -o - %s | FileCheck -check-prefix=WITHOUT %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -x objective-c++ -emit-llvm -o - %s -fsanitize=thread | FileCheck -check-prefix=TSAN %s

__attribute__((objc_root_class))
@interface NSObject
- (void)dealloc;
@end

class NeedCleanup {
public:
  ~NeedCleanup() __attribute__((no_sanitize("thread"))) {}
};

@interface MyObject : NSObject {
  NeedCleanup v;
};
+ (void) initialize;
- (void) dealloc;
@end

@implementation MyObject
+ (void)initialize {
}
- (void)dealloc {
  [super dealloc];
}
@end

// WITHOUT-NOT: "sanitize_thread_no_checking_at_run_time"

// TSAN: initialize{{.*}}) [[ATTR:#[0-9]+]]
// TSAN: dealloc{{.*}}) [[ATTR:#[0-9]+]]
// TSAN: cxx_destruct{{.*}}) [[ATTR:#[0-9]+]]
// TSAN: attributes [[ATTR]] = { noinline nounwind {{.*}} "sanitize_thread_no_checking_at_run_time" {{.*}} }
