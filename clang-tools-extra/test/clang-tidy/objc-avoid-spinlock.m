// RUN: %check_clang_tidy %s objc-avoid-spinlock %t

typedef int OSSpinLock;

@implementation Foo
- (void)f {
    int i = 1;
    OSSpinlockLock(&i);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use os_unfair_lock_lock() or dispatch queue APIs instead of the deprecated OSSpinLock [objc-avoid-spinlock]
    OSSpinlockTry(&i);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use os_unfair_lock_lock() or dispatch queue APIs instead of the deprecated OSSpinLock [objc-avoid-spinlock]
    OSSpinlockUnlock(&i);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use os_unfair_lock_lock() or dispatch queue APIs instead of the deprecated OSSpinLock [objc-avoid-spinlock]
}
@end
