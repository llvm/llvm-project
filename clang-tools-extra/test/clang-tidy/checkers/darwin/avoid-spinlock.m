// RUN: %check_clang_tidy %s darwin-avoid-spinlock %t

typedef int OSSpinLock;

void OSSpinlockLock(OSSpinLock *l);
void OSSpinlockTry(OSSpinLock *l);
void OSSpinlockUnlock(OSSpinLock *l);

@implementation Foo
- (void)f {
    int i = 1;
    OSSpinlockLock(&i);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use os_unfair_lock_lock() or dispatch queue APIs instead of the deprecated OSSpinLock [darwin-avoid-spinlock]
    OSSpinlockTry(&i);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use os_unfair_lock_lock() or dispatch queue APIs instead of the deprecated OSSpinLock [darwin-avoid-spinlock]
    OSSpinlockUnlock(&i);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use os_unfair_lock_lock() or dispatch queue APIs instead of the deprecated OSSpinLock [darwin-avoid-spinlock]
}
@end
