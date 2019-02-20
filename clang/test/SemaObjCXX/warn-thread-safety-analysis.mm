// RUN: %clang_cc1 -fsyntax-only -verify -Wthread-safety -Wthread-safety-beta -Wno-objc-root-class %s

class __attribute__((lockable)) Lock {
public:
  void Acquire() __attribute__((exclusive_lock_function())) {}
  void Release() __attribute__((unlock_function())) {}
};

class __attribute__((scoped_lockable)) AutoLock {
public:
  AutoLock(Lock &lock) __attribute__((exclusive_lock_function(lock)))
  : lock_(lock) {
    lock.Acquire();
  }
  ~AutoLock() __attribute__((unlock_function())) { lock_.Release(); }

private:
  Lock &lock_;
};

@interface MyInterface {
@private
  Lock lock_;
  int value_;
}

- (void)incrementValue;
- (void)decrementValue;

@end

@implementation MyInterface

- (void)incrementValue {
  AutoLock lock(lock_);
  value_ += 1;
}

- (void)decrementValue {
  lock_.Acquire(); // expected-note{{mutex acquired here}}
  value_ -= 1;
} // expected-warning{{mutex 'self->lock_' is still held at the end of function}}

@end
