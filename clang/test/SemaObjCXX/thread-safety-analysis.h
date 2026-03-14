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
