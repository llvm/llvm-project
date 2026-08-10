struct __attribute__((lockable)) mutex {
  void lock() __attribute__((exclusive_lock_function));
  void unlock() __attribute__((unlock_function));
};
