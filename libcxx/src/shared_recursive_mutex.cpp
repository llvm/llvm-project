#include <__threading_support>
#include <shared_recursive_mutex>

constexpr uint32_t SHARED_USER_MAX_CNT = UINT32_MAX;
std::__thread_id THREAD_ID_NOEXIST     = std::__thread_id();

using namespace std::__shared_recursive_mutex;

void shared_recursive_mutex::lock() {
  std::__thread_id self = std::this_thread::get_id();
  std::unique_lock guard(__mutex_);

  // write lock reentrant
  if (__owner_ == self) {
    if (__recursions_ == SHARED_USER_MAX_CNT) {
      return;
    }
    ++__recursions_;
    return;
  }

  while ((__owner_ != THREAD_ID_NOEXIST) || (__readers_ != 0)) {
    __cond_.wait(guard);
  }

  __owner_      = self;
  __recursions_ = 1;
}

bool shared_recursive_mutex::try_lock() {
  std::__thread_id self = std::this_thread::get_id();
  std::lock_guard guard(__mutex_);

  // write lock reentrant
  if (__owner_ == self) {
    if (__recursions_ == SHARED_USER_MAX_CNT) {
      return false;
    }
    ++__recursions_;
    return true;
  }

  if ((__owner_ != THREAD_ID_NOEXIST) || (__readers_ != 0)) {
    return false;
  }

  __owner_      = self;
  __recursions_ = 1;

  return true;
}

void shared_recursive_mutex::unlock() {
  std::__thread_id self = std::this_thread::get_id();
  std::lock_guard guard(__mutex_);

  if ((__owner_ != self) || (__recursions_ == 0)) {
    return;
  }

  if (--__recursions_ == 0) {
    __owner_.__reset();
    // release write lock and notifies all the servers.
    __cond_.notify_all();
  }
}

void shared_recursive_mutex::lock_shared() {
  std::__thread_id self = std::this_thread::get_id();
  std::unique_lock guard(__mutex_);

  // write-read nesting
  if (__owner_ == self) {
    ++__readers_;
    return;
  }

  // If other threads have held the write lock or the number of read locks exceeds the upper limit, wait.
  while (__owner_ != THREAD_ID_NOEXIST || __readers_ == SHARED_USER_MAX_CNT) {
    __cond_.wait(guard);
  }

  ++__readers_;
}

bool shared_recursive_mutex::try_lock_shared() {
  std::__thread_id self = std::this_thread::get_id();
  std::lock_guard guard(__mutex_);

  // write-read nesting
  if (__owner_ == self) {
    ++__readers_;
    return true;
  }

  // If another thread already holds the write lock or the number of read locks exceeds the upper limit,
  // the operation fails.
  if (__owner_ != THREAD_ID_NOEXIST || __readers_ == SHARED_USER_MAX_CNT) {
    return false;
  }

  ++__readers_;
  return true;
}

void shared_recursive_mutex::unlock_shared() {
  std::lock_guard guard(__mutex_);

  if (__readers_ == 0) {
    return;
  }

  --__readers_;
  __cond_.notify_all();
}
