// TODO: license block

#include "test/IntegrationTest/test.h"

#include "src/pthread/pthread_cond_destroy.h"
#include "src/pthread/pthread_cond_init.h"
#include "src/pthread/pthread_cond_signal.h"
#include "src/pthread/pthread_cond_wait.h"
#include "src/pthread/pthread_create.h"
#include "src/pthread/pthread_join.h"
#include "src/pthread/pthread_mutex_destroy.h"
#include "src/pthread/pthread_mutex_init.h"
#include "src/pthread/pthread_mutex_lock.h"
#include "src/pthread/pthread_mutex_unlock.h"

#include <pthread.h>

pthread_mutex_t waiter_mtx, main_thread_mtx;
pthread_cond_t waiter_cnd, main_thread_cnd;

void *waiter_thread_func(void *) {
  LIBC_NAMESPACE::pthread_mutex_lock(&waiter_mtx);

  LIBC_NAMESPACE::pthread_mutex_lock(&main_thread_mtx);
  LIBC_NAMESPACE::pthread_cond_signal(&main_thread_cnd);
  LIBC_NAMESPACE::pthread_mutex_unlock(&main_thread_mtx);

  LIBC_NAMESPACE::pthread_cond_wait(&waiter_cnd, &waiter_mtx);
  LIBC_NAMESPACE::pthread_mutex_unlock(&waiter_mtx);

  return reinterpret_cast<void *>(0x600D);
}

void single_waiter_test() {
  ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_init(&waiter_mtx, nullptr), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_init(&main_thread_mtx, nullptr), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_cond_init(&waiter_cnd, nullptr), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_cond_init(&main_thread_cnd, nullptr), 0);

  ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_lock(&main_thread_mtx), 0);

  pthread_t waiter_thread;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_create(&waiter_thread, nullptr,
                                           waiter_thread_func, nullptr),
            0);

  ASSERT_EQ(
      LIBC_NAMESPACE::pthread_cond_wait(&main_thread_cnd, &main_thread_mtx), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_unlock(&main_thread_mtx), 0);

  ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_lock(&waiter_mtx), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_cond_signal(&waiter_cnd), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_unlock(&waiter_mtx), 0);

  void *retval;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_join(waiter_thread, &retval), 0);
  ASSERT_EQ(reinterpret_cast<long>(retval), 0x600D);

  ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_destroy(&waiter_mtx), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_destroy(&main_thread_mtx), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_cond_destroy(&waiter_cnd), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_cond_destroy(&main_thread_cnd), 0);
}

TEST_MAIN() {
  single_waiter_test();
  return 0;
}
