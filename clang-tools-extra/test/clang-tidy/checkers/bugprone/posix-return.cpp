// RUN: %check_clang_tidy %s bugprone-posix-return %t

#define NULL nullptr
#define ZERO 0
#define NEGATIVE_ONE -1

typedef int pid_t;
typedef long off_t;
typedef decltype(sizeof(int)) size_t;
typedef struct __posix_spawn_file_actions* posix_spawn_file_actions_t;
typedef struct __posix_spawnattr* posix_spawnattr_t;
# define __CPU_SETSIZE 1024
# define __NCPUBITS (8 * sizeof (__cpu_mask))
typedef unsigned long int __cpu_mask;
typedef struct
{
  __cpu_mask __bits[__CPU_SETSIZE / __NCPUBITS];
} cpu_set_t;
typedef struct _opaque_pthread_t *__darwin_pthread_t;
typedef __darwin_pthread_t pthread_t;
typedef struct pthread_attr_t_ *pthread_attr_t;

extern "C" int posix_fadvise(int fd, off_t offset, off_t len, int advice);
extern "C" int posix_fallocate(int fd, off_t offset, off_t len);
extern "C" int posix_madvise(void *addr, size_t len, int advice);
extern "C" int posix_memalign(void **memptr, size_t alignment, size_t size);
extern "C" int posix_openpt(int flags);
extern "C" int posix_spawn(pid_t *pid, const char *path,
                const posix_spawn_file_actions_t *file_actions,
                const posix_spawnattr_t *attrp,
                char *const argv[], char *const envp[]);
extern "C" int posix_spawnp(pid_t *pid, const char *file,
                 const posix_spawn_file_actions_t *file_actions,
                 const posix_spawnattr_t *attrp,
                 char *const argv[], char *const envp[]);
extern "C" int pthread_create(pthread_t *thread, const pthread_attr_t *attr, void *(*start_routine)(void *), void *arg);
extern "C" int pthread_attr_setaffinity_np(pthread_attr_t *attr, size_t cpusetsize, const cpu_set_t *cpuset);
extern "C" int pthread_attr_setschedpolicy(pthread_attr_t *attr, int policy);
extern "C" int pthread_attr_init(pthread_attr_t *attr);
extern "C" int pthread_yield(void);


void warningLessThanZero() {
  if (posix_fadvise(0, 0, 0, 0) < 0) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: the comparison always evaluates to false because posix_fadvise always returns non-negative values
  // CHECK-FIXES: posix_fadvise(0, 0, 0, 0) > 0
  if (posix_fallocate(0, 0, 0) < 0) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:32: warning:
  // CHECK-FIXES: posix_fallocate(0, 0, 0) > 0
  if (posix_madvise(NULL, 0, 0) < 0) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:33: warning:
  // CHECK-FIXES: posix_madvise(NULL, 0, 0) > 0
  if (posix_memalign(NULL, 0, 0) < 0) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning:
  // CHECK-FIXES: posix_memalign(NULL, 0, 0) > 0
  if (posix_spawn(NULL, NULL, NULL, NULL, {NULL}, {NULL}) < 0) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:59: warning:
  // CHECK-FIXES: posix_spawn(NULL, NULL, NULL, NULL, {NULL}, {NULL}) > 0
  if (posix_spawnp(NULL, NULL, NULL, NULL, {NULL}, {NULL}) < 0) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:60: warning:
  // CHECK-FIXES: posix_spawnp(NULL, NULL, NULL, NULL, {NULL}, {NULL}) > 0
  if (pthread_create(NULL, NULL, NULL, NULL) < 0) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:46: warning: the comparison always evaluates to false because pthread_create always returns non-negative values
  // CHECK-FIXES: pthread_create(NULL, NULL, NULL, NULL) > 0
  if (pthread_attr_setaffinity_np(NULL, 0, NULL) < 0) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:50: warning:
  // CHECK-FIXES: pthread_attr_setaffinity_np(NULL, 0, NULL) > 0
  if (pthread_attr_setschedpolicy(NULL, 0) < 0) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:44: warning:
  // CHECK-FIXES: pthread_attr_setschedpolicy(NULL, 0) > 0)
  if (pthread_attr_init(NULL) < 0) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning:
  // CHECK-FIXES: pthread_attr_init(NULL) > 0
  if (pthread_yield() < 0) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning:
  // CHECK-FIXES: pthread_yield() > 0

}

void warningAlwaysTrue() {
  if (posix_fadvise(0, 0, 0, 0) >= 0) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: the comparison always evaluates to true because posix_fadvise always returns non-negative values
  if (pthread_create(NULL, NULL, NULL, NULL) >= 0) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:46: warning: the comparison always evaluates to true because pthread_create always returns non-negative values
  if (pthread_attr_setaffinity_np(NULL, 0, NULL) >= 0) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:50: warning:
  if (pthread_attr_setschedpolicy(NULL, 0) >= 0) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:44: warning:
  if (pthread_attr_init(NULL) >= 0) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning:
  if (pthread_yield() >= 0) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning:

}

void warningEqualsNegative() {
  if (posix_fadvise(0, 0, 0, 0) == -1) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: posix_fadvise
  if (posix_fadvise(0, 0, 0, 0) != -1) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:33: warning:
  if (posix_fadvise(0, 0, 0, 0) <= -1) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:33: warning:
  if (posix_fadvise(0, 0, 0, 0) < -1) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:33: warning:
  if (posix_fallocate(0, 0, 0) == -1) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:32: warning:
  if (posix_madvise(NULL, 0, 0) == -1) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:33: warning:
  if (posix_memalign(NULL, 0, 0) == -1) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning:
  if (posix_spawn(NULL, NULL, NULL, NULL, {NULL}, {NULL}) == -1) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:59: warning:
  if (posix_spawnp(NULL, NULL, NULL, NULL, {NULL}, {NULL}) == -1) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:60: warning:
  if (pthread_create(NULL, NULL, NULL, NULL) == -1) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:46: warning: pthread_create
  if (pthread_create(NULL, NULL, NULL, NULL) != -1) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:46: warning:
  if (pthread_create(NULL, NULL, NULL, NULL) <= -1) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:46: warning:
  if (pthread_create(NULL, NULL, NULL, NULL) < -1) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:46: warning:

}

void WarningWithMacro() {
  if (posix_fadvise(0, 0, 0, 0) < ZERO) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:33: warning:
  // CHECK-FIXES: posix_fadvise(0, 0, 0, 0) > ZERO
  if (posix_fadvise(0, 0, 0, 0) >= ZERO) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:33: warning:
  if (posix_fadvise(0, 0, 0, 0) == NEGATIVE_ONE) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:33: warning:
  if (posix_fadvise(0, 0, 0, 0) != NEGATIVE_ONE) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:33: warning:
  if (posix_fadvise(0, 0, 0, 0) <= NEGATIVE_ONE) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:33: warning:
  if (posix_fadvise(0, 0, 0, 0) < NEGATIVE_ONE) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:33: warning:
  if (pthread_create(NULL, NULL, NULL, NULL) < ZERO) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:46: warning:
  // CHECK-FIXES: pthread_create(NULL, NULL, NULL, NULL) > ZERO
  if (pthread_create(NULL, NULL, NULL, NULL) >= ZERO) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:46: warning:
  if (pthread_create(NULL, NULL, NULL, NULL) == NEGATIVE_ONE) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:46: warning:
  if (pthread_create(NULL, NULL, NULL, NULL) != NEGATIVE_ONE) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:46: warning:
  if (pthread_create(NULL, NULL, NULL, NULL) <= NEGATIVE_ONE) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:46: warning:
  if (pthread_create(NULL, NULL, NULL, NULL) < NEGATIVE_ONE) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:46: warning:

}

void noWarning() {
  if (posix_openpt(0) < 0) {}
  if (posix_openpt(0) <= 0) {}
  if (posix_openpt(0) == -1) {}
  if (posix_openpt(0) != -1) {}
  if (posix_openpt(0) <= -1) {}
  if (posix_openpt(0) < -1) {}
  if (posix_fadvise(0, 0, 0, 0) <= 0) {}
  if (posix_fadvise(0, 0, 0, 0) == 1) {}
}

namespace i {
int posix_fadvise(int fd, off_t offset, off_t len, int advice);
int pthread_yield(void);

void noWarning() {
  if (posix_fadvise(0, 0, 0, 0) < 0) {}
  if (posix_fadvise(0, 0, 0, 0) >= 0) {}
  if (posix_fadvise(0, 0, 0, 0) == -1) {}
  if (posix_fadvise(0, 0, 0, 0) != -1) {}
  if (posix_fadvise(0, 0, 0, 0) <= -1) {}
  if (posix_fadvise(0, 0, 0, 0) < -1) {}
    if (pthread_yield() < 0) {}
    if (pthread_yield() >= 0) {}
    if (pthread_yield() == -1) {}
    if (pthread_yield() != -1) {}
    if (pthread_yield() <= -1) {}
    if (pthread_yield() < -1) {}
}

} // namespace i

class G {
 public:
  int posix_fadvise(int fd, off_t offset, off_t len, int advice);
  int pthread_yield(void);

  void noWarning() {
    if (posix_fadvise(0, 0, 0, 0) < 0) {}
    if (posix_fadvise(0, 0, 0, 0) >= 0) {}
    if (posix_fadvise(0, 0, 0, 0) == -1) {}
    if (posix_fadvise(0, 0, 0, 0) != -1) {}
    if (posix_fadvise(0, 0, 0, 0) <= -1) {}
    if (posix_fadvise(0, 0, 0, 0) < -1) {}
    if (pthread_yield() < 0) {}
    if (pthread_yield() >= 0) {}
    if (pthread_yield() == -1) {}
    if (pthread_yield() != -1) {}
    if (pthread_yield() <= -1) {}
    if (pthread_yield() < -1) {}
  }
};
