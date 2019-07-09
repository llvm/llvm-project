// RUN: %check_clang_tidy %s bugprone-posix-return %t

#define NULL nullptr
#define ZERO 0
#define NEGATIVE_ONE -1

typedef int pid_t;
typedef long off_t;
typedef decltype(sizeof(int)) size_t;
typedef struct __posix_spawn_file_actions* posix_spawn_file_actions_t;
typedef struct __posix_spawnattr* posix_spawnattr_t;

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
}

void warningAlwaysTrue() {
  if (posix_fadvise(0, 0, 0, 0) >= 0) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: the comparison always evaluates to true because posix_fadvise always returns non-negative values
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

void noWarning() {
  if (posix_fadvise(0, 0, 0, 0) < 0) {}
  if (posix_fadvise(0, 0, 0, 0) >= 0) {}
  if (posix_fadvise(0, 0, 0, 0) == -1) {}
  if (posix_fadvise(0, 0, 0, 0) != -1) {}
  if (posix_fadvise(0, 0, 0, 0) <= -1) {}
  if (posix_fadvise(0, 0, 0, 0) < -1) {}
}

} // namespace i

class G {
 public:
  int posix_fadvise(int fd, off_t offset, off_t len, int advice);

  void noWarning() {
    if (posix_fadvise(0, 0, 0, 0) < 0) {}
    if (posix_fadvise(0, 0, 0, 0) >= 0) {}
    if (posix_fadvise(0, 0, 0, 0) == -1) {}
    if (posix_fadvise(0, 0, 0, 0) != -1) {}
    if (posix_fadvise(0, 0, 0, 0) <= -1) {}
    if (posix_fadvise(0, 0, 0, 0) < -1) {}
  }
};
