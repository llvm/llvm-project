// RUN: %check_clang_tidy %s bugprone-spuriously-wake-up-functions %t -- --
#define NULL 0

struct Node1 {
  void *Node1;
  struct Node1 *next;
};

typedef struct mtx_t {
} mtx_t;
typedef struct cnd_t {
} cnd_t;
struct timespec {};

int cnd_wait(cnd_t *cond, mtx_t *mutex){};
int cnd_timedwait(cnd_t *cond, mtx_t *mutex,
                  const struct timespec *time_point){};

struct Node1 list_c;
static mtx_t lock;
static cnd_t condition_c;
struct timespec ts;

void consume_list_element(void) {

  if (list_c.next == NULL) {
    if (0 != cnd_wait(&condition_c, &lock)) {
      // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: 'cnd_wait' should be placed inside a while statement [bugprone-spuriously-wake-up-functions]
    }
  }
  if (list_c.next == NULL)
    if (0 != cnd_wait(&condition_c, &lock))
      // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: 'cnd_wait' should be placed inside a while statement [bugprone-spuriously-wake-up-functions]
      ;
  if (list_c.next == NULL && 0 != cnd_wait(&condition_c, &lock))
    // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: 'cnd_wait' should be placed inside a while statement [bugprone-spuriously-wake-up-functions]
    ;
  while (list_c.next == NULL) {
    if (0 != cnd_wait(&condition_c, &lock)) {
    }
  }
  while (list_c.next == NULL)
    if (0 != cnd_wait(&condition_c, &lock)) {
    }
  while (list_c.next == NULL)
    if (0 != cnd_wait(&condition_c, &lock))
      ;
  if (list_c.next == NULL) {
    cnd_wait(&condition_c, &lock);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'cnd_wait' should be placed inside a while statement [bugprone-spuriously-wake-up-functions]
  }
  if (list_c.next == NULL)
    cnd_wait(&condition_c, &lock);
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'cnd_wait' should be placed inside a while statement [bugprone-spuriously-wake-up-functions]
  while (list_c.next == NULL) {
    cnd_wait(&condition_c, &lock);
  }
  while (list_c.next == NULL)
    cnd_wait(&condition_c, &lock);

  do {
    if (0 != cnd_wait(&condition_c, &lock)) {
    }
  } while (list_c.next == NULL);
  do
    if (0 != cnd_wait(&condition_c, &lock)) {
    }
  while (list_c.next == NULL);
  do
    if (0 != cnd_wait(&condition_c, &lock))
      ;
  while (list_c.next == NULL);
  do {
    cnd_wait(&condition_c, &lock);
  } while (list_c.next == NULL);
  do
    cnd_wait(&condition_c, &lock);
  while (list_c.next == NULL);
  for (;; list_c.next == NULL) {
    if (0 != cnd_wait(&condition_c, &lock)) {
    }
  }
  for (;; list_c.next == NULL)
    if (0 != cnd_wait(&condition_c, &lock)) {
    }
  for (;; list_c.next == NULL)
    if (0 != cnd_wait(&condition_c, &lock))
      ;
  for (;; list_c.next == NULL) {
    cnd_wait(&condition_c, &lock);
  }
  for (;; list_c.next == NULL)
    cnd_wait(&condition_c, &lock);

  if (list_c.next == NULL) {
    if (0 != cnd_timedwait(&condition_c, &lock, &ts)) {
      // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: 'cnd_timedwait' should be placed inside a while statement [bugprone-spuriously-wake-up-functions]
    }
  }
  if (list_c.next == NULL)
    if (0 != cnd_timedwait(&condition_c, &lock, &ts))
      // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: 'cnd_timedwait' should be placed inside a while statement [bugprone-spuriously-wake-up-functions]
      ;
  if (list_c.next == NULL && 0 != cnd_timedwait(&condition_c, &lock, &ts))
    // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: 'cnd_timedwait' should be placed inside a while statement [bugprone-spuriously-wake-up-functions]
    ;
  while (list_c.next == NULL) {
    if (0 != cnd_timedwait(&condition_c, &lock, &ts)) {
    }
  }
  while (list_c.next == NULL)
    if (0 != cnd_timedwait(&condition_c, &lock, &ts)) {
    }
  while (list_c.next == NULL)
    if (0 != cnd_timedwait(&condition_c, &lock, &ts))
      ;
  if (list_c.next == NULL) {
    cnd_timedwait(&condition_c, &lock, &ts);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'cnd_timedwait' should be placed inside a while statement [bugprone-spuriously-wake-up-functions]
  }
  if (list_c.next == NULL)
    cnd_timedwait(&condition_c, &lock, &ts);
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'cnd_timedwait' should be placed inside a while statement [bugprone-spuriously-wake-up-functions]
  while (list_c.next == NULL) {
    cnd_timedwait(&condition_c, &lock, &ts);
  }
  while (list_c.next == NULL)
    cnd_timedwait(&condition_c, &lock, &ts);

  do {
    if (0 != cnd_timedwait(&condition_c, &lock, &ts)) {
    }
  } while (list_c.next == NULL);
  do
    if (0 != cnd_timedwait(&condition_c, &lock, &ts)) {
    }
  while (list_c.next == NULL);
  do
    if (0 != cnd_timedwait(&condition_c, &lock, &ts))
      ;
  while (list_c.next == NULL);
  do {
    cnd_timedwait(&condition_c, &lock, &ts);
  } while (list_c.next == NULL);
  do
    cnd_timedwait(&condition_c, &lock, &ts);
  while (list_c.next == NULL);
  for (;; list_c.next == NULL) {
    if (0 != cnd_timedwait(&condition_c, &lock, &ts)) {
    }
  }
  for (;; list_c.next == NULL)
    if (0 != cnd_timedwait(&condition_c, &lock, &ts)) {
    }
  for (;; list_c.next == NULL)
    if (0 != cnd_timedwait(&condition_c, &lock, &ts))
      ;
  for (;; list_c.next == NULL) {
    cnd_timedwait(&condition_c, &lock, &ts);
  }
  for (;; list_c.next == NULL)
    cnd_timedwait(&condition_c, &lock, &ts);
}
int main() { return 0; }
