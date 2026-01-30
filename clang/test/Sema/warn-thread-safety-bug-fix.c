// RUN: %clang_cc1 -fsyntax-only -verify -Wthread-safety -Wthread-safety-pointer -Wthread-safety-beta %s

typedef int __attribute__((capability("lock"))) lock_t;

typedef struct {
    lock_t lock;
} * map_t;

typedef struct task {
    map_t map;
} *task_t;

#define ACQUIRES_LOCK(map) \
    __attribute__((acquire_capability((map)->lock)))
#define RELEASES_LOCK(map) \
    __attribute__((release_capability((map)->lock)))

extern void lock_map(map_t map) ACQUIRES_LOCK(map);
extern void unlock_map_indirect(map_t *mapp) RELEASES_LOCK(*mapp);
extern void f(void *, void *, void *);

static void saveContexBug(task_t task)
{
    map_t map;
    map = task->map;
    lock_map(map); // expected-note{{lock acquired here}}
    map_t *mapp = &map;
    // Previously, a local-variable-definition-context was created and
    // pushed for each of the argument below, resulting context
    // mismatch. The analyzer missed the fact that 'mapp' may no
    // longer point to the lock. So it does not report an issue at the
    // 'unlock_map_indirect' call.
    f(&map, &map, &mapp);
    unlock_map_indirect(mapp); // expected-warning{{releasing lock 'mapp->lock' that was not held}}
} // expected-warning{{lock 'task->map->lock' is still held at the end of function}}

