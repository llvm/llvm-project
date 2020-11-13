/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#ifndef ABTI_H_INCLUDED
#define ABTI_H_INCLUDED

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <limits.h>

#include "abt_config.h"
#include "abt.h"

#ifndef ABT_CONFIG_DISABLE_ERROR_CHECK
#define ABTI_IS_ERROR_CHECK_ENABLED 1
#else
#define ABTI_IS_ERROR_CHECK_ENABLED 0
#endif

#ifdef ABT_CONFIG_DISABLE_EXT_THREAD
#define ABTI_IS_EXT_THREAD_ENABLED 0
#else
#define ABTI_IS_EXT_THREAD_ENABLED 1
#endif

#include "abtu.h"
#include "abti_error.h"
#include "abti_valgrind.h"

/* Constants */
#define ABTI_SCHED_NUM_PRIO 3

#define ABTI_SCHED_REQ_FINISH (1 << 0)
#define ABTI_SCHED_REQ_EXIT (1 << 1)

#define ABTI_THREAD_REQ_JOIN (1 << 0)
#define ABTI_THREAD_REQ_TERMINATE (1 << 1)
#define ABTI_THREAD_REQ_CANCEL (1 << 2)
#define ABTI_THREAD_REQ_MIGRATE (1 << 3)
#define ABTI_THREAD_REQ_BLOCK (1 << 4)
#define ABTI_THREAD_REQ_ORPHAN (1 << 5)
#define ABTI_THREAD_REQ_NOPUSH (1 << 6)
#define ABTI_THREAD_REQ_NON_YIELD                                              \
    (ABTI_THREAD_REQ_CANCEL | ABTI_THREAD_REQ_MIGRATE |                        \
     ABTI_THREAD_REQ_TERMINATE | ABTI_THREAD_REQ_BLOCK |                       \
     ABTI_THREAD_REQ_ORPHAN | ABTI_THREAD_REQ_NOPUSH)

#define ABTI_THREAD_INIT_ID 0xFFFFFFFFFFFFFFFF
#define ABTI_TASK_INIT_ID 0xFFFFFFFFFFFFFFFF

#define ABTI_INDENT 4

#define ABT_THREAD_TYPE_FULLY_FLEDGED 0
#define ABT_THREAD_TYPE_DYNAMIC_PROMOTION 1

enum ABTI_xstream_type {
    ABTI_XSTREAM_TYPE_PRIMARY,
    ABTI_XSTREAM_TYPE_SECONDARY
};

enum ABTI_sched_used {
    ABTI_SCHED_NOT_USED,
    ABTI_SCHED_MAIN,
    ABTI_SCHED_IN_POOL
};

#define ABTI_THREAD_TYPE_EXT ((ABTI_thread_type)0)
#define ABTI_THREAD_TYPE_THREAD ((ABTI_thread_type)(0x1 << 0))
#define ABTI_THREAD_TYPE_ROOT ((ABTI_thread_type)(0x1 << 1))
#define ABTI_THREAD_TYPE_MAIN ((ABTI_thread_type)(0x1 << 2))
#define ABTI_THREAD_TYPE_MAIN_SCHED ((ABTI_thread_type)(0x1 << 3))
#define ABTI_THREAD_TYPE_YIELDABLE ((ABTI_thread_type)(0x1 << 4))
#define ABTI_THREAD_TYPE_NAMED ((ABTI_thread_type)(0x1 << 5))
#define ABTI_THREAD_TYPE_MIGRATABLE ((ABTI_thread_type)(0x1 << 6))

#define ABTI_THREAD_TYPE_MEM_MEMPOOL_DESC ((ABTI_thread_type)(0x1 << 7))
#define ABTI_THREAD_TYPE_MEM_MALLOC_DESC ((ABTI_thread_type)(0x1 << 8))
#define ABTI_THREAD_TYPE_MEM_MEMPOOL_DESC_STACK ((ABTI_thread_type)(0x1 << 9))
#define ABTI_THREAD_TYPE_MEM_MALLOC_DESC_STACK ((ABTI_thread_type)(0x1 << 10))

#define ABTI_THREAD_TYPES_MEM                                                  \
    (ABTI_THREAD_TYPE_MEM_MEMPOOL_DESC | ABTI_THREAD_TYPE_MEM_MALLOC_DESC |    \
     ABTI_THREAD_TYPE_MEM_MEMPOOL_DESC_STACK |                                 \
     ABTI_THREAD_TYPE_MEM_MALLOC_DESC_STACK)

enum ABTI_mutex_attr_val {
    ABTI_MUTEX_ATTR_NONE = 0,
    ABTI_MUTEX_ATTR_RECURSIVE = 1 << 0
};

/* Macro functions */
#define ABTI_UNUSED(a) (void)(a)

/* Data Types */
typedef struct ABTI_global ABTI_global;
typedef struct ABTI_local ABTI_local;
typedef struct ABTI_local_func ABTI_local_func;
typedef struct ABTI_xstream ABTI_xstream;
typedef enum ABTI_xstream_type ABTI_xstream_type;
typedef struct ABTI_sched ABTI_sched;
typedef char *ABTI_sched_config;
typedef enum ABTI_sched_used ABTI_sched_used;
typedef void *ABTI_sched_id;       /* Scheduler id */
typedef uintptr_t ABTI_sched_kind; /* Scheduler kind */
typedef struct ABTI_pool ABTI_pool;
typedef struct ABTI_thread ABTI_thread;
typedef struct ABTI_thread_attr ABTI_thread_attr;
typedef struct ABTI_ythread ABTI_ythread;
typedef struct ABTI_thread_mig_data ABTI_thread_mig_data;
typedef uint32_t ABTI_thread_type;
typedef struct ABTI_ythread_htable ABTI_ythread_htable;
typedef struct ABTI_ythread_queue ABTI_ythread_queue;
typedef struct ABTI_key ABTI_key;
typedef struct ABTI_ktelem ABTI_ktelem;
typedef struct ABTI_ktable ABTI_ktable;
typedef struct ABTI_mutex_attr ABTI_mutex_attr;
typedef struct ABTI_mutex ABTI_mutex;
typedef struct ABTI_cond ABTI_cond;
typedef struct ABTI_rwlock ABTI_rwlock;
typedef struct ABTI_eventual ABTI_eventual;
typedef struct ABTI_future ABTI_future;
typedef struct ABTI_barrier ABTI_barrier;
typedef struct ABTI_xstream_barrier ABTI_xstream_barrier;
typedef struct ABTI_timer ABTI_timer;
#ifndef ABT_CONFIG_DISABLE_TOOL_INTERFACE
typedef struct ABTI_tool_context ABTI_tool_context;
#endif
/* ID associated with native thread (e.g, Pthreads), which can distinguish
 * execution streams and external threads */
struct ABTI_native_thread_id_opaque;
typedef struct ABTI_native_thread_id_opaque *ABTI_native_thread_id;
/* ID associated with thread (i.e., ULTs, tasklets, and external threads) */
struct ABTI_thread_id_opaque;
typedef struct ABTI_thread_id_opaque *ABTI_thread_id;

/* Architecture-Dependent Definitions */
#include "abtd.h"

/* Spinlock */
typedef struct ABTI_spinlock ABTI_spinlock;
#include "abti_spinlock.h"

/* Basic data structure and memory pool. */
#include "abti_sync_lifo.h"
#include "abti_mem_pool.h"

/* Definitions */
struct ABTI_mutex_attr {
    uint32_t attrs;          /* bit-or'ed attributes */
    uint32_t nesting_cnt;    /* nesting count */
    ABTI_thread_id owner_id; /* owner's ID */
    uint32_t max_handovers;  /* max. # of handovers */
    uint32_t max_wakeups;    /* max. # of wakeups */
};

struct ABTI_mutex {
    ABTD_atomic_uint32 val;        /* 0: unlocked, 1: locked */
    ABTI_mutex_attr attr;          /* attributes */
    ABTI_ythread_htable *p_htable; /* a set of queues */
    ABTI_ythread *p_handover;      /* next ULT for the mutex handover */
    ABTI_ythread *p_giver;         /* current ULT that hands over the mutex */
};

struct ABTI_global {
    int max_xstreams;             /* Largest rank used in Argobots. */
    int num_xstreams;             /* Current # of ESs */
    ABTI_xstream *p_xstream_head; /* List of ESs (head). The list is sorted. */
    ABTI_spinlock
        xstream_list_lock; /* Spinlock protecting ES list. Any read and
                            * write to this list requires a lock.*/

    int num_cores;                /* Number of CPU cores */
    ABT_bool set_affinity;        /* Whether CPU affinity is used */
    ABT_bool use_logging;         /* Whether logging is used */
    ABT_bool use_debug;           /* Whether debug output is used */
    int key_table_size;           /* Default key table size */
    size_t thread_stacksize;      /* Default stack size for ULT (in bytes) */
    size_t sched_stacksize;       /* Default stack size for sched (in bytes) */
    uint32_t sched_event_freq;    /* Default check frequency for sched */
    long sched_sleep_nsec;        /* Default nanoseconds for scheduler sleep */
    ABTI_ythread *p_main_ythread; /* ULT of the main function */

    uint32_t mutex_max_handovers; /* Default max. # of local handovers */
    uint32_t mutex_max_wakeups;   /* Default max. # of wakeups */
    uint32_t os_page_size;        /* OS page size */
    uint32_t huge_page_size;      /* Huge page size */
#ifdef ABT_CONFIG_USE_MEM_POOL
    uint32_t mem_page_size;  /* Page size for memory allocation */
    uint32_t mem_sp_size;    /* Stack page size */
    uint32_t mem_max_stacks; /* Max. # of stacks kept in each ES */
    uint32_t mem_max_descs;  /* Max. # of descriptors kept in each ES */
    int mem_lp_alloc;        /* How to allocate large pages */

    ABTI_mem_pool_global_pool mem_pool_stack; /* Pool of stack (default size) */
    ABTI_mem_pool_global_pool mem_pool_desc;  /* Pool of descriptors that can
                                               * store ABTI_task. */
#ifndef ABT_CONFIG_DISABLE_EXT_THREAD
    /* They are used for external threads. */
    ABTI_spinlock mem_pool_stack_lock;
    ABTI_mem_pool_local_pool mem_pool_stack_ext;
    ABTI_spinlock mem_pool_desc_lock;
    ABTI_mem_pool_local_pool mem_pool_desc_ext;
#endif
#endif

    ABT_bool print_config; /* Whether to print config on ABT_init */

#ifndef ABT_CONFIG_DISABLE_TOOL_INTERFACE
    ABTI_spinlock tool_writer_lock;

    ABT_tool_thread_callback_fn tool_thread_cb_f;
    void *tool_thread_user_arg;
    ABT_tool_task_callback_fn tool_task_cb_f;
    void *tool_task_user_arg;
    ABTD_atomic_uint64 tool_thread_event_mask_tagged;
#endif
};

struct ABTI_local; /* Empty. */

struct ABTI_local_func {
    char padding1[ABT_CONFIG_STATIC_CACHELINE_SIZE];
    ABTI_local *(*get_local_f)(void);
    void (*set_local_xstream_f)(ABTI_xstream *);
    void *(*get_local_ptr_f)(void);
    char padding2[ABT_CONFIG_STATIC_CACHELINE_SIZE];
};

struct ABTI_xstream {
    /* Linked list to manage all execution streams. */
    ABTI_xstream *p_prev;
    ABTI_xstream *p_next;

    int rank;                 /* Rank */
    ABTI_xstream_type type;   /* Type */
    ABTD_atomic_int state;    /* State (ABT_xstream_state) */
    ABTI_sched *p_main_sched; /* Main scheduler, which is the bottom of the
                               * linked list of schedulers */
    ABTD_xstream_context ctx; /* ES context */

    ABTI_ythread
        *p_root_ythread; /* Root thread that schedulers the main scheduler. */
    ABTI_pool *p_root_pool; /* Root pool that stores the main scheduler. */

    ABTU_align_member_var(ABT_CONFIG_STATIC_CACHELINE_SIZE)
        ABTI_thread *p_thread; /* Current running ULT/tasklet */

#ifdef ABT_CONFIG_USE_MEM_POOL
    ABTI_mem_pool_local_pool mem_pool_stack;
    ABTI_mem_pool_local_pool mem_pool_desc;
#endif
};

struct ABTI_sched {
    ABTI_sched_used used;       /* To know if it is used and how */
    ABT_bool automatic;         /* To know if automatic data free */
    ABTI_sched_kind kind;       /* Kind of the scheduler  */
    ABT_sched_type type;        /* Can yield or not (ULT or task) */
    ABTD_atomic_uint32 request; /* Request */
    ABT_pool *pools;            /* Thread pools */
    int num_pools;              /* Number of thread pools */
    ABTI_ythread *p_ythread;    /* Associated ULT */
    void *data;                 /* Data for a specific scheduler */

    /* Scheduler functions */
    ABT_sched_init_fn init;
    ABT_sched_run_fn run;
    ABT_sched_free_fn free;
    ABT_sched_get_migr_pool_fn get_migr_pool;

#ifdef ABT_CONFIG_USE_DEBUG_LOG
    uint64_t id; /* ID */
#endif
};

struct ABTI_pool {
    ABT_pool_access access; /* Access mode */
    ABT_bool automatic;     /* To know if automatic data free */
    /* NOTE: int32_t to check if still positive */
    ABTD_atomic_int32 num_scheds;     /* Number of associated schedulers */
    ABTD_atomic_int32 num_blocked;    /* Number of blocked ULTs */
    ABTD_atomic_int32 num_migrations; /* Number of migrating ULTs */
    void *data;                       /* Specific data */
    uint64_t id;                      /* ID */

    /* Functions to manage units */
    ABT_unit_get_type_fn u_get_type;
    ABT_unit_get_thread_fn u_get_thread;
    ABT_unit_get_task_fn u_get_task;
    ABT_unit_is_in_pool_fn u_is_in_pool;
    ABT_unit_create_from_thread_fn u_create_from_thread;
    ABT_unit_create_from_task_fn u_create_from_task;
    ABT_unit_free_fn u_free;

    /* Functions to manage the pool */
    ABT_pool_init_fn p_init;
    ABT_pool_get_size_fn p_get_size;
    ABT_pool_push_fn p_push;
    ABT_pool_pop_fn p_pop;
    ABT_pool_pop_wait_fn p_pop_wait;
    ABT_pool_pop_timedwait_fn p_pop_timedwait;
    ABT_pool_remove_fn p_remove;
    ABT_pool_free_fn p_free;
    ABT_pool_print_all_fn p_print_all;
};

struct ABTI_thread {
    ABTI_thread *p_prev;
    ABTI_thread *p_next;
    ABTD_atomic_int is_in_pool;   /* Whether this thread is in a pool. */
    ABTI_thread_type type;        /* Thread type */
    ABT_unit unit;                /* Unit enclosing this thread */
    ABTI_xstream *p_last_xstream; /* Last ES where it ran */
    ABTI_thread *p_parent;        /* Parent thread */
    void (*f_thread)(void *);     /* Thread function */
    void *p_arg;                  /* Thread function argument */
    ABTD_atomic_int state;        /* State (ABT_thread_state) */
    ABTD_atomic_uint32 request;   /* Request */
    ABTI_pool *p_pool;            /* Associated pool */
    ABTD_atomic_ptr p_keytable;   /* Thread-specific data (ABTI_ktable *) */
    ABT_unit_id id;               /* ID */
};

struct ABTI_thread_attr {
    void *p_stack;                /* Stack address */
    size_t stacksize;             /* Stack size (in bytes) */
    ABTI_thread_type thread_type; /* Thread type */
#ifndef ABT_CONFIG_DISABLE_MIGRATION
    ABT_bool migratable;              /* Migratability */
    void (*f_cb)(ABT_thread, void *); /* Callback function */
    void *p_cb_arg;                   /* Callback function argument */
#endif
};

struct ABTI_thread_mig_data {
    void (*f_migration_cb)(ABT_thread, void *); /* Callback function */
    void *p_migration_cb_arg;                   /* Callback function argument */
    ABTD_atomic_ptr
        p_migration_pool; /* Destination of migration (ABTI_pool *) */
};

struct ABTI_ythread {
    ABTI_thread thread;       /* Common thread definition */
    ABTD_ythread_context ctx; /* Context */
    void *p_stack;            /* Stack address */
    size_t stacksize;         /* Stack size (in bytes) */
};

struct ABTI_key {
    void (*f_destructor)(void *value);
    uint32_t id;
};

struct ABTI_ktelem {
    /* information of ABTI_key */
    void (*f_destructor)(void *value);
    uint32_t key_id;
    void *value;
    ABTD_atomic_ptr p_next; /* Next element (ABTI_ktelem *) */
};

struct ABTI_ktable {
    int size;           /* size of the table */
    ABTI_spinlock lock; /* Protects any new entry creation. */
    void *p_used_mem;
    void *p_extra_mem;
    size_t extra_mem_size;
    ABTD_atomic_ptr p_elems[1]; /* element array (ABTI_ktelem *) */
};

struct ABTI_cond {
    ABTI_spinlock lock;
    ABTI_mutex *p_waiter_mutex;
    size_t num_waiters;
    ABTI_thread *p_head; /* Head of waiters */
    ABTI_thread *p_tail; /* Tail of waiters */
};

struct ABTI_rwlock {
    ABTI_mutex mutex;
    ABTI_cond cond;
    size_t reader_count;
    int write_flag;
};

struct ABTI_eventual {
    ABTI_spinlock lock;
    ABT_bool ready;
    void *value;
    int nbytes;
    ABTI_thread *p_head; /* Head of waiters */
    ABTI_thread *p_tail; /* Tail of waiters */
};

struct ABTI_future {
    ABTI_spinlock lock;
    ABTD_atomic_uint32 counter;
    uint32_t compartments;
    void **array;
    void (*p_callback)(void **arg);
    ABTI_thread *p_head; /* Head of waiters */
    ABTI_thread *p_tail; /* Tail of waiters */
};

struct ABTI_barrier {
    uint32_t num_waiters;
    volatile uint32_t counter;
    ABTI_ythread **waiters;
    ABT_unit_type *waiter_type;
    ABTI_spinlock lock;
};

struct ABTI_xstream_barrier {
    uint32_t num_waiters;
    ABTD_xstream_barrier bar;
};

struct ABTI_timer {
    ABTD_time start;
    ABTD_time end;
};

#ifndef ABT_CONFIG_DISABLE_TOOL_INTERFACE
struct ABTI_tool_context {
    ABTI_thread *p_caller;
    ABTI_pool *p_pool;
    ABTI_thread
        *p_parent; /* Parent of the target thread.  Used to get the depth */
    ABT_sync_event_type sync_event_type;
    void *p_sync_object; /* ABTI type */
};
#endif

/* Global Data */
extern ABTI_global *gp_ABTI_global;
extern ABTI_local_func gp_ABTI_local_func;

/* ES Local Data */
extern ABTD_XSTREAM_LOCAL ABTI_local *lp_ABTI_local;

/* Execution Stream (ES) */
ABTU_ret_err int ABTI_xstream_create_primary(ABTI_xstream **pp_xstream);
void ABTI_xstream_start_primary(ABTI_xstream **pp_local_xstream,
                                ABTI_xstream *p_xstream,
                                ABTI_ythread *p_ythread);
void ABTI_xstream_free(ABTI_local *p_local, ABTI_xstream *p_xstream,
                       ABT_bool force_free);
void ABTI_xstream_schedule(void *p_arg);
void ABTI_xstream_run_unit(ABTI_xstream **pp_local_xstream, ABT_unit unit,
                           ABTI_pool *p_pool);
void ABTI_xstream_check_events(ABTI_xstream *p_xstream, ABTI_sched *p_sched);
void ABTI_xstream_print(ABTI_xstream *p_xstream, FILE *p_os, int indent,
                        ABT_bool print_sub);

/* Scheduler */
ABT_sched_def *ABTI_sched_get_basic_def(void);
ABT_sched_def *ABTI_sched_get_basic_wait_def(void);
ABT_sched_def *ABTI_sched_get_prio_def(void);
ABT_sched_def *ABTI_sched_get_randws_def(void);
void ABTI_sched_finish(ABTI_sched *p_sched);
void ABTI_sched_exit(ABTI_sched *p_sched);
ABTU_ret_err int ABTI_sched_create_basic(ABT_sched_predef predef, int num_pools,
                                         ABT_pool *pools,
                                         ABT_sched_config config,
                                         ABTI_sched **pp_newsched);
void ABTI_sched_free(ABTI_local *p_local, ABTI_sched *p_sched,
                     ABT_bool force_free);
ABTU_ret_err int ABTI_sched_get_migration_pool(ABTI_sched *, ABTI_pool *,
                                               ABTI_pool **);
ABT_bool ABTI_sched_has_to_stop(ABTI_local **pp_local, ABTI_sched *p_sched);
size_t ABTI_sched_get_size(ABTI_sched *p_sched);
size_t ABTI_sched_get_total_size(ABTI_sched *p_sched);
size_t ABTI_sched_get_effective_size(ABTI_local *p_local, ABTI_sched *p_sched);
void ABTI_sched_print(ABTI_sched *p_sched, FILE *p_os, int indent,
                      ABT_bool print_sub);
void ABTI_sched_reset_id(void);

/* Scheduler config */
ABTU_ret_err int ABTI_sched_config_read(ABT_sched_config config, int type,
                                        int num_vars, void **variables);
ABTU_ret_err int ABTI_sched_config_read_global(ABT_sched_config config,
                                               ABT_pool_access *access,
                                               ABT_bool *automatic);

/* Pool */
ABTU_ret_err int ABTI_pool_create_basic(ABT_pool_kind kind,
                                        ABT_pool_access access,
                                        ABT_bool automatic,
                                        ABTI_pool **pp_newpool);
void ABTI_pool_free(ABTI_pool *p_pool);
ABTU_ret_err int ABTI_pool_get_fifo_def(ABT_pool_access access,
                                        ABT_pool_def *p_def);
ABTU_ret_err int ABTI_pool_get_fifo_wait_def(ABT_pool_access access,
                                             ABT_pool_def *p_def);
void ABTI_pool_print(ABTI_pool *p_pool, FILE *p_os, int indent);
void ABTI_pool_reset_id(void);

/* Work Unit */
void ABTI_unit_set_associated_pool(ABT_unit unit, ABTI_pool *p_pool);

/* Threads */
ABTU_ret_err int ABTI_thread_get_mig_data(ABTI_local *p_local,
                                          ABTI_thread *p_thread,
                                          ABTI_thread_mig_data **pp_mig_data);
void ABTI_thread_revive(ABTI_local *p_local, ABTI_pool *p_pool,
                        void (*thread_func)(void *), void *arg,
                        ABTI_thread *p_thread);
void ABTI_thread_join(ABTI_local **pp_local, ABTI_thread *p_thread);
void ABTI_thread_free(ABTI_local *p_local, ABTI_thread *p_thread);
void ABTI_thread_print(ABTI_thread *p_thread, FILE *p_os, int indent);
void ABTI_thread_reset_id(void);
ABT_unit_id ABTI_thread_get_id(ABTI_thread *p_thread);

/* Yieldable threads */
ABTU_ret_err int ABTI_ythread_create_root(ABTI_local *p_local,
                                          ABTI_xstream *p_xstream,
                                          ABTI_ythread **pp_root_ythread);
ABTU_ret_err int ABTI_ythread_create_main(ABTI_local *p_local,
                                          ABTI_xstream *p_xstream,
                                          ABTI_ythread **p_ythread);
ABTU_ret_err int ABTI_ythread_create_main_sched(ABTI_local *p_local,
                                                ABTI_xstream *p_xstream,
                                                ABTI_sched *p_sched);
ABTU_ret_err int ABTI_ythread_create_sched(ABTI_local *p_local,
                                           ABTI_pool *p_pool,
                                           ABTI_sched *p_sched);
ABTU_noreturn void ABTI_ythread_exit(ABTI_xstream *p_local_xstream,
                                     ABTI_ythread *p_ythread);
void ABTI_ythread_free_main(ABTI_local *p_local, ABTI_ythread *p_ythread);
void ABTI_ythread_free_root(ABTI_local *p_local, ABTI_ythread *p_ythread);
void ABTI_ythread_set_blocked(ABTI_ythread *p_ythread);
void ABTI_ythread_suspend(ABTI_xstream **pp_local_xstream,
                          ABTI_ythread *p_ythread,
                          ABT_sync_event_type sync_event_type, void *p_sync);
void ABTI_ythread_set_ready(ABTI_local *p_local, ABTI_ythread *p_ythread);
void ABTI_ythread_print_stack(ABTI_ythread *p_ythread, FILE *p_os);

/* Thread attributes */
void ABTI_thread_attr_print(ABTI_thread_attr *p_attr, FILE *p_os, int indent);
ABTU_ret_err int
ABTI_thread_attr_dup(const ABTI_thread_attr *p_attr,
                     ABTI_thread_attr **pp_dup_attr) ABTU_ret_err;

/* Thread hash table */
ABTU_ret_err int ABTI_ythread_htable_create(uint32_t num_rows,
                                            ABTI_ythread_htable **pp_htable);
void ABTI_ythread_htable_free(ABTI_ythread_htable *p_htable);
void ABTI_ythread_htable_push(ABTI_ythread_htable *p_htable, int idx,
                              ABTI_ythread *p_ythread);
void ABTI_ythread_htable_push_low(ABTI_ythread_htable *p_htable, int idx,
                                  ABTI_ythread *p_ythread);
ABTI_ythread *ABTI_ythread_htable_pop(ABTI_ythread_htable *p_htable,
                                      ABTI_ythread_queue *p_queue);
ABTI_ythread *ABTI_ythread_htable_pop_low(ABTI_ythread_htable *p_htable,
                                          ABTI_ythread_queue *p_queue);
ABT_bool ABTI_ythread_htable_switch_low(ABTI_xstream **pp_local_xstream,
                                        ABTI_ythread_queue *p_queue,
                                        ABTI_ythread *p_ythread,
                                        ABTI_ythread_htable *p_htable,
                                        ABT_sync_event_type sync_event_type,
                                        void *p_sync);
/* Key */
void ABTI_ktable_free(ABTI_local *p_local, ABTI_ktable *p_ktable);

/* Mutex */
void ABTI_mutex_wait(ABTI_xstream **pp_local_xstream, ABTI_mutex *p_mutex,
                     int val);
void ABTI_mutex_wait_low(ABTI_xstream **pp_local_xstream, ABTI_mutex *p_mutex,
                         int val);
void ABTI_mutex_wake_de(ABTI_local *p_local, ABTI_mutex *p_mutex);

/* Information */
void ABTI_info_print_config(FILE *fp);
void ABTI_info_check_print_all_thread_stacks(void);

#include "abti_log.h"
#include "abti_local.h"
#include "abti_self.h"
#include "abti_pool.h"
#include "abti_sched.h"
#include "abti_config.h"
#include "abti_stream.h"
#include "abti_thread.h"
#include "abti_tool.h"
#include "abti_ythread.h"
#include "abti_thread_attr.h"
#include "abti_mutex.h"
#include "abti_mutex_attr.h"
#include "abti_cond.h"
#include "abti_rwlock.h"
#include "abti_eventual.h"
#include "abti_future.h"
#include "abti_barrier.h"
#include "abti_stream_barrier.h"
#include "abti_timer.h"
#include "abti_mem.h"
#include "abti_key.h"

#endif /* ABTI_H_INCLUDED */
