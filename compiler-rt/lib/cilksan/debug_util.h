#ifndef __DEBUG_UTIL_H__
#define __DEBUG_UTIL_H__

#include <assert.h>

#define CILKSAN_DEBUG 1

// if NULL, err_io is set to stderr
#define ERROR_FILE NULL

// debug_level is a bitmap
//   1 is basic debugging (old level 1)
//   2 is debug the backtrace
enum debug_levels {
    DEBUG_BASIC      = 1,
    DEBUG_BACKTRACE  = 2,
    DEBUG_BAGS       = 4,
    DEBUG_CALLBACK   = 8,
    DEBUG_MEMORY     = 16,
    DEBUG_DEQUE      = 32,
    DEBUG_REDUCER    = 64,
    DEBUG_DISJOINTSET = 128
};

#if CILKSAN_DEBUG
static int debug_level = 0; // DEBUG_BASIC | DEBUG_BAGS | DEBUG_CALLBACK | DEBUG_DISJOINTSET | DEBUG_MEMORY;
#else
static int debug_level = 0;
#endif

#if CILKSAN_DEBUG
#define WHEN_CILKSAN_DEBUG(stmt) do { stmt; } while(0)
#define cilksan_assert(c) \
    do { if (!(c)) { die("%s:%d assertion failure: %s\n", \
                        __FILE__, __LINE__, #c);} } while (0)
#else
#define WHEN_CILKSAN_DEBUG(stmt)
#define cilksan_assert(c)
#endif

#if CILKSAN_DEBUG
// debugging assert to check that the tool is catching all the runtime events
// that are supposed to match up (i.e., has event begin and event end)
enum EventType_t { ENTER_FRAME = 1, ENTER_HELPER = 2, SPAWN_PREPARE = 3,
                   DETACH = 4, CILK_SYNC = 5, LEAVE_FRAME_OR_HELPER = 6,
                   RUNTIME_LOOP = 7, NONE = 8 };
#endif

__attribute__((noreturn))
void die(const char *fmt, ...);
void debug_printf(int level, const char *fmt, ...);

#if CILKSAN_DEBUG
#define DBG_TRACE(level,...) debug_printf(level, __VA_ARGS__)
#else
#define DBG_TRACE(level,...)
#endif

#endif
