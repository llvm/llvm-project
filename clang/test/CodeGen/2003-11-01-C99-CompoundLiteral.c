// RUN: %clang_cc1 -emit-llvm %s  -o /dev/null

typedef struct { int foo; } spinlock_t;
typedef struct wait_queue_head_t { spinlock_t lock; } wait_queue_head_t;
void call_usermodehelper(void) { 
  struct wait_queue_head_t work = { lock: (spinlock_t) { 0 }, }; 
}

