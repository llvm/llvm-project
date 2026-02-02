#define __SIZEOF_PTHREAD_MUTEX_T 40

namespace std {
  typedef union {
    struct __pthread_mutex_s {
      int __lock;
      unsigned int __count;
    } __data;
    char __size[__SIZEOF_PTHREAD_MUTEX_T];
    long int __align;
  } pthread_mutex_t;
};
