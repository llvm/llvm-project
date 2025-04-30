#ifndef _SPARC_NPTL_H

union sparc_pthread_barrier
{
  struct pthread_barrier b;
  struct sparc_pthread_barrier_s
    {
      unsigned int curr_event;
      int lock;
      unsigned int left;
      unsigned int init_count;
      unsigned char left_lock;
      unsigned char pshared;
    } s;
};

struct sparc_new_sem
{
  unsigned int value;
  unsigned char lock;
  unsigned char private;
  unsigned char pad[2];
  unsigned long int nwaiters;
};

struct sparc_old_sem
{
  unsigned int value;
  unsigned char lock;
  unsigned char private;
};

#endif
