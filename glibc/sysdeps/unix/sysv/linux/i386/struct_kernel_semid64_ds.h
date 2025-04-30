/* Analogous to kernel struct semid64_ds used on semctl.  */
struct kernel_semid64_ds
{
  struct ipc_perm sem_perm;
  unsigned long sem_otime;
  unsigned long sem_otime_high;
  unsigned long sem_ctime;
  unsigned long sem_ctime_high;
  __syscall_ulong_t sem_nsems;
  __syscall_ulong_t __unused3;
  __syscall_ulong_t __unused4;
};
