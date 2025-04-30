/* Analogous to kernel struct semid64_ds used on semctl.  */
struct kernel_semid64_ds
{
  struct ipc_perm sem_perm;
  unsigned long sem_otime_high;
  unsigned long sem_otime;
  unsigned long sem_ctime_high;
  unsigned long sem_ctime;
  unsigned long sem_nsems;
  unsigned long __unused1;
  unsigned long __unused2;
};
