/* Analogous to kernel struct shmid64_ds used on shmctl.  */
struct kernel_shmid64_ds
{
  struct ipc_perm shm_perm;
  size_t shm_segsz;
#if __TIMESIZE == 64
  long int shm_atime;
  long int shm_dtime;
  long int shm_ctime;
#else
  unsigned long int shm_atime;
  unsigned long int shm_dtime;
  unsigned long int shm_ctime;
#endif
  __pid_t shm_cpid;
  __pid_t shm_lpid;
  unsigned long int shm_nattch;
#if __TIMESIZE == 64
  unsigned long int __unused1;
  unsigned long int __unused2;
#else
  unsigned short int shm_atime_high;
  unsigned short int shm_dtime_high;
  unsigned short int shm_ctime_high;
  unsigned short int __ununsed1;
#endif
};
