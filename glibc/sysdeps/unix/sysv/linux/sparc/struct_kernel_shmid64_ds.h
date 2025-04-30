/* Analogous to kernel struct shmid64_ds used on shmctl.  */
struct kernel_shmid64_ds
{
  struct ipc_perm shm_perm;
  unsigned long int shm_atime_high;
  unsigned long int shm_atime;
  unsigned long int shm_dtime_high;
  unsigned long int shm_dtime;
  unsigned long int shm_ctime_high;
  unsigned long int shm_ctime;
  size_t shm_segsz;
  __pid_t shm_cpid;
  __pid_t shm_lpid;
  unsigned long int shm_nattch;
  unsigned long int __unused1;
  unsigned long int __unused2;
};
