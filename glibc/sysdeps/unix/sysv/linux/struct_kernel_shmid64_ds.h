/* Analogous to kernel struct shmid64_ds used on shmctl.  */
struct kernel_shmid64_ds
{
  struct ipc_perm shm_perm;
  size_t shm_segsz;
  unsigned long int shm_atime;
  unsigned long int shm_atime_high;
  unsigned long int shm_dtime;
  unsigned long int shm_dtime_high;
  unsigned long int shm_ctime;
  unsigned long int shm_ctime_high;
  __pid_t shm_cpid;
  __pid_t shm_lpid;
  unsigned long int shm_nattch;
  unsigned long int __unused1;
  unsigned long int __unused2;
};
