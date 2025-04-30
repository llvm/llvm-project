/* Analogous to kernel struct shmid64_ds used on shmctl.  */
struct kernel_shmid64_ds
{
  struct ipc_perm shm_perm;		/* operation permission struct */
  unsigned long int shm_atime_high;
  unsigned long int shm_atime;		/* time of last shmat() */
  unsigned long int shm_dtime_high;
  unsigned long int shm_dtime;		/* time of last shmdt() */
  unsigned long int shm_ctime_high;
  unsigned long int shm_ctime;		/* time of last change by shmctl() */
  unsigned long int __pad;
  size_t shm_segsz;			/* size of segment in bytes */
  __pid_t shm_cpid;			/* pid of creator */
  __pid_t shm_lpid;			/* pid of last shmop */
  shmatt_t shm_nattch;		/* number of current attaches */
  unsigned long int __unused1;
  unsigned long int __unused2;
};
