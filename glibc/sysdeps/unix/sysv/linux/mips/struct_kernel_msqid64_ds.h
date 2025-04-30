/* Analogous to kernel struct msqid64_ds used on msgctl.  */
struct kernel_msqid64_ds
{
  struct ipc_perm msg_perm;
#if __TIMESIZE == 32
# ifdef __MIPSEL__
  unsigned long int msg_stime;
  unsigned long int msg_stime_high;
  unsigned long int msg_rtime;
  unsigned long int msg_rtime_high;
  unsigned long int msg_ctime;
  unsigned long int msg_ctime_high;
# else
  unsigned long int msg_stime_high;
  unsigned long int msg_stime;
  unsigned long int msg_rtime_high;
  unsigned long int msg_rtime;
  unsigned long int msg_ctime_high;
  unsigned long int msg_ctime;
# endif
#else
  unsigned long int msg_stime;
  unsigned long int msg_rtime;
  unsigned long int msg_ctime;
#endif
  unsigned long int msg_cbytes;
  unsigned long int msg_qnum;
  unsigned long int msg_qbytes;
  __pid_t msg_lspid;
  __pid_t msg_lrpid;
  unsigned long int __unused1;
  unsigned long int __unused2;
};
