/* Analogous to kernel struct msqid64_ds used on msgctl.  It is only used
   for 32-bit architectures on 64-bit time_t msgctl64 implementation.  */
struct kernel_msqid64_ds
{
  struct ipc_perm msg_perm;
  unsigned long int msg_stime;
  unsigned long int msg_stime_high;
  unsigned long int msg_rtime;
  unsigned long int msg_rtime_high;
  unsigned long int msg_ctime;
  unsigned long int msg_ctime_high;
  unsigned long int msg_cbytes;
  unsigned long int msg_qnum;
  unsigned long int msg_qbytes;
  __pid_t msg_lspid;
  __pid_t msg_lrpid;
  unsigned long int __unused4;
  unsigned long int __unused5;
};
