/* Analogous to kernel struct compat_msqid64_ds used on msgctl.  */
struct kernel_msqid64_ds
{
  struct ipc_perm msg_perm;
  unsigned long msg_stime;
  unsigned long msg_stime_high;
  unsigned long msg_rtime;
  unsigned long msg_rtime_high;
  unsigned long msg_ctime;
  unsigned long msg_ctime_high;
  unsigned long msg_cbytes;
  unsigned long msg_qnum;
  unsigned long msg_qbytes;
  __pid_t msg_lspid;
  __pid_t msg_lrpid;
  unsigned long __unused1;
  unsigned long __unused2;
};
