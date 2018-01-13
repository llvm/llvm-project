//===-- sanitizer_platform_limits_netbsd.h --------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of Sanitizer common code.
//
// Sizes and layouts of platform-specific NetBSD data structures.
//===----------------------------------------------------------------------===//

#ifndef SANITIZER_PLATFORM_LIMITS_NETBSD_H
#define SANITIZER_PLATFORM_LIMITS_NETBSD_H

#if SANITIZER_NETBSD

#include "sanitizer_internal_defs.h"
#include "sanitizer_platform.h"

#define _GET_LINK_MAP_BY_DLOPEN_HANDLE(handle, shift) \
  ((link_map *)((handle) == nullptr ? nullptr : ((char *)(handle) + (shift))))

#if defined(__x86_64__)
#define GET_LINK_MAP_BY_DLOPEN_HANDLE(handle) \
  _GET_LINK_MAP_BY_DLOPEN_HANDLE(handle, 312)
#elif defined(__i386__)
#define GET_LINK_MAP_BY_DLOPEN_HANDLE(handle) \
  _GET_LINK_MAP_BY_DLOPEN_HANDLE(handle, 164)
#endif

namespace __sanitizer {
extern unsigned struct_utsname_sz;
extern unsigned struct_stat_sz;
extern unsigned struct_rusage_sz;
extern unsigned siginfo_t_sz;
extern unsigned struct_itimerval_sz;
extern unsigned pthread_t_sz;
extern unsigned pthread_mutex_t_sz;
extern unsigned pthread_cond_t_sz;
extern unsigned pid_t_sz;
extern unsigned timeval_sz;
extern unsigned uid_t_sz;
extern unsigned gid_t_sz;
extern unsigned mbstate_t_sz;
extern unsigned struct_timezone_sz;
extern unsigned struct_tms_sz;
extern unsigned struct_itimerspec_sz;
extern unsigned struct_sigevent_sz;
extern unsigned struct_sched_param_sz;
extern unsigned struct_statfs_sz;
extern unsigned struct_sockaddr_sz;
extern unsigned ucontext_t_sz;

extern unsigned struct_rlimit_sz;
extern unsigned struct_utimbuf_sz;
extern unsigned struct_timespec_sz;

struct __sanitizer_iocb {
  u64 aio_offset;
  uptr aio_buf;
  long aio_nbytes;
  u32 aio_fildes;
  u32 aio_lio_opcode;
  long aio_reqprio;
#if SANITIZER_WORDSIZE == 64
  u8 aio_sigevent[32];
#else
  u8 aio_sigevent[20];
#endif
  u32 _state;
  u32 _errno;
  long _retval;
};

struct __sanitizer___sysctl_args {
  int *name;
  int nlen;
  void *oldval;
  uptr *oldlenp;
  void *newval;
  uptr newlen;
};

struct __sanitizer_sem_t {
  uptr data[5];
};

struct __sanitizer_ipc_perm {
  u32 uid;
  u32 gid;
  u32 cuid;
  u32 cgid;
  u32 mode;
  unsigned short _seq;
  long _key;
};

struct __sanitizer_shmid_ds {
  __sanitizer_ipc_perm shm_perm;
  unsigned long shm_segsz;
  u32 shm_lpid;
  u32 shm_cpid;
  unsigned int shm_nattch;
  u64 shm_atime;
  u64 shm_dtime;
  u64 shm_ctime;
  void *_shm_internal;
};

extern unsigned struct_msqid_ds_sz;
extern unsigned struct_mq_attr_sz;
extern unsigned struct_timex_sz;
extern unsigned struct_statvfs_sz;

struct __sanitizer_iovec {
  void *iov_base;
  uptr iov_len;
};

struct __sanitizer_ifaddrs {
  struct __sanitizer_ifaddrs *ifa_next;
  char *ifa_name;
  unsigned int ifa_flags;
  void *ifa_addr;     // (struct sockaddr *)
  void *ifa_netmask;  // (struct sockaddr *)
  void *ifa_dstaddr;  // (struct sockaddr *)
  void *ifa_data;
  unsigned int ifa_addrflags;
};

typedef unsigned __sanitizer_pthread_key_t;

typedef long long __sanitizer_time_t;
typedef int __sanitizer_suseconds_t;

struct __sanitizer_timeval {
  __sanitizer_time_t tv_sec;
  __sanitizer_suseconds_t tv_usec;
};

struct __sanitizer_itimerval {
  struct __sanitizer_timeval it_interval;
  struct __sanitizer_timeval it_value;
};

struct __sanitizer_passwd {
  char *pw_name;
  char *pw_passwd;
  int pw_uid;
  int pw_gid;
  __sanitizer_time_t pw_change;
  char *pw_class;
  char *pw_gecos;
  char *pw_dir;
  char *pw_shell;
  __sanitizer_time_t pw_expire;
};

struct __sanitizer_group {
  char *gr_name;
  char *gr_passwd;
  int gr_gid;
  char **gr_mem;
};

struct __sanitizer_timeb {
  __sanitizer_time_t time;
  unsigned short millitm;
  short timezone;
  short dstflag;
};

struct __sanitizer_ether_addr {
  u8 octet[6];
};

struct __sanitizer_tm {
  int tm_sec;
  int tm_min;
  int tm_hour;
  int tm_mday;
  int tm_mon;
  int tm_year;
  int tm_wday;
  int tm_yday;
  int tm_isdst;
  long int tm_gmtoff;
  const char *tm_zone;
};

struct __sanitizer_msghdr {
  void *msg_name;
  unsigned msg_namelen;
  struct __sanitizer_iovec *msg_iov;
  unsigned msg_iovlen;
  void *msg_control;
  unsigned msg_controllen;
  int msg_flags;
};
struct __sanitizer_cmsghdr {
  unsigned cmsg_len;
  int cmsg_level;
  int cmsg_type;
};

struct __sanitizer_dirent {
  u64 d_fileno;
  u16 d_reclen;
  // more fields that we don't care about
};

typedef int __sanitizer_clock_t;
typedef int __sanitizer_clockid_t;

typedef u32 __sanitizer___kernel_uid_t;
typedef u32 __sanitizer___kernel_gid_t;
typedef u64 __sanitizer___kernel_off_t;
typedef struct {
  u32 fds_bits[8];
} __sanitizer___kernel_fd_set;

typedef struct {
  unsigned int pta_magic;
  int pta_flags;
  void *pta_private;
} __sanitizer_pthread_attr_t;

struct __sanitizer_sigset_t {
  // uint32_t * 4
  unsigned int __bits[4];
};

struct __sanitizer_siginfo {
  // The size is determined by looking at sizeof of real siginfo_t on linux.
  u64 opaque[128 / sizeof(u64)];
};

using __sanitizer_sighandler_ptr = void (*)(int sig);
using __sanitizer_sigactionhandler_ptr = void (*)(int sig,
                                                  __sanitizer_siginfo *siginfo,
                                                  void *uctx);

struct __sanitizer_sigaction {
  union {
    __sanitizer_sighandler_ptr handler;
    __sanitizer_sigactionhandler_ptr sigaction;
  };
  __sanitizer_sigset_t sa_mask;
  int sa_flags;
};

typedef __sanitizer_sigset_t __sanitizer_kernel_sigset_t;

struct __sanitizer_kernel_sigaction_t {
  union {
    void (*handler)(int signo);
    void (*sigaction)(int signo, void *info, void *ctx);
  };
  unsigned long sa_flags;
  void (*sa_restorer)(void);
  __sanitizer_kernel_sigset_t sa_mask;
};

extern const uptr sig_ign;
extern const uptr sig_dfl;
extern const uptr sig_err;
extern const uptr sa_siginfo;

extern int af_inet;
extern int af_inet6;
uptr __sanitizer_in_addr_sz(int af);

struct __sanitizer_dl_phdr_info {
  uptr dlpi_addr;
  const char *dlpi_name;
  const void *dlpi_phdr;
  short dlpi_phnum;
};

extern unsigned struct_ElfW_Phdr_sz;

struct __sanitizer_addrinfo {
  int ai_flags;
  int ai_family;
  int ai_socktype;
  int ai_protocol;
  unsigned ai_addrlen;
  char *ai_canonname;
  void *ai_addr;
  struct __sanitizer_addrinfo *ai_next;
};

struct __sanitizer_hostent {
  char *h_name;
  char **h_aliases;
  int h_addrtype;
  int h_length;
  char **h_addr_list;
};

struct __sanitizer_pollfd {
  int fd;
  short events;
  short revents;
};

typedef unsigned __sanitizer_nfds_t;

struct __sanitizer_glob_t {
  uptr gl_pathc;
  uptr gl_matchc;
  uptr gl_offs;
  int gl_flags;
  char **gl_pathv;
  int (*gl_errfunc)(const char *, int);
  void (*gl_closedir)(void *dirp);
  struct dirent *(*gl_readdir)(void *dirp);
  void *(*gl_opendir)(const char *);
  int (*gl_lstat)(const char *, void * /* struct stat* */);
  int (*gl_stat)(const char *, void * /* struct stat* */);
};

extern int glob_nomatch;
extern int glob_altdirfunc;

extern unsigned path_max;

struct __sanitizer_wordexp_t {
  uptr we_wordc;
  char **we_wordv;
  uptr we_offs;
  char *we_strings;
  uptr we_nbytes;
};

typedef char __sanitizer_FILE;
#define SANITIZER_HAS_STRUCT_FILE 0

extern int shmctl_ipc_stat;

// This simplifies generic code
#define struct_shminfo_sz -1
#define struct_shm_info_sz -1
#define shmctl_shm_stat -1
#define shmctl_ipc_info -1
#define shmctl_shm_info -1

extern unsigned struct_utmp_sz;
extern unsigned struct_utmpx_sz;

extern int map_fixed;

// ioctl arguments
struct __sanitizer_ifconf {
  int ifc_len;
  union {
    void *ifcu_req;
  } ifc_ifcu;
};

#define IOC_NRBITS 8
#define IOC_TYPEBITS 8
#define IOC_SIZEBITS 14
#define IOC_DIRBITS 2
#define IOC_NONE 0U
#define IOC_WRITE 1U
#define IOC_READ 2U
#define IOC_NRMASK ((1 << IOC_NRBITS) - 1)
#define IOC_TYPEMASK ((1 << IOC_TYPEBITS) - 1)
#define IOC_SIZEMASK ((1 << IOC_SIZEBITS) - 1)
#undef IOC_DIRMASK
#define IOC_DIRMASK ((1 << IOC_DIRBITS) - 1)
#define IOC_NRSHIFT 0
#define IOC_TYPESHIFT (IOC_NRSHIFT + IOC_NRBITS)
#define IOC_SIZESHIFT (IOC_TYPESHIFT + IOC_TYPEBITS)
#define IOC_DIRSHIFT (IOC_SIZESHIFT + IOC_SIZEBITS)
#define EVIOC_EV_MAX 0x1f
#define EVIOC_ABS_MAX 0x3f

#define IOC_DIR(nr) (((nr) >> IOC_DIRSHIFT) & IOC_DIRMASK)
#define IOC_TYPE(nr) (((nr) >> IOC_TYPESHIFT) & IOC_TYPEMASK)
#define IOC_NR(nr) (((nr) >> IOC_NRSHIFT) & IOC_NRMASK)
#define IOC_SIZE(nr) (((nr) >> IOC_SIZESHIFT) & IOC_SIZEMASK)

extern unsigned struct_ifreq_sz;
extern unsigned struct_termios_sz;
extern unsigned struct_winsize_sz;

extern unsigned struct_arpreq_sz;

extern unsigned struct_mtget_sz;
extern unsigned struct_mtop_sz;
extern unsigned struct_rtentry_sz;
extern unsigned struct_sbi_instrument_sz;
extern unsigned struct_seq_event_rec_sz;
extern unsigned struct_synth_info_sz;
extern unsigned struct_vt_mode_sz;
extern unsigned struct_audio_buf_info_sz;
extern unsigned struct_ppp_stats_sz;
extern unsigned struct_sioc_sg_req_sz;
extern unsigned struct_sioc_vif_req_sz;

// ioctl request identifiers

// A special value to mark ioctls that are not present on the target platform,
// when it can not be determined without including any system headers.
extern const unsigned IOCTL_NOT_PRESENT;

extern unsigned IOCTL_FIOASYNC;
extern unsigned IOCTL_FIOCLEX;
extern unsigned IOCTL_FIOGETOWN;
extern unsigned IOCTL_FIONBIO;
extern unsigned IOCTL_FIONCLEX;
extern unsigned IOCTL_FIOSETOWN;
extern unsigned IOCTL_SIOCADDMULTI;
extern unsigned IOCTL_SIOCATMARK;
extern unsigned IOCTL_SIOCDELMULTI;
extern unsigned IOCTL_SIOCGIFADDR;
extern unsigned IOCTL_SIOCGIFBRDADDR;
extern unsigned IOCTL_SIOCGIFCONF;
extern unsigned IOCTL_SIOCGIFDSTADDR;
extern unsigned IOCTL_SIOCGIFFLAGS;
extern unsigned IOCTL_SIOCGIFMETRIC;
extern unsigned IOCTL_SIOCGIFMTU;
extern unsigned IOCTL_SIOCGIFNETMASK;
extern unsigned IOCTL_SIOCGPGRP;
extern unsigned IOCTL_SIOCSIFADDR;
extern unsigned IOCTL_SIOCSIFBRDADDR;
extern unsigned IOCTL_SIOCSIFDSTADDR;
extern unsigned IOCTL_SIOCSIFFLAGS;
extern unsigned IOCTL_SIOCSIFMETRIC;
extern unsigned IOCTL_SIOCSIFMTU;
extern unsigned IOCTL_SIOCSIFNETMASK;
extern unsigned IOCTL_SIOCSPGRP;
extern unsigned IOCTL_TIOCCONS;
extern unsigned IOCTL_TIOCEXCL;
extern unsigned IOCTL_TIOCGETD;
extern unsigned IOCTL_TIOCGPGRP;
extern unsigned IOCTL_TIOCGWINSZ;
extern unsigned IOCTL_TIOCMBIC;
extern unsigned IOCTL_TIOCMBIS;
extern unsigned IOCTL_TIOCMGET;
extern unsigned IOCTL_TIOCMSET;
extern unsigned IOCTL_TIOCNOTTY;
extern unsigned IOCTL_TIOCNXCL;
extern unsigned IOCTL_TIOCOUTQ;
extern unsigned IOCTL_TIOCPKT;
extern unsigned IOCTL_TIOCSCTTY;
extern unsigned IOCTL_TIOCSETD;
extern unsigned IOCTL_TIOCSPGRP;
extern unsigned IOCTL_TIOCSTI;
extern unsigned IOCTL_TIOCSWINSZ;
extern unsigned IOCTL_SIOCGETSGCNT;
extern unsigned IOCTL_SIOCGETVIFCNT;
extern unsigned IOCTL_MTIOCGET;
extern unsigned IOCTL_MTIOCTOP;
extern unsigned IOCTL_SIOCADDRT;
extern unsigned IOCTL_SIOCDELRT;
extern unsigned IOCTL_SNDCTL_DSP_GETBLKSIZE;
extern unsigned IOCTL_SNDCTL_DSP_GETFMTS;
extern unsigned IOCTL_SNDCTL_DSP_NONBLOCK;
extern unsigned IOCTL_SNDCTL_DSP_POST;
extern unsigned IOCTL_SNDCTL_DSP_RESET;
extern unsigned IOCTL_SNDCTL_DSP_SETFMT;
extern unsigned IOCTL_SNDCTL_DSP_SETFRAGMENT;
extern unsigned IOCTL_SNDCTL_DSP_SPEED;
extern unsigned IOCTL_SNDCTL_DSP_STEREO;
extern unsigned IOCTL_SNDCTL_DSP_SUBDIVIDE;
extern unsigned IOCTL_SNDCTL_DSP_SYNC;
extern unsigned IOCTL_SNDCTL_FM_4OP_ENABLE;
extern unsigned IOCTL_SNDCTL_FM_LOAD_INSTR;
extern unsigned IOCTL_SNDCTL_MIDI_INFO;
extern unsigned IOCTL_SNDCTL_MIDI_PRETIME;
extern unsigned IOCTL_SNDCTL_SEQ_CTRLRATE;
extern unsigned IOCTL_SNDCTL_SEQ_GETINCOUNT;
extern unsigned IOCTL_SNDCTL_SEQ_GETOUTCOUNT;
extern unsigned IOCTL_SNDCTL_SEQ_NRMIDIS;
extern unsigned IOCTL_SNDCTL_SEQ_NRSYNTHS;
extern unsigned IOCTL_SNDCTL_SEQ_OUTOFBAND;
extern unsigned IOCTL_SNDCTL_SEQ_PANIC;
extern unsigned IOCTL_SNDCTL_SEQ_PERCMODE;
extern unsigned IOCTL_SNDCTL_SEQ_RESET;
extern unsigned IOCTL_SNDCTL_SEQ_RESETSAMPLES;
extern unsigned IOCTL_SNDCTL_SEQ_SYNC;
extern unsigned IOCTL_SNDCTL_SEQ_TESTMIDI;
extern unsigned IOCTL_SNDCTL_SEQ_THRESHOLD;
extern unsigned IOCTL_SNDCTL_SYNTH_INFO;
extern unsigned IOCTL_SNDCTL_SYNTH_MEMAVL;
extern unsigned IOCTL_SNDCTL_TMR_CONTINUE;
extern unsigned IOCTL_SNDCTL_TMR_METRONOME;
extern unsigned IOCTL_SNDCTL_TMR_SELECT;
extern unsigned IOCTL_SNDCTL_TMR_SOURCE;
extern unsigned IOCTL_SNDCTL_TMR_START;
extern unsigned IOCTL_SNDCTL_TMR_STOP;
extern unsigned IOCTL_SNDCTL_TMR_TEMPO;
extern unsigned IOCTL_SNDCTL_TMR_TIMEBASE;
extern unsigned IOCTL_SOUND_MIXER_READ_ALTPCM;
extern unsigned IOCTL_SOUND_MIXER_READ_BASS;
extern unsigned IOCTL_SOUND_MIXER_READ_CAPS;
extern unsigned IOCTL_SOUND_MIXER_READ_CD;
extern unsigned IOCTL_SOUND_MIXER_READ_DEVMASK;
extern unsigned IOCTL_SOUND_MIXER_READ_ENHANCE;
extern unsigned IOCTL_SOUND_MIXER_READ_IGAIN;
extern unsigned IOCTL_SOUND_MIXER_READ_IMIX;
extern unsigned IOCTL_SOUND_MIXER_READ_LINE1;
extern unsigned IOCTL_SOUND_MIXER_READ_LINE2;
extern unsigned IOCTL_SOUND_MIXER_READ_LINE3;
extern unsigned IOCTL_SOUND_MIXER_READ_LINE;
extern unsigned IOCTL_SOUND_MIXER_READ_LOUD;
extern unsigned IOCTL_SOUND_MIXER_READ_MIC;
extern unsigned IOCTL_SOUND_MIXER_READ_MUTE;
extern unsigned IOCTL_SOUND_MIXER_READ_OGAIN;
extern unsigned IOCTL_SOUND_MIXER_READ_PCM;
extern unsigned IOCTL_SOUND_MIXER_READ_RECLEV;
extern unsigned IOCTL_SOUND_MIXER_READ_RECMASK;
extern unsigned IOCTL_SOUND_MIXER_READ_RECSRC;
extern unsigned IOCTL_SOUND_MIXER_READ_SPEAKER;
extern unsigned IOCTL_SOUND_MIXER_READ_STEREODEVS;
extern unsigned IOCTL_SOUND_MIXER_READ_SYNTH;
extern unsigned IOCTL_SOUND_MIXER_READ_TREBLE;
extern unsigned IOCTL_SOUND_MIXER_READ_VOLUME;
extern unsigned IOCTL_SOUND_MIXER_WRITE_ALTPCM;
extern unsigned IOCTL_SOUND_MIXER_WRITE_BASS;
extern unsigned IOCTL_SOUND_MIXER_WRITE_CD;
extern unsigned IOCTL_SOUND_MIXER_WRITE_ENHANCE;
extern unsigned IOCTL_SOUND_MIXER_WRITE_IGAIN;
extern unsigned IOCTL_SOUND_MIXER_WRITE_IMIX;
extern unsigned IOCTL_SOUND_MIXER_WRITE_LINE1;
extern unsigned IOCTL_SOUND_MIXER_WRITE_LINE2;
extern unsigned IOCTL_SOUND_MIXER_WRITE_LINE3;
extern unsigned IOCTL_SOUND_MIXER_WRITE_LINE;
extern unsigned IOCTL_SOUND_MIXER_WRITE_LOUD;
extern unsigned IOCTL_SOUND_MIXER_WRITE_MIC;
extern unsigned IOCTL_SOUND_MIXER_WRITE_MUTE;
extern unsigned IOCTL_SOUND_MIXER_WRITE_OGAIN;
extern unsigned IOCTL_SOUND_MIXER_WRITE_PCM;
extern unsigned IOCTL_SOUND_MIXER_WRITE_RECLEV;
extern unsigned IOCTL_SOUND_MIXER_WRITE_RECSRC;
extern unsigned IOCTL_SOUND_MIXER_WRITE_SPEAKER;
extern unsigned IOCTL_SOUND_MIXER_WRITE_SYNTH;
extern unsigned IOCTL_SOUND_MIXER_WRITE_TREBLE;
extern unsigned IOCTL_SOUND_MIXER_WRITE_VOLUME;
extern unsigned IOCTL_SOUND_PCM_READ_BITS;
extern unsigned IOCTL_SOUND_PCM_READ_CHANNELS;
extern unsigned IOCTL_SOUND_PCM_READ_FILTER;
extern unsigned IOCTL_SOUND_PCM_READ_RATE;
extern unsigned IOCTL_SOUND_PCM_WRITE_CHANNELS;
extern unsigned IOCTL_SOUND_PCM_WRITE_FILTER;
extern unsigned IOCTL_VT_ACTIVATE;
extern unsigned IOCTL_VT_GETMODE;
extern unsigned IOCTL_VT_OPENQRY;
extern unsigned IOCTL_VT_RELDISP;
extern unsigned IOCTL_VT_SETMODE;
extern unsigned IOCTL_VT_WAITACTIVE;
extern unsigned IOCTL_KDDISABIO;
extern unsigned IOCTL_KDENABIO;
extern unsigned IOCTL_KDGETLED;
extern unsigned IOCTL_KDGKBMODE;
extern unsigned IOCTL_KDGKBTYPE;
extern unsigned IOCTL_KDMKTONE;
extern unsigned IOCTL_KDSETLED;
extern unsigned IOCTL_KDSETMODE;
extern unsigned IOCTL_KDSKBMODE;

extern const int si_SEGV_MAPERR;
extern const int si_SEGV_ACCERR;
}  // namespace __sanitizer

#define CHECK_TYPE_SIZE(TYPE) \
  COMPILER_CHECK(sizeof(__sanitizer_##TYPE) == sizeof(TYPE))

#define CHECK_SIZE_AND_OFFSET(CLASS, MEMBER)                      \
  COMPILER_CHECK(sizeof(((__sanitizer_##CLASS *)NULL)->MEMBER) == \
                 sizeof(((CLASS *)NULL)->MEMBER));                \
  COMPILER_CHECK(offsetof(__sanitizer_##CLASS, MEMBER) ==         \
                 offsetof(CLASS, MEMBER))

// For sigaction, which is a function and struct at the same time,
// and thus requires explicit "struct" in sizeof() expression.
#define CHECK_STRUCT_SIZE_AND_OFFSET(CLASS, MEMBER)                      \
  COMPILER_CHECK(sizeof(((struct __sanitizer_##CLASS *)NULL)->MEMBER) == \
                 sizeof(((struct CLASS *)NULL)->MEMBER));                \
  COMPILER_CHECK(offsetof(struct __sanitizer_##CLASS, MEMBER) ==         \
                 offsetof(struct CLASS, MEMBER))

#define SIGACTION_SYMNAME __sigaction14

#endif  // SANITIZER_NETBSD

#endif
