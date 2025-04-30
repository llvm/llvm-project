/* NB: Don't define MINSIGSTKSZ nor SIGSTKSZ to sysconf (SC_SIGSTKSZ) for
   glibc build.  IS_IN can only be used when _ISOMAC isn't defined.  */
#ifdef _ISOMAC
# include_next <bits/sigstksz.h>
#elif IS_IN (libsupport)
# include_next <bits/sigstksz.h>
#endif
