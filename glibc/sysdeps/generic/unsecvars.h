#if !HAVE_TUNABLES
# define GLIBC_TUNABLES_ENVVAR "GLIBC_TUNABLES\0"
#else
# define GLIBC_TUNABLES_ENVVAR
#endif

/* Environment variable to be removed for SUID programs.  The names are
   all stuffed in a single string which means they have to be terminated
   with a '\0' explicitly.  */
#define UNSECURE_ENVVARS \
  "GCONV_PATH\0"							      \
  "GETCONF_DIR\0"							      \
  GLIBC_TUNABLES_ENVVAR							      \
  "HOSTALIASES\0"							      \
  "LD_AUDIT\0"								      \
  "LD_DEBUG\0"								      \
  "LD_DEBUG_OUTPUT\0"							      \
  "LD_DYNAMIC_WEAK\0"							      \
  "LD_HWCAP_MASK\0"							      \
  "LD_LIBRARY_PATH\0"							      \
  "LD_ORIGIN_PATH\0"							      \
  "LD_PRELOAD\0"							      \
  "LD_PROFILE\0"							      \
  "LD_SHOW_AUXV\0"							      \
  "LD_USE_LOAD_BIAS\0"							      \
  "LOCALDOMAIN\0"							      \
  "LOCPATH\0"								      \
  "MALLOC_TRACE\0"							      \
  "NIS_PATH\0"								      \
  "NLSPATH\0"								      \
  "RESOLV_HOST_CONF\0"							      \
  "RES_OPTIONS\0"							      \
  "TMPDIR\0"								      \
  "TZDIR\0"
