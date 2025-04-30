#if IS_IN (ldconfig)
# include <sysdeps/i386/dl-procinfo.c>
#else
# include <sysdeps/x86_64/dl-procinfo.c>
#endif
