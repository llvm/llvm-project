#ifndef LLDB_TEST_ATTACH_H
#define LLDB_TEST_ATTACH_H

// On some systems (e.g., some versions of linux) it is not possible to attach
// to a process without it giving us special permissions. This defines the
// lldb_enable_attach macro, which should perform any such actions, if needed by
// the platform.
#if defined(__linux__)
#include <sys/prctl.h>

// Android API <= 16 does not have these defined.
#ifndef PR_SET_PTRACER
#define PR_SET_PTRACER 0x59616d61
#endif
#ifndef PR_SET_PTRACER_ANY
#define PR_SET_PTRACER_ANY ((unsigned long)-1)
#endif

// For now we execute on best effort basis.  If this fails for some reason, so
// be it.
#define lldb_enable_attach()                                                   \
  do {                                                                         \
    const int prctl_result =                                                   \
        prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0);                    \
    (void)prctl_result;                                                        \
  } while (0)

#else // not linux

#define lldb_enable_attach()

#endif // defined(__linux__)

#endif // LLDB_TEST_ATTACH_H
