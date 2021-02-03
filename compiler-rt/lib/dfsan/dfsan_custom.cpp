//===-- dfsan.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of DataFlowSanitizer.
//
// This file defines the custom functions listed in done_abilist.txt.
//===----------------------------------------------------------------------===//

#include <arpa/inet.h>
#include <assert.h>
#include <ctype.h>
#include <dlfcn.h>
#include <link.h>
#include <poll.h>
#include <pthread.h>
#include <pwd.h>
#include <sched.h>
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/epoll.h>
#include <sys/resource.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "dfsan/dfsan.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_linux.h"

using namespace __dfsan;

#define CALL_WEAK_INTERCEPTOR_HOOK(f, ...)                                     \
  do {                                                                         \
    if (f)                                                                     \
      f(__VA_ARGS__);                                                          \
  } while (false)
#define DECLARE_WEAK_INTERCEPTOR_HOOK(f, ...) \
SANITIZER_INTERFACE_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE void f(__VA_ARGS__);

// Async-safe, non-reentrant spin lock.
class SignalSpinLocker {
 public:
  SignalSpinLocker() {
    sigset_t all_set;
    sigfillset(&all_set);
    pthread_sigmask(SIG_SETMASK, &all_set, &saved_thread_mask_);
    sigactions_mu.Lock();
  }
  ~SignalSpinLocker() {
    sigactions_mu.Unlock();
    pthread_sigmask(SIG_SETMASK, &saved_thread_mask_, nullptr);
  }

 private:
  static StaticSpinMutex sigactions_mu;
  sigset_t saved_thread_mask_;

  SignalSpinLocker(const SignalSpinLocker &) = delete;
  SignalSpinLocker &operator=(const SignalSpinLocker &) = delete;
};

StaticSpinMutex SignalSpinLocker::sigactions_mu;

extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE int
__dfsw_stat(const char *path, struct stat *buf, dfsan_label path_label,
            dfsan_label buf_label, dfsan_label *ret_label) {
  int ret = stat(path, buf);
  if (ret == 0)
    dfsan_set_label(0, buf, sizeof(struct stat));
  *ret_label = 0;
  return ret;
}

SANITIZER_INTERFACE_ATTRIBUTE int __dfsw_fstat(int fd, struct stat *buf,
                                               dfsan_label fd_label,
                                               dfsan_label buf_label,
                                               dfsan_label *ret_label) {
  int ret = fstat(fd, buf);
  if (ret == 0)
    dfsan_set_label(0, buf, sizeof(struct stat));
  *ret_label = 0;
  return ret;
}

SANITIZER_INTERFACE_ATTRIBUTE char *__dfsw_strchr(const char *s, int c,
                                                  dfsan_label s_label,
                                                  dfsan_label c_label,
                                                  dfsan_label *ret_label) {
  for (size_t i = 0;; ++i) {
    if (s[i] == c || s[i] == 0) {
      if (flags().strict_data_dependencies) {
        *ret_label = s_label;
      } else {
        *ret_label = dfsan_union(dfsan_read_label(s, i + 1),
                                 dfsan_union(s_label, c_label));
      }

      // If s[i] is the \0 at the end of the string, and \0 is not the
      // character we are searching for, then return null.
      if (s[i] == 0 && c != 0) {
        return nullptr;
      }
      return const_cast<char *>(s + i);
    }
  }
}

SANITIZER_INTERFACE_ATTRIBUTE char *__dfsw_strpbrk(const char *s,
                                                   const char *accept,
                                                   dfsan_label s_label,
                                                   dfsan_label accept_label,
                                                   dfsan_label *ret_label) {
  const char *ret = strpbrk(s, accept);
  if (flags().strict_data_dependencies) {
    *ret_label = ret ? s_label : 0;
  } else {
    size_t s_bytes_read = (ret ? ret - s : strlen(s)) + 1;
    *ret_label =
        dfsan_union(dfsan_read_label(s, s_bytes_read),
                    dfsan_union(dfsan_read_label(accept, strlen(accept) + 1),
                                dfsan_union(s_label, accept_label)));
  }
  return const_cast<char *>(ret);
}

static int dfsan_memcmp_bcmp(const void *s1, const void *s2, size_t n,
                             dfsan_label s1_label, dfsan_label s2_label,
                             dfsan_label n_label, dfsan_label *ret_label) {
  const char *cs1 = (const char *) s1, *cs2 = (const char *) s2;
  for (size_t i = 0; i != n; ++i) {
    if (cs1[i] != cs2[i]) {
      if (flags().strict_data_dependencies) {
        *ret_label = 0;
      } else {
        *ret_label = dfsan_union(dfsan_read_label(cs1, i + 1),
                                 dfsan_read_label(cs2, i + 1));
      }
      return cs1[i] - cs2[i];
    }
  }

  if (flags().strict_data_dependencies) {
    *ret_label = 0;
  } else {
    *ret_label = dfsan_union(dfsan_read_label(cs1, n),
                             dfsan_read_label(cs2, n));
  }
  return 0;
}

DECLARE_WEAK_INTERCEPTOR_HOOK(dfsan_weak_hook_memcmp, uptr caller_pc,
                              const void *s1, const void *s2, size_t n,
                              dfsan_label s1_label, dfsan_label s2_label,
                              dfsan_label n_label)

SANITIZER_INTERFACE_ATTRIBUTE int __dfsw_memcmp(const void *s1, const void *s2,
                                                size_t n, dfsan_label s1_label,
                                                dfsan_label s2_label,
                                                dfsan_label n_label,
                                                dfsan_label *ret_label) {
  CALL_WEAK_INTERCEPTOR_HOOK(dfsan_weak_hook_memcmp, GET_CALLER_PC(), s1, s2, n,
                             s1_label, s2_label, n_label);
  return dfsan_memcmp_bcmp(s1, s2, n, s1_label, s2_label, n_label, ret_label);
}

SANITIZER_INTERFACE_ATTRIBUTE int __dfsw_bcmp(const void *s1, const void *s2,
                                              size_t n, dfsan_label s1_label,
                                              dfsan_label s2_label,
                                              dfsan_label n_label,
                                              dfsan_label *ret_label) {
  return dfsan_memcmp_bcmp(s1, s2, n, s1_label, s2_label, n_label, ret_label);
}

DECLARE_WEAK_INTERCEPTOR_HOOK(dfsan_weak_hook_strcmp, uptr caller_pc,
                              const char *s1, const char *s2,
                              dfsan_label s1_label, dfsan_label s2_label)

SANITIZER_INTERFACE_ATTRIBUTE int __dfsw_strcmp(const char *s1, const char *s2,
                                                dfsan_label s1_label,
                                                dfsan_label s2_label,
                                                dfsan_label *ret_label) {
  CALL_WEAK_INTERCEPTOR_HOOK(dfsan_weak_hook_strcmp, GET_CALLER_PC(), s1, s2,
                             s1_label, s2_label);
  for (size_t i = 0;; ++i) {
    if (s1[i] != s2[i] || s1[i] == 0 || s2[i] == 0) {
      if (flags().strict_data_dependencies) {
        *ret_label = 0;
      } else {
        *ret_label = dfsan_union(dfsan_read_label(s1, i + 1),
                                 dfsan_read_label(s2, i + 1));
      }
      return s1[i] - s2[i];
    }
  }
  return 0;
}

SANITIZER_INTERFACE_ATTRIBUTE int
__dfsw_strcasecmp(const char *s1, const char *s2, dfsan_label s1_label,
                  dfsan_label s2_label, dfsan_label *ret_label) {
  for (size_t i = 0;; ++i) {
    char s1_lower = tolower(s1[i]);
    char s2_lower = tolower(s2[i]);

    if (s1_lower != s2_lower || s1[i] == 0 || s2[i] == 0) {
      if (flags().strict_data_dependencies) {
        *ret_label = 0;
      } else {
        *ret_label = dfsan_union(dfsan_read_label(s1, i + 1),
                                 dfsan_read_label(s2, i + 1));
      }
      return s1_lower - s2_lower;
    }
  }
  return 0;
}

DECLARE_WEAK_INTERCEPTOR_HOOK(dfsan_weak_hook_strncmp, uptr caller_pc,
                              const char *s1, const char *s2, size_t n,
                              dfsan_label s1_label, dfsan_label s2_label,
                              dfsan_label n_label)

SANITIZER_INTERFACE_ATTRIBUTE int __dfsw_strncmp(const char *s1, const char *s2,
                                                 size_t n, dfsan_label s1_label,
                                                 dfsan_label s2_label,
                                                 dfsan_label n_label,
                                                 dfsan_label *ret_label) {
  if (n == 0) {
    *ret_label = 0;
    return 0;
  }

  CALL_WEAK_INTERCEPTOR_HOOK(dfsan_weak_hook_strncmp, GET_CALLER_PC(), s1, s2,
                             n, s1_label, s2_label, n_label);

  for (size_t i = 0;; ++i) {
    if (s1[i] != s2[i] || s1[i] == 0 || s2[i] == 0 || i == n - 1) {
      if (flags().strict_data_dependencies) {
        *ret_label = 0;
      } else {
        *ret_label = dfsan_union(dfsan_read_label(s1, i + 1),
                                 dfsan_read_label(s2, i + 1));
      }
      return s1[i] - s2[i];
    }
  }
  return 0;
}

SANITIZER_INTERFACE_ATTRIBUTE int
__dfsw_strncasecmp(const char *s1, const char *s2, size_t n,
                   dfsan_label s1_label, dfsan_label s2_label,
                   dfsan_label n_label, dfsan_label *ret_label) {
  if (n == 0) {
    *ret_label = 0;
    return 0;
  }

  for (size_t i = 0;; ++i) {
    char s1_lower = tolower(s1[i]);
    char s2_lower = tolower(s2[i]);

    if (s1_lower != s2_lower || s1[i] == 0 || s2[i] == 0 || i == n - 1) {
      if (flags().strict_data_dependencies) {
        *ret_label = 0;
      } else {
        *ret_label = dfsan_union(dfsan_read_label(s1, i + 1),
                                 dfsan_read_label(s2, i + 1));
      }
      return s1_lower - s2_lower;
    }
  }
  return 0;
}

SANITIZER_INTERFACE_ATTRIBUTE void *__dfsw_calloc(size_t nmemb, size_t size,
                                                  dfsan_label nmemb_label,
                                                  dfsan_label size_label,
                                                  dfsan_label *ret_label) {
  void *p = calloc(nmemb, size);
  dfsan_set_label(0, p, nmemb * size);
  *ret_label = 0;
  return p;
}

SANITIZER_INTERFACE_ATTRIBUTE size_t
__dfsw_strlen(const char *s, dfsan_label s_label, dfsan_label *ret_label) {
  size_t ret = strlen(s);
  if (flags().strict_data_dependencies) {
    *ret_label = 0;
  } else {
    *ret_label = dfsan_read_label(s, ret + 1);
  }
  return ret;
}

static void *dfsan_memmove(void *dest, const void *src, size_t n) {
  dfsan_label *sdest = shadow_for(dest);
  const dfsan_label *ssrc = shadow_for(src);
  internal_memmove((void *)sdest, (const void *)ssrc, n * sizeof(dfsan_label));
  return internal_memmove(dest, src, n);
}

static void *dfsan_memcpy(void *dest, const void *src, size_t n) {
  dfsan_label *sdest = shadow_for(dest);
  const dfsan_label *ssrc = shadow_for(src);
  internal_memcpy((void *)sdest, (const void *)ssrc, n * sizeof(dfsan_label));
  return internal_memcpy(dest, src, n);
}

static void dfsan_memset(void *s, int c, dfsan_label c_label, size_t n) {
  internal_memset(s, c, n);
  dfsan_set_label(c_label, s, n);
}

SANITIZER_INTERFACE_ATTRIBUTE
void *__dfsw_memcpy(void *dest, const void *src, size_t n,
                    dfsan_label dest_label, dfsan_label src_label,
                    dfsan_label n_label, dfsan_label *ret_label) {
  *ret_label = dest_label;
  return dfsan_memcpy(dest, src, n);
}

SANITIZER_INTERFACE_ATTRIBUTE
void *__dfsw_memmove(void *dest, const void *src, size_t n,
                     dfsan_label dest_label, dfsan_label src_label,
                     dfsan_label n_label, dfsan_label *ret_label) {
  *ret_label = dest_label;
  return dfsan_memmove(dest, src, n);
}

SANITIZER_INTERFACE_ATTRIBUTE
void *__dfsw_memset(void *s, int c, size_t n,
                    dfsan_label s_label, dfsan_label c_label,
                    dfsan_label n_label, dfsan_label *ret_label) {
  dfsan_memset(s, c, c_label, n);
  *ret_label = s_label;
  return s;
}

SANITIZER_INTERFACE_ATTRIBUTE char *__dfsw_strcat(char *dest, const char *src,
                                                  dfsan_label dest_label,
                                                  dfsan_label src_label,
                                                  dfsan_label *ret_label) {
  size_t dest_len = strlen(dest);
  char *ret = strcat(dest, src);
  dfsan_label *sdest = shadow_for(dest + dest_len);
  const dfsan_label *ssrc = shadow_for(src);
  internal_memcpy((void *)sdest, (const void *)ssrc,
                  strlen(src) * sizeof(dfsan_label));
  *ret_label = dest_label;
  return ret;
}

SANITIZER_INTERFACE_ATTRIBUTE char *
__dfsw_strdup(const char *s, dfsan_label s_label, dfsan_label *ret_label) {
  size_t len = strlen(s);
  void *p = malloc(len+1);
  dfsan_memcpy(p, s, len+1);
  *ret_label = 0;
  return static_cast<char *>(p);
}

SANITIZER_INTERFACE_ATTRIBUTE char *
__dfsw_strncpy(char *s1, const char *s2, size_t n, dfsan_label s1_label,
               dfsan_label s2_label, dfsan_label n_label,
               dfsan_label *ret_label) {
  size_t len = strlen(s2);
  if (len < n) {
    dfsan_memcpy(s1, s2, len+1);
    dfsan_memset(s1+len+1, 0, 0, n-len-1);
  } else {
    dfsan_memcpy(s1, s2, n);
  }

  *ret_label = s1_label;
  return s1;
}

SANITIZER_INTERFACE_ATTRIBUTE ssize_t
__dfsw_pread(int fd, void *buf, size_t count, off_t offset,
             dfsan_label fd_label, dfsan_label buf_label,
             dfsan_label count_label, dfsan_label offset_label,
             dfsan_label *ret_label) {
  ssize_t ret = pread(fd, buf, count, offset);
  if (ret > 0)
    dfsan_set_label(0, buf, ret);
  *ret_label = 0;
  return ret;
}

SANITIZER_INTERFACE_ATTRIBUTE ssize_t
__dfsw_read(int fd, void *buf, size_t count,
             dfsan_label fd_label, dfsan_label buf_label,
             dfsan_label count_label,
             dfsan_label *ret_label) {
  ssize_t ret = read(fd, buf, count);
  if (ret > 0)
    dfsan_set_label(0, buf, ret);
  *ret_label = 0;
  return ret;
}

SANITIZER_INTERFACE_ATTRIBUTE int __dfsw_clock_gettime(clockid_t clk_id,
                                                       struct timespec *tp,
                                                       dfsan_label clk_id_label,
                                                       dfsan_label tp_label,
                                                       dfsan_label *ret_label) {
  int ret = clock_gettime(clk_id, tp);
  if (ret == 0)
    dfsan_set_label(0, tp, sizeof(struct timespec));
  *ret_label = 0;
  return ret;
}

static void unpoison(const void *ptr, uptr size) {
  dfsan_set_label(0, const_cast<void *>(ptr), size);
}

// dlopen() ultimately calls mmap() down inside the loader, which generally
// doesn't participate in dynamic symbol resolution.  Therefore we won't
// intercept its calls to mmap, and we have to hook it here.
SANITIZER_INTERFACE_ATTRIBUTE void *
__dfsw_dlopen(const char *filename, int flag, dfsan_label filename_label,
              dfsan_label flag_label, dfsan_label *ret_label) {
  void *handle = dlopen(filename, flag);
  link_map *map = GET_LINK_MAP_BY_DLOPEN_HANDLE(handle);
  if (map)
    ForEachMappedRegion(map, unpoison);
  *ret_label = 0;
  return handle;
}

struct pthread_create_info {
  void *(*start_routine_trampoline)(void *, void *, dfsan_label, dfsan_label *);
  void *start_routine;
  void *arg;
};

static void *pthread_create_cb(void *p) {
  pthread_create_info pci(*(pthread_create_info *)p);
  free(p);
  dfsan_label ret_label;
  return pci.start_routine_trampoline(pci.start_routine, pci.arg, 0,
                                      &ret_label);
}

SANITIZER_INTERFACE_ATTRIBUTE int __dfsw_pthread_create(
    pthread_t *thread, const pthread_attr_t *attr,
    void *(*start_routine_trampoline)(void *, void *, dfsan_label,
                                      dfsan_label *),
    void *start_routine, void *arg, dfsan_label thread_label,
    dfsan_label attr_label, dfsan_label start_routine_label,
    dfsan_label arg_label, dfsan_label *ret_label) {
  pthread_create_info *pci =
      (pthread_create_info *)malloc(sizeof(pthread_create_info));
  pci->start_routine_trampoline = start_routine_trampoline;
  pci->start_routine = start_routine;
  pci->arg = arg;
  int rv = pthread_create(thread, attr, pthread_create_cb, (void *)pci);
  if (rv != 0)
    free(pci);
  *ret_label = 0;
  return rv;
}

SANITIZER_INTERFACE_ATTRIBUTE int __dfsw_pthread_join(pthread_t thread,
                                                      void **retval,
                                                      dfsan_label thread_label,
                                                      dfsan_label retval_label,
                                                      dfsan_label *ret_label) {
  int ret = pthread_join(thread, retval);
  if (ret == 0 && retval)
    dfsan_set_label(0, retval, sizeof(*retval));
  *ret_label = 0;
  return ret;
}

struct dl_iterate_phdr_info {
  int (*callback_trampoline)(void *callback, struct dl_phdr_info *info,
                             size_t size, void *data, dfsan_label info_label,
                             dfsan_label size_label, dfsan_label data_label,
                             dfsan_label *ret_label);
  void *callback;
  void *data;
};

int dl_iterate_phdr_cb(struct dl_phdr_info *info, size_t size, void *data) {
  dl_iterate_phdr_info *dipi = (dl_iterate_phdr_info *)data;
  dfsan_set_label(0, *info);
  dfsan_set_label(0, const_cast<char *>(info->dlpi_name),
                  strlen(info->dlpi_name) + 1);
  dfsan_set_label(
      0, const_cast<char *>(reinterpret_cast<const char *>(info->dlpi_phdr)),
      sizeof(*info->dlpi_phdr) * info->dlpi_phnum);
  dfsan_label ret_label;
  return dipi->callback_trampoline(dipi->callback, info, size, dipi->data, 0, 0,
                                   0, &ret_label);
}

SANITIZER_INTERFACE_ATTRIBUTE int __dfsw_dl_iterate_phdr(
    int (*callback_trampoline)(void *callback, struct dl_phdr_info *info,
                               size_t size, void *data, dfsan_label info_label,
                               dfsan_label size_label, dfsan_label data_label,
                               dfsan_label *ret_label),
    void *callback, void *data, dfsan_label callback_label,
    dfsan_label data_label, dfsan_label *ret_label) {
  dl_iterate_phdr_info dipi = { callback_trampoline, callback, data };
  *ret_label = 0;
  return dl_iterate_phdr(dl_iterate_phdr_cb, &dipi);
}

// This function is only available for glibc 2.27 or newer.  Mark it weak so
// linking succeeds with older glibcs.
SANITIZER_WEAK_ATTRIBUTE void _dl_get_tls_static_info(size_t *sizep,
                                                      size_t *alignp);

SANITIZER_INTERFACE_ATTRIBUTE void __dfsw__dl_get_tls_static_info(
    size_t *sizep, size_t *alignp, dfsan_label sizep_label,
    dfsan_label alignp_label) {
  assert(_dl_get_tls_static_info);
  _dl_get_tls_static_info(sizep, alignp);
  dfsan_set_label(0, sizep, sizeof(*sizep));
  dfsan_set_label(0, alignp, sizeof(*alignp));
}

SANITIZER_INTERFACE_ATTRIBUTE
char *__dfsw_ctime_r(const time_t *timep, char *buf, dfsan_label timep_label,
                     dfsan_label buf_label, dfsan_label *ret_label) {
  char *ret = ctime_r(timep, buf);
  if (ret) {
    dfsan_set_label(dfsan_read_label(timep, sizeof(time_t)), buf,
                    strlen(buf) + 1);
    *ret_label = buf_label;
  } else {
    *ret_label = 0;
  }
  return ret;
}

SANITIZER_INTERFACE_ATTRIBUTE
char *__dfsw_fgets(char *s, int size, FILE *stream, dfsan_label s_label,
                   dfsan_label size_label, dfsan_label stream_label,
                   dfsan_label *ret_label) {
  char *ret = fgets(s, size, stream);
  if (ret) {
    dfsan_set_label(0, ret, strlen(ret) + 1);
    *ret_label = s_label;
  } else {
    *ret_label = 0;
  }
  return ret;
}

SANITIZER_INTERFACE_ATTRIBUTE
char *__dfsw_getcwd(char *buf, size_t size, dfsan_label buf_label,
                    dfsan_label size_label, dfsan_label *ret_label) {
  char *ret = getcwd(buf, size);
  if (ret) {
    dfsan_set_label(0, ret, strlen(ret) + 1);
    *ret_label = buf_label;
  } else {
    *ret_label = 0;
  }
  return ret;
}

SANITIZER_INTERFACE_ATTRIBUTE
char *__dfsw_get_current_dir_name(dfsan_label *ret_label) {
  char *ret = get_current_dir_name();
  if (ret) {
    dfsan_set_label(0, ret, strlen(ret) + 1);
  }
  *ret_label = 0;
  return ret;
}

SANITIZER_INTERFACE_ATTRIBUTE
int __dfsw_gethostname(char *name, size_t len, dfsan_label name_label,
                       dfsan_label len_label, dfsan_label *ret_label) {
  int ret = gethostname(name, len);
  if (ret == 0) {
    dfsan_set_label(0, name, strlen(name) + 1);
  }
  *ret_label = 0;
  return ret;
}

SANITIZER_INTERFACE_ATTRIBUTE
int __dfsw_getrlimit(int resource, struct rlimit *rlim,
                     dfsan_label resource_label, dfsan_label rlim_label,
                     dfsan_label *ret_label) {
  int ret = getrlimit(resource, rlim);
  if (ret == 0) {
    dfsan_set_label(0, rlim, sizeof(struct rlimit));
  }
  *ret_label = 0;
  return ret;
}

SANITIZER_INTERFACE_ATTRIBUTE
int __dfsw_getrusage(int who, struct rusage *usage, dfsan_label who_label,
                     dfsan_label usage_label, dfsan_label *ret_label) {
  int ret = getrusage(who, usage);
  if (ret == 0) {
    dfsan_set_label(0, usage, sizeof(struct rusage));
  }
  *ret_label = 0;
  return ret;
}

SANITIZER_INTERFACE_ATTRIBUTE
char *__dfsw_strcpy(char *dest, const char *src, dfsan_label dst_label,
                    dfsan_label src_label, dfsan_label *ret_label) {
  char *ret = strcpy(dest, src);  // NOLINT
  if (ret) {
    internal_memcpy(shadow_for(dest), shadow_for(src),
                    sizeof(dfsan_label) * (strlen(src) + 1));
  }
  *ret_label = dst_label;
  return ret;
}

SANITIZER_INTERFACE_ATTRIBUTE
long int __dfsw_strtol(const char *nptr, char **endptr, int base,
                       dfsan_label nptr_label, dfsan_label endptr_label,
                       dfsan_label base_label, dfsan_label *ret_label) {
  char *tmp_endptr;
  long int ret = strtol(nptr, &tmp_endptr, base);
  if (endptr) {
    *endptr = tmp_endptr;
  }
  if (tmp_endptr > nptr) {
    // If *tmp_endptr is '\0' include its label as well.
    *ret_label = dfsan_union(
        base_label,
        dfsan_read_label(nptr, tmp_endptr - nptr + (*tmp_endptr ? 0 : 1)));
  } else {
    *ret_label = 0;
  }
  return ret;
}

SANITIZER_INTERFACE_ATTRIBUTE
double __dfsw_strtod(const char *nptr, char **endptr,
                       dfsan_label nptr_label, dfsan_label endptr_label,
                       dfsan_label *ret_label) {
  char *tmp_endptr;
  double ret = strtod(nptr, &tmp_endptr);
  if (endptr) {
    *endptr = tmp_endptr;
  }
  if (tmp_endptr > nptr) {
    // If *tmp_endptr is '\0' include its label as well.
    *ret_label = dfsan_read_label(
        nptr,
        tmp_endptr - nptr + (*tmp_endptr ? 0 : 1));
  } else {
    *ret_label = 0;
  }
  return ret;
}

SANITIZER_INTERFACE_ATTRIBUTE
long long int __dfsw_strtoll(const char *nptr, char **endptr, int base,
                       dfsan_label nptr_label, dfsan_label endptr_label,
                       dfsan_label base_label, dfsan_label *ret_label) {
  char *tmp_endptr;
  long long int ret = strtoll(nptr, &tmp_endptr, base);
  if (endptr) {
    *endptr = tmp_endptr;
  }
  if (tmp_endptr > nptr) {
    // If *tmp_endptr is '\0' include its label as well.
    *ret_label = dfsan_union(
        base_label,
        dfsan_read_label(nptr, tmp_endptr - nptr + (*tmp_endptr ? 0 : 1)));
  } else {
    *ret_label = 0;
  }
  return ret;
}

SANITIZER_INTERFACE_ATTRIBUTE
unsigned long int __dfsw_strtoul(const char *nptr, char **endptr, int base,
                       dfsan_label nptr_label, dfsan_label endptr_label,
                       dfsan_label base_label, dfsan_label *ret_label) {
  char *tmp_endptr;
  unsigned long int ret = strtoul(nptr, &tmp_endptr, base);
  if (endptr) {
    *endptr = tmp_endptr;
  }
  if (tmp_endptr > nptr) {
    // If *tmp_endptr is '\0' include its label as well.
    *ret_label = dfsan_union(
        base_label,
        dfsan_read_label(nptr, tmp_endptr - nptr + (*tmp_endptr ? 0 : 1)));
  } else {
    *ret_label = 0;
  }
  return ret;
}

SANITIZER_INTERFACE_ATTRIBUTE
long long unsigned int __dfsw_strtoull(const char *nptr, char **endptr,
                                       int base, dfsan_label nptr_label,
                                       dfsan_label endptr_label,
                                       dfsan_label base_label,
                                       dfsan_label *ret_label) {
  char *tmp_endptr;
  long long unsigned int ret = strtoull(nptr, &tmp_endptr, base);
  if (endptr) {
    *endptr = tmp_endptr;
  }
  if (tmp_endptr > nptr) {
    // If *tmp_endptr is '\0' include its label as well.
    *ret_label = dfsan_union(
        base_label,
        dfsan_read_label(nptr, tmp_endptr - nptr + (*tmp_endptr ? 0 : 1)));
  } else {
    *ret_label = 0;
  }
  return ret;
}

SANITIZER_INTERFACE_ATTRIBUTE
time_t __dfsw_time(time_t *t, dfsan_label t_label, dfsan_label *ret_label) {
  time_t ret = time(t);
  if (ret != (time_t) -1 && t) {
    dfsan_set_label(0, t, sizeof(time_t));
  }
  *ret_label = 0;
  return ret;
}

SANITIZER_INTERFACE_ATTRIBUTE
int __dfsw_inet_pton(int af, const char *src, void *dst, dfsan_label af_label,
                     dfsan_label src_label, dfsan_label dst_label,
                     dfsan_label *ret_label) {
  int ret = inet_pton(af, src, dst);
  if (ret == 1) {
    dfsan_set_label(dfsan_read_label(src, strlen(src) + 1), dst,
                    af == AF_INET ? sizeof(struct in_addr) : sizeof(in6_addr));
  }
  *ret_label = 0;
  return ret;
}

SANITIZER_INTERFACE_ATTRIBUTE
struct tm *__dfsw_localtime_r(const time_t *timep, struct tm *result,
                              dfsan_label timep_label, dfsan_label result_label,
                              dfsan_label *ret_label) {
  struct tm *ret = localtime_r(timep, result);
  if (ret) {
    dfsan_set_label(dfsan_read_label(timep, sizeof(time_t)), result,
                    sizeof(struct tm));
    *ret_label = result_label;
  } else {
    *ret_label = 0;
  }
  return ret;
}

SANITIZER_INTERFACE_ATTRIBUTE
int __dfsw_getpwuid_r(id_t uid, struct passwd *pwd,
                      char *buf, size_t buflen, struct passwd **result,
                      dfsan_label uid_label, dfsan_label pwd_label,
                      dfsan_label buf_label, dfsan_label buflen_label,
                      dfsan_label result_label, dfsan_label *ret_label) {
  // Store the data in pwd, the strings referenced from pwd in buf, and the
  // address of pwd in *result.  On failure, NULL is stored in *result.
  int ret = getpwuid_r(uid, pwd, buf, buflen, result);
  if (ret == 0) {
    dfsan_set_label(0, pwd, sizeof(struct passwd));
    dfsan_set_label(0, buf, strlen(buf) + 1);
  }
  *ret_label = 0;
  dfsan_set_label(0, result, sizeof(struct passwd*));
  return ret;
}

SANITIZER_INTERFACE_ATTRIBUTE
int __dfsw_epoll_wait(int epfd, struct epoll_event *events, int maxevents,
                      int timeout, dfsan_label epfd_label,
                      dfsan_label events_label, dfsan_label maxevents_label,
                      dfsan_label timeout_label, dfsan_label *ret_label) {
  int ret = epoll_wait(epfd, events, maxevents, timeout);
  if (ret > 0)
    dfsan_set_label(0, events, ret * sizeof(*events));
  *ret_label = 0;
  return ret;
}

SANITIZER_INTERFACE_ATTRIBUTE
int __dfsw_poll(struct pollfd *fds, nfds_t nfds, int timeout,
                dfsan_label dfs_label, dfsan_label nfds_label,
                dfsan_label timeout_label, dfsan_label *ret_label) {
  int ret = poll(fds, nfds, timeout);
  if (ret >= 0) {
    for (; nfds > 0; --nfds) {
      dfsan_set_label(0, &fds[nfds - 1].revents, sizeof(fds[nfds - 1].revents));
    }
  }
  *ret_label = 0;
  return ret;
}

SANITIZER_INTERFACE_ATTRIBUTE
int __dfsw_select(int nfds, fd_set *readfds, fd_set *writefds,
                  fd_set *exceptfds, struct timeval *timeout,
                  dfsan_label nfds_label, dfsan_label readfds_label,
                  dfsan_label writefds_label, dfsan_label exceptfds_label,
                  dfsan_label timeout_label, dfsan_label *ret_label) {
  int ret = select(nfds, readfds, writefds, exceptfds, timeout);
  // Clear everything (also on error) since their content is either set or
  // undefined.
  if (readfds) {
    dfsan_set_label(0, readfds, sizeof(fd_set));
  }
  if (writefds) {
    dfsan_set_label(0, writefds, sizeof(fd_set));
  }
  if (exceptfds) {
    dfsan_set_label(0, exceptfds, sizeof(fd_set));
  }
  dfsan_set_label(0, timeout, sizeof(struct timeval));
  *ret_label = 0;
  return ret;
}

SANITIZER_INTERFACE_ATTRIBUTE
int __dfsw_sched_getaffinity(pid_t pid, size_t cpusetsize, cpu_set_t *mask,
                             dfsan_label pid_label,
                             dfsan_label cpusetsize_label,
                             dfsan_label mask_label, dfsan_label *ret_label) {
  int ret = sched_getaffinity(pid, cpusetsize, mask);
  if (ret == 0) {
    dfsan_set_label(0, mask, cpusetsize);
  }
  *ret_label = 0;
  return ret;
}

SANITIZER_INTERFACE_ATTRIBUTE
int __dfsw_sigemptyset(sigset_t *set, dfsan_label set_label,
                       dfsan_label *ret_label) {
  int ret = sigemptyset(set);
  dfsan_set_label(0, set, sizeof(sigset_t));
  return ret;
}

// Clear DFSan runtime TLS state at the end of a scope.
//
// Implementation must be async-signal-safe and use small data size, because
// instances of this class may live on the signal handler stack.
//
// DFSan uses TLS to pass metadata of arguments and return values. When an
// instrumented function accesses the TLS, if a signal callback happens, and the
// callback calls other instrumented functions with updating the same TLS, the
// TLS is in an inconsistent state after the callback ends. This may cause
// either under-tainting or over-tainting.
//
// The current implementation simply resets TLS at restore. This prevents from
// over-tainting. Although under-tainting may still happen, a taint flow can be
// found eventually if we run a DFSan-instrumented program multiple times. The
// alternative option is saving the entire TLS. However the TLS storage takes
// 2k bytes, and signal calls could be nested. So it does not seem worth.
class ScopedClearThreadLocalState {
 public:
  ScopedClearThreadLocalState() {}
  ~ScopedClearThreadLocalState() { dfsan_clear_thread_local_state(); }
};

// SignalSpinLocker::sigactions_mu guarantees atomicity of sigaction() calls.
const int kMaxSignals = 1024;
static atomic_uintptr_t sigactions[kMaxSignals];

static void SignalHandler(int signo) {
  ScopedClearThreadLocalState stlsb;

  // Clear shadows for all inputs provided by system. This is why DFSan
  // instrumentation generates a trampoline function to each function pointer,
  // and uses the trampoline to clear shadows. However sigaction does not use
  // a function pointer directly, so we have to do this manually.
  dfsan_clear_arg_tls(0, sizeof(dfsan_label));

  typedef void (*signal_cb)(int x);
  signal_cb cb =
      (signal_cb)atomic_load(&sigactions[signo], memory_order_relaxed);
  cb(signo);
}

static void SignalAction(int signo, siginfo_t *si, void *uc) {
  ScopedClearThreadLocalState stlsb;

  // Clear shadows for all inputs provided by system. Similar to SignalHandler.
  dfsan_clear_arg_tls(0, 3 * sizeof(dfsan_label));
  dfsan_set_label(0, si, sizeof(*si));
  dfsan_set_label(0, uc, sizeof(ucontext_t));

  typedef void (*sigaction_cb)(int, siginfo_t *, void *);
  sigaction_cb cb =
      (sigaction_cb)atomic_load(&sigactions[signo], memory_order_relaxed);
  cb(signo, si, uc);
}

SANITIZER_INTERFACE_ATTRIBUTE
int __dfsw_sigaction(int signum, const struct sigaction *act,
                     struct sigaction *oldact, dfsan_label signum_label,
                     dfsan_label act_label, dfsan_label oldact_label,
                     dfsan_label *ret_label) {
  CHECK_LT(signum, kMaxSignals);
  SignalSpinLocker lock;
  uptr old_cb = atomic_load(&sigactions[signum], memory_order_relaxed);
  struct sigaction new_act;
  struct sigaction *pnew_act = act ? &new_act : nullptr;
  if (act) {
    internal_memcpy(pnew_act, act, sizeof(struct sigaction));
    if (pnew_act->sa_flags & SA_SIGINFO) {
      uptr cb = (uptr)(pnew_act->sa_sigaction);
      if (cb != (uptr)SIG_IGN && cb != (uptr)SIG_DFL) {
        atomic_store(&sigactions[signum], cb, memory_order_relaxed);
        pnew_act->sa_sigaction = SignalAction;
      }
    } else {
      uptr cb = (uptr)(pnew_act->sa_handler);
      if (cb != (uptr)SIG_IGN && cb != (uptr)SIG_DFL) {
        atomic_store(&sigactions[signum], cb, memory_order_relaxed);
        pnew_act->sa_handler = SignalHandler;
      }
    }
  }

  int ret = sigaction(signum, pnew_act, oldact);

  if (ret == 0 && oldact) {
    if (oldact->sa_flags & SA_SIGINFO) {
      if (oldact->sa_sigaction == SignalAction)
        oldact->sa_sigaction = (decltype(oldact->sa_sigaction))old_cb;
    } else {
      if (oldact->sa_handler == SignalHandler)
        oldact->sa_handler = (decltype(oldact->sa_handler))old_cb;
    }
  }

  if (oldact) {
    dfsan_set_label(0, oldact, sizeof(struct sigaction));
  }
  *ret_label = 0;
  return ret;
}

SANITIZER_INTERFACE_ATTRIBUTE
sighandler_t __dfsw_signal(int signum,
                           void *(*handler_trampoline)(void *, int, dfsan_label,
                                                       dfsan_label *),
                           sighandler_t handler, dfsan_label signum_label,
                           dfsan_label handler_label, dfsan_label *ret_label) {
  CHECK_LT(signum, kMaxSignals);
  SignalSpinLocker lock;
  uptr old_cb = atomic_load(&sigactions[signum], memory_order_relaxed);
  if (handler != SIG_IGN && handler != SIG_DFL) {
    atomic_store(&sigactions[signum], (uptr)handler, memory_order_relaxed);
    handler = &SignalHandler;
  }

  sighandler_t ret = signal(signum, handler);

  if (ret == SignalHandler)
    ret = (sighandler_t)old_cb;

  *ret_label = 0;
  return ret;
}

SANITIZER_INTERFACE_ATTRIBUTE
int __dfsw_sigaltstack(const stack_t *ss, stack_t *old_ss, dfsan_label ss_label,
                       dfsan_label old_ss_label, dfsan_label *ret_label) {
  int ret = sigaltstack(ss, old_ss);
  if (ret != -1 && old_ss)
    dfsan_set_label(0, old_ss, sizeof(*old_ss));
  *ret_label = 0;
  return ret;
}

SANITIZER_INTERFACE_ATTRIBUTE
int __dfsw_gettimeofday(struct timeval *tv, struct timezone *tz,
                        dfsan_label tv_label, dfsan_label tz_label,
                        dfsan_label *ret_label) {
  int ret = gettimeofday(tv, tz);
  if (tv) {
    dfsan_set_label(0, tv, sizeof(struct timeval));
  }
  if (tz) {
    dfsan_set_label(0, tz, sizeof(struct timezone));
  }
  *ret_label = 0;
  return ret;
}

SANITIZER_INTERFACE_ATTRIBUTE void *__dfsw_memchr(void *s, int c, size_t n,
                                                  dfsan_label s_label,
                                                  dfsan_label c_label,
                                                  dfsan_label n_label,
                                                  dfsan_label *ret_label) {
  void *ret = memchr(s, c, n);
  if (flags().strict_data_dependencies) {
    *ret_label = ret ? s_label : 0;
  } else {
    size_t len =
        ret ? reinterpret_cast<char *>(ret) - reinterpret_cast<char *>(s) + 1
            : n;
    *ret_label =
        dfsan_union(dfsan_read_label(s, len), dfsan_union(s_label, c_label));
  }
  return ret;
}

SANITIZER_INTERFACE_ATTRIBUTE char *__dfsw_strrchr(char *s, int c,
                                                   dfsan_label s_label,
                                                   dfsan_label c_label,
                                                   dfsan_label *ret_label) {
  char *ret = strrchr(s, c);
  if (flags().strict_data_dependencies) {
    *ret_label = ret ? s_label : 0;
  } else {
    *ret_label =
        dfsan_union(dfsan_read_label(s, strlen(s) + 1),
                    dfsan_union(s_label, c_label));
  }

  return ret;
}

SANITIZER_INTERFACE_ATTRIBUTE char *__dfsw_strstr(char *haystack, char *needle,
                                                  dfsan_label haystack_label,
                                                  dfsan_label needle_label,
                                                  dfsan_label *ret_label) {
  char *ret = strstr(haystack, needle);
  if (flags().strict_data_dependencies) {
    *ret_label = ret ? haystack_label : 0;
  } else {
    size_t len = ret ? ret + strlen(needle) - haystack : strlen(haystack) + 1;
    *ret_label =
        dfsan_union(dfsan_read_label(haystack, len),
                    dfsan_union(dfsan_read_label(needle, strlen(needle) + 1),
                                dfsan_union(haystack_label, needle_label)));
  }

  return ret;
}

SANITIZER_INTERFACE_ATTRIBUTE int __dfsw_nanosleep(const struct timespec *req,
                                                   struct timespec *rem,
                                                   dfsan_label req_label,
                                                   dfsan_label rem_label,
                                                   dfsan_label *ret_label) {
  int ret = nanosleep(req, rem);
  *ret_label = 0;
  if (ret == -1) {
    // Interrupted by a signal, rem is filled with the remaining time.
    dfsan_set_label(0, rem, sizeof(struct timespec));
  }
  return ret;
}

static void clear_msghdr_labels(size_t bytes_written, struct msghdr *msg) {
  dfsan_set_label(0, msg, sizeof(*msg));
  dfsan_set_label(0, msg->msg_name, msg->msg_namelen);
  dfsan_set_label(0, msg->msg_control, msg->msg_controllen);
  for (size_t i = 0; bytes_written > 0; ++i) {
    assert(i < msg->msg_iovlen);
    struct iovec *iov = &msg->msg_iov[i];
    size_t iov_written =
        bytes_written < iov->iov_len ? bytes_written : iov->iov_len;
    dfsan_set_label(0, iov->iov_base, iov_written);
    bytes_written -= iov_written;
  }
}

SANITIZER_INTERFACE_ATTRIBUTE int __dfsw_recvmmsg(
    int sockfd, struct mmsghdr *msgvec, unsigned int vlen, int flags,
    struct timespec *timeout, dfsan_label sockfd_label,
    dfsan_label msgvec_label, dfsan_label vlen_label, dfsan_label flags_label,
    dfsan_label timeout_label, dfsan_label *ret_label) {
  int ret = recvmmsg(sockfd, msgvec, vlen, flags, timeout);
  for (int i = 0; i < ret; ++i) {
    dfsan_set_label(0, &msgvec[i].msg_len, sizeof(msgvec[i].msg_len));
    clear_msghdr_labels(msgvec[i].msg_len, &msgvec[i].msg_hdr);
  }
  *ret_label = 0;
  return ret;
}

SANITIZER_INTERFACE_ATTRIBUTE ssize_t __dfsw_recvmsg(
    int sockfd, struct msghdr *msg, int flags, dfsan_label sockfd_label,
    dfsan_label msg_label, dfsan_label flags_label, dfsan_label *ret_label) {
  ssize_t ret = recvmsg(sockfd, msg, flags);
  if (ret >= 0)
    clear_msghdr_labels(ret, msg);
  *ret_label = 0;
  return ret;
}

SANITIZER_INTERFACE_ATTRIBUTE int
__dfsw_socketpair(int domain, int type, int protocol, int sv[2],
                  dfsan_label domain_label, dfsan_label type_label,
                  dfsan_label protocol_label, dfsan_label sv_label,
                  dfsan_label *ret_label) {
  int ret = socketpair(domain, type, protocol, sv);
  *ret_label = 0;
  if (ret == 0) {
    dfsan_set_label(0, sv, sizeof(*sv) * 2);
  }
  return ret;
}

SANITIZER_INTERFACE_ATTRIBUTE int __dfsw_getsockopt(
    int sockfd, int level, int optname, void *optval, socklen_t *optlen,
    dfsan_label sockfd_label, dfsan_label level_label,
    dfsan_label optname_label, dfsan_label optval_label,
    dfsan_label optlen_label, dfsan_label *ret_label) {
  int ret = getsockopt(sockfd, level, optname, optval, optlen);
  if (ret != -1 && optval && optlen) {
    dfsan_set_label(0, optlen, sizeof(*optlen));
    dfsan_set_label(0, optval, *optlen);
  }
  *ret_label = 0;
  return ret;
}

SANITIZER_INTERFACE_ATTRIBUTE int __dfsw_getsockname(
    int sockfd, struct sockaddr *addr, socklen_t *addrlen,
    dfsan_label sockfd_label, dfsan_label addr_label, dfsan_label addrlen_label,
    dfsan_label *ret_label) {
  socklen_t origlen = addrlen ? *addrlen : 0;
  int ret = getsockname(sockfd, addr, addrlen);
  if (ret != -1 && addr && addrlen) {
    socklen_t written_bytes = origlen < *addrlen ? origlen : *addrlen;
    dfsan_set_label(0, addrlen, sizeof(*addrlen));
    dfsan_set_label(0, addr, written_bytes);
  }
  *ret_label = 0;
  return ret;
}

SANITIZER_INTERFACE_ATTRIBUTE int __dfsw_getpeername(
    int sockfd, struct sockaddr *addr, socklen_t *addrlen,
    dfsan_label sockfd_label, dfsan_label addr_label, dfsan_label addrlen_label,
    dfsan_label *ret_label) {
  socklen_t origlen = addrlen ? *addrlen : 0;
  int ret = getpeername(sockfd, addr, addrlen);
  if (ret != -1 && addr && addrlen) {
    socklen_t written_bytes = origlen < *addrlen ? origlen : *addrlen;
    dfsan_set_label(0, addrlen, sizeof(*addrlen));
    dfsan_set_label(0, addr, written_bytes);
  }
  *ret_label = 0;
  return ret;
}

// Type of the trampoline function passed to the custom version of
// dfsan_set_write_callback.
typedef void (*write_trampoline_t)(
    void *callback,
    int fd, const void *buf, ssize_t count,
    dfsan_label fd_label, dfsan_label buf_label, dfsan_label count_label);

// Calls to dfsan_set_write_callback() set the values in this struct.
// Calls to the custom version of write() read (and invoke) them.
static struct {
  write_trampoline_t write_callback_trampoline = nullptr;
  void *write_callback = nullptr;
} write_callback_info;

SANITIZER_INTERFACE_ATTRIBUTE void
__dfsw_dfsan_set_write_callback(
    write_trampoline_t write_callback_trampoline,
    void *write_callback,
    dfsan_label write_callback_label,
    dfsan_label *ret_label) {
  write_callback_info.write_callback_trampoline = write_callback_trampoline;
  write_callback_info.write_callback = write_callback;
}

SANITIZER_INTERFACE_ATTRIBUTE int
__dfsw_write(int fd, const void *buf, size_t count,
             dfsan_label fd_label, dfsan_label buf_label,
             dfsan_label count_label, dfsan_label *ret_label) {
  if (write_callback_info.write_callback) {
    write_callback_info.write_callback_trampoline(
        write_callback_info.write_callback,
        fd, buf, count,
        fd_label, buf_label, count_label);
  }

  *ret_label = 0;
  return write(fd, buf, count);
}
} // namespace __dfsan

// Type used to extract a dfsan_label with va_arg()
typedef int dfsan_label_va;

// Formats a chunk either a constant string or a single format directive (e.g.,
// '%.3f').
struct Formatter {
  Formatter(char *str_, const char *fmt_, size_t size_)
      : str(str_), str_off(0), size(size_), fmt_start(fmt_), fmt_cur(fmt_),
        width(-1) {}

  int format() {
    char *tmp_fmt = build_format_string();
    int retval =
        snprintf(str + str_off, str_off < size ? size - str_off : 0, tmp_fmt,
                 0 /* used only to avoid warnings */);
    free(tmp_fmt);
    return retval;
  }

  template <typename T> int format(T arg) {
    char *tmp_fmt = build_format_string();
    int retval;
    if (width >= 0) {
      retval = snprintf(str + str_off, str_off < size ? size - str_off : 0,
                        tmp_fmt, width, arg);
    } else {
      retval = snprintf(str + str_off, str_off < size ? size - str_off : 0,
                        tmp_fmt, arg);
    }
    free(tmp_fmt);
    return retval;
  }

  char *build_format_string() {
    size_t fmt_size = fmt_cur - fmt_start + 1;
    char *new_fmt = (char *)malloc(fmt_size + 1);
    assert(new_fmt);
    internal_memcpy(new_fmt, fmt_start, fmt_size);
    new_fmt[fmt_size] = '\0';
    return new_fmt;
  }

  char *str_cur() { return str + str_off; }

  size_t num_written_bytes(int retval) {
    if (retval < 0) {
      return 0;
    }

    size_t num_avail = str_off < size ? size - str_off : 0;
    if (num_avail == 0) {
      return 0;
    }

    size_t num_written = retval;
    // A return value of {v,}snprintf of size or more means that the output was
    // truncated.
    if (num_written >= num_avail) {
      num_written -= num_avail;
    }

    return num_written;
  }

  char *str;
  size_t str_off;
  size_t size;
  const char *fmt_start;
  const char *fmt_cur;
  int width;
};

// Formats the input and propagates the input labels to the output. The output
// is stored in 'str'. 'size' bounds the number of output bytes. 'format' and
// 'ap' are the format string and the list of arguments for formatting. Returns
// the return value vsnprintf would return.
//
// The function tokenizes the format string in chunks representing either a
// constant string or a single format directive (e.g., '%.3f') and formats each
// chunk independently into the output string. This approach allows to figure
// out which bytes of the output string depends on which argument and thus to
// propagate labels more precisely.
//
// WARNING: This implementation does not support conversion specifiers with
// positional arguments.
static int format_buffer(char *str, size_t size, const char *fmt,
                         dfsan_label *va_labels, dfsan_label *ret_label,
                         va_list ap) {
  Formatter formatter(str, fmt, size);

  while (*formatter.fmt_cur) {
    formatter.fmt_start = formatter.fmt_cur;
    formatter.width = -1;
    int retval = 0;

    if (*formatter.fmt_cur != '%') {
      // Ordinary character. Consume all the characters until a '%' or the end
      // of the string.
      for (; *(formatter.fmt_cur + 1) && *(formatter.fmt_cur + 1) != '%';
           ++formatter.fmt_cur) {}
      retval = formatter.format();
      dfsan_set_label(0, formatter.str_cur(),
                      formatter.num_written_bytes(retval));
    } else {
      // Conversion directive. Consume all the characters until a conversion
      // specifier or the end of the string.
      bool end_fmt = false;
      for (; *formatter.fmt_cur && !end_fmt; ) {
        switch (*++formatter.fmt_cur) {
        case 'd':
        case 'i':
        case 'o':
        case 'u':
        case 'x':
        case 'X':
          switch (*(formatter.fmt_cur - 1)) {
          case 'h':
            // Also covers the 'hh' case (since the size of the arg is still
            // an int).
            retval = formatter.format(va_arg(ap, int));
            break;
          case 'l':
            if (formatter.fmt_cur - formatter.fmt_start >= 2 &&
                *(formatter.fmt_cur - 2) == 'l') {
              retval = formatter.format(va_arg(ap, long long int));
            } else {
              retval = formatter.format(va_arg(ap, long int));
            }
            break;
          case 'q':
            retval = formatter.format(va_arg(ap, long long int));
            break;
          case 'j':
            retval = formatter.format(va_arg(ap, intmax_t));
            break;
          case 'z':
          case 't':
            retval = formatter.format(va_arg(ap, size_t));
            break;
          default:
            retval = formatter.format(va_arg(ap, int));
          }
          dfsan_set_label(*va_labels++, formatter.str_cur(),
                          formatter.num_written_bytes(retval));
          end_fmt = true;
          break;

        case 'a':
        case 'A':
        case 'e':
        case 'E':
        case 'f':
        case 'F':
        case 'g':
        case 'G':
          if (*(formatter.fmt_cur - 1) == 'L') {
            retval = formatter.format(va_arg(ap, long double));
          } else {
            retval = formatter.format(va_arg(ap, double));
          }
          dfsan_set_label(*va_labels++, formatter.str_cur(),
                          formatter.num_written_bytes(retval));
          end_fmt = true;
          break;

        case 'c':
          retval = formatter.format(va_arg(ap, int));
          dfsan_set_label(*va_labels++, formatter.str_cur(),
                          formatter.num_written_bytes(retval));
          end_fmt = true;
          break;

        case 's': {
          char *arg = va_arg(ap, char *);
          retval = formatter.format(arg);
          va_labels++;
          internal_memcpy(shadow_for(formatter.str_cur()), shadow_for(arg),
                          sizeof(dfsan_label) *
                              formatter.num_written_bytes(retval));
          end_fmt = true;
          break;
        }

        case 'p':
          retval = formatter.format(va_arg(ap, void *));
          dfsan_set_label(*va_labels++, formatter.str_cur(),
                          formatter.num_written_bytes(retval));
          end_fmt = true;
          break;

        case 'n': {
          int *ptr = va_arg(ap, int *);
          *ptr = (int)formatter.str_off;
          va_labels++;
          dfsan_set_label(0, ptr, sizeof(ptr));
          end_fmt = true;
          break;
        }

        case '%':
          retval = formatter.format();
          dfsan_set_label(0, formatter.str_cur(),
                          formatter.num_written_bytes(retval));
          end_fmt = true;
          break;

        case '*':
          formatter.width = va_arg(ap, int);
          va_labels++;
          break;

        default:
          break;
        }
      }
    }

    if (retval < 0) {
      return retval;
    }

    formatter.fmt_cur++;
    formatter.str_off += retval;
  }

  *ret_label = 0;

  // Number of bytes written in total.
  return formatter.str_off;
}

extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE
int __dfsw_sprintf(char *str, const char *format, dfsan_label str_label,
                   dfsan_label format_label, dfsan_label *va_labels,
                   dfsan_label *ret_label, ...) {
  va_list ap;
  va_start(ap, ret_label);
  int ret = format_buffer(str, ~0ul, format, va_labels, ret_label, ap);
  va_end(ap);
  return ret;
}

SANITIZER_INTERFACE_ATTRIBUTE
int __dfsw_snprintf(char *str, size_t size, const char *format,
                    dfsan_label str_label, dfsan_label size_label,
                    dfsan_label format_label, dfsan_label *va_labels,
                    dfsan_label *ret_label, ...) {
  va_list ap;
  va_start(ap, ret_label);
  int ret = format_buffer(str, size, format, va_labels, ret_label, ap);
  va_end(ap);
  return ret;
}

// Default empty implementations (weak). Users should redefine them.
SANITIZER_INTERFACE_WEAK_DEF(void, __sanitizer_cov_trace_pc_guard, u32 *) {}
SANITIZER_INTERFACE_WEAK_DEF(void, __sanitizer_cov_trace_pc_guard_init, u32 *,
                             u32 *) {}
SANITIZER_INTERFACE_WEAK_DEF(void, __sanitizer_cov_pcs_init, void) {}
SANITIZER_INTERFACE_WEAK_DEF(void, __sanitizer_cov_trace_pc_indir, void) {}

SANITIZER_INTERFACE_WEAK_DEF(void, __dfsw___sanitizer_cov_trace_cmp, void) {}
SANITIZER_INTERFACE_WEAK_DEF(void, __dfsw___sanitizer_cov_trace_cmp1, void) {}
SANITIZER_INTERFACE_WEAK_DEF(void, __dfsw___sanitizer_cov_trace_cmp2, void) {}
SANITIZER_INTERFACE_WEAK_DEF(void, __dfsw___sanitizer_cov_trace_cmp4, void) {}
SANITIZER_INTERFACE_WEAK_DEF(void, __dfsw___sanitizer_cov_trace_cmp8, void) {}
SANITIZER_INTERFACE_WEAK_DEF(void, __dfsw___sanitizer_cov_trace_const_cmp1,
                             void) {}
SANITIZER_INTERFACE_WEAK_DEF(void, __dfsw___sanitizer_cov_trace_const_cmp2,
                             void) {}
SANITIZER_INTERFACE_WEAK_DEF(void, __dfsw___sanitizer_cov_trace_const_cmp4,
                             void) {}
SANITIZER_INTERFACE_WEAK_DEF(void, __dfsw___sanitizer_cov_trace_const_cmp8,
                             void) {}
SANITIZER_INTERFACE_WEAK_DEF(void, __dfsw___sanitizer_cov_trace_switch, void) {}
}  // extern "C"
