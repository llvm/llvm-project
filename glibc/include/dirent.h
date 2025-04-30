#ifndef _DIRENT_H
# ifndef _ISOMAC
#  include <dirstream.h>
# endif
# include <dirent/dirent.h>
# ifndef _ISOMAC
# include <sys/stat.h>
# include <stdbool.h>

struct scandir_cancel_struct
{
  DIR *dp;
  void *v;
  size_t cnt;
};

/* Now define the internal interfaces.  */
extern DIR *__opendir (const char *__name) attribute_hidden;
extern DIR *__opendirat (int dfd, const char *__name) attribute_hidden;
extern DIR *__fdopendir (int __fd) attribute_hidden;
extern int __closedir (DIR *__dirp) attribute_hidden;
extern struct dirent *__readdir (DIR *__dirp) attribute_hidden;
extern struct dirent *__readdir_unlocked (DIR *__dirp) attribute_hidden;
extern struct dirent64 *__readdir64 (DIR *__dirp);
libc_hidden_proto (__readdir64)
extern int __readdir_r (DIR *__dirp, struct dirent *__entry,
			struct dirent **__result);
extern int __readdir64_r (DIR *__dirp, struct dirent64 *__entry,
			  struct dirent64 **__result);
extern int __scandir64 (const char * __dir,
			struct dirent64 *** __namelist,
			int (*__selector) (const struct dirent64 *),
			int (*__cmp) (const struct dirent64 **,
				      const struct dirent64 **));
extern __ssize_t __getdirentries (int __fd, char *__restrict __buf,
				size_t __nbytes,
				__off_t *__restrict __basep)
     __THROW __nonnull ((2, 4));

/* These functions are only implemented on Linux.  */
extern __ssize_t __getdents (int __fd, void *__buf, size_t __nbytes)
     attribute_hidden;
extern __ssize_t __getdents64 (int __fd, void *__buf, size_t __nbytes);
libc_hidden_proto (__getdents64)

extern int __alphasort64 (const struct dirent64 **a, const struct dirent64 **b)
     __attribute_pure__;
extern int __versionsort64 (const struct dirent64 **a,
			    const struct dirent64 **b)
     __attribute_pure__;
extern DIR *__alloc_dir (int fd, bool close_fd, int flags,
			 const struct __stat64_t64 *statp)
     __nonnull (4) attribute_hidden;
extern __typeof (rewinddir) __rewinddir;
extern __typeof (seekdir) __seekdir;
extern __typeof (dirfd) __dirfd;
libc_hidden_proto (dirfd);

extern void __scandir_cancel_handler (void *arg) attribute_hidden;
extern int __scandir_tail (DIR *dp,
			   struct dirent ***namelist,
			   int (*select) (const struct dirent *),
			   int (*cmp) (const struct dirent **,
				       const struct dirent **))
  attribute_hidden;
#  if !_DIRENT_MATCHES_DIRENT64
extern int __scandir_tail (DIR *dp,
			   struct dirent ***namelist,
			   int (*select) (const struct dirent *),
			   int (*cmp) (const struct dirent **,
					 const struct dirent **))
  attribute_hidden;
#  endif
extern int __scandir64_tail (DIR *dp,
			     struct dirent64 ***namelist,
			     int (*select) (const struct dirent64 *),
			     int (*cmp) (const struct dirent64 **,
					 const struct dirent64 **))
  attribute_hidden;

libc_hidden_proto (__rewinddir)
extern __typeof (scandirat) __scandirat;

#  if IS_IN (rtld)
extern __typeof (__rewinddir) __rewinddir attribute_hidden;
#  endif
# endif

#endif
