/* Copyright (c) 1998-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Thorsten Kukuk <kukuk@suse.de>, 1998.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#ifndef _NSCD_H
#define _NSCD_H	1

#include <pthread.h>
#include <stdbool.h>
#include <time.h>
#include <sys/uio.h>

/* The declarations for the request and response types are in the file
   "nscd-client.h", which should contain everything needed by client
   functions.  */
#include "nscd-client.h"


/* Handle databases.  */
typedef enum
{
  pwddb,
  grpdb,
  hstdb,
  servdb,
  netgrdb,
  lastdb
} dbtype;


/* Default limit on the number of times a value gets reloaded without
   being used in the meantime.  NSCD does not throw a value out as
   soon as it times out.  It tries to reload the value from the
   server.  Only if the value has not been used for so many rounds it
   is removed.  */
#define DEFAULT_RELOAD_LIMIT 5


/* Time before restarting the process in paranoia mode.  */
#define RESTART_INTERVAL (60 * 60)


/* Stack size for worker threads.  */
#define NSCD_THREAD_STACKSIZE 1024 * 1024 * (sizeof (void *) / 4)

/* Maximum size of stack frames we allow the thread to use.  We use
   80% of the thread stack size.  */
#define MAX_STACK_USE ((8 * NSCD_THREAD_STACKSIZE) / 10)

/* Records the file registered per database that when changed
   or modified requires invalidating the database.  */
struct traced_file
{
  /* Tracks the last modified time of the traced file.  */
  time_t mtime;
  /* Support multiple registered files per database.  */
  struct traced_file *next;
  int call_res_init;
  /* Requires Inotify support to do anything useful.  */
#define TRACED_FILE	0
#define TRACED_DIR	1
  int inotify_descr[2];
# ifndef PATH_MAX
#  define PATH_MAX 1024
# endif
  /* The parent directory is used to scan for creation/deletion.  */
  char dname[PATH_MAX];
  /* Just the name of the file with no directory component.  */
  char *sfname;
  /* The full-path name of the registered file.  */
  char fname[];
};

/* Initialize a `struct traced_file`.  As input we need the name
   of the file, and if invalidation requires calling res_init.
   If CRINIT is 1 then res_init will be called after invalidation
   or if the traced file is changed in any way, otherwise it will
   not.  */
static inline void
init_traced_file(struct traced_file *file, const char *fname, int crinit)
{
   char *dname;
   file->mtime = 0;
   file->inotify_descr[TRACED_FILE] = -1;
   file->inotify_descr[TRACED_DIR] = -1;
   strcpy (file->fname, fname);
   /* Compute the parent directory name and store a copy.  The copy makes
      it much faster to add/remove watches while nscd is running instead
      of computing this over and over again in a temp buffer.  */
   file->dname[0] = '\0';
   dname = strrchr (fname, '/');
   if (dname != NULL)
     {
       size_t len = (size_t)(dname - fname);
       if (len > sizeof (file->dname))
	 abort ();
       memcpy (file->dname, file->fname, len);
       file->dname[len] = '\0';
     }
   /* The basename is the name just after the last forward slash.  */
   file->sfname = &dname[1];
   file->call_res_init = crinit;
}

#define define_traced_file(id, filename) 			\
static union							\
{								\
  struct traced_file file;					\
  char buf[sizeof (struct traced_file) + sizeof (filename)];	\
} id##_traced_file;

/* Structure describing dynamic part of one database.  */
struct database_dyn
{
  pthread_rwlock_t lock;
  pthread_cond_t prune_cond;
  pthread_mutex_t prune_lock;
  pthread_mutex_t prune_run_lock;
  time_t wakeup_time;

  int enabled;
  int check_file;
  int clear_cache;
  int persistent;
  int shared;
  int propagate;
  struct traced_file *traced_files;
  const char *db_filename;
  size_t suggested_module;
  size_t max_db_size;

  unsigned long int postimeout;	/* In seconds.  */
  unsigned long int negtimeout;	/* In seconds.  */

  int wr_fd;			/* Writable file descriptor.  */
  int ro_fd;			/* Unwritable file descriptor.  */

  const struct iovec *disabled_iov;

  struct database_pers_head *head;
  char *data;
  size_t memsize;
  pthread_mutex_t memlock;
  bool mmap_used;
  bool last_alloc_failed;
};


/* Paths of the file for the persistent storage.  */
#define _PATH_NSCD_PASSWD_DB	"/var/db/nscd/passwd"
#define _PATH_NSCD_GROUP_DB	"/var/db/nscd/group"
#define _PATH_NSCD_HOSTS_DB	"/var/db/nscd/hosts"
#define _PATH_NSCD_SERVICES_DB	"/var/db/nscd/services"
#define _PATH_NSCD_NETGROUP_DB	"/var/db/nscd/netgroup"

/* Path used when not using persistent storage.  */
#define _PATH_NSCD_XYZ_DB_TMP	"/var/run/nscd/dbXXXXXX"

/* Maximum alignment requirement we will encounter.  */
#define BLOCK_ALIGN_LOG 3
#define BLOCK_ALIGN (1 << BLOCK_ALIGN_LOG)
#define BLOCK_ALIGN_M1 (BLOCK_ALIGN - 1)

/* Default value for the maximum size of the database files.  */
#define DEFAULT_MAX_DB_SIZE	(32 * 1024 * 1024)

/* Number of bytes of data we initially reserve for each hash table bucket.  */
#define DEFAULT_DATASIZE_PER_BUCKET 1024

/* Default module of hash table.  */
#define DEFAULT_SUGGESTED_MODULE 211


/* Number of seconds between two cache pruning runs if we do not have
   better information when it is really needed.  */
#define CACHE_PRUNE_INTERVAL	15


/* Global variables.  */
extern struct database_dyn dbs[lastdb] attribute_hidden;
extern const char *const dbnames[lastdb];
extern const char *const serv2str[LASTREQ];

extern const struct iovec pwd_iov_disabled;
extern const struct iovec grp_iov_disabled;
extern const struct iovec hst_iov_disabled;
extern const struct iovec serv_iov_disabled;
extern const struct iovec netgroup_iov_disabled;


/* Initial number of threads to run.  */
extern int nthreads;
/* Maximum number of threads to use.  */
extern int max_nthreads;

/* Inotify descriptor.  */
extern int inotify_fd;

/* User name to run server processes as.  */
extern const char *server_user;

/* Name and UID of user who is allowed to request statistics.  */
extern const char *stat_user;
extern uid_t stat_uid;

/* Time the server was started.  */
extern time_t start_time;

/* Number of times clients had to wait.  */
extern unsigned long int client_queued;

/* Maximum needed alignment.  */
extern const size_t block_align;

/* Number of times a value is reloaded without being used.  UINT_MAX
   means unlimited.  */
extern unsigned int reload_count;

/* Pagesize minus one.  */
extern uintptr_t pagesize_m1;

/* Nonzero if paranoia mode is enabled.  */
extern int paranoia;
/* Time after which the process restarts.  */
extern time_t restart_time;
/* How much time between restarts.  */
extern time_t restart_interval;
/* Old current working directory.  */
extern const char *oldcwd;
/* Old user and group ID.  */
extern uid_t old_uid;
extern gid_t old_gid;


/* Prototypes for global functions.  */

/* Wrapper functions with error checking for standard functions.  */
#include <programs/xmalloc.h>

/* nscd.c */
extern void termination_handler (int signum) __attribute__ ((__noreturn__));
extern int nscd_open_socket (void);
void notify_parent (int child_ret);
void do_exit (int child_ret, int errnum, const char *format, ...);

/* connections.c */
extern void nscd_init (void);
extern void register_traced_file (size_t dbidx, struct traced_file *finfo);
#ifdef HAVE_INOTIFY
extern void install_watches (struct traced_file *finfo);
#endif
extern void close_sockets (void);
extern void start_threads (void) __attribute__ ((__noreturn__));

/* nscd_conf.c */
extern int nscd_parse_file (const char *fname,
			    struct database_dyn dbs[lastdb]);

/* nscd_stat.c */
extern void send_stats (int fd, struct database_dyn dbs[lastdb]);
extern int receive_print_stats (void) __attribute__ ((__noreturn__));

/* cache.c */
extern struct datahead *cache_search (request_type, const void *key,
				      size_t len, struct database_dyn *table,
				      uid_t owner);
extern int cache_add (int type, const void *key, size_t len,
		      struct datahead *packet, bool first,
		      struct database_dyn *table, uid_t owner,
		      bool prune_wakeup);
extern time_t prune_cache (struct database_dyn *table, time_t now, int fd);

/* pwdcache.c */
extern void addpwbyname (struct database_dyn *db, int fd, request_header *req,
			 void *key, uid_t uid);
extern void addpwbyuid (struct database_dyn *db, int fd, request_header *req,
			void *key, uid_t uid);
extern time_t readdpwbyname (struct database_dyn *db, struct hashentry *he,
			     struct datahead *dh);
extern time_t readdpwbyuid (struct database_dyn *db, struct hashentry *he,
			    struct datahead *dh);

/* grpcache.c */
extern void addgrbyname (struct database_dyn *db, int fd, request_header *req,
			 void *key, uid_t uid);
extern void addgrbygid (struct database_dyn *db, int fd, request_header *req,
			void *key, uid_t uid);
extern time_t readdgrbyname (struct database_dyn *db, struct hashentry *he,
			     struct datahead *dh);
extern time_t readdgrbygid (struct database_dyn *db, struct hashentry *he,
			    struct datahead *dh);

/* hstcache.c */
extern void addhstbyname (struct database_dyn *db, int fd, request_header *req,
			  void *key, uid_t uid);
extern void addhstbyaddr (struct database_dyn *db, int fd, request_header *req,
			  void *key, uid_t uid);
extern void addhstbynamev6 (struct database_dyn *db, int fd,
			    request_header *req, void *key, uid_t uid);
extern void addhstbyaddrv6 (struct database_dyn *db, int fd,
			    request_header *req, void *key, uid_t uid);
extern time_t readdhstbyname (struct database_dyn *db, struct hashentry *he,
			      struct datahead *dh);
extern time_t readdhstbyaddr (struct database_dyn *db, struct hashentry *he,
			      struct datahead *dh);
extern time_t readdhstbynamev6 (struct database_dyn *db, struct hashentry *he,
				struct datahead *dh);
extern time_t readdhstbyaddrv6 (struct database_dyn *db, struct hashentry *he,
				struct datahead *dh);

/* aicache.c */
extern void addhstai (struct database_dyn *db, int fd, request_header *req,
		      void *key, uid_t uid);
extern time_t readdhstai (struct database_dyn *db, struct hashentry *he,
			  struct datahead *dh);


/* initgrcache.c */
extern void addinitgroups (struct database_dyn *db, int fd,
			   request_header *req, void *key, uid_t uid);
extern time_t readdinitgroups (struct database_dyn *db, struct hashentry *he,
			       struct datahead *dh);

/* servicecache.c */
extern void addservbyname (struct database_dyn *db, int fd,
			   request_header *req, void *key, uid_t uid);
extern time_t readdservbyname (struct database_dyn *db, struct hashentry *he,
			       struct datahead *dh);
extern void addservbyport (struct database_dyn *db, int fd,
			   request_header *req, void *key, uid_t uid);
extern time_t readdservbyport (struct database_dyn *db, struct hashentry *he,
			       struct datahead *dh);

/* netgroupcache.c */
extern void addinnetgr (struct database_dyn *db, int fd, request_header *req,
			void *key, uid_t uid);
extern time_t readdinnetgr (struct database_dyn *db, struct hashentry *he,
			    struct datahead *dh);
extern void addgetnetgrent (struct database_dyn *db, int fd,
			    request_header *req, void *key, uid_t uid);
extern time_t readdgetnetgrent (struct database_dyn *db, struct hashentry *he,
				struct datahead *dh);

/* mem.c */
extern void *mempool_alloc (struct database_dyn *db, size_t len,
			    int data_alloc);
extern void gc (struct database_dyn *db);


/* nscd_setup_thread.c */
extern int setup_thread (struct database_dyn *db);

/* cachedumper.c */
extern void nscd_print_cache (const char *name);

/* Special version of TEMP_FAILURE_RETRY for functions returning error
   values.  */
#define TEMP_FAILURE_RETRY_VAL(expression) \
  (__extension__							      \
    ({ long int __result;						      \
       do __result = (long int) (expression);				      \
       while (__result == EINTR);					      \
       __result; }))

#endif /* nscd.h */
