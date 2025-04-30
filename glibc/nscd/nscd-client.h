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

/* This file defines everything that client code should need to
   know to talk to the nscd daemon.  */

#ifndef _NSCD_CLIENT_H
#define _NSCD_CLIENT_H	1

#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <sys/types.h>
#include <atomic.h>
#include <nscd-types.h>
#include <sys/uio.h>


/* Version number of the daemon interface */
#define NSCD_VERSION 2

/* Path of the file where the PID of the running system is stored.  */
#define _PATH_NSCDPID	 "/var/run/nscd/nscd.pid"

/* Path for the Unix domain socket.  */
#define _PATH_NSCDSOCKET "/var/run/nscd/socket"

/* Path for the configuration file.  */
#define _PATH_NSCDCONF	 "/etc/nscd.conf"

/* Maximum allowed length for the key.  */
#define MAXKEYLEN 1024


/* Available services.  */
typedef enum
{
  GETPWBYNAME,
  GETPWBYUID,
  GETGRBYNAME,
  GETGRBYGID,
  GETHOSTBYNAME,
  GETHOSTBYNAMEv6,
  GETHOSTBYADDR,
  GETHOSTBYADDRv6,
  SHUTDOWN,		/* Shut the server down.  */
  GETSTAT,		/* Get the server statistic.  */
  INVALIDATE,           /* Invalidate one special cache.  */
  GETFDPW,
  GETFDGR,
  GETFDHST,
  GETAI,
  INITGROUPS,
  GETSERVBYNAME,
  GETSERVBYPORT,
  GETFDSERV,
  GETNETGRENT,
  INNETGR,
  GETFDNETGR,
  LASTREQ
} request_type;


/* Header common to all requests */
typedef struct
{
  int32_t version;	/* Version number of the daemon interface.  */
  request_type type;	/* Service requested.  */
  int32_t key_len;	/* Key length.  */
} request_header;


/* Structure sent in reply to password query.  Note that this struct is
   sent also if the service is disabled or there is no record found.  */
typedef struct
{
  int32_t version;
  int32_t found;
  nscd_ssize_t pw_name_len;
  nscd_ssize_t pw_passwd_len;
  uid_t pw_uid;
  gid_t pw_gid;
  nscd_ssize_t pw_gecos_len;
  nscd_ssize_t pw_dir_len;
  nscd_ssize_t pw_shell_len;
} pw_response_header;


/* Structure sent in reply to group query.  Note that this struct is
   sent also if the service is disabled or there is no record found.  */
typedef struct
{
  int32_t version;
  int32_t found;
  nscd_ssize_t gr_name_len;
  nscd_ssize_t gr_passwd_len;
  gid_t gr_gid;
  nscd_ssize_t gr_mem_cnt;
} gr_response_header;


/* Structure sent in reply to host query.  Note that this struct is
   sent also if the service is disabled or there is no record found.  */
typedef struct
{
  int32_t version;
  int32_t found;
  nscd_ssize_t h_name_len;
  nscd_ssize_t h_aliases_cnt;
  int32_t h_addrtype;
  int32_t h_length;
  nscd_ssize_t h_addr_list_cnt;
  int32_t error;
} hst_response_header;


/* Structure sent in reply to addrinfo query.  Note that this struct is
   sent also if the service is disabled or there is no record found.  */
typedef struct
{
  int32_t version;
  int32_t found;
  nscd_ssize_t naddrs;
  nscd_ssize_t addrslen;
  nscd_ssize_t canonlen;
  int32_t error;
} ai_response_header;

/* Structure filled in by __nscd_getai.  */
struct nscd_ai_result
{
  int naddrs;
  char *canon;
  uint8_t *family;
  char *addrs;
};

/* Structure sent in reply to initgroups query.  Note that this struct is
   sent also if the service is disabled or there is no record found.  */
typedef struct
{
  int32_t version;
  int32_t found;
  nscd_ssize_t ngrps;
} initgr_response_header;


/* Structure sent in reply to services query.  Note that this struct is
   sent also if the service is disabled or there is no record found.  */
typedef struct
{
  int32_t version;
  int32_t found;
  nscd_ssize_t s_name_len;
  nscd_ssize_t s_proto_len;
  nscd_ssize_t s_aliases_cnt;
  int32_t s_port;
} serv_response_header;


/* Structure send in reply to netgroup query.  Note that this struct is
   sent also if the service is disabled or there is no record found.  */
typedef struct
{
  int32_t version;
  int32_t found;
  nscd_ssize_t nresults;
  nscd_ssize_t result_len;
} netgroup_response_header;

typedef struct
{
  int32_t version;
  int32_t found;
  int32_t result;
} innetgroup_response_header;


/* Type for offsets in data part of database.  */
typedef uint32_t ref_t;
/* Value for invalid/no reference.  */
#define ENDREF	UINT32_MAX

/* Timestamp type.  */
typedef uint64_t nscd_time_t;

/* Maximum timestamp.  */
#define MAX_TIMEOUT_VALUE \
  (sizeof (time_t) == sizeof (long int) ? LONG_MAX : INT_MAX)

/* Alignment requirement of the beginning of the data region.  */
#define ALIGN 16


/* Head of record in data part of database.  */
struct datahead
{
  nscd_ssize_t allocsize;	/* Allocated Bytes.  */
  nscd_ssize_t recsize;		/* Size of the record.  */
  nscd_time_t timeout;		/* Time when this entry becomes invalid.  */
  uint8_t notfound;		/* Nonzero if data has not been found.  */
  uint8_t nreloads;		/* Reloads without use.  */
  uint8_t usable;		/* False if the entry must be ignored.  */
  uint8_t unused;		/* Unused.  */
  uint32_t ttl;			/* TTL value used.  */

  /* We need to have the following element aligned for the response
     header data types and their use in the 'struct dataset' types
     defined in the XXXcache.c files.  */
  union
  {
    pw_response_header pwdata;
    gr_response_header grdata;
    hst_response_header hstdata;
    ai_response_header aidata;
    initgr_response_header initgrdata;
    serv_response_header servdata;
    netgroup_response_header netgroupdata;
    innetgroup_response_header innetgroupdata;
    nscd_ssize_t align1;
    nscd_time_t align2;
  } data[0];
};

static inline time_t
datahead_init_common (struct datahead *head, nscd_ssize_t allocsize,
		      nscd_ssize_t recsize, uint32_t ttl)
{
  /* Initialize so that we don't write out junk in uninitialized data to the
     cache.  */
  memset (head, 0, sizeof (*head));

  head->allocsize = allocsize;
  head->recsize = recsize;
  head->usable = true;

  head->ttl = ttl;

  /* Compute and return the timeout time.  */
  return head->timeout = time (NULL) + ttl;
}

static inline time_t
datahead_init_pos (struct datahead *head, nscd_ssize_t allocsize,
		   nscd_ssize_t recsize, uint8_t nreloads, uint32_t ttl)
{
  time_t ret = datahead_init_common (head, allocsize, recsize, ttl);

  head->notfound = false;
  head->nreloads = nreloads;

  return ret;
}

static inline time_t
datahead_init_neg (struct datahead *head, nscd_ssize_t allocsize,
		   nscd_ssize_t recsize, uint32_t ttl)
{
  time_t ret = datahead_init_common (head, allocsize, recsize, ttl);

  /* We don't need to touch nreloads here since it is set to our desired value
     (0) when we clear the structure.  */
  head->notfound = true;

  return ret;
}

/* Structure for one hash table entry.  */
struct hashentry
{
  request_type type:8;		/* Which type of dataset.  */
  bool first;			/* True if this was the original key.  */
  nscd_ssize_t len;		/* Length of key.  */
  ref_t key;			/* Pointer to key.  */
  int32_t owner;		/* If secure table, this is the owner.  */
  ref_t next;			/* Next entry in this hash bucket list.  */
  ref_t packet;			/* Records for the result.  */
  union
  {
    struct hashentry *dellist;	/* Next record to be deleted.  This can be a
				   pointer since only nscd uses this field.  */
    ref_t *prevp;		/* Pointer to field containing forward
				   reference.  */
  };
};


/* Current persistent database version.  */
#define DB_VERSION	2

/* Maximum time allowed between updates of the timestamp.  */
#define MAPPING_TIMEOUT (5 * 60)


/* Used indices for the EXTRA_DATA element of 'database_pers_head'.
   Each database has its own indices.  */
#define NSCD_HST_IDX_CONF_TIMESTAMP	0


/* Header of persistent database file.  */
struct database_pers_head
{
  int32_t version;
  int32_t header_size;
  volatile int32_t gc_cycle;
  volatile int32_t nscd_certainly_running;
  volatile nscd_time_t timestamp;
  /* Room for extensions.  */
  volatile uint32_t extra_data[4];

  nscd_ssize_t module;
  nscd_ssize_t data_size;

  nscd_ssize_t first_free;	/* Offset of first free byte in data area.  */

  nscd_ssize_t nentries;
  nscd_ssize_t maxnentries;
  nscd_ssize_t maxnsearched;

  uint64_t poshit;
  uint64_t neghit;
  uint64_t posmiss;
  uint64_t negmiss;

  uint64_t rdlockdelayed;
  uint64_t wrlockdelayed;

  uint64_t addfailed;

  ref_t array[0];
};


/* Mapped database record.  */
struct mapped_database
{
  const struct database_pers_head *head;
  const char *data;
  size_t mapsize;
  int counter;		/* > 0 indicates it is usable.  */
  size_t datasize;
};
#define NO_MAPPING ((struct mapped_database *) -1l)

struct locked_map_ptr
{
  int lock;
  struct mapped_database *mapped;
};
#define libc_locked_map_ptr(class, name) class struct locked_map_ptr name

/* Try acquiring lock for mapptr, returns true if it succeeds, false
   if not.  */
static inline bool
__nscd_acquire_maplock (volatile struct locked_map_ptr *mapptr)
{
  int cnt = 0;
  while (__builtin_expect (atomic_compare_and_exchange_val_acq (&mapptr->lock,
								1, 0) != 0, 0))
    {
      // XXX Best number of rounds?
      if (__glibc_unlikely (++cnt > 5))
	return false;

      atomic_spin_nop ();
    }

  return true;
}


/* Open socket connection to nscd server.  */
extern int __nscd_open_socket (const char *key, size_t keylen,
			       request_type type, void *response,
			       size_t responselen) attribute_hidden;

/* Try to get a file descriptor for the shared meory segment
   containing the database.  */
extern struct mapped_database *__nscd_get_mapping (request_type type,
						   const char *key,
						   struct mapped_database **mappedp) attribute_hidden;

/* Get reference of mapping.  */
extern struct mapped_database *__nscd_get_map_ref (request_type type,
						   const char *name,
						   volatile struct locked_map_ptr *mapptr,
						   int *gc_cyclep)
  attribute_hidden;

/* Unmap database.  */
extern void __nscd_unmap (struct mapped_database *mapped)
  attribute_hidden;

/* Drop reference of mapping.  */
static int
__attribute__ ((unused))
__nscd_drop_map_ref (struct mapped_database *map, int *gc_cycle)
{
  if (map != NO_MAPPING)
    {
      int now_cycle = map->head->gc_cycle;
      if (__glibc_unlikely (now_cycle != *gc_cycle))
	{
	  /* We might have read inconsistent data.  */
	  *gc_cycle = now_cycle;
	  return -1;
	}

      if (atomic_decrement_val (&map->counter) == 0)
	__nscd_unmap (map);
    }

  return 0;
}


/* Search the mapped database.  */
extern struct datahead *__nscd_cache_search (request_type type,
					     const char *key,
					     size_t keylen,
					     const struct mapped_database *mapped,
					     size_t datalen)
  attribute_hidden;

/* Wrappers around read, readv and write that only read/write less than LEN
   bytes on error or EOF.  */
extern ssize_t __readall (int fd, void *buf, size_t len)
  attribute_hidden;
extern ssize_t __readvall (int fd, const struct iovec *iov, int iovcnt)
  attribute_hidden;
extern ssize_t writeall (int fd, const void *buf, size_t len)
  attribute_hidden;

/* Get netlink timestamp counter from mapped area or zero.  */
extern uint32_t __nscd_get_nl_timestamp (void)
  attribute_hidden;

#endif /* nscd.h */
