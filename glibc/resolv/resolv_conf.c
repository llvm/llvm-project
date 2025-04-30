/* Extended resolver state separate from struct __res_state.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

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

#include <resolv_conf.h>

#include <alloc_buffer.h>
#include <assert.h>
#include <libc-lock.h>
#include <resolv-internal.h>
#include <sys/stat.h>
#include <libc-symbols.h>
#include <file_change_detection.h>

/* _res._u._ext.__glibc_extension_index is used as an index into a
   struct resolv_conf_array object.  The intent of this construction
   is to make reasonably sure that even if struct __res_state objects
   are copied around and patched by applications, we can still detect
   accesses to stale extended resolver state.  The array elements are
   either struct resolv_conf * pointers (if the LSB is cleared) or
   free list entries (if the LSB is set).  The free list is used to
   speed up finding available entries in the array.  */
#define DYNARRAY_STRUCT resolv_conf_array
#define DYNARRAY_ELEMENT uintptr_t
#define DYNARRAY_PREFIX resolv_conf_array_
#define DYNARRAY_INITIAL_SIZE 0
#include <malloc/dynarray-skeleton.c>

/* A magic constant for XORing the extension index
   (_res._u._ext.__glibc_extension_index).  This makes it less likely
   that a valid index is created by accident.  In particular, a zero
   value leads to an invalid index.  */
#define INDEX_MAGIC 0x26a8fa5e48af8061ULL

/* Global resolv.conf-related state.  */
struct resolv_conf_global
{
  /* struct __res_state objects contain the extension index
     (_res._u._ext.__glibc_extension_index ^ INDEX_MAGIC), which
     refers to an element of this array.  When a struct resolv_conf
     object (extended resolver state) is associated with a struct
     __res_state object (legacy resolver state), its reference count
     is increased and added to this array.  Conversely, if the
     extended state is detached from the basic state (during
     reinitialization or deallocation), the index is decremented, and
     the array element is overwritten with NULL.  */
  struct resolv_conf_array array;

  /* Start of the free list in the array.  Zero if the free list is
     empty.  Otherwise, free_list_start >> 1 is the first element of
     the free list (and the free list entries all have their LSB set
     and are shifted one to the left).  */
  uintptr_t free_list_start;

  /* Cached current configuration object for /etc/resolv.conf.  */
  struct resolv_conf *conf_current;

  /* File system identification for /etc/resolv.conf.  */
  struct file_change_detection file_resolve_conf;
};

/* Lazily allocated storage for struct resolv_conf_global.  */
static struct resolv_conf_global *global;

/* The lock synchronizes access to global and *global.  It also
   protects the __refcount member of struct resolv_conf.  */
__libc_lock_define_initialized (static, lock);

/* Ensure that GLOBAL is allocated and lock it.  Return NULL if
   memory allocation failes.  */
static struct resolv_conf_global *
get_locked_global (void)
{
  __libc_lock_lock (lock);
  /* Use relaxed MO through because of load outside the lock in
     __resolv_conf_detach.  */
  struct resolv_conf_global *global_copy = atomic_load_relaxed (&global);
  if (global_copy == NULL)
    {
      global_copy = calloc (1, sizeof (*global));
      if (global_copy == NULL)
        return NULL;
      atomic_store_relaxed (&global, global_copy);
      resolv_conf_array_init (&global_copy->array);
    }
  return global_copy;
}

/* Relinquish the lock acquired by get_locked_global.  */
static void
put_locked_global (struct resolv_conf_global *global_copy)
{
  __libc_lock_unlock (lock);
}

/* Decrement the reference counter.  The caller must acquire the lock
   around the function call.  */
static void
conf_decrement (struct resolv_conf *conf)
{
  assert (conf->__refcount > 0);
  if (--conf->__refcount == 0)
    free (conf);
}

struct resolv_conf *
__resolv_conf_get_current (void)
{
  struct file_change_detection initial;
  if (!__file_change_detection_for_path (&initial, _PATH_RESCONF))
    return NULL;

  struct resolv_conf_global *global_copy = get_locked_global ();
  if (global_copy == NULL)
    return NULL;
  struct resolv_conf *conf;
  if (global_copy->conf_current != NULL
      && __file_is_unchanged (&initial, &global_copy->file_resolve_conf))
    /* We can reuse the cached configuration object.  */
    conf = global_copy->conf_current;
  else
    {
      /* Parse configuration while holding the lock.  This avoids
         duplicate work.  */
      struct file_change_detection after_load;
      conf = __resolv_conf_load (NULL, &after_load);
      if (conf != NULL)
        {
          if (global_copy->conf_current != NULL)
            conf_decrement (global_copy->conf_current);
          global_copy->conf_current = conf; /* Takes ownership.  */

          /* Update file change detection data, but only if it matches
             the initial measurement.  This avoids an ABA race in case
             /etc/resolv.conf is temporarily replaced while the file
             is read (after the initial measurement), and restored to
             the initial version later.  */
          if (__file_is_unchanged (&initial, &after_load))
            global_copy->file_resolve_conf = after_load;
          else
            /* If there is a discrepancy, trigger a reload during the
               next use.  */
            global_copy->file_resolve_conf.size = -1;
        }
    }

  if (conf != NULL)
    {
      /* Return an additional reference.  */
      assert (conf->__refcount > 0);
      ++conf->__refcount;
      assert (conf->__refcount > 0);
    }
  put_locked_global (global_copy);
  return conf;
}

/* Internal implementation of __resolv_conf_get, without validation
   against *RESP.  */
static struct resolv_conf *
resolv_conf_get_1 (const struct __res_state *resp)
{
  /* Not initialized, and therefore no assoicated context.  */
  if (!(resp->options & RES_INIT))
    return NULL;

  struct resolv_conf_global *global_copy = get_locked_global ();
  if (global_copy == NULL)
    /* A memory allocation failure here means that no associated
       contexts exists, so returning NULL is correct.  */
    return NULL;
  size_t index = resp->_u._ext.__glibc_extension_index ^ INDEX_MAGIC;
  struct resolv_conf *conf = NULL;
  if (index < resolv_conf_array_size (&global_copy->array))
    {
      uintptr_t *slot = resolv_conf_array_at (&global_copy->array, index);
      if (!(*slot & 1))
        {
          conf = (struct resolv_conf *) *slot;
          assert (conf->__refcount > 0);
          ++conf->__refcount;
        }
    }
  put_locked_global (global_copy);
  return conf;
}

/* Return true if both IPv4 addresses are equal.  */
static bool
same_address_v4 (const struct sockaddr_in *left,
                 const struct sockaddr_in *right)
{
  return left->sin_addr.s_addr == right->sin_addr.s_addr
    && left->sin_port == right->sin_port;
}

/* Return true if both IPv6 addresses are equal.  This ignores the
   flow label.  */
static bool
same_address_v6 (const struct sockaddr_in6 *left,
                 const struct sockaddr_in6 *right)
{
  return memcmp (&left->sin6_addr, &right->sin6_addr,
                 sizeof (left->sin6_addr)) == 0
    && left->sin6_port == right->sin6_port
    && left->sin6_scope_id == right->sin6_scope_id;
}

static bool
same_address (const struct sockaddr *left, const struct sockaddr *right)
{
  if (left->sa_family != right->sa_family)
    return false;
  switch (left->sa_family)
    {
    case AF_INET:
      return same_address_v4 ((const struct sockaddr_in *) left,
                              (const struct sockaddr_in *) right);
    case AF_INET6:
      return same_address_v6 ((const struct sockaddr_in6 *) left,
                              (const struct sockaddr_in6 *) right);
    }
  return false;
}

/* Check that *RESP and CONF match.  Used by __resolv_conf_get.  */
static bool
resolv_conf_matches (const struct __res_state *resp,
                     const struct resolv_conf *conf)
{
  /* NB: Do not compare the options, retrans, retry, ndots.  These can
     be changed by applicaiton.  */

  /* Check that the name servers in *RESP have not been modified by
     the application.  */
  {
    size_t nserv = conf->nameserver_list_size;
    if (nserv > MAXNS)
      nserv = MAXNS;
    /* _ext.nscount is 0 until initialized by res_send.c.  */
    if (resp->nscount != nserv
        || (resp->_u._ext.nscount != 0 && resp->_u._ext.nscount != nserv))
      return false;
    for (size_t i = 0; i < nserv; ++i)
      {
        if (resp->nsaddr_list[i].sin_family == 0)
          {
            if (resp->_u._ext.nsaddrs[i]->sin6_family != AF_INET6)
              return false;
            if (!same_address ((struct sockaddr *) resp->_u._ext.nsaddrs[i],
                               conf->nameserver_list[i]))
              return false;
          }
        else if (resp->nsaddr_list[i].sin_family != AF_INET)
          return false;
        else if (!same_address ((struct sockaddr *) &resp->nsaddr_list[i],
                                conf->nameserver_list[i]))
          return false;
      }
  }

  /* Check that the search list in *RESP has not been modified by the
     application.  */
  {
    if (resp->dnsrch[0] == NULL)
      {
        /* Empty search list.  No default domain name.  */
        return conf->search_list_size == 0 && resp->defdname[0] == '\0';
      }

    if (resp->dnsrch[0] != resp->defdname)
      /* If the search list is not empty, it must start with the
         default domain name.  */
      return false;

    size_t nsearch;
    for (nsearch = 0; nsearch < MAXDNSRCH; ++nsearch)
      if (resp->dnsrch[nsearch] == NULL)
        break;
    if (nsearch > MAXDNSRCH)
      /* Search list is not null-terminated.  */
      return false;

    size_t search_list_size = 0;
    for (size_t i = 0; i < conf->search_list_size; ++i)
      {
        if (resp->dnsrch[i] != NULL)
          {
            search_list_size += strlen (resp->dnsrch[i]) + 1;
            if (strcmp (resp->dnsrch[i], conf->search_list[i]) != 0)
              return false;
          }
        else
          {
            /* resp->dnsrch is truncated if the number of elements
               exceeds MAXDNSRCH, or if the combined storage space for
               the search list exceeds what can be stored in
               resp->defdname.  */
            if (i == MAXDNSRCH || search_list_size > sizeof (resp->dnsrch))
              break;
            /* Otherwise, a mismatch indicates a match failure.  */
            return false;
          }
      }
  }

  /* Check that the sort list has not been modified.  */
  {
    size_t nsort = conf->sort_list_size;
    if (nsort > MAXRESOLVSORT)
      nsort = MAXRESOLVSORT;
    if (resp->nsort != nsort)
      return false;
    for (size_t i = 0; i < nsort; ++i)
      if (resp->sort_list[i].addr.s_addr != conf->sort_list[i].addr.s_addr
          || resp->sort_list[i].mask != conf->sort_list[i].mask)
        return false;
  }

  return true;
}

struct resolv_conf *
__resolv_conf_get (struct __res_state *resp)
{
  struct resolv_conf *conf = resolv_conf_get_1 (resp);
  if (conf == NULL)
    return NULL;
  if (resolv_conf_matches (resp, conf))
    return conf;
  __resolv_conf_put (conf);
  return NULL;
}

void
__resolv_conf_put (struct resolv_conf *conf)
{
  if (conf == NULL)
    return;

  __libc_lock_lock (lock);
  conf_decrement (conf);
  __libc_lock_unlock (lock);
}

struct resolv_conf *
__resolv_conf_allocate (const struct resolv_conf *init)
{
  /* Allocate in decreasing order of alignment.  */
  _Static_assert (__alignof__ (const char *const *)
                  <= __alignof__ (struct resolv_conf), "alignment");
  _Static_assert (__alignof__ (struct sockaddr_in6)
                  <= __alignof__ (const char *const *), "alignment");
  _Static_assert (__alignof__ (struct sockaddr_in)
                  ==  __alignof__ (struct sockaddr_in6), "alignment");
  _Static_assert (__alignof__ (struct resolv_sortlist_entry)
                  <= __alignof__ (struct sockaddr_in), "alignment");

  /* Space needed by the nameserver addresses.  */
  size_t address_space = 0;
  for (size_t i = 0; i < init->nameserver_list_size; ++i)
    if (init->nameserver_list[i]->sa_family == AF_INET)
      address_space += sizeof (struct sockaddr_in);
    else
      {
        assert (init->nameserver_list[i]->sa_family == AF_INET6);
        address_space += sizeof (struct sockaddr_in6);
      }

  /* Space needed by the search list strings.  */
  size_t string_space = 0;
  for (size_t i = 0; i < init->search_list_size; ++i)
    string_space += strlen (init->search_list[i]) + 1;

  /* Allocate the buffer.  */
  void *ptr;
  struct alloc_buffer buffer = alloc_buffer_allocate
    (sizeof (struct resolv_conf)
     + init->nameserver_list_size * sizeof (init->nameserver_list[0])
     + address_space
     + init->search_list_size * sizeof (init->search_list[0])
     + init->sort_list_size * sizeof (init->sort_list[0])
     + string_space,
     &ptr);
  struct resolv_conf *conf
    = alloc_buffer_alloc (&buffer, struct resolv_conf);
  if (conf == NULL)
    /* Memory allocation failure.  */
    return NULL;
  assert (conf == ptr);

  /* Initialize the contents.  */
  conf->__refcount = 1;
  conf->retrans = init->retrans;
  conf->retry = init->retry;
  conf->options = init->options;
  conf->ndots = init->ndots;

  /* Allocate the arrays with pointers.  These must come first because
     they have the highets alignment.  */
  conf->nameserver_list_size = init->nameserver_list_size;
  const struct sockaddr **nameserver_array = alloc_buffer_alloc_array
    (&buffer, const struct sockaddr *, init->nameserver_list_size);
  conf->nameserver_list = nameserver_array;

  conf->search_list_size = init->search_list_size;
  const char **search_array = alloc_buffer_alloc_array
    (&buffer, const char *, init->search_list_size);
  conf->search_list = search_array;

  /* Fill the name server list array.  */
  for (size_t i = 0; i < init->nameserver_list_size; ++i)
    if (init->nameserver_list[i]->sa_family == AF_INET)
      {
        struct sockaddr_in *sa = alloc_buffer_alloc
          (&buffer, struct sockaddr_in);
        *sa = *(struct sockaddr_in *) init->nameserver_list[i];
        nameserver_array[i] = (struct sockaddr *) sa;
      }
    else
      {
        struct sockaddr_in6 *sa = alloc_buffer_alloc
          (&buffer, struct sockaddr_in6);
        *sa = *(struct sockaddr_in6 *) init->nameserver_list[i];
        nameserver_array[i] = (struct sockaddr *) sa;
      }

  /* Allocate and fill the sort list array.  */
  {
    conf->sort_list_size = init->sort_list_size;
    struct resolv_sortlist_entry *array = alloc_buffer_alloc_array
      (&buffer, struct resolv_sortlist_entry, init->sort_list_size);
    conf->sort_list = array;
    for (size_t i = 0; i < init->sort_list_size; ++i)
      array[i] = init->sort_list[i];
  }

  /* Fill the search list array.  This must come last because the
     strings are the least aligned part of the allocation.  */
  {
    for (size_t i = 0; i < init->search_list_size; ++i)
      search_array[i] = alloc_buffer_copy_string
        (&buffer, init->search_list[i]);
  }

  assert (!alloc_buffer_has_failed (&buffer));
  return conf;
}

/* Update *RESP from the extended state.  */
static __attribute__ ((nonnull (1, 2), warn_unused_result)) bool
update_from_conf (struct __res_state *resp, const struct resolv_conf *conf)
{
  resp->defdname[0] = '\0';
  resp->pfcode = 0;
  resp->_vcsock = -1;
  resp->_flags = 0;
  resp->ipv6_unavail = false;
  resp->__glibc_unused_qhook = NULL;
  resp->__glibc_unused_rhook = NULL;

  resp->retrans = conf->retrans;
  resp->retry = conf->retry;
  resp->options = conf->options;
  resp->ndots = conf->ndots;

  /* Copy the name server addresses.  */
  {
    resp->nscount = 0;
    resp->_u._ext.nscount = 0;
    size_t nserv = conf->nameserver_list_size;
    if (nserv > MAXNS)
      nserv = MAXNS;
    for (size_t i = 0; i < nserv; i++)
      {
        if (conf->nameserver_list[i]->sa_family == AF_INET)
          {
            resp->nsaddr_list[i]
              = *(struct sockaddr_in *)conf->nameserver_list[i];
            resp->_u._ext.nsaddrs[i] = NULL;
          }
        else
          {
            assert (conf->nameserver_list[i]->sa_family == AF_INET6);
            resp->nsaddr_list[i].sin_family = 0;
            /* Make a defensive copy of the name server address, in
               case the application overwrites it.  */
            struct sockaddr_in6 *sa = malloc (sizeof (*sa));
            if (sa == NULL)
              {
                for (size_t j = 0; j < i; ++j)
                  free (resp->_u._ext.nsaddrs[j]);
                return false;
              }
            *sa = *(struct sockaddr_in6 *)conf->nameserver_list[i];
            resp->_u._ext.nsaddrs[i] = sa;
          }
        resp->_u._ext.nssocks[i] = -1;
      }
    resp->nscount = nserv;
    /* Leave resp->_u._ext.nscount at 0.  res_send.c handles this.  */
  }

  /* Fill in the prefix of the search list.  It is truncated either at
     MAXDNSRCH, or if reps->defdname has insufficient space.  */
  {
    struct alloc_buffer buffer
      = alloc_buffer_create (resp->defdname, sizeof (resp->defdname));
    size_t size = conf->search_list_size;
    size_t i;
    for (i = 0; i < size && i < MAXDNSRCH; ++i)
      {
        resp->dnsrch[i] = alloc_buffer_copy_string
          (&buffer, conf->search_list[i]);
        if (resp->dnsrch[i] == NULL)
          /* No more space in resp->defdname.  Truncate.  */
          break;
      }
    resp->dnsrch[i] = NULL;
  }

  /* Copy the sort list.  */
  {
    size_t nsort = conf->sort_list_size;
    if (nsort > MAXRESOLVSORT)
      nsort = MAXRESOLVSORT;
    for (size_t i = 0; i < nsort; ++i)
      {
        resp->sort_list[i].addr = conf->sort_list[i].addr;
        resp->sort_list[i].mask = conf->sort_list[i].mask;
      }
    resp->nsort = nsort;
  }

  /* The overlapping parts of both configurations should agree after
     initialization.  */
  assert (resolv_conf_matches (resp, conf));
  return true;
}

/* Decrement the configuration object at INDEX and free it if the
   reference counter reaches 0.  *GLOBAL_COPY must be locked and
   remains so.  */
static void
decrement_at_index (struct resolv_conf_global *global_copy, size_t index)
{
  if (index < resolv_conf_array_size (&global_copy->array))
    {
      /* Index found.  */
      uintptr_t *slot = resolv_conf_array_at (&global_copy->array, index);
      /* Check that the slot is not already part of the free list.  */
      if (!(*slot & 1))
        {
          struct resolv_conf *conf = (struct resolv_conf *) *slot;
          conf_decrement (conf);
          /* Put the slot onto the free list.  */
          *slot = global_copy->free_list_start;
          global_copy->free_list_start = (index << 1) | 1;
        }
    }
}

bool
__resolv_conf_attach (struct __res_state *resp, struct resolv_conf *conf)
{
  assert (conf->__refcount > 0);

  struct resolv_conf_global *global_copy = get_locked_global ();
  if (global_copy == NULL)
    return false;

  /* Try to find an unused index in the array.  */
  size_t index;
  {
    if (global_copy->free_list_start & 1)
      {
        /* Unlink from the free list.  */
        index = global_copy->free_list_start >> 1;
        uintptr_t *slot = resolv_conf_array_at (&global_copy->array, index);
        global_copy->free_list_start = *slot;
        assert (global_copy->free_list_start == 0
                || global_copy->free_list_start & 1);
        /* Install the configuration pointer.  */
        *slot = (uintptr_t) conf;
      }
    else
      {
        size_t size = resolv_conf_array_size (&global_copy->array);
        /* No usable index found.  Increase the array size.  */
        resolv_conf_array_add (&global_copy->array, (uintptr_t) conf);
        if (resolv_conf_array_has_failed (&global_copy->array))
          {
            put_locked_global (global_copy);
            __set_errno (ENOMEM);
            return false;
          }
        /* The new array element was added at the end.  */
        index = size;
      }
  }

  /* We have added a new reference to the object.  */
  ++conf->__refcount;
  assert (conf->__refcount > 0);
  put_locked_global (global_copy);

  if (!update_from_conf (resp, conf))
    {
      /* Drop the reference we acquired.  Reacquire the lock.  The
         object has already been allocated, so it cannot be NULL this
         time.  */
      global_copy = get_locked_global ();
      decrement_at_index (global_copy, index);
      put_locked_global (global_copy);
      return false;
    }
  resp->_u._ext.__glibc_extension_index = index ^ INDEX_MAGIC;

  return true;
}

void
__resolv_conf_detach (struct __res_state *resp)
{
  if (atomic_load_relaxed (&global) == NULL)
    /* Detach operation after a shutdown, or without any prior
       attachment.  We cannot free the data (and there might not be
       anything to free anyway).  */
    return;

  struct resolv_conf_global *global_copy = get_locked_global ();
  size_t index = resp->_u._ext.__glibc_extension_index ^ INDEX_MAGIC;
  decrement_at_index (global_copy, index);

  /* Clear the index field, so that accidental reuse is less
     likely.  */
  resp->_u._ext.__glibc_extension_index = 0;

  put_locked_global (global_copy);
}

/* Deallocate the global data.  */
libc_freeres_fn (freeres)
{
  /* No locking because this function is supposed to be called when
     the process has turned single-threaded.  */
  if (global == NULL)
    return;

  if (global->conf_current != NULL)
    {
      conf_decrement (global->conf_current);
      global->conf_current = NULL;
    }

  /* Note that this frees only the array itself.  The pointed-to
     configuration objects should have been deallocated by res_nclose
     and per-thread cleanup functions.  */
  resolv_conf_array_free (&global->array);

  free (global);

  /* Stop potential future __resolv_conf_detach calls from accessing
     deallocated memory.  */
  global = NULL;
}
