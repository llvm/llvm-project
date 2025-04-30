/* Monitoring file descriptor usage.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <support/check.h>
#include <support/descriptors.h>
#include <support/support.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <xunistd.h>

struct procfs_descriptor
{
  int fd;
  char *link_target;
  dev_t dev;
  ino64_t ino;
};

/* Used with qsort.  */
static int
descriptor_compare (const void *l, const void *r)
{
  const struct procfs_descriptor *left = l;
  const struct procfs_descriptor *right = r;
  /* Cannot overflow due to limited file descriptor range.  */
  return left->fd - right->fd;
}

#define DYNARRAY_STRUCT descriptor_list
#define DYNARRAY_ELEMENT struct procfs_descriptor
#define DYNARRAY_PREFIX descriptor_list_
#define DYNARRAY_ELEMENT_FREE(e) free ((e)->link_target)
#define DYNARRAY_INITIAL_SIZE 0
#include <malloc/dynarray-skeleton.c>

struct support_descriptors
{
  struct descriptor_list list;
};

struct support_descriptors *
support_descriptors_list (void)
{
  struct support_descriptors *result = xmalloc (sizeof (*result));
  descriptor_list_init (&result->list);

  DIR *fds = opendir ("/proc/self/fd");
  if (fds == NULL)
    FAIL_EXIT1 ("opendir (\"/proc/self/fd\"): %m");

  while (true)
    {
      errno = 0;
      struct dirent64 *e = readdir64 (fds);
      if (e == NULL)
        {
          if (errno != 0)
            FAIL_EXIT1 ("readdir: %m");
          break;
        }

      if (e->d_name[0] == '.')
        continue;

      char *endptr;
      long int fd = strtol (e->d_name, &endptr, 10);
      if (*endptr != '\0' || fd < 0 || fd > INT_MAX)
        FAIL_EXIT1 ("readdir: invalid file descriptor name: /proc/self/fd/%s",
                    e->d_name);

      /* Skip the descriptor which is used to enumerate the
         descriptors.  */
      if (fd == dirfd (fds))
        continue;

      char *target;
      {
        char *path = xasprintf ("/proc/self/fd/%ld", fd);
        target = xreadlink (path);
        free (path);
      }
      struct stat64 st;
      if (fstat64 (fd, &st) != 0)
        FAIL_EXIT1 ("readdir: fstat64 (%ld) failed: %m", fd);

      struct procfs_descriptor *item = descriptor_list_emplace (&result->list);
      if (item == NULL)
        FAIL_EXIT1 ("descriptor_list_emplace: %m");
      item->fd = fd;
      item->link_target = target;
      item->dev = st.st_dev;
      item->ino = st.st_ino;
    }

  closedir (fds);

  /* Perform a merge join between descrs and current.  This assumes
     that the arrays are sorted by file descriptor.  */

  qsort (descriptor_list_begin (&result->list),
         descriptor_list_size (&result->list),
         sizeof (struct procfs_descriptor), descriptor_compare);

  return result;
}

void
support_descriptors_free (struct support_descriptors *descrs)
{
  descriptor_list_free (&descrs->list);
  free (descrs);
}

void
support_descriptors_dump (struct support_descriptors *descrs,
                          const char *prefix, FILE *fp)
{
  struct procfs_descriptor *end = descriptor_list_end (&descrs->list);
  for (struct procfs_descriptor *d = descriptor_list_begin (&descrs->list);
       d != end; ++d)
    {
      char *quoted = support_quote_string (d->link_target);
      fprintf (fp, "%s%d: target=\"%s\" major=%lld minor=%lld ino=%lld\n",
               prefix, d->fd, quoted,
               (long long int) major (d->dev),
               (long long int) minor (d->dev),
               (long long int) d->ino);
      free (quoted);
    }
}

static void
dump_mismatch (bool *first,
               struct support_descriptors *descrs,
               struct support_descriptors *current)
{
  if (*first)
    *first = false;
  else
    return;

  puts ("error: Differences found in descriptor set");
  puts ("Reference descriptor set:");
  support_descriptors_dump (descrs, "  ", stdout);
  puts ("Current descriptor set:");
  support_descriptors_dump (current, "  ", stdout);
  puts ("Differences:");
}

static void
report_closed_descriptor (bool *first,
                          struct support_descriptors *descrs,
                          struct support_descriptors *current,
                          struct procfs_descriptor *left)
{
  support_record_failure ();
  dump_mismatch (first, descrs, current);
  printf ("error: descriptor %d was closed\n", left->fd);
}

static void
report_opened_descriptor (bool *first,
                          struct support_descriptors *descrs,
                          struct support_descriptors *current,
                          struct procfs_descriptor *right)
{
  support_record_failure ();
  dump_mismatch (first, descrs, current);
  char *quoted = support_quote_string (right->link_target);
  printf ("error: descriptor %d was opened (\"%s\")\n", right->fd, quoted);
  free (quoted);
}

void
support_descriptors_check (struct support_descriptors *descrs)
{
  struct support_descriptors *current = support_descriptors_list ();

  /* Perform a merge join between descrs and current.  This assumes
     that the arrays are sorted by file descriptor.  */

  struct procfs_descriptor *left = descriptor_list_begin (&descrs->list);
  struct procfs_descriptor *left_end = descriptor_list_end (&descrs->list);
  struct procfs_descriptor *right = descriptor_list_begin (&current->list);
  struct procfs_descriptor *right_end = descriptor_list_end (&current->list);

  bool first = true;
  while (left != left_end && right != right_end)
    {
      if (left->fd == right->fd)
        {
          if (strcmp (left->link_target, right->link_target) != 0)
            {
              support_record_failure ();
              char *left_quoted = support_quote_string (left->link_target);
              char *right_quoted = support_quote_string (right->link_target);
              dump_mismatch (&first, descrs, current);
              printf ("error: descriptor %d changed from \"%s\" to \"%s\"\n",
                      left->fd, left_quoted, right_quoted);
              free (left_quoted);
              free (right_quoted);
            }
          if (left->dev != right->dev)
            {
              support_record_failure ();
              dump_mismatch (&first, descrs, current);
              printf ("error: descriptor %d changed device"
                      " from %lld:%lld to %lld:%lld\n",
                      left->fd,
                      (long long int) major (left->dev),
                      (long long int) minor (left->dev),
                      (long long int) major (right->dev),
                      (long long int) minor (right->dev));
            }
          if (left->ino != right->ino)
            {
              support_record_failure ();
              dump_mismatch (&first, descrs, current);
              printf ("error: descriptor %d changed ino from %lld to %lld\n",
                      left->fd,
                      (long long int) left->ino, (long long int) right->ino);
            }
          ++left;
          ++right;
        }
      else if (left->fd < right->fd)
        {
          /* Gap on the right.  */
          report_closed_descriptor (&first, descrs, current, left);
          ++left;
        }
      else
        {
          /* Gap on the left.  */
          TEST_VERIFY_EXIT (left->fd > right->fd);
          report_opened_descriptor (&first, descrs, current, right);
          ++right;
        }
    }

  while (left != left_end)
    {
      /* Closed descriptors (more descriptors on the left).  */
      report_closed_descriptor (&first, descrs, current, left);
      ++left;
    }

  while (right != right_end)
    {
      /* Opened descriptors (more descriptors on the right).  */
      report_opened_descriptor (&first, descrs, current, right);
      ++right;
    }

  support_descriptors_free (current);
}
