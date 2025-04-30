/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */
/** \file
 * \brief Path name manipulation utilities
 *
 *  Implement the legacy path name utility functions.
 */

#include "legacy-util-api.h"
#include <stddef.h>
#include <string.h>
#ifndef _WIN64
#include <unistd.h> /* access() */
#endif

void
basenam(const char *orig_path, const char *optional_suffix, char *basename)
{
  const char *fn = strrchr(orig_path, '/');
  size_t length;

  if (fn == NULL)
    fn = orig_path;
  else
    ++fn;
  length = strlen(fn);

  if (optional_suffix != NULL) {
    size_t suffix_length = strlen(optional_suffix);
    if (suffix_length >= length &&
        strcmp(fn + length - suffix_length, optional_suffix) == 0)
      length -= suffix_length;
  }

  memcpy(basename, fn, length);
  basename[length] = '\0';
}

void
dirnam(const char *orig_path, char *dirname)
{
  const char *slash = strrchr(orig_path, '/');
  if (slash == NULL) {
    strcpy(dirname, "./");
  } else if (slash == orig_path) {
    strcpy(dirname, "/");
  } else {
    size_t length = slash - orig_path;
    memcpy(dirname, orig_path, length);
    dirname[length] = '\0';
  }
}

int
fndpath(const char *target, char *path, size_t max_length, const char *dirlist)
{
  size_t target_length = target ? strlen(target) : 0;
  if (target_length == 0)
    return -1;

  /* The legacy fndpath supplies a default dirlist of '.', which seems
   * unsafe.
   */
  if (dirlist == NULL || !*dirlist)
    dirlist = ".";

  while (*dirlist != '\0') {
    const char *end = strchr(dirlist, ':');
    size_t component_length = end ? end - dirlist : strlen(dirlist);
    while (component_length > 1 &&
           dirlist[component_length - 1] == '/') {
      /* ignore trailing '/', unless it's the only character */
      --component_length;
    }
    if (component_length > 0 &&
        component_length + 1 /* '/' */ + target_length + 1 <= max_length) {
      char *p = path;
      memcpy(p, dirlist, component_length);
      p += component_length;
      *p++ = '/';
      memcpy(p, target, target_length);
      p[target_length] = '\0';
      if (access(path, 0) == 0)
        return 0; /* path exists */
    }
    if (end == NULL)
      break;
    dirlist = end + 1;
  }

  return -1;
}

char *
mkperm(char *pattern, const char *oldext, const char *newext)
{
  size_t length = strlen(pattern), ext_length = strlen(oldext);
  if (ext_length <= length) {
    char *at = pattern + length - ext_length;
    if (memcmp(at, oldext, ext_length) == 0)
      strcpy(at, newext);
  }
  return pattern;
}
