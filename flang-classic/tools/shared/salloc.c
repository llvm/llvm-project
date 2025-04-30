/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
   \file
   \brief compiler storage allocation utility routines.
 */

#include "gbldefs.h"
#include "global.h"
#include "error.h"

#define SIZE                                             \
  2000          /* size in bytes of each block of memory \
                 * obtained from malloc (NEW). */
#define ANUM 30 /* number of different areas supported, 0...ANUM-1 */

typedef char *PTR;
#undef PTRSZ
#undef ALIGN
#define PTRSZ sizeof(PTR)
#define ALIGN(o) (((o) + (PTRSZ - 1)) & (~(PTRSZ - 1)))

static char *areap[ANUM] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                            NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                            NULL, NULL, NULL, NULL, NULL, NULL, NULL};
static int avail[ANUM];

/**
   \param area is an area
   \param size is the size in bytes of item to be allocated.
 */
char *
getitem(int area, int size)
{
  char *p;

  assert(area >= 0 && area < ANUM, "getitem: bad area", area, ERR_Fatal);
  size = ALIGN(size); /* round up to multiple of PTRSZ */

  if (areap[area] == NULL) {
    int sz = SIZE;
    if (size + PTRSZ > SIZE)
      sz = size + PTRSZ;
    NEW(p, char, sz);
    areap[area] = p;
    if (p == NULL)
      interr("getitem: no mem avail", area, ERR_Fatal);
    *((PTR *)p) = NULL;
    avail[area] = PTRSZ;
  } else if (avail[area] + size > SIZE) {
    int sz = SIZE;
    if (size + PTRSZ > SIZE)
      sz = size + PTRSZ;
    NEW(p, char, sz);
    if (p == NULL)
      interr("getitem: no mem avail", area, ERR_Fatal);
    *((PTR *)p) = areap[area];
    areap[area] = p;
    avail[area] = PTRSZ;
  }
  p = areap[area] + avail[area];
  avail[area] += size;
#if DEBUG
  if (DBGBIT(0, 0x20000)) {
    char *q, cc;
    int s;
    /* fill with junk */
    cc = 0xa6;
    for (s = size, q = p; s; --s, ++q) {
      *q = cc;
      cc = (cc << 1) | (cc >> 7);
    }
  }
#endif
  return p;
}

void
freearea(int area)
{
  char *p, *q;

  assert(area >= 0 && area < ANUM, "freearea: bad area", area, ERR_Fatal);
  for (p = areap[area]; p != NULL; p = q) {
    q = *((PTR *)p); /* get next before free!!! */
    FREE(p);
  }
  areap[area] = NULL;
}

#if DEBUG
void
reportarea(int full)
{
  int area;
  for (area = 0; area < ANUM; ++area) {
    if (areap[area] == NULL) {
      if (full)
        fprintf(gbl.dbgfil, "area[%2d] is empty\n", area);
    } else {
      char *p, *q;
      int total = 0;
      for (p = areap[area]; p != NULL; p = q) {
        q = *((PTR *)p); /* get next before free!!! */
        total += SIZE;
      }
      fprintf(gbl.dbgfil, "area[%2d] >= %d bytes with %d free\n", area, total,
              avail[area]);
    }
  }
}
#endif

/*
 * Functions to allow referring to pointers returned by getitem as ints
 * for the purpose of storing pointers in the int fields of the symbol
 * table and other structures.
 */
static struct {
  void **base;
  int size;
  int avail;
} tbl = {0, 0, 0};

/**
   \brief stores the pointer in the pointer table and returns its
   index.
 */
int
put_getitem_p(void *p)
{
  int r;

  /* NULL <==> 0 */
  if (p == NULL)
    return 0;
  if (tbl.size == 0) {
    tbl.size = 100;
    NEW(tbl.base, void *, tbl.size);
    r = 1;
    tbl.avail = 2;
  } else {
    r = tbl.avail++;
    NEED(tbl.avail, tbl.base, void *, tbl.size, tbl.size + 100);
  }
  tbl.base[r] = p;
  return r;
}

/**
   \brief returns the pointer stored at index i.
 */
void *
get_getitem_p(int i)
{
  /* 0 <==> NULL */
  if (i == 0)
    return NULL;
#if DEBUG
  assert(tbl.size, "get_getitem_p: null tbl", i, ERR_unused);
  assert(i > 0 && i < tbl.avail, "get_getitem_p: i out of range", i, ERR_unused);
#endif
  return tbl.base[i];
}

/**
   \brief frees the table of pointers.  NOTE: it's safer to just call
   this function at the end of compilation to allow a mix of pointers
   which are only live during a phase and those which are live during
   the entire compile such as the pointers locating items in area 8.
 */
void
free_getitem_p(void)
{
  if (tbl.size) {
    free(tbl.base);
    tbl.size = 0;
  }
}
