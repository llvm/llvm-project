/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/*
 * Red-Black tree routines
 * Given a pointer to the root structure for a tree
 * we can efficiently find, add, delete tree elements
 */

#include "gbldefs.h"
#include "rbtree.h"

/*
 * red-black rotation routines
 */
static void
rb_left_rotate(rbroot *T, rbtree x)
{
  rbtree y, yleft, xparent;
  y = x->right;
  yleft = y->left;
  xparent = x->parent;
  x->right = yleft;
  y->left = x;
  x->parent = y;
  if (yleft != NULL)
    yleft->parent = x;
  y->parent = xparent;
  if (xparent == NULL) {
    T->root = y;
  } else if (x == xparent->left) {
    xparent->left = y;
  } else {
    xparent->right = y;
  }
} /* rb_left_rotate */

static void
rb_right_rotate(rbroot *T, rbtree x)
{
  rbtree y, yright, xparent;
  y = x->left;
  yright = y->right;
  xparent = x->parent;
  x->left = yright;
  y->right = x;
  x->parent = y;
  if (yright != NULL)
    yright->parent = x;
  y->parent = xparent;
  if (xparent == NULL) {
    T->root = y;
  } else if (x == xparent->right) {
    xparent->right = y;
  } else {
    xparent->left = y;
  }
} /* rb_right_rotate */

/*
 * allocate a new rbtree entry in the T tree
 */
static rbtree
rb_new(rbroot *T, size_t datasize, void *data)
{
  rbtree x;
  size_t bytes;

  bytes = sizeof(struct rbtree_struct) + datasize;
  bytes = ALN(bytes, 8);

  if (!T->datafree || T->freesize < bytes) {
    char *dblock;
    if (T->blocksize == 0)
      T->blocksize = 4088;
    if (T->blocksize < bytes)
      T->blocksize = 2 * bytes;
    dblock = (char *)sccalloc(T->blocksize + sizeof(void *));
    *(char **)dblock = T->datablocklist;
    T->datablocklist = dblock;
    T->datafree = dblock + sizeof(void *);
    T->freesize = T->blocksize;
  }
  x = (rbtree)T->datafree;
  T->datafree += bytes;
  T->freesize -= bytes;
  if (T->freesize == 0)
    T->datafree = NULL;

  x->left = NULL;
  x->right = NULL;
  x->parent = NULL;
  x->color = 0;
  memcpy(x->data, data, datasize);
  return x;
} /* rb_new */

/*
 * insert a new entry into the tree
 */
static rbtree
_rbtree_insert(rbroot *T, size_t datasize, void *data, rb_compare compare)
{
  rbtree x, y, z;
  y = NULL;
  x = T->root;
  while (x != NULL) {
    y = x;
    if (compare(data, x->data) < 0) {
      x = x->left;
    } else {
      x = x->right;
    }
  }
  z = rb_new(T, datasize, data);
  if (y == NULL) {
    T->root = z;
    z->parent = NULL;
  } else if (compare(z->data, y->data) < 0) {
    y->left = z;
    z->parent = y;
  } else {
    y->right = z;
    z->parent = y;
  }
  return z;
} /* _rbtree_insert */

/*
 * insert a new entry into the tree T, then balance the tree
 */
rbtree
rb_insert(rbroot *T, size_t datasize, void *data, rb_compare compare)
{
  rbtree x, xx;
  xx = _rbtree_insert(T, datasize, data, compare);
  SETRED(xx);
  x = xx;
  while (x != T->root && ISRED(x->parent)) {
    rbtree y;
    if (x->parent == x->parent->parent->left) {
      y = x->parent->parent->right;
      if (y && ISRED(y)) {
        SETBLACK(x->parent);
        SETBLACK(y);
        SETRED(x->parent->parent);
        x = x->parent->parent;
      } else {
        if (x == x->parent->right) {
          x = x->parent;
          rb_left_rotate(T, x);
        }
        SETBLACK(x->parent);
        SETRED(x->parent->parent);
        rb_right_rotate(T, x->parent->parent);
      }
    } else {
      y = x->parent->parent->left;
      if (y && ISRED(y)) {
        SETBLACK(x->parent);
        SETBLACK(y);
        SETRED(x->parent->parent);
        x = x->parent->parent;
      } else {
        if (x == x->parent->left) {
          x = x->parent;
          rb_right_rotate(T, x);
        }
        SETBLACK(x->parent);
        SETRED(x->parent->parent);
        rb_left_rotate(T, x->parent->parent);
      }
    }
  }
  SETBLACK(T->root);
  return xx;
} /* rb_insert */

/*
 * find an entry in the tree
 */
rbtree
rb_find(rbroot *T, void *data, rb_compare compare)
{
  rbtree x;
  int r;
  x = T->root;
  while (x) {
    r = compare(data, x->data);
    if (r == 0)
      return x;
    if (r < 0) {
      x = x->left;
    } else {
      x = x->right;
    }
  }
  return x;
} /* rb_find */

/*
 * walk the tree in sorted order (in-order traversal)
 */
static int
_rb_walk(rbtree x, rb_walk_proc proc, void *userdata)
{
  int r;
  if (x->left) {
    r = _rb_walk(x->left, proc, userdata);
    if (r)
      return r;
  }
  r = (*proc)(x, userdata);
  if (r)
    return r;
  if (x->right) {
    r = _rb_walk(x->right, proc, userdata);
    if (r)
      return r;
  }
  return 0;
} /* _rb_walk */

int
rb_walk(rbroot *T, rb_walk_proc proc, void *userdata)
{
  int r = 0;
  if (T && T->root)
    r = _rb_walk(T->root, proc, userdata);
  return r;
} /* rb_walk */

/*
 * free the allocated space for a tree
 */
void
rb_free(rbroot *T)
{
  char *p, *q;
  for (p = T->datablocklist; p; p = q) {
    q = *((char **)p);
    sccfree(p);
  }
  T->datablocklist = NULL;
} /* rb_free */
