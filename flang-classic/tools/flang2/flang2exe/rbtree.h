/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef RBTREE_H_
#define RBTREE_H_

typedef struct rbtree_root_struct rbroot;
typedef struct rbtree_struct *rbtree;
struct rbtree_root_struct {
  rbtree root;
  size_t blocksize, freesize;
  char *datablocklist, *datafree;
};

/* red-black tree */
struct rbtree_struct {
  rbtree left, right, parent;
  int color;
  int data[];
};

#define RBRED 1
#define RBBLK 2
#define RBCOLOR 3

#define ISBLACK(rb) (rb == NULL || (rb->color == RBBLK))
#define ISRED(rb) (rb != NULL && (rb->color == RBRED))
#define SETBLACK(rb)   \
  if (rb) {            \
    rb->color = RBBLK; \
  }
#define SETRED(rb)     \
  if (rb) {            \
    rb->color = RBRED; \
  }
#define COPYCOLOR(rb, fb) rb->color = fb->color

#define ALN(b, aln) ((((b) + (aln)-1)) & (~((aln)-1)))

typedef int (*rb_compare)(void *, void *);
typedef int (*rb_walk_proc)(rbtree, void *userdata);

/**
   \brief ...
 */
int rb_walk(rbroot *T, rb_walk_proc proc, void *userdata);

/**
   \brief ...
 */
rbtree rb_find(rbroot *T, void *data, rb_compare compare);

/**
   \brief ...
 */
rbtree rb_insert(rbroot *T, size_t datasize, void *data, rb_compare compare);

/**
   \brief ...
 */
void rb_free(rbroot *T);


#endif // RBTREE_H_
