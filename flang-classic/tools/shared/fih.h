/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
 * \brief File Information Header (FIH)
 */

typedef struct {
  const char *dirname;  /**< directory name */
  const char *filename; /**< file name (only) */
  const char *fullname; /**< full file name */
  const char *funcname; /**< function name */
  void *ccffinfo; /**< opaque pointer used for CCFF info */
  int functag;    /**< integer function tag; ilm index of the function header */
  int parent;     /**< file into which this is inlined or included */
  int flags;      /**< file flags */
  int lineno;     /**< line number of parent where included or inlined */
  int srcline;    /**< line number in filename */
  int level;      /**< count of number of parents, plus 1 */
  int next;       /**< next inlined/included routine into same routine */
  int child;      /**< first inlined/included routine into this routine */
  int dirindex;   /**< used for debug line tables */
  int findex; /**< For bottom-up auto-inlining, mapping between IFIH and FIH */
  int funcsize; /**< For bottom-up auto-inlining, passing info to ccff_info */
} FIH;

typedef struct {
  FIH *stg_base;
  int stg_size;
  int stg_avail;
  int currfindex, currftag; /**< findex/ftag for this block */
  int nextfindex, nextftag; /**< findex/ftag for next block created */
  int saved_avail;
} FIHB;

#define FIH_INCLUDED 1
#define FIH_INLINED 2
#define FIH_IPAINLINED 4
#define FIH_SPLIT 8
#define FIH_ALTCODE 0x10
#define FIH_CCFF 0x20
#define FIH_FDONE 0x40
#define FIH_FINCDONE 0x80
#define FIH_DO_CCFF 0x100
/* File introduced by USE stmt in Fortran */
#define FIH_USE_MOD 0x200

#define FIH_DIRNAME(i) fihb.stg_base[i].dirname
#define FIH_FILENAME(i) fihb.stg_base[i].filename
#define FIH_NAME(i) fihb.stg_base[i].fullname
#define FIH_FULLNAME(i) fihb.stg_base[i].fullname
#define FIH_FUNCNAME(i) fihb.stg_base[i].funcname
#define FIH_FUNCTAG(i) fihb.stg_base[i].functag
#define FIH_PARENT(i) fihb.stg_base[i].parent
#define FIH_FLAGS(i) fihb.stg_base[i].flags
#define FIH_LINENO(i) fihb.stg_base[i].lineno
#define FIH_SRCLINE(i) fihb.stg_base[i].srcline
#define FIH_LEVEL(i) fihb.stg_base[i].level
#define FIH_NEXT(i) fihb.stg_base[i].next
#define FIH_CHILD(i) fihb.stg_base[i].child
#define FIH_INC(i) (fihb.stg_base[i].flags & FIH_INCLUDED)
#define FIH_INL(i) (fihb.stg_base[i].flags & FIH_INLINED)
#define FIH_IPAINL(i) (fihb.stg_base[i].flags & FIH_IPAINLINED)
#define FIH_CLEARFLAG(i, f) (fihb.stg_base[i].flags &= ~(f))
#define FIH_SETFLAG(i, f) (fihb.stg_base[i].flags |= (f))
#define FIH_CHECKFLAG(i, f) (fihb.stg_base[i].flags & (f))
#define FIH_CLEARDONE(i) (fihb.stg_base[i].flags &= ~FIH_FDONE)
#define FIH_SETDONE(i) (fihb.stg_base[i].flags |= FIH_FDONE)
#define FIH_DONE(i) (fihb.stg_base[i].flags & FIH_FDONE)
#define FIH_CLEARINCDONE(i) (fihb.stg_base[i].flags &= ~FIH_FINCDONE)
#define FIH_SETINCDONE(i) (fihb.stg_base[i].flags |= FIH_FINCDONE)
#define FIH_INCDONE(i) (fihb.stg_base[i].flags & FIH_FINCDONE)
#define FIH_USEMOD(i) (fihb.stg_base[i].flags & FIH_USE_MOD)
#define FIH_CCFFINFO(i) fihb.stg_base[i].ccffinfo

extern FIHB fihb;

/*             IFIH (inline file information header)   */

#define IFIH_DIRNAME(i) ifihb.stg_base[i].dirname
#define IFIH_FILENAME(i) ifihb.stg_base[i].filename
#define IFIH_NAME(i) ifihb.stg_base[i].fullname
#define IFIH_FULLNAME(i) ifihb.stg_base[i].fullname
#define IFIH_FUNCNAME(i) ifihb.stg_base[i].funcname
#define IFIH_FUNCTAG(i) ifihb.stg_base[i].functag
#define IFIH_PARENT(i) ifihb.stg_base[i].parent
#define IFIH_FLAGS(i) ifihb.stg_base[i].flags
#define IFIH_LINENO(i) ifihb.stg_base[i].lineno
#define IFIH_SRCLINE(i) ifihb.stg_base[i].srcline
#define IFIH_LEVEL(i) ifihb.stg_base[i].level
#define IFIH_NEXT(i) ifihb.stg_base[i].next
#define IFIH_CHILD(i) ifihb.stg_base[i].child
#define IFIH_INC(i) (ifihb.stg_base[i].flags & FIH_INCLUDED)
#define IFIH_INL(i) (ifihb.stg_base[i].flags & FIH_INLINED)
#define IFIH_IPAINL(i) (ifihb.stg_base[i].flags & FIH_IPAINLINED)
#define IFIH_CLEARFLAG(i, f) (ifihb.stg_base[i].flags &= ~(f))
#define IFIH_SETFLAG(i, f) (ifihb.stg_base[i].flags |= (f))
#define IFIH_CHECKFLAG(i, f) (ifihb.stg_base[i].flags & (f))
#define IFIH_CLEARDONE(i) (ifihb.stg_base[i].flags &= ~FIH_FDONE)
#define IFIH_SETDONE(i) (ifihb.stg_base[i].flags |= FIH_FDONE)
#define IFIH_DONE(i) (ifihb.stg_base[i].flags & FIH_FDONE)
#define IFIH_CLEARINCDONE(i) (ifihb.stg_base[i].flags &= ~FIH_FINCDONE)
#define IFIH_SETINCDONE(i) (ifihb.stg_base[i].flags |= FIH_FINCDONE)
#define IFIH_INCDONE(i) (ifihb.stg_base[i].flags & FIH_FINCDONE)
#define IFIH_CCFFINFO(i) ifihb.stg_base[i].ccffinfo
#define IFIH_FINDEX(i) ifihb.stg_base[i].findex
#define IFIH_FUNCSIZE(i) ifihb.stg_base[i].funcsize

extern FIHB ifihb;

/* Moved from ccffinfo.c */
typedef struct ccff_message {
  struct ccff_message *next;
  struct ccff_message *msgchild;
  const char *msgid;
  char *message;
  char *varname;
  char *funcname;
  struct ccff_argument *args;
  int msgtype, fihx, lineno, order, suborder, seq, combine;
} MESSAGE;

typedef struct ccff_argument {
  struct ccff_argument *next;
  char *argstring;
  char *argvalue;
} ARGUMENT;
