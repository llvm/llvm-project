/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
 * \brief Module to support assembly file annotation ( -Manno switch).
 */

#include "gbldefs.h"
#include "error.h"
#include "global.h"
#include "symtab.h" /* prerequisite for ili.h */
#include "ili.h"
#include "asm_anno.h"
#include "fih.h"

extern char *comment_char; /* assem.c */

/** \brief Assembly block annotation */
typedef struct ANNO_S {
  struct ANNO_S *next;
  int lineno;
  int module;     /** \brief module number, 0 .. n-1 */
  long fileloc;   /** \brief Byte offset from begin of file */
  int bihd;       /** \brief bihd associated with this line */
  int tmp;        /** \brief index holder */
  const char *filename; /** \brief Name of file for this lineno */
  FILE *fd;       /** \brief file descriptor associated with this lineno */
} ANNO;

extern void annomod_init(void);
static ANNO *annomod_initx(ANNO *ahead);
static int qs_anno(const void *p1, const void *p2);
extern void annomod_asm(int blkno);
extern void anno_blkcnt(int blkno);
extern void annomod_end(void);
static char *get_anno_line(FILE *fptr);
static void put_anno_line(char *p);
static int find_idx(ANNO *, int, int, int);
static void emit_str(const char *p, int lptr);
static void dmp_all_anno(ANNO *, FILE *, int);
static void dmp_anno(ANNO *, FILE *);

#define ALEN 256
static struct {
  FILE *fp;    /* Current file pointer */
  FILE *fname; /* Current file name */
  int curlin;  /* Current line number */
  INT curpos;  /* Current file position.  Good after each output */
  char buf[ALEN + 4];
} lanno;

static FILE *fanno = NULL;
static ANNO *amod = NULL;
static ANNO *ahead = NULL;
static int modcnt = -1;

#undef SOURCE_FILE
#define SOURCE_FILE gbl.file_name

extern void
annomod_init(void)
{
  modcnt++;
  amod = annomod_initx(ahead);
  if (amod != NULL && modcnt == 0)
    ahead = amod;
}

/** \brief Initiate annotation of the assembly file.
 *
 * This is called before code is generated.  Augment the current annotation
 * chain with a sorted list of the new linenumber records (ANNO) for this
 * module.
 *
 * \param ahead head of annotation list
 * \return new head of annotation list
 */
static ANNO *
annomod_initx(ANNO *ahead)
{
  static ANNO *static_ahead;
  static int lastcnt = 0;
  static INT file_pos = 0;

  ANNO *aptr, *aptr1, *anptr;
  int cnt = 0;
  int cnt1 = 0;
  int bihd, i, j;
  int old_line_num;
  int first_bih;
  char *p;

  /*  *************************************************** *
   *  loop thru all of the ili blocks and setup line info
   *  *************************************************** */

  /* ********
   * step 1:
   *      open source file.
   *      count the number of non-zero blocks
   *      allocate and zero first ANNO node.
   *      chain to old ANNO chain.
   */

  if (modcnt == 0) {
    if ((fanno = fopen(SOURCE_FILE, "rb")) == NULL) {
      error((error_code_t)2, ERR_Warning, 0, SOURCE_FILE, CNULL);
      return ahead;
    }
  } else if (fanno == NULL)
    return ahead; /* don't process if src not open */

  for (bihd = BIH_NEXT(0), cnt = 0; bihd; bihd = BIH_NEXT(bihd)) {
    if (BIH_LINENO(bihd))
      cnt++;
  }

  if (cnt == 0)
    return NULL;

  if (cnt > lastcnt) { /* get a new list after freeing old list */
    if (static_ahead)
      FREE(static_ahead);
    NEW(static_ahead, ANNO, cnt);
    BZERO(static_ahead, ANNO, cnt);
    lastcnt = cnt;
  } else /* Just zero out the current chain */
    BZERO(static_ahead, ANNO, lastcnt);

  anptr = aptr = ahead = static_ahead;

  /* anptr should point to tail of list */

  /* ********
   * step 2:
   *      walk BIH list and chain non-zero lineno blocks
   *           into ANNO chain.  There are 'cnt' non-0 lineno
   *           blocks.
   */
  aptr1 = aptr;
  first_bih = -1;

  for (bihd = BIH_NEXT(0), cnt1 = 0; bihd; bihd = BIH_NEXT(bihd)) {

    if (BIH_LINENO(bihd)) {
      if (first_bih == -1)
        first_bih = bihd;

      /* per flyspray 15590, check if this is inline block, get line number
         from fihb.stg_base[].lineno. Inliner has kept a good record of
         line numbers.   Otherwise, an error lines too long will occurs
         because p = NULL, a return from get_anno_line() when line number
         in BIH_LINENO(bihd) is larger than line number in compile file.
         Note: Include does not yet work correctly especially for c++ .
       */
      if (FIH_INL(BIH_FINDEX(bihd))) {
        /* Go to main source file of this */
        for (j = BIH_FINDEX(bihd); j != 0; j = FIH_PARENT(j)) {
          if (FIH_PARENT(j) == 1) {
            aptr1->lineno = FIH_LINENO(j);
            break;
          }
        }
      } else {
        aptr1->lineno = BIH_LINENO(bihd);
      }

      aptr1->bihd = bihd;
      aptr1->module = modcnt;
      aptr1->tmp = cnt1;
      aptr1->next = NULL;
      aptr1++;
      cnt1++;
    }
  }

  assert(cnt == cnt1, "annomod_init: cnt != cnt1 :", cnt1, ERR_Severe);

/* ********
 * step 3:
 *      Sort the newest list into ascending lineno.
 */

#if DEBUG
  if (DBGBIT(16, 4)) {
    fprintf(gbl.dbgfil,
            "annomod_init: Pre QSORT Linear dump of anno recs: cnt: %d\n", cnt);
    dmp_all_anno(ahead, gbl.dbgfil, cnt);
  }
#endif

  qsort((char *)aptr, cnt, sizeof(ANNO), qs_anno);

#if DEBUG
  if (DBGBIT(16, 4)) {
    fprintf(gbl.dbgfil,
            "annomod_init: Post QSORT Linear dump of anno recs: cnt: %d\n",
            cnt);
    dmp_all_anno(ahead, gbl.dbgfil, cnt);
  }
#endif
  /* ********
   * step 3a:
   *      Now go back and link the ANNO records according to index occurence.
   */
  {
    int idx, i;

    aptr = ahead;
    aptr1 = NULL;
    for (i = 0, idx = 0; i < cnt; i++) {

      idx = find_idx(aptr, cnt, i, idx);
      if (i == 0)
        ahead = &aptr[idx];

      assert(idx <= cnt && idx >= 0, "outside anno loop: i:", i, ERR_Informational);

      if (aptr1 == NULL)
        aptr1 = &aptr[idx];

      aptr1->next = &aptr[idx];
      aptr1 = &aptr[idx];
      aptr1->next = NULL;
    }
  }

#if DEBUG
  if (DBGBIT(16, 4)) {
    fprintf(gbl.dbgfil, "annomod_init3: Linked Dump of anno recs: cnt: %d\n",
            cnt);
    dmp_all_anno(ahead, gbl.dbgfil, 0);
  }
#endif

  /* ********:
   * step 4:
   *      At this point:
   *          o  ahead points a NULL terminated linked list;
   *          o  aptr points at the latest contiguous list of sorted
   *             (by ascending lineno) ANNO records.
   *          o  The 'next' link chains the new ANNO chain in the order
   *             of their bih occurrence.
   *          o  cnt is the number of linenumbers in aptr list.
   *             aptr[0..cnt-1]
   *
   *      Now, get file locations for each line by streaming thru
   *           source file and recording file position for each line
   *           that corresponds to a BIH LINENO.
   *      When completed, restore file and curlin to the current
   *           positions.
   */

  if (lanno.curlin < 1)
    lanno.curlin = 1;
  file_pos = ftell(fanno);
  old_line_num = lanno.curlin;

  for (i = 0; i < cnt; i++) {
    int l;

    l = aptr[i].lineno;

    /* per flyspray 15590, keep line the same is it is include file */
    if (FIH_INC(BIH_FINDEX(aptr[i].bihd))) {
      l = lanno.curlin;
    }

    while (lanno.curlin <= l) {
      if (lanno.curlin < l) {
        p = get_anno_line(fanno);
        if (p == NULL) {
          flg.anno = 0;
          return NULL;
        }
      }

      if (lanno.curlin == l) {
        if (lanno.curpos < 0) {
#if DEBUG
          interr("annomod_init(): cannot ftell into source for line:",
                 lanno.curlin, ERR_Warning);
#endif
          return NULL;
        }
        aptr[i].fileloc = (long)lanno.curpos;
        aptr[i].fd = fanno;
        aptr[i].filename = SOURCE_FILE; /* Not alloc'd for now */
        break;
      }
    }
  }

  fseek(fanno, file_pos, 0);
  lanno.curlin = old_line_num;
  lanno.curpos = file_pos = ftell(fanno);

#if DEBUG
  if (DBGBIT(16, 4)) {
    fprintf(gbl.dbgfil, "annomod_init4: Dump of Linked anno recs: cnt:%d\n",
            cnt);
    dmp_all_anno(ahead, gbl.dbgfil, 0);
    fprintf(gbl.dbgfil, "annomod_init4: Dump of linear anno recs: cnt: %d\n",
            cnt);
    dmp_all_anno(aptr, gbl.dbgfil, cnt);
  }
#endif

  return aptr;
} /* endroutine annomod_init */

/** \brief Quicksort comparison routine for annotations.
 *
 * Compares source file line numbers that the blocks correspond to.
 *
 * \param p1 first block
 * \param p2 second block
 *
 * Note: parameters are void* because that is required by qsort.
 */
static int
qs_anno(const void *p1, const void *p2)
{
  if (((const ANNO *)p1)->lineno < ((const ANNO *)p2)->lineno)
    return (-1);
  if (((const ANNO *)p1)->lineno > ((const ANNO *)p2)->lineno)
    return (1);
#ifdef HOST_MSDOS
  return 0;
#else
  return (-1); /* compare equality as being 'less than' */
#endif
} /* endroutine qs_anno */

/* ----------------------------------------------------------- */

/* To annotate the PFO block count in the asm file. */
extern void
anno_blkcnt(int blkno)
{
  fprintf(gbl.asmfil, "%s block %d execution count: %lf\n", comment_char, blkno,
          BIH_BLKCNT(blkno));
}

/* ----------------------------------------------------------- */

/** \brief Annotate assembly file
 *
 * Just seek to file position indicated and output from there to next file
 * position or eof.
 */
extern void
annomod_asm(int blkno)
{
  int l;
  char *p;

  if (BIH_LINENO(blkno) == 0 || amod == NULL)
    return;

  /*	amod should pt to record for the code that begins with
      this bih.   annomod_asm should output code from this
      node to the *(amod+1) if amod->next is not null. */

  if (amod->bihd != blkno) {
#if DEBUG
    interr("Inconsistent anno records for blkno: ", blkno, ERR_Informational);
#endif
    flg.anno = 0;
    return;
  }

again:
  lanno.curpos = ftell(fanno);
  if (lanno.curpos < 0) {
    interr("annomod_asm(): cannot ftell into source for module:", modcnt, ERR_Warning);
    amod = NULL;
    return;
  }

  if (lanno.curpos != amod->fileloc) {
    if (fseek(fanno, (long)amod->fileloc, 0)) {
      interr("annomod_asm(): cannot fseek into source for lineno:",
             amod->lineno, ERR_Warning);
      amod = NULL;
      return;
    }

    goto again;
  }

  lanno.curlin = amod->lineno;

  if (amod->next == NULL)
    l = lanno.curlin + 1; /* Only go to the next line if last block */
  else {
    if ((amod + 1)->lineno == 0)
      l = lanno.curlin + 1;
    else
      l = (amod + 1)->lineno;
  }

  emit_str("\n", 0);
  while (lanno.curlin < l) {
    p = get_anno_line(fanno);
    if (p == NULL)
      break;
    put_anno_line(p);
    if (lanno.curlin == l)
      break;
  }

  amod = amod->next;
} /* endroutine annomod_asm() */

/* ----------------------------------------------------- */

/** \brief Finish annotate assembly file for this module.
 *
 * Just seek to the current file position.
 */
extern void
annomod_end(void)
{
  if (amod == NULL)
    return;

  lanno.curpos = ftell(fanno);
  assert(lanno.curpos >= 0, "annomod_end: cannot ftell into source for module:",
         modcnt, ERR_Warning);

  amod = NULL;
}

/* ----------------------------------------------------- */

/** \brief Get a line from the file
 *
 * \param fptr file pointer
 *
 * \return ptr to static file buffer if enough room or return NULL if problem
 * encountered.
 */
static char *
get_anno_line(FILE *fptr)
{
#define BLEN 80
  char *p, *p1;
  char lbuf[BLEN + 2];

  lanno.buf[0] = '\n';
  lanno.buf[ALEN - 1] = lanno.buf[ALEN - 2] = lanno.buf[ALEN - 3] = ' ';
  lanno.buf[1] = '\0';

  p = fgets(lanno.buf, ALEN - 2, fptr);
  if (p && (int)strlen(lanno.buf) >= ALEN - 3 && lanno.buf[ALEN - 4] != '\n') {
    strcat(lanno.buf, "...\n");
    while (1) {
      p1 = fgets(lbuf, BLEN - 2, fptr);
      if (p1 && (int)strlen(lbuf) >= BLEN - 3 && lbuf[BLEN - 4] != '\n')
        continue;
      else
        break;
    }
  }

  if (p)
    lanno.curlin++;

  lanno.curpos = ftell(fptr);
  return p;

} /* endroutine get_anno_line */

/** \brief Output an annotated line
 *
 * \param p assembly buffer to print
 */
static void
put_anno_line(char *p)
{
  if (!p)
    return;
  emit_str(comment_char, 0);
  emit_str(" ", 0);
  emit_str(p, 0);
}

static int
find_idx(ANNO *p, int cnt, /* # of records in linear list */
         int idx,          /* idx to search for in idx field */
         int last_idx      /* last idx found */
         )
{
  int i;

  for (i = last_idx; i < cnt; i++) {
    if (p[i].tmp == idx)
      return i;
  }

  /* if can't find forward, start from beginning and research */
  for (i = 0; i < cnt; i++) {
    if (p[i].tmp == idx)
      return i;
  }

  return -1;
} /* endroutine find_idx */

/** \brief Emit the string to the output file
 *
 * \param p is the string to output
 * \param lptr is the label sptr used for delay br filling.
 */
static void
emit_str(const char *p, int lptr)
{
  if (p == NULL && lptr != 0)
    return; /* Called from sched_blkinit */
  assert(p, "emit_str: NULL ptr to be output", 0, ERR_Informational);

  if (gbl.asmfil == NULL)
    gbl.asmfil = stdout;

  if (!XBIT(96, 2) || lptr >= 0 || strcmp(p, "\twnop") != 0)
    fputs(p, gbl.asmfil);

} /* endroutine emit_str */

/* -------------------------------------------------------- */

#if DEBUG
static void
dmp_all_anno(ANNO *panno, FILE *fp, int flag)
{
  static const char *msg[] = {" linked list dump ", " linear list dump "};
  ANNO *p;
  int cnt = 0;

  fprintf(fp, "dmp_all_anno: ********** ptr:%p %s flag:%d\n", (void *)panno,
          msg[flag != 0], flag);
  if (flag)
    goto dmplist;
  p = panno;
  while (p) {
    dmp_anno(p, fp);
    p = p->next;
    cnt++;
  }
  goto RTN;

dmplist:
  p = panno;
  for (cnt = 0; cnt < flag; cnt++)
    dmp_anno(p++, fp);

RTN:
  fprintf(fp, "\nend dmp_all_anno: ***************************** cnt:%d\n\n",
          cnt);
  fflush(fp);
} /* endroutine dmp_all_anno */

static void
dmp_anno(ANNO *panno, FILE *fp)
{
  fprintf(fp, "dmp_anno: ");
  fprintf(fp,
          "addr:%p lineno:%d module:%d fileloc:%lu bihd:%d next:%p  idx :%d\n",
          (void *)panno, panno->lineno, panno->module, panno->fileloc,
          panno->bihd, (void *)panno->next, panno->tmp);
  fflush(fp);
}
#endif /* DEBUG */
