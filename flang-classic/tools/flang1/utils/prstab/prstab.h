/**
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */
/** \file
 * \brief LR header files
 *
 *       This file declares all constants and varaiables global to all
 *       LR (prstab) routines, also contains all the routine declarations.
 *
 */

#ifndef PRSTAB_H
#define PRSTAB_H

#define INT int
#define LOGICAL int
#define UINT unsigned int
#define FLOAT float
#define CHAR unsigned char

#define VOID void

#undef min
#undef max
#define min(a, b) ((a) <= (b) ? (a) : (b))
#define max(a, b) ((a) >= (b) ? (a) : (b))

#if defined(_WIN64)
#define SNPRINTF _snprintf
#else
#define SNPRINTF snprintf
#endif

#define MAXSHD 2000
#define MAXSHDP1 2001
#define MAXSHD2P1 4001
#define MAXSST 16000
#define MAXPRDC 10000
#define MAXPROD 4000
#define MAXLST 24000
#define MAXBAS 44000
#define MAXTRN 40000
#define MAXRED 10000
#define MAXSCR 6000
#define MAXHASH 2048
#define MAXLEN 1024

/* Global Declarations */

struct files_s {
  FILE *infile;
  FILE *gramin;
  FILE *tokin;
  FILE *tokout;
  FILE *lstfil;
  FILE *datfil;
  FILE *dbgfil;
  FILE *semfil;
};
extern struct files_s files;

extern INT *scrtch;
extern INT *hashpt;
extern CHAR *linech;
extern char *filnambuf;

struct s4_s {
  INT *item;
  INT *nextt;
};
extern struct s4_s s4;

struct s1_1_s{
  INT *sstore;
  INT *sthead;
};
extern struct s1_1_s s1_1;

struct lstcom_s {
  INT garbag;
  INT lstptr;
};
extern struct lstcom_s lstcom;

struct qcom_s {
  INT qhead;
  INT qtail;
};
extern struct qcom_s qcom;

struct switches_s {
  LOGICAL listsw;
  LOGICAL runosw;
  LOGICAL xrefsw;
  LOGICAL toksw;
  LOGICAL semsw;
  LOGICAL datasw;
  LOGICAL lrltsw;
  LOGICAL dbgsw;
  LOGICAL dbgasw;
  LOGICAL dbgbsw;
  LOGICAL dbgcsw;
  LOGICAL dbgdsw;
  LOGICAL dbgesw;
};
extern struct switches_s switches;

struct adqcom_s {
  INT adequt;
};
extern struct adqcom_s adqcom;

struct g_1_s {
  INT *lftuse;
  INT *rgtuse;
  INT *frsprd;
  INT *nprods;
  INT *prodcn;
  INT *prdind;
  INT *vocab;
  INT nvoc;
  INT numprd;
  INT goal;
  INT nterms;
};
extern struct g_1_s g_1;

struct s_1_s {
  INT nstate;
  INT nxttrn;
  INT nxtred;
  INT ncsets;
  INT listcs;
  INT ifinal;
  INT indbas;
  INT lencsl;
  INT lsets;
  INT *thedpt;
  INT *nullnt;
  INT *basis;
  INT *tran;
  INT *red;
};
extern struct s_1_s s_1;

struct string_1_s {
  INT sstptr;
  INT shdptr;
};
extern struct string_1_s string_1;

struct readcm_s{
  INT linbuf[81];
  INT curchr;
  INT lineno;
  INT fstchr;
};
extern struct readcm_s readcm;

struct ptsem_s {
  INT pnum;
  INT brkflg;
  INT semnum;
};
extern struct ptsem_s ptsem;

/*
 * a local array line is used to constuct a line of text to be written
 * to a file.  Because the output routines are not yet "char *" - based,
 * line is an INT array.  Use a macro to declare line so that it's
 * trivial to change to a char array when the I/O is improved.
 */
#define DECL_LINE(n) INT line[n]

/* Functions Declaration */

/* prstab1 */
INT enter(INT *buff, INT *len);
INT error(const char *msg, INT msglen, INT ibad, INT sym, INT alias);
INT fndnul(void), bildhp(void), finish(void);
INT endbas(INT *iptr);
INT complt(INT *istate, INT *maxset);
INT enque(INT *iptr), deque(INT *iptr);
INT csun(INT *iptr1, INT *iptr2, INT *ich);
INT delcs(INT *iptr);
INT condec(INT number, INT *line, INT istart, INT iend);
INT additl(INT *iarg, INT *lptr, INT *ichnge);
INT addltl(INT *lptr1, INT *lptr2, INT *ichnge);
INT copyl(INT *lptr1, INT *lptr2);
INT chrcmp(INT *iptr1, INT *iptr2, INT *irslt);
INT addbas(INT iptr, INT npr, INT ndot, INT nset);
INT addtrn(INT *ibasis, INT *itran, INT *imax);
INT endtrn(INT *ibasis);
INT addred(INT *ibasis, INT *iprod, INT *icntxt, INT *maxr);
INT endred(INT *ibasis);

/* prstab2 */
INT hashof(INT *i);
INT lenlst(INT *ihead);
INT length(INT *iptr);
INT log76(INT *iarg);
INT lcompr(INT *iptr1, INT *iptr2);
INT movstr(INT iptr, INT *line, INT *istart, INT iend);
INT output(FILE *unit, INT *line, INT nchars);
INT new (INT *iptr), newbas(INT *index), newcs(INT *is, INT *iptr);
INT genthd(void);
INT hepify(INT iptr, INT imax);
INT imtrcs(INT *ibasis, INT *iptr);
INT putsem(INT curlhs, INT *curprd);
INT hlddmp(INT *hold, INT *hldidx);
INT newtrn(INT *ibasis, INT *imax);
INT merge(INT *ibasis, INT *ires, INT *ichng);
INT newred(INT *ibasis, INT *imax);
INT less(INT *iptr1, INT *iptr2);

/* prstab3 */
INT strcomp(INT *iptr1, INT *iptr2, INT *irslt);
INT sortcg(INT *nsets);
INT trnred(INT *ibasis, INT *jmax);
INT rel(INT *iptr);
INT scan(INT *token);

INT init(void), xref(void), tablea(void), gentab(void), rdgram(void),
    fndgol(void), tokmac(void), conect(void), chncsl(void), tableu(void),
    finish(void), ground(void), analyz(void), prntgm(void), sortgm(void),
    pntset(void), zprntk(void);

void intArrayToCharArray(INT *intarr, CHAR *charr, INT count);

#endif
