/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
 *  \file gbldefs.h
 *  \brief miscellaneous macros and prototypes used by Fortran front-end
 *  Function prototypes and host/version dependent miscellaneous macros used
 *  throughout the Fortran front-end.
 */

#ifndef FE_GBLDEFS_H
#define FE_GBLDEFS_H

#include <stdint.h>
#include "universal.h"
#include "platform.h"
#include "pgifeat.h"
#include <scutil.h>

#define NEW_ARG_PARSER

/* enable negative zero */
#define USEFNEG 1

#define FE90
#define PGFTN
#define PGHPF
#define SCFTN
#define SCC_SCFTN
#define PGC_PGFTN

#ifndef DEBUG
#define DEBUG 1
#endif
#define DBGBIT(n, m) (flg.dbg[n] & m)

#define GBL_SIZE_T_FORMAT "zu"

#define XBIT(n, m) (flg.x[n] & m)
#define F77OUTPUT XBIT(49, 0x80)
/* This x-bit controls the insertion of scope labels. On by default. */
#define XBIT_USE_SCOPE_LABELS !XBIT(198, 0x40000)

/* used to determine what kind of runtime descriptor to use */
/* flag to test if using WINNT calling conventions: */
#define WINNT_CALL XBIT(121, 0x10000)
#define WINNT_CREF XBIT(121, 0x40000)
#define WINNT_NOMIXEDSTRLEN XBIT(121, 0x80000)

#define CNULL ((char *)0)
#undef uf
#define uf(s) error(0, 1, gbl.lineno, "Unimplemented feature", s)

/* Fortran Standard max identifier length (used with -Mstandard) */
#define STANDARD_MAXIDLEN 63

/* internal max identifier length - allow room for suffix like $sd and $td$ft */
#define MAXIDLEN 163

/*  should replace local MAX_FNAME_LENs with: */
#define MAX_FILENAME_LEN 2048

/* maximum number of array subscripts */
#define MAXSUBS 7
#define MAXDIMS 7

typedef int8_t INT8;
typedef int16_t INT16;
typedef uint16_t UINT16;

/* define a host type which represents 'size_t' for array extents. */
#define ISZ_T BIGINT
#define UISZ_T BIGUINT
#define ISZ_PF BIGIPFSZ
#define ISZ_2_INT64(s, r) bgitoi64(s, r)
#define INT64_2_ISZ(s, r) r = i64tobgi(s)

#define BITS_PER_BYTE 8

/* ETLS/TLS threadprivate features */

typedef int LOGICAL;
#undef TRUE
#define TRUE 1
#undef FALSE
#define FALSE 0

/*
 * Define truth values for Fortran.  The negate operation is dependent
 * upon the values chosen.
 */
#define SCFTN_TRUE gbl.ftn_true
#define SCFTN_FALSE 0
#define SCFTN_NEGATE(n) ((~(n)) & SCFTN_TRUE)

#define BCOPY(p, q, dt, n) memcpy(p, q, ((UINT)sizeof(dt) * (n)))
#define BZERO(p, dt, n) memset((p), 0, ((UINT)sizeof(dt) * (n)))
#define FREE(p) sccfree((char *)p), p = NULL

#if DEBUG
#define NEW(p, dt, n)                                    \
  if (1) {                                               \
    p = (dt *)sccalloc((BIGUINT64)((INT)sizeof(dt) * (n)));   \
    if (DBGBIT(7, 2))                                    \
      bjunk((char *)(p), (BIGUINT64)((INT)sizeof(dt) * (n))); \
  } else
#define NEED(n, p, dt, size, newsize)                                        \
  if (n > size) {                                                            \
    p = (dt *)sccrelal((char *)p, ((BIGUINT64)((newsize) * (INT)sizeof(dt))));    \
    if (DBGBIT(7, 2))                                                        \
      bjunk((char *)(p + size), (BIGUINT64)((newsize - size) * (INT)sizeof(dt))); \
    size = newsize;                                                          \
  } else

#else
#define NEW(p, dt, n) p = (dt *)sccalloc((BIGUINT64)((INT)sizeof(dt) * (n)))
#define NEED(n, p, dt, size, newsize)                                     \
  if (n > size) {                                                         \
    p = (dt *)sccrelal((char *)p, ((BIGUINT64)((newsize) * (INT)sizeof(dt)))); \
    size = newsize;                                                       \
  } else
#endif

#define NEEDB(n, p, dt, size, newsize)                                    \
  if (n > size) {                                                         \
    p = (dt *)sccrelal((char *)p, ((BIGUINT64)((newsize) * (INT)sizeof(dt)))); \
    BZERO(p + size, dt, newsize - size);                                  \
    size = newsize;                                                       \
  } else

#include "sharedefs.h"

#define CLRFPERR() (Fperr = FPE_NOERR)
/* NOTE :fperror prints an error message and then sets Fperr to FPE_NOERR    */
/*       it returns zero if Fperr was equal to FPE_NOERR , otherwise nonzero */
#define CHKFPERR() (Fperr != FPE_NOERR ? fperror() : 0)

/*  declare external functions which are used globally:  */

void finish(void); /* from main.c    */

/* mall.c */
char *sccalloc(BIGUINT64);
void sccfree(char *);
char *sccrelal(char *, BIGUINT64);
#ifdef DEBUG
void bjunk(void *p, BIGUINT64 n);
#endif

char *getitem(int, int); /* from salloc.c: */
#define GETITEM(area, type) (type *) getitem(area, sizeof(type))
#define GETITEMS(area, type, n) (type *) getitem(area, (n) * sizeof(type))
void freearea(int);
int put_getitem_p(void *);
void *get_getitem_p(int);
void free_getitem_p(void);

char *mkfname(const char *, const char *, const char *); /* from miscutil.c: */
bool is_xflag_bit(int);
void set_xflag(int, INT);
void set_yflag(int, INT);
void list_init(FILE *); /* listing.c: */
void list_line(const char *); /* listing.c */
void list_page(void);   /* listing.c */

ISZ_T get_bss_addr(void); /* from assem.c */
void assemble(void);
void assemble_init(void);
void assemble_end(void);
ISZ_T set_bss_addr(ISZ_T);
ISZ_T pad_cmn_mem(int, ISZ_T, int *);
void fix_equiv_locals(int loc_list, ISZ_T loc_addr);
void fix_equiv_statics(int loc_list, ISZ_T loc_addr, LOGICAL dinitflg);

void dbg_print_ast(int, FILE *);
void deferred_to_pointer(void);
void dumpfgraph();
void dumploops();
void dumpnmes();
void dumpuses(void);
void dumpdefs(void);

/* dump.c */
void dcommons(void);
void dumpdts(void);
void dstds(int std1, int std2);
void dsyms(int l, int u);
void dstdps(int std1, int std2);
void dumpasts(void);
void dumpstdtrees(void);
void dumpshapes(void);
void dumplists(void);
void dsstds(int std1, int std2);

void dump_ast_tree(int i); /* ast.c */

void dumpaccrout(void); /* accroutine.c */

void reportarea(int full);            /* salloc.c */
void lower_ipa_info(FILE *lowerfile); /* ipa.c */

void ipa_init(void);              /* ipa.c */
void ipa_startfunc(int currfunc); /* ipa.c */
void ipa_header1(int currfunc);   /* ipa.c */
void ipa_header2(int currfunc);   /* ipa.c */
void ipa(void);                   /* ipa.c */
long IPA_sstride(int sptr);       /* ipa.c */
long IPA_pstride(int sptr);       /* ipa.c */

void ipasave_endfunc(void); /* ipasave.c */
void fill_ipasym(void);     /* ipasave.c */

void ipa_import_highpoint(void);   /* interf.c */
void ipa_import(void);             /*interf.c */
void ipa_set_vestigial_host(void); /* interf.c */
void import_module_print(void);    /* interf.c */
void import_host_subprogram(FILE *fd, const char *file_name, int oldsymavl,
                            int oldastavl, int olddtyavl, int modbase,
                            int moddiff);              /* interf.c */
void import_fini(void);                                /* interf.c */
void ipa_import_highpoint(void);                       /* exterf.c */
void set_tag(void);                                    /* exterf.c */
void ipa_export_endmodule(void);                       /* exterf.c */
void ipa_export_highpoint(void);                       /* exterf.c */
void ipa_export_endcontained(void);                    /* exterf.c */
void exterf_init(void);                                /* exterf.c */
void exterf_init_host(void);                           /* exterf.c */
void export_public_module(int module, int exceptlist); /* exterf.c */
void export_inline(FILE *export_fd, char *export_name,
                   char *file_name); /* exterf.c */

LOGICAL is_initialized(int *bv, int nme); /* flow.c */

int IPA_isnoconflict(int sptr); /* main.c */
void reinit(void);              /* main.c */

int can_map_initsym(int old_firstosym);         /* symtab.c */
int map_initsym(int oldsym, int old_firstosym); /* symtab.c */
int hashcon(INT *value, int dtype, int sptr);

void end_contained(void);  /* main.c */
void set_exitcode(int ec); /* main.c */

void eliminate_unused_variables(int which); /* bblock.c */
void bblock_init(void);                     /* bblock.c */
int bblock(void);                           /* bblock.c */
void merge_commons(void);                   /* bblock.c */
void renumber_lines(void);                  /* bblock.c */

void lower_constructor(void);             /* lower.c */
void lower_pstride_info(FILE *lowerfile); /* pstride.c */

void parse_init(void);

void fpp(void); /* fpp.c */

#ifdef _WIN64
#define snprintf _snprintf
#endif

#endif /* FE_GBLDEFS_H */
