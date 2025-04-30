/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
    \brief Semantic analyzer routines which process IO statements.
 */

#include "gbldefs.h"
#include "global.h"
#include "gramsm.h"
#include "gramtk.h"
#include "error.h"
#include "symtab.h"
#include "symutl.h"
#include "dtypeutl.h"
#include "semant.h"
#include "scan.h"
#include "dinit.h"
#include "semstk.h"
#include "ast.h"
#include "feddesc.h"
#include "rte.h"
#include "rtlRtns.h"

/* generate asts for calling an io routine; must be performed in the
 * the following order:
 *     begin_io_call()
 *     add_io_arg()        [zero or more calls]
 *     end_io_call()	   [optional assignment of function]
 */

static struct { /* record info for begin_io_call - end_io_call */
  int ast;
  int ast_type;
  int std;
} io_call;

/*-------- define data structures and macros local to this file: --------*/

/* define macros used to access table the I/O parameter table, "pt".
 * All of the macros that can be used in the INQUIRE statement are at
 * the beginning (their values range from 0 .. PT_LAST_INQUIRE_VAL).
 * These are in the order specified by the fio_inquire routine in the
 * HPF Execution Environment spec. and only include those which are passed
 * as arguments.  Note that ERR is not passed to fio_inquire.
 *
 * N O T E:  The static array of struct, pt, is initialized the names of
 *           these parameters.  If changes are made to the PT_ macros,
 *           R E M E M B E R  to change ptname.
 */
#define PT_UNIT 0
#define PT_FILE 1
#define PT_IOSTAT 2
#define PT_EXIST 3
#define PT_OPENED 4
#define PT_NUMBER 5
#define PT_NAMED 6
#define PT_NAME 7
#define PT_ACCESS 8
#define PT_SEQUENTIAL 9
#define PT_DIRECT 10
#define PT_FORM 11
#define PT_FORMATTED 12
#define PT_UNFORMATTED 13
#define PT_RECL 14
#define PT_NEXTREC 15
#define PT_BLANK 16
#define PT_POSITION 17
#define PT_ACTION 18
#define PT_READ 19
#define PT_WRITE 20
#define PT_READWRITE 21
#define PT_DELIM 22
#define PT_PAD 23
#define PT_ID 24
#define PT_PENDING 25
#define PT_POS 26
#define PT_SIZE 27
#define PT_ASYNCHRONOUS 28
#define PT_DECIMAL 29
#define PT_ENCODING 30
#define PT_SIGN 31
#define PT_STREAM 32
#define PT_ROUND 33
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * end of INQUIRE parameters: make sure that PT_LAST_INQUIRE_VAL is set
 * to the last inquire value.
 * Values <= including PT_LAST_INQUIRE_VALf95 are f95 inquire specifiers
 * values > PT_LAST_INQUIRE_VALf95 and <= PT_LAST_INQUIRE_VAL are the new
 * f2003 inquire specifiers.
 *
 * PT_IOLENGTH is only present since certain utility functions, such as
 * chk_var, require a 'pt' argument.
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#define PT_DISPOSE 34
#define PT_END 35
#define PT_ERR 36
#define PT_FMT 37
#define PT_NML 38
#define PT_REC 39
#define PT_STATUS 40
#define PT_ADVANCE 41
#define PT_EOR 42
#define PT_IOLENGTH 43
#define PT_CONVERT 44
#define PT_SHARED 45
#define PT_IOMSG 46
#define PT_NEWUNIT 47

#define PT_LAST_INQUIRE_VALf95 PT_PAD
#define PT_LAST_INQUIRE_VAL 33
#define PT_MAXV 47

/*
 * define bit flag for each I/O statement. Used for checking
 * illegal cases of I/O keyword specifiers.
 */
#define BT_ACCEPT 0x00001
#define BT_BKSPACE 0x00002
#define BT_CLOSE 0x00004
#define BT_DECODE 0x00008
#define BT_ENCODE 0x00010
#define BT_ENDFILE 0x00020
#define BT_INQUIRE 0x00040
#define BT_OPEN 0x00080
#define BT_PRINT 0x00100
#define BT_READ 0x00200
#define BT_REWIND 0x00400
#define BT_WRITE 0x00800
#define BT_WAIT 0x01000
#define BT_FLUSH 0x02000

/* define macros for the edit descriptors */
#define PUT(n) (_put((INT)(n), DT_INT))

/* define format type macros:  */
typedef enum {
  FT_UNFORMATTED = 1,
  FT_LIST_DIRECTED = 2,
  FT_ENCODED = 3,
  FT_CHARACTER = 4,
  FT_NML = 5,
  FT_FMTSTR = 6,
  FT_LAST = 7
} FormatType;

/* Array indexed by [is_read][<format type from above>] */
static int functype[2][FT_LAST] = {
    {DT_IO_FWRITE, DT_IO_UWRITE, DT_IO_FWRITE, DT_IO_FWRITE, DT_IO_FWRITE,
     DT_IO_FWRITE, DT_IO_FWRITE},
    {DT_IO_FREAD, DT_IO_UREAD, DT_IO_FREAD, DT_IO_FREAD, DT_IO_FREAD,
     DT_IO_FREAD, DT_IO_FREAD}};

/* miscellaneous macros:  */

#define DEFAULT_UNIT (is_read ? 5 : 6)
#define IOERR(n) errsev(n)
#define IOERR2(n, s) error(n, 3, gbl.lineno, s, CNULL)
#define ERR170(s) error(170, 2, gbl.lineno, s, CNULL)
#define ERR204(s1, s2) error(204, 3, gbl.lineno, s1, s2)

#define PTV(d) pt[d].val
#define PT_CHECK(a, b) \
  if (PTV(a) == 0)     \
  PTV(a) = b
#define PTS(a) pt[a].set
#define PT_SET(a) PTS(a) = 1
#define PTVARREF(a) pt[a].varref
#define PT_VARREF(a, v) PTVARREF(a) = (v)
#define PTTMPUSED(a) pt[a].tmp_in_use
#define PT_TMPUSED(a, v) (pt[a].tmp_in_use = (v))
#define PTARG(a) \
  (PTVARREF(a) && PTVARREF(a) != 1 ? PT_TMPUSED(a, 1), PTV(a) : PTV(a))
#define PTNAME(a) pt[a].name
#define PTSTMT(a) pt[a].stmt
#define UNIT_CHECK       \
  if (PTV(PT_UNIT) == 0) \
  IOERR(200)

/* typedef and macros to access io lists: */

typedef struct iol {
  struct iol *next;
  SST *element;
  DOINFO *doinfo;
  int id;
  int l_std; /* list of stds added after the left paren of an
              * implied-do is parsed */
} IOL;
#define IE_EXPR 0
#define IE_DOBEGIN 1
#define IE_DOEND 2
#define IE_OPTDO 3

/* local data */

static struct pt_tag { /* parameter table for I/O statements */
  int val;             /* pointer to AST of I/O keyword specifier, except
                        * for a keyword specifier for a label, in which
                        * case val is the sptr of the label.
                        */
  int set;
  int varref;     /* zero     ==> io specifier is not a variable,
                   * non-zero ==> io specifier is a variable reference,
                   *              where
                   *              the value 1 means that the variable is
                   *              being used directly, and any other
                   *              non zero value implies that a temp
                   *              value is being used in which case
                   *                1) val contains the ast referencing
                   *                   the temp
                   *                2) and varref (this field) contains
                   *                   the ast referencing the original
                   *                   variable.
                   */
  int tmp_in_use; /* 1==>a tmp is being used in the current call (see above) */
  const char *name;
  int stmt; /* I/O stmts which may use parameter/keyword */
} pt[PT_MAXV + 1] = {
    {0, 0, 0, 0, "UNIT",
     BT_BKSPACE | BT_CLOSE | BT_ENDFILE | BT_INQUIRE | BT_OPEN | BT_READ |
         BT_REWIND | BT_WRITE | BT_WAIT | BT_FLUSH},
    {0, 0, 0, 0, "FILE", BT_INQUIRE | BT_OPEN},
    {0, 0, 0, 0, "IOSTAT",
     BT_BKSPACE | BT_CLOSE | BT_DECODE | BT_ENCODE | BT_ENDFILE | BT_INQUIRE |
         BT_OPEN | BT_READ | BT_REWIND | BT_WRITE | BT_WAIT | BT_FLUSH},
    {0, 0, 0, 0, "EXIST", BT_INQUIRE},
    {0, 0, 0, 0, "OPENED", BT_INQUIRE},
    {0, 0, 0, 0, "NUMBER", BT_INQUIRE},
    {0, 0, 0, 0, "NAMED", BT_INQUIRE},
    {0, 0, 0, 0, "NAME", BT_INQUIRE | BT_OPEN},
    {0, 0, 0, 0, "ACCESS", BT_INQUIRE | BT_OPEN},
    {0, 0, 0, 0, "SEQUENTIAL", BT_INQUIRE},
    {0, 0, 0, 0, "DIRECT", BT_INQUIRE},
    {0, 0, 0, 0, "FORM", BT_INQUIRE | BT_OPEN},
    {0, 0, 0, 0, "FORMATTED", BT_INQUIRE},
    {0, 0, 0, 0, "UNFORMATTED", BT_INQUIRE},
    {0, 0, 0, 0, "RECL", BT_INQUIRE | BT_OPEN},
    {0, 0, 0, 0, "NEXTREC", BT_INQUIRE},
    {0, 0, 0, 0, "BLANK", BT_INQUIRE | BT_OPEN | BT_READ},
    {0, 0, 0, 0, "POSITION", BT_INQUIRE | BT_OPEN},
    {0, 0, 0, 0, "ACTION", BT_INQUIRE | BT_OPEN},
    {0, 0, 0, 0, "READ", BT_INQUIRE},
    {0, 0, 0, 0, "WRITE", BT_INQUIRE},
    {0, 0, 0, 0, "READWRITE", BT_INQUIRE},
    {0, 0, 0, 0, "DELIM", BT_INQUIRE | BT_OPEN | BT_WRITE},
    {0, 0, 0, 0, "PAD", BT_INQUIRE | BT_OPEN | BT_READ},
    {0, 0, 0, 0, "ID", BT_INQUIRE | BT_READ | BT_WRITE | BT_WAIT},
    {0, 0, 0, 0, "PENDING", BT_INQUIRE},
    {0, 0, 0, 0, "POS", BT_INQUIRE | BT_READ | BT_WRITE},
    {0, 0, 0, 0, "SIZE", BT_INQUIRE | BT_READ | BT_WRITE},
    {0, 0, 0, 0, "ASYNCHRONOUS", BT_INQUIRE | BT_OPEN | BT_READ | BT_WRITE},
    {0, 0, 0, 0, "DECIMAL", BT_INQUIRE | BT_OPEN | BT_READ | BT_WRITE},
    {0, 0, 0, 0, "ENCODING", BT_INQUIRE | BT_OPEN},
    {0, 0, 0, 0, "SIGN", BT_INQUIRE | BT_OPEN | BT_WRITE},
    {0, 0, 0, 0, "STREAM", BT_INQUIRE},
    {0, 0, 0, 0, "ROUND", BT_INQUIRE | BT_OPEN | BT_READ | BT_WRITE},
    {0, 0, 0, 0, "DISPOSE", BT_CLOSE | BT_OPEN},
    {0, 0, 0, 0, "END", BT_READ | BT_WAIT},
    {0, 0, 0, 0, "ERR",
     BT_BKSPACE | BT_CLOSE | BT_DECODE | BT_ENCODE | BT_ENDFILE | BT_INQUIRE |
         BT_OPEN | BT_READ | BT_REWIND | BT_WRITE | BT_WAIT | BT_FLUSH},
    {0, 0, 0, 0, "FMT", BT_READ | BT_WRITE},
    {0, 0, 0, 0, "NML", BT_READ | BT_WRITE},
    {0, 0, 0, 0, "REC", BT_READ | BT_WRITE},
    {0, 0, 0, 0, "STATUS", BT_CLOSE | BT_OPEN},
    {0, 0, 0, 0, "ADVANCE", BT_READ | BT_WRITE},
    {0, 0, 0, 0, "EOR", BT_READ | BT_WAIT},
    {0, 0, 0, 0, "IOLENGTH", BT_INQUIRE},
    {0, 0, 0, 0, "CONVERT", BT_OPEN},
    {0, 0, 0, 0, "SHARED", BT_OPEN},
    {0, 0, 0, 0, "IOMSG",
     BT_BKSPACE | BT_CLOSE | BT_DECODE | BT_ENCODE | BT_ENDFILE | BT_INQUIRE |
         BT_OPEN | BT_READ | BT_REWIND | BT_WRITE | BT_WAIT | BT_FLUSH},
    {0, 0, 0, 0, "NEWUNIT", BT_OPEN},
};
static FormatType fmttyp;     /* formatted or unformatted I/O */
static int nml_group;         /* sptr to namelist group ident */
static int intern_array;      /* AST of array section used as internal unit*/
static int intern_tmp;        /* AST of temp replacing 'intern_array' */
static LOGICAL intern;        /* internal I/O flag */
static LOGICAL external_io;   /* set for any external I/O statement */
static LOGICAL nondevice_io;  /* set for any I/O statement not allowed in CUDA
                                 device code */
static LOGICAL is_read;       /* read flag */
static LOGICAL no_rw;         /* statement DOES NOT read or write */
static LOGICAL print_star;    /* PRINT * */
static LOGICAL unit_star;     /* unit is * */
static int fmt_is_var;        /* FMT specifier is a variable */
static int fasize;            /* size in units of 32-bit words of the format
                               * list */
static int rescan;            /* where formatted I/O begins rescanning if
                               * there are more I/O items than edit
                               * descriptors */
static int lastrpt;           /* marks where the last repeat count seen while
                               * processing edit descriptors is in the
                               * format list */
static int edit_state;        /* state transition value for edit descriptor
                               * processing -- used for checking "delimiter
                               * conformance" */
static int last_edit;         /* last edit descriptor seen */
static int fmt_length;        /* ast representing length of an unencoded
                               * format string (FT_CHARACTER).
                               */
static int filename_type = 0; /* TY_CHAR or TY_NCHAR */
static int iolist;            /* ASTLI list for io items in read/write */
static LOGICAL noparens;      /* no parens enclosing control list */
static LOGICAL open03;        /* any f2003 open specfiers */
static LOGICAL rw03;          /* any f2003 read/write specifiers */
static INT bitv;              /* bit vector for IOSTAT, END, and ERR: */
#define BITV_IOSTAT 0x01
#define BITV_ERR 0x02
#define BITV_END 0x04
#define BITV_EOR 0x08
#define BITV_IOMSG 0x10
/*
 * the following values are not seen by the runtime and are just used to
 * check for illegal combinations of I/O specifers.
 */
#define BITV_SIZE 0x020
#define BITV_ADVANCE 0x040

#define BYTE_SWAPPED_IO (XBIT(125, 0x2) != 0)
#define LARGE_ARRAY (XBIT(68, 1) != 0)

/* Selectable unfomatted i/o routines:
 * Indexed by:
 * 1) LARGE_ARRAY_IDX
 *
 * 2) 0 - normal unformatted i/o (file format matches native byte order)
 *    1 - byte-swapped unf i/o (file format is reversed from native byte order).
 *    2 - byte read/write
 *
 * '-x 125 2' is used to select one of the first two entries; aggregate
 * unformatted i/o selects the third entry.
 */

#define BYTE_SWAPPED_IO_IDX ((BYTE_SWAPPED_IO) ? 1 : 0)
#define LARGE_ARRAY_IDX ((LARGE_ARRAY) ? 1 : 0)
#define BYTE_RW_IDX 2

typedef struct {
  FtnRtlEnum init;
  FtnRtlEnum read;
  FtnRtlEnum write;
  FtnRtlEnum end;
} UnformattedRtns;

UnformattedRtns unf_nm[][3] = {{
                                   {RTE_f90io_unf_init, RTE_f90io_unf_reada,
                                    RTE_f90io_unf_writea, RTE_f90io_unf_end},
                                   {RTE_f90io_usw_init, RTE_f90io_usw_reada,
                                    RTE_f90io_usw_writea, RTE_f90io_usw_end},
                                   {RTE_f90io_unf_init, RTE_f90io_byte_reada,
                                    RTE_f90io_byte_writea, RTE_f90io_unf_end},
                               },
                               {
                                   {RTE_f90io_unf_init, RTE_f90io_unf_reada,
                                    RTE_f90io_unf_writea, RTE_f90io_unf_end},
                                   {RTE_f90io_usw_init, RTE_f90io_usw_reada,
                                    RTE_f90io_usw_writea, RTE_f90io_usw_end},
                                   {RTE_f90io_unf_init, RTE_f90io_byte_read64a,
                                    RTE_f90io_byte_write64a, RTE_f90io_unf_end},
                               }};

UnformattedRtns array_unf_nm[][3] = {
    {
        {RTE_f90io_unf_init, RTE_f90io_unf_read_aa, RTE_f90io_unf_write_aa,
         RTE_f90io_unf_end},
        {RTE_f90io_usw_init, RTE_f90io_usw_read_aa, RTE_f90io_usw_write_aa,
         RTE_f90io_usw_end},
        {RTE_f90io_unf_init, RTE_f90io_byte_reada, RTE_f90io_byte_writea,
         RTE_f90io_unf_end},
    },
    {
        {RTE_f90io_unf_init, RTE_f90io_unf_read64_aa, RTE_f90io_unf_write64_aa,
         RTE_f90io_unf_end},
        {RTE_f90io_usw_init, RTE_f90io_usw_read64_aa, RTE_f90io_usw_write64_aa,
         RTE_f90io_usw_end},
        {RTE_f90io_unf_init, RTE_f90io_byte_read64a, RTE_f90io_byte_write64a,
         RTE_f90io_unf_end},
    }};

/* Selectable formatted init routines:
 * 0 - external i/o
 *     0 - encoded format or not present
 *     1 - fmt specifier is a variable
 * 1 - internal i/o
 *     0 - encoded format or not present
 *     1 - fmt specifier is a variable
 */
static struct {
  FtnRtlEnum read[2];
  FtnRtlEnum write[2];
} fmt_init[] = {{
                    {RTE_f90io_fmtr_init2003a, RTE_f90io_fmtr_initv2003a},
                    {RTE_f90io_fmtw_inita, RTE_f90io_fmtw_initva},
                },
                {
                    {RTE_f90io_fmtr_intern_inita, RTE_f90io_fmtr_intern_initva},
                    {RTE_f90io_fmtw_intern_inita, RTE_f90io_fmtw_intern_initva},
                }};

/* Init routines for DECODE/ENCODE */
static struct {
  FtnRtlEnum read[2];
  FtnRtlEnum write[2];
} fmt_inite = {
    {RTE_f90io_fmtr_intern_inite, RTE_f90io_fmtr_intern_initev},
    {RTE_f90io_fmtw_intern_inite, RTE_f90io_fmtw_intern_initev},
};

/* Selectable prefixes for the I/O routines (craft, non-craft) passed
 * to mkfunc_name();
 */
static int io_sc;

static int set_io_sc(void);
static int copy_replic_sect_to_tmp(int);
static void copy_back_to_replic_sect(int, int);
static void fix_iostat(void);
static int add_cgoto(int);
static int end_or_err(int, int, int);
static int fio_end_err(int, int);
static void chk_expr(SST *, int, int);
static void chk_var(SST *, int, int);
static void gen_spec_item_tmp(SST *, int, int);
static void chk_unitid(SST *);
static void chk_fmtid(SST *);
static void chk_iospec(void);
static void put_edit(int);
static void put_ffield(SST *);
static void kwd_errchk(int);
static int get_fmt_array(int, int);
static int ast_ioret(void);
static int mk_iofunc(FtnRtlEnum, int, int);
static int mk_hpfiofunc(FtnRtlEnum, int, int);
static LOGICAL need_descriptor_ast(int);
static void rw_array(int, int, int, FtnRtlEnum);
static void get_derived_iolptrs(SST *, int, SST *);
static void gen_derived_io(int, FtnRtlEnum, int);
static void gen_lastval(DOINFO *);
static int misc_io_checks(const char *);
static void iomsg_check(void);
static void newunit_check(void);
static void put_vlist(SST *);
static FtnRtlEnum getBasicScalarRWRtn(LOGICAL, FormatType);
static FtnRtlEnum getAggrRWRtn(LOGICAL);
static FtnRtlEnum getWriteByDtypeRtn(int, FormatType);
static FtnRtlEnum getArrayRWRtn(LOGICAL, FormatType, int, LOGICAL);

static ITEM *gen_dtio_args(SST *, int, int, int);
static int gen_dtsfmt_args(int *, int *);
static int call_dtsfmt(int, int);
static int get_defined_io_call(int, int, ITEM *);

static int begin_io_call(int, int, int);
static void add_io_arg(int);
static int end_io_call(void);
static void _put(INT, int);

/*---------------------------------------------------------------------------*/

/** \brief Semantic analysis of IO statements.
    \param rednum   reduction number
    \param top      top of stack after reduction
 */
void
semantio(int rednum, SST *top)
{
  int sptr, i, iofunc;
  int len;
  int dtype;
  int dum;
  ADSC *ad;
  SST *stkptr, *e1;
  IOL *iolptr;
  IOL *dobegin, *doend;
  DOINFO *doinfo;
  FtnRtlEnum rtlRtn;
  int ast;
  int count;
  int ast1, ast2, ast3;
  int dim; /* dimension # of the index variable */
  int asd; /* array subscript descriptor */
  int subs[7];
  int numdim;
  int sptr1;
  int nelems;
  int last_inquire_val;
  ITEM *itemp;
  char *strptr;
  LOGICAL needDescr;

  switch (rednum) {

  /* ------------------------------------------------------------------ */
  /*
   *	<null> ::=
   */
  case NULL1:
    /*
     * this reduction is made at the beginning of any I/O statement
     * not involving the transfer of data.  It is used to initialize for
     * the processing of these statements.
     */
    no_rw = TRUE;
    goto io_shared;

  /* ------------------------------------------------------------------ */
  /*
   *	<write> ::=
   */
  case WRITE1:
    /*
     * this reduction is made at the beginning of any I/O statement
     * performing output. It is used to initialize for the processing
     * of these statements.
     */
    is_read = FALSE;
    goto rw_init;

  /* ------------------------------------------------------------------ */
  /*
   *	<read> ::=
   */
  case READ1:
    /*
     * this reduction is made at the beginning of any I/O statement
     * performing input. It is used to initialize for the processing
     * of these statements.
     */
    is_read = TRUE;
  rw_init:
    sem.io_stmt = TRUE;
    fmttyp = FT_UNFORMATTED;
    intern = FALSE;
    no_rw = FALSE;
    print_star = FALSE;
    unit_star = FALSE;

    /* shared entry for all io statements except bufferin/bufferout */

  io_shared:
    set_io_sc();
    bitv = 0;
    for (i = 0; i <= PT_MAXV; i++) {
      PTV(i) = 0;
      PTS(i) = 0;
      PTVARREF(i) = 0;
      PT_TMPUSED(i, 0);
    }
    if (flg.smp || flg.accmp || XBIT(125, 0x1)) {
      /* begin i/o critical section */
      if (flg.smp || flg.accmp)
        sptr = sym_mkfunc_nodesc("_mp_bcs_nest", DT_NONE);
      else
        sptr = mk_iofunc(RTE_f90io_begin, DT_NONE, 0);
      (void)begin_io_call(A_CALL, sptr, 0);
      ast = end_io_call();
      STD_LINENO(io_call.std) = gbl.lineno;
      /*
       * if an I/O statement is labeled, ensure that the first 'statement'
       * generated is labeled.
       */
      if (scn.currlab && !DEFDG(scn.currlab)) {
        STD_LABEL(io_call.std) = scn.currlab;
        DEFDP(scn.currlab, 1);
      }
    }
    /*
     * Create a character variable which is data initialized with
     * the name of the source file if character constants can't be
     * passed to RTE_loc().  Certain systems (hp) may place character
     * constants on the stack, in which case the run-time or generated
     * code can't stash away the address of the constant.
     */
    sptr = getstring(gbl.curr_file, strlen(gbl.curr_file));
    if (!XBIT(49, 0x100000))
      ast1 = mk_cnst(sptr);
    else {
      sptr1 = getcctmp_sc('t', sptr, ST_UNKNOWN, DTYPEG(sptr), io_sc);
      if (STYPEG(sptr1) == ST_UNKNOWN) {
        STYPEP(sptr1, ST_VAR);
        DINITP(sptr1, 1);
        if (SCG(sptr1) != SC_NONE)
          sym_is_refd(sptr1);
        dinit_put(DINIT_LOC, sptr1);
        dinit_put(DINIT_STR, (INT)sptr);
        dinit_put(DINIT_END, (INT)0);
      }
      ast1 = mk_id(sptr1);
    }
    sptr = mk_iofunc(RTE_f90io_src_info03a, DT_NONE, 0);
    (void)begin_io_call(A_CALL, sptr, 2);
    (void)add_io_arg(mk_cval((INT)gbl.lineno, DT_INT));
    (void)add_io_arg(ast1);
    ast = end_io_call();
    STD_LINENO(io_call.std) = gbl.lineno;
    /*
     * if an I/O statement is labeled, ensure that the first 'statement'
     * generated is labeled.
     */
    if (!XBIT(125, 0x1) && !flg.smp && !flg.accmp && scn.currlab &&
        !DEFDG(scn.currlab)) {
      STD_LABEL(io_call.std) = scn.currlab;
      DEFDP(scn.currlab, 1);
    }

    iolist = 0;
    noparens = FALSE;
    external_io = FALSE;
    nondevice_io = FALSE;
    open03 = FALSE;
    rw03 = FALSE;
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<IO stmt> ::= <null>  BACKSPACE <unit info>              |
   */
  case IO_STMT1:
    (void)misc_io_checks("BACKSPACE");
    iomsg_check();
    kwd_errchk(BT_BKSPACE);
    if (BYTE_SWAPPED_IO)
      rtlRtn = RTE_f90io_swbackspace; /* byte swap backspace */
    else
      rtlRtn = RTE_f90io_backspace;
    goto rewind_shared;
  /*
   *	<IO stmt> ::= <null>  ENDFILE <unit info>                |
   */
  case IO_STMT2:
    (void)misc_io_checks("ENDFILE");
    iomsg_check();
    kwd_errchk(BT_ENDFILE);
    rtlRtn = RTE_f90io_endfile;
    goto rewind_shared;
  /*
   *	<IO stmt> ::= <null>  REWIND <unit info>                 |
   */
  case IO_STMT3:
    (void)misc_io_checks("REWIND");
    iomsg_check();
    kwd_errchk(BT_REWIND);
    rtlRtn = RTE_f90io_rewind;
  rewind_shared:
    sptr = mk_iofunc(rtlRtn, DT_INT, 0);
    UNIT_CHECK;
    fix_iostat();
    (void)begin_io_call(A_FUNC, sptr, 3);
    (void)add_io_arg(PTARG(PT_UNIT));
    (void)add_io_arg(mk_cval(bitv, DT_INT));
    (void)add_io_arg(PTARG(PT_IOSTAT));
    ast = end_io_call();
    ast = add_cgoto(ast);
    external_io = TRUE;
    nondevice_io = TRUE;
    goto end_IO_STMT;
  /*
   *	<IO stmt> ::= <null>  CLOSE <iolp> <spec list> )              |
   */
  case IO_STMT4:
    (void)misc_io_checks("CLOSE");
    iomsg_check();
    UNIT_CHECK;
    kwd_errchk(BT_CLOSE);
    if (PTV(PT_STATUS)) {
      if (PTV(PT_DISPOSE))
        IOERR2(202, "STATUS and DISPOSE in CLOSE");
    } else
      PTV(PT_STATUS) = PTV(PT_DISPOSE);
    PT_CHECK(PT_STATUS, astb.ptr0c);
    sptr = mk_iofunc(RTE_f90io_closea, DT_INT, 0);
    fix_iostat();
    (void)begin_io_call(A_FUNC, sptr, 4);
    (void)add_io_arg(PTARG(PT_UNIT));
    (void)add_io_arg(mk_cval(bitv, DT_INT));
    (void)add_io_arg(PTARG(PT_IOSTAT));
    (void)add_io_arg(PTARG(PT_STATUS));
    ast = end_io_call();
    ast = add_cgoto(ast);
    external_io = TRUE;
    nondevice_io = TRUE;
    goto end_IO_STMT;
  /*
   *	<IO stmt> ::= <null>  OPEN <iolp> <spec list> )               |
   */
  case IO_STMT5:
    (void)misc_io_checks("OPEN");
    iomsg_check();
    UNIT_CHECK;
    newunit_check();
    kwd_errchk(BT_OPEN);
    PT_CHECK(PT_ACCESS, astb.ptr0c);
    PT_CHECK(PT_ACTION, astb.ptr0c);
    PT_CHECK(PT_BLANK, astb.ptr0c);
    PT_CHECK(PT_DELIM, astb.ptr0c);
    PT_CHECK(PT_FORM, astb.ptr0c);
    fix_iostat();
    PT_CHECK(PT_PAD, astb.ptr0c);
    PT_CHECK(PT_POSITION, astb.ptr0c);
    PT_CHECK(PT_RECL, astb.ptr0);
    PT_CHECK(PT_STATUS, astb.ptr0c);
    PT_CHECK(PT_FILE, astb.ptr0c);
    PT_CHECK(PT_DISPOSE, astb.ptr0c);

    if (PTS(PT_NEWUNIT)) {
      sptr = sym_mkfunc(mkRteRtnNm(RTE_f90io_get_newunit), DT_INT);
      INDEPP(sptr, 1);
      TYPDP(sptr, 1);
      INTERNALP(sptr, 0);

      ast = mk_func_node(A_FUNC, mk_id(sptr), 0, 0);
      ast = mk_assn_stmt(PTV(PT_NEWUNIT), ast, A_DTYPEG(PTV(PT_NEWUNIT)));
      add_stmt_after(ast, io_call.std);
      if (A_DTYPEG(PTV(PT_NEWUNIT)) != DT_INT) {
        PTV(PT_UNIT) = mk_convert(PTV(PT_NEWUNIT), DT_INT);
      }
    }

    if (PTS(PT_FILE) && PTS(PT_NAME))
      IOERR2(202, "FILE and NAME in OPEN");

    sptr = mk_iofunc(RTE_f90io_open2003a, DT_INT, 0);
    (void)begin_io_call(A_FUNC, sptr, 14);
    (void)add_io_arg(PTARG(PT_UNIT));
    (void)add_io_arg(mk_cval(bitv, DT_INT));
    (void)add_io_arg(PTARG(PT_ACCESS));
    (void)add_io_arg(PTARG(PT_ACTION));
    (void)add_io_arg(PTARG(PT_BLANK));
    (void)add_io_arg(PTARG(PT_DELIM));

    /* on open statement, FILE and NAME are the same.  code previously
       only set PT_FILE, but this created incorrect error messages */
    if (PTV(PT_NAME))
      (void)add_io_arg(PTARG(PT_NAME));
    else
      (void)add_io_arg(PTARG(PT_FILE));
    (void)add_io_arg(PTARG(PT_FORM));
    (void)add_io_arg(PTARG(PT_IOSTAT));
    (void)add_io_arg(PTARG(PT_PAD));
    (void)add_io_arg(PTARG(PT_POSITION));
    (void)add_io_arg(PTARG(PT_RECL));
    (void)add_io_arg(PTARG(PT_STATUS));
    (void)add_io_arg(PTARG(PT_DISPOSE));
    ast = end_io_call();
    if (PTV(PT_CONVERT)) {
      /* ast is an A_ASN of the form
       * z_io = ...open(...)
       */
      sptr = mk_iofunc(RTE_f90io_open_cvta, DT_INT, 0);
      (void)begin_io_call(A_FUNC, sptr, 2);
      (void)add_io_arg(A_DESTG(ast));
      (void)add_io_arg(PTARG(PT_CONVERT));
      ast = end_io_call();
    }
    if (PTV(PT_SHARED)) {
      /* ast is an A_ASN of the form
       * z_io = ...open(...)
       */
      sptr = mk_iofunc(RTE_f90io_open_sharea, DT_INT, 0);
      (void)begin_io_call(A_FUNC, sptr, 2);
      (void)add_io_arg(A_DESTG(ast));
      (void)add_io_arg(PTARG(PT_SHARED));
      ast = end_io_call();
    }
    if (PTV(PT_ASYNCHRONOUS)) {
      /* ast is an A_ASN of the form
       * z_io = ...open(...)
       */
      sptr = mk_iofunc(RTE_f90io_open_asynca, DT_INT, 0);
      (void)begin_io_call(A_FUNC, sptr, 2);
      (void)add_io_arg(A_DESTG(ast));
      (void)add_io_arg(PTARG(PT_ASYNCHRONOUS));
      ast = end_io_call();
    }
    if (open03) {
      /* ast is an A_ASN of the form
       * z_io = ...open(...)
       */
      PT_CHECK(PT_DECIMAL, astb.ptr0c);
      PT_CHECK(PT_ROUND, astb.ptr0c);
      PT_CHECK(PT_SIGN, astb.ptr0c);
      PT_CHECK(PT_ENCODING, astb.ptr0c);
      sptr = mk_iofunc(RTE_f90io_open03a, DT_INT, 0);
      (void)begin_io_call(A_FUNC, sptr, 5);
      (void)add_io_arg(A_DESTG(ast));
      (void)add_io_arg(PTARG(PT_DECIMAL));
      (void)add_io_arg(PTARG(PT_ROUND));
      (void)add_io_arg(PTARG(PT_SIGN));
      (void)add_io_arg(PTARG(PT_ENCODING));
      ast = end_io_call();
    }
    ast = add_cgoto(ast);
    external_io = TRUE;
    nondevice_io = TRUE;
    goto end_IO_STMT;
  /*
   *	<IO stmt> ::= <null>  INQUIRE <iolp> <spec list> )         |
   */
  case IO_STMT6:

    (void)misc_io_checks("INQUIRE");
    iomsg_check();
    if (PTV(PT_UNIT)) {
      if (PTV(PT_FILE))
        IOERR2(201, "UNIT and FILE used in INQUIRE");
      PTV(PT_FILE) = astb.ptr0c;
    } else if (PTV(PT_FILE))
      PTV(PT_UNIT) = astb.i0;
    else
      IOERR(200);

    kwd_errchk(BT_INQUIRE);

    last_inquire_val = PT_LAST_INQUIRE_VALf95;
    for (i = PT_LAST_INQUIRE_VALf95 + 1; i <= PT_LAST_INQUIRE_VAL; i++) {
      if (PTV(i)) {
        last_inquire_val = PT_LAST_INQUIRE_VAL;
        break;
      }
    }
    for (i = 2; i <= last_inquire_val; i++) {
      switch (i) {
      case PT_NEXTREC:
      case PT_NUMBER:
      case PT_RECL:
      case PT_SIZE:
      case PT_IOSTAT:
      case PT_EXIST:
      case PT_NAMED:
      case PT_OPENED:
      case PT_ID:
      case PT_PENDING:
      case PT_POS:
        PT_CHECK(i, astb.ptr0);
        break;
      default:
        PT_CHECK(i, astb.ptr0c);
      }
    }

    if (last_inquire_val < PT_LAST_INQUIRE_VAL) {
      sptr = mk_iofunc(RTE_f90io_inquire2003a, DT_INT, 0);
    } else {
      sptr = mk_iofunc(RTE_f90io_inquire03_2a, DT_INT, 0);
    }
    filename_type = 0;
    (void)begin_io_call(A_FUNC, sptr, last_inquire_val + 2);
    (void)add_io_arg(PTARG(PT_UNIT));
    (void)add_io_arg(PTARG(PT_FILE));
    (void)add_io_arg(mk_cval(bitv, DT_INT));
    for (i = 2; i <= last_inquire_val; i++)
      (void)add_io_arg(PTARG(i));
    ast = end_io_call();

    ast = add_cgoto(ast);
    external_io = TRUE;
    nondevice_io = TRUE;
    goto end_IO_STMT;
  /*
   *	<IO stmt> ::= <write> WRITE <io spec>                    |
   */
  case IO_STMT7:
    (void)misc_io_checks("WRITE");
    kwd_errchk(BT_WRITE);
    external_io = !intern;
    goto io_end;
  /*
   *	<IO stmt> ::= <write> WRITE <io spec> <output list>      |
   */
  case IO_STMT8:
    (void)misc_io_checks("WRITE");
    kwd_errchk(BT_WRITE);
    iolptr = (IOL *)SST_BEGG(RHS(4));
    external_io = !intern;
    goto io_items;
  /*
   *	<IO stmt> ::= <write> PRINT <print spec>                 |
   */
  case IO_STMT9:
    (void)misc_io_checks("PRINT");
    kwd_errchk(BT_PRINT);
    external_io = !intern;
    goto io_end;
  /*
   *	<IO stmt> ::= <write> PRINT <print spec> , <output list> |
   */
  case IO_STMT10:
    (void)misc_io_checks("PRINT");
    kwd_errchk(BT_PRINT);
    iolptr = (IOL *)SST_BEGG(RHS(5));
    external_io = !intern;
    goto io_items;
  /*
   *	<IO stmt> ::= <read>  READ <io spec> <input list>        |
   */
  case IO_STMT11:
    (void)misc_io_checks("READ");
    kwd_errchk(BT_READ);
    chk_iospec();
    iolptr = (IOL *)SST_BEGG(RHS(4));
    external_io = !intern;
    nondevice_io = TRUE;
    goto io_items;
  /*
   *	<IO stmt> ::= <read>  READ <read spec2>                  |
   */
  case IO_STMT12:
    (void)misc_io_checks("READ");
    kwd_errchk(BT_READ);
    external_io = !intern;
    nondevice_io = TRUE;
    goto io_end;
  /*
   *	<IO stmt> ::= <read>  READ <read spec3> , <input list>   |
   */
  case IO_STMT13:
    (void)misc_io_checks("READ");
    kwd_errchk(BT_READ);
    iolptr = (IOL *)SST_BEGG(RHS(5));
    external_io = !intern;
    nondevice_io = TRUE;
    goto io_items;
  /*
   *	<IO stmt> ::= <read>  ACCEPT <read spec4>                |
   */
  case IO_STMT14:
    (void)misc_io_checks("ACCEPT");
    iomsg_check();
    kwd_errchk(BT_ACCEPT);
    external_io = !intern;
    nondevice_io = TRUE;
    goto io_end;
  /*
   *	<IO stmt> ::= <read>  ACCEPT <read spec3> , <input list>  |
   */
  case IO_STMT15:
    (void)misc_io_checks("ACCEPT");
    iomsg_check();
    kwd_errchk(BT_ACCEPT);
    iolptr = (IOL *)SST_BEGG(RHS(5));
    external_io = !intern;
    nondevice_io = TRUE;
    goto io_items;
  /*
   *	<IO stmt> ::= <write> ENCODE <encode spec> <optional comma> <output
   *list> |
   */
  case IO_STMT16:
    (void)misc_io_checks("ENCODE");
    iomsg_check();
    kwd_errchk(BT_ENCODE);
    iolptr = (IOL *)SST_BEGG(RHS(5));
    external_io = FALSE;
    nondevice_io = TRUE;
    goto io_items;
  /*
   *	<IO stmt> ::= <write> ENCODE <encode spec>               |
   */
  case IO_STMT17:
    (void)misc_io_checks("ENCODE");
    iomsg_check();
    kwd_errchk(BT_ENCODE);
    external_io = FALSE;
    nondevice_io = TRUE;
    goto io_end;
  /*
   *	<IO stmt> ::= <read>  DECODE <encode spec> <optional comma> <input list>
   */
  case IO_STMT18:
    (void)misc_io_checks("DECODE");
    iomsg_check();
    kwd_errchk(BT_DECODE);
    iolptr = (IOL *)SST_BEGG(RHS(5));
    external_io = FALSE;
    nondevice_io = TRUE;
    goto io_items;
  /*
   *	<IO stmt> ::= <read>  DECODE <encode spec> |
   */
  case IO_STMT19:
    (void)misc_io_checks("DECODE");
    iomsg_check();
    kwd_errchk(BT_DECODE);
    external_io = FALSE;
    nondevice_io = TRUE;
    goto io_end;

  io_items:
    if (fmttyp == FT_NML) {
      IOERR(212);
      break;
    }
    count = 4;
    rtlRtn = getBasicScalarRWRtn(is_read, fmttyp);
    iofunc = mk_iofunc(rtlRtn, DT_INT, is_read ? 0 : INTENT_IN);
    SEQUENTP(iofunc, 1); /* TPR1786 */

    for (; iolptr; iolptr = iolptr->next) {
      switch (iolptr->id) {
      case IE_OPTDO:
        stkptr = iolptr->element;
        dtype = SST_DTYPEG(stkptr);
        doinfo = iolptr->doinfo;
        /* TBD:
            I would rather call a function to compute the lastvalue:
              index_var = f90_lastval(m1, m2, m3)
        */
        /*
         * Compute last value of index variable:
         *    index_var = max( (m2 - m1 + m3)/m3, 0)
         */
        gen_lastval(doinfo);
        goto array_io_item;
      case IE_EXPR:
        stkptr = iolptr->element;
        dtype = SST_DTYPEG(stkptr);
        if (DTY(dtype) == TY_ARRAY) {
          ad = AD_DPTR(dtype);
          if (SST_IDG(stkptr) == S_IDENT && AD_ASSUMSZ(ad)) {
            /* illegal use of assumed size array */
            ast2 = astb.i0;
            IOERR(215);
          }
          goto array_io_item;
        }
        /* does this have a shape */
        if (SST_SHAPEG(stkptr))
          goto array_io_item;

        /* at this point, io item is a scalar */

        if (dtype == DT_HOLL) {
          /* semantic stack type is hollerith, but want its associated
           * character constant.
           */
          sptr1 = CONVAL1G(SST_SYMG(stkptr));
          dtype = DTYPEG(sptr1);
#if DEBUG
          assert(DTY(dtype) == TY_CHAR, "io_item: HOLL not char", sptr1, 3);
#endif
          SST_DTYPEP(stkptr, dtype);
          SST_ASTP(stkptr, mk_cnst(sptr1));
        }
        ast2 = astb.i1;
        ast3 = astb.i0;
        (void)mkarg(stkptr, &dum);
        ast = SST_ASTG(stkptr);
        if (is_read) {
          if (A_TYPEG(ast) == A_ID) {
            DOCHK(A_SPTRG(ast));
          }
          if (SST_IDG(stkptr) == S_IDENT || SST_IDG(stkptr) == S_DERIVED)
            set_assn(sym_of_ast(SST_ASTG(stkptr)));
          else if (SST_IDG(stkptr) == S_LVALUE)
            set_assn(sym_of_ast(SST_ASTG(stkptr)));
        }
        if (!DT_ISBASIC(dtype) && SST_IDG(stkptr) != S_DERIVED) {
          int bytfunc;
          int udt;

          FtnRtlEnum aggrRWRtn = getAggrRWRtn(is_read);

          if (SST_IDG(stkptr) == S_IDENT) {
            sptr = SST_SYMG(stkptr);
            udt = DTYPEG(sptr);
          } else if (SST_IDG(stkptr) == S_EXPR) {
            if (A_TYPEG(ast) == A_ID) {
              sptr = A_SPTRG(ast);
              udt = DTYPEG(sptr);
            } else {
              sptr = 0;
              udt = A_DTYPEG(ast);
            }
          } else {
            sptr = SST_LSYMG(stkptr);
            udt = DTYPEG(sptr);
          }
          if (DTYG(udt) == TY_DERIVED) {
            int iotype_ast = 0, vlist_ast = 0;
            int fsptr, argcnt, has_io, asn, tast;
            const char *iotype;
            ITEM *arglist;

            dtype = udt;
            has_io = dtype_has_defined_io(dtype);
#if DEBUG
            assert(has_io, "unknown status from dtype_has_defined_io()", 0, 3);
#endif
            if (has_io & functype[is_read][fmttyp]) {
              if (fmttyp == FT_UNFORMATTED) {
                argcnt = 4;
              } else {
                int tast;
                argcnt = 6;
                if (fmttyp == FT_NML) {
                  iotype = "NAMELIST";
                  vlist_ast = astb.i0;
                  iotype_ast = mk_cnst(getstring(iotype, strlen(iotype)));
                } else if (fmttyp == FT_LIST_DIRECTED) {
                  iotype = "LISTDIRECTED";
                  vlist_ast = astb.i0;
                  iotype_ast = mk_cnst(getstring(iotype, strlen(iotype)));
                } else {
                  asn = gen_dtsfmt_args(&iotype_ast, &vlist_ast);
                  asn = call_dtsfmt(iotype_ast, vlist_ast);
                  tast = add_cgoto(asn);
                }
              }
              arglist = gen_dtio_args(stkptr, ast, iotype_ast, vlist_ast);

              fsptr = resolve_defined_io((is_read ? 0 : 1), stkptr, arglist);
              if (!fsptr) {
                /*unable to find io routine*/
                error(155, 2, gbl.lineno,
                      "- Unable to resolve user defined io ", CNULL);

              } else {
                /*  set the upper bound of descriptor here? */
                ast = get_defined_io_call(fsptr, argcnt, arglist);
                (void)add_stmt(ast);
                /* error handling in parent in case of error */
                if (argcnt == 4) {
                  tast = (arglist->next->next)->ast;
                } else {
                  tast = (arglist->next->next->next->next)->ast;
                }
                sptr1 = mk_iofunc(RTE_f90io_dts_stat, DT_NONE, 0);
                ast1 = begin_io_call(A_CALL, sptr1, 1);
                (void)add_io_arg(mk_unop(OP_VAL, tast, DT_INT4));
                (void)add_stmt(ast1);
                (void)add_cgoto(tast);
              }
              break;
            } else {
              gen_derived_io(sptr, aggrRWRtn, is_read);
            }
          } else {
            bytfunc = mk_iofunc(aggrRWRtn, DT_INT, is_read ? 0 : INTENT_IN);
            if (XBIT(68, 0x1)) {
              /* just transfer a stream of n bytes */
              i = mk_isz_cval(size_of(dtype), DT_INT8);
              (void)begin_io_call(A_FUNC, bytfunc, 4);
              (void)add_io_arg(i); /* length, stride */
              (void)add_io_arg(astb.i0);
              (void)add_io_arg(ast);     /* item */
              (void)add_io_arg(astb.k1); /* explicit item_length */
            } else {
              i = mk_cval(size_of(dtype), DT_INT8);
              (void)begin_io_call(A_FUNC, bytfunc, 4);
              (void)add_io_arg(ast2); /* length, stride */
              (void)add_io_arg(ast3);
              (void)add_io_arg(ast); /* item */
              (void)add_io_arg(i);   /* explicit item_length */
            }
          }
        } else if (fmttyp != FT_UNFORMATTED && !is_read &&
                   (dtype == DT_INT8 || dtype == DT_INT4 || dtype == DT_SINT ||
                    dtype == DT_BINT || dtype == DT_LOG8 || dtype == DT_LOG ||
                    dtype == DT_SLOG || dtype == DT_BLOG || dtype == DT_REAL4 ||
                    dtype == DT_REAL8 || dtype == DT_QUAD || dtype == DT_CMPLX8 ||
                    dtype == DT_CMPLX16 ||
#ifdef TARGET_SUPPORTS_QUADFP
                    dtype == DT_QCMPLX ||
#endif
                    (DTY(dtype) == TY_CHAR && fmttyp == FT_LIST_DIRECTED))) {

          i = sym_mkfunc_nodesc(mkRteRtnNm(getWriteByDtypeRtn(dtype, fmttyp)),
                                DT_INT);
          if (DTY(dtype) != TY_CHAR) {
            ast = mk_unop(OP_VAL, ast, dtype);
          }
          ast1 = mk_cval((INT)dtype_to_arg(dtype), DT_INT4);
          ast1 = mk_unop(OP_VAL, ast1, DT_INT4);
          (void)begin_io_call(A_FUNC, i, 2);
          (void)add_io_arg(ast);  /* item */
          (void)add_io_arg(ast1); /* type */
        } else {
          ast1 = mk_cval((INT)dtype_to_arg(dtype), DT_INT);
          (void)begin_io_call(A_FUNC, iofunc, 4);
          (void)add_io_arg(ast1); /* type, length, stride */
          (void)add_io_arg(ast2);
          (void)add_io_arg(ast3);
          (void)add_io_arg(ast); /* item */
        }
        ast = end_io_call();
        ast = add_cgoto(ast);
        break;

      case IE_DOBEGIN:
        doinfo = iolptr->doinfo;
        ast = do_begin(doinfo);
        {
          /* for any stds attached to this DO, add them after
           * the DO begin; these STDs are simply linked in by
           * altering the NEXT & PREV fields of the DO's std
           * and the first and last stds. in the list.
           */
          int s, s1;
          s1 = add_stmt_after(ast, (int)STD_PREV(0));
          s = iolptr->l_std;
          if (s) {
            STD_NEXT(s1) = s;
            STD_PREV(s) = s1;
            s1 = s;
            /* find last STD in the list attached to the DO */
            while (STD_NEXT(s1))
              s1 = STD_NEXT(s1);
            STD_PREV(0) = s1;
          }
        }
        NEED_DOIF(i, DI_DO);
        break;

      case IE_DOEND:
        do_end(iolptr->doinfo);
        break;

      default:
        interr("io_items, badIE", iolptr->id, 3);
      }

      continue;

    array_io_item:
      if (DTY(dtype) == TY_ARRAY)
        dtype = DTY(dtype + 1);
      if (no_data_components(dtype))
        continue;
      if (!((SST_IDG(stkptr) == S_EXPR && DTYG(dtype) == TY_DERIVED)))
        (void)mkarg(stkptr, &dum);
      if (is_read) {
        if (SST_IDG(stkptr) == S_IDENT)
          set_assn(sym_of_ast(SST_ASTG(stkptr)));
        else if (SST_IDG(stkptr) == S_LVALUE)
          set_assn(sym_of_ast(SST_ASTG(stkptr)));
      }
      ast = SST_ASTG(stkptr);
      needDescr = need_descriptor_ast(ast);
      rtlRtn = getArrayRWRtn(is_read, fmttyp, dtype, !needDescr);
      if (DTY(dtype) == TY_CHAR) {
        if (!needDescr) {
          /* collapse a read/write of an array into a single call
           * which doesn't require a descriptor
           */
          if (dtype != DT_ASSCHAR && dtype != DT_DEFERCHAR) {
            ast3 = SST_CVLENG(stkptr);
            if (!ast3)
              ast3 = mk_cval(size_of(dtype), DT_INT);
          } else {
            ast3 = string_expr_length(ast);
          }
          rw_array(ast, ast3, dtype, rtlRtn);
          goto end_io_item;
        }
        ast1 = mk_cval((INT)ty_to_lib[TY_CHAR], DT_INT);
        ast2 = astb.i1; /* tpr 1786: this gets elementalized later */
        if (dtype != DT_ASSCHAR && dtype != DT_DEFERCHAR) {
          ast3 = SST_CVLENG(stkptr);
          if (!ast3)
            ast3 = mk_cval(size_of(dtype), DT_INT);
        } else {
          ast3 = string_expr_length(ast);
        }
        (void)begin_io_call(A_FUNC, iofunc, 4);
        (void)add_io_arg(ast1); /* type, length, stride */
        (void)add_io_arg(ast2);
        (void)add_io_arg(ast3);
        (void)add_io_arg(ast); /* item */
      }
      else if (DTY(dtype) == TY_NCHAR) {
        ast1 = mk_cval((INT)ty_to_lib[TY_NCHAR], DT_INT);
        ast2 = astb.i1; /* tpr 1786: this gets elementalized later */
        if (dtype != DT_ASSNCHAR && dtype != DT_DEFERNCHAR)
          ast3 = mk_cval(size_of(dtype), DT_INT);
        else {
          i = sym_mkfunc_nodesc(mkRteRtnNm(RTE_nlena), DT_INT);
          ast3 = begin_call(A_FUNC, i, 1);
          add_arg(ast);
        }
        (void)begin_io_call(A_FUNC, iofunc, 4);
        (void)add_io_arg(ast1); /* type, length, stride */
        (void)add_io_arg(ast2);
        (void)add_io_arg(ast3);
        (void)add_io_arg(ast); /* item */
      }
      else if (!DT_ISBASIC(dtype)) {
        int bytfunc;
        if (DTY(dtype) == TY_DERIVED &&
            (dtype_has_defined_io(dtype) & functype[is_read][fmttyp])) {
          const char *iotype;
          int tast, iotype_ast = 0, vlist_ast = 0, argcnt, asn, fsptr;
          int shape, forall, triplet_list, n, lb, ub, st, newast;
          int index_var, triplet, dovar, list, sym, triple;
          ITEM *arglist;
          int subs[7], std;
          int i;
          if (fmttyp == FT_UNFORMATTED) {
            argcnt = 4;
          } else {
            argcnt = 6;
            if (fmttyp == FT_LIST_DIRECTED) {
              iotype = "LISTDIRECTED";
              vlist_ast = astb.i0;
              iotype_ast = mk_cnst(getstring(iotype, strlen(iotype)));
            } else if (fmttyp != FT_NML) {
              /* only generates the argument, no call */
              asn = gen_dtsfmt_args(&iotype_ast, &vlist_ast);
            }
          }
          /* need to make a loop here */

          arglist = gen_dtio_args(stkptr, ast, iotype_ast, vlist_ast);
          shape = A_SHAPEG(arglist->ast);

          /* make forall */
          start_astli();
          numdim = SHD_NDIM(shape);
          ad = AD_DPTR(dtype);
          for (i = 0; i < numdim; i++) {
            subs[i] = ASD_SUBS(A_ASDG(ast1), i);
          }
          for (i = numdim - 1; i >= 0; i--) {
            /* make each forall index */
            lb = check_member(ast1, SHD_LWB(shape, i));
            ub = check_member(ast1, SHD_UPB(shape, i));
            st = check_member(ast1, SHD_STRIDE(shape, i));
            if (A_DTYPEG(lb) == DT_INT8 || A_DTYPEG(ub) == DT_INT8 ||
                A_DTYPEG(st) == DT_INT8)
              dtype = DT_INT8;
            else
              dtype = astb.bnd.dtype;
            if (A_DTYPEG(lb) == DT_INT8 || A_DTYPEG(ub) == DT_INT8 ||
                A_DTYPEG(st) == DT_INT8)
              sym = sym_get_scalar("i", 0, DT_INT8);
            else
              sym = sym_get_scalar("i", 0, DT_INT);
            list = add_astli();
            triple = mk_triple(lb, ub, st);
            ASTLI_SPTR(list) = sym;
            ASTLI_TRIPLE(list) = triple;
            subs[i] = mk_id(sym);
          }
          forall = mk_stmt(A_FORALL, 0);
          A_LISTP(forall, ASTLI_HEAD);
          A_IFEXPRP(forall, 0);

          n = 0;
          triplet_list = A_LISTG(forall);
          for (; triplet_list; triplet_list = ASTLI_NEXT(triplet_list)) {
            n++;
            newast = mk_stmt(A_DO, 0);
            index_var = ASTLI_SPTR(triplet_list);
            triplet = ASTLI_TRIPLE(triplet_list);
            dovar = mk_id(index_var);
            lb = A_LBDG(triplet);
            ub = A_UPBDG(triplet);
            st = A_STRIDEG(triplet);
            A_M1P(newast, lb);
            A_M2P(newast, ub);
            A_M3P(newast, st);
            A_M4P(newast, 0);
            A_DOVARP(newast, dovar);

            std = add_stmt(newast);
          }
          if (A_TYPEG(ast) == A_SUBSCR)
            ast1 = mk_subscr(A_LOPG(ast), subs, numdim, DDTG(A_DTYPEG(ast)));
          else
            ast1 = mk_subscr(ast, subs, numdim, DDTG(A_DTYPEG(ast)));

          /* need to generate a new stkptr */
          arglist->t.stkp = (SST *)getitem(0, sizeof(SST));
          SST_ASTP(arglist->t.stkp, ast1);
          SST_DTYPEP(arglist->t.stkp, A_DTYPEG(ast1));
          SST_SHAPEP(arglist->t.stkp, 0);
          arglist->ast = ast1;
          SST_IDP(arglist->t.stkp, S_EXPR);
          mkexpr(arglist->t.stkp);
          fsptr =
              resolve_defined_io((is_read ? 0 : 1), arglist->t.stkp, arglist);
          if (!fsptr) {
            error(155, 2, gbl.lineno, "- Unable to resolve user defined io",
                  CNULL);
          } else {
            /* need to make a loop */
            if (fmttyp != FT_UNFORMATTED && fmttyp != FT_LIST_DIRECTED &&
                fmttyp != FT_NML) {
              asn = call_dtsfmt(iotype_ast, vlist_ast);
              tast = add_cgoto(asn);
            }
            tast = get_defined_io_call(fsptr, argcnt, arglist);
            (void)add_stmt_after(tast, std);
            if (argcnt == 4) {
              tast = (arglist->next->next)->ast;
            } else {
              tast = (arglist->next->next->next->next)->ast;
            }
            asn = mk_assn_stmt(ast_ioret(), tast, DT_INT);
            (void)add_stmt(asn);
            tast = add_cgoto(asn);
          }
          triplet_list = A_LISTG(forall);
          for (; triplet_list; triplet_list = ASTLI_NEXT(triplet_list)) {
            newast = mk_stmt(A_ENDDO, 0);
            (void)add_stmt(newast);
          }

          continue;
        } else {
          bytfunc =
              mk_iofunc(getAggrRWRtn(is_read), DT_INT, is_read ? 0 : INTENT_IN);
          ast2 = AD_NUMELM(AD_DPTR(SST_DTYPEG(stkptr)));
          if (XBIT(68, 0x1)) {
            /* just transfer a stream of n bytes */
            ast3 = mk_isz_cval(size_of(dtype), DT_INT8);
            ast2 = mk_binop(OP_MUL, ast2, ast3, DT_INT8);
            (void)begin_io_call(A_FUNC, bytfunc, 4);
            (void)add_io_arg(ast2); /* length, stride */
            (void)add_io_arg(astb.i0);
            (void)add_io_arg(ast);     /* item */
            (void)add_io_arg(astb.k1); /* explicit item_length */
          } else {
            ast3 = mk_cval(size_of(dtype), DT_INT8);
            (void)begin_io_call(A_FUNC, bytfunc, 4);
            (void)add_io_arg(ast2); /* length, stride */
            (void)add_io_arg(ast3);
            (void)add_io_arg(ast);  /* item */
            (void)add_io_arg(ast3); /* explicit item_length */
          }
        }
      } else {
        if (!need_descriptor_ast(ast)) {
          /* collapse a read/write of an array into a single call
           * which doesn't require a descriptor
           */
          ast3 = mk_cval(size_of(dtype), DT_INT);
          rw_array(ast, ast3, dtype, rtlRtn);
          goto end_io_item;
        }
        (void)begin_io_call(
            A_FUNC, mk_hpfiofunc(rtlRtn, DT_INT, is_read ? 0 : INTENT_IN), 1);
        (void)add_io_arg(ast); /* item only */
      }
    end_io_item:
      ast = end_io_call();
      ast = add_cgoto(ast);
    }

  io_end:
    if (fmttyp == FT_NML) {
      if (is_read)
        rtlRtn = RTE_f90io_nmlr;
      else
        rtlRtn = RTE_f90io_nmlw;
      sptr = mk_iofunc(rtlRtn, DT_INT, is_read ? 0 : INTENT_IN);
      /*
       * First, create and "address" of the descriptor for the
       * namelist group; then, call the appropriate routine
       */
      (void)begin_io_call(A_FUNC, sptr, 1);
      (void)add_io_arg(PTARG(PT_NML));
      ast = end_io_call();
      ast = add_cgoto(ast);
      if (is_read)
        rtlRtn = RTE_f90io_nmlr_end;
      else
        rtlRtn = RTE_f90io_nmlw_end;
      sptr = mk_iofunc(rtlRtn, DT_INT, 0);
      (void)begin_io_call(A_FUNC, sptr, 0);
      ast = end_io_call();
    } else {
      if (fmttyp == FT_UNFORMATTED)
        rtlRtn = unf_nm[LARGE_ARRAY_IDX][BYTE_SWAPPED_IO_IDX].end;
      else if (fmttyp == FT_LIST_DIRECTED) {
        if (is_read)
          rtlRtn = RTE_f90io_ldr_end;
        else
          rtlRtn = RTE_f90io_ldw_end;
      } else {
        if (is_read)
          rtlRtn = RTE_f90io_fmtr_end;
        else
          rtlRtn = RTE_f90io_fmtw_end;
      }
      sptr = mk_iofunc(rtlRtn, DT_INT, 0);
      (void)begin_io_call(A_FUNC, sptr, 0);
      ast = end_io_call();
    }
    if (intern_tmp) {
      copy_back_to_replic_sect(intern_array, intern_tmp);
      intern_tmp = 0;
    }
    ast = add_cgoto(ast);
    goto end_IO_STMT;
  /*
   *	<IO stmt> ::= <null>  INQUIRE <iolp> IOLENGTH = <var ref> ) <output
   *list> |
   */
  case IO_STMT20:
    (void)misc_io_checks("INQUIRE");
    iomsg_check();
    chk_var(RHS(6), PT_IOLENGTH, DT_INT);
    iolptr = (IOL *)SST_BEGG(RHS(8));
    ast = astb.i0;
    ast = mk_assn_stmt(PTV(PT_IOLENGTH), ast, DT_INT);
    (void)add_stmt_after(ast, (int)STD_PREV(0));
    for (; iolptr; iolptr = iolptr->next) {
      switch (iolptr->id) {
      case IE_OPTDO:
        stkptr = iolptr->element;
        dtype = SST_DTYPEG(stkptr);
        doinfo = iolptr->doinfo;
        /* TBD:
            I would rather call a function to compute the lastvalue:
              index_var = f90_lastval(m1, m2, m3)
        */
        /*
         * Compute last value of index variable:
         *    index_var = max( (m2 - m1 + m3)/m3, 0)
         */
        gen_lastval(doinfo);
        goto array_iol;
      case IE_EXPR:
        stkptr = iolptr->element;
        dtype = SST_DTYPEG(stkptr);
        if (DTY(dtype) == TY_ARRAY) {
          ad = AD_DPTR(dtype);
          if (SST_IDG(stkptr) == S_IDENT && AD_ASSUMSZ(ad)) {
            /* illegal use of assumed  array */
            ast2 = astb.i0;
            IOERR(215);
          }
          goto array_iol;
        }

        /* at this point, io item is a scalar */

        nelems = astb.i1;
        goto accum_it;

      case IE_DOBEGIN:
        doinfo = iolptr->doinfo;
        ast = do_begin(doinfo);
        {
          /* for any stds attached to this DO, add them after
           * the DO begin; these STDs are simply linked in by
           * altering the NEXT & PREV fields of the DO's std
           * and the first and last stds. in the list.
           */
          int s, s1;
          s1 = add_stmt_after(ast, (int)STD_PREV(0));
          s = iolptr->l_std;
          if (s) {
            STD_NEXT(s1) = s;
            STD_PREV(s) = s1;
            s1 = s;
            /* find last STD in the list attached to the DO */
            while (STD_NEXT(s1))
              s1 = STD_NEXT(s1);
            STD_PREV(0) = s1;
          }
        }
        NEED_DOIF(i, DI_DO);
        break;

      case IE_DOEND:
        do_end(iolptr->doinfo);
        break;

      default:
        interr("iol_items, badIE", iolptr->id, 3);
      }

      continue;

    array_iol:
      dtype = DTY(dtype + 1);
      (void)mkarg(stkptr, &dum);
      ast = SST_ASTG(stkptr);
      nelems = size_of_ast(ast);

    accum_it:
      if (dtype == DT_HOLL) {
        /* semantic stack type is hollerith, but want its associated
         * character constant.
         */
        sptr1 = CONVAL1G(SST_SYMG(stkptr));
        dtype = DTYPEG(sptr1);
#if DEBUG
        assert(DTY(dtype) == TY_CHAR, "io_item: HOLL not char", sptr1, 3);
#endif
        ast = mk_cval(size_of(dtype), DT_INT);
      } else if (!DT_ISBASIC(dtype))
        /* treat aggr. as byte stream */
        ast = mk_cval(size_of(dtype), DT_INT);
      else if (DTY(dtype) == TY_CHAR) {
        (void)mkarg(stkptr, &dum);
        ast2 = SST_ASTG(stkptr);
        i = sym_mkfunc_nodesc(mkRteRtnNm(RTE_lena), DT_INT);
        ast = begin_call(A_FUNC, i, 1);
        add_arg(ast2);
      }
      else if (DTY(dtype) == TY_NCHAR) {
        (void)mkarg(stkptr, &dum);
        ast2 = SST_ASTG(stkptr);
        i = sym_mkfunc_nodesc(mkRteRtnNm(RTE_nlena), DT_INT);
        ast = begin_call(A_FUNC, i, 1);
        add_arg(ast2);
        ast1 = mk_cval(size_of(dtype), DT_INT);
        ast = mk_binop(OP_MUL, ast, ast1, DT_INT);
      }
      else
        ast = mk_cval(size_of(dtype), DT_INT);

      ast = mk_binop(OP_MUL, ast, nelems, DT_INT);
      ast = mk_binop(OP_ADD, PTV(PT_IOLENGTH), ast, DT_INT);
      ast = mk_assn_stmt(PTV(PT_IOLENGTH), ast, DT_INT);
      (void)add_stmt_after(ast, (int)STD_PREV(0));
    }
    external_io = TRUE;
    nondevice_io = TRUE;
    goto end_IO_STMT;

  /*
   *    <IO stmt> ::= <null>  WAIT <iolp> <spec list> ) |
   */
  case IO_STMT21:
    (void)misc_io_checks("WAIT");
    iomsg_check();
    UNIT_CHECK;
    kwd_errchk(BT_WAIT);
    PT_CHECK(PT_ID, astb.ptr0);
    sptr = mk_iofunc(RTE_f90io_wait, DT_INT, 0);
    fix_iostat();
    (void)begin_io_call(A_FUNC, sptr, 4);
    (void)add_io_arg(PTARG(PT_UNIT));
    (void)add_io_arg(mk_cval(bitv, DT_INT));
    (void)add_io_arg(PTARG(PT_IOSTAT));
    (void)add_io_arg(PTARG(PT_ID));
    ast = end_io_call();
    ast = add_cgoto(ast);
    external_io = TRUE;
    nondevice_io = TRUE;
    goto end_IO_STMT;
  /*
   *	<IO stmt> ::= <null>  FLUSH <unit info>
   */
  case IO_STMT22:
    (void)misc_io_checks("FLUSH");
    iomsg_check();
    kwd_errchk(BT_FLUSH);
    rtlRtn = RTE_f90io_flush;
    goto rewind_shared;

  /* ------------------------------------------------------------------ */
  /*
   *	<unit info> ::= <unit id> |
   */
  case UNIT_INFO1:
    noparens = TRUE;
    break;
  /*
   *	<unit info> ::= <iolp> <unit data list> )
   */
  case UNIT_INFO2:
    break;

  /*
   *	<unit info> ::= <iolp> <unit id> )
   */
  case UNIT_INFO3:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<unit data list> ::= <unit data list> , <unit data> |
   */
  case UNIT_DATA_LIST1:
    break;
  /*
   *	<unit data list> ::= <unit id> , <unit data> |
   */
  case UNIT_DATA_LIST2:
    break;
  /*
   *	<unit data list> ::= <unit data>
   */
  case UNIT_DATA_LIST3:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<unit data> ::= UNIT = <unit id> |
   */
  case UNIT_DATA1:
    PT_SET(PT_UNIT);
    break;
  /*
   *	<unit data> ::= ERR = <reflabel> |
   */
  case UNIT_DATA2:
    bitv |= BITV_ERR;
    PT_SET(PT_ERR);
    PTV(PT_ERR) = SST_SYMG(RHS(3)); /* sptr not ast */
    nondevice_io = TRUE;
    break;
  /*
   *	<unit data> ::= IOSTAT = <var ref> |
   */
  case UNIT_DATA3:
    bitv |= BITV_IOSTAT;
    PT_SET(PT_IOSTAT);
    chk_var(RHS(3), PT_IOSTAT, DT_INT);
    PTV(PT_IOSTAT) = SST_ASTG(RHS(3));
    set_assn(sym_of_ast(SST_ASTG(RHS(3))));
    nondevice_io = TRUE;
    break;
  /*
   *	<unit data> ::= IOMSG = <var ref>
   */
  case UNIT_DATA4:
    bitv |= BITV_IOMSG;
    PT_SET(PT_IOMSG);
    chk_var(RHS(3), PT_IOMSG, DT_CHAR);
    PTV(PT_IOMSG) = SST_ASTG(RHS(3));
    nondevice_io = TRUE;
    break;

  /*
   *	<unit data> ::= NEWUNIT = <var ref> |
   */
  case UNIT_DATA5:
    if (PTV(PT_UNIT) && !PTS(PT_NEWUNIT)) {
      IOERR2(201, "UNIT and NEWUNIT are mutually exclusive");
      break;
    }
    PT_SET(PT_NEWUNIT);
    chk_var(RHS(3), PT_NEWUNIT, SST_DTYPEG(RHS(3)));
    PTV(PT_UNIT) = PTV(PT_NEWUNIT);
    PT_VARREF(PT_UNIT, PTVARREF(PT_NEWUNIT));
    set_assn(sym_of_ast(SST_ASTG(RHS(3))));
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<unit id> ::= <expression> |
   */
  case UNIT_ID1:
    nondevice_io = TRUE;
    chk_unitid(RHS(1));
    break;

  /*
   *	<unit id> ::= *
   */
  case UNIT_ID2:
    if (scn.stmtyp != TK_READ && scn.stmtyp != TK_WRITE &&
        scn.stmtyp != TK_PRINT)
      IOERR2(201, "UNIT=*");
    chk_unitid((SST *)NULL);
    unit_star = TRUE;
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<spec list> ::= <spec list> , <spec item> |
   */
  case SPEC_LIST1:
    break;
  /*
   *	<spec list> ::= <spec item>
   */
  case SPEC_LIST2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<spec item> ::= <unit data> |
   */
  case SPEC_ITEM1:
    break;
  /*
   *	<spec item> ::= STATUS = <expression>  |
   */
  case SPEC_ITEM2:
    PT_SET(PT_STATUS);
    chk_expr(RHS(3), PT_STATUS, DT_CHAR);
    nondevice_io = TRUE;
    break;
  /*
   *	<spec item> ::= FILE = <expression>    |
   */
  case SPEC_ITEM3:
    PT_SET(PT_FILE);
    filename_type = TY_CHAR;
    if (DTY(SST_DTYPEG(RHS(3))) == TY_NCHAR) {
      filename_type = TY_NCHAR;
      chk_expr(RHS(3), PT_FILE, DT_NCHAR);
    } else
      chk_expr(RHS(3), PT_FILE, DT_CHAR);
    nondevice_io = TRUE;
    break;
  /*
   *	<spec item> ::= NAME = <expression>    |
   */
  case SPEC_ITEM4:
    /* string of either CHAR or NCHAR type may be used for file name: */
    PT_SET(PT_NAME);
    filename_type = TY_CHAR;
    if (DTY(SST_DTYPEG(RHS(3))) == TY_NCHAR) {
      filename_type = TY_NCHAR;
      chk_expr(RHS(3), PT_NAME, DT_NCHAR);
    } else
      chk_expr(RHS(3), PT_NAME, DT_CHAR);
    nondevice_io = TRUE;
    break;
  /*
   *	<spec item> ::= ACCESS = <expression>  |
   */
  case SPEC_ITEM5:
    PT_SET(PT_ACCESS);
    chk_expr(RHS(3), PT_ACCESS, DT_CHAR);
    nondevice_io = TRUE;
    break;
  /*
   *	<spec item> ::= FORM = <expression>    |
   */
  case SPEC_ITEM6:
    PT_SET(PT_FORM);
    chk_expr(RHS(3), PT_FORM, DT_CHAR);
    nondevice_io = TRUE;
    break;
  /*
   *	<spec item> ::= RECL = <expression>    |
   */
  case SPEC_ITEM7:
    i = PT_RECL;
    dtype = DT_INT8;
    nondevice_io = TRUE;
    goto var_or_expr_spec;
  /*
   *	<spec item> ::= BLANK = <expression>   |
   */
  case SPEC_ITEM8:
    i = PT_BLANK;
    dtype = DT_CHAR;
    rw03 = TRUE;
    nondevice_io = TRUE;
    goto inq_var_or_expr_spec;
  /*
   *	<spec item> ::= DISPOSE = <expression> |
   */
  case SPEC_ITEM9:
    PT_SET(PT_DISPOSE);
    if (flg.standard)
      error(171, 2, gbl.lineno, "DISPOSE", CNULL);
    chk_expr(RHS(3), PT_DISPOSE, DT_CHAR);
    nondevice_io = TRUE;
    break;
  /*
   *	<spec item> ::= READONLY                |
   */
  case SPEC_ITEM10:
    PT_SET(PT_ACTION);
    PTV(PT_ACTION) = mk_cnst(getstring("read", strlen("read")));
    nondevice_io = TRUE;
    break;
  /*
   *	<spec item> ::= READ = <var ref>       |
   */
  case SPEC_ITEM11:
    PT_SET(PT_READ);
    chk_var(RHS(3), PT_READ, DT_CHAR);
    nondevice_io = TRUE;
    break;
  /*
   *	<spec item> ::= WRITE = <var ref>      |
   */
  case SPEC_ITEM12:
    PT_SET(PT_WRITE);
    chk_var(RHS(3), PT_WRITE, DT_CHAR);
    nondevice_io = TRUE;
    break;
  /*
   *	<spec item> ::= READWRITE = <var ref>   |
   */
  case SPEC_ITEM13:
    PT_SET(PT_READWRITE);
    chk_var(RHS(3), PT_READWRITE, DT_CHAR);
    nondevice_io = TRUE;
    break;
  /*
   *	<spec item> ::= ACTION = <expression>   |
   */
  case SPEC_ITEM14:
    i = PT_ACTION;
    dtype = DT_CHAR;
    nondevice_io = TRUE;
    goto var_or_expr_spec;
  /*
   *	<spec item> ::= DELIM = <expression>    |
   */
  case SPEC_ITEM15:
    i = PT_DELIM;
    dtype = DT_CHAR;
    rw03 = TRUE;
    nondevice_io = TRUE;
    goto inq_var_or_expr_spec;
  /*
   *	<spec item> ::= PAD = <expression>      |
   */
  case SPEC_ITEM16:
    i = PT_PAD;
    dtype = DT_CHAR;
    rw03 = TRUE;
    nondevice_io = TRUE;
    goto inq_var_or_expr_spec;
  /*
   *	<spec item> ::= POSITION = <expression> |
   */
  case SPEC_ITEM17:
    i = PT_POSITION;
    dtype = DT_CHAR;
    nondevice_io = TRUE;
  /* goto var_or_expr_spec; */
  var_or_expr_spec:
    PT_SET(i);
    if (scn.stmtyp == TK_OPEN)
      chk_expr(RHS(3), i, dtype);
    else
      chk_var(RHS(3), i, dtype);
    break;
  /*
   *	<spec item> ::= EXIST = <var ref>  |
   */
  case SPEC_ITEM18:
    PT_SET(PT_EXIST);
    chk_var(RHS(3), PT_EXIST, DT_LOG);
    nondevice_io = TRUE;
    break;
  /*
   *	<spec item> ::= OPENED = <var ref> |
   */
  case SPEC_ITEM19:
    PT_SET(PT_OPENED);
    chk_var(RHS(3), PT_OPENED, DT_LOG);
    nondevice_io = TRUE;
    break;
  /*
   *	<spec item> ::= NUMBER = <var ref> |
   */
  case SPEC_ITEM20:
    PT_SET(PT_NUMBER);
    chk_var(RHS(3), PT_NUMBER, DT_INT8);
    nondevice_io = TRUE;
    break;
  /*
   *	<spec item> ::= NAMED = <var ref>  |
   */
  case SPEC_ITEM21:
    PT_SET(PT_NAMED);
    chk_var(RHS(3), PT_NAMED, DT_LOG);
    nondevice_io = TRUE;
    break;
  /*
   *	<spec item> ::= SEQUENTIAL = <var ref>   |
   */
  case SPEC_ITEM22:
    PT_SET(PT_SEQUENTIAL);
    chk_var(RHS(3), PT_SEQUENTIAL, DT_CHAR);
    nondevice_io = TRUE;
    break;
  /*
   *	<spec item> ::= DIRECT = <var ref>       |
   */
  case SPEC_ITEM23:
    PT_SET(PT_DIRECT);
    chk_var(RHS(3), PT_DIRECT, DT_CHAR);
    nondevice_io = TRUE;
    break;
  /*
   *	<spec item> ::= FORMATTED = <var ref>    |
   */
  case SPEC_ITEM24:
    PT_SET(PT_FORMATTED);
    chk_var(RHS(3), PT_FORMATTED, DT_CHAR);
    nondevice_io = TRUE;
    break;
  /*
   *	<spec item> ::= UNFORMATTED = <var ref>  |
   */
  case SPEC_ITEM25:
    PT_SET(PT_UNFORMATTED);
    chk_var(RHS(3), PT_UNFORMATTED, DT_CHAR);
    nondevice_io = TRUE;
    break;
  /*
   *	<spec item> ::= NEXTREC = <var ref>      |
   */
  case SPEC_ITEM26:
    PT_SET(PT_NEXTREC);
    chk_var(RHS(3), PT_NEXTREC, DT_INT8);
    nondevice_io = TRUE;
    break;
  /*
   *	<spec item> ::= FMT = <format id>  |
   */
  case SPEC_ITEM27:
    PT_SET(PT_FMT);
    nondevice_io = TRUE;
    break;
  /*
   *	<spec item> ::= REC = <expression> |
   */
  case SPEC_ITEM28:
    PT_SET(PT_REC);
    chk_expr(RHS(3), PT_REC, DT_INT);
    nondevice_io = TRUE;
    break;
  /*
   *	<spec item> ::= End = <reflabel>      |
   */
  case SPEC_ITEM29:
    bitv |= BITV_END;
    PT_SET(PT_END);
    PTV(PT_END) = SST_SYMG(RHS(3)); /* sptr not ast */
    nondevice_io = TRUE;
    break;
  /*
   *	<spec item> ::= NML = <ident>      |
   */
  case SPEC_ITEM30:
    nondevice_io = TRUE;
    PT_SET(PT_NML);
    sptr = refsym(SST_SYMG(RHS(3)), OC_OTHER);
    if (STYPEG(sptr) != ST_NML) {
      IOERR2(213, SYMNAME(sptr));
      break;
    }
  nml_io:
    PTV(PT_NML) = mk_id(get_nml_array(sptr));
    if (fmttyp != FT_UNFORMATTED) {
      if (fmttyp == FT_NML)
        IOERR2(202, "NML");
      else
        IOERR(212);
    }
    fmttyp = FT_NML;
    nml_group = sptr;
    REFP(sptr, 1); /* ref ==> must be initialized */
    for (i = CMEMFG(sptr); i; i = NML_NEXT(i)) {
      sptr = NML_SPTR(i);
      if (SCG(sptr) == SC_NONE)
        sem_set_storage_class(sptr);
    }
    break;
  /*
   *	<spec item> ::= <expression>       |
   */
  case SPEC_ITEM31:
    nondevice_io = TRUE;
    if (SST_IDG(RHS(1)) == S_IDENT) {
      if (STYPEG(SST_SYMG(RHS(1))) == ST_NML) {
        sptr = refsym(SST_SYMG(RHS(1)), OC_OTHER);
        goto nml_io;
      }
    }
    if (!PTV(PT_UNIT))
      chk_unitid(RHS(1));
    else
      chk_fmtid(RHS(1));
    break;
  /*
   *	<spec item> ::= * |
   */
  case SPEC_ITEM32:
    if (scn.stmtyp != TK_READ && scn.stmtyp != TK_WRITE &&
        scn.stmtyp != TK_PRINT)
      IOERR2(201, "UNIT=*");
    if (PTV(PT_UNIT))
      chk_fmtid((SST *)NULL);
    else {
      chk_unitid((SST *)NULL);
      unit_star = TRUE;
    }
    break;
  /*
   *	<spec item> ::= ADVANCE = <expression> |
   */
  case SPEC_ITEM33:
    if (DI_IN_NEST(sem.doif_depth, DI_DOCONCURRENT))
      error(1050, ERR_Severe, gbl.lineno,
            "I/O statement ADVANCE specifier in", CNULL);
    nondevice_io = TRUE;
    PT_SET(PT_ADVANCE);
    bitv |= BITV_ADVANCE;
    chk_expr(RHS(3), PT_ADVANCE, DT_CHAR);
    break;
  /*
   *	<spec item> ::= EOR = <reflabel> |
   */
  case SPEC_ITEM34:
    nondevice_io = TRUE;
    bitv |= BITV_EOR;
    PT_SET(PT_EOR);
    PTV(PT_EOR) = SST_SYMG(RHS(3)); /* sptr not ast */
    break;
  /*
   *	<spec item> ::= CONVERT = <expression> |
   */
  case SPEC_ITEM35:
    nondevice_io = TRUE;
    PT_SET(PT_CONVERT);
    chk_expr(RHS(3), PT_CONVERT, DT_CHAR);
    break;
  /*
   *	<spec item> ::= SHARED |
   */
  case SPEC_ITEM36:
    nondevice_io = TRUE;
    PT_SET(PT_SHARED);
    PTV(PT_SHARED) = mk_cnst(getstring("shared", strlen("shared")));
    break;
  /*
   *	<spec item> ::= ID = <var ref>              |
   */
  case SPEC_ITEM37:
    nondevice_io = TRUE;
    PT_SET(PT_ID);
    chk_var(RHS(3), PT_ID, DT_INT);
    break;
  /*
   *	<spec item> ::= PENDING = <var ref>         |
   */
  case SPEC_ITEM38:
    nondevice_io = TRUE;
    PT_SET(PT_PENDING);
    chk_var(RHS(3), PT_PENDING, DT_LOG);
    break;
  /*
   *	<spec item> ::= POS = <expression>          |
   */
  case SPEC_ITEM39:
    nondevice_io = TRUE;
    i = PT_POS;
    dtype = DT_INT8;
    goto inq_var_or_expr_spec;
  /*
   *	<spec item> ::= SIZE = <var ref> |
   */
  case SPEC_ITEM40:
    nondevice_io = TRUE;
    PT_SET(PT_SIZE);
    bitv |= BITV_SIZE;
    gen_spec_item_tmp(RHS(3), PT_SIZE, DT_INT8);
    break;
  /*
   *	<spec item> ::= ASYNCHRONOUS = <expression> |
   */
  case SPEC_ITEM41:
    nondevice_io = TRUE;
    i = PT_ASYNCHRONOUS;
    dtype = DT_CHAR;
    goto inq_var_or_expr_spec;
  /*
   *	<spec item> ::= DECIMAL = <expression>      |
   */
  case SPEC_ITEM42:
    nondevice_io = TRUE;
    i = PT_DECIMAL;
    dtype = DT_CHAR;
    rw03 = TRUE;
    open03 = TRUE;
    goto inq_var_or_expr_spec;
  /*
   *	<spec item> ::= ENCODING = <expression>     |
   */
  case SPEC_ITEM43:
    nondevice_io = TRUE;
    i = PT_ENCODING;
    dtype = DT_CHAR;
    rw03 = TRUE;
    open03 = TRUE;
    goto inq_var_or_expr_spec;
    break;
  /*
   *	<spec item> ::= SIGN = <expression>         |
   */
  case SPEC_ITEM44:
    nondevice_io = TRUE;
    i = PT_SIGN;
    dtype = DT_CHAR;
    open03 = TRUE;
    rw03 = TRUE;
    goto inq_var_or_expr_spec;
  /*
   *      <spec item> ::= STREAM = <var ref>          |
   */
  case SPEC_ITEM45:
    nondevice_io = TRUE;
    PT_SET(PT_STREAM);
    chk_var(RHS(3), PT_STREAM, DT_CHAR);
    break;
  /*
   *	<spec item> ::= ROUND = <expression>
   */
  case SPEC_ITEM46:
    nondevice_io = TRUE;
    i = PT_ROUND;
    dtype = DT_CHAR;
    open03 = TRUE;
    rw03 = TRUE;
  /*  fall thru  */
  inq_var_or_expr_spec:
    PT_SET(i);
    if (scn.stmtyp == TK_INQUIRE)
      chk_var(RHS(3), i, dtype);
    else
      chk_expr(RHS(3), i, dtype);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<format id> ::= <expression> |
   */
  case FORMAT_ID1:
    nondevice_io = TRUE;
    chk_fmtid(RHS(1));
    break;
  /*
   *	<format id> ::= *
   */
  case FORMAT_ID2:
    chk_fmtid((SST *)NULL);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<fid or nid> ::= <expression> |
   */
  case FID_OR_NID1:
    if (!XBIT(137, 0x010000))
      nondevice_io = TRUE;
    if (SST_IDG(RHS(1)) == S_IDENT) {
      sptr = refsym(SST_SYMG(RHS(1)), OC_OTHER);
      if (STYPEG(sptr) == ST_NML)
        goto nml_io;
    }
    chk_fmtid(RHS(1));
    break;
  /*
   *	<fid or nid> ::= *
   */
  case FID_OR_NID2:
    chk_fmtid((SST *)NULL);
    unit_star = TRUE;
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<print spec> ::= <fid or nid>
   */
  case PRINT_SPEC1:
    chk_unitid((SST *)NULL);
    noparens = TRUE;
    if (unit_star)
      print_star = TRUE;
    goto io_init;

  /* ------------------------------------------------------------------ */
  /*
   *	<io spec> ::= <iolp> <spec list> ) <optional comma> |
   */
  case IO_SPEC1:
    goto io_init;

  /* ------------------------------------------------------------------ */
  /*
   *	<read spec2> ::= <iolp> <spec list> ) |
   */
  case READ_SPEC21:
    goto io_init;
  /*
   *	<read spec2> ::= <fid or nid> |
   */
  case READ_SPEC22:
  /* fall thru */

  /* ------------------------------------------------------------------ */
  /*
   *	<read spec3> ::= <format id>
   */
  case READ_SPEC31:
  /* fall thru */

  /* ------------------------------------------------------------------ */
  /*
   *	<read spec4> ::= <fid or nid>
   */
  case READ_SPEC41:
    chk_unitid((SST *)NULL);
    noparens = TRUE;

  io_init:
    if (PTS(PT_POS) && PTS(PT_REC)) {
      IOERR2(201, "Both POS and REC cannot be specified");
    }
    UNIT_CHECK;
    PT_CHECK(PT_REC, astb.ptr0);

    if (PTV(PT_POS)) {
      ast = PTV(PT_POS);
      if (A_DTYPEG(ast) != DT_INT8)
        ast = mk_convert(ast, DT_INT8);
      ast1 = mk_unop(OP_VAL, astb.i1, DT_INT4);
      ast2 = mk_unop(OP_VAL, ast, DT_INT8);
      sptr = mk_iofunc(RTE_f90io_aux_init, DT_INT, 0);
      (void)begin_io_call(A_CALL, sptr, 2);
      (void)add_io_arg(ast1);
      (void)add_io_arg(ast2);
      (void)add_stmt_after(io_call.ast, (int)STD_PREV(0));
    }
    iomsg_check();
    if (fmttyp == FT_UNFORMATTED) {
      if (intern)
        IOERR(211);
      if (PTV(PT_ASYNCHRONOUS) || PTV(PT_ID)) {
        PT_CHECK(PT_ASYNCHRONOUS, astb.ptr0c);
        PT_CHECK(PT_ID, astb.ptr0);
        sptr = mk_iofunc(RTE_f90io_unf_asynca, DT_INT, 0);
        (void)begin_io_call(A_CALL, sptr, 2);
        (void)add_io_arg(PTARG(PT_ASYNCHRONOUS));
        (void)add_io_arg(PTARG(PT_ID));
        (void)add_stmt_after(io_call.ast, (int)STD_PREV(0));
      }
      sptr = mk_iofunc(unf_nm[LARGE_ARRAY_IDX][BYTE_SWAPPED_IO_IDX].init,
                       DT_INT, 0);
      fix_iostat();
      (void)begin_io_call(A_FUNC, sptr, 5);
      (void)add_io_arg(mk_cval(is_read, DT_INT));
      (void)add_io_arg(PTARG(PT_UNIT));
      (void)add_io_arg(PTARG(PT_REC));
      (void)add_io_arg(mk_cval(bitv, DT_INT));
      (void)add_io_arg(PTARG(PT_IOSTAT));
      ast = end_io_call();
      if (rw03) {
        if (PTV(PT_BLANK))
          IOERR2(201, PTNAME(PT_BLANK));
        if (PTV(PT_DECIMAL))
          IOERR2(201, PTNAME(PT_DECIMAL));
        if (PTV(PT_DELIM))
          IOERR2(201, PTNAME(PT_DELIM));
        if (PTV(PT_PAD))
          IOERR2(201, PTNAME(PT_PAD));
        if (PTV(PT_ROUND))
          IOERR2(201, PTNAME(PT_ROUND));
        if (PTV(PT_SIGN))
          IOERR2(201, PTNAME(PT_SIGN));
      }
    } else { /*  FORMATTED I/O  */
      if (fmttyp == FT_LIST_DIRECTED)
        PTV(PT_FMT) = astb.ptr0;
      else if (fmttyp == FT_NML)
        PTV(PT_FMT) = astb.ptr0;
      else if (fmttyp == FT_FMTSTR) {
        /*
         * a character variable be initialized with the character
         * constant construct from the edit list in the FORMAT
         * statement.  Can't treat this the same as FT_CHARACTER
         * because we may not know the actual length at the time
         * of the read/write statement.
         * Just issue a call to the encode routine which expects a
         * a character variable and record the fact that it was called.
         */
        if (fmt_is_var) {
          /*
           * integer variable containing the address of a character
           * string.
           */
          fmt_is_var = 0;
          sptr = mk_iofunc(RTE_f90io_encode_fmtv, DT_INT, 0);
          (void)begin_io_call(A_FUNC, sptr, 1);
        } else {
          sptr = mk_iofunc(RTE_f90io_encode_fmta, DT_INT, 0);
          (void)begin_io_call(A_FUNC, sptr, 3);
          (void)add_io_arg(mk_cval((INT)ty_to_lib[TY_CHAR], DT_INT));
          (void)add_io_arg(astb.i1);
        }
        (void)add_io_arg(PTARG(PT_FMT));
        ast = end_io_call();
        PTV(PT_FMT) = astb.ptr0;
        fmttyp = FT_CHARACTER; /* FMTSTR is now the same as character*/
      } else if (fmttyp != FT_ENCODED) {
        /*
         * if an unencoded format string is to be used in this
         * statement, first issue a call to the encode routine
         * and record the fact that it was called.
         */
        sptr = mk_iofunc(RTE_f90io_encode_fmta, DT_INT, 0);
        ast = PTV(PT_FMT);
        ast1 = mk_cval((INT)ty_to_lib[DTYG(A_DTYPEG(ast))], DT_INT);
        (void)begin_io_call(A_FUNC, sptr, 3);
        (void)add_io_arg(ast1); /* kind of format string */
        if (fmt_length)
          (void)add_io_arg(fmt_length); /* number of elements */
        else
          /*
           * if fmt_length is zero, then the unencoded format
           * character string is just a scalar, constant, subscripted
           * reference, or substring reference.  Use the calling
           * convention to pass its length.
           */
          (void)add_io_arg(astb.i1);
        (void)add_io_arg(ast); /* format string */
        ast = end_io_call();
        PTV(PT_FMT) = astb.ptr0;
      }
      if (intern) { /*  INTERNAL I/O  */
                    /*
                     * construct AST's to represent the number of array
                     * elements of the character array or variable specified
                     * as the unit id and the size of each record.
                     */
        LOGICAL is_array;
        LOGICAL noncontig; /* non-contiguous array */
        int asd, i, ndim;

        noncontig = FALSE;
        ast = ast3 = PTV(PT_UNIT);
        if (DTY(A_DTYPEG(ast)) != TY_ARRAY)
          is_array = FALSE;
        else
          is_array = TRUE;
      again:
        switch (A_TYPEG(ast)) {
        case A_ID:
          sptr = A_SPTRG(ast);
          goto do_intern;
        case A_MEM:
          sptr = A_SPTRG(A_MEMG(ast));
          if (is_array && DTY(DTYPEG(sptr)) != TY_ARRAY) {
            noncontig = TRUE;
            break;
          }
        do_intern: /* 'ast' is a scalar or whole array */
          if (DTY(DTYPEG(sptr)) != TY_ARRAY) /* a scalar */
            ast2 = astb.i1;
          else if (ASUMSZG(sptr))
            ast2 = astb.i0;
          else if (POINTERG(sptr)) {
            /* A possible section.  Copy it to a contiguous
             * temporary.
             */
            noncontig = TRUE;
          } else {
            ast2 = AD_NUMELM(AD_PTR(sptr));
          }
          break;

        case A_SUBSCR:
          /* assume an array element  (or an irregular section,
           * which is an error)
           */
          ast2 = astb.i1;
          asd = A_ASDG(ast);
          ndim = ASD_NDIM(asd);
          for (i = 0; i < ndim; ++i) {
            int ss = ASD_SUBS(asd, i);

            if (A_TYPEG(ss) == A_TRIPLE)
              noncontig = TRUE;
            else if (A_SHAPEG(ss)) {
              /* a vector subscript -- illegal! */
              error(155, 3, gbl.lineno,
                    "An internal file cannot be a character "
                    "array with a vector subscript",
                    "");
              noncontig = FALSE;
              is_array = FALSE;
              break;
            }
          }
          if (noncontig)
            break;
          noncontig = is_array;
          break;

        case A_SUBSTR:
          ast = A_LOPG(ast); /* the 'parent' char scalar or
                              * array section variable. 'ast3'
                              * stores the A_SUBSTR AST. */
          if (is_array) {
            noncontig = TRUE;
            break;
          }
          goto again;

        default:
          interr("semantio-int.io, bad ast", ast, 3);
        }
        if (noncontig) {
          /* A noncontiguous array:
           * +  array section
           * +  array substring
           * +  array%member
           * +  array%member(subscript)
           * + ...
           * Copy it to a contiguous temporary.
           */
          intern_array = ast3;
          intern_tmp = copy_replic_sect_to_tmp(intern_array);
          PTV(PT_UNIT) = intern_tmp;
          ast2 = AD_NUMELM(AD_PTR(A_SPTRG(intern_tmp)));
        }
        /*
         * since the UNIT is character, its length is passed as
         * the last argument.  We rely on the passing mechanism
         * to effect this.  IF we have a situation where this
         * isn't true, we'll need to make this section conditional
         * on the target.
         */
        fix_iostat();
        if (fmttyp == FT_LIST_DIRECTED) {
          if (is_read)
            rtlRtn = RTE_f90io_ldr_intern_inita;
          else
            rtlRtn = RTE_f90io_ldw_intern_inita;
          sptr = mk_iofunc(rtlRtn, DT_INT, 0);

          (void)begin_io_call(A_FUNC, sptr, 4);
          (void)add_io_arg(PTARG(PT_UNIT));
          (void)add_io_arg(ast2);
          (void)add_io_arg(mk_cval(bitv, DT_INT));
          (void)add_io_arg(PTARG(PT_IOSTAT));
          ast = end_io_call();
        } else if (fmttyp == FT_NML) { /*  Namelist I/O  */
          if (is_read)
            rtlRtn = RTE_f90io_nmlr_intern_inita;
          else
            rtlRtn = RTE_f90io_nmlw_intern_inita;
          sptr = mk_iofunc(rtlRtn, DT_INT, 0);

          (void)begin_io_call(A_FUNC, sptr, 4);
          (void)add_io_arg(PTARG(PT_UNIT));
          (void)add_io_arg(ast2);
          (void)add_io_arg(mk_cval(bitv, DT_INT));
          (void)add_io_arg(PTARG(PT_IOSTAT));
          ast = end_io_call();
        } else {
          if (is_read)
            rtlRtn = fmt_init[1].read[fmt_is_var];
          else
            rtlRtn = fmt_init[1].write[fmt_is_var];
          sptr = mk_iofunc(rtlRtn, DT_INT, 0);
          (void)begin_io_call(A_FUNC, sptr, 5);
          (void)add_io_arg(PTARG(PT_UNIT));
          (void)add_io_arg(ast2);
          (void)add_io_arg(mk_cval(bitv, DT_INT));
          (void)add_io_arg(PTARG(PT_IOSTAT));
          (void)add_io_arg(PTARG(PT_FMT));
          ast = end_io_call();
        }
      } else { /*  EXTERNAL I/O  */
        fix_iostat();
        if (fmttyp == FT_LIST_DIRECTED) {
          if (is_read)
            rtlRtn = RTE_f90io_ldr_init;
          else if (!print_star)
            rtlRtn = RTE_f90io_ldw_init;
          else
            rtlRtn = RTE_f90io_print_init;
          sptr = mk_iofunc(rtlRtn, DT_INT, 0);
          (void)begin_io_call(A_FUNC, sptr, 4);
          (void)add_io_arg(PTARG(PT_UNIT));
          (void)add_io_arg(PTARG(PT_REC));
          (void)add_io_arg(mk_cval(bitv, DT_INT));
          (void)add_io_arg(PTARG(PT_IOSTAT));
          ast = end_io_call();
        } else if (fmttyp == FT_NML) {
          if (is_read)
            rtlRtn = RTE_f90io_nmlr_init;
          else
            rtlRtn = RTE_f90io_nmlw_init;
          sptr = mk_iofunc(rtlRtn, DT_INT, 0);
          (void)begin_io_call(A_FUNC, sptr, 4);
          (void)add_io_arg(PTARG(PT_UNIT));
          (void)add_io_arg(PTARG(PT_REC));
          (void)add_io_arg(mk_cval(bitv, DT_INT));
          if (bitv & BITV_IOSTAT)
            (void)add_io_arg(PTARG(PT_IOSTAT));
          else
            (void)add_io_arg(mk_fake_iostat());
          ast = end_io_call();
        } else {
          if (is_read) {
            rtlRtn = fmt_init[0].read[fmt_is_var];
            sptr = mk_iofunc(rtlRtn, DT_INT, 0);
            (void)begin_io_call(A_FUNC, sptr, 7);
          } else {
            rtlRtn = fmt_init[0].write[fmt_is_var];
            sptr = mk_iofunc(rtlRtn, DT_INT, 0);
            (void)begin_io_call(A_FUNC, sptr, 6);
          }
          (void)add_io_arg(PTARG(PT_UNIT));
          (void)add_io_arg(PTARG(PT_REC));
          (void)add_io_arg(mk_cval(bitv, DT_INT));
          (void)add_io_arg(PTARG(PT_IOSTAT));
          (void)add_io_arg(PTARG(PT_FMT));
          if (is_read) {
            PT_CHECK(PT_SIZE, astb.ptr0);
            (void)add_io_arg(PTARG(PT_SIZE));
          }
          PT_CHECK(PT_ADVANCE, astb.ptr0c);
          (void)add_io_arg(PTARG(PT_ADVANCE));
          ast = end_io_call();
        }
      }
      if (rw03) {
        if (is_read) {
          if (PTV(PT_DELIM))
            IOERR2(201, PTNAME(PT_DELIM));
          if (PTV(PT_SIGN))
            IOERR2(201, PTNAME(PT_SIGN));
        } else {
          if (PTV(PT_BLANK))
            IOERR2(201, PTNAME(PT_BLANK));
          if (PTV(PT_PAD))
            IOERR2(201, PTNAME(PT_PAD));
        }
        if (fmttyp == FT_LIST_DIRECTED) {
          if (is_read) {
            rtlRtn = RTE_f90io_ldr_init03a;
            PT_CHECK(PT_BLANK, astb.ptr0c);
            PT_CHECK(PT_DECIMAL, astb.ptr0c);
            PT_CHECK(PT_PAD, astb.ptr0c);
            PT_CHECK(PT_ROUND, astb.ptr0c);
            sptr = mk_iofunc(rtlRtn, DT_INT, 0);
            (void)begin_io_call(A_FUNC, sptr, 5);
            (void)add_io_arg(A_DESTG(ast));
            (void)add_io_arg(PTARG(PT_BLANK));
            (void)add_io_arg(PTARG(PT_DECIMAL));
            (void)add_io_arg(PTARG(PT_PAD));
            (void)add_io_arg(PTARG(PT_ROUND));
          } else {
            rtlRtn = RTE_f90io_ldw_init03a;
            if (PTV(PT_ROUND))
              IOERR2(201, PTNAME(PT_ROUND));
            PT_CHECK(PT_DECIMAL, astb.ptr0c);
            PT_CHECK(PT_DELIM, astb.ptr0c);
            PT_CHECK(PT_SIGN, astb.ptr0c);
            sptr = mk_iofunc(rtlRtn, DT_INT, 0);
            (void)begin_io_call(A_FUNC, sptr, 4);
            (void)add_io_arg(A_DESTG(ast));
            (void)add_io_arg(PTARG(PT_DECIMAL));
            (void)add_io_arg(PTARG(PT_DELIM));
            (void)add_io_arg(PTARG(PT_SIGN));
          }
          ast = end_io_call();
        } else if (fmttyp == FT_NML) {
          if (is_read) {
            rtlRtn = RTE_f90io_nmlr_init03a;
            PT_CHECK(PT_BLANK, astb.ptr0c);
            PT_CHECK(PT_DECIMAL, astb.ptr0c);
            PT_CHECK(PT_PAD, astb.ptr0c);
            PT_CHECK(PT_ROUND, astb.ptr0c);
            sptr = mk_iofunc(rtlRtn, DT_INT, 0);
            (void)begin_io_call(A_FUNC, sptr, 5);
            (void)add_io_arg(A_DESTG(ast));
            (void)add_io_arg(PTARG(PT_BLANK));
            (void)add_io_arg(PTARG(PT_DECIMAL));
            (void)add_io_arg(PTARG(PT_PAD));
            (void)add_io_arg(PTARG(PT_ROUND));
          } else {
            rtlRtn = RTE_f90io_nmlw_init03a;
            if (PTV(PT_ROUND))
              IOERR2(201, PTNAME(PT_ROUND));
            PT_CHECK(PT_DECIMAL, astb.ptr0c);
            PT_CHECK(PT_DELIM, astb.ptr0c);
            PT_CHECK(PT_SIGN, astb.ptr0c);
            sptr = mk_iofunc(rtlRtn, DT_INT, 0);
            (void)begin_io_call(A_FUNC, sptr, 4);
            (void)add_io_arg(A_DESTG(ast));
            (void)add_io_arg(PTARG(PT_DECIMAL));
            (void)add_io_arg(PTARG(PT_DELIM));
            (void)add_io_arg(PTARG(PT_SIGN));
          }
          ast = end_io_call();
        } else {
          if (is_read) {
            rtlRtn = RTE_f90io_fmtr_init03a;
            PT_CHECK(PT_BLANK, astb.ptr0c);
            PT_CHECK(PT_DECIMAL, astb.ptr0c);
            PT_CHECK(PT_PAD, astb.ptr0c);
            PT_CHECK(PT_ROUND, astb.ptr0c);
            sptr = mk_iofunc(rtlRtn, DT_INT, 0);
            (void)begin_io_call(A_FUNC, sptr, 5);
            (void)add_io_arg(A_DESTG(ast));
            (void)add_io_arg(PTARG(PT_BLANK));
            (void)add_io_arg(PTARG(PT_DECIMAL));
            (void)add_io_arg(PTARG(PT_PAD));
            (void)add_io_arg(PTARG(PT_ROUND));
          } else {
            rtlRtn = RTE_f90io_fmtw_init03a;
            if (PTV(PT_DELIM))
              IOERR2(201, PTNAME(PT_DELIM));
            PT_CHECK(PT_DECIMAL, astb.ptr0c);
            PT_CHECK(PT_SIGN, astb.ptr0c);
            sptr = mk_iofunc(rtlRtn, DT_INT, 0);
            (void)begin_io_call(A_FUNC, sptr, 4);
            (void)add_io_arg(A_DESTG(ast));
            (void)add_io_arg(PTARG(PT_DECIMAL));
            (void)add_io_arg(PTARG(PT_SIGN));
            (void)add_io_arg(PTARG(PT_ROUND));
          }
          ast = end_io_call();
        }
        /*
         * ast is an A_ASN of the form
         * z_io = ..._init03(...)
         */
      }
    }
    ast = add_cgoto(ast);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<encode spec> ::= <iolp> <encode unit> <encode ctl> )
   */
  case ENCODE_SPEC1:
    if (flg.standard)
      error(171, 2, gbl.lineno, "ENCODE/DECODE", CNULL);
    if (fmttyp == FT_FMTSTR) {
      /*
       * a character variable be initialized with the character
       * constant construct from the edit list in the FORMAT
       * statement.  Can't treat this the same as FT_CHARACTER
       * because we may not know the actual length at the time
       * of the read/write statement.
       * Just issue a call to the encode routine which expects a
       * a character variable and record the fact that it was called.
       */
      if (fmt_is_var) {
        /*
         * integer variable contiaining address of a character
         * string.
         */
        fmt_is_var = 0;
        sptr = mk_iofunc(RTE_f90io_encode_fmtv, DT_INT, 0);
        (void)begin_io_call(A_FUNC, sptr, 1);
      } else {
        sptr = mk_iofunc(RTE_f90io_encode_fmta, DT_INT, 0);
        (void)begin_io_call(A_FUNC, sptr, 3);
        (void)add_io_arg(mk_cval((INT)ty_to_lib[TY_CHAR], DT_INT));
        (void)add_io_arg(astb.i1);
      }
      (void)add_io_arg(PTARG(PT_FMT));
      ast = end_io_call();
      PTV(PT_FMT) = astb.ptr0;
      fmttyp = FT_CHARACTER; /* FMTSTR is now the same as character*/
    } else if (fmttyp != FT_ENCODED) {
      /*
       * if an unencoded format string is to be used in this
       * statement, first issue a call to the encode routine
       * and record the fact that it was called.
       */
      sptr = mk_iofunc(RTE_f90io_encode_fmta, DT_INT, 0);
      ast = PTV(PT_FMT);
      ast1 = mk_cval((INT)ty_to_lib[DTYG(A_DTYPEG(ast))], DT_INT);
      (void)begin_io_call(A_FUNC, sptr, 3);
      (void)add_io_arg(ast1); /* kind of format string */
      if (fmt_length)
        (void)add_io_arg(fmt_length); /* number of elements */
      else
        (void)add_io_arg(astb.i1);
      (void)add_io_arg(ast); /* format string */
      ast = end_io_call();
      PTV(PT_FMT) = astb.ptr0;
    }
    ast1 = mk_cval(is_read, DT_INT); /* read flag */
    ast2 = astb.i1;                  /* # of records is always 1 */
    fix_iostat();
    /*
     * decode-/encode- specific init routines are called.
     * Need to pass the 'loc' of the buffer to avoid passing
     * the length of the buffer if character; this also implies
     * the corresponding argument of the called function will be
     * a '**'.  The actual length of the buffer is passed as an
     * explicit argument.
     */
    (void)sym_mkfunc_nodesc(mkRteRtnNm(RTE_loc), DT_ADDR);
    SST_UNITP(RHS(2), mk_unop(OP_LOC, SST_UNITG(RHS(2)), DT_PTR));
    if (fmttyp == FT_LIST_DIRECTED) {
      if (is_read)
        rtlRtn = RTE_f90io_ldr_intern_inite;
      else
        rtlRtn = RTE_f90io_ldw_intern_inite;
      sptr = mk_iofunc(rtlRtn, DT_INT, 0);
      (void)begin_io_call(A_FUNC, sptr, 5);
      (void)add_io_arg((int)SST_UNITG(RHS(2)));
      (void)add_io_arg(ast2);
      (void)add_io_arg(mk_cval(bitv, DT_INT));
      (void)add_io_arg(PTARG(PT_IOSTAT));
      (void)add_io_arg((int)SST_LENG(RHS(2)));
    } else {
      if (is_read)
        rtlRtn = fmt_inite.read[fmt_is_var];
      else
        rtlRtn = fmt_inite.write[fmt_is_var];
      sptr = mk_iofunc(rtlRtn, DT_INT, 0);
      (void)begin_io_call(A_FUNC, sptr, 6);
      (void)add_io_arg((int)SST_UNITG(RHS(2)));
      (void)add_io_arg(ast2);
      (void)add_io_arg(mk_cval(bitv, DT_INT));
      (void)add_io_arg(PTARG(PT_IOSTAT));
      (void)add_io_arg(PTARG(PT_FMT));
      (void)add_io_arg((int)SST_LENG(RHS(2)));
    }
    ast = end_io_call();
    ast = add_cgoto(ast);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<encode unit> ::= <expression> , <format id> , <var ref>
   */
  case ENCODE_UNIT1:
    if (is_varref(RHS(5))) {
      (void)mkarg(RHS(5), &dum);
      ast1 = SST_ASTG(RHS(5));
    } else {
      IOERR(217);
      ast1 = astb.i0;
    }
    (void)chk_scalartyp(RHS(1), DT_INT, FALSE);
    SST_LENP(LHS, SST_ASTG(RHS(1)));
    SST_UNITP(LHS, ast1);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<encode ctl> ::=   |
   */
  case ENCODE_CTL1:
    break;
  /*
   *	<encode ctl> ::= , <spec list>
   */
  case ENCODE_CTL2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<output list> ::= <output list> , <output item> |
   */
  case OUTPUT_LIST1:
    ((IOL *)SST_ENDG(RHS(1)))->next = (IOL *)SST_BEGG(RHS(3));
    SST_ENDP(RHS(1), SST_ENDG(RHS(3)));
    break;
  /*
   *	<output list> ::= <elp> <output list> , <output item> ) |
   */
  case OUTPUT_LIST2:
    ((IOL *)SST_ENDG(RHS(2)))->next = (IOL *)SST_BEGG(RHS(4));
    SST_ENDP(RHS(2), SST_ENDG(RHS(4)));
    *LHS = *RHS(2);
    break;
  /*
   *	<output list> ::= <output item>
   */
  case OUTPUT_LIST3:
    sem.defined_io_seen = 0;
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<output item> ::= <expression> |
   */
  case OUTPUT_ITEM1:
    dtype = SST_DTYPEG(RHS(1));
    if (DTY(dtype) == TY_ARRAY)
      dtype = DTY(dtype + 1);
    if (DTY(dtype) == TY_DERIVED && UNLPOLYG(DTY(dtype + 3))) {
      error(155, 4, gbl.lineno,
            "Unlimited polymorphic object not allowed"
            " in list directed I/O",
            CNULL);
    }
    if (DTYG(SST_DTYPEG(RHS(1))) == TY_DERIVED &&
        (A_TYPEG(SST_ASTG(RHS(1))) == A_FUNC ||
        /* Allocate a temporary ast to store the value of the derived type array
         * reference whose subscript is a function reference, otherwise the
         * function would be incorrectly called in each component I/O. */
        (A_TYPEG(SST_ASTG(RHS(1))) == A_SUBSCR &&
         A_CALLFGG(SST_ASTG(RHS(1)))))) {
      ast = sem_tempify(RHS(1));
      (void)add_stmt(ast);
      SST_IDP(RHS(1), S_IDENT);
      SST_ASTP(RHS(1), 0);
      SST_SYMP(RHS(1), A_SPTRG(A_DESTG(ast)));
    }
    if (DTY(dtype) == TY_DERIVED) {
      i = dtype_has_defined_io(dtype);
      if (i & (DT_IO_FWRITE | DT_IO_UWRITE)) {
        sem.defined_io_seen = 1;
      }

      sptr1 = 0;
      if (SST_IDG(RHS(1)) == S_SCONST) {
        sptr1 = SST_SYMG(RHS(1));
        if (STYPEG(sptr1) == ST_TYPEDEF) {
          sptr =
              getcctmp_sc('t', sem.dtemps++, ST_VAR, SST_DTYPEG(RHS(1)), io_sc);
          sptr = init_derived_w_acl(sptr, SST_ACLG(RHS(1)));
          SST_IDP(RHS(1), S_IDENT);
          SST_ASTP(RHS(1), 0);
          SST_SYMP(RHS(1), sptr);
          SST_ALIASP(RHS(1), 0);
          SST_CVLENP(RHS(1), 0);
          SST_SHAPEP(RHS(1), 0);
        }
      }
    }
    goto io_item_shared;
  /*
   *	<output item> ::= <elp> <output list> , <implied do control> )
   */
  case OUTPUT_ITEM2:
    goto implied_do_shared;

  /* ------------------------------------------------------------------ */
  /*
   *	<input list> ::= <input list> , <input item> |
   */
  case INPUT_LIST1:
    ((IOL *)SST_ENDG(RHS(1)))->next = (IOL *)SST_BEGG(RHS(3));
    SST_ENDP(RHS(1), SST_ENDG(RHS(3)));
    break;
  /*
   *	<input list> ::= <input item>
   */
  case INPUT_LIST2:
    sem.defined_io_seen = 0;
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<input item> ::= <var ref> |
   */
  case INPUT_ITEM1:
    if (SST_IDG(RHS(1)) != S_DERIVED && !is_varref(RHS(1)))
      IOERR(214);

    dtype = SST_DTYPEG(RHS(1));
    if (DTY(dtype) == TY_ARRAY)
      dtype = DTY(dtype + 1);
    if (DTY(dtype) == TY_DERIVED) {
      i = dtype_has_defined_io(dtype);
      if (i & (DT_IO_FREAD | DT_IO_UREAD)) {
        sem.defined_io_seen = 1;
      }
    }

  io_item_shared:
    dtype = SST_DTYPEG(RHS(1));

    ast1 = ast3 = SST_ASTG(RHS(1));
    if (A_TYPEG(ast1) == A_SUBSTR)
      ast1 = A_LOPG(ast1);
    if (SST_IDG(RHS(1)) == S_LVALUE && DTY(dtype) == TY_ARRAY &&
        A_TYPEG(ast1) == A_SUBSCR) {
      /*
       * If an input item is an array section containing vector
       * subscripts, the dimensions containing the vector subscripts
       * must be scalarized.  Ultimately, the io item will be classified
       * as an 'expression', and the ast for the item will be a subscript
       * ast bounded by do 'begin-end' pairs, one for each vector
       * subscript; if more than one vector subscript appears, the
       * resulting loops will be nested, where the left-most vector
       * subscript is the innermost loop, etc.  The bounds of each do
       * are derived from the shape of the index vector.
       */
      IOL *iol_var;    /* iol item representing the io item if any
                        * vector subscripts are present.
                        */
      LOGICAL isvec;   /* any subscripts which are triples */
      LOGICAL ischar;  /* base type is character */
      LOGICAL anydiff; /* any difficult subscripts found */
      LOGICAL anyvec;  /* any vector subscripts found */

      isvec = FALSE;
      ischar = FALSE;
      anydiff = FALSE;
      anyvec = FALSE;
      dtype = DTY(dtype + 1);
      if (DTY(dtype) == TY_CHAR || DTY(dtype) == TY_NCHAR)
        ischar = TRUE;
      iolptr = NULL; /* previous dobegin which was emitted */
      doend = NULL;  /* last doend emitted */
      asd = A_ASDG(ast1);
      numdim = ASD_NDIM(asd); /* get number of subscripts */
      for (i = 0; i < numdim; i++) {
        int asd2;
        /* look for difficult subscripts */
        ast2 = ASD_SUBS(asd, i);
        switch (A_TYPEG(ast2)) {
        case A_ID:
        case A_TRIPLE:
          break;
        case A_SUBSCR:
          /* if only a single dimension with triplet subscript, ok */
          asd2 = A_ASDG(ast2);
          if (ASD_NDIM(asd2) != 1) {
            anydiff = TRUE;
          } else if (A_TYPEG(ASD_SUBS(asd2, 0)) != A_TRIPLE) {
            anydiff = TRUE;
          }
          break;
        default:
          if (A_SHAPEG(ast2))
            anydiff = TRUE;
        }
      }
      for (i = 0; !anydiff && i < numdim; i++) {
        int shp;
        ast2 = ASD_SUBS(asd, i);
        if (A_TYPEG(ast2) == A_TRIPLE && !ischar)
          isvec = TRUE;
        else if ((shp = A_SHAPEG(ast2)) ||
                 (ischar && A_TYPEG(ast2) == A_TRIPLE)) {
          IOL *iol_tmp;
          anyvec = TRUE;
          /* vector subscript */
          if (iolptr == NULL) {
            /* create the semantic stack and IOL representing the
             * subscripted form of the input item.
             */
            stkptr = (SST *)getitem(0, sizeof(SST));
            *stkptr = *RHS(1);
            iol_var = (IOL *)getitem(0, sizeof(IOL));
            iol_var->id = IE_EXPR;
            iol_var->next = NULL;
            iol_var->element = stkptr;
            iol_var->l_std = 0;
            iolptr = iol_var;
          }
          /* create a DOINFO record for this vector; the index
           * variable is a compiler-created temporary and the bounds
           * are extracted from the shape of the index vector.
           */
          doinfo = get_doinfo(0);
          sptr = get_temp(DT_INT);
          doinfo->index_var = sptr;
          if (A_TYPEG(ast2) == A_TRIPLE) {
            doinfo->init_expr = A_LBDG(ast2);
            doinfo->limit_expr = A_UPBDG(ast2);
            doinfo->step_expr = A_STRIDEG(ast2);
            if (doinfo->step_expr == 0)
              doinfo->step_expr = astb.i1;
          } else {
            doinfo->init_expr = SHD_LWB(shp, 0);
            doinfo->limit_expr = SHD_UPB(shp, 0);
            doinfo->step_expr = SHD_STRIDE(shp, 0);
          }
          doinfo->count =
              mk_binop(OP_SUB, doinfo->limit_expr, doinfo->init_expr, DT_INT);
          doinfo->count =
              mk_binop(OP_ADD, doinfo->count, doinfo->step_expr, DT_INT);
          doinfo->count =
              mk_binop(OP_DIV, doinfo->count, doinfo->step_expr, DT_INT);
          /*
           * Create the DOBEGIN IOL for this index. Its 'next'
           * pointer is just the previous dobegin or the IOL
           * of the subscripted form.
           */
          dobegin = (IOL *)getitem(0, sizeof(IOL));
          dobegin->id = IE_DOBEGIN;
          dobegin->next = iolptr;
          dobegin->doinfo = doinfo;
          dobegin->l_std = 0;
          /*
           * Create the DOEND IOL for this index; ensure that
           * the previous doend, if present, is linked to this doend.
           * If this is the first do pair, the IOL representing the
           * item is linked to this doend.
           */
          iol_tmp = (IOL *)getitem(0, sizeof(IOL));
          if (doend)
            doend->next = iol_tmp;
          else
            iolptr->next = iol_tmp;
          doend = iol_tmp;
          doend->id = IE_DOEND;
          doend->next = NULL;
          doend->doinfo = doinfo;
          doend->l_std = 0;
          iolptr = dobegin;
          /*
           * prepare the subscript for this dimension; it's just
           * a subscripted reference of the index vector and the
           * subscript is the DO index variable.
           */
          dum = mk_id(sptr);
          if (A_TYPEG(ast2) == A_TRIPLE) {
            ast2 = dum;
          } else if (A_TYPEG(ast2) == A_SUBSCR) {
            /* must have been single subscript */
            ast2 = mk_subscr(A_LOPG(ast2), &dum, 1, DT_INT);
          } else {
            ast2 = mk_subscr(ast2, &dum, 1, DT_INT);
          }
        }
        subs[i] = ast2;
      }
      if (iolptr) {
        /* vector subscripts were found; if any triples were present,
         * the type of the reference is still array
         */
        dtype = SST_DTYPEG(RHS(1));
        if (!isvec)
          dtype = DTY(dtype + 1);
        /*
         * Create the subscripted form of the io item; fill in the
         * item's semantic stack entry.
         */
        ast1 = mk_subscr((int)A_LOPG(ast1), subs, numdim, dtype);
        stkptr = iol_var->element;
        SST_IDP(stkptr, S_LVALUE);
        SST_DTYPEP(stkptr, dtype);
        /* if this is a substring, recreate the substring */
        if (A_TYPEG(ast3) == A_SUBSTR)
          ast1 = mk_substr(ast1, A_LEFTG(ast3), A_RIGHTG(ast3), dtype);

        SST_ASTP(stkptr, ast1);
        if (!isvec)
          SST_SHAPEP(stkptr, 0);
        /*
         * define the begin and end fields for this item; they are
         * just the last dobegin and doend IOLs created.
         */
        SST_BEGP(LHS, (ITEM *)dobegin);
        SST_ENDP(LHS, (ITEM *)doend);
        break;
      }
    }
    count = 0;
    sptr1 = 0;
    if (SST_IDG(RHS(1)) == S_IDENT || SST_IDG(RHS(1)) == S_DERIVED)
      sptr1 = SST_SYMG(RHS(1));
    else if (SST_IDG(RHS(1)) == S_LVALUE)
      sptr1 = SST_LSYMG(RHS(1));

    dtype = SST_DTYPEG(RHS(1));
    if (sptr1 && DTYG(dtype) == TY_DERIVED) {
      /* derived,  get_derived_iolptrs() sets up LHS with a list of
         iolptrs representing the derived object*/

      dtype = DDTG(DTYPEG(sptr1));
      if (dtype_has_defined_io(dtype) & functype[is_read][fmttyp]) {
        stkptr = (SST *)getitem(0, sizeof(SST));
        *stkptr = *RHS(1);
        iolptr = (IOL *)getitem(0, sizeof(IOL));
        iolptr->id = IE_EXPR;
        iolptr->next = NULL;
        iolptr->element = stkptr;
        iolptr->l_std = 0;
        SST_BEGP(LHS, (ITEM *)iolptr);
        SST_ENDP(LHS, (ITEM *)iolptr);

        break;
      }
      (void)mkarg(RHS(1), &dum);
      sptr1 = refsym(sptr1, OC_OTHER);
      SST_SYMP(RHS(1), sptr1);
      get_derived_iolptrs(LHS, sptr1, RHS(1));
    } else {
      /*
       * guard against the item being c_loc()
       */
      if (DTY(dtype) == TY_DERIVED && A_TYPEG(ast1) == A_INTR &&
          is_iso_cptr(dtype)) {
        error(155, 3, gbl.lineno,
              "A derived type containing private "
              "components cannot be an I/O item",
              CNULL);
      }
      stkptr = (SST *)getitem(0, sizeof(SST));
      *stkptr = *RHS(1);
      iolptr = (IOL *)getitem(0, sizeof(IOL));
      iolptr->id = IE_EXPR;
      iolptr->next = NULL;
      iolptr->element = stkptr;
      iolptr->l_std = 0;
      SST_BEGP(LHS, (ITEM *)iolptr);
      SST_ENDP(LHS, (ITEM *)iolptr);
    }
    break;
  /*
   *	<input item> ::= <elp> <input list> ) |
   */
  case INPUT_ITEM2:
    *LHS = *RHS(2);
    break;
  /*
   *	<input item> ::= <elp> <input list> , <implied do control> )
   */
  case INPUT_ITEM3:
  implied_do_shared:
    doinfo = (DOINFO *)SST_BEGG(RHS(4));
    sptr = doinfo->index_var;
    if (flg.smp || flg.accmp)
      is_dovar_sptr(doinfo->index_var);
    dtype = DTYPEG(sptr);
    iolptr = (IOL *)SST_BEGG(RHS(2));

    if (DT_ISINT(dtype) && iolptr == (IOL *)SST_ENDG(RHS(2)) &&
        iolptr->id == IE_EXPR) {
      /*
       * check for an optimizable DO loop; the conditions which must be
       * satisfied are:
       * 1. The DO index is integer.
       * 2. the I/O list consists of a single subscripted array without
       *    a substring specification.  Also, the array may not be a
       *    structure member.
       * 3. one subscript is the DO index variable.
       * 4. the remaining subscripts are either ICON or ILD of a variable
       *    not the DO index variable.
       * 5. no subscript checking
       */

      /*
       * Determine if the I/O list item is a subscripted array --
       */
      stkptr = iolptr->element;
      if (SST_IDG(stkptr) == S_IDENT || SST_IDG(stkptr) == S_CONST ||
          SST_IDG(stkptr) == S_ACONST)
        goto not_optz;
      ast1 = SST_ASTG(stkptr);
      if (A_TYPEG(ast1) != A_SUBSCR)
        goto not_optz;
      if (DTY(A_DTYPEG(ast1)) == TY_CHAR || !DT_ISBASIC(A_DTYPEG(ast1)))
        goto not_optz;
      asd = A_ASDG(ast1);

      /* scan through the subscripts */

      numdim = ASD_NDIM(asd); /* get number of subscripts */
      dim = -1;
      for (i = 0; i < numdim; i++) {
        ast2 = ASD_SUBS(asd, i);
        switch (A_TYPEG(ast2)) {
        case A_CNST: /* constant is okay */
          break;
        case A_ID: /* check load */
          if (A_SPTRG(ast2) != sptr)
            break;
          if (dim >= 0) /* another use of index */
            goto not_optz;
          dim = i;
          break; /* subscript is okay */
        case A_CONV:
          ast3 = A_LOPG(ast2);
          if (A_TYPEG(ast3) != A_ID || A_SPTRG(ast3) != sptr || dim >= 0) {
            goto not_optz;
          }
          dim = i;
          break;
        default: /* anything else is not okay */
          goto not_optz;
        }
        subs[i] = ast2; /* remember subscript just in case */
      }
      if (dim < 0) /* index variable not used as subscript */
        goto not_optz;
      /*
       * Finally have determined that the implied DO can be optimized
       * into a single call -- a new subscript ast needs to be created
       * with a triple replacing the index variable in the corresponding
       * dimension.  The triple will represent the section
       *     <init_expr>:<limit_expr>:<step_expr>.
       * All other dimension will have a triple of <subs>:<subs>.
       */
      for (i = 0; i < numdim; i++)
        if (i == dim) {
          int l, u, s;
          l = doinfo->init_expr;
          u = doinfo->limit_expr;
          s = doinfo->step_expr;
          if (XBIT(68, 0x1) && A_DTYPEG(subs[dim]) == DT_INT8 &&
              size_of(A_DTYPEG(l)) < 8) {
            l = mk_convert(l, DT_INT8);
            u = mk_convert(u, DT_INT8);
            if (s)
              s = mk_convert(s, DT_INT8);
          }
          subs[dim] = mk_triple(l, u, s);
        } else
          subs[i] = ASD_SUBS(asd, i);
      ast1 = mk_subscr((int)A_LOPG(ast1), subs, numdim,
                       (int)A_DTYPEG(A_LOPG(ast1)));

      SST_DTYPEP(stkptr, A_DTYPEG(ast1));
      SST_ASTP(stkptr, ast1);
      iolptr->id = IE_OPTDO;
      iolptr->doinfo = doinfo;
      *LHS = *RHS(2);
      break;
    }

  not_optz:
    dobegin = (IOL *)getitem(0, sizeof(IOL));
    dobegin->id = IE_DOBEGIN;
    dobegin->next = iolptr;
    dobegin->doinfo = doinfo;
    dobegin->l_std = 0;
    doend = (IOL *)getitem(0, sizeof(IOL));
    doend->id = IE_DOEND;
    doend->next = NULL;
    doend->doinfo = doinfo;
    doend->l_std = 0;

    SST_BEGP(LHS, (ITEM *)dobegin);
    ((IOL *)(SST_ENDG(RHS(2))))->next = doend;
    SST_ENDP(LHS, (ITEM *)doend);
    /*
     * STDs could have been added since the beginning left paren.
     * Since the DO begin is generated later in the parse, these STDs
     * will appear before the DO.  For the STDs added since the beginning
     * left paren, remove these from the STD list and attach them to the
     * dobegin; when the DO is generated these STDs will be added after the
     * DO begin.  Examples of when STDs may be added are when referencing
     * functions/intrinsics which turn into 'calls'.
     */
    {
      int s, s1, s2, t;
      s = SST_ASTG(RHS(1));  /* STD generated when left paren seen */
      s1 = STD_NEXT(s);      /* STD generated after left paren */
      s2 = SST_ASTG(RHS(4)); /* STD generated before do expressions*/
      if (s != s2) {
        /* Unlink the list of stds added after the left paren and
         * before the implied do expressions and add to the dobegin's
         * iol structure. Stds s1 through s2, inclusive, mark the
         * list to be moved.
         *
         * Before:
         *     s <-> s1 <-> ... <-> s2 <-> t
         * After:
         *     s     s1 <-> ... <-> s2     t
         *     |                           |
         *     +<------------------------->+
         */
        dobegin->l_std = s1;
        t = STD_NEXT(s2);
        STD_NEXT(s) = t;
        STD_PREV(t) = s;
        STD_NEXT(s2) = 0;
      } else
        dobegin->l_std = 0;
    }

    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<format stmt> ::= <format> ( <format list> ) |
   */
  case FORMAT_STMT1:
  end_format:
    put_edit(FED_END); /* note that the last rp is not put out */
    PUT(rescan);
    sptr = SST_SYMG(RHS(1)); /* locate format list */
    PLLENP(sptr, fasize);    /* number of 32-bit elements */
    DINITP(sptr, 1);
    /* sym_is_refd(sptr);  REF set when it appears in a read/write stmt */
    dinit_put(DINIT_END, 0);
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<format stmt> ::= <format> ( ) |
   */
  case FORMAT_STMT2:
    rescan = 0;
    fasize = 2; /* number of 32-bit elements */
    goto end_format;
  /*
   *	<format stmt> ::= FORMAT <fmtstr> |
   */
  case FORMAT_STMT3:
    /*
     * The format array is actually a data initialized character variable.
     * The initializing value is the character constant constructed by the
     * scanner from the format edit list.
     */
    if (scn.currlab == 0)
      IOERR(203);
    sptr = get_fmt_array(scn.currlab, ST_VAR);
    FMTPTP(scn.currlab, sptr);
    DEFDP(scn.currlab, 1);
    scn.currlab = 0;
    sptr1 = SST_SYMG(RHS(2));
    DTYPEP(sptr, DTYPEG(sptr1));
    DINITP(sptr, 1);
    DCLDP(sptr, 1);
    HCCSYMP(sptr, 1);
    dinit_put(DINIT_LOC, sptr);
    dinit_put(DINIT_STR, (INT)sptr1);
    dinit_put(DINIT_END, 0);
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<format stmt> ::= <format> ( <format list unl> )
   */
  case FORMAT_STMT4:
    goto end_format;

  /* ------------------------------------------------------------------ */
  /*
   *	<format> ::= FORMAT
   */
  case FORMAT1:
    if (scn.currlab == 0)
      IOERR(203);
    sptr = get_fmt_array(scn.currlab, ST_PLIST);
    CCSYMP(sptr, 1);
    SCP(sptr, SC_STATIC);
    DTYPEP(sptr, DT_INT);
    FMTPTP(scn.currlab, sptr);
    DEFDP(scn.currlab, 1);
    scn.currlab = 0;
    dinit_put(DINIT_FMT, (INT)sptr);
#if DEBUG
    if (DBGBIT(3, 128))
      fprintf(gbl.dbgfil, "Format(%4d) %s:\n", gbl.lineno, getprint(sptr));
#endif
    fasize = 0;
    rescan = 0; /* default rescan is to the item after the very
                 * first left paren. NOTE: the outermost left
                 * parenthesis does not occur in the list */
    lastrpt = -999;
    last_edit = FED_LPAREN;
    edit_state = 0;
    SST_SYMP(LHS, sptr); /* pass up "array" of encoded format */
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<format list> ::= <format list> <format item> |
   */
  case FORMAT_LIST1:
    break;
  /*
   *	<format list> ::= <format item>
   */
  case FORMAT_LIST2:
    break;
  /*
   *    <format list> ::= <format list unl> <unlimited format item>
   */
  case FORMAT_LIST3:
    error(W_0548_Incorrect_use_of_unlimited_repetition,
          flg.standard ? ERR_Severe : ERR_Warning, gbl.lineno, CNULL, CNULL);
    break;
  /*
   *    <format list> ::= <format list unl> <format item>
   */
  case FORMAT_LIST4:
    error(W_0548_Incorrect_use_of_unlimited_repetition,
          flg.standard ? ERR_Severe : ERR_Warning, gbl.lineno, CNULL, CNULL);
    break;

  /*
   *    <format list unl> ::= <format list> <unlimited format item> |
   */
  case FORMAT_LIST_UNL1:
    break;
  /*
   *    <format list unl> ::= <unlimited format item> |
   */
  case FORMAT_LIST_UNL2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<format item> ::= ,  |
   */
  case FORMAT_ITEM1:
    if (flg.standard) {
      switch (edit_state) {
      case 0: /* initial state - process first edit descriptor */
        ERR170("comma after left parenthesis in FORMAT");
        edit_state = 2;
        break;
      case 1: /* an edit descr. was just seen */
        if (last_edit == FED_LPAREN)
          ERR170("comma after left parenthesis in FORMAT");
        edit_state = 2;
        break;
      case 2: /* a comma was just seen */
        ERR170("comma after comma in FORMAT");
        edit_state = 1;
        break;
      case 3: /* a colon or slash was just seen */
        edit_state = 2;
        break;
      }
    }
    last_edit = -101; /* fake code for ED_COMMA */
    break;
  /*
   *	<format item> ::= <repeat factor> /  |
   */
  case FORMAT_ITEM2:
    put_edit(FED_SLASH);
    break;
  /*
   *	<format item> ::= :  |
   */
  case FORMAT_ITEM3:
    put_edit(FED_COLON);
    break;
  /*
   *	<format item> ::= <kanji string>  |
   */
  case FORMAT_ITEM5:
    put_edit(FED_KANJI_STRING);
    sptr = SST_SYMG(RHS(1));
    goto string_shared;
  /*
   *	<format item> ::= <char literal>  |
   */
  case FORMAT_ITEM4:
    put_edit(FED_STR);
    sptr = SST_SYMG(RHS(1));
    goto string_shared;
  /*
   *	<format item> ::= <Hollerith>  |
   */
  case FORMAT_ITEM6:
    put_edit(FED_STR);
    sptr = CONVAL1G(SST_SYMG(RHS(1)));
  string_shared:
    dtype = DTYPEG(sptr);
    len = string_length(dtype);
    PUT(len);
    dinit_put(DT_CHAR, (INT)sptr);

    if (DTY(DT_INT) == TY_INT8) {
      /* pad out data init if len not divisible by 8 */
      len = (len + 7) & ~7;
      fasize += len >> 3;
    } else {
      /* pad out data init if len not divisible by 4 */
      len = (len + 3) & ~3;
      fasize += len >> 2;
    }
    break;
  /*
   *	<format item> ::= <repeat factor> <F1 item> |
   */
  case FORMAT_ITEM7:
    break;
  /*
   *	<format item> ::= <repeat factor> <F2 item> |
   */
  case FORMAT_ITEM8:
    break;
  /*
   *	<format item> ::= <scale factor>  |
   */
  case FORMAT_ITEM9:
    put_edit(FED_P);
    put_ffield(RHS(1));
    break;
  /*
   *	<format item> ::= <F3 item>
   */
  case FORMAT_ITEM10:
    break;
  /* ------------------------------------------------------------------ */
  /*
   *	<repeat factor> ::=   |
   */
  case REPEAT_FACTOR1:
    break;
  /*
   *	<repeat factor> ::= <ffield>
   */
  case REPEAT_FACTOR2:
    lastrpt = fasize;
    put_ffield(RHS(1));
    if (SST_IDG(RHS(1)) == S_CONST && SST_CVALG(RHS(1)) == 0)
      IOERR(206);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<scale factor> ::= <ffield> P  |
   */
  case SCALE_FACTOR1:
    break;
  /*
   *	<scale factor> ::= <addop> <ffield> P
   */
  case SCALE_FACTOR2:
    if (SST_OPTYPEG(RHS(1)) == OP_ADD)
      *LHS = *RHS(2);
    else { /* negate <ffield> */
#if DEBUG
      assert(SST_IDG(RHS(2)) == S_CONST, "semantio:exp.-<const>P", 0, 3);
#endif
      SST_IDP(LHS, S_CONST);
      SST_CVALP(LHS, -SST_CVALG(RHS(2)));
      SST_ASTP(LHS, mk_cval(SST_CVALG(LHS), DT_INT));
    }
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<F1 item> ::= <flp> <format list> ) |
   */
  case F1_ITEM1:
    rescan = SST_RNG2G(RHS(1));
    put_edit(FED_RPAREN);
    PUT(SST_RNG1G(RHS(1)));
    break;
  /*
   *	<F1 item> ::= I <ffield> |
   */
  case F1_ITEM2:
    put_edit(FED_Iw_m);
    put_ffield(RHS(2));
    PUT(0);
    PUT(1);
    break;
  /*
   *	<F1 item> ::= I <ffield> . <ffield> |
   */
  case F1_ITEM3:
    put_edit(FED_Iw_m);
    put_ffield(RHS(2));
    put_ffield(RHS(4));
    break;
  /*
   *	<F1 item> ::= A <ffield> |
   */
  case F1_ITEM4:
    put_edit(FED_Aw);
    put_ffield(RHS(2));
    break;
  /*
   *	<F1 item> ::= N <ffield> |
   */
  case F1_ITEM5:
    put_edit(FED_Nw);
    put_ffield(RHS(2));
    break;
  /*
   *	<F1 item> ::= L <ffield> |
   */
  case F1_ITEM6:
    put_edit(FED_Lw);
    put_ffield(RHS(2));
    break;
  /*
   *	<F1 item> ::= O <ffield> |
   */
  case F1_ITEM7:
    put_edit(FED_Ow_m);
    put_ffield(RHS(2));
    PUT(0);
    PUT(1);
    break;
  /*
   *	<F1 item> ::= O <ffield> . <ffield> |
   */
  case F1_ITEM8:
    put_edit(FED_Ow_m);
    put_ffield(RHS(2));
    put_ffield(RHS(4));
    break;
  /*
   *	<F1 item> ::= Z <ffield> |
   */
  case F1_ITEM9:
    put_edit(FED_Zw_m);
    put_ffield(RHS(2));
    PUT(0);
    PUT(1);
    break;
  /*
   *	<F1 item> ::= Z <ffield> . <ffield> |
   */
  case F1_ITEM10:
    put_edit(FED_Zw_m);
    put_ffield(RHS(2));
    put_ffield(RHS(4));
    break;
  /*
   *	<F1 item> ::= <aformat>  |
   */
  case F1_ITEM11:
    put_edit(FED_A);
    break;
  /*
   *	<F1 item> ::= <nformat>  |
   */
  case F1_ITEM12:
    put_edit(FED_N);
    break;
  /*
   *	<F1 item> ::= <lformat>  |
   */
  case F1_ITEM13:
    put_edit(FED_L);
    break;
  /*
   *	<F1 item> ::= <iformat>  |
   */
  case F1_ITEM14:
    put_edit(FED_I);
    break;
  /*
   *	<F1 item> ::= <oformat>  |
   */
  case F1_ITEM15:
    put_edit(FED_O);
    break;
  /*
   *	<F1 item> ::= <zformat>  |
   */
  case F1_ITEM16:
    put_edit(FED_Z);
    break;
  /*
   *	<F1 item> ::= B <ffield> |
   */
  case F1_ITEM17:
    put_edit(FED_Bw_m);
    put_ffield(RHS(2));
    PUT(0);
    PUT(1);
    break;
  /*
   *	<F1 item> ::= B <ffield> . <ffield> |
   */
  case F1_ITEM18:
    put_edit(FED_Bw_m);
    put_ffield(RHS(2));
    put_ffield(RHS(4));
    break;
  /*
   *	<F1 item> ::= <flp> <format list unl> )
   */
  case F1_ITEM19:
    error(W_0548_Incorrect_use_of_unlimited_repetition, ERR_Warning, gbl.lineno,
          CNULL, CNULL);
    rescan = SST_RNG2G(RHS(1));
    put_edit(FED_RPAREN);
    PUT(SST_RNG1G(RHS(1)));
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<flp> ::= (
   */
  case FLP1:
    SST_RNG1P(LHS, fasize + 1); /* point to item which follows the
                                 * left paren */
    if (lastrpt == fasize - 2)
      /*
       * a repeat count immediately preceded the paren. locate the
       * repeat count for the run-time (marks reversion).
       */
      SST_RNG2P(LHS, fasize - 2);
    else
      /* * o.w., just point to the left paren */
      SST_RNG2P(LHS, fasize);
    put_edit(FED_LPAREN);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<F2 item> ::= F <ffield> . <ffield> |
   */
  case F2_ITEM1:
    put_edit(FED_Fw_d);
    put_ffield(RHS(2));
    put_ffield(RHS(4));
    break;
  /*
   *	<F2 item> ::= E <ffield> . <ffield> |
   */
  case F2_ITEM2:
    put_edit(FED_Ew_d);
    put_ffield(RHS(2));
    put_ffield(RHS(4));
    break;
  /*
   *	<F2 item> ::= G <ffield> . <ffield> |
   */
  case F2_ITEM3:
    if (SST_CVALG(RHS(2)) == 0) {
      put_edit(FED_G0_d);
    } else {
      put_edit(FED_Gw_d);
    }
    put_ffield(RHS(2));
    put_ffield(RHS(4));
    break;
  /*
   *      <F2 item> ::= E <ffield> |
   */
  case F2_ITEM4:
    switch (last_edit) {
    case FED_Ew_d:
    case FED_Gw_d:
    case FED_ESw_d:
    case FED_ENw_d:
      put_edit(FED_Ee);
      put_ffield(RHS(2));
      break;
    default:
      IOERR(210);
      edit_state = 1;
      last_edit = FED_Ee;
    }
    break;

  /*
   *	<F2 item> ::= D <ffield> . <ffield> |
   */
  case F2_ITEM5:
    put_edit(FED_Dw_d);
    put_ffield(RHS(2));
    put_ffield(RHS(4));
    break;
  /*
   *	<F2 item> ::= <fformat>  |
   */
  case F2_ITEM6:
    put_edit(FED_F);
    break;
  /*
   *	<F2 item> ::= <eformat>  |
   */
  case F2_ITEM7:
    put_edit(FED_E);
    break;
  /*
   *	<F2 item> ::= <gformat>  |
   */
  case F2_ITEM8:
    put_edit(FED_G);
    break;
  /*
   *	<F2 item> ::= <dformat>  |
   */
  case F2_ITEM9:
    put_edit(FED_D);
    break;
  /*
   *	<F2 item> ::= EN <ffield> . <ffield> |
   */
  case F2_ITEM10:
    put_edit(FED_ENw_d);
    put_ffield(RHS(2));
    put_ffield(RHS(4));
    break;
  /*
   *	<F2 item> ::= ES <ffield> . <ffield>
   */
  case F2_ITEM11:
    put_edit(FED_ESw_d);
    put_ffield(RHS(2));
    put_ffield(RHS(4));
    break;
  /*
   *	<F2 item> ::= <dtformat>
   */
  case F2_ITEM12:
    put_edit(FED_DT);
    PUT(0);
    PUT(1);
    sptr = getstring("DT", 2);
    dtype = DTYPEG(sptr);
    len = string_length(dtype);
    PUT(0);
    PUT(len);
    dinit_put(DT_CHAR, (INT)sptr);
    if (DTY(DT_INT) == TY_INT8) {
      len = (len + 7) & ~7;
      fasize += len >> 3;
    } else {
      len = (len + 3) & ~3;
      fasize += len >> 2;
    }

    break;
  /*
   *	<F2 item> ::= DT <dts>
   */
  case F2_ITEM13:
    put_edit(FED_DT);

    if (SST_BEGG(RHS(2)) == 0 && SST_ENDG(RHS(2)) == 0) {
      char *str;
      /* <char_literal> */

      PUT(0);
      PUT(1);
      sptr = SST_SYMG(RHS(2));
      sptr = A_SPTRG(SST_ASTG(RHS(2)));
      strptr = malloc(strlen(stb.n_base + CONVAL1G(sptr)) + 3);

      str = strcpy(strptr, "DT");
      str = strcat(strptr, stb.n_base + CONVAL1G(sptr));
      sptr = getstring(str, strlen(strptr));
      dtype = DTYPEG(sptr);
      len = string_length(dtype);
      PUT(0);
      PUT(len);
      dinit_put(DT_CHAR, (INT)sptr);
      /* pad out data init if len not divisible by 8 */
      if (DTY(DT_INT) == TY_INT8) {
        len = (len + 7) & ~7;
        fasize += len >> 3;
      } else {
        len = (len + 3) & ~3;
        fasize += len >> 2;
      }

    } else if (SST_IDG(RHS(2)) == -1) {
      /* <vlist>  */
      PUT(0);
      PUT(2);
      sptr = getstring("DT", 2);
      dtype = DTYPEG(sptr);
      len = string_length(dtype);
      PUT(0);
      PUT(len);
      dinit_put(DT_CHAR, (INT)sptr);
      if (DTY(DT_INT) == TY_INT8) {
        len = (len + 7) & ~7;
        fasize += len >> 3;
      } else {
        len = (len + 3) & ~3;
        fasize += len >> 2;
      }
      put_vlist(RHS(2));
      break;
    } else {
      /* <char_literal> <vlist> */
      char *str;
      PUT(0);
      PUT(2);
      /*strcpy */
      sptr = A_SPTRG(SST_ASTG(RHS(2)));
      strptr = malloc(strlen(stb.n_base + CONVAL1G(sptr)) + 3);
      str = strcpy(strptr, "DT");
      str = strcat(strptr, stb.n_base + CONVAL1G(sptr));

      sptr = getstring(str, strlen(str));
      dtype = DTYPEG(sptr);
      len = string_length(dtype);
      PUT(0);
      PUT(len);
      dinit_put(DT_CHAR, (INT)sptr);
      /* pad out data init if len not divisible by 8 */
      if (DTY(DT_INT) == TY_INT8) {
        len = (len + 7) & ~7;
        fasize += len >> 3;
      } else {
        len = (len + 3) & ~3;
        fasize += len >> 2;
      }
      put_vlist(RHS(2));
    }

    break;
  /*
   *    <F2 item> ::= <g0format>
   */
  case F2_ITEM14:
    put_edit(FED_G0);
    break;


  /* ------------------------------------------------------------------ */
  /*
   *	<F3 item> ::= T <ffield>  |
   */
  case F3_ITEM1:
    put_edit(FED_T);
    put_ffield(RHS(2));
    break;
  /*
   *	<F3 item> ::= TL <ffield> |
   */
  case F3_ITEM2:
    put_edit(FED_TL);
    put_ffield(RHS(2));
    break;
  /*
   *	<F3 item> ::= TR <ffield> |
   */
  case F3_ITEM3:
    put_edit(FED_TR);
    put_ffield(RHS(2));
    break;
  /*
   *	<F3 item> ::= <ffield> X  |
   */
  case F3_ITEM4:
    put_edit(FED_X);
    put_ffield(RHS(1));
    break;
  /*
   *	<F3 item> ::= X  |
   */
  case F3_ITEM5:
    put_edit(FED_X);
    PUT(0);
    PUT(1);
    break;
  /*
   *	<F3 item> ::= S  |
   */
  case F3_ITEM6:
    put_edit(FED_S);
    break;
  /*
   *	<F3 item> ::= SP |
   */
  case F3_ITEM7:
    put_edit(FED_SP);
    break;
  /*
   *	<F3 item> ::= SS |
   */
  case F3_ITEM8:
    put_edit(FED_SS);
    break;
  /*
   *	<F3 item> ::= BN |
   */
  case F3_ITEM9:
    put_edit(FED_BN);
    break;
  /*
   *	<F3 item> ::= BZ |
   */
  case F3_ITEM10:
    put_edit(FED_BZ);
    break;
  /*
   *	<F3 item> ::= DC |
   */
  case F3_ITEM11:
    put_edit(FED_DC);
    break;
  /*
   *	<F3 item> ::= DP |
   */
  case F3_ITEM12:
    put_edit(FED_DP);
    break;
  /*
   *	<F3 item> ::= Q  |
   */
  case F3_ITEM13:
    put_edit(FED_Q);
    break;
  /*
   *	<F3 item> ::= '$' |
   */
  case F3_ITEM14:
    put_edit(FED_DOLLAR);
    break;
  /*
   *	<F3 item> ::= RU |
   */
  case F3_ITEM15:
    put_edit(FED_RU);
    break;
  /*
   *	<F3 item> ::= RD |
   */
  case F3_ITEM16:
    put_edit(FED_RD);
    break;
  /*
   *	<F3 item> ::= RZ |
   */
  case F3_ITEM17:
    put_edit(FED_RZ);
    break;
  /*
   *	<F3 item> ::= RN |
   */
  case F3_ITEM18:
    put_edit(FED_RN);
    break;
  /*
   *	<F3 item> ::= RC |
   */
  case F3_ITEM19:
    put_edit(FED_RC);
    break;
  /*
   *	<F3 item> ::= RP
   */
  case F3_ITEM20:
    put_edit(FED_RP);
    break;
  /* ------------------------------------------------------------------ */
  /*
   *	<dts> ::= <char_literal>
   */
  case DTS1:
    SST_CVALP(LHS, SST_CVALG(RHS(1)));
    SST_IDP(LHS, S_CONST);
    SST_DTYPEP(LHS, SST_DTYPEG(RHS(1)));
    SST_ASTP(LHS, mk_cval(SST_CVALG(LHS), SST_DTYPEG(RHS(1))));
    SST_SYMP(LHS, A_SPTRG(SST_ASTG(LHS)));

    SST_ENDP(LHS, 0);
    SST_BEGP(LHS, 0);

    break;
  /*
   *	<dts> ::= <char_literal> <dlp> <dt vlist> )
   */
  case DTS2:
    SST_BEGP(LHS, SST_BEGG(RHS(3)));
    SST_ENDP(LHS, SST_ENDG(RHS(3)));

    break;
  /*
   *	<dts> ::= <dlp> <dt vlist> )
   */
  case DTS3:
    *LHS = *RHS(2);
    SST_IDP(LHS, -1);

    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<dt vlist> ::= <dt vlist> , <addop> <integer>
   *	<dt vlist> ::= <dt vlist> , <integer>
   *	<dt vlist> ::= <addop> <integer>
   *	<dt vlist> ::= <integer>
   */
  case DT_VLIST1:
  case DT_VLIST2:
  case DT_VLIST3:
  case DT_VLIST4:
    count = DT_VLIST4 + 1 - rednum; // RHS symbol count:  4, 3, 2, or 1
    if ((count == 2 || count == 4) && SST_OPTYPEG(RHS(count - 1)) == OP_SUB)
      SST_CVALP(RHS(count), -SST_CVALG(RHS(count))); // negate <integer>
    e1 = (SST *)getitem(0, sizeof(SST));
    *e1 = *RHS(count);
    itemp = (ITEM *)getitem(0, sizeof(ITEM));
    itemp->next = ITEM_END;
    itemp->t.stkp = e1;
    if (count <= 2) {
      SST_BEGP(LHS, itemp);
    } else {
      (SST_ENDG(RHS(1)))->next = itemp;
    }
    SST_ENDP(LHS, itemp);

    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<ffield> ::= <integer>
   */
  case FFIELD1:
    SST_IDP(LHS, S_CONST);
    SST_ASTP(LHS, mk_cval(SST_CVALG(LHS), DT_INT));
    SST_DTYPEP(LHS, DT_INT);
    break;

  /*
   *	<unlimited format item> ::= <star repeat> <flp> <format list> )
   */
  case UNLIMITED_FORMAT_ITEM1:
    rescan = SST_RNG2G(RHS(2));
    put_edit(FED_RPAREN);
    PUT(SST_RNG1G(RHS(2)));
    break;

  /*
   *	<star repeat> ::= *
   */
  case STAR_REPEAT1:
    lastrpt = fasize;
    PUT(0);
    PUT(0x7fffffff);
    break;

  default:
    interr("semantio: bad rednum ", rednum, 3);
    break;
  }
  return;

end_IO_STMT:
  /*
   * common termination point for all I/O statments.
   */
  if (gbl.currsub && PUREG(gbl.currsub) && external_io)
    error(155, 2, gbl.lineno, SYMNAME(gbl.currsub),
          "- PURE subprograms may not contain external I/O statements");
  sem.io_stmt = FALSE;
  SST_ASTP(LHS, 0);
}

static int
set_io_sc(void)
{
  io_sc = SC_LOCAL;
  if (sem.task)
    io_sc = SC_PRIVATE;
  return io_sc;
}

/*
 * 'array_ast' is a character variable used as an internal I/O unit.
 * Its type is either:
 *  A_SUBSCR:  a regular section of a character array, or
 *  A_SUBSTR:  a substring whose parent is a regular section of a char array.
 * This copies 'array_ast' to a contiguous temporary array.  The
 * temporary array is replicated, as is the original array, since
 * distribution of character arrays isn't supported.
 */
static int
copy_replic_sect_to_tmp(int array_ast)
{
  int asn, std, tmp_ast, tmp_sptr;
  int dtype;
  int eldtype;
  int subscr[7];

  /* allocate(tmp(...))
   * tmp = array
   */
  switch (A_TYPEG(array_ast)) {
  case A_SUBSCR: /* a regular section */
    dtype = DTYPEG(sptr_of_subscript(array_ast));
    break;
  default: /* a substring of a regular section or just some array */
    dtype = A_DTYPEG(array_ast);
    break;
  }
  eldtype = DDTG(dtype);
  tmp_sptr = mk_shape_sptr(A_SHAPEG(array_ast), subscr, eldtype);
  tmp_ast = mk_id(tmp_sptr);

  asn = mk_assn_stmt(tmp_ast, array_ast, eldtype);
  std = add_stmt(asn);

  mk_mem_allocate(tmp_ast, subscr, std, array_ast);
  /* ...we call this after 'std = add_stmt( forall )' because
   * it adds stmts *before* 'std', so 'std' must be defined! */

  return tmp_ast;
}

/*
 * Copies back the temporary array created by a prior call to
 * 'copy_replic_sect_to_tmp' to the original regular section.
 */
static void
copy_back_to_replic_sect(int array_ast, int tmp_ast)
{
  int asn, std;
  int dtype;
  int eldtype;

  /* array = tmp
   * deallocate(tmp)
   */
  switch (A_TYPEG(array_ast)) {
  case A_SUBSCR: /* a regular section */
    dtype = DTYPEG(sptr_of_subscript(array_ast));
    break;
  default: /* a substring of a regular section or just some array */
    dtype = A_DTYPEG(array_ast);
    break;
  }
  eldtype = DDTG(dtype);
  asn = mk_assn_stmt(array_ast, tmp_ast, eldtype);

  std = add_stmt_after(asn, STD_LAST);

  mk_mem_deallocate(tmp_ast, std);
}

/* ensure iostat locates an AST */
static void
fix_iostat(void)
{
  if (PTV(PT_IOSTAT) == 0)
    PTV(PT_IOSTAT) = astb.i0;
}

static int
chk_SIZE_var()
{
  int asn = 0;

  if (PTS(PT_SIZE) && PTTMPUSED(PT_SIZE) && PTVARREF(PT_SIZE) != 1) {
    /* if SIZE specified, assign SIZE tmp back to user data item */
    asn = mk_assn_stmt(PTVARREF(PT_SIZE),
                       mk_convert(PTV(PT_SIZE), A_DTYPEG(PTVARREF(PT_SIZE))),
                       A_DTYPEG(PTVARREF(PT_SIZE)));
  }
  return asn;
}

/* Add test and branches for ERR=, END=, and EOR=.  */
static int
add_cgoto(int ast)
{
  int lbsptr;
  int last_ast;
  int inquire_cnt = 13;

  if (A_TYPEG(ast) == A_ASN)
    ast = A_DESTG(ast);
  last_ast = 0;
  if (strcmp(SYMNAME(A_SPTRG(A_LOPG(io_call.ast))),
             mkRteRtnNm(RTE_f90io_fmtr_end)) == 0 ||
      strncmp(SYMNAME(A_SPTRG(A_LOPG(io_call.ast))),
              mkRteRtnNm(RTE_f90io_inquirea), inquire_cnt) == 0) {
    int asn = chk_SIZE_var();
    if (asn) {
      add_stmt_after(asn, (int)STD_PREV(0));
      PT_TMPUSED(PT_SIZE, 0);
    }
  }
  if ((lbsptr = PTV(PT_ERR)) != 0) /* == 1 => error */
    last_ast = end_or_err(ast, lbsptr, astb.i1);
  if ((lbsptr = PTV(PT_END)) != 0) /* == 2 ==> end of file */
    last_ast = end_or_err(ast, lbsptr, mk_cval((INT)2, DT_INT));
  if ((lbsptr = PTV(PT_EOR)) != 0) /* == 3 ==> end of record */
    last_ast = end_or_err(ast, lbsptr, mk_cval((INT)3, DT_INT));

  return last_ast;
}

/* src   - ast of variable assigned the value of the func */
/* elab  - label of END=, ERR=, or EOR= */
/* eqval - ast of value which is compared with for equality */
/*
 * END= or ERR=; generate the ast sequence to test the value of an
 * i/o function and branch to the END=, ERR=, or EOR= label.  We're
 * always comparing for inequality):
 *     if ( io_call() .neq. eqval ) goto no-error-lab
 *     <code to set-up/clean-up for error>
 *     goto err_lab
 *   no-error-lab:
 * NOTE: must end an i/o critical section if end or err is taken.
 */
static int
end_or_err(int src, int elab, int eqval)
{
  int tmp, ast2;
  int ast;
  int astlab;
  int std;
  int sz_asgn = chk_SIZE_var();
  int no_error, ifexpr;

  tmp = mk_binop(OP_EQ, src, eqval, DT_LOG);
  /*
   * if condition is false, branch to no-error case - branch around
   * critical section code code.
   */

  no_error = getlab();
  ast = mk_stmt(A_IF, 0);
  ifexpr = mk_unop(OP_LNOT, tmp, DT_LOG);
  A_IFEXPRP(ast, ifexpr);
  ast2 = mk_stmt(A_GOTO, 0);
  astlab = mk_label(no_error);
  A_L1P(ast2, astlab);
  A_IFSTMTP(ast, ast2);
  ast = add_stmt_after(ast, (int)STD_PREV(0));
  if (sz_asgn) {
    add_stmt_after(sz_asgn, (int)STD_PREV(0));
  }
  ast = fio_end_err(0, elab); /* fall-thru to critical section end & br code */
  RFCNTI(no_error);
  ast = mk_stmt(A_CONTINUE, 0);
  std = add_stmt_after(ast, (int)STD_PREV(0));
  STD_LABEL(std) = no_error;

  return ast;
}

/*
 * If a ERR= or END= is taken and we need to "end" the i/o critical section,
 * a special block is written:
 *     new-err/end-label:
 *        fio_end();
 *        goto err/end-label;
 * NOTE that the conditional branch has been changed to reflect the new
 * label.
 */
/* lab  - label of block of code to call fio_end & branch;
 *        may be 0 -- block is not labeled
 */
/* elab - label of branch target */
static int
fio_end_err(int lab, int elab)
{
  int ast;
  int astlab;
  if (flg.smp || flg.accmp || XBIT(125, 0x1)) {
    if (flg.smp || flg.accmp)
      begin_io_call(A_CALL, sym_mkfunc_nodesc("_mp_ecs_nest", DT_NONE), 0);
    else
      (void)begin_io_call(A_CALL, mk_iofunc(RTE_f90io_end, DT_NONE, 0), 0);
    ast = end_io_call();
  }
  if (lab)
    STD_LABEL(A_STDG(ast)) = lab;
  ast = mk_stmt(A_GOTO, 0);
  astlab = mk_label(elab);
  A_L1P(ast, astlab);
  (void)add_stmt_after(ast, (int)STD_PREV(0));

  return ast;
}

static void
chk_expr(SST *stk, int p, int dtype)
{
  int dt;

  if (PTV(p))
    IOERR2(202, PTNAME(p));

  if (is_varref(stk))
    PT_VARREF(p, 1);

  if (DTY(dtype) == TY_CHAR || DTY(dtype) == TY_NCHAR)
    (void)mkarg(stk, &dt);
  else if (DT_ISINT(dtype)) {
    (void)chktyp(stk, dtype, FALSE);
    dt = SST_DTYPEG(stk);
  } else { /* pass by value:  */
    (void)mkexpr(stk);
    dt = SST_DTYPEG(stk);
  }
  if (DTY(dt) != DTY(dtype))
    IOERR2(201, PTNAME(p));
  PTV(p) = SST_ASTG(stk);
}

static void
gen_spec_item_tmp(SST *stk, int p, int dtype)
{
  int dt;
  int sptr;
  int i;

  (void)mkarg(stk, &dt);
  PT_VARREF(p, SST_ASTG(stk));
  sptr = getcctmp_sc('t', sem.dtemps++, ST_VAR, DTY(dtype), io_sc);
  SST_IDP(stk, S_IDENT);
  SST_SYMP(stk, sptr);
  SST_DTYPEP(stk, DTY(dtype));

  (void)mkarg(stk, &dt);
  i = SST_ASTG(stk);
  PTV(p) = SST_ASTG(stk);

  if (A_TYPEG(PTVARREF(p)) == A_ID) {
    DOCHK(A_SPTRG(i));
  }
}

static void
chk_var(SST *stk, int p, int dtype)
{
  int dt;
  int sptr;

  if (PTV(p)) /* repeated specifier */
    IOERR2(202, PTNAME(p));

  if (is_varref(stk)) {
    int i;
    PT_VARREF(p, 1);
    if (DTY(SST_DTYPEG(stk)) != DTY(dtype)) {
      if ((DT_ISINT(dtype) && DT_ISINT(SST_DTYPEG(stk))) ||
          (DT_ISLOG(dtype) && DT_ISLOG(SST_DTYPEG(stk)))) {
        (void)mkarg(stk, &dt);
        PT_VARREF(p, SST_ASTG(stk));
        sptr = getcctmp_sc('t', sem.dtemps++, ST_VAR, DTY(dtype), io_sc);
        SST_IDP(stk, S_IDENT);
        SST_SYMP(stk, sptr);
        SST_DTYPEP(stk, DTY(dtype));
      } else {
        IOERR2(201, PTNAME(p));
        SST_ASTP(stk, astb.i0);
      }
    }

    (void)mkarg(stk, &dt);
    i = SST_ASTG(stk);
    PTV(p) = SST_ASTG(stk);

    if (A_TYPEG(PTVARREF(p)) == A_ID) {
      DOCHK(A_SPTRG(i));
    }
  } else {
    IOERR2(201, PTNAME(p));
    SST_ASTP(stk, astb.i0);
  }
}

static void
chk_unitid(SST *stk)
{
  int dtype;
  INT unit;

  if (PTS(PT_NEWUNIT))
    IOERR2(201, "UNIT and NEWUNIT are mutually exclusive");
  else if (PTV(PT_UNIT)) /* error if repeated unit specification */
    IOERR2(202, "UNIT");

  if (stk == NULL) {
    /*
     * Unit id is the default unit ('*')
     */
    if (XBIT(124, 0x4)) /* use -5 (stdin) or -6 (stdout) */
      unit = -DEFAULT_UNIT;
    else /* use 5 or 6 */
      unit = DEFAULT_UNIT;
    PTV(PT_UNIT) = mk_cval(unit, DT_INT);
  } else {
    dtype = SST_DTYPEG(stk);
    if (DT_ISINT(dtype))
      /*
       * Unit id is an integer expression
       */
      (void)chktyp(stk, DT_INT, FALSE);
    else if (DTYG(dtype) == TY_CHAR &&
             (SST_IDG(stk) == S_IDENT || SST_IDG(stk) == S_LVALUE)) {
      /*
       * Unit id specifies an internal file -- a character variable,
       * a character array, a character array element, or a character
       * substring.
       */
      (void)mkarg(stk, &dtype);
      intern = TRUE;
      if (no_rw)
        IOERR(211);
      else if (!is_read && gbl.currsub && PUREG(gbl.currsub)) {
        int sptr;
        sptr = sym_of_ast(SST_ASTG(stk));
        if (SCG(sptr) == SC_CMBLK)
          error(155, 2, gbl.lineno, SYMNAME(gbl.currsub),
                "- PURE subprograms may not contain internal WRITE statements "
                "where the file unit is a common block variable");
      }
    } else
      IOERR2(201, "UNIT");
    PTV(PT_UNIT) = SST_ASTG(stk);
  }
}

static void
chk_fmtid(SST *stk)
{
  int dtype;
  int sptr, lab;
  ADSC *ad;

  fmt_is_var = 0;
  if (fmttyp != FT_UNFORMATTED)
    IOERR2(202, "FMT");
  if (stk == NULL) {
    fmttyp = FT_LIST_DIRECTED;
    PTV(PT_FMT) = astb.i0;
    if (!is_read && unit_star)
      print_star = TRUE;
  } else {
    dtype = SST_DTYPEG(stk);
    if (legal_labelvar(dtype)) {
      /* If requested, a string is constructed by the scanner
       * from the edit list in the FORMAT statement.
       */
      if (XBIT(58, 0x200))
        /* similar check exists in scan.c:get_fmtstr() */
        fmttyp = FT_FMTSTR;
      else
        fmttyp = FT_ENCODED;
      if (SST_IDG(stk) == S_CONST) {
        /*
         * format id is an integer constant representing the statement
         * label of a FORMAT statement.  First, derive the "name" of the
         * label as in the <label> = <integer> reduction.
         */
        long labno =
            DTY(dtype) == TY_INT8 ? CONVAL2G(SST_CVALG(stk)) : SST_CVALG(stk);
        lab = declref(getsymf(".L%05ld", labno), ST_LABEL, 'r');
        RFCNTI(lab);
        /*
         * link into list of referenced labels if not already
         * there
         */
        if (SYMLKG(lab) == NOSYM) {
          SYMLKP(lab, sem.flabels);
          sem.flabels = lab;
        }
        if (flg.xref)
          xrefput(lab, 'r');
        sptr = get_fmt_array(lab, ST_PLIST);
        REFP(sptr, 1);
        PTV(PT_FMT) = mk_id(sptr);
        return;
      }
      if (SST_IDG(stk) == S_IDENT) {
        /*
         * format id is an integer variable containing a label address
         * For 64-byte targets, create a temp variable iff the type
         * of the user variable is integer*4.
         * When targeting llvm, always create a temp variable of
         * ptr-size integer type.
         */
        (void)mklabelvar(stk);
        PTV(PT_FMT) = SST_ASTG(stk);
        fmt_is_var = 1;
        return;
      }
    }
    /* format id is not a label or integer variable (encoded format case).
     * Now, it's either a character string expression (which is the
     * standard case) containing a non-encoded format specification.
     * Or, a non-standard case is it's a variable, array element, or
     * array which contains a hollerith constant containing a non-encoded
     * format specification.
     */
    fmttyp = FT_CHARACTER;
    if (DTYG(dtype) == TY_CHAR) { /* standard case */
      (void)mkarg(stk, &dtype);
      if (DTY(dtype) == TY_ARRAY) {
        fmt_length = astb.i1; /* TBD: need LEN SST_ASTG(stk)) */
        ad = AD_DPTR(dtype);
        if (AD_ASSUMSZ(ad) == 0)
          fmt_length = AD_NUMELM(ad);
        else
          IOERR(215);
      } else
        /*
         * mark non-encoded form as a character scalar, constant,
         * element, or substring (LET later processing determine
         * the length.
         */
        fmt_length = 0;
    } else if (SST_IDG(stk) == S_IDENT || SST_IDG(stk) == S_LVALUE) {
      (void)mkarg(stk, &dtype);
      fmt_length = astb.i1;
      if (DTY(dtype) == TY_ARRAY) {
        switch (A_TYPEG(SST_ASTG(stk))) {
        case A_ID:
        case A_MEM:
          ad = AD_DPTR(dtype);
          if (AD_ASSUMSZ(ad) == 0)
            fmt_length = AD_NUMELM(ad);
          else
            IOERR(215);
          break;
        default:
          IOERR2(201, "FMT");
        }
      }
      if (flg.standard)
        error(170, 2, gbl.lineno, "nonstandard format specifier", CNULL);
    } else
      IOERR2(201, "FMT");
    PTV(PT_FMT) = SST_ASTG(stk);
  }
}

static void
chk_iospec()
{
  LOGICAL advancingspecified = FALSE;
  int adv_ast;

  if (!flg.standard)
    return;

  if (bitv & BITV_ADVANCE) {
    adv_ast = PTV(PT_ADVANCE);
    if (A_TYPEG(adv_ast) == A_CNST && DTY(A_DTYPEG(adv_ast)) == TY_CHAR &&
        (strcmp(stb.n_base + CONVAL1G(A_SPTRG(adv_ast)), "YES") == 0 ||
         strcmp(stb.n_base + CONVAL1G(A_SPTRG(adv_ast)), "yes") == 0)) {
      advancingspecified = TRUE;
    }
  }

  if (intern && bitv & BITV_ADVANCE) {
    IOERR2(201, "internal unit with ADVANCE specifier");
  }
  if (bitv & BITV_EOR) {
    if (!(bitv & BITV_ADVANCE)) {
      IOERR2(201, "EOR specified without ADVANCE");
    } else if (advancingspecified) {
      IOERR2(201, "EOR specified with ADVANCE=YES");
    }
  }
  if (bitv & BITV_SIZE) {
    if (!(bitv & BITV_ADVANCE)) {
      IOERR2(201, "SIZE specified without ADVANCE");
    } else if (advancingspecified) {
      IOERR2(201, "SIZE specified with ADVANCE=YES");
    }
  }
}

static void
put_edit(int code)
{
  PUT(code);
  /*
   * If the standard switch is used, save enough state must be saved
   * while processing edit descriptors to check for delimiter conformance.
   * For this task, the edit processing will be in one of 4 states:
   *   0 - initial state: a left paren was just seen.
   *   1 - an edit descriptor was just seen, a delimiter or right paren is
   *       expected.
   *   2 - a comma was just seen (this state is entered by the semantic
   *       actions for "<format item> ::= ,"
   *   3 - a colon or slash delimiter was just seen.
   *   4 - a edit descriptor DT was just seen ???
   */
  if (flg.standard) {
    switch (edit_state) {
    case 0: /* initial state - process edit descriptor after left paren */
      switch (code) {
      case FED_LPAREN:
        break;
      case FED_COLON:
      case FED_SLASH:
        edit_state = 3;
        break;
      default:
        edit_state = 1;
      }
      break;
    case 1: /* an edit descr. was just seen */
      switch (code) {
      case FED_END:
      case FED_RPAREN:
        break;
      case FED_COLON:
      case FED_SLASH:
        edit_state = 3;
        break;
      default:
        if (last_edit == FED_P && (code == FED_Fw_d || code == FED_Ew_d ||
                                   code == FED_Dw_d || code == FED_Gw_d))
          ;
        else if (code == FED_Ee)
          ; /* checking has already been done for what Ee can
             * follow */
        else {
          ERR170("missing delimiter in FORMAT");
          edit_state = 3;
        }
        break;
      }
      break;
    case 2: /* a comma was just seen */
      if (code == FED_RPAREN)
        ERR170("right parenthesis occurred after comma in FORMAT");
      FLANG_FALLTHROUGH;
    /*
     * fall through for state transition
     */
    case 3: /* a colon or slash was just seen */
      switch (code) {
      case FED_LPAREN:
        edit_state = 0;
        break;
      case FED_COLON:
      case FED_SLASH:
        edit_state = 3;
        break;
      default:
        edit_state = 1;
      }
      break;
    }
    switch (code) { /* check for any of the edit extensions */
    case FED_Q:
    case FED_DOLLAR:
      ERR170("using nonstandard edit descriptor");
      break;
    case FED_Z:
    case FED_O:
    case FED_L:
    case FED_I:
    case FED_F:
    case FED_E:
    case FED_G:
    case FED_D:
      ERR170("using edit descriptor without specifying width");
      break;
    }
  }
  last_edit = code;
}

static void
put_ffield(SST *stk)
{
  if (SST_IDG(stk) == S_CONST) {
    PUT(0);
    PUT(SST_CVALG(stk));
  } else
    interr("put_ffield, not cnst, stkid", SST_IDG(stk), 3);
}

static void
put_vlist(SST *stk)
{
  int i = 0;
  ITEM *iptr;

  if (SST_BEGG(stk) == 0) {
    interr("put_vlist, vlist is empty", 0, 3);
  } else {
    for (iptr = SST_BEGG(stk); iptr != ITEM_END; iptr = iptr->next)
      i++;

    PUT(0);
    PUT(i);

    /* This field will be used by runtime: RTE_f90io_dts_fmtr/w */
    PUT(0);
    PUT(0);

    for (iptr = SST_BEGG(stk); iptr != ITEM_END; iptr = iptr->next) {
      PUT((SST_CVALG(iptr->t.stkp)));
      PUT(0);
    }
  }
}

static void
kwd_errchk(int bt)
{
  int i;

  for (i = 0; i <= PT_MAXV; i++)
    if ((PTS(i) && !(PTSTMT(i) & bt)) ||
        (bt == BT_INQUIRE && i > 1 && i != PT_ERR && i != PT_END && PTS(i) &&
         !PTVARREF(i)))
      IOERR2(201, PTNAME(i));
}

static void
_put(INT n, int dtype)
{
  static const char *desc[] = {
      " ",        "END",     "LP",      "RP",      "K",       "STR",
      "T",        "TL",      "TR",      "X",       "S",       "SP",
      "SS",       "BN",      "BZ",      "SLASH",   "COLON",   "Q",
      "DOLLAR",   "A",       "L",       "I",       "F",       "Ee",
      "E",        "EN",      "ES",      "G",       "D",       "O",
      "Z",        "AFORMAT", "LFORMAT", "IFORMAT", "OFORMAT", "ZFORMAT",
      "FFORMAT",  "EFORMAT", "GFORMAT", "DFORMAT", "B",       "DT",
      "DTFORMAT",
  };

  if (DBGBIT(3, 128)) {
    fprintf(gbl.dbgfil, "    %4d: ", fasize);
    if (n < 0 && (-n) < (sizeof(desc) / sizeof(char *)))
      fprintf(gbl.dbgfil, "%s\n", desc[-n]);
    else
      fprintf(gbl.dbgfil, "%ld\n", (long)n);
  }
  if (DTY(dtype) == TY_INT8)
    n = cngcon(n, DT_INT4, DT_INT8);
  dinit_put(dtype, n);
  fasize++;
}

char *
cntl_name(int p)
{
  static char buf[32];
  int len;

  len = strlen(PTNAME(p));
  strcpy(buf, PTNAME(p));
  buf[len] = '=';
  buf[len + 1] = '\0';

  return buf;
}

const char *
ed_name(int ed)
{
  static const char *desc[] = {
      " ",  "END", "(",  ")",  "P",  "STR", "T", "TL", "TR", "X", "S",
      "SP", "SS",  "BN", "BZ", "/",  ":",   "Q", "$",  "A",  "L", "I",
      "F",  "E",   "E",  "EN", "ES", "G",   "D", "O",  "Z",  "A", "L",
      "I",  "F",   "E",  "G",  "D",  "O",   "Z", "B",  "DT",
  };

  return desc[ed];
}

/* create the array (ST_PLIST) representing the format; name is
 * derived from its label
 */
static int
get_fmt_array(int lab, int stype)
{
  char *nm, *p;
  char name[12]; /* z__fmtddddd */
  int sptr;

  nm = SYMNAME(lab);
  nm += 2; /* skip past .L */
  while (*nm == '0')
    nm++; /* skip over leading 0's */
  strcpy(name, "z__fmt");
  p = name + 6;
  while ((*p++ = *nm++))
    ;
  sptr = getsymbol(name);
  if (stype == ST_PLIST) {
    STYPEP(sptr, ST_PLIST);
    /* there may have been a format statement from the outer procedure with
     * the same label.  If this is an internal procedure, make a new symbol
     */
    if (gbl.internal > 1 && !INTERNALG(sptr) && INTERNALG(lab)) {
      sptr = insert_sym(sptr);
      STYPEP(sptr, ST_PLIST);
    }
    if (gbl.internal <= 1 || INTERNALG(lab))
      SCOPEP(sptr, stb.curr_scope);
    if (gbl.internal > 1) {
      if (INTERNALG(lab))
        INTERNALP(sptr, 1);
      else
        INTERNREFP(sptr, 1);
    }
    if (sem.parallel || sem.orph || sem.target || sem.teams) {
      set_parref_flag(sptr, sptr, BLK_UPLEVEL_SPTR(sem.scope_level));
      PARREFP(sptr, 1);
    }
  } else {
    sptr = declsym_newscope(sptr, stype, 0);
    if (gbl.internal > 1) {
      INTERNALP(sptr, 1);
    }
    ASSNP(sptr, 1);
    HCCSYMP(sptr, 1);
  }
  return sptr;
}

/** \brief Create the array (ST_PLIST) representing the namelist.
 *
 *  The name is derived from the namelist group name.
 */
int
get_nml_array(int nml)
{
  int sptr;

  if (ADDRESSG(nml))
    return ADDRESSG(nml);
  sptr = get_next_sym(SYMNAME(nml), "nml");
  STYPEP(sptr, ST_PLIST);
  CCSYMP(sptr, 1);
  SCP(sptr, SC_STATIC);
  DTYPEP(sptr, DT_PTR);
  ADDRESSP(nml, sptr);
  /* plist will be init'd only if the namelist variable
   * is referenced.
   */
  return sptr;
}

static int
ast_ioret(void)
{
  static int ast;
  int sptr;

  if (io_sc != SC_PRIVATE) {
    sptr = getsymbol("z__io");
    sptr = declsym_newscope(sptr, ST_VAR, DT_INT);
    /*
     * Some test suitable for deciding the variable is already created.
    if (!SCG(sptr)) {
     */
    SCP(sptr, io_sc);
    ast = mk_id(sptr); /* note that ast is static */
    ASSNP(sptr, 1);
    HCCSYMP(sptr, 1);
/*
}
 */
  } else {
    sptr = getsymbol("z__io_p");
    if (STYPEG(sptr) == ST_UNKNOWN) {
      STYPEP(sptr, ST_VAR);
      DTYPEP(sptr, DT_INT);
      DCLDP(sptr, 1);
      SCP(sptr, io_sc);
      SCOPEP(sptr, stb.curr_scope);
      ASSNP(sptr, 1);
      HCCSYMP(sptr, 1);
    }
    ast = mk_id(sptr);
  }

  return ast;
}

/* ast_type - A_FUNC or A_CALL */
/* count    - number of arguments */
/* func     - sptr of function to invoke */
static int
begin_io_call(int ast_type, int func, int count)
{
  io_call.ast = begin_call(ast_type, func, count);
  io_call.ast_type = ast_type;
  return io_call.ast;
}

/* arg - ast of argument to add */
static void
add_io_arg(int arg)
{
  add_arg(arg);
}

/*
 * complete the call to the io support routine. if the function returns
 * a value (ast_type is A_FUNC), first generate an assignment of the
 * function to a temporary; then, add this statement to the STD list.
 * If the function invocation is a call, then just add the call to the
 * STD list.
 */
static int
end_io_call(void)
{
  int asn;
  int i;

  if (io_call.ast_type == A_FUNC)
    asn = mk_assn_stmt(ast_ioret(), io_call.ast, DT_INT);
  else
    asn = io_call.ast;
  io_call.std = add_stmt_after(asn, (int)STD_PREV(0));

  for (i = 0; i <= PT_MAXV; i++) {
    if (i == PT_SIZE) {
      continue;
    }
    if (PTS(i) && PTTMPUSED(i)) {
      int aaa;
      if (PTVARREF(i) != 1) {
        aaa =
            mk_assn_stmt(PTVARREF(i), mk_convert(PTV(i), A_DTYPEG(PTVARREF(i))),
                         A_DTYPEG(PTVARREF(i)));
        add_stmt_after(aaa, io_call.std);
      }
      if (i != PT_SIZE && i != PT_IOSTAT)
        PT_TMPUSED(i, 0);
    }
  }
  return asn;
}

static int
mk_iofunc(FtnRtlEnum rtlRtn, int dtype, int intent)
{
  int sptr;
  sptr = sym_mkfunc(mkRteRtnNm(rtlRtn), dtype);
  NODESCP(sptr, 1);
#ifdef SDSCSAFEG
  SDSCSAFEP(sptr, 1);
#endif
  INDEPP(sptr, 1);
  TYPDP(sptr, 1);     /* force external statement */
  INTERNALP(sptr, 0); /* these are not internal functions */
  if (intent)
    INTENTP(sptr, intent);
  return sptr;
}

static int
mk_hpfiofunc(FtnRtlEnum rtlRtn, int dtype, int intent)
{
  int sptr;

  sptr = sym_mkfunc(mkRteRtnNm(rtlRtn), dtype);
  PUREP(sptr, 0);
  INDEPP(sptr, 1);
#ifdef SDSCSAFEG
  SDSCSAFEP(sptr, 1);
#endif
  TYPDP(sptr, 1); /* force external statement */
  if (intent)
    INTENTP(sptr, intent);
  return sptr;
}

static LOGICAL
need_descriptor_ast(int refast)
{
  /*
   * Determine if an array reference, given its ast, requires a descriptor.
   */
  if (!XBIT(58, 0x800000) && XBIT(58, 0x40) && A_TYPEG(refast) == A_ID) {
    /* only performed for f90 whole array references */
    int ss = A_SPTRG(refast);
    return ss > NOSYM && (ASSUMSHPG(ss) || POINTERG(ss) || ASUMSZG(ss));
  }
  return TRUE;
}

static void
rw_array(int refast, int stride, int eldtype, FtnRtlEnum rtlRtn)
{
  /* collapse a read/write of an array into a single call
   * which doesn't require a descriptor
   */
  int syma;
  int ast1, ast2;

  ast1 = mk_cval((INT)dtype_to_arg(eldtype), DT_INT);
  ast2 = size_of_ast(refast);
  syma = mk_iofunc(rtlRtn, DT_INT, is_read ? 0 : INTENT_IN);
  (void)begin_io_call(A_FUNC, syma, 4);
  (void)add_io_arg(ast1); /* type, length, stride */
  (void)add_io_arg(ast2);
  (void)add_io_arg(stride);
  (void)add_io_arg(refast); /* item */
}

/* ------------------------------------------------------------------------ */
static IOL *
find_end_iollist(IOL *in_iolptr)
{
  IOL *iolptr;

  iolptr = in_iolptr;

  if (iolptr == NULL)
    return iolptr;

  while (iolptr->next != NULL)
    iolptr = iolptr->next;

  return iolptr;
}

static IOL *
link_iollist(IOL *first, IOL *last)
{
  IOL *end;

  if (first == NULL)
    return last;
  if (last == NULL)
    return first;

  end = find_end_iollist(first);
  end->next = last;
  return first;
}

static IOL *
get_dobegin(int ast1, int ast2, int ast3)
{
  int sptr, count;
  DOINFO *doinfo;
  IOL *dobegin;

  /* create a DOINFO record using ast1,ast2,ast3.; the index
   * variable is a compiler-created temporary .
   */
  doinfo = get_doinfo(0);
  sptr = get_temp(astb.bnd.dtype);
  doinfo->index_var = sptr;
  doinfo->init_expr = ast1;
  doinfo->limit_expr = ast2;
  doinfo->step_expr = ast3 ? ast3 : astb.bnd.one;
  count = doinfo->limit_expr;
  if (doinfo->init_expr != doinfo->step_expr) {
    if (doinfo->init_expr != astb.bnd.zero) {
      count = mk_binop(OP_SUB, count, doinfo->init_expr, astb.bnd.dtype);
    }
    count = mk_binop(OP_ADD, count, doinfo->step_expr, astb.bnd.dtype);
  }
  if (doinfo->step_expr != astb.bnd.one) {
    count = mk_binop(OP_DIV, count, doinfo->step_expr, astb.bnd.dtype);
  }
  doinfo->count = count;
  /*
   * Create the DOBEGIN IOL for this index.
   */
  dobegin = (IOL *)getitem(0, sizeof(IOL));
  dobegin->id = IE_DOBEGIN;
  dobegin->next = NULL;
  dobegin->doinfo = doinfo;
  dobegin->l_std = 0;

  return dobegin;
}

static IOL *
get_doend(DOINFO *doinfo)
{
  IOL *doend;
  /*
   * Create the DOEND IOL for this index;
   */
  doend = (IOL *)getitem(0, sizeof(IOL));
  doend->id = IE_DOEND;
  doend->next = NULL;
  doend->doinfo = doinfo;
  doend->l_std = 0;
  return doend;
}

/* we have a one-dimensional expression, replace a vector subscript
 * by the indexast */
static int
replace_vector_subscript(int ast, int indexast)
{
  int oldl, newl, oldr, newr, nargs, argt, i, changes, argtnew, sptr;
  int asd, nsubs, subs[7], lb;
  int dtype;

  if (ast == 0)
    return 0;
  dtype = A_DTYPEG(ast);
  if (DTY(dtype) == TY_ARRAY)
    dtype = DTY(dtype + 1);
  switch (A_TYPEG(ast)) {
  /* expressions */
  case A_CNST:
  case A_CMPLXC:
    break;
  case A_BINOP:
    oldl = A_LOPG(ast);
    oldr = A_ROPG(ast);
    newl = replace_vector_subscript(oldl, indexast);
    newr = replace_vector_subscript(oldr, indexast);
    if (newl != oldl || newr != oldr) {
      ast = mk_binop(A_OPTYPEG(ast), newl, newr, dtype);
    }
    break;
  case A_UNOP:
    oldl = A_LOPG(ast);
    newl = replace_vector_subscript(oldl, indexast);
    if (newl != oldl) {
      ast = mk_unop(A_OPTYPEG(ast), newl, dtype);
    }
    break;
  case A_CONV:
    oldl = A_LOPG(ast);
    newl = replace_vector_subscript(oldl, indexast);
    if (newl != oldl) {
      ast = mk_convert(newl, dtype);
    }
    break;
  case A_PAREN:
    oldl = A_LOPG(ast);
    newl = replace_vector_subscript(oldl, indexast);
    if (newl != oldl) {
      ast = mk_paren(newl, dtype);
    }
    break;
  case A_SUBSTR:
    oldl = A_LOPG(ast);
    newl = replace_vector_subscript(oldl, indexast);
    if (newl != oldl) {
      ast = mk_substr(newl, A_LEFTG(ast), A_RIGHTG(ast), dtype);
    }
    break;
  case A_INTR:
  case A_FUNC:
    nargs = A_ARGCNTG(ast);
    argt = A_ARGSG(ast);
    argtnew = mk_argt(nargs);
    changes = FALSE;
    for (i = 0; i < nargs; ++i) {
      ARGT_ARG(argtnew, i) =
          replace_vector_subscript(ARGT_ARG(argt, i), indexast);
      if (ARGT_ARG(argtnew, i) != ARGT_ARG(argt, i))
        changes = TRUE;
    }
    if (!changes) {
      unmk_argt(nargs);
    } else {
      ast = mk_func_node(A_TYPEG(ast), A_LOPG(ast), nargs, argtnew);
      A_DTYPEP(ast, dtype);
    }
    break;
  case A_MEM:
  case A_ID:
    if (A_TYPEG(ast) == A_MEM) {
      sptr = A_SPTRG(A_MEMG(ast));
    } else {
      sptr = A_SPTRG(ast);
    }
    dtype = DTYPEG(sptr);
    if (DTY(dtype) == TY_ARRAY) {
      /* add a subscript here */
      if (ADD_NUMDIM(dtype) != 1) {
        interr("replace_vector_subscript: >1 dimensional array ",
               ADD_NUMDIM(dtype), 3);
      }
      ast = mk_subscr(ast, &indexast, 1, DTY(dtype + 1));
    } else {
      oldl = A_LOPG(ast);
      newl = replace_vector_subscript(oldl, indexast);
      if (oldl != newl) {
        ast = mk_member(oldl, A_MEMG(ast), dtype);
      }
    }
    break;
  case A_SUBSCR:
    /* go through the subscripts looking for a vector one */
    asd = A_ASDG(ast);
    nsubs = ASD_NDIM(asd);
    changes = FALSE;
    for (i = 0; i < nsubs; ++i) {
      int oldss, newss;
      oldss = ASD_SUBS(asd, i);
      newss = replace_vector_subscript(oldss, indexast);
      subs[i] = newss;
      if (newss != oldss)
        changes = TRUE;
    }
    if (changes) {
      ast = mk_subscr(A_LOPG(ast), subs, nsubs, dtype);
    }
    break;

  case A_TRIPLE:
    /* change l:u:t into l+indexast */
    lb = A_LBDG(ast);
    if (lb == astb.i0) {
      ast = indexast;
    } else {
      ast = mk_binop(OP_ADD, lb, indexast, DT_INT);
    }
    break;
  default:
    interr("replace_vector_subscript: unexpected subscript operation",
           A_TYPEG(ast), 3);
    break;
  }
  return ast;
} /* replace_vector_subscript */

/* this function expands f90 style derived type references into member
 * references, so the runtime doesn`t have to know how to print derived
 * types. Array of derived types will get dobegin/doend bracketing
 * the reference.  This now handled both componentized and noncomponentized
 * derived types.
 */

static IOL *
add_iolptrs(int dtype, SST *in_stkptr, int *mndscp)
{
  IOL *iolptr;
  IOL *startlist;
  IOL *endlist;
  int ast = 0;
  int sptr1, sptrm;
  int j, numdim;
  int derived_dtype, dtypem;
  int mem_id;
  int subs[7];
  SST *stkptr;
  SST tmpstk;
  IOL *dobegin;
  IOL *doend;
  IOL *tmp;
  int shp, leafast, rootast;
  int priv_err;
#if defined(DEBUG)
  int print_once = 0;
#endif

  if (DTYG(dtype) != TY_DERIVED) {
    /* recursive member that is not itself a derived type */
    stkptr = (SST *)getitem(0, sizeof(SST));
    SST_SHAPEP(stkptr, 0);
    SST_CVLENP(stkptr, 0);
    iolptr = (IOL *)getitem(0, sizeof(IOL));
    iolptr->next = NULL;
    *stkptr = *in_stkptr;
    iolptr->id = IE_EXPR;
    iolptr->element = stkptr;
    iolptr->l_std = 0;
    return iolptr;
  }

  /* at a derived type; go through the members */
  startlist = NULL;
  endlist = NULL;
  switch (SST_IDG(in_stkptr)) {
  case S_IDENT:
  case S_DERIVED:
    sptr1 = SST_SYMG(in_stkptr);
    ast = mk_id(sptr1);
    break;
  case S_LVALUE:
    ast = SST_ASTG(in_stkptr);
    break;
  case S_EXPR:
    ast = SST_ASTG(in_stkptr);
    break;
  default:
    interr("add_iolptrs: unexpected semant stack type", SST_IDG(in_stkptr), 3);
  }

  derived_dtype = DDTG(dtype);

  dobegin = NULL;
  doend = NULL;
  /* TPR 2661:
   * if we have a reference x(i)%mem1(1:n)%mem2
   * the datatype is array(1:n) of dtypemem2
   * however, mem2 is a scalar, 'ast' is not a subscript.
   * So, if ast has the shape of its parent,
   * save the subtree from tail to the parent, reapply it later */
  leafast = ast;
  rootast = ast;
  while (1) {
    if (A_TYPEG(ast) == A_MEM) {
      /* if parent is different shape, stop here */
      if (A_SHAPEG(ast) && A_SHAPEG(ast) == A_SHAPEG(A_PARENTG(ast))) {
        ast = A_PARENTG(ast);
        rootast = ast;
        continue;
      }
    } else if (A_TYPEG(ast) == A_SUBSCR) {
      int lop;
      lop = A_LOPG(ast);
      /* compare subscript with parent of A_MEM, if any */
      if (A_TYPEG(lop) == A_MEM && A_SHAPEG(ast) &&
          A_SHAPEG(ast) == A_SHAPEG(A_PARENTG(lop))) {
        ast = A_PARENTG(lop);
        rootast = ast;
        continue;
      }
    }
    break;
  }
  if (DTY(dtype) == TY_ARRAY && ast && A_TYPEG(ast) != A_SUBSCR) {
    /* array of derived type.  Rest of compiler doesn't handle
     * these 'inner' subscripts yet.
     * make a subscripted reference, the dobegins will be
     * thrown around at the end of this function */
    numdim = ADD_NUMDIM(dtype);
    for (j = 0; j < numdim; ++j) {
      tmp = get_dobegin(ADD_LWAST(dtype, j), ADD_UPAST(dtype, j), 0);
      dobegin = link_iollist(dobegin, tmp);
      subs[j] = mk_id(tmp->doinfo->index_var);
      tmp = get_doend(tmp->doinfo);
      doend = link_iollist(tmp, doend);
    }
    ast = mk_subscr(ast, subs, numdim, DDTG(A_DTYPEG(ast)));
  } else if (ast && A_TYPEG(ast) == A_SUBSCR && (shp = A_SHAPEG(ast))) {
    /* we want to create/rewrite subscript and use dobegin/doend.
     * We may have to use doend/dobegin for internal triples, too.
     */
    int asd, numdim, i;
    asd = A_ASDG(ast);
    numdim = ASD_NDIM(asd); /* get number of subscripts */
    for (i = 0; i < numdim; ++i) {
      int shpss, astss, sptrivar, astivar;
      int init, final, stride, normalize;
      normalize = 0;
      astss = ASD_SUBS(asd, i);
      shpss = A_SHAPEG(astss);
      if (A_TYPEG(astss) == A_TRIPLE) {
        init = A_LBDG(astss);
        final = A_UPBDG(astss);
        stride = A_STRIDEG(astss);
        if (stride == 0)
          stride = astb.i1;
      } else if (shpss) { /* better be one-dimensional */
        assert(SHD_NDIM(shpss) == 1,
               "add_iolptrs: subscript shape >1 dimension", SHD_NDIM(shpss), 3);
        init = SHD_LWB(shpss, 0);
        final = SHD_UPB(shpss, 0);
        stride = SHD_STRIDE(shpss, 0);
        if (A_TYPEG(astss) == A_SUBSCR) {
          int asdss;
          asdss = A_ASDG(astss);
          if (ASD_NDIM(asdss) != 1) {
            normalize = 1;
          }
        } else if (A_TYPEG(astss) != A_ID) {
          normalize = 1;
        }
      } else {
        init = 0;
        final = 0;
        stride = 0;
      }
      if (!init) {
        /* use the original subscript */
        subs[i] = astss;
      } else {
        /* create a DOINFO record for this dimension; the index
         * variable is a compiler-created temporary.
         */
        tmp = get_dobegin(init, final, stride);
        dobegin = link_iollist(dobegin, tmp);
        sptrivar = tmp->doinfo->index_var;
        if (normalize) {
          /* change limits from DO i=i,f,s to DO i=0,(f-i)/s,1 */
          int limit;
          limit = final;
          if (init != astb.i0) {
            limit = mk_binop(OP_SUB, limit, init, DT_INT);
          }
          if (stride != astb.i1) {
            limit = mk_binop(OP_DIV, limit, stride, DT_INT);
          }
          tmp->doinfo->init_expr = astb.i0;
          tmp->doinfo->limit_expr = limit;
          tmp->doinfo->step_expr = astb.i1;
        }
        tmp = get_doend(tmp->doinfo);
        doend = link_iollist(tmp, doend);
        /*
         * prepare the subscript for this dimension; it's just
         * a subscripted reference of the index vector and the
         * subscript is the DO index variable.
         */
        astivar = mk_id(sptrivar);
        if (normalize) {
          /* do something special */
          subs[i] = replace_vector_subscript(astss, astivar);
        } else if (A_TYPEG(astss) == A_TRIPLE) {
          subs[i] = astivar;
        } else if (A_TYPEG(astss) == A_ID) {
          /* subscript the array reference */
          subs[i] = mk_subscr(astss, &astivar, 1, DT_INT);
        } else if (A_TYPEG(astss) == A_SUBSCR) {
          subs[i] = mk_subscr(A_LOPG(astss), &astivar, 1, DT_INT);
        } else {
          /* no cases left */
          interr("add_iolptrs, unexpected subscript", A_TYPEG(astss), 3);
        }
      }
    }
    ast = mk_subscr(A_LOPG(ast), subs, numdim, DDTG(A_DTYPEG(ast)));
  }
  if (leafast != rootast) {
    ast = replace_ast_subtree(leafast, rootast, ast);
  }

  priv_err = 0;
  for (sptrm = DTY(derived_dtype + 1); sptrm != NOSYM; sptrm = SYMLKG(sptrm)) {
    int i;
    if (is_tbp_or_final(sptrm)) {
#if DEBUG
      if (!print_once) {
        if (!sem.defined_io_seen) {
          if (BINDG(sptrm)) {
            error(155, 2, gbl.lineno,
                  "Dubious use of derived type with "
                  "type bound procedure in I/O statement",
                  NULL);
          } else {
            error(155, 2, gbl.lineno,
                  "Dubious use of derived type with "
                  "final subroutine in I/O statement",
                  NULL);
          }
        }
        print_once = 1;
      }
#endif
      continue; /* skip tbp member */
    }
    if (KINDG(sptrm) || LENPARMG(sptrm)) /* skip type parameters */
      continue;
    dtypem = DTYPEG(sptrm);
    if (PRIVATEG(sptrm) && test_private_dtype(ENCLDTYPEG(sptrm)) &&
        !sem.defined_io_seen) {
      priv_err = 1;
    }
    i = NMPTRG(sptrm);
    if (POINTERG(sptrm) && !sem.defined_io_seen) {
      error(453, 3, gbl.lineno, SYMNAME(sym_of_ast(ast)), "");
      break;
    }
    mem_id = mkmember(derived_dtype, ast, i);

    SST_IDP(&tmpstk, S_EXPR);
    SST_ASTP(&tmpstk, mem_id);
    SST_DTYPEP(&tmpstk, dtypem);
    SST_SHAPEP(&tmpstk, 0);
    SST_CVLENP(&tmpstk, 0);
    iolptr = add_iolptrs(dtypem, &tmpstk, mndscp);
    if (startlist) {
      endlist->next = iolptr;
    } else {
      /* first element(s) */
      startlist = iolptr;
    }
    endlist = find_end_iollist(iolptr);

    /* FS#13258  If member is allocatable, skip pointer,offset,descriptor */
    if (ALLOCG(sptrm)) {
      int member = sptrm;
      while (SYMLKG(sptrm) == MIDNUMG(member) ||
             SYMLKG(sptrm) == PTROFFG(member) || SYMLKG(sptrm) == SDSCG(member))
        sptrm = SYMLKG(sptrm);
    }
  }
  if (dobegin) {
    startlist = link_iollist(dobegin, startlist);
    startlist = link_iollist(startlist, doend);
  }
  if (priv_err) {
    error(155, 3, gbl.lineno,
          "A derived type containing private components cannot be an I/O item",
          CNULL);
  }
  return startlist;
}
/* ------------------------------------------------------------------------ */
static void
get_derived_iolptrs(SST *result, int sptr1, SST *instkptr)
{
  IOL *iolptr;
  IOL *start;

  start = add_iolptrs(SST_DTYPEG(instkptr), instkptr, NULL);
  iolptr = find_end_iollist(start);
  SST_BEGP(result, (ITEM *)start);
  SST_ENDP(result, (ITEM *)iolptr);
}

static void
gen_derived_io(int sptr, FtnRtlEnum rtlRtn, int read)
{
  int bytfunc;
  int i;

  if (!sptr) {
    /*
     * In general, should not get here if sptr is 0.  But, there is
     * one case where this will happen -- the output item is
     * iso_c_binding:c_loc() and we would have already reported
     * the error,"A derived type containing private components cannot
     * be an I/O item" (search for c_loc earlier in this file).
     */
    return;
  }
  interr("gen_derived_io, f90 output not implemented", sptr, 2);
  bytfunc = mk_iofunc(rtlRtn, DT_INT, read ? 0 : INTENT_IN);
  i = mk_cval(size_of(DTYPEG(sptr)), DT_INT);
  (void)add_io_arg(astb.i1); /* length, stride */
  (void)add_io_arg(astb.i0);
  (void)add_io_arg(mk_id(sptr)); /* item */
  (void)add_io_arg(i);           /* explicit item_length */
}

/*
 * Compute last value of index variable:
 *    index_var = max( (m2 - m1 + m3)/m3, 0)
 */
static void
gen_lastval(DOINFO *doinfo)
{
  int ast1, ast;
  int dtype;

  ast1 = doinfo->count;
  dtype = DTYPEG(doinfo->index_var);
  ast = astb.i0;
  switch (dtype) {
  case DT_INT8:
  case DT_LOG8:
    dtype = DT_INT8;
    break;
  case DT_BINT:
  case DT_BLOG:
    dtype = DT_BINT;
    break;
  case DT_SINT:
  case DT_SLOG:
    dtype = DT_SINT;
    break;
  default:
    dtype = DT_INT4;
    break;
  }
  if (dtype != DT_INT4)
    ast = mk_convert(ast, dtype);
  ast1 = ast_intr(I_MAX, dtype, 2, ast1, ast);
  if (doinfo->step_expr)
    ast1 = mk_binop(OP_MUL, ast1, doinfo->step_expr, dtype);
  ast1 = mk_binop(OP_ADD, ast1, doinfo->init_expr, dtype);
  ast = mk_assn_stmt(mk_id(doinfo->index_var), ast1, dtype);
  (void)add_stmt(ast);
}

/*
 * perform more checks & initialization for all io statements
 * except bufferin/bufferout
 */
static int
misc_io_checks(const char *iostmt)
{
  (void)not_in_forall(iostmt);
  /*    if (PTV(PT_IOMSG)) {
          int   sptr;
          int   ast;
          sptr = mk_iofunc(RTE_f90io_iomsg, DT_NONE, 0);
          (void)begin_io_call(A_CALL, sptr, 1);
          (void)add_io_arg(PTV(PT_IOMSG));
          ast = end_io_call();
          STD_LINENO(io_call.std) = gbl.lineno;
          nondevice_io = TRUE;
      }*/
  return 0;
}

static void
iomsg_check()
{
  if (PTV(PT_IOMSG)) {
    int sptr;
    sptr = mk_iofunc(RTE_f90io_iomsga, DT_NONE, 0);
    (void)begin_io_call(A_CALL, sptr, 1);
    (void)add_io_arg(PTV(PT_IOMSG));
    (void)add_stmt_after(io_call.ast, (int)STD_PREV(0));
  }
}

static void
newunit_check()
{
  if (PTV(PT_NEWUNIT) && !(PTV(PT_FILE) || PTV(PT_STATUS)))
    IOERR2(201, "NEWUNIT requires FILE or STATUS=SCRATCH specifier");
}

static int
gen_dtsfmt_args(int *iotype_ast, int *vlist_ast)
{
  int sptr;
  int tempbase, templen;
  int tast, baseast, asn;
  int dtype;
  ADSC *ad;
  tast = 0;
  asn = 1;

  /* create deferchar for iotype , the callee will allocate it */
  sptr = getcctmp_sc('d', sem.dtemps++, ST_VAR, DT_DEFERCHAR, SC_BASED);

  ADJLENP(sptr, 1);
  ADDRTKNP(sptr, 1);
  SEQP(sptr, 1);
  ARGP(sptr, 1);
  tempbase = get_next_sym(SYMNAME(sptr), "cp");
  templen = get_next_sym(SYMNAME(sptr), "cl");

  /* make the pointer points to sptr */
  STYPEP(tempbase, ST_VAR);
  DTYPEP(tempbase, DT_PTR);
  SCP(tempbase, io_sc);
  NODESCP(tempbase, 1);
  SEQP(sptr, 1);

  /* set length variable */
  STYPEP(templen, ST_VAR);
  DTYPEP(templen, DT_INT);
  SCP(templen, io_sc);

  MIDNUMP(sptr, tempbase);
  CVLENP(sptr, templen);
  tast = add_nullify_ast(mk_id(tempbase));
  add_stmt(tast);

  *iotype_ast = mk_id(sptr);
  baseast = mk_id(tempbase);

  if (XBIT(124, 0x10))
    dtype = get_array_dtype(1, DT_INT8);
  else
    dtype = get_array_dtype(1, DT_INT);
  sptr = getcctmp_sc('d', sem.dtemps++, ST_ARRAY, dtype, SC_BASED);
  ALLOCP(sptr, 1);
  ADDRTKNP(sptr, 1);
  SEQP(sptr, 1);
  ARGP(sptr, 1);
  ad = AD_DPTR(dtype);
  ADD_DEFER(dtype) = 1;
  get_static_descriptor(sptr);
  get_all_descriptors(sptr);
  tempbase = MIDNUMG(sptr);

  tast = add_nullify_ast(mk_id(tempbase));
  add_stmt(tast);
  *vlist_ast = mk_id(sptr);

  return asn;
}

static int
call_dtsfmt(int iotype_ast, int vlist_ast)
{
  int sptr, flagast;
  int tast, baseast, listast, lenast, arraydsc, asn;
  INT val[2];

  val[0] = 0;
  if (XBIT(124, 0x10) && XBIT(68, 0x1)) {
    /* vlist is i8 and its descriptor is i8 */
    val[1] = 3;
  } else if (XBIT(68, 0x1)) {
    /* vlist is i4 and its descriptor is i8 */
    val[1] = 2;
  } else if (XBIT(124, 0x10)) {
    /* vlist is i8 and its descriptor is i4 */
    val[1] = 1;
  } else {
    /* vlist and its descriptor are i4 */
    val[1] = 0;
  }

  lenast = mk_id(CVLENG(A_SPTRG(iotype_ast)));
  arraydsc = mk_id(SDSCG(A_SPTRG(vlist_ast)));
  listast = mk_id(MIDNUMG(A_SPTRG(vlist_ast)));
  baseast = mk_id(MIDNUMG(A_SPTRG(iotype_ast)));
  flagast = mk_id(getcon(val, DT_INT4));

  if (is_read)
    sptr = mk_iofunc(RTE_f90io_dts_fmtr, DT_INT, 0);
  else
    sptr = mk_iofunc(RTE_f90io_dts_fmtw, DT_INT, 0);

  tast = 0;
  tast = begin_call(A_FUNC, sptr, 5);
  (void)add_arg(baseast);
  (void)add_arg(listast);
  (void)add_arg(lenast);
  (void)add_arg(arraydsc);
  (void)add_arg(flagast);
  asn = mk_assn_stmt(ast_ioret(), tast, DT_INT);
  (void)add_stmt(asn);
  return asn;
}

static ITEM *
gen_dtio_args(SST *stkptr, int arg1, int iotype_ast, int vlist_ast)
{
  ITEM *p, *arglist;
  INT v[2];
  int ast_type, iostat_ast, iomsg_ast, unit_ast;
  int sptr, tast;
  int argdtyp = DT_INT;
  if (XBIT(124, 0x10))
    argdtyp = DT_INT8;

  /* dtv */
  p = (ITEM *)getitem(0, sizeof(ITEM));
  p->t.stkp = stkptr;
  p->next = NULL;
  p->ast = arg1;
  arglist = p;

  /* unit */
  if (intern == TRUE) {
    v[0] = 0;
    v[1] = -1;
    unit_ast = mk_cnst(getcon(v, argdtyp));

  } else {
    unit_ast = PTARG(PT_UNIT);
    if (A_DTYPEG(unit_ast) != argdtyp) {
      unit_ast = mk_convert(unit_ast, argdtyp);
    }
  }
  p->next = (ITEM *)getitem(0, sizeof(ITEM));
  p = p->next;
  p->t.stkp = (SST *)getitem(0, sizeof(SST));
  SST_ASTP(p->t.stkp, unit_ast);
  SST_DTYPEP(p->t.stkp, A_DTYPEG(unit_ast));
  ast_type = A_TYPEG(unit_ast);
  SST_SHAPEP(p->t.stkp, 0);
  if (ast_type == A_CNST) {
    SST_IDP(p->t.stkp, S_CONST);
    SST_SYMP(p->t.stkp, A_SPTRG(unit_ast));
    SST_LSYMP(p->t.stkp, 0);
    SST_CVALP(p->t.stkp, CONVAL2G(A_SPTRG(unit_ast)));
  } else if (ast_type == A_ID) {
    SST_IDP(p->t.stkp, S_IDENT);
    SST_SYMP(p->t.stkp, A_SPTRG(unit_ast));
    mkident(p->t.stkp);
  } else {
    SST_IDP(p->t.stkp, S_EXPR); /* need to check this */
    mkexpr(p->t.stkp);
  }
  p->ast = unit_ast;

  if (fmttyp != FT_UNFORMATTED) {
    /* iotype */
    p->next = (ITEM *)getitem(0, sizeof(ITEM));
    p = p->next;
    p->t.stkp = (SST *)getitem(0, sizeof(SST));
    p->ast = iotype_ast;
    SST_ASTP(p->t.stkp, iotype_ast);
    SST_DTYPEP(p->t.stkp, A_DTYPEG(iotype_ast));
    SST_SYMP(p->t.stkp, A_SPTRG(iotype_ast));
    SST_PARENP(p->t.stkp, 0);
    SST_SHAPEP(p->t.stkp, 0);
    SST_IDP(p->t.stkp, S_CONST);

    /* v_list */
    if (vlist_ast == astb.i0) {
      /* make array of size 0 */
      /* set it as array size 0 first */
      int dtype = get_array_dtype(1, argdtyp);
      ADSC *ad = AD_DPTR(dtype);
      AD_LWAST(ad, 0) = astb.i1;
      AD_LWBD(ad, 0) = astb.i1;
      AD_UPAST(ad, 0) = astb.i0;
      AD_UPBD(ad, 0) = astb.i0;
      AD_MLPYR(ad, 0) = astb.i1;

      sptr = getcctmp_sc('d', sem.dtemps++, ST_ARRAY, dtype, io_sc);
      vlist_ast = mk_id(sptr);
      DESCUSEDP(sptr, 1);
      ARGP(sptr, 1);
      if (MIDNUMG(sptr)) {
        tast = add_nullify_ast(mk_id(MIDNUMG(sptr)));
        add_stmt(tast);
      }
    }
    p->next = (ITEM *)getitem(0, sizeof(ITEM));
    p = p->next;
    p->t.stkp = (SST *)getitem(0, sizeof(SST));
    p->next = NULL;
    p->ast = vlist_ast;
    SST_ASTP(p->t.stkp, vlist_ast);
    SST_DTYPEP(p->t.stkp, A_DTYPEG(vlist_ast));
    SST_SYMP(p->t.stkp, A_SPTRG(vlist_ast));
    SST_PARENP(p->t.stkp, 0);
    SST_SHAPEP(p->t.stkp, 0);
    SST_IDP(p->t.stkp, S_IDENT);
  }

  /* iostat */
  iostat_ast = PTARG(PT_IOSTAT);
  p->next = (ITEM *)getitem(0, sizeof(ITEM));
  p = p->next;
  p->t.stkp = (SST *)getitem(0, sizeof(SST));
  SST_IDP(p->t.stkp, S_IDENT);
  if (iostat_ast == astb.i0) {
    sptr = getcctmp_sc('d', sem.dtemps++, ST_VAR, argdtyp, io_sc);
    iostat_ast = mk_id(sptr);
    if (argdtyp != A_DTYPEG(astb.i0))
      tast = mk_assn_stmt(iostat_ast, mk_convert(astb.i0, argdtyp), argdtyp);
    else
      tast = mk_assn_stmt(iostat_ast, astb.i0, argdtyp);
    (void)add_stmt(tast);
  }
  p->ast = iostat_ast;
  SST_ASTP(p->t.stkp, iostat_ast);
  SST_DTYPEP(p->t.stkp, A_DTYPEG(iostat_ast));
  SST_SYMP(p->t.stkp, A_SPTRG(iostat_ast));
  SST_IDP(p->t.stkp, S_IDENT);
  SST_PARENP(p->t.stkp, 0);
  SST_SHAPEP(p->t.stkp, 0);

  /* iomsg */
  iomsg_ast = PTARG(PT_IOMSG);
  if (iomsg_ast == 0) {
    sptr = getcctmp_sc('d', sem.dtemps++, ST_VAR, DT_CHAR, io_sc);
    iomsg_ast = mk_id(sptr);
  }
  p->next = (ITEM *)getitem(0, sizeof(ITEM));
  p = p->next;
  p->t.stkp = (SST *)getitem(0, sizeof(SST));
  p->next = ITEM_END;
  p->ast = iomsg_ast;
  SST_ASTP(p->t.stkp, iomsg_ast);
  SST_DTYPEP(p->t.stkp, A_DTYPEG(iomsg_ast));
  SST_SYMP(p->t.stkp, A_SPTRG(iomsg_ast));
  SST_IDP(p->t.stkp, S_IDENT);
  SST_PARENP(p->t.stkp, 0);
  SST_SHAPEP(p->t.stkp, 0);

  return arglist;
}

/* Returns an ast representing a call intended for a defined I/O subroutine.
 * fpstr = the function sptr returned from resolved_defined_io()
 * argcnt = number of arguments for the defined I/O subroutine
 * arglist = list of arguments. At the very least, argument 1, the derived
 * type variable, must be set when fsptr is a type bound procedure.
 */
static int
get_defined_io_call(int fsptr, int argcnt, ITEM *arglist)
{
  int ast, mem, i, dtv, dtv_sptr;
  ITEM *p;

  if (CLASSG(fsptr) && VTABLEG(fsptr) && VTOFFG(fsptr)) {

    /* generate tbp call */

    dtv = arglist->ast;
    dtv_sptr = memsym_of_ast(dtv);
    ast = begin_call(A_CALL, fsptr, argcnt);
    if (!BINDG(fsptr) || !CCSYMG(fsptr)) {
      get_implementation(DTYPEG(dtv_sptr), fsptr, 0, &mem);
      fsptr = mem;
    }
    mem = mk_member(dtv, mk_id(fsptr), A_DTYPEG(dtv));
    A_LOPP(ast, mem);

    /* add args in arglist */
    for (p = arglist, i = 0; i < argcnt; p = p->next) {
      add_arg(p->ast);
      ++i;
    }
  } else {

    /* generate normal call */

    ast = begin_call(A_CALL, fsptr, argcnt);
    for (p = arglist; p != ITEM_END; p = p->next) {
      add_arg(p->ast);
    }
  }
  return ast;
}

static FtnRtlEnum
getBasicScalarRWRtn(LOGICAL is_read, FormatType fmttyp)
{
  FtnRtlEnum rtlRtn = RTE_no_rtn;

  if (is_read) {
    if (fmttyp == FT_UNFORMATTED)
      rtlRtn = unf_nm[LARGE_ARRAY_IDX][BYTE_SWAPPED_IO_IDX].read;
    else if (fmttyp == FT_LIST_DIRECTED)
      rtlRtn = RTE_f90io_ldra;
    else
      rtlRtn = RTE_f90io_fmt_reada;
  } else {
    if (fmttyp == FT_UNFORMATTED)
      rtlRtn = unf_nm[LARGE_ARRAY_IDX][BYTE_SWAPPED_IO_IDX].write;
    else if (fmttyp == FT_LIST_DIRECTED)
      rtlRtn = RTE_f90io_ldwa;
    else
      rtlRtn = RTE_f90io_fmt_writea;
  }
  return rtlRtn;
}

/* treat aggr. as byte stream */
static FtnRtlEnum
getAggrRWRtn(LOGICAL is_read)
{
  FtnRtlEnum rtlRtn = RTE_no_rtn;

  if (is_read) {
    rtlRtn = unf_nm[LARGE_ARRAY_IDX][BYTE_RW_IDX].read;
  } else {
    rtlRtn = unf_nm[LARGE_ARRAY_IDX][BYTE_RW_IDX].write;
  }

  return rtlRtn;
}

/*
 * For formatted & list-directed writes of certain
 * types of scalars, pass the scalars to the
 * run-time by value instead of by reference.
 * This at least removes one case of having to set
 * the scalar's ADDRTKN flag.
 */
static FtnRtlEnum
getWriteByDtypeRtn(int dtype, FormatType fmttyp)
{
  FtnRtlEnum rtlRtn = RTE_no_rtn;

  switch (dtype) {
  case DT_REAL4:
    rtlRtn = (fmttyp == FT_LIST_DIRECTED) ? RTE_f90io_sc_f_ldw
                                          : RTE_f90io_sc_f_fmt_write;
    break;
  case DT_REAL8:
    rtlRtn = (fmttyp == FT_LIST_DIRECTED) ? RTE_f90io_sc_d_ldw
                                          : RTE_f90io_sc_d_fmt_write;
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case DT_QUAD:
    rtlRtn = (fmttyp == FT_LIST_DIRECTED) ? RTE_f90io_sc_q_ldw
                                          : RTE_f90io_sc_q_fmt_write;
    break;
#endif
  case DT_INT8:
    rtlRtn = (fmttyp == FT_LIST_DIRECTED) ? RTE_f90io_sc_l_ldw
                                          : RTE_f90io_sc_l_fmt_write;
    break;
  case DT_LOG8:
    rtlRtn = (fmttyp == FT_LIST_DIRECTED) ? RTE_f90io_sc_l_ldw
                                          : RTE_f90io_sc_l_fmt_write;
    break;
  case DT_INT4:
    rtlRtn = (fmttyp == FT_LIST_DIRECTED) ? RTE_f90io_sc_i_ldw
                                          : RTE_f90io_sc_i_fmt_write;
    break;
  case DT_BLOG:
  case DT_SLOG:
  case DT_LOG4:
    rtlRtn = (fmttyp == FT_LIST_DIRECTED) ? RTE_f90io_sc_i_ldw
                                          : RTE_f90io_sc_i_fmt_write;
    break;
  case DT_CMPLX8:
    rtlRtn = (fmttyp == FT_LIST_DIRECTED) ? RTE_f90io_sc_cf_ldw
                                          : RTE_f90io_sc_cf_fmt_write;
    break;
  case DT_CMPLX16:
    rtlRtn = (fmttyp == FT_LIST_DIRECTED) ? RTE_f90io_sc_cd_ldw
                                          : RTE_f90io_sc_cd_fmt_write;
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case DT_QCMPLX:
    rtlRtn = (fmttyp == FT_LIST_DIRECTED) ? RTE_f90io_sc_cq_ldw
                                          : RTE_f90io_sc_cq_fmt_write;
    break;
#endif
  default:
    if (DTY(dtype) == TY_CHAR) {
      rtlRtn = (fmttyp == FT_LIST_DIRECTED) ? RTE_f90io_sc_ch_ldw
                                            : RTE_f90io_sc_ch_fmt_write;
    } else {
      rtlRtn = (fmttyp == FT_LIST_DIRECTED) ? RTE_f90io_sc_ldw
                                            : RTE_f90io_sc_fmt_write;
    }
    break;
  }

  return rtlRtn;
}

static FtnRtlEnum
getArrayRWRtn(LOGICAL is_read, FormatType fmtTyp, int dtype, LOGICAL needDescr)
{
  FtnRtlEnum rtlRtn = RTE_no_rtn;

  if (needDescr) {
    if (is_read) {
      if (fmtTyp == FT_UNFORMATTED)
        rtlRtn = array_unf_nm[LARGE_ARRAY_IDX][BYTE_SWAPPED_IO_IDX].read;
      else if (fmtTyp == FT_LIST_DIRECTED)
        rtlRtn = (LARGE_ARRAY) ? RTE_f90io_ldr64_aa : RTE_f90io_ldr_aa;
      else
        rtlRtn =
            (LARGE_ARRAY) ? RTE_f90io_fmt_read64_aa : RTE_f90io_fmt_read_aa;
    } else {
      if (fmtTyp == FT_UNFORMATTED)
        rtlRtn = array_unf_nm[LARGE_ARRAY_IDX][BYTE_SWAPPED_IO_IDX].write;
      else if (fmtTyp == FT_LIST_DIRECTED)
        rtlRtn = (LARGE_ARRAY) ? RTE_f90io_ldw64_aa : RTE_f90io_ldw_aa;
      else
        rtlRtn =
            (LARGE_ARRAY) ? RTE_f90io_fmt_write64_aa : RTE_f90io_fmt_write_aa;
    }
  } else {
    if (is_read) {
      switch (fmtTyp) {
      case FT_UNFORMATTED:
        if (BYTE_SWAPPED_IO) {
          rtlRtn = (LARGE_ARRAY) ? RTE_io_usw_read64 : RTE_io_usw_read;
        } else {
          rtlRtn = (LARGE_ARRAY) ? RTE_io_unf_read64 : RTE_io_unf_read;
        }
        break;
      case FT_LIST_DIRECTED:
        rtlRtn = (LARGE_ARRAY) ? RTE_io_ldr64 : RTE_io_ldr;
        break;
      default:
        rtlRtn = (LARGE_ARRAY) ? RTE_io_fmt_read64 : RTE_io_fmt_read;
        break;
      }
    } else {
      switch (fmtTyp) {
      case FT_UNFORMATTED:
        if (BYTE_SWAPPED_IO) {
          rtlRtn = (LARGE_ARRAY) ? RTE_io_usw_write64 : RTE_io_usw_write;
        } else {
          rtlRtn = (LARGE_ARRAY) ? RTE_io_unf_write64 : RTE_io_unf_write;
        }
        break;
      case FT_LIST_DIRECTED:
        rtlRtn = (LARGE_ARRAY) ? RTE_io_ldw64 : RTE_io_ldw;
        break;
      default:
        rtlRtn = (LARGE_ARRAY) ? RTE_io_fmt_write64 : RTE_io_fmt_write;
        break;
      }
    }
  }

  return rtlRtn;
}
