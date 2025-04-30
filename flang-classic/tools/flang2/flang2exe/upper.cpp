/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
 * \brief upper - import the lowered F90/HPF code
 */

#include "upper.h"
#include "error.h"
#include "ilm.h"
#include "ilmtp.h"
#include "semant.h"
#include "semutil0.h"
#include "main.h"
#include "soc.h"
#include "dinit.h"
#include "dinitutl.h"
#include "nme.h"
#include "fih.h"
#include "pragma.h"
#include "ccffinfo.h"
#include "llmputil.h"
#include "llassem.h"
#include "cgraph.h"
#include "semsym.h"
#include "llmputil.h"
#include "dtypeutl.h"
#include "exp_rte.h"
#include "symfun.h"
#include <stdarg.h>

static int endilmfile; /* flag for end of file */
static int ilmlinenum = 0;

static char *line = NULL;
static int linelen = 0;
static int pos;

static int do_level = 0;
static int in_array_ctor = 0;
static int oprnd_cnt = 0;
static int passbyflags = 1;
static int cfuncflags = 1;
static int cudaflags = 1;
static int cudaemu = 0; /* 1 => global; 2 => device */
extern int init_list_count;

static int llvm_stb_processing = 0;

static int read_line(void);
static void checkversion(const char *text);
static int checkname(const char *text);
static ISZ_T getval(const char *valname);
static long getlval(const char *valname);
static int getbit(const char *bitname);

#define STB_UPPER() (gbl.stbfil != NULL)
#ifdef FLANG2_UPPER_UNUSED
static void do_llvm_sym_is_refd(void);
#endif
static void build_agoto(void);
static void free_modvar_alias_list(void);
static void save_modvar_alias(SPTR sptr, char *alias_name);

static void init_upper(void);
static void read_fileentries(void);
static void read_datatype(void);
static void read_symbol(void);
static void read_overlap(void);
static void read_Entry(void);
static void read_program(void);
static void read_ipainfo(void);
static int newindex(int);
static int newinfo(void);
static void fix_datatype(void);
static void fix_symbol(void);
static int create_thread_private_vector(int, int);
static DTYPE create_threadprivate_dtype(void);
static int getnamelen(void);
static char *getname(void);
static int getoperand(const char *optype, char letter);

static void read_ilm(void);
static void read_label(void);

static void Begindata(void);
static void Writedata(void);
static void dataDo(void);
static void dataEnddo(void);
static void dataConstant(void);
static void dataReference(void);
static void dataStructure(void);
static void dataVariable(void);
static void read_init(void);
static void data_pop_const(void);
static void data_push_const(void);
static void read_global(void);
static int read_CCFF(void);
#include "fdirect.h"
static void read_contained(void);

typedef struct CGR_LIST {
  struct CGR_LIST *next;
  SPTR func_sptr;
} CGR_LIST;

/* type of descriptor elements */
#define DESC_ELM_DT (XBIT(68, 1) ? DT_INT8 : DT_INT)

typedef struct {
  const char *keyword;
  const char *shortkeyword;
  int keyvalue;
} namelist;

/* clang-format off */
static const namelist IPAtypes[] = {
  { "pstride",  "p",  1 }, { "sstride",     "s",   2 }, { "Target", "T",  3 },
  { "target",   "t",  4 }, { "allcallsafe", "a",   5 }, { "safe",   "f",  6 },
  { "callsafe", "c",  7 }, { NULL,          NULL, -1 },
};

/* list of datatype keywords */
static const namelist Datatypes[] = {
  { "Array",     "A",   TY_ARRAY },  { "Complex8",   "C8", TY_CMPLX },
  { "Complex16", "C16", TY_DCMPLX },
  { "Complex32", "C32", TY_QCMPLX }, { "Derived",    "D",  TY_STRUCT },
  { "Hollerith", "H",   TY_HOLL },   { "Integer1",   "I1", TY_BINT },
  { "Integer2",  "I2",  TY_SINT },   { "Integer4",   "I4", TY_INT },
  { "Integer8",  "I8",  TY_INT8 },   { "Logical1",   "L1", TY_BLOG },
  { "Logical2",  "L2",  TY_SLOG },   { "Logical4",   "L4", TY_LOG },
  { "Logical8",  "L8",  TY_LOG8 },   { "Numeric",    "N",  TY_NUMERIC },
  { "Pointer",   "P",   TY_PTR },    { "proc",       "p",  TY_PROC },
  { "Real2",     "R2",  TY_HALF },
  { "Real4",     "R4",  TY_REAL },   { "Real8",      "R8", TY_DBLE },
  { "Real16",    "R16", TY_QUAD },   { "Struct",     "S",  TY_STRUCT },
  { "Word4",     "W4",  TY_WORD },   { "Word8",      "W8", TY_DWORD },
  { "Union",     "U",   TY_UNION },  { "any",        "a",  TY_ANY },
  { "character", "c",   TY_CHAR },   { "kcharacter", "k",  TY_NCHAR },
  { "none",      "n",   TY_NONE },   { NULL,         NULL, -1 },
};

/* list of symbol type keywords */
static const namelist Symboltypes[] = {
  { "Array",     "A",  ST_ARRAY },   { "Block",     "B",  ST_BLOCK },
  { "Common",    "C",  ST_CMBLK },   { "Derived",   "D",  ST_STRUCT },
  { "Entry",     "E",  ST_ENTRY },   { "Generic",   "G",  ST_GENERIC },
  { "Intrinsic", "I",  ST_INTRIN },  { "Known",     "K",  ST_PD },
  { "Label",     "L",  ST_LABEL },   { "Member",    "M",  ST_MEMBER },
  { "Namelist",  "N",  ST_NML },     { "Procedure", "P",  ST_PROC },
  { "Struct",    "S",  ST_STRUCT },  { "Tag",       "T",  ST_STAG },
  { "Union",     "U",  ST_UNION },   { "Variable",  "V",  ST_VAR },
  { "constant",  "c",  ST_CONST },   { "dpname",    "d",  ST_DPNAME },
  { "list",      "l",  ST_PLIST },
  { "module",    "m",  -99 },        { "parameter", "p",  ST_PARAM },
  { "typedef",   "t",  ST_TYPEDEF }, { NULL,        NULL, -1 },
};
/* list of symbol class keywords */
static const namelist Symbolclasses[] = {
  { "Based",  "B",  SC_BASED },  { "Common",  "C",  SC_CMBLK },
  { "Dummy",  "D",  SC_DUMMY },  { "Extern",  "E",  SC_EXTERN },
  { "Local",  "L",  SC_LOCAL },  { "Private", "P",  SC_PRIVATE },
  { "Static", "S",  SC_STATIC }, { "none",    "n",  SC_NONE },
  { NULL,     NULL, -1 },
};

/* list of subprogram type keywords */
static const namelist Subprogramtypes[] = {
  { "Blockdata", "B",  RU_BDATA }, { "Function",   "F",  RU_FUNC },
  { "Program",   "P",  RU_PROG },  { "Subroutine", "S",  RU_SUBR },
  { NULL,        NULL, -1 },
};
/* clang-format on */

static int symbolcount = 0, datatypecount = 0;
static int oldsymbolcount = 0, olddatatypecount = 0;
static SPTR *symbolxref;
static DTYPE *datatypexref;

static int *agototab;
static int agotosz = 0;
static int agotomax;

typedef struct {
  int type; /* INFO_... below */
  int next; /* next IPAinfo entry for this symbol */
  union {
    struct {
      int indirect; /* integer count of '*'s x 2, plus 1 if imprecise */
      int target;   /* sptr of target, or pseudo target number */
    } target;
    struct {
      int low, high;
    } range;
    struct {
      int info;
    } funcinfo;
    long pstride;
    struct {
      int val1;
      int val2;
    } val;
  } t;
} IPAinfo;

/* values for IPAinfo.type */
#define INFO_GTARGET 1
#define INFO_OGTARGET 2
#define INFO_LTARGET 3
#define INFO_OTARGET 4
#define INFO_FLDYNTARGET 5
#define INFO_FGDYNTARGET 6
#define INFO_FUNKTARGET 7
#define INFO_FOTARGET 8
#define INFO_FSTARGET 9
#define INFO_FOSTARGET 10
#define INFO_RANGE 11
#define INFO_SAFE 12
#define INFO_FUNC 13
#define INFO_NEWSYM 14
#define INFO_NOCONFLICT 15
#define INFO_NOADDR 16
#define INFO_PSTRIDE 17
#define INFO_SSTRIDE 18
#define INFO_ALLCALLSAFE 19
#define INFO_CALLSAFE 20

typedef struct {
  int sptr, info;
} IPAindex;

typedef struct {
  int base, increment;
} SYMinfo;

typedef struct {
  int stmt, lhs, rhs;
} repltype;

typedef struct {
  int lhs, rhs;
} repltemptype;

typedef struct {
  int index, dtype, link;
} typelisttype;

typedef struct {
  int version;

  IPAindex *index;
  int indexsize, indexavl;

  IPAinfo *info;
  int infosize, infoavl;
} IPAB;

/* values for IPNFO_FUNCINFO() */
#define FINFO_WRITEARG 0x01
#define FINFO_READGLOB 0x02
#define FINFO_WRITEGLOB 0x04
#define FINFO_READSTATIC 0x08
#define FINFO_WRITESTATIC 0x10

int IPA_Pointer_Targets_Disambiguated = 0;
int IPA_Safe_Globals_Confirmed = 0;
int IPA_Range_Propagated = 0;
int IPA_Func_Propagated = 0;
int IPA_Pointer_Strides_Propagated = 0;

#if DEBUG

/* print a message, continue */
#define Trace(a) TraceOutput a

static void
TraceOutput(const char *fmt, ...)
{
  va_list argptr;
  va_start(argptr, fmt);

  if (DBGBIT(47, 0x100)) {
    if (gbl.dbgfil) {
      vfprintf(gbl.dbgfil, fmt, argptr);
      fprintf(gbl.dbgfil, "\n");
    } else {
      fprintf(stderr, "Trace: ");
      vfprintf(stderr, fmt, argptr);
      fprintf(stderr, "\n");
    }
    va_end(argptr);
  }
} /* TraceOutput */
#else

/* eliminate the trace output */
#define Trace(a)
#endif

typedef struct alias_syminfo {
  SPTR sptr;
  char *alias;
  struct alias_syminfo *next;
} alias_syminfo;
static alias_syminfo *modvar_alias_list;

/* for processing data initialization */
typedef struct typestack {
  DTYPE dtype;
  SPTR member;
} typestack;

/* for saving outer procedure symbol information for the next internal routine
 */
typedef struct upper_syminfo {
  ISZ_T address;
  ISZ_T clen_address;
  SC_KIND sc;
  int ref : 1;
  int save : 1;
  int memarg;
  int clen_memarg;
} upper_syminfo;

static void restore_saved_syminfo(int);
static int getkeyword(const char *keyname, const namelist NL[]);

static IPAB ipab;
static int errors;

/* keep a stack of information */
static int stack_top, stack_size;
static int **stack;

static typestack *ts; /* type stack */
static int tsl = -1;         /* level in type stack */
static int tssize = 0;       /* level in type stack */

static SPTR *saved_symbolxref;
static int saved_symbolcount = 0;
static upper_syminfo *saved_syminfo;
static int saved_syminfocount = 0;
static upper_syminfo *saved_tpinfo;
static int saved_tpcount = 0;
static int tpcount;
static DTYPE threadprivate_dtype;
static int *ilmxref;
static int ilmxrefsize, origilmavl;

#ifdef __cplusplus
inline SPTR getSptrVal(const char *s) {
  return static_cast<SPTR>(getval(s));
}

inline DTYPE getDtypeVal(const char *s) {
  return static_cast<DTYPE>(getval(s));
}

inline SPTR getSptrOperand(const char *s, char ch) {
  return static_cast<SPTR>(getoperand(s, ch));
}

inline DTYPE getDtypeOperand(const char *s, char ch) {
  return static_cast<DTYPE>(getoperand(s, ch));
}

inline TY_KIND getTYKind(void) {
  return static_cast<TY_KIND>(getkeyword("datatype", Datatypes));
}

inline SYMTYPE getSymType(void) {
  return static_cast<SYMTYPE>(getkeyword("type", Symboltypes));
}

inline SC_KIND getSCKind(void) {
  return static_cast<SC_KIND>(getkeyword("class", Symbolclasses));
}

inline RUTYPE getRUType(void) {
  return static_cast<RUTYPE>(getkeyword("procedure", Subprogramtypes));
}

inline int getIPAType(void) {
  return getkeyword("type", IPAtypes);
}
#else //  !C++
#define getSptrVal      getval
#define getDtypeVal     getval
#define getSptrOperand  getoperand
#define getDtypeOperand getoperand
#define getTYKind()     getkeyword("datatype", Datatypes)
#define getSymType()    getkeyword("type", Symboltypes)
#define getSCKind()     getkeyword("class", Symbolclasses)
#define getRUType()     getkeyword("procedure", Subprogramtypes)
#define getIPAType()    getkeyword("type", IPAtypes)
#endif // C++

#define IPNDX_SPTR(i) ipab.index[i].sptr
#define IPNDX_INFO(i) ipab.index[i].info
#define IPNFO_TYPE(i) ipab.info[i].type
#define IPNFO_NEXT(i) ipab.info[i].next
#define IPNFO_INDIRECT(i) (ipab.info[i].t.target.indirect >> 1)
#define IPNFO_IMPRECISE(i) (ipab.info[i].t.target.indirect & 0x01)
#define IPNFO_SET(i, indirect, imprecise) \
  (ipab.info[i].t.target.indirect = indirect << 1 + (imprecise ? 1 : 0))
#define IPNFO_SET_IMPRECISE(i) (ipab.info[i].t.target.indirect |= 1)
#define IPNFO_TARGET(i) ipab.info[i].t.target.target
#define IPNFO_LOW(i) ipab.info[i].t.range.low
#define IPNFO_HIGH(i) ipab.info[i].t.range.high
#define IPNFO_FUNCINFO(i) ipab.info[i].t.funcinfo.info
#define IPNFO_PSTRIDE(i) ipab.info[i].t.pstride
#define IPNFO_SSTRIDE(i) ipab.info[i].t.pstride
#define IPNFO_VAL(i) ipab.info[i].t.val.val1
#define IPNFO_VAL2(i) ipab.info[i].t.val.val2

/**
 * \brief Entry point for reading in ILM file
 *
 * Size of private array allocated by frontend - the frontend will allocate
 * space for a descriptor and its pointer & offset variables since there
 * is an assumed sequence of allocation.
 */
void
upper(int stb_processing)
{
  ISZ_T size;
  SPTR first;
  int firstinternal, gstaticbase;
  extern void set_private_size(ISZ_T);

  llvm_stb_processing = stb_processing;
  init_upper();

  /* read first line */
  endilmfile = read_line();
  if (endilmfile) {
    /* must be done! */
    gbl.eof_flag = 1;
    return;
  }
  if (line[0] == 'C') {
    /* check for end of module */
    if (strncmp(line, "CONSTRUCTORACC", 14) == 0) {
      gbl.bss_addr = 0;
      gbl.saddr = 0;
      gbl.locaddr = 0;
      gbl.statics = NOSYM;
      gbl.locals = NOSYM;
      gbl.cuda_constructor = true;
      gbl.paddr = 0;
      gbl.internal = 0;
      return;
    }
  }
  checkversion("TOILM");

  endilmfile = read_line();
  gbl.internal = getval("Internal");

  if (gbl.internal > 1) {
    --gbl.numcontained;
    endilmfile = read_line();
    gbl.outersub = getSptrVal("Outer");
    endilmfile = read_line();
    firstinternal = getval("First");
  } else {
    gbl.outersub = SPTR_NULL;
    gbl.numcontained = 0;
    firstinternal = stb.firstusym;
  }

  endilmfile = read_line();
  symbolcount = getval("Symbols");
  oldsymbolcount = stb.stg_avail - 1;
  NEW(symbolxref, SPTR, symbolcount + 1);
  BZERO(symbolxref, SPTR, symbolcount + 1);

  endilmfile = read_line();
  datatypecount = getval("Datatypes");
  olddatatypecount = stb.dt.stg_avail - 1;
  NEW(datatypexref, DTYPE, datatypecount + 1);
  BZERO(datatypexref, DTYPE, datatypecount + 1);

  ilmxrefsize = 100;
  NEW(ilmxref, int, ilmxrefsize);
  BZERO(ilmxref, int, ilmxrefsize);
  origilmavl = 0;

  endilmfile = read_line();
  size = getval("BSS");
  gbl.bss_addr = size;

  endilmfile = read_line();
  size = getval("GBL");
  gbl.saddr = size;

  endilmfile = read_line();
  size = getval("LOC");
  gbl.locaddr = size;

  endilmfile = read_line();
  first = getSptrVal("STATICS");
  gbl.statics = first;

  endilmfile = read_line();
  first = getSptrVal("LOCALS");
  gbl.locals = first;

  endilmfile = read_line();
  size = getval("PRIVATES");
  set_private_size(size);

  endilmfile = read_line();
  gstaticbase = 0;
  while (!endilmfile) {
    /* read datatypes, symbols */
    switch (line[0]) {
    case 'd':
      read_datatype();
      break;
    case 's':
      read_symbol();
      break;
    case 'o':
      read_overlap();
      break;
    case 'E':
      read_Entry();
      break;
    case 'p':
      read_program();
      break;
    case 'f':
      read_fileentries();
      break;
    case 'i':
      read_ipainfo();
      break;
    case 'e':
      endilmfile = 1;
      break;
    case 'c':
      read_contained();
      break;
    case 'g':
      read_global();
      break;
    case 'G':
      gstaticbase = getval("GNAME");
      break;
    case 'x':
      if (line[1] == 'l') {
      }
      break;
    default:
      fprintf(stderr, "ILM error: line %d unknown line type %c\n", ilmlinenum,
              line[0]);
      ++errors;
      break;
    }
    /* don't read next line if this was the end line */
    if (!endilmfile)
      endilmfile = read_line();
  }
  fix_symbol();
  fix_datatype();

#if DEBUG
  if (DBGBIT(47, 0x200)) {
    dmp_dtype();
    symdmp(gbl.dbgfil, 0);
  }
#endif
  if (STB_UPPER()) {
    if (endilmfile) {
      goto do_pastilm;
    }
  }
  endilmfile = read_line();
  if (checkname("CCFF")) {
    endilmfile = read_CCFF();
    if (!endilmfile)
      read_line(); /* read line past CCFF messages */
  }

  if (STB_UPPER()) {
    goto do_pastilm;
  }

  /* import the ILMs */

  /* check first line */
  checkversion("AST2ILM");

  endilmfile = read_line();
  while (!endilmfile) {
    switch (line[0]) {
    case 'B':
      /* Begindata */
      Begindata();
      break;
    case 'C':
      /* Data Constant repeatcount datatype symbol [value | symbol] */
      dataConstant();
      break;
    case 'D':
      /* Data Do indvar lower upper step */
      dataDo();
      break;
    case 'E':
      /* Data Enddo */
      dataEnddo();
      break;
    case 'e':
      /* end */
      endilmfile = 1;
      break;
    case 'i':
      /* ilm */
      read_ilm();
      break;
    case 'I':
      /* initialization */
      read_init();
      break;
    case 'l':
      /* label */
      read_label();
      break;
    case 'R':
      /* data Reference ilm type */
      dataReference();
      break;
    case 's':
      /* structure repeatcount datatype symbol no_dinitp */
      dataStructure();
      break;
    case 't':
      /* tructurend */
      data_pop_const();
      break;
    case 'V':
      /* data Variable ilm type */
      dataVariable();
      break;
    case 'W':
      /* Writedata: end of data statement */
      Writedata();
      break;
    default:
      fprintf(stderr, "ILM error: line %d unknown line type %c\n", ilmlinenum,
              line[0]);
      ++errors;
      break;
    }
    /* don't read next line if this was the end line */
    if (!endilmfile)
      endilmfile = read_line();
  }

do_pastilm:
  if (ts)
    FREE(ts);
  if (stack)
    FREE(stack);
  FREE(datatypexref);
  FREE(ilmxref);

  if (gbl.internal) {
    /* must be done here before freeing symbolxref and saved_symbolxref */
    fixup_llvm_uplevel_symbol();
  }
  if (agotosz) {
    build_agoto();
  }

  switch (gbl.internal) {
  case 0:
    /* no internal routines */
    FREE(symbolxref);
    symbolxref = NULL;
    /* get rid of stuff from previous containing routine, if any */
    if (saved_symbolxref) {
      FREE(saved_symbolxref);
      saved_symbolxref = NULL;
      saved_symbolcount = 0;
    }
    if (saved_syminfo) {
      FREE(saved_syminfo);
      saved_syminfo = NULL;
      saved_syminfocount = 0;
    }
    gbl.outersub = SPTR_NULL;
    break;
  case 1:
    /* outer routine having internal routines */
    /* get rid of stuff from previous containing routine, if any */
    if (saved_symbolxref) {
      FREE(saved_symbolxref);
      saved_symbolxref = NULL;
      saved_symbolcount = 0;
    }
    if (saved_syminfo) {
      FREE(saved_syminfo);
      saved_syminfo = NULL;
      saved_syminfocount = 0;
    }
    saved_symbolxref = symbolxref;
    saved_symbolcount = symbolcount;
    /* this is how many symbols we need to save information for */
    saved_syminfocount = stb.stg_avail;
    symbolxref = NULL;
    gbl.outersub = gbl.currsub;
    if (saved_tpinfo) {
      FREE(saved_tpinfo);
      saved_tpinfo = NULL;
      saved_tpcount = 0;
    }
    break;
  default:
    /* inner routine; restore saved information */
    restore_saved_syminfo(firstinternal);
    FREE(symbolxref);
    symbolxref = NULL;
    /* keep the old 'syminfo' and 'saved_symbolxref' for next routine */
    break;
  }

/* import the DIRECTIVES */

/* read first line */
  if (!STB_UPPER()) {
    endilmfile = read_line();
    checkversion("DIRECTIVES");
    ilmlinenum += direct_import(gbl.srcfil);
  } else if (endilmfile) {
    goto do_dchar;
  }
  endilmfile = read_line(); /* end */
  if (line[0] == 'e') {
    endilmfile = 1;
  } else {
    errors++;
  }

  if (STB_UPPER()) {
    goto do_dchar;
  }

  do_dinit();
  /* if we are using the global ILM structure,
   * look for assumed-length or deferred-length character dummy arguments.
   * get a temp for the character length */
do_dchar:
  if (XBIT(14, 0x20000) || !XBIT(14, 0x10000)) {
    int e, dpdsc, paramct, i;
    for (e = gbl.entries; e > NOSYM; e = SYMLKG(e)) {
      dpdsc = DPDSCG(e);
      paramct = PARAMCTG(e);
      for (i = 0; i < paramct; ++i) {
        int param, dtype;
        param = aux.dpdsc_base[dpdsc + i];
        dtype = DDTG(DTYPEG(param));
        if (dtype == DT_DEFERCHAR || dtype == DT_DEFERNCHAR ||
            dtype == DT_ASSCHAR || dtype == DT_ASSNCHAR) {
          if (!CLENG(param)) {
            int clen;
            clen = getdumlen();
            CLENP(param, clen);
            PARREFP(clen, PARREFG(param));
          }
        }
      }
    }
  }

  if (gstaticbase) {
    create_static_base(gstaticbase);
  }
  freearea(4); /* free memory used to build static initializations */
  if (errors) {
    interr("Errors in ILM file", errors, ERR_Fatal);
  }
  llvm_stb_processing = 0;
} /* upper */

/**
   \brief For outer routines that contain inner routines, make sure all
   variables get an address, even if never used in this routine, in case they
   are used by the contained routines.
 */
void
upper_assign_addresses(void)
{
  if (gbl.internal == 1) {
    SPTR sptr;
    for (sptr = (SPTR) stb.firstusym; sptr < stb.stg_avail; ++sptr) {
      switch (STYPEG(sptr)) {
      case ST_VAR:
      case ST_ARRAY:
      case ST_STRUCT:
      case ST_UNION:
      case ST_PLIST:
        if (REFG(sptr) == 0) {
          switch (SCG(sptr)) {
          case SC_LOCAL:
          case SC_STATIC:
            hostsym_is_refd(sptr);
            break;
          default:
            break;
          }
        }
        break;
      default:
        break;
      }
    }
  }
} /* upper_assign_addresses */

static void
restore_saved_syminfo(int firstinternal)
{
  int s;
  SPTR newsptr, oldsptr;
  SC_KIND sc;
  int ref, save;
  ISZ_T address;

  if (gbl.internal < 2)
    return;
  for (s = 1; s <= saved_symbolcount; ++s) {
    /* has this symbol been imported for this internal routine? */
    if (s > symbolcount)
      break;
    if (s >= firstinternal)
      break;
    newsptr = symbolxref[s];
    if (newsptr == 0)
      continue;
    oldsptr = saved_symbolxref[s];
    if (oldsptr >= saved_syminfocount)
      continue;
    sc = saved_syminfo[oldsptr].sc;
    address = saved_syminfo[oldsptr].address;
    ref = saved_syminfo[oldsptr].ref;
    save = saved_syminfo[oldsptr].save;
    switch (STYPEG(newsptr)) {
    case ST_PLIST:
    case ST_VAR:
    case ST_ARRAY:
    case ST_STRUCT:
    case ST_UNION:
      if (sc == SC_DUMMY) {
        SCP(newsptr, sc);
        ADDRESSP(newsptr, address);
        REFP(newsptr, ref);
        MEMARGP(newsptr, saved_syminfo[oldsptr].memarg);
        if (saved_syminfo[oldsptr].clen_address) {
          int clen;
          clen = gethost_dumlen(newsptr, saved_syminfo[oldsptr].clen_address);
          CLENP(newsptr, clen);
          MEMARGP(clen, saved_syminfo[oldsptr].clen_memarg);
        }
#if DEBUG
        if (sc != SCG(newsptr)) {
          Trace(("outer procedure dummy %d name %s had (sc)=(%d) now (%d)",
                 newsptr, SYMNAME(newsptr), sc, (int)SCG(newsptr)));
          fprintf(stderr,
                  "ILM error: internal routine gets bad sclass for "
                  "outer variable %s\n",
                  SYMNAME(newsptr));
          ++errors;
        }
#endif
      } else if (REFG(newsptr) && (save || !SAVEG(newsptr))) {
        /* allow for the case where the SAVE flag was optimized away */
        /* compare the saved REF, ADDRESS, SC fields */
        if (REREFG(newsptr)) {
          /* handle special case when REREF flag is set. See
           * comment for REREF in fix_symbol().
           */
          ADDRESSP(newsptr, address);
        }
        if (sc != SCG(newsptr) || address != ADDRESSG(newsptr) || ref == 0) {
          if (sc || address || ref) {
            Trace(("outer procedure symbol %d name %s had "
                   "(sc,address,ref)=(%d,%" ISZ_PF "d,%d) now (%d,%" ISZ_PF
                   "d,%d)",
                   newsptr, SYMNAME(newsptr), sc, address, ref,
                   (int)SCG(newsptr), (int)ADDRESSG(newsptr),
                   (int)REFG(newsptr)));
            fprintf(stderr,
                    "ILM error: internal routine gets bad address for "
                    "outer variable %s\n",
                    SYMNAME(newsptr));
            ++errors;
          }
        }
      } else if (ref) {
        /* get the saved REF, ADDRESS, SC fields */
        if (sc == SC_LOCAL) {
          SCP(newsptr, sc);
          SAVEP(newsptr, save);
          ADDRESSP(newsptr, address);
          REFP(newsptr, ref);
          if (!UPLEVELG(newsptr)) {
            SYMLKP(newsptr, gbl.locals);
            gbl.locals = newsptr;
          }
        } else if (sc == SC_STATIC) {
          SCP(newsptr, sc);
          ADDRESSP(newsptr, address);
          REFP(newsptr, ref);
          if (!UPLEVELG(newsptr)) {
            SYMLKP(newsptr, gbl.statics);
            gbl.statics = newsptr;
          }
        } else {
          Trace(("unknown restore (sc,address,ref)=(%d,%" ISZ_PF "d,%d)", sc,
                 address, ref));
        }
      }
      if (IS_THREAD_TP(newsptr)) {
        int tptr;
        int psptr;

        switch (SCG(newsptr)) {
        case SC_LOCAL:
        case SC_STATIC:
          if (UPLEVELG(newsptr) && !MIDNUMG(newsptr)) {
            tptr = create_thread_private_vector(newsptr, oldsptr);
            MIDNUMP(tptr, newsptr);
            MIDNUMP(newsptr, tptr);
            if (!XBIT(69, 0x80))
              SCP(tptr, SC_STATIC);
          }
          break;
        case SC_BASED:
          psptr = MIDNUMG(newsptr);
          if ((SCG(psptr) == SC_LOCAL || SCG(psptr) == SC_STATIC) &&
              UPLEVELG(psptr)) {
            if (POINTERG(newsptr)) {
              /*
               * Cannot rely on the SYMLK chain appearing as
               *     $p -> $o -> $sd
               * Apparently, these links only occur for the
               * pointer's internal variables if the pointer
               * does not have the SAVE attribute.  Without
               * these fields, the correct size of the threads'
               * copies cannot be computed.
               * Just explicitly look for the internal pointer
               * and descriptor. If the descriptor is present,
               * can assume that there is an offest variable which
               * only needs to be accounted for in the size
               * computation of the threads' copies.
               * Setup up the MIDNUM fields as follows where
               * foo is the symtab entry which has the POINTER
               * flag set:
               *    foo    -> foo$p
               *    TPpfoo -> foo
               *    foo$p  -> TPpfoo
               *    foo$sd -> TPpfoo
               * Note that foo's SDSC -> foo$sd.
               * Before we had:
               *    foo    -> TPpfoo
               *    TPpfoo -> foo$p
               * which is a problem for computing the size
               * when starting with TPpfoo.
               */
              int sdsptr;
              tptr = create_thread_private_vector(psptr, oldsptr);
              THREADP(psptr, 1);
              MIDNUMP(newsptr, psptr);
              MIDNUMP(tptr, newsptr);
              MIDNUMP(psptr, tptr);
              sdsptr = SDSCG(newsptr);
              if (sdsptr) {
                THREADP(sdsptr, 1);
                MIDNUMP(sdsptr, tptr);
              }
            } else {
              /*
               * Given the above code for POINTER, this code is
               * probably dead, but leave it just in case.
               */
              tptr = create_thread_private_vector(psptr, oldsptr);
              THREADP(psptr, 1);
              MIDNUMP(newsptr, tptr);
              MIDNUMP(tptr, psptr);
              MIDNUMP(psptr, tptr);
              if (SYMLKG(psptr) != NOSYM) {
                psptr = symbolxref[SYMLKG(psptr)];
                THREADP(psptr, 1);
                MIDNUMP(psptr, tptr);
                if (SYMLKG(psptr) != NOSYM) {
                  psptr = symbolxref[SYMLKG(psptr)];
                  THREADP(psptr, 1);
                  MIDNUMP(psptr, tptr);
                }
              }
            }
          }
          break;
        default:
          break;
        }
      }
      break;

    case ST_PROC:
      /* assertion: must be a dummy procedure */
      ADDRESSP(newsptr, address);
      MEMARGP(newsptr, saved_syminfo[oldsptr].memarg);
      break;
    default:
      break;
    }
  }

} /* restore_saved_syminfo */

/**
   \brief Save information about symbols for this outer routine to restore
   inside other inner routines.
 */
void
upper_save_syminfo(void)
{
  int s, sptr;

  if (gbl.internal != 1)
    return;
  /* allocate a saved_syminfo; only need info for symbols imported;
   * saved_syminfocount set for gbl.internal==1 in upper() */
  NEW(saved_syminfo, upper_syminfo, saved_syminfocount + 1);
  BZERO(saved_syminfo, upper_syminfo, saved_syminfocount + 1);
  for (s = 1; s <= saved_symbolcount; ++s) {
    sptr = saved_symbolxref[s];
    if (sptr == 0)
      continue;
    switch (STYPEG(sptr)) {
    case ST_VAR:
    case ST_ARRAY:
    case ST_STRUCT:
    case ST_UNION:
    case ST_PLIST:
      if (REFG(sptr) || GSCOPEG(sptr) || SCG(sptr) == SC_DUMMY) {
        saved_syminfo[sptr].sc = SCG(sptr);
        saved_syminfo[sptr].address = ADDRESSG(sptr);
        saved_syminfo[sptr].ref = REFG(sptr) | GSCOPEG(sptr);
        saved_syminfo[sptr].save = SAVEG(sptr);
        saved_syminfo[sptr].clen_address = 0;
        saved_syminfo[sptr].clen_memarg = 0;
        if (SCG(sptr) == SC_DUMMY) {
          if (DDTG(DTYPEG(sptr)) == DT_ASSCHAR ||
              DDTG(DTYPEG(sptr)) == DT_ASSNCHAR) {
            saved_syminfo[sptr].clen_address = ADDRESSG(CLENG(sptr));
            saved_syminfo[sptr].clen_memarg = MEMARGG(CLENG(sptr));
          } else if (DDTG(DTYPEG(sptr)) == DT_DEFERCHAR ||
                     DDTG(DTYPEG(sptr)) == DT_DEFERNCHAR) {
            saved_syminfo[sptr].clen_address = ADDRESSG(CLENG(sptr));
            saved_syminfo[sptr].clen_memarg = MEMARGG(CLENG(sptr));
          }
          saved_syminfo[sptr].memarg = MEMARGG(sptr);
        }
      }
      break;
    case ST_PROC:
      if (SCG(sptr) == SC_DUMMY) {
        /* sc & reg aren't needed but are copied to prevent * UMRs. */
        saved_syminfo[sptr].sc = SCG(sptr);
        saved_syminfo[sptr].address = ADDRESSG(sptr);
        saved_syminfo[sptr].ref = REFG(sptr);
        saved_syminfo[sptr].memarg = MEMARGG(sptr);
      }
      break;
    default:
      break;
    }
  }
  if (tpcount) {
    int cnt;
    NEW(saved_tpinfo, upper_syminfo, tpcount + 1);
    cnt = 0;
    for (sptr = gbl.threadprivate; sptr > NOSYM; sptr = TPLNKG(sptr)) {
      /*
      if (STYPEG(MIDNUMG(sptr)) == ST_CMBLK)
          continue;
      */
      saved_tpinfo[cnt].sc = SCG(sptr);
      saved_tpinfo[cnt].address = ADDRESSG(sptr);
      saved_tpinfo[cnt].ref = REFG(sptr);
      saved_tpinfo[cnt].memarg = MIDNUMG(sptr);
      cnt++;
    }
    saved_tpcount = cnt;
  }
} /* upper_save_syminfo */

static void
init_upper(void)
{
  gbl.entries = NOSYM;
  gbl.cuda_constructor = false;
  soc.avail = 1;

  errors = 0;

  stack_top = 0;
  stack_size = 0;
  stack = NULL;
  tsl = -1;
  ts = NULL;
  tssize = 0;
  if (linelen == 0) {
    linelen = 4096;
    line = (char *)malloc(linelen * sizeof(char));
  }
  if (ipab.index == NULL) {
    ipab.indexsize = 100;
    NEW(ipab.index, IPAindex, ipab.indexsize);
  }
  ipab.indexavl = 0;
  if (ipab.info == NULL) {
    ipab.infosize = 100;
    NEW(ipab.info, IPAinfo, ipab.infosize);
  }
  ipab.infoavl = 1;
  if (modvar_alias_list) {
    free_modvar_alias_list();
  }
} /* init_upper */

/*
 * called from main
 * read the 'inline' information saved in the ilm file
 */
void
upper_init(void)
{
  int end;
  end = read_line();
  while (line[0] == 'i') {
    char *name, *cname, *filename;
    int level, which, namelen, cnamelen, filenamelen, base, size;
    long offset, objoffset;
    /* an 'inline' line */
    level = getval("inline");
    offset = getlval("offset");
    which = getval("which");
    cnamelen = getnamelen();
    cname = line + pos;
    pos += cnamelen;
    namelen = getnamelen();
    name = line + pos;
    pos += namelen;
    filenamelen = getnamelen();
    filename = line + pos;
    pos += filenamelen;
    objoffset = getlval("objoffset");
    base = getval("base");
    size = getval("size");
    name[namelen] = '\0';
    cname[cnamelen] = '\0';
    filename[filenamelen] = '\0';
    end = read_line();
  }

} /* upper_init */

static int
read_line(void)
{
  int i, ch;
  i = 0;
  pos = 0;
  while (1) {
    if (STB_UPPER())
      ch = fgetc(gbl.stbfil); /* fgetc() returns an int */
    else
      ch = fgetc(gbl.srcfil); /* fgetc() returns an int */
    if (i >= linelen) {
      if (linelen == 0) {
        linelen = 4096;
        line = (char *)malloc(linelen * sizeof(char));
      } else {
        linelen = linelen * 2;
        line = (char*) realloc(line, linelen);
      }
    }
    if (ch == EOF || (char)ch == '\n') {
      line[i] = '\0';
      break;
    }
    line[i] = (char)ch;
    ++i;
  }

  ++ilmlinenum;
  if (ch == EOF && i == 0)
    return 1;
  return 0;
} /* read_line */

static void
checkversion(const char *text)
{
  int ret;
  char check[50];
  int v1, v2;

  v1 = v2 = 0;
  check[0] = '\0';
  ret = sscanf(line, "%s version %d/%d", check, &v1, &v2);
  if (ret != 3 || v1 != VersionMajor || strcmp(text, check) != 0) {
    fprintf(stderr,
            "IILM file version error\n"
            "Expecting %s version %d/%d\n"
            "      got %s version %d/%d\n",
            text, VersionMajor, VersionMinor, check, v1, v2);
    exit(1);
  }
  if (v2 != VersionMinor) {
    switch (VersionMajor) {
    case 1:
      /*
       * The PASSBYVAL & PASSBYREF flags are new to 1.10
       * If the version
       */
      if (v2 < 10 && VersionMinor >= 10) {
        passbyflags = 0;
        return;
      }
      /* CFUNC for variables are new t. 1.1 :
         make the externally visable  variables
         compatible with the equivalent C extern
       */
      if (v2 < 11 && VersionMinor >= 11) {
        cfuncflags = 0;
        return;
      }
      if (v2 < 15 && VersionMinor >= 15) {
        cudaflags = 0;
        return;
      }
    }
    fprintf(stderr,
            "ILM file version error\n"
            "Expecting %s version %d/%d\n"
            "      got %s version %d/%d\n",
            text, VersionMajor, VersionMinor, check, v1, v2);
    exit(1);
  }
} /* checkversion */

/* skip white space */
static void
skipwhitespace(void)
{
  while (line[pos] <= ' ' && line[pos] != '\0')
    ++pos;
} /* skipwhitespace */

/* check that the name matches */
static int
checkname(const char *name)
{
  int i;
  if ((line[pos] == name[0]) && (line[pos + 1] == ':')) {
    /* short version of file, just initial letter of each field */
    pos += 2;
    return 1;
  }
  for (i = 0; name[i] && line[pos + i]; ++i) {
    if (line[pos + i] != name[i])
      return 0;
  }
  if (line[pos + i] == '\n' || line[pos + i] == ' ' || line[pos + i] == '\0') {
    pos += i;
    return 1;
  }
  if (line[pos + i] == ':') {
    pos += i + 1; /* skip past colon */
    return 1;
  }
  return 0;
} /* checkname */

/* check that the name matches */
static int
checkbitname(const char *name)
{
  int i;
  if ((line[pos] == name[0]) &&
      (line[pos + 1] == '-' || line[pos + 1] == '+')) {
    /* short version of file, just initial letter of each field */
    ++pos;
    return 1;
  }
  for (i = 0; name[i] && line[pos + i]; ++i) {
    if (line[pos + i] != name[i])
      return 0;
  }
  if (line[pos + i] == '+' || line[pos + i] == '-') {
    pos += i;
    return 1;
  }
  return 0;
} /* checkbitname */

static ISZ_T
getval(const char *valname)
{
  ISZ_T val, neg;

  if (endilmfile) {
    fprintf(stderr, "ILM file: looking past end-of-file for value %s\n",
            valname);
    ++errors;
    return 0;
  }

  skipwhitespace();

  if (!checkname(valname)) {
    fprintf(stderr,
            "ILM file line %d: expecting value for %s\n"
            "instead got: %s\n",
            ilmlinenum, valname, line + pos);
    ++errors;
    return 0;
  }

  val = 0;
  neg = 1;
  if (line[pos] == '-') {
    ++pos;
    neg = -1;
  }
  while (line[pos] >= '0' && line[pos] <= '9') {
    val = val * 10 + (line[pos] - '0');
    ++pos;
  }
  val *= neg;
  Trace((" %s=%d", valname, val));
  return val;
} /* getval */

static long
getlval(const char *valname)
{
  long val, neg;

  if (endilmfile) {
    fprintf(stderr, "ILM file: looking past end-of-file for value %s\n",
            valname);
    ++errors;
    return 0;
  }

  skipwhitespace();

  if (!checkname(valname)) {
    fprintf(stderr,
            "ILM file line %d: expecting value for %s\n"
            "instead got: %s\n",
            ilmlinenum, valname, line + pos);
    ++errors;
    return 0;
  }

  val = 0;
  neg = 1;
  if (line[pos] == '-') {
    ++pos;
    neg = -1;
  }
  while (line[pos] >= '0' && line[pos] <= '9') {
    val = val * 10 + (line[pos] - '0');
    ++pos;
  }
  val *= neg;
  Trace((" %s=%d", valname, val));
  return val;
} /* getlval */

static int
getbit(const char *bitname)
{
  if (endilmfile) {
    fprintf(stderr, "ILM file: looking past end-of-file for bit %s\n", bitname);
    ++errors;
    return 0;
  }

  skipwhitespace();

  if (!checkbitname(bitname)) {
    fprintf(stderr,
            "ILM file line %d: expecting bit %s\n"
            "instead got: %s\n",
            ilmlinenum, bitname, line + pos);
    ++errors;
    return 0;
  }

  if (line[pos] == '-') {
    ++pos;
    Trace((" %s-", bitname));
    return 0;
  }
  if (line[pos] == '+') {
    ++pos;
    Trace((" %s+", bitname));
    return 1;
  }
  fprintf(stderr,
          "ILM file line %d: expecting +/- value for %s\n"
          "instead got: %s\n",
          ilmlinenum, bitname, line + pos);
  ++errors;
  return 0;
} /* getbit */

/* get a pair of numbers first:second */
static void
getpair(SPTR *first, SPTR *second)
{
  int val, neg;
  if (endilmfile) {
    fprintf(stderr, "ILM file: looking past end-of-file for number pair\n");
    *first = *second = SPTR_NULL;
    ++errors;
    return;
  }

  skipwhitespace();

  val = 0;
  neg = 1;
  if (line[pos] == '-') {
    ++pos;
    neg = -1;
  }
  while (line[pos] >= '0' && line[pos] <= '9') {
    val = val * 10 + (line[pos] - '0');
    ++pos;
  }
  *first = (SPTR)(val * neg);

  if (line[pos] != ':') {
    fprintf(stderr,
            "ILM file line %d: expecting number pair\n"
            "instead got: %s\n",
            ilmlinenum, line + pos);
    *second = SPTR_NULL;
    ++errors;
    return;
  }
  ++pos;

  val = 0;
  neg = 1;
  if (line[pos] == '-') {
    ++pos;
    neg = -1;
  }
  while (line[pos] >= '0' && line[pos] <= '9') {
    val = val * 10 + (line[pos] - '0');
    ++pos;
  }
  *second = (SPTR)(val * neg);
} /* getpair */

static int
getnum(void)
{
  int val;

  if (endilmfile) {
    fprintf(stderr, "ILM file: looking past end-of-file for number\n");
    ++errors;
    return 0;
  }

  skipwhitespace();

  val = 0;
  while (line[pos] >= '0' && line[pos] <= '9') {
    val = val * 10 + (line[pos] - '0');
    ++pos;
  }
  Trace((" %d", val));
  return val;
} /* getnum */

static int
gethex(void)
{
  int val;
  char ch;

  if (endilmfile) {
    fprintf(stderr, "ILM file: looking past end-of-file for hex value\n");
    ++errors;
    return 0;
  }

  skipwhitespace();

  val = 0;
  while (1) {
    ch = line[pos];
    if (ch >= '0' && ch <= '9') {
      val = val * 16 + (line[pos] - '0');
    } else if (ch >= 'a' && ch <= 'f') {
      val = val * 16 + (line[pos] - 'a') + 10;
    } else if (ch >= 'A' && ch <= 'F') {
      val = val * 16 + (line[pos] - 'A') + 10;
    } else {
      break;
    }
    ++pos;
  }
  Trace((" %x", val));
  return val;
} /* gethex */

static int
match(const char *K)
{
  int j;
  for (j = 0; K[j]; ++j) {
    if (K[j] != line[pos + j]) {
      return 0;
    }
  }
  if (line[pos + j] <= ' ') { /* all matched */
    pos += j;
    return 1;
  }
  return 0;
} /* match */

static int
getkeyword(const char *keyname, const namelist NL[])
{
  int i;
  if (endilmfile) {
    fprintf(stderr, "ILM file: looking past end-of-file for %s keyword\n",
            keyname);
    ++errors;
    return 0;
  }

  skipwhitespace();

  for (i = 0; NL[i].keyword; ++i) {
    if (line[pos] == NL[i].keyword[0]) {
      /* check this keyword and shortkeyword */
      if (match(NL[i].keyword)) {
        Trace((" %s=%s", keyname, NL[i].keyword));
        return NL[i].keyvalue;
      }
      if (match(NL[i].shortkeyword)) {
        Trace((" %s=%s", keyname, NL[i].keyword));
        return NL[i].keyvalue;
      }
    }
  }
  fprintf(stderr, "ILM File line %d: no match for %s keyword\n", ilmlinenum,
          keyname);
  ++errors;
  return -1;
} /* getkeyword */

static int
getnamelen(void)
{
  int val;
  if (endilmfile) {
    fprintf(stderr, "ILM file: looking past end-of-file for name\n");
    ++errors;
    return 0;
  }

  skipwhitespace();

  val = 0;
  while (line[pos] >= '0' && line[pos] <= '9') {
    val = val * 10 + (line[pos] - '0');
    ++pos;
  }
  if (line[pos] == ':')
    ++pos;
  Trace((" %d:", val));
  return val;
} /* getnamelen */

static char *
getname(void)
{
  int len;
  char *name;
  len = getnamelen();
  name = line + pos;
  pos += len + 1;
  name[len] = '\0';
  return name;
} /* getname */

static void
read_datatype(void)
{
  DTYPE dtype, dt;
  TY_KIND dval;
  SPTR member;
  int align;
  DTYPE subtype;
  int ndim;
  SPTR lower, upper;
  int i;
  SPTR tag;
  ISZ_T size;
  ADSC *ad;
  SPTR iface;
  int paramct, dpdsc;
  SPTR fval;

  dtype = getDtypeVal("datatype");
  dval = getTYKind();
  switch (dval) {
  default:
    break;
  case TY_CMPLX:
    datatypexref[dtype] = DT_CMPLX;
    break;
  case TY_DCMPLX:
    datatypexref[dtype] = DT_DCMPLX;
    break;
  case TY_QCMPLX:
    datatypexref[dtype] = DT_QCMPLX;
    break;
  case TY_HOLL:
    datatypexref[dtype] = DT_HOLL;
    break;
  case TY_BINT:
    datatypexref[dtype] = DT_BINT;
    break;
  case TY_SINT:
    datatypexref[dtype] = DT_SINT;
    break;
  case TY_INT:
    datatypexref[dtype] = DT_INT;
    break;
  case TY_INT8:
    datatypexref[dtype] = DT_INT8;
    break;
  case TY_BLOG:
    datatypexref[dtype] = DT_BLOG;
    break;
  case TY_SLOG:
    datatypexref[dtype] = DT_SLOG;
    break;
  case TY_LOG:
    datatypexref[dtype] = DT_LOG;
    break;
  case TY_LOG8:
    datatypexref[dtype] = DT_LOG8;
    break;
  case TY_NUMERIC:
    datatypexref[dtype] = DT_NUMERIC;
    break;
  case TY_REAL:
    datatypexref[dtype] = DT_REAL;
    break;
  case TY_DBLE:
    datatypexref[dtype] = DT_DBLE;
    break;
  case TY_QUAD:
    datatypexref[dtype] = DT_QUAD;
    break;
  case TY_WORD:
    datatypexref[dtype] = DT_WORD;
    break;
  case TY_DWORD:
    datatypexref[dtype] = DT_DWORD;
    break;
  case TY_ANY:
    datatypexref[dtype] = DT_ANY;
    break;
  case TY_NONE:
    datatypexref[dtype] = DT_NONE;
    break;

  case TY_STRUCT:
  case TY_UNION:
    member = getSptrVal("member");
    size = getval("size");
    tag = getSptrVal("tag");
    align = getval("align");
    dt = get_type(6, dval, NOSYM);
    datatypexref[dtype] = dt;
    DTySetAlgTy(dt, member, size, tag, align, 0);
    break;
  case TY_CHAR:
    size = getval("len");
    if (size == -1) {
      datatypexref[dtype] = DT_ASSCHAR;
    } else if (size == -2) {
      datatypexref[dtype] = DT_ASSCHAR;
    } else if (size == -3) {
      datatypexref[dtype] = DT_DEFERCHAR;
    } else if (size == -4) {
      datatypexref[dtype] = DT_DEFERCHAR;
    } else {
      datatypexref[dtype] = get_type(2, dval, size);
    }
    break;
  case TY_NCHAR:
    size = getval("len");
    if (size == -1) {
      datatypexref[dtype] = DT_ASSNCHAR;
    } else if (size == -2) {
      datatypexref[dtype] = DT_ASSNCHAR;
    } else if (size == -3) {
      datatypexref[dtype] = DT_DEFERNCHAR;
    } else if (size == -4) {
      datatypexref[dtype] = DT_DEFERNCHAR;
    } else {
      datatypexref[dtype] = get_type(2, dval, size);
    }
    break;
  case TY_ARRAY:
    subtype = getDtypeVal("type");
    ndim = getval("dims");
    dt = get_array_dtype(ndim, subtype);
    /* get the pointer to the array bounds descriptor */
    ad = AD_DPTR(dt);
    AD_NUMDIM(ad) = ndim;
    AD_SCHECK(ad) = 0;
    for (i = 0; i < ndim; ++i) {
      getpair(&lower, &upper);
      AD_LWBD(ad, i) = lower; /* to be fixed after symbols added */
      AD_UPBD(ad, i) = upper; /* to be fixed after symbols added */
      AD_MLPYR(ad, i) = getSptrVal("mpy");
    }
    AD_ZBASE(ad) = getval("zbase");
    AD_NUMELM(ad) = getSptrVal("numelm");
    datatypexref[dtype] = dt;
    break;
  case TY_PTR:
    subtype = getDtypeVal("ptrto");
    if (subtype == DT_ANY) {
      datatypexref[dtype] = DT_ADDR;
    } else {
      datatypexref[dtype] = get_type(2, dval, subtype);
    }
    break;
  case TY_PROC:
    subtype = getDtypeVal("result");
    iface = getSptrVal("iface");
    paramct = getval("paramct");
    dpdsc = getval("dpdsc");
    fval = getSptrVal("fval");
    dt = get_type(6, dval, subtype);
    datatypexref[dtype] = dt;
    DTySetProcTy(dt, subtype, iface, paramct, dpdsc, fval);
    break;
  }
} /* read_datatype */

static void
fix_datatype(void)
{
  int d;
  DTYPE dtype;
  int ndim, i;
  SPTR lower, upper;
  int member;
  SPTR mlpyr;
  int zbase;
  SPTR numelm;
  DTYPE subtype;
  SPTR tag;
  ADSC *ad;
  SPTR iface;
  int dpdsc;
  SPTR fval;

  for (d = 0; d <= datatypecount; ++d) {
    dtype = datatypexref[d];
    if (dtype > olddatatypecount) {
      switch (DTY(dtype)) {
      case TY_STRUCT:
      case TY_UNION:
        member = DTyAlgTyMember(dtype);
        member = symbolxref[member];
        DTySetFst(dtype, member);
        tag = DTyAlgTyTag(dtype);
        if (tag) {
          tag = symbolxref[tag];
          DTySetAlgTyTag(dtype, tag);
        }
        if (PARENTG(tag)) {
          /* fix up "parent member" */
          SPTR ptag;
          DTYPE pdtype;
          int pmem;
          PARENTP(member, member);
          pdtype = DTYPEG(member);
          ptag = DTyAlgTyTag(pdtype);
          if (ptag > oldsymbolcount) {
            DTySetAlgTyTag(pdtype, ptag);
          }
          pmem = DTyAlgTyMember(pdtype);
          if (pmem > oldsymbolcount) {
            DTySetFst(pdtype, pmem);
          }
        } else {
          PARENTP(member, 0);
        }
        break;
      case TY_ARRAY:
        subtype = DTySeqTyElement(dtype);
        subtype = datatypexref[subtype];
        if (subtype == 0) {
          fprintf(stderr, "ILM file: missing subtype for array datatype %d\n",
                  d);
          ++errors;
        }
        DTySetFst(dtype, subtype);
        ad = AD_DPTR(dtype);
        ndim = AD_NUMDIM(ad);
        for (i = 0; i < ndim; ++i) {
          lower = AD_LWBD(ad, i);
          lower = symbolxref[lower];
          AD_LWBD(ad, i) = lower;
          upper = AD_UPBD(ad, i);
          if (upper > 0) {
            upper = symbolxref[upper];
            AD_UPBD(ad, i) = upper;
          }
          mlpyr = AD_MLPYR(ad, i);
          if (mlpyr > 0) {
            mlpyr = symbolxref[mlpyr];
            AD_MLPYR(ad, i) = mlpyr;
          }
        }
        zbase = AD_ZBASE(ad);
        if (zbase > 0) {
          zbase = symbolxref[zbase];
          AD_ZBASE(ad) = zbase;
        }
        numelm = AD_NUMELM(ad);
        if (numelm > 0) {
          numelm = symbolxref[numelm];
          AD_NUMELM(ad) = numelm;
        }
        break;
      case TY_PTR:
        subtype = DTySeqTyElement(dtype);
        subtype = datatypexref[subtype];
        if (subtype == 0) {
          fprintf(stderr, "ILM file: missing subtype for pointer datatype %d\n",
                  d);
          ++errors;
        }
        DTySetFst(dtype, subtype);
        break;
      case TY_PROC:
        subtype = DTyReturnType(dtype);
        subtype = datatypexref[subtype];
        /* NOTE: subtype  may be 0, i.e. DT_NONE */
        DTySetFst(dtype, subtype);
        iface = DTyInterface(dtype);
        if (iface) {
          iface = symbolxref[iface];
        }
        DTySetInterface(dtype, iface);
        dpdsc = DTyParamDesc(dtype);
        if (dpdsc && iface) {
          dpdsc = DPDSCG(iface);
        }
        DTySetParamDesc(dtype, dpdsc);
        fval = DTyFuncVal(dtype);
        if (fval) {
          fval = symbolxref[fval];
        }
        DTySetFuncVal(dtype, fval);
        break;
      default:
        break;
      }
    }
  }
} /* fix_datatype */

static SPTR
newsymbol(void)
{
  SPTR sptr;
  int hashid;
  int namelen = getnamelen();
  char *ch = line + pos;
  HASH_ID(hashid, ch, namelen);
  ADDSYM(sptr, hashid);
  NMPTRP(sptr, putsname(line + pos, namelen));
  SYMLKP(sptr, NOSYM);
  return sptr;
} /* newsymbol */

static int
newintrinsic(int wantstype)
{
  int namelen, sptr, hashid, first;
  char *name;
  namelen = getnamelen();
  name = line + pos;
  name[namelen] = '\0';
  HASH_ID(hashid, name, namelen);
  first = stb.hashtb[hashid];
  for (sptr = first; sptr; sptr = HASHLKG(sptr)) {
    if (strcmp(SYMNAME(sptr), name) == 0) {
      switch (STYPEG(sptr)) {
      case ST_PD:
      case ST_INTRIN:
      case ST_GENERIC:
        return sptr;
      default:
        break;
      }
    }
  }
  fprintf(stderr, "ILM file: can't find intrinsic %s\n", name);
  ++errors;
  return 0;
} /* newintrinsic */

static char
gethexchar(FILE *file)
{
  char c1, c2, val;
  c1 = getc(file);
  c2 = getc(file);
  if (c1 >= '0' && c1 <= '9') {
    c1 = c1 - '0';
  } else if (c1 >= 'a' && c1 <= 'f') {
    c1 = c1 - 'a' + 10;
  } else if (c1 >= 'A' && c1 <= 'F') {
    c1 = c1 - 'A' + 10;
  } else {
    c1 = '\0';
  }
  if (c2 >= '0' && c2 <= '9') {
    c2 = c2 - '0';
  } else if (c2 >= 'a' && c2 <= 'f') {
    c2 = c2 - 'a' + 10;
  } else if (c2 >= 'A' && c2 <= 'F') {
    c2 = c2 - 'A' + 10;
  } else {
    c2 = '\0';
  }
  val = c1 << 4 | c2;
  return val;
} /* gethexchar */

#if defined(TARGET_WIN) && defined(PGFTN)
/*
 * convert to upper case
 */
static void
upcase_name(char *name)
{
  char *p;
  int ch;
  for (p = name; ch = *p; ++p)
    if (ch >= 'a' && ch <= 'z')
      *p = ch + ('A' - 'a');
}
#endif

/* Get symbol for sptr from symbolxref or create a new one and add it. */
static SPTR
get_or_create_symbol(SPTR sptr)
{
  SPTR newsptr;
  if (symbolxref[sptr])
    return symbolxref[sptr];
  newsptr = newsymbol();
  symbolxref[sptr] = newsptr;
  return newsptr;
}

static void
read_symbol(void)
{
  SPTR newsptr;
  SYMTYPE stype;
  SC_KIND sclass;
  DTYPE dtype;
  int val[4], namelen, i, dpdsc, inmod;
  /* flags: */
  int addrtkn, adjustable, afterentry, altname, altreturn, aret, argument,
      assigned, assumedrank, assumedshape, assumedsize, autoarray, Cfunc,
      ccsym, clen, cmode, common, constant, count, currsub, decl;
  SPTR descriptor = SPTR_NULL;
  int intentin, texture, device, dll, dllexportmod, enclfunc, end, endlab,
    format, func, gsame, gdesc, hccsym, hollerith, init, isdesc, linenum;
  SPTR link;
  int managed,
      member, midnum, mscall, namelist, needmod, nml, noconflict, passbyval,
      passbyref, cstructret, optional, origdim, origdum, paramcount, pinned,
      plist, pointer, Private, ptrsafe, pure, pdaln, recursive, ref, refs,
      returnval, routx = 0, save, sdscs1, sdsccontig, contigattr, sdscsafe, seq,
                 shared, startlab, startline, stdcall, decorate, cref,
                 nomixedstrlen, target, param, thread, task, tqaln, typed,
    uplevel, vararg, Volatile, fromMod, modcmn, elemental;
  SPTR parent;
  int internref, Class, denorm, Scope, restricted, vtable, iface, vtoff, tbplnk,
      invobj, invobjinc, reref, libm, libc, tls, etls;
  int reflected, mirrored, create, copyin, resident, acclink, devicecopy,
      devicesd, devcopy;
  int unlpoly = 0, allocattr, f90pointer, final, finalized, kindparm;
  int lenparm, isoctype = 0;
  int inmodproc, cudamodule, datacnst, fwdref;
  int agoto, parref, parsyms, parsymsct, paruplevel, is_interface;
  int typedef_init;
  int alldefaultinit = 0;
  int tpalloc, procdesc, has_opts;
  SPTR assocptr, ptrtarget;
  int prociface;
  ISZ_T address, size;
  SPTR sptr = getSptrVal("symbol");
  bool has_alias = false;
  char *alias_name;
  int palign;
#if DEBUG
  if (sptr > symbolcount) {
    fprintf(stderr, "Symbol count was %d, but new symbol number is %d\n",
            symbolcount, sptr);
    exit(1);
  }
#endif
  stype = getSymType();
  sclass = getSCKind();
  dtype = getDtypeVal("dtype");
  palign = getval("palign");
#if DEBUG
  if (dtype > datatypecount) {
    fprintf(stderr, "Datatype count was %d, but new datatype is %d\n",
            datatypecount, dtype);
    interr("upper() FAIL", 0, ERR_Fatal);
  }
#endif
  if (dtype > 0) {
    dtype = datatypexref[dtype]; /* fix data type */
    if (dtype == 0) {
      fprintf(stderr, "ILM file line %d: missing data type for symbol %d\n",
              ilmlinenum, sptr);
      ++errors;
    }
  }
  newsptr = SPTR_NULL;
  passbyval = 0;
  passbyref = 0;
  cstructret = 0;
  switch (stype) {

  case ST_ARRAY:
  case ST_STRUCT:
  case ST_UNION:
  case ST_VAR:
    addrtkn = getbit("addrtaken");
    argument = getbit("argument"); /* + */
    assigned = getbit("assigned");
    decl = getbit("decl");
    dll = getval("dll");
    mscall = getbit("mscall");
    cref = getbit("cref");
    ccsym = getbit("ccsym");
    hccsym = getbit("hccsym");
    init = getbit("init");
    datacnst = getbit("datacnst");
    namelist = getbit("namelist"); /* + */
    optional = getbit("optional"); /* + */
    pointer = getbit("pointer");   /* + */
    Private = getbit("private");   /* + */
    pdaln = getval("pdaln");       /* + */
    tqaln = getbit("tqaln");       /* + */
    ref = getbit("ref");
    save = getbit("save");
    seq = getbit("seq");       /* + */
    target = getbit("target"); /* + */
    param = getbit("param");
    uplevel = getbit("uplevel");
    internref = getbit("internref");
    ptrsafe = getbit("ptrsafe");
    thread = getbit("thread");
    etls = getval("etls");
    tls = getbit("tls");
    task = getbit("task");
    Volatile = getbit("volatile");
    address = getval("address");
    clen = getval("clen");
    common = getval("common");
    link = getSptrVal("link");
    midnum = getval("midnum");
    if (flg.debug && gbl.rutype != RU_BDATA &&
        stype == ST_VAR && sclass == SC_CMBLK) {
      /* Retrieve debug info for renaming and restricted importing
       * of module variables */
      has_alias = getbit("has_alias");
      if (has_alias) {
        const int namelen = getnamelen();
        NEW(alias_name, char, namelen + 1);
        strncpy(alias_name, line + pos, namelen);
        alias_name[namelen] = '\0';
        pos += namelen;
      }
    }
    if (sclass == SC_DUMMY) {
      origdum = getval("origdummy");
    }
    origdim = 0;
    if (stype == ST_ARRAY) {
      adjustable = getbit("adjustable");
      afterentry = getbit("afterentry");
      assumedrank = getbit("assumedrank");
      assumedshape = getbit("assumedshape"); /* + */
      assumedsize = getbit("assumedsize");
      autoarray = getbit("autoarray");
      noconflict = getbit("noconflict");
      sdscs1 = getbit("s1");
      isdesc = getbit("isdesc");
      sdsccontig = getbit("contig");
      origdim = getval("origdim");
      descriptor = getSptrVal("descriptor");
    }
    parref = getbit("parref");
    enclfunc = getval("enclfunc");
    if (passbyflags) {
      passbyval = getbit("passbyval");
      passbyref = getbit("passbyref");
    }
    if (cfuncflags) {
      Cfunc = getbit("Cfunc");
      altname = getval("altname");
    }
    contigattr = getbit("contigattr");
    if (cudaflags) {
      device = getbit("device");
      pinned = getbit("pinned");
      shared = getbit("shared");
      constant = getbit("constant");
      texture = getbit("texture");
      managed = getbit("managed");
    }
    intentin = getbit("intentin");

    Class = getbit("class");
    parent = getSptrVal("parent");

    if (stype == ST_VAR) { /* TBD - for polymorphic variable */
      descriptor = getSptrVal("descriptor");
    }

    reref = getbit("reref");

    reflected = getbit("reflected");
    mirrored = getbit("mirrored");
    create = getbit("create");
    copyin = getbit("copyin");
    resident = getbit("resident");
    acclink = getbit("link");
    devicecopy = getbit("devicecopy");
    devicesd = getbit("devicesd");
    devcopy = getval("devcopy");

    allocattr = getbit("allocattr");
    f90pointer = getbit("f90pointer"); /* will denote the POINTER attribute */
                                       /* but need to remove FE legacy use */
    procdesc = getbit("procdescr");
    newsptr = get_or_create_symbol(sptr);
    if (Class) {
      CLASSP(newsptr, Class);
    }

    if (target) {
      TARGETP(newsptr, 1);
    }

    if (reref) {
      REREFP(newsptr, 1);
    }

    if (stype == ST_VAR) { /* TBD - for polymorphic variable */
      SDSCP(newsptr, descriptor);
      VARDSCP(newsptr, 1);
    }

    if (stype == ST_VAR && DTY(dtype) == TY_STRUCT) {
      STYPEP(newsptr, ST_STRUCT);
    } else {
      STYPEP(newsptr, stype);
    }
    if (Class && stype == ST_ARRAY && isdesc) {
      /* put the type that this type descriptor is associated with
       * in subtype field. (polymoprhic variable) */
      DTYPE dt;
      ADSC *ad;
      DTySetFst(dtype, parent);

      dt = get_array_dtype(1, datatypexref[parent]);
      /* get the pointer to the array bounds descriptor */
      ad = AD_DPTR(dt);
      AD_NUMDIM(ad) = 1;
      AD_SDSC(ad) = SPTR_NULL;
    }
    SCP(newsptr, sclass);
    DTYPEP(newsptr, dtype);
    DCLDP(newsptr, decl);
#if defined(TARGET_WIN_X86)
    if (dll)
      DLLP(newsptr, dll);
#endif
    DINITP(newsptr, init);
    DATACONSTP(newsptr, datacnst);
    CCSYMP(newsptr, ccsym | hccsym);
    if (sclass == SC_LOCAL) {
      SAVEP(newsptr, save | init);
    } else {
      SAVEP(newsptr, save);
    }
    REFP(newsptr, ref);
    ADDRTKNP(newsptr, addrtkn);
    UPLEVELP(newsptr, uplevel);
    INTERNREFP(newsptr, internref);
    if (internref && STB_UPPER()) {
      add_llvm_uplevel_symbol(sptr);
    }
    PTRSAFEP(newsptr, ptrsafe);
    THREADP(newsptr, thread);
#ifdef TASKG
    TASKP(newsptr, task);
#endif
    VOLP(newsptr, Volatile);
    ASSNP(newsptr, assigned);
#ifdef PDALNP
    if (pdaln > 0)
      PDALNP(newsptr, pdaln);
#endif
#ifdef QALNP
    if (pdaln != PDALN_EXPLICIT_0 && pdaln >= 3)
      QALNP(newsptr, 1);
#endif
    OPTARGP(newsptr, optional);
    POINTERP(newsptr, pointer);
    SYMLKP(newsptr, link);
    SOCPTRP(newsptr, 0);
    ADDRESSP(newsptr, address);
    PARAMP(newsptr, param);
    CONTIGATTRP(newsptr, contigattr);
    if (cfuncflags) {
      CFUNCP(newsptr, Cfunc);
      ALTNAMEP(newsptr, altname);

      if (Cfunc) {
        /* add  C_BIND vars to list of global externs */
        SYMLKP(newsptr, gbl.externs);
        gbl.externs = newsptr;
      }
    }

    if (sclass == SC_CMBLK) {
      if (CFUNCG(newsptr)) {
        /* variables visable from C  */
        SCP(newsptr, SC_EXTERN); /* try this */
      } else {

        MIDNUMP(newsptr, common);
      }
    } else {
      if (CFUNCG(newsptr)) {
        /* variables visable from C  */
        SCP(newsptr, SC_EXTERN); /* try this */
      } else {
        MIDNUMP(newsptr, midnum);
      }
    }
    if (sclass == SC_DUMMY) {
      ORIGDUMMYP(newsptr, origdum);
    }
    ORIGDIMP(newsptr, origdim);
    if (stype == ST_ARRAY) {
      ASSUMRANKP(newsptr, assumedrank);
      ASSUMSHPP(newsptr, assumedshape);
      ASUMSZP(newsptr, assumedsize);
      ADJARRP(newsptr, adjustable);
      AFTENTP(newsptr, afterentry);
      AUTOBJP(newsptr, autoarray);
      DESCARRAYP(newsptr, isdesc);
      IS_PROC_DESCRP(newsptr, procdesc);
      if (isdesc) {
        SDSCS1P(newsptr, sdscs1);
        SDSCCONTIGP(newsptr, sdsccontig);
      }
      SDSCP(newsptr, descriptor);
      /* fill in SDSC field of datatype, if necessary */
      if (descriptor && (pointer || assumedshape) && !XBIT(52, 4)) {
        AD_SDSC(AD_DPTR(dtype)) = descriptor;
      }
    }
    if (clen)
      CLENP(newsptr, clen);
    if (stype == ST_ARRAY && sclass == SC_BASED) {
      /* set the NOCONFLICT bit? */
      if (noconflict) {
        NOCONFLICTP(newsptr, 1);
      }
    }
    if (sclass != SC_BASED && !pointer && !target && !addrtkn) {
      /* set the NOCONFLICT flag, meaning no pointers can conflict with it */
      NOCONFLICTP(newsptr, 1);
    }
    if (SCG(newsptr) == SC_PRIVATE && REFG(newsptr)) {
      /* frontend has allocated this private variable - need to
       * adjust its offset
       */
      fix_private_sym(newsptr);
    }
    if (PARAMG(newsptr) || (DINITG(newsptr) && CCSYMG(newsptr))) {
      init_list_count++;
    }
    PARREFP(newsptr, parref);
    ENCLFUNCP(newsptr, enclfunc);
    if (XBIT(119, 0x2000000) && enclfunc)
      LIBSYMP(newsptr, LIBSYMG(symbolxref[enclfunc]));
    if (passbyflags) {
      PASSBYVALP(newsptr, passbyval);
      PASSBYREFP(newsptr, passbyref);
      if (optional)
        PASSBYVALP(newsptr, 0);
    }
    if (cudaflags) {
      if (constant)
        device = 1;
      DEVICEP(newsptr, device);
      PINNEDP(newsptr, pinned);
      SHAREDP(newsptr, shared);
      CONSTANTP(newsptr, constant);
      TEXTUREP(newsptr, texture);
      MANAGEDP(newsptr, managed);
      ACCCREATEP(newsptr, create);
      ACCCOPYINP(newsptr, copyin);
      ACCRESIDENTP(newsptr, resident);
      ACCLINKP(newsptr, acclink);
    }
    INTENTINP(newsptr, intentin);
    ALLOCATTRP(newsptr, allocattr);
    if (flg.debug && has_alias)
      save_modvar_alias(newsptr, alias_name);
    break;

  case ST_CMBLK:
    altname = getval("altname");
    ccsym = getbit("ccsym");
    Cfunc = getbit("Cfunc");
    dll = getval("dll");
    init = getbit("init");
    member = getval("member");
    mscall = getbit("mscall");
    pdaln = getval("pdaln"); /* + */
    save = getbit("save");
    size = getval("size");
    stdcall = getbit("stdcall");
    thread = getbit("thread");
    etls = getval("etls");
    tls = getbit("tls");
    Volatile = getbit("volatile");
    fromMod = getbit("frommod");
    modcmn = getbit("modcmn");
    Scope = getval("scope");
    restricted = getbit("restricted");
    if (cudaflags) {
      device = getbit("device");
      constant = getbit("constant");
      create = getbit("create");
      copyin = getbit("copyin");
      resident = getbit("resident");
      acclink = getbit("link");
    }

    newsptr = get_or_create_symbol(sptr);

    STYPEP(newsptr, stype);
    SCP(newsptr, sclass);
    DTYPEP(newsptr, dtype);

    ALTNAMEP(newsptr, altname);
    CCSYMP(newsptr, ccsym);
    CFUNCP(newsptr, Cfunc);
#if defined(TARGET_WIN_X86)
    if (dll)
      DLLP(newsptr, dll);
#endif
    DINITP(newsptr, init);
    MSCALLP(newsptr, mscall);
#ifdef PDALNP
    if (pdaln > 0)
      PDALNP(newsptr, pdaln);
#endif
#ifdef QALNP
    if (pdaln != PDALN_EXPLICIT_0 && pdaln >= 3)
      QALNP(newsptr, 1);
#endif
    SAVEP(newsptr, save);
    STDCALLP(newsptr, stdcall);
    THREADP(newsptr, thread);
    VOLP(newsptr, Volatile);
    FROMMODP(newsptr, fromMod);
    MODCMNP(newsptr, modcmn);
    SCOPEP(newsptr, Scope);
    RESTRICTEDP(newsptr, restricted);

    CMEMFP(newsptr, member);
    SIZEP(newsptr, size);
    if (cudaflags) {
      DEVICEP(newsptr, device);
      CONSTANTP(newsptr, constant);
      ACCCREATEP(newsptr, create);
      ACCCOPYINP(newsptr, copyin);
      ACCRESIDENTP(newsptr, resident);
      ACCLINKP(newsptr, acclink);
    }

    SYMLKP(newsptr, gbl.cmblks);
    gbl.cmblks = newsptr;
    if (modcmn && !fromMod) {
      /* Indicate that the compiler-created module common is being
       * defined in this subprogram.
       */
      DEFDP(newsptr, 1);
    }
    break;

  case ST_CONST:
    hollerith = getbit("hollerith");
    switch (DTY(dtype)) {
    case TY_HOLL:             /* symbol table ptr of char constant */
      val[0] = getval("sym"); /* to be fixed */
                              /* always add a new symbol; don't use getcon()
                               * because the symbol pointers have not been resolved yet */
      newsptr = newsymbol();
      CONVAL1P(newsptr, val[0]);
      ACONOFFP(newsptr, 0);
      STYPEP(newsptr, ST_CONST);
      DTYPEP(newsptr, dtype);
      break;
    case TY_DWORD:
    case TY_INT8:
    case TY_LOG8:
    case TY_DBLE:
    case TY_CMPLX:
      val[0] = gethex();
      val[1] = gethex();
      newsptr = getcon(val, dtype);
      break;
    case TY_INT:
    case TY_REAL:
    case TY_WORD:
    case TY_LOG:
      val[0] = 0;
      val[1] = gethex();
      newsptr = getcon(val, dtype);
      break;
    case TY_BINT:
    case TY_SINT:
      val[0] = 0;
      val[1] = gethex();
      dtype = DT_INT;
      newsptr = getcon(val, dtype);
      break;
    case TY_BLOG:
    case TY_SLOG:
      val[0] = 0;
      val[1] = gethex();
      dtype = DT_LOG;
      newsptr = getcon(val, dtype);
      break;
    case TY_DCMPLX:
    case TY_QCMPLX:
      val[0] = getval("sym");
      val[1] = getval("sym");
      /* always add a new symbol; don't use getcon()
       * because the symbol pointers have not been resolved yet */
      newsptr = newsymbol();
      CONVAL1P(newsptr, val[0]);
      CONVAL2P(newsptr, val[1]);
      STYPEP(newsptr, ST_CONST);
      DTYPEP(newsptr, dtype);
      break;
    case TY_QUAD:
      val[0] = gethex();
      val[1] = gethex();
      val[2] = gethex();
      val[3] = gethex();
      newsptr = getcon(val, dtype);
      break;
    case TY_PTR:
      val[0] = getval("sym");
      address = getval("offset");
      /* always add a new symbol; don't use getcon()
       * because the symbol pointers have not been resolved yet */
      newsptr = newsymbol();
      CONVAL1P(newsptr, val[0]);
      ACONOFFP(newsptr, address);
      STYPEP(newsptr, ST_CONST);
      DTYPEP(newsptr, dtype);
      break;
    case TY_CHAR:
    case TY_NCHAR:
      namelen = getnamelen();
      /* read the next 'namelen' characters */
      if (namelen > 0) {
        char dash;
/* get the dash */
        if (STB_UPPER())
          dash = getc(gbl.stbfil);
        else
          dash = getc(gbl.srcfil);
        if (namelen >= linelen) {
          linelen = namelen * 2;
          line = (char*) realloc(line, linelen);
        }
        if (dash == '-') {
          for (i = 0; i <= namelen; ++i) {
            if (STB_UPPER())
              line[i] = getc(gbl.stbfil);
            else
              line[i] = getc(gbl.srcfil);
          }
        } else {
          for (i = 0; i < namelen; ++i) {
            if (STB_UPPER())
              line[i] = gethexchar(gbl.stbfil);
            else
              line[i] = gethexchar(gbl.srcfil);
          }
          if (STB_UPPER())
            line[i] = getc(gbl.stbfil);
          else
            line[i] = getc(gbl.srcfil);
        }
        ++ilmlinenum;
      }
      newsptr = getstring(line, namelen);
      if (hollerith)
        HOLLP(newsptr, 1);
      if (DTY(dtype) == TY_NCHAR) {
        val[0] = newsptr;
        val[1] = val[2] = val[3] = 0;
        newsptr = getcon(val, dtype);
      }
      break;
    default:
      fprintf(stderr,
              "ILM file line %d: unknown constant type %d for old symbol %d\n",
              ilmlinenum, dtype, sptr);
      ++errors;
      break;
    }
    SYMLKP(newsptr, SPTR_NULL);
    symbolxref[sptr] = newsptr;
    break;

  case ST_ENTRY:
    currsub = getbit("currsub");
    adjustable = getbit("adjustable");
    afterentry = getbit("afterentry");
    altname = getval("altname");
    Cfunc = getbit("Cfunc");
    decl = getbit("decl");
    dll = getval("dll");
    cmode = getval("cmode");
    end = getval("end"); /* + */
    inmod = getval("inmodule");
    linenum = getval("line");
    mscall = getbit("mscall");
    pure = getbit("pure");           /* + */
    recursive = getbit("recursive"); /* + */
    elemental = getbit("elemental"); /* + */
    returnval = getval("returnval");
    if (passbyflags) {
      passbyval = getbit("passbyval");
      passbyref = getbit("passbyref");
    }
    stdcall = getbit("stdcall");
    decorate = getbit("decorate");
    cref = getbit("cref");
    nomixedstrlen = getbit("nomixedstrlen");
    cudaemu = getval("cudaemu");
    routx = getval("rout");
    paramcount = getval("paramcount");
    altreturn = getval("altreturn");
    vtoff = getval("vtoff");
    invobj = getval("invobj");
    invobjinc = getbit("invobjinc");
    Class = getbit("class");
    denorm = getbit("denorm");
    aret = getbit("aret");
    vararg = getbit("vararg");
    has_opts = getbit("has_opts");

    if (altreturn) {
      gbl.arets = true;
    }
    if (denorm) {
      gbl.denorm = true;
    }

    if (paramcount == 0) {
      dpdsc = 0;
    } else {
      dpdsc = aux.dpdsc_avl;
      aux.dpdsc_avl += paramcount;
      NEED(aux.dpdsc_avl, aux.dpdsc_base, int, aux.dpdsc_size,
           aux.dpdsc_size + paramcount + 100);

      for (i = 0; i < paramcount; ++i) {
        aux.dpdsc_base[dpdsc + i] = getnum();
      }
    }
    newsptr = get_or_create_symbol(sptr);

    STYPEP(newsptr, stype);
    SCP(newsptr, sclass);
    CFUNCP(newsptr, Cfunc);
    DTYPEP(newsptr, dtype);
    DCLDP(newsptr, decl);
#if defined(TARGET_WIN_X86)
    if (dll)
      DLLP(newsptr, dll);
#endif
    MSCALLP(newsptr, mscall);
    PUREP(newsptr, pure);
    ELEMENTALP(newsptr, elemental);
    RECURP(newsptr, recursive);
    if (passbyflags) {
      PASSBYVALP(newsptr, passbyval);
      PASSBYREFP(newsptr, passbyref);
    }
#ifdef CUDAP
    CUDAP(newsptr, cmode);
#endif
    STDCALLP(newsptr, stdcall);
    DECORATEP(newsptr, decorate);
    CREFP(newsptr, cref);
    NOMIXEDSTRLENP(newsptr, nomixedstrlen);
    COPYPRMSP(newsptr, 0);
    ADJARRP(newsptr, adjustable);
    AFTENTP(newsptr, afterentry);
    ADDRESSP(newsptr, 0);
    ALTNAMEP(newsptr, altname);
    DPDSCP(newsptr, dpdsc);
    PARAMCTP(newsptr, paramcount);
    FUNCLINEP(newsptr, linenum);
    FVALP(newsptr, returnval);
    INMODULEP(newsptr, inmod);
    /* add to list of gbl.entries */
    if (currsub) {
      gbl.currsub = newsptr;
      /* don't add if this is a block data */
      if (gbl.rutype != RU_BDATA) {
        /* add to front of list */
        SYMLKP(newsptr, (SPTR) gbl.entries);
        gbl.entries = newsptr;
      }
      if (recursive)
        flg.recursive = true;
    } else if (gbl.entries <= NOSYM) {
      SYMLKP(newsptr, NOSYM);
      gbl.entries = newsptr;
    } else {
      int s;
      for (s = gbl.entries; SYMLKG(s) > NOSYM; s = SYMLKG(s))
        ;
      SYMLKP(s, newsptr);
      SYMLKP(newsptr, NOSYM);
    }
    VTOFFP(newsptr, vtoff);
    INVOBJP(newsptr, invobj);
    INVOBJINCP(newsptr, invobjinc);
    if (invobj) {
      CLASSP(newsptr, Class);
    }
    HAS_OPT_ARGSP(newsptr, has_opts);
    break;

  case ST_LABEL:
    ccsym = getbit("ccsym");
    assigned = getbit("assigned"); /* + */
    format = getbit("format");
    Volatile = getbit("volatile");
    refs = getval("refs");
    agoto = getval("agoto");

    newsptr = get_or_create_symbol(sptr);

    STYPEP(newsptr, stype);
    SCP(newsptr, sclass);
    DTYPEP(newsptr, dtype);

    CCSYMP(newsptr, ccsym);
    RFCNTP(newsptr, refs);
    ADDRESSP(newsptr, 0);
    FMTPTP(newsptr, format);
    VOLP(newsptr, Volatile);
    if (!gbl.stbfil && agoto) {
      if (agotosz == 0) {
        agotosz = 64;
        NEW(agototab, int, agotosz);
        agotomax = 0;
      }
      NEED(agoto, agototab, int, agotosz, agoto + 32);
      agototab[agoto - 1] = newsptr;
      if (agoto > agotomax)
        agotomax = agoto;
    }
    break;

  case ST_MEMBER:
    ccsym = getbit("ccsym");
    sdscs1 = getbit("s1");
    isdesc = getbit("isdesc");
    sdsccontig = getbit("contig");
    contigattr = getbit("contigattr");
    pointer = getbit("pointer");
    address = getval("address");
    descriptor = getSptrVal("descriptor");
    noconflict = getbit("noconflict");
    link = getSptrVal("link");
    tbplnk = getval("tbplnk");
    vtable = getval("vtable");
    iface = getval("iface");
    Class = getbit("class");
    mscall = getbit("mscall");
    cref = getbit("cref");
    allocattr = getbit("allocattr");
    f90pointer = getbit("f90pointer"); /* will denote the POINTER attribute */
                                       /* but need to remove FE legacy use */
    final = getval("final");
    finalized = getbit("finalized");
    kindparm = getbit("kindparm");
    lenparm = getbit("lenparm");
    tpalloc = getbit("tpalloc");
    assocptr = getSptrVal("assocptr");
    ptrtarget = getSptrVal("ptrtarget");
    prociface = getbit("prociface");
    newsptr = get_or_create_symbol(sptr);

    STYPEP(newsptr, stype);
    SCP(newsptr, sclass);
    DTYPEP(newsptr, dtype);
    SDSCP(newsptr, descriptor);
    /* fill in SDSC field of datatype, if necessary */
    if (descriptor && pointer && !XBIT(52, 4) &&
        ((!Class && !finalized && dtype != DT_DEFERCHAR &&
          dtype != DT_DEFERNCHAR) ||
         DTY(dtype) == TY_ARRAY)) {
      AD_SDSC(AD_DPTR(dtype)) = descriptor;
    }
    /* set the NOCONFLICT bit? */
    if (noconflict) {
      NOCONFLICTP(newsptr, 1);
    }

    CCSYMP(newsptr, ccsym);
    ADDRESSP(newsptr, address);
    SYMLKP(newsptr, link);
    POINTERP(newsptr, pointer);
    DESCARRAYP(newsptr, isdesc);
    if (isdesc) {
      SDSCS1P(newsptr, sdscs1);
      SDSCCONTIGP(newsptr, sdsccontig);
    }
    VARIANTP(newsptr, NOSYM);
    PSMEMP(newsptr, newsptr);
    VTABLEP(newsptr, vtable);
    IFACEP(newsptr, iface);
    TBPLNKP(newsptr, tbplnk);
    CLASSP(newsptr, Class);
    ALLOCATTRP(newsptr, allocattr);
    CONTIGATTRP(newsptr, contigattr);
    FINALP(newsptr, final);
    FINALIZEDP(newsptr, finalized);
    KINDPARMP(newsptr, kindparm);
    LENPARMP(newsptr, lenparm);
    TPALLOCP(newsptr, tpalloc);
    ASSOC_PTRP(newsptr, assocptr);
    if (ptrtarget > NOSYM) {
      PTR_TARGETP(newsptr, ptrtarget);
    }
    if (assocptr > NOSYM || ptrtarget > NOSYM) {
      PTR_INITIALIZERP(newsptr, 1);
    }
    IS_PROC_PTR_IFACEP(newsptr, prociface);
    break;

  case ST_NML:
    linenum = getval("line");
    ref = getbit("ref");
    plist = getval("plist");
    count = getval("count");

    nml = aux.nml_avl;
    aux.nml_avl += count;
    NEED(aux.nml_avl, aux.nml_base, NMLDSC, aux.nml_size,
         aux.nml_size + count + 100);

    for (i = 0; i < count; ++i) {
      NML_SPTR(nml + i) = getnum();
      NML_NEXT(nml + i) = nml + i + 1;
      NML_LINENO(nml + i) = linenum;
    }
    NML_NEXT(nml + count - 1) = 0;

    newsptr = get_or_create_symbol(sptr);

    STYPEP(newsptr, stype);
    SCP(newsptr, sclass);
    DTYPEP(newsptr, dtype);

    REFP(newsptr, ref);
    ADDRESSP(newsptr, plist);
    CMEMFP(newsptr, nml);
    CMEMLP(newsptr, nml + count - 1);

    SYMLKP(newsptr, sem.nml);
    sem.nml = newsptr;
    break;

  case ST_PARAM:
    decl = getbit("decl");       /* + */
    Private = getbit("private"); /* + */
    ref = getbit("ref");
    if (TY_ISWORD(DTY(dtype))) {
      val[0] = getval("val");
    } else {
      val[0] = getval("sym");
    }

    newsptr = get_or_create_symbol(sptr);

    STYPEP(newsptr, stype);
    SCP(newsptr, sclass);
    DTYPEP(newsptr, dtype);

    REFP(newsptr, ref);
    CONVAL1P(newsptr, val[0]);
    break;

  case ST_PLIST:
    ccsym = getbit("ccsym");
    init = getbit("init");
    ref = getbit("ref");
    uplevel = getbit("uplevel");
    internref = getbit("internref");
    parref = getbit("parref");
    count = getval("count");
    etls = getval("etls");
    tls = getbit("tls");

    newsptr = get_or_create_symbol(sptr);

    STYPEP(newsptr, stype);
    SCP(newsptr, sclass);
    DTYPEP(newsptr, dtype);

    DINITP(newsptr, init);
    CCSYMP(newsptr, ccsym);
    REFP(newsptr, ref);
    UPLEVELP(newsptr, uplevel);
    INTERNREFP(newsptr, internref);
    if (internref && STB_UPPER()) {
      add_llvm_uplevel_symbol(sptr);
    }
    PARREFP(newsptr, parref);
    PLLENP(newsptr, count);
    break;

  case ST_PROC:
    altname = getval("altname");
    ccsym = getbit("ccsym");
    decl = getbit("decl");
    dll = getval("dll");
    dllexportmod = getbit("dllexportmod");
    cmode = getval("cmode");
    func = getbit("func");
    inmod = getval("inmodule");
    mscall = getbit("mscall");
    needmod = getbit("needmod");
    pure = getbit("pure");
    ref = getbit("ref");
    if (passbyflags) {
      passbyval = getbit("passbyval");
      passbyref = getbit("passbyref");
    }
    cstructret = getbit("cstructret");
    sdscsafe = getbit("sdscsafe");
    stdcall = getbit("stdcall");
    decorate = getbit("decorate");
    cref = getbit("cref");
    nomixedstrlen = getbit("nomixedstrlen");
    typed = getbit("typed");
    recursive = getbit("recursive");
    returnval = getval("returnval");
    Cfunc = getbit("Cfunc");
    uplevel = getbit("uplevel");
    internref = getbit("internref");
    routx = getval("rout");
    paramcount = getval("paramcount");
    vtoff = getval("vtoff");
    invobj = getval("invobj");
    invobjinc = getbit("invobjinc");
    Class = getbit("class");
    libm = getbit("mlib");
    libc = getbit("clib");
    inmodproc = getbit("inmodproc");
    cudamodule = getbit("cudamodule");
    fwdref = getbit("fwdref");
    aret = getbit("aret");
    vararg = getbit("vararg");
    has_opts = getbit("has_opts");
    parref = getbit("parref");
    is_interface = getbit("is_interface");
    descriptor = (sclass == SC_DUMMY) ? getSptrVal("descriptor") : SPTR_NULL;
    assocptr = getSptrVal("assocptr");
    ptrtarget = getSptrVal("ptrtarget");
    prociface = getbit("prociface");

    if (paramcount == 0) {
      dpdsc = 0;
    } else {
      dpdsc = aux.dpdsc_avl;
      aux.dpdsc_avl += paramcount;
      NEED(aux.dpdsc_avl, aux.dpdsc_base, int, aux.dpdsc_size,
           aux.dpdsc_size + paramcount + 100);

      for (i = 0; i < paramcount; ++i) {
        aux.dpdsc_base[dpdsc + i] = getnum();
      }
    }

    newsptr = get_or_create_symbol(sptr);
    STYPEP(newsptr, stype);
    SCP(newsptr, sclass);
    DTYPEP(newsptr, dtype);

    ALTNAMEP(newsptr, altname);
    CCSYMP(newsptr, ccsym);
    DCLDP(newsptr, decl);
#if defined(TARGET_WIN_X86)
    if (dll)
      DLLP(newsptr, dll);
#endif
#ifdef CUDAP
    CUDAP(newsptr, cmode);
#endif
    FUNCP(newsptr, func);
    INMODULEP(newsptr, inmod);
    MSCALLP(newsptr, mscall);
    NEEDMODP(newsptr, needmod);
    PUREP(newsptr, pure);
    REFP(newsptr, ref);
    REDUCP(newsptr, 0);
    PASSBYVALP(newsptr, passbyval);
    PASSBYREFP(newsptr, passbyref);
    CSTRUCTRETP(newsptr, cstructret);
#ifdef SDSCSAFEP
    SDSCSAFEP(newsptr, sdscsafe);
#endif
    STDCALLP(newsptr, stdcall);
    DECORATEP(newsptr, decorate);
    CREFP(newsptr, cref);
    NOMIXEDSTRLENP(newsptr, nomixedstrlen);
    CFUNCP(newsptr, Cfunc);
    UPLEVELP(newsptr, uplevel);
    INTERNREFP(newsptr, internref);
    DPDSCP(newsptr, dpdsc);
    PARAMCTP(newsptr, paramcount);
    FVALP(newsptr, returnval);
    if (internref && STB_UPPER()) {
      add_llvm_uplevel_symbol(sptr);
    }
    LIBMP(newsptr, libm);
    LIBCP(newsptr, libc);
#ifdef CUDAMODULEP
    CUDAMODULEP(newsptr, cudamodule);
#endif
    FWDREFP(newsptr, fwdref);
    TYPDP(newsptr, needmod && typed);

    if (XBIT(119, 0x2000000)) {
      // Set LIBSYM for -Msecond_underscore processing.
      char *s = SYMNAME(newsptr);
      if (needmod) {
        switch (*s) {
        case 'a':
          if (strncmp(s, "accel_lib", 9) == 0)
            LIBSYMP(newsptr, true);
          break;
        case 'i':
          if (strncmp(s, "ieee_arithmetic", 15) == 0 ||
              strncmp(s, "ieee_exceptions", 15) == 0 ||
              strncmp(s, "ieee_features",   13) == 0 ||
              strncmp(s, "iso_c_binding",   13) == 0 ||
              strncmp(s, "iso_fortran_env", 15) == 0)
            LIBSYMP(newsptr, true);
          break;
        case 'o':
          if (strncmp(s, "omp_lib", 7) == 0)
            LIBSYMP(newsptr, true);
          break;
        case 'p':
          if (strncmp(s, "pgi_acc_common", 14) == 0)
            LIBSYMP(newsptr, true);
          break;
        }
      } else if (inmod) {
        LIBSYMP(newsptr, LIBSYMG(symbolxref[inmod]));
      } else if (strncmp(s, "omp_", 4) == 0) {
        // This code should execute when OpenMP routines are used without
        // 'use omp_lib', and should typically set LIBSYM.
        static const char *omp_name[] = {
          "destroy_lock",             "destroy_nest_lock",
          "get_active_level",         "get_ancestor_thread_num",
          "get_cancellation",         "get_default_device",
          "get_dynamic",              "get_initial_device",
          "get_level",                "get_max_active_levels",
          "get_max_task_priority",    "get_max_threads",
          "get_nested",               "get_num_devices",
          "get_num_places",           "get_num_procs",
          "get_num_teams",            "get_num_threads",
          "get_partition_num_places", "get_partition_place_nums",
          "get_place_num",            "get_place_num_procs",
          "get_place_proc_ids",       "get_proc_bind",
          "get_schedule",             "get_team_num",
          "get_team_size",            "get_thread_limit",
          "get_thread_num",           "get_wtick",
          "get_wtime",                "in_parallel",              
          "init_lock",                "init_nest_lock",
          "init_nest_lock_with_hint", "is_initial_device",
          "set_default_device",       "set_dynamic",
          "set_lock",                 "set_max_active_levels",
          "set_nest_lock",            "set_nested",
          "set_num_threads",          "set_schedule",
          "test_lock",                "test_nest_lock",
          "unset_lock",               "unset_nest_lock",
        };
        int c, l, m, u;
        s += 4;
        for (l=0, u=sizeof(omp_name)/sizeof(char*)-1, m=u/2; l<=u; m=(l+u)/2) {
          c = strcmp(s, omp_name[m]);
          if (c == 0) {
            LIBSYMP(newsptr, true);
            break;
          }
          if (c < 0)
            u = m - 1;
          else
            l = m + 1;
        }
      }
    }

    if (sclass != SC_DUMMY && sptr != gbl.outersub && !Class && !inmodproc) {
      /* add to list of gbl.externs. gbl.externs may contain
       * SC_STATIC routines, e.g., internal procedures.
       * If unified.c creates multiple versions of the internal
       * procedure, it needs to see the internal procedure on
       * the gbl.externs list so that the selection is done in the
       * host. If class is set, then this is an internal ST_PROC
       * used in F2003 type bound procedures. Do not add these to
       * the extern list since they're ultimately not referenced. We
       * also do not add these to the extern list if they're used as
       * a module procedure or part of a generic interface.
       */
      SYMLKP(newsptr, gbl.externs);
      gbl.externs = newsptr;
    }
    VTOFFP(newsptr, vtoff);
    INVOBJP(newsptr, invobj);
    INVOBJINCP(newsptr, invobjinc);
    if (invobj) {
      CLASSP(newsptr, Class);
    }
    VARARGP(newsptr, vararg);
    PARREFP(newsptr, parref);
    IS_INTERFACEP(newsptr, is_interface);
    SDSCP(newsptr, descriptor);
    HAS_OPT_ARGSP(newsptr, has_opts);
    ASSOC_PTRP(newsptr, assocptr);
    if (ptrtarget > NOSYM) {
      PTR_TARGETP(newsptr, ptrtarget);
    }
    if (assocptr > NOSYM || ptrtarget > NOSYM) {
      PTR_INITIALIZERP(newsptr, 1);
    }
    IS_PROC_PTR_IFACEP(newsptr, prociface);
    break;

  case ST_GENERIC:
    gsame = getval("gsame");
    count = getval("count");
    if (count < 0)
      goto Handle_as_Intrinsic;
    if (count == 0) {
      gdesc = 0;
    } else {
      gdesc = aux.symi_avl;
      aux.symi_avl += count;
      NEED(aux.symi_avl, aux.symi_base, SYMI, aux.symi_size,
           aux.symi_size + count + 100);
      for (i = 0; i < count; ++i) {
        SYMI_SPTR(gdesc + i) = getnum();
        SYMI_NEXT(gdesc + i) = gdesc + i + 1;
      }
      SYMI_NEXT(gdesc + count - 1) = 0;
    }
    newsptr = get_or_create_symbol(sptr);

    STYPEP(newsptr, stype);
    SCP(newsptr, sclass);
    DTYPEP(newsptr, dtype);

    if (count >= 0) {
      GSAMEP(newsptr, gsame);
      GNDSCP(newsptr, gdesc);
      GNCNTP(newsptr, count);
    }
    break;

  case ST_PD:
  case ST_INTRIN:
  Handle_as_Intrinsic:
    /* exported as an intrinsic, generic, or predeclared function.
     * actually this symbol should be replaced by the fortran name
     * of a function that does the same work;  the only time the symbol
     * gets used is for certain intrinsic/predeclared calls or when
     * the function appears in a procedure argument list */
    i = newintrinsic(stype);
    if (i) {
      /* get the function name to use */
      if (STYPEG(i) == ST_GENERIC) {
        int gnr = i;
        if (GSAMEG(i) == 0) {
          fprintf(stderr, "ILM file: generic %s not allowed as argument\n",
                  SYMNAME(i));
          ++errors;
        }
        i = GSAMEG(i); /* function to use if same name */
        if (ARGTYPG(i) == DT_INT) {
          if (!flg.i4)
            i = GSINTG(gnr);
          else if (XBIT(124, 0x10))
            i = GINT8G(gnr);
        } else if (XBIT(124, 0x8)) {
          if (ARGTYPG(i) == DT_REAL)
            i = GDBLEG(gnr);
          else if (ARGTYPG(i) == DT_CMPLX)
            i = GDCMPLXG(gnr);
        }
      }
    }
    if (i) {
      int name = PNMPTRG(i);
      int cr_size = 0;
      char *actualname;
      if (name == 0) {
        fprintf(stderr, "ILM file: intrinsic %s not allowed as argument\n",
                SYMNAME(i));
        ++errors;
      } else {
        actualname = local_sname(stb.n_base + name);
#ifdef CREFP
#ifdef TARGET_WIN_X8664
        /* Need to add trailing underscore because can't do it in assem.c */
        if (WINNT_CREF && !WINNT_NOMIXEDSTRLEN) {
          strcat(actualname, "_m");
          cr_size = 2; /* size of "_m" */
        }
        if (WINNT_CREF) {
          strcat(actualname, "_");
          cr_size += 1; /* size of "_" */
        }
#endif
#endif

        newsptr = getsym(actualname, strlen(stb.n_base + name) + cr_size);
        newsptr = declref(newsptr, ST_PROC, 'r');

        symbolxref[sptr] = newsptr;

        DTYPEP(newsptr, INTTYPG(i));
        SCP(newsptr, SC_EXTERN);
        if (XBIT(119, 0x2000000))
          LIBSYMP(newsptr, strncmp(SYMNAME(newsptr), "ftn_", 4) == 0);
        SYMLKP(newsptr, gbl.externs);
        gbl.externs = newsptr;
        if (WINNT_CALL)
          MSCALLP(newsptr, 1);
#ifdef CREFP
        if (WINNT_CREF)
          CCSYMP(newsptr, 1);
#endif
      }
    }
    break;

  case ST_STAG:
  case ST_TYPEDEF:
    if (stype == ST_TYPEDEF) {
      /* ST_TYPEDEF */
      fromMod = getbit("frommod");
      parent = getSptrVal("parent");
      descriptor = getSptrVal("descriptor");
      Class = getbit("class");
      alldefaultinit = getbit("alldefaultinit");
      unlpoly = getbit("unlpoly");
      isoctype = getbit("isoctype");
      typedef_init = getval("typedef_init");
      newsptr = get_or_create_symbol(sptr);
    } else {
      /* ST_STAG */
      fromMod = 0;
      parent = SPTR_NULL;
      Class = 0;
      typedef_init = 0;
      newsptr = get_or_create_symbol(sptr);
    }
    STYPEP(newsptr, stype);
    SCP(newsptr, sclass);
    DTYPEP(newsptr, dtype);
    FROMMODP(newsptr, fromMod);
    PARENTP(newsptr, parent);
    SDSCP(newsptr, descriptor);
    CLASSP(newsptr, Class);
    ALLDEFAULTINITP(newsptr, alldefaultinit);
    UNLPOLYP(newsptr, unlpoly);
    ISOCTYPEP(newsptr, isoctype);
    TYPDEF_INITP(newsptr, typedef_init);
    break;

  case ST_BLOCK:
    enclfunc = getval("enclfunc");
    startline = getval("startline");
    end = getval("end");
    startlab = getval("startlab");
    endlab = getval("endlab");
    paruplevel = getval("paruplevel");
    parent = getSptrVal("parent");
    parsymsct = getval("parsymsct");
    parsyms = 0;
    if (parsymsct || parent) {
      LLUplevel *up;

      parsyms = llmp_get_next_key();
      up = llmp_create_uplevel_bykey(parsyms);
      up->parent = parent;
      for (i = 0; i < parsymsct; ++i) {
	/* todo this should be removed as it's wrong.
	 * Keep it until tested. */
	llmp_add_shared_var(up, getnum());
      }
    }

    newsptr = get_or_create_symbol(sptr);
    STYPEP(newsptr, stype);
    ENCLFUNCP(newsptr, enclfunc);
    STARTLINEP(newsptr, startline);
    ENDLINEP(newsptr, end);
    STARTLABP(newsptr, startlab);
    ENDLABP(newsptr, endlab);
    PARSYMSP(newsptr, parsyms);
    PARSYMSCTP(newsptr, parsymsct);
    PARUPLEVELP(newsptr, paruplevel);

    break;

  case -99: /* MODULE */
    /* import this as a block data symbol */
    break;

  default:
    fprintf(stderr, "ILM file line %d: unknown symbol type\n", ilmlinenum);
    ++errors;
    break;
  }
  if (newsptr != SPTR_NULL) {
    PALIGNP(newsptr, palign);
  }
  Trace((" newsptr = %d", newsptr));
} /* read_symbol */

static void
read_overlap(void)
{
  int sptr, count, i;
  sptr = getval("overlap");
  sptr = symbolxref[sptr];
  count = getval("count");
  SOCPTRP(sptr, soc.avail);
  if (soc.size == 0) { /* allocate it */
    soc.size = 1000;
    if (count >= soc.size)
      soc.size = count + 1000;
    NEW(soc.base, SOC_ITEM, soc.size);
  } else {
    NEED(soc.avail + count, soc.base, SOC_ITEM, soc.size,
         soc.avail + count + 1000);
  }
  for (i = 0; i < count; ++i) {
    int n;
    n = getnum();
    SOC_SPTR(soc.avail) = symbolxref[n];
    SOC_NEXT(soc.avail) = soc.avail + 1;
    ++soc.avail;
  }
  /* unlink the last one */
  SOC_NEXT(soc.avail - 1) = 0;
} /* read_overlap */

static void
read_program(void)
{
  if (!checkname("procedure")) {
    fprintf(stderr,
            "ILM file line %d: expecting value for procedure\n"
            "instead got: %s\n",
            ilmlinenum, line + pos);
    ++errors;
    return;
  }
  gbl.rutype = getRUType();
  gbl.has_program |= (gbl.rutype == RU_PROG);
  if (gbl.rutype == RU_PROG) {
    flg.recursive = false;
  } else if (flg.smp) {
    flg.recursive = true;
  }
} /* read_program */

/* add ipab.info pointer stride info for sptr */
static void
addpstride(int sptr, long stride)
{
  int i, j;
  if (!XBIT(66, 0x1000000))
    return;
  j = newindex(sptr);
  i = newinfo();
  IPNFO_TYPE(i) = INFO_PSTRIDE;
  IPNFO_NEXT(i) = IPNDX_INFO(j);
  IPNFO_PSTRIDE(i) = stride;
  IPNDX_INFO(j) = i;
  Trace(("symbol %d:%s has stride %ld", sptr, SYMNAME(sptr), stride));
} /* addpstride */

/* add ipab.info pointer section stride info for sptr */
static void
addsstride(int sptr, long stride)
{
  int i, j;
  if (!XBIT(66, 0x1000000))
    return;
  j = newindex(sptr);
  i = newinfo();
  IPNFO_TYPE(i) = INFO_SSTRIDE;
  IPNFO_NEXT(i) = IPNDX_INFO(j);
  IPNFO_SSTRIDE(i) = stride;
  IPNDX_INFO(j) = i;
  Trace(("symbol %d:%s has section stride %ld", sptr, SYMNAME(sptr), stride));
} /* addsstride */

static void
addf90target(int sptr, int targettype, int targetid)
{
  int i, j;
  j = newindex(sptr);
  i = newinfo();
  IPNFO_TYPE(i) = targettype;
  IPNFO_NEXT(i) = IPNDX_INFO(j);
  IPNFO_TARGET(i) = targetid;
  IPNDX_INFO(j) = i;
  Trace(("symbol %d:%s has targettype %d id %d", sptr, targettype, targetid));
} /* addf90target */

static void
addsafe(int sptr, int safetype, int val)
{
  int i, j;
  j = newindex(sptr);
  i = newinfo();
  IPNFO_TYPE(i) = safetype;
  IPNFO_VAL(i) = val;
  IPNFO_NEXT(i) = IPNDX_INFO(j);
  IPNDX_INFO(j) = i;
  Trace(("symbol %d:%s has safetype %d", sptr, safetype));
} /* addsafe */

static void
read_ipainfo(void)
{
  int sptr, itype, targettype, targetid, func;
  long stride;
  sptr = getval("info");
  sptr = symbolxref[sptr];
  itype = getIPAType();
  switch (itype) {
  case 1: /* pstride */
    stride = getlval("stride");
    addpstride(sptr, stride);
    break;
  case 2: /* sstride */
    stride = getlval("stride");
    addsstride(sptr, stride);
    break;
  case 3: /* Target, from local analysis */
    targettype = getval("type");
    targetid = getval("id");
    switch (targettype) {
    case 1: /* local dynamic memory */
      addf90target(sptr, INFO_FUNKTARGET, targetid);
      break;
    case 2: /* local dynamic memory */
      addf90target(sptr, INFO_FLDYNTARGET, targetid);
      break;
    case 3: /* global dynamic memory */
      addf90target(sptr, INFO_FGDYNTARGET, targetid);
      break;
    case 4: /* nonlocal symbol */
      addf90target(sptr, INFO_FOTARGET, targetid);
      break;
    case 5: /* precise symbol */
    case 6: /* imprecise symbol */
      if (symbolxref[targetid]) {
        addf90target(sptr, INFO_FSTARGET, symbolxref[targetid]);
      } else {
        addf90target(sptr, INFO_FOSTARGET, symbolxref[targetid]);
      }
      break;
    }
    break;
  case 4: /* Target, from IPA */
    targettype = getval("type");
    targetid = getval("id");
    switch (targettype) {
    case 1: /* local symbol */
      if (symbolxref[targetid]) {
        addf90target(sptr, INFO_LTARGET, symbolxref[targetid]);
      } else {
        addf90target(sptr, INFO_OTARGET, targetid);
      }
      break;
    case 2: /* global symbol */
      if (symbolxref[targetid]) {
        addf90target(sptr, INFO_GTARGET, symbolxref[targetid]);
      } else {
        addf90target(sptr, INFO_OGTARGET, targetid);
      }
      break;
    case 3: /* other data */
      addf90target(sptr, INFO_OTARGET, targetid);
      break;
    case 4: /* anonymous global variable */
      addf90target(sptr, INFO_OGTARGET, targetid);
      break;
    }
    break;
  case 5: /* all call safe, from IPA */
    addsafe(sptr, INFO_ALLCALLSAFE, 0);
    break;
  case 6: /* safe, from IPA */
    addsafe(sptr, INFO_SAFE, 0);
    break;
  case 7: /* callsafe, from IPA */
    func = getval("func");
    if (symbolxref[func]) {
      addsafe(sptr, INFO_CALLSAFE, symbolxref[func]);
    }
    break;
  }
} /* read_ipainfo */

static void
fix_symbol(void)
{
  SPTR sptr;
  int i, fval, smax;
  int altname;
  DTYPE dtype;
  int parsyms, paruplevel;
  int clen, common, dpdsc;
  SPTR desc;
  int enclfunc, inmod, scope;
  SPTR lab, link;
  int midnum, member, nml, paramcount, plist, val, origdum;
  int typedef_init;

  threadprivate_dtype = DT_NONE;
  tpcount = 0;
  if (gbl.statics) {
    /* NOSYM required instead of 0 */
    if (!symbolxref[gbl.statics]) {
      gbl.statics = NOSYM;
    } else {
      gbl.statics = symbolxref[gbl.statics];
    }
  } else {
    gbl.statics = NOSYM;
  }

  if (gbl.locals) {
    /* NOSYM required instead of 0 */
    if (!symbolxref[gbl.locals]) {
      gbl.locals = NOSYM;
    } else {
      gbl.locals = symbolxref[gbl.locals];
    }
  } else {
    gbl.locals = NOSYM;
  }

  if (gbl.outersub) {
    gbl.outersub = symbolxref[gbl.outersub];
  }
  smax = stb.stg_avail;
  for (sptr = (SPTR)(oldsymbolcount + 1); sptr < smax; ++sptr) {
    bool refd_done = false;
    switch (STYPEG(sptr)) {
    case ST_TYPEDEF: /* FS#16646 - fix type descriptor symbol */
      desc = SDSCG(sptr);
      if (desc > NOSYM) {
        desc = symbolxref[desc];
        SDSCP(sptr, desc);
      }
      typedef_init = TYPDEF_INITG(sptr);
      if (typedef_init > NOSYM) {
        typedef_init = symbolxref[typedef_init];
        TYPDEF_INITP(sptr, typedef_init);
      }
      break;
    case ST_ARRAY:
    case ST_STRUCT:
    case ST_UNION:
      dtype = DTYPEG(sptr);
      if (REREFG(sptr)) {
        /* REF bit not set in front end because we need to
         * compute assn_static_off() in the back end's
         * sym_is_refd(). So, we will do it here. This typically
         * occurs with type extensions that have initializations
         * in their parent component.
         */
        REFP(sptr, 0);
        sym_is_refd(sptr);
        refd_done = true; /* don't put on gbl lists again */
      }
      if (DTY(dtype) == TY_ARRAY) {
        desc = SDSCG(sptr);
        if (desc > NOSYM) {
          desc = symbolxref[desc];
          SDSCP(sptr, desc);
        }
        if (desc > NOSYM && AD_SDSC(AD_DPTR(dtype))) {
          AD_SDSC(AD_DPTR(dtype)) = desc;
        }
        if (CLASSG(sptr) && DESCARRAYG(sptr)) {
          /* insert type descriptor in gbl list */
          int sptr2;
          for (sptr2 = gbl.typedescs; sptr2 > NOSYM; sptr2 = TDLNKG(sptr2)) {
            if (sptr2 == sptr)
              break;
          }
          if (sptr2 != sptr) {
            /* unset CC flag so getsname() produces a
             * correct Fortran global symbol with a
             * trailing underscore.
             */
            CCSYMP(sptr, 0);
            TDLNKP(sptr, gbl.typedescs);
            gbl.typedescs = sptr;
          }
        }
      }
      FLANG_FALLTHROUGH;
    case ST_VAR:
      if (STYPEG(sptr) != ST_ARRAY && VARDSCG(sptr)) {
        desc = SDSCG(sptr);
        if (desc > NOSYM) {
          desc = symbolxref[desc];
          SDSCP(sptr, desc);
        }
      }
      link = SYMLKG(sptr);
      if ((link > NOSYM) && !CFUNCG(sptr)) {
        /* CFUNCG : keep BIND(C) variables on the
           gbl.extern list
        */
          SYMLKP(sptr, symbolxref[link]);
      }
      if (SCG(sptr) == SC_CMBLK) {
        common = MIDNUMG(sptr);
        if (CFUNCG(sptr)) {
          /* variables visable from C  */
          SCP(sptr, SC_EXTERN); /* try this */
        } else {
          MIDNUMP(sptr, symbolxref[common]);
        }
      } else if (IS_THREAD_TP(sptr)) {
        if ((SCG(sptr) == SC_LOCAL || SCG(sptr) == SC_STATIC) &&
            !UPLEVELG(sptr) && !MIDNUMG(sptr)) {
          int tptr;
          tptr = create_thread_private_vector(sptr, 0);
          MIDNUMP(tptr, sptr);
          MIDNUMP(sptr, tptr);
          if (!XBIT(69, 0x80))
            SCP(tptr, SC_STATIC);
        } else if (SCG(sptr) == SC_BASED) {
          int psptr;
          psptr = symbolxref[MIDNUMG(sptr)];
          if (SCG(psptr) == SC_CMBLK) {
            /* if the $p var is in a common block, the
             * treadprivate vector will be generated when
             * the $p var is processed
             */
            MIDNUMP(sptr, psptr);
          } else if ((SCG(psptr) == SC_LOCAL || SCG(psptr) == SC_STATIC) &&
                     UPLEVELG(psptr)) {
            /* defer until restore_saved_syminfo() */
            MIDNUMP(sptr, psptr);
          } else if (POINTERG(sptr)) {
            /* Cannot rely on the SYMLK chain appearing as
             *     $p -> $o -> $sd
             * Apparently, these links only occur for the pointer's internal
             * variables if the pointer does not have the SAVE attribute.
             * Without these fields, the correct size of the threads' copies
             * cannot be computed.
             * Just explicitly look for the internal pointer and descriptor.
             * If the descriptor is present, can assume that there is an
             * offest var which only needs to be accounted for in the size
             * computation of the threads' copies.
             * Setup up the MIDNUM fields as follows where foo is the symtab
             * entry which has the POINTER flag set:
             *    foo    -> foo$p
             *    TPpfoo -> foo
             *    foo$p  -> TPpfoo
             *    foo$sd -> TPpfoo
             * Note that foo's SDSC -> foo$sd.
             * Before we had:
             *    foo    -> TPpfoo
             *    TPpfoo -> foo$p
             * which is a problem for computing the size when starting with
             * TPpfoo.
             */
            int tptr;
            int sdsptr;
            tptr = create_thread_private_vector(sptr, 0);
            THREADP(psptr, 1);
            MIDNUMP(sptr, psptr);
            MIDNUMP(tptr, sptr);
            MIDNUMP(psptr, tptr);
            sdsptr = SDSCG(sptr);
            if (sdsptr) {
              THREADP(sdsptr, 1);
              MIDNUMP(sdsptr, tptr);
            }
            if (!XBIT(69, 0x80))
              if (SCG(psptr) == SC_LOCAL || SCG(psptr) == SC_STATIC)
                SCP(tptr, SC_STATIC);
          } else {
            /*
             * Given the above code for POINTER, this code is
             * probably dead, but leave it just in case.
             */
            int tptr;
            tptr = create_thread_private_vector(psptr, 0);
            THREADP(psptr, 1);
            MIDNUMP(sptr, tptr);
            MIDNUMP(tptr, psptr);
            MIDNUMP(psptr, tptr);
            if (SYMLKG(psptr) != NOSYM) {
              psptr = symbolxref[SYMLKG(psptr)];
              THREADP(psptr, 1);
              MIDNUMP(psptr, tptr);
              if (SYMLKG(psptr) != NOSYM) {
                psptr = symbolxref[SYMLKG(psptr)];
                THREADP(psptr, 1);
                MIDNUMP(psptr, tptr);
              }
            }
          }
        }
      } else {
        midnum = MIDNUMG(sptr);
        if (midnum) {
          const int newMid = symbolxref[midnum];
          MIDNUMP(sptr, newMid);
#ifdef REVMIDLNKP
          if (POINTERG(sptr) && newMid) {
            assert(!REVMIDLNKG(newMid), "REVMIDLNK already set", newMid,
                   ERR_Fatal);
            REVMIDLNKP(newMid, sptr);
          }
#endif
          if (ALLOCATTRG(sptr))
            ALLOCATTRP(newMid, 1);
        }
      }
      if (SCG(sptr) == SC_DUMMY) {
        origdum = ORIGDUMMYG(sptr);
        if (origdum) {
          origdum = symbolxref[origdum];
          ORIGDUMMYP(sptr, origdum);
          ORIGDUMMYP(origdum, sptr);
        }
      } else if (SCG(sptr) == SC_STATIC && REFG(sptr) && !refd_done &&
                 !DINITG(sptr)) {
        /* FE90 front end doesn't have a gbl.bssvars */
        SYMLKP(sptr, gbl.bssvars);
        gbl.bssvars = sptr;
      }
      clen = CLENG(sptr);
      if (clen) {
        clen = symbolxref[clen];
        CLENP(sptr, clen);
      }
      if (!XBIT(124, 64) && SCG(sptr) == SC_BASED) {
        /* if the MIDNUM (pointer) is not a TEMP,
         * and we are not using safe 'cray-pointer' semantics,
         * reset NOCONFLICT */
        midnum = MIDNUMG(sptr);
        if (midnum && !CCSYMG(midnum)) {
          NOCONFLICTP(sptr, 0);
        }
      }
      if (SCG(sptr) == SC_BASED && MIDNUMG(sptr) && CCSYMG(MIDNUMG(sptr))) {
        /* nonuser cray pointer, the pointer variable has no conflict */
        NOCONFLICTP(MIDNUMG(sptr), 1);
      }
      if (SCG(sptr) == SC_BASED && !NOCONFLICTG(sptr) && MIDNUMG(sptr) &&
          !CCSYMG(MIDNUMG(sptr))) {
        /* ### for now, reset NOCONFLICT bit on cray pointer */
        /* ### error in f90correct/bq00.f with -Mscalarsse -Mx,72,1 */
        NOCONFLICTP(MIDNUMG(sptr), 0);
      }
      enclfunc = ENCLFUNCG(sptr);
      if (enclfunc) {
        enclfunc = symbolxref[enclfunc];
        ENCLFUNCP(sptr, enclfunc);
      }
      altname = ALTNAMEG(sptr);
      if (altname)
        ALTNAMEP(sptr, symbolxref[altname]);
      break;
    case ST_CMBLK:
      member = CMEMFG(sptr);
      CMEMFP(sptr, symbolxref[member]);
      altname = ALTNAMEG(sptr);
      if (altname)
        ALTNAMEP(sptr, symbolxref[altname]);
      scope = SCOPEG(sptr);
      if (scope) {
        scope = symbolxref[scope];
        SCOPEP(sptr, scope);
      }
      break;
    case ST_CONST:
      switch (DTY(DTYPEG(sptr))) {
      case TY_HOLL:
        val = CONVAL1G(sptr);
        CONVAL1P(sptr, symbolxref[val]);
        break;
      case TY_DCMPLX:
      case TY_QCMPLX:
        val = CONVAL1G(sptr);
        CONVAL1P(sptr, symbolxref[val]);
        val = CONVAL2G(sptr);
        CONVAL2P(sptr, symbolxref[val]);
        break;
      case TY_PTR:
        val = CONVAL1G(sptr);
        CONVAL1P(sptr, symbolxref[val]);
        break;
      default:
        break;
      }
      break;
    case ST_LABEL:
      break;
    case ST_MEMBER:
      link = SYMLKG(sptr);
      if (link > NOSYM) {
        link = symbolxref[link];
        SYMLKP(sptr, link);
        VARIANTP(link, sptr);
        if (ALLOCATTRG(sptr) && ADDRESSG(link) == ADDRESSG(sptr) &&
            DTY(DTYPEG(link)) == TY_PTR)
          ALLOCATTRP(link, 1);
      }
      dtype = DTYPEG(sptr);
      if (DTY(dtype) == TY_ARRAY) {
        desc = SDSCG(sptr);
        if (desc > NOSYM) {
          desc = symbolxref[desc];
          SDSCP(sptr, desc);
        }
        if (desc > NOSYM && AD_SDSC(AD_DPTR(dtype))) {
          AD_SDSC(AD_DPTR(dtype)) = desc;
        } else if (desc <= NOSYM && AD_SDSC(AD_DPTR(dtype)) > oldsymbolcount) {
          desc = AD_SDSC(AD_DPTR(dtype));
          AD_SDSC(AD_DPTR(dtype)) = symbolxref[desc];
        }

      } else if (DTYPEG(sptr) == DT_ASSCHAR) {
        desc = SDSCG(sptr);
        if (desc > NOSYM) {
          desc = symbolxref[desc];
          SDSCP(sptr, desc);
        }
      } else if (DTYPEG(sptr) == DT_DEFERCHAR) {
        desc = SDSCG(sptr);
        if (desc > NOSYM) {
          desc = symbolxref[desc];
          SDSCP(sptr, desc);
        }
      } else if (CLASSG(sptr) || FINALIZEDG(sptr)) {
        desc = SDSCG(sptr);
        if (desc > NOSYM) {
          desc = symbolxref[desc];
          SDSCP(sptr, desc);
        }
      }
      if (SCG(sptr) == SC_BASED && MIDNUMG(sptr) && CCSYMG(MIDNUMG(sptr))) {
        /* nonuser cray pointer, the pointer variable has no conflict */
        NOCONFLICTP(MIDNUMG(sptr), 1);
      }
      if (CLASSG(sptr) || FINALIZEDG(sptr)) {
        /* Fix up type bound procedure links */
        int sym = TBPLNKG(sptr);
        if (sym > oldsymbolcount) {
          sym = symbolxref[sym];
          TBPLNKP(sptr, sym);
        }
        sym = VTABLEG(sptr);
        if (sym > oldsymbolcount) {
          sym = symbolxref[sym];
          VTABLEP(sptr, sym);
        }
        sym = IFACEG(sptr);
        if (sym > oldsymbolcount) {
          sym = symbolxref[sym];
          IFACEP(sptr, sym);
        }
      }
      break;
    case ST_NML:
      plist = ADDRESSG(sptr);
      ADDRESSP(sptr, symbolxref[plist]);
      /* fix namelist members */
      for (nml = CMEMFG(sptr); nml; nml = NML_NEXT(nml)) {
        member = NML_SPTR(nml);
        NML_SPTR(nml) = symbolxref[member];
      }
      break;
    case ST_PARAM:
      if (!TY_ISWORD(DTY(DTYPEG(sptr)))) {
        /* fix up sptr */
        val = CONVAL1G(sptr);
        CONVAL1P(sptr, symbolxref[val]);
      }
      break;
    case ST_PLIST:
      if (!UPLEVELG(sptr))
        sym_is_refd(sptr);
      break;
    case ST_PROC:
      if (PTR_INITIALIZERG(sptr) && ASSOC_PTRG(sptr) && PTR_TARGETG(sptr)) {
        SPTR ptr = symbolxref[ASSOC_PTRG(sptr)];
        ASSOC_PTRP(sptr, MIDNUMG(ptr) > NOSYM ? MIDNUMG(ptr) : ptr);
        PTR_TARGETP(sptr, symbolxref[PTR_TARGETG(sptr)]);
        ADDRESSP(sptr, ADDRESSG(ptr));
        break;
      }
      FLANG_FALLTHROUGH;
    case ST_ENTRY:
      paramcount = PARAMCTG(sptr);
      dpdsc = DPDSCG(sptr);
      for (i = 0; i < paramcount; ++i) {
        int param;
        param = aux.dpdsc_base[dpdsc + i];
        param = symbolxref[param];
        aux.dpdsc_base[dpdsc + i] = param;
      }
      fval = FVALG(sptr);
      if (fval) {
        fval = symbolxref[fval];
        FVALP(sptr, fval);
      }
      inmod = INMODULEG(sptr);
      if (inmod) {
        inmod = symbolxref[inmod];
        INMODULEP(sptr, inmod);
      }
      altname = ALTNAMEG(sptr);
      if (altname)
        ALTNAMEP(sptr, symbolxref[altname]);
      if (STYPEG(sptr) == ST_PROC && SDSCG(sptr)) {
        SDSCP(sptr, symbolxref[SDSCG(sptr)]);
      }
      break;
    case ST_GENERIC:
      for (desc = (SPTR)GNDSCG(sptr); desc; desc = (SPTR)SYMI_NEXT(desc)) {
        int spec;
        spec = SYMI_SPTR(desc);
        spec = symbolxref[spec];
        SYMI_SPTR(desc) = spec;
      }
      break;
    case ST_BLOCK:
      enclfunc = ENCLFUNCG(sptr);
      if (enclfunc) {
        enclfunc = symbolxref[enclfunc];
        ENCLFUNCP(sptr, enclfunc);
      }
      lab = STARTLABG(sptr);
      STARTLABP(sptr, symbolxref[lab]);
      lab = ENDLABG(sptr);
      ENDLABP(sptr, symbolxref[lab]);
      paruplevel = PARUPLEVELG(sptr);
      if (paruplevel) {
        paruplevel = symbolxref[paruplevel];
        PARUPLEVELP(sptr, paruplevel);
      }
      if (PARSYMSG(sptr) || llmp_has_uplevel(sptr)) {
        LLUplevel *up = llmp_get_uplevel(sptr);
        for (i = 0; i < up->vals_count; ++i) {
          int parsptr = up->vals[i];
          parsptr = symbolxref[parsptr];
          up->vals[i] = parsptr;
        }
        if (up->parent) {
          up->parent = symbolxref[up->parent];
          if (llmp_has_uplevel(up->parent) == 0) {
            parsyms = llmp_get_next_key();
            PARSYMSP(up->parent, parsyms);
            up = llmp_create_uplevel_bykey(parsyms);
          }
        }
      }
      break;
    default:
      break;
    }
  }
  for (common = gbl.cmblks; common > NOSYM; common = SYMLKG(common)) {
#if defined(TARGET_WIN_X86)
    int cmem;
    for (cmem = CMEMFG(common); cmem > NOSYM; cmem = SYMLKG(cmem)) {
      if ((DLLG(cmem) == DLL_EXPORT) && (DLLG(common) != DLL_EXPORT)) {
        DLLP(common, DLL_EXPORT);
      }
    }
#endif
    if (common > oldsymbolcount) {
      member = CMEMFG(common);
      for (; SYMLKG(member) > NOSYM; member = SYMLKG(member))
        ;
      CMEMLP(common, member);
      if (IS_THREAD_TP(common)) {
        int tptr;
        /* mark all members as thread-private */
        for (member = CMEMFG(common); member > NOSYM; member = SYMLKG(member)) {
          THREADP(member, 1);
        }

        tptr = create_thread_private_vector(common, 0);
        /* Link the common block and its vector */
        MIDNUMP(tptr, common);
        MIDNUMP(common, tptr);
      }
#if defined(TARGET_WIN_X86)
      else if (DLLG(common) == DLL_EXPORT) {
        /* mark all members as dllexport */
        for (member = CMEMFG(common); member > NOSYM; member = SYMLKG(member))
          DLLP(member, DLL_EXPORT);
      } else if (DLLG(common) == DLL_IMPORT) {
        /* mark all members as dllimport */
        for (member = CMEMFG(common); member > NOSYM; member = SYMLKG(member))
          DLLP(member, DLL_IMPORT);
      }
#endif
    }
  }
} /* fix_symbol */

static int
create_thread_private_vector(int sptr, int host_tpsym)
{
  char TPname[MAXIDLEN + 5];
  char *np;
  int len, hashid;
  SPTR tptr;

  if (threadprivate_dtype == 0) {
    threadprivate_dtype = create_threadprivate_dtype();
  }
  TPname[0] = 'T';
  TPname[1] = 'P';
  TPname[2] = 'p';
  np = SYMNAME(sptr);
  len = strlen(np);
  if (len > MAXIDLEN)
    len = MAXIDLEN;
  strncpy(TPname + 3, np, len);
  HASH_ID(hashid, TPname, len + 3);
  ADDSYM(tptr, hashid);
  NMPTRP(tptr, putsname(TPname, len + 3));
  STYPEP(tptr, ST_VAR);
  SCP(tptr, SC_EXTERN);
  DTYPEP(tptr, threadprivate_dtype);
  DCLDP(tptr, 1);

  if (host_tpsym) {
    /*
     * If the threadprivate variable/common were declared in the host,
     * need to use its threadprivate vector which is also declared in
     * the host along with its host attributes.  Also, in this case,
     * avoid adding the vector to the gbl.threadprivate list; doing so
     * yields multiple declaratations via _mp_cdecl[p].
     */
    int s;
    for (s = 0; s < saved_tpcount; s++) {
      if (host_tpsym == saved_tpinfo[s].memarg) {
        SCP(tptr, saved_tpinfo[s].sc);
        ADDRESSP(tptr, saved_tpinfo[s].address);
        REFP(tptr, saved_tpinfo[s].ref);
        if (STYPEG(sptr) != ST_CMBLK)
          UPLEVELP(tptr, 1);
        return tptr;
      }
    }
  }

  /* Add the vector to the gbl.threadprivate list */
  TPLNKP(tptr, gbl.threadprivate);
  gbl.threadprivate = tptr;
  tpcount++;

  return tptr;
}

/* create the datatype for the vector of pointers,
 * this code copied from 'semant.c' for the pgf77
 */
static DTYPE
create_threadprivate_dtype(void)
{
  DTYPE dt;
  SPTR zero, one, maxcpu, maxcpup1;
  int val[4];
  ADSC *ad;
  return DT_ADDR;

  val[0] = 0;
  val[1] = 0;
  zero = getcon(val, DESC_ELM_DT);
  val[1] = 1;
  one = getcon(val, DESC_ELM_DT);
  val[1] = MAXCPUS - 1;
  maxcpu = getcon(val, DESC_ELM_DT);
  val[1] = MAXCPUS;
  maxcpup1 = getcon(val, DESC_ELM_DT);
  dt = get_array_dtype(1, __POINT_T);
  ad = AD_DPTR(dt);
  AD_NUMDIM(ad) = 1;
  AD_SCHECK(ad) = 0;
  AD_LWBD(ad, 0) = zero;
  AD_UPBD(ad, 0) = maxcpu;
  AD_MLPYR(ad, 0) = one;
  AD_ZBASE(ad) = zero;
  AD_NUMELM(ad) = maxcpup1;
  return dt;
}

#include "upperilm.h"

static int
getilm(void)
{
  int val;

  if (endilmfile) {
    fprintf(stderr, "ILM file: looking past end-of-file for ilm number\n");
    ++errors;
    return 0;
  }

  if (line[pos] != 'i') {
    fprintf(stderr,
            "ILM file line %d: expecting ilm number\n"
            "instead got: %s\n",
            ilmlinenum, line + pos);
    ++errors;
    return 0;
  }

  ++pos;
  val = 0;
  while (line[pos] >= '0' && line[pos] <= '9') {
    val = val * 10 + (line[pos] - '0');
    ++pos;
  }
  return val;
} /* getilm */

static int
getoperand(const char *optype, char letter)
{
  int val, neg;

  if (endilmfile) {
    fprintf(stderr, "ILM file: looking past end-of-file for %s operand\n",
            optype);
    ++errors;
    return 0;
  }

  skipwhitespace();

  if (line[pos] != letter) {
    fprintf(stderr,
            "ILM file line %d: expecting %s operand\n"
            "instead got: %s\n",
            ilmlinenum, optype, line + pos);
    ++errors;
    return 0;
  }

  ++pos;
  val = 0;
  neg = 1;
  if (line[pos] == '-') {
    ++pos;
    neg = -1;
  }
  while (line[pos] >= '0' && line[pos] <= '9') {
    val = val * 10 + (line[pos] - '0');
    ++pos;
  }
  val *= neg;
  switch (letter) {
  case chsym:
    if (val == 0)
      return 0;
    if (symbolxref[val] != 0) {
      return symbolxref[val];
    }
    break;
  case chdtype:
    if (datatypexref[val] != 0) {
      return datatypexref[val];
    }
    if (val == 0) {
      return 0;
    }
    break;
  case chilm:
    if (val <= 0 || val >= origilmavl) {
      fprintf(stderr, "ILM FILE line %d: Bad ilm operand %d\n", ilmlinenum,
              val);
      ++errors;
    } else if (ilmxref[val] == 0) {
      fprintf(stderr, "ILM FILE line %d: Invalid ilm operand %d\n", ilmlinenum,
              val);
      ++errors;
    } else {
      val = ilmxref[val];
    }
    return val;
  case chline:
  case chnum:
    return val;
  default:
    break;
  }
  fprintf(stderr, "ILM file line %d: unknown %s operand %d\n", ilmlinenum,
          optype, val);
  ++errors;
  return 0;
} /* getoperand */

static int
getoperation(void)
{
  char ch;
  char *p;
  int len;
  int hi, lo;

  if (endilmfile) {
    fprintf(stderr, "ILM file: looking past end-of-file for operation\n");
    ++errors;
    return 0;
  }

  skipwhitespace();

  /* end of statement? */
  p = line + pos;

  if (strncmp(p, "---", 3) == 0) {
    /* yes, simply return */
    return -1;
  }

  /* check for unimplemented operation */
  if (strncmp(p, "--", 2) == 0) {
    /* yes, simply return */
    return -2;
  }

  ch = line[pos];
  len = 0;
  while ((ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') ||
         (ch >= '0' && ch <= '9') || (ch == '_')) {
    ++pos;
    ++len;
    ch = line[pos];
  }
  line[pos] = '\0';
  /* binary search */
  hi = NUMOPERATIONS - 1;
  lo = 0;
  while (lo <= hi) {
    int mid, compare;
    mid = (hi + lo) / 2;
    compare = strcmp(p, info[mid].name);
    if (compare == 0) {
      line[pos] = ch;
      return mid;
    }
    if (compare < 0) {
      hi = mid - 1;
    } else {
      lo = mid + 1;
    }
  }
  line[pos] = ch;
  fprintf(stderr, "ILM file line %d: unknown operation: %s\n", ilmlinenum, p);
  ++errors;
  return -5;
} /* getoperation */

/* read one line from the ILM file */
static void
read_ilm(void)
{
  int ilm, op, numoperands, i, opc;
  ilm = getilm();
  if (line[pos] == ':') {
    ++pos;
  }

  op = getoperation();
  numoperands = 0;

  if (op >= 0 && info[op].ilmtype == IM_BOS) {
    /* first argument is the line number */
    gbl.lineno = getoperand("line", chline);
    gbl.findex = getoperand("number", chnum);
    Trace(("Statement at line %d", gbl.lineno));
    origilmavl = 4;
  } else if (op >= 0) {
    opc = info[op].ilmtype;
    Trace(("read i%d: %s (%d) with %d operands", ilm, info[op].name, op,
           info[op].numoperands));
    if (ilm != origilmavl) {
      fprintf(stderr, "ILM FILE line %d: Reading ilm %d into slot %d\n",
              ilmlinenum, ilm, origilmavl);
      ++errors;
    }
    if (opc == IM_AGOTO) {
      gbl.asgnlbls = NME_NULL;
    }

    numoperands = info[op].numoperands;
    /* this is where the next ILM should appear: */
    ++origilmavl;
    NEED(ilm + 1, ilmxref, int, ilmxrefsize, ilm + 100);
    ilmxref[ilm] = ilmb.ilmavl;
    ad1ilm(opc);
    for (i = 0; i < numoperands; ++i) {
      int opnd;
      switch (info[op].operand[i]) {
      case pilm:
        ++origilmavl;
        opnd = getoperand("ilm", chilm);
        Trace((" %c%d", chilm, opnd));
        ad1ilm(opnd);
        break;
      case psym:
        ++origilmavl;
        opnd = getoperand("symbol", chsym);
        Trace((" %c%d", chsym, opnd));
        ad1ilm(opnd);
        if (opc == IM_LABEL)
          DEFDP(opnd, 1);
        break;
      case pdtype:
        ++origilmavl;
        opnd = getoperand("datatype", chdtype);
        Trace((" %c%d", chdtype, opnd));
        ad1ilm(opnd);
        break;
      case pline:
        ++origilmavl;
        opnd = getoperand("line", chline);
        Trace((" %c%d", chline, opnd));
        ad1ilm(opnd);
        break;
      case pnum:
        ++origilmavl;
        opnd = getoperand("number", chnum);
        Trace((" %c%d", chnum, opnd));
        ad1ilm(opnd);
        break;
      case pilms:
        skipwhitespace();
        while (line[pos] == chilm) {
          ++origilmavl;
          opnd = getoperand("ilm", chilm);
          Trace((" %c%d", chilm, opnd));
          ad1ilm(opnd);
          skipwhitespace();
        }
        break;
      case pargs:
        skipwhitespace();
        while (line[pos] == chilm) {
          ++origilmavl;
          opnd = getoperand("ilm", chilm);
          Trace((" %c%d", chilm, opnd));
          ad1ilm(opnd);
          skipwhitespace();
          opnd = getoperand("datatype", chdtype);
          /* ignore the datatype */
          skipwhitespace();
        }
        break;
      case psyms:
        skipwhitespace();
        while (line[pos] == chsym) {
          ++origilmavl;
          opnd = getoperand("symbol", chsym);
          Trace((" %c%d", chsym, opnd));
          ad1ilm(opnd);
          skipwhitespace();
        }
        break;
      case pnums:
        skipwhitespace();
        while (line[pos] == chnum) {
          ++origilmavl;
          opnd = getoperand("number", chnum);
          Trace((" %c%d", chnum, opnd));
          ad1ilm(opnd);
          skipwhitespace();
        }
        break;
      default:
        break;
      }
    }
  } else if (op == -1) {
    /* end of statement */
    Trace(("---------------"));
    /* write ilms out */
    wrilms(-1);
  } else if (op == -2) {
    /* unimplemented ilm */
    Trace(("read i%d: -- unimplemented", ilm));
  }
} /* read_ilm */

static int
getlabelnum(void)
{
  int val;
  if (endilmfile) {
    fprintf(stderr, "ILM file: looking past end-of-file for label number\n");
    ++errors;
    return 0;
  }

  if (line[pos] != 'l') {
    fprintf(stderr,
            "ILM file line %d: expecting label number\n"
            "instead got: %s\n",
            ilmlinenum, line + pos);
    ++errors;
    return 0;
  }

  ++pos;
  val = 0;
  while (line[pos] >= '0' && line[pos] <= '9') {
    val = val * 10 + (line[pos] - '0');
    ++pos;
  }
  return val;
} /* getlabelnum */

int
getswel(int sz)
{
  int sw;
  sw = sem.switch_avl;
  sem.switch_avl += sz;
  if (sem.switch_size == 0) { /* allocate it */
    if (sz < 400)
      sem.switch_size = 400;
    else
      sem.switch_size = sz;
    NEW(switch_base, SWEL, sem.switch_size);
  } else {
    NEED(sem.switch_avl, switch_base, SWEL, sem.switch_size,
         sem.switch_size + 300);
  }
  return sw;
}

static void
read_label(void)
{
  int l;
  SPTR label;
  int value, first, sw;
  /* add a label to the label list */
  l = getlabelnum();
  label = getSptrVal("label");
  label = symbolxref[label];
  value = getval("value");
  first = getbit("first");
  sw = getswel(1);
  switch_base[sw].clabel = label;
  switch_base[sw].val = value;
  switch_base[sw].next = 0;
  if (!first) {
    switch_base[sw - 1].next = sw;
  }
  if (l != sw) {
    fprintf(stderr,
            "ILM file line %d: switch label %d entered at switch offset %d\n",
            ilmlinenum, l, sw);
    ++errors;
  }
} /* read_label */

static VAR *dataivl;
static VAR *lastivl;
static CONST *dataict;
static CONST *lastict;
static CONST *outerict;

static void
data_add_ivl(VAR *ivl)
{
  ivl->next = NULL;
  if (lastivl) {
    lastivl->next = ivl;
  } else {
    dataivl = ivl;
  }
  lastivl = ivl;
} /* data_add_ivl */

static void
data_push_const(void)
{
  /* rotate: NULL=>dataict=>outerict=>lastict->subc */
  lastict->subc = outerict;
  outerict = dataict;
  dataict = NULL;
  lastict = NULL;
} /* data_push_const */

static void
data_pop_const(void)
{
  CONST *save;
  for (lastict = outerict; lastict->next; lastict = lastict->next)
    ;
  /* unrotate: lastict->subc=>outerict=>dataict=>lastict->subc */
  save = lastict->subc;
  lastict->subc = dataict;
  dataict = outerict;
  outerict = save;
} /* data_pop_const */

static void
data_add_const(CONST *ict)
{
  ict->next = NULL;
  if (lastict) {
    lastict->next = ict;
  } else {
    dataict = ict;
  }
  lastict = ict;
} /* data_add_const */

static void
push(int *value)
{
  ++stack_top;
  if (stack_top >= stack_size) {
    if (stack_size == 0) {
      stack_size = 100;
      NEW(stack, int *, stack_size);
    } else {
      stack_size += 100;
      NEED(stack_top, stack, int *, stack_size, stack_size + 100);
    }
  }
  stack[stack_top] = value;
} /* push */

static int *
pop(void)
{
  if (stack_top <= 0) {
    fprintf(stderr, "ILM file line %d: stack underflow while lowering\n",
            ilmlinenum);
    exit(1);
  }
  --stack_top;
  return stack[stack_top + 1];
} /* pop */

static void
push_typestack(void)
{
  ++tsl;
  if (tsl >= tssize) {
    if (tssize == 0) {
      tssize = 100;
      NEW(ts, typestack, tssize);
    } else {
      tssize += 100;
      NEED(tsl, ts, typestack, tssize, tssize + 100);
    }
  }
} /* push_typestack */

static void
read_init(void)
{
  int val;
  DTYPE dtypev;
  int a;
  DTYPE dt;
  static SPTR sptr = SPTR_NULL;  /* the symbol being initialized */
  static DTYPE dtype; /* the datatype of that symbol */
  static int offset = 0;
  int movemember = 1;

  if (!checkname("Init")) {
    fprintf(stderr,
            "ILM file line %d: Error in initialization record\ngot %s\n",
            ilmlinenum, line);
    ++errors;
    return;
  }
  skipwhitespace();
  switch (line[pos]) {
  case 'a': /* array start/end */
    if (!checkname("array")) {
      fprintf(stderr,
              "ILM file line %d: "
              "Error in array initialization\n"
              "got %s\n",
              ilmlinenum, line);
      ++errors;
      return;
    }
    skipwhitespace();
    if (line[pos] == 's' && checkname("start")) {
      if (tsl < 0) {
        fprintf(stderr,
                "ILM file line %d: "
                "unexpected array initialization\n",
                ilmlinenum);
        ++errors;
        return;
      }
      dt = ts[tsl].dtype;
      if (DTY(dt) != TY_ARRAY) {
        fprintf(stderr,
                "ILM file line %d: "
                "array initialization for nonarray type\n",
                ilmlinenum);
        ++errors;
        return;
      }
      push_typestack();
      ts[tsl].dtype = DTySeqTyElement(dt);
      ts[tsl].member = SPTR_NULL;
      movemember = 0;
    } else if (line[pos] == 'e' && checkname("end")) {
      if (tsl < 0) {
        fprintf(stderr,
                "ILM file line %d: "
                "misplaced end-array\n",
                ilmlinenum);
        ++errors;
        return;
      }
      --tsl;
    } else {
      fprintf(stderr,
              "ILM file line %d: "
              "Error in array initialization\n"
              "got %s\n",
              ilmlinenum, line);
      ++errors;
      return;
    }
    break;
  case 'c': /* data charstring */
    val = getval("charstring");
    val = symbolxref[val];
    dtypev = DTYPEG(val);
    if (sptr > 0) {
      DTYPE totype;
      if (tsl == 0) {
        totype = dtype;
      } else {
        dt = ts[tsl].dtype;
        totype = DTY(dt) == TY_ARRAY ? DTySeqTyElement(dt) : dt;
      }
      if (DTYG(dtypev) == TY_HOLL) {
        /* convert hollerith string to proper length */
        val = cngcon(val, DTYPEG(val), totype);
      } else if (DTYG(dtypev) == TY_CHAR || DTYG(dtypev) == TY_NCHAR ||
                 (totype > 0 && DTYG(dtypev) != DTY(totype))) {
        /* convert to proper character string length or
         * convert constant to datatype of symbol */
        val = cngcon(val, dtypev, totype);
        dtypev = totype;
      }
    }
    dinit_put(dtypev, val);
    offset += size_of(dtypev);
    break;
  case 'e': /* end */
    sptr = SPTR_NULL;
    dtype = DT_NONE;
    tsl = -1;
    break;
  case 'f': /* format */
    val = getval("format");
    sptr = symbolxref[val];
    offset = 0;
    dinit_put(DINIT_LOC, sptr);
    sptr = SPTR_NULL; /* don't type-convert */
    dtype = DT_NONE;
    break;
  case 'l': /* location */
    val = getval("location");
    sptr = symbolxref[val];
    dtype = DDTG(DTYPEG(sptr));
    offset = 0;
    dinit_put(DINIT_LOC, sptr);
    push_typestack();
    ts[tsl].dtype = DTYPEG(sptr);
    ts[tsl].member = SPTR_NULL;
    break;
  case 'L': { /* Label */
    SPTR sptr;
    val = getval("Label");
    sptr = symbolxref[val];
    val = sptr;
    dinit_put(DINIT_LABEL, sptr);
    if (!UPLEVELG(sptr))
      sym_is_refd(sptr);
  } break;
  case 'n': /* namelist */
    val = getval("namelist");
    sptr = symbolxref[val];
    offset = 0;
    dinit_put(DINIT_FUNCCOUNT, gbl.func_count);
    dinit_put(DINIT_LOC, sptr);
    dinit_put(DINIT_FUNCCOUNT, gbl.func_count);
    sptr = SPTR_NULL; /* don't type-convert */
    dtype = DT_NONE;
    break;
  case 'r': /* repeat count */
    val = getval("repeat");
    dinit_put(DINIT_REPEAT, val);
    break;
  case 's': /* data symbol and type */
    val = getval("symbol");
    dtypev = getDtypeVal("datatype");
    if (datatypexref[dtypev] == 0) {
      fprintf(stderr,
              "ILM file line %d: missing data type %d for initialization\n",
              ilmlinenum, dtypev);
      ++errors;
    }
    dtypev = datatypexref[dtypev];
    val = symbolxref[val];
    if (sptr > 0) {
      DTYPE totype;
      if (tsl == 0) {
        totype = dtype;
      } else {
        dt = ts[tsl].dtype;
        totype = DTY(dt) == TY_ARRAY ? DTySeqTyElement(dt) : dt;
      }
      if (DTYG(dtypev) == TY_HOLL) {
        /* convert hollerith string to proper length */
        val = cngcon(val, DTYPEG(val), totype);
      } else if (DTYG(dtypev) == TY_CHAR || DTYG(dtypev) == TY_NCHAR ||
                 (totype > 0 && DTYG(dtypev) != DTY(totype))) {
        /* convert to proper character string length or
         * convert constant to datatype of symbol */
        val = cngcon(val, dtypev, totype);
        dtypev = totype;
      }
      if (flg.opt >= 2 && dtypev == dtype && tsl == 0 &&
          STYPEG(sptr) == ST_VAR && SCG(sptr) == SC_LOCAL) {
        NEED(aux.dvl_avl + 1, aux.dvl_base, DVL, aux.dvl_size,
             aux.dvl_size + 32);
        DVL_SPTR(aux.dvl_avl) = sptr;
        DVL_CONVAL(aux.dvl_avl) = val;
        REDUCP(sptr, 1); /* => in dvl table */
        aux.dvl_avl++;
      }
    }
    a = alignment(dtypev);
    while (a & offset) {
      dinit_put(DT_BLOG, 0);
      ++offset;
    }
    dinit_put(dtypev, val);
    offset += size_of(dtypev);
    break;
  case 't': /* typedef start/end */
    if (!checkname("typedef")) {
      fprintf(stderr,
              "ILM file line %d: "
              "Error in derived type initialization\n"
              "got %s\n",
              ilmlinenum, line);
      ++errors;
      return;
    }
    skipwhitespace();
    if (line[pos] == 's' && checkname("start")) {
      if (tsl < 0) {
        fprintf(stderr,
                "ILM file line %d: "
                "unexpected derived type initialization\n",
                ilmlinenum);
        ++errors;
        return;
      }
      dt = ts[tsl].dtype;
      if (DTYG(dt) != TY_STRUCT) {
        fprintf(stderr,
                "ILM file line %d: "
                "structure initialization for non-derived type\n",
                ilmlinenum);
        ++errors;
        return;
      }
      push_typestack();
      ts[tsl].member = DTY(dt) == TY_ARRAY ? DTyAlgTyMember(DTySeqTyElement(dt)) : DTyAlgTyMember(dt);
      ts[tsl].dtype = DTYPEG(ts[tsl].member);
      movemember = 0;
    } else if (line[pos] == 'e' && checkname("end")) {
      if (tsl < 0) {
        fprintf(stderr,
                "ILM file line %d: "
                "misplaced end-derived-type\n",
                ilmlinenum);
        ++errors;
        return;
      }
      --tsl;
    } else {
      fprintf(stderr,
              "ILM file line %d: "
              "Error in derived type initialization\n"
              "got %s\n",
              ilmlinenum, line);
      ++errors;
      return;
    }
    break;
  case 'v': /* data value and type */
    val = getval("value");
    dtypev = getDtypeVal("datatype");
    if (datatypexref[dtypev] == 0) {
      fprintf(stderr,
              "ILM file line %d: missing data type %d "
              "for initialization\n",
              ilmlinenum, dtypev);
      ++errors;
    }
    dtypev = datatypexref[dtypev];
    if (sptr > 0) {
      DTYPE totype;
      if (tsl == 0) {
        totype = dtype;
      } else {
        dt = ts[tsl].dtype;
        totype = (DTY(dt) == TY_ARRAY) ? DTySeqTyElement(dt) : dt;
      }
      if (DTYG(dtypev) == TY_CHAR || DTYG(dtypev) == TY_NCHAR ||
          (totype > 0 && DTYG(dtypev) != DTY(totype))) {
        if (DTY(totype) == TY_CHAR && DTySeqTyElement(totype) == 1) {
          /* special case of initializing char*1 to numeric */
          if (DT_ISINT(dtypev) && !DT_ISLOG(dtypev)) {
            /* integer value, not symbol */
            char buf[2];
            if (val < 0 || val > 255) {
              buf[0] = val & 0xff;
            } else {
              buf[0] = val;
            }
            buf[1] = 0;
            val = getstring(buf, 1);
            dtypev = DT_CHAR;
          }
        }
        val = cngcon(val, dtypev, totype);
        dtypev = totype;
      }
      if (flg.opt >= 2 && dtypev == dtype && tsl == 0 &&
          STYPEG(sptr) == ST_VAR && SCG(sptr) == SC_LOCAL) {
        NEED(aux.dvl_avl + 1, aux.dvl_base, DVL, aux.dvl_size,
             aux.dvl_size + 32);
        DVL_SPTR(aux.dvl_avl) = sptr;
        DVL_CONVAL(aux.dvl_avl) = val;
        REDUCP(sptr, 1); /* => in dvl table */
        aux.dvl_avl++;
      }
    }
    a = alignment(dtypev);
    while (a & offset) {
      dinit_put(DT_BLOG, 0);
      ++offset;
    }
    dinit_put(dtypev, val);
    offset += size_of(dtypev);
    break;
  }
  if (movemember && tsl > 0 && ts[tsl].member > 0) {
    ts[tsl].member = SYMLKG(ts[tsl].member);
    ts[tsl].dtype = DTYPEG(ts[tsl].member);
  }
} /* read_init */

static void
Begindata(void)
{
  dataivl = lastivl = NULL;
  dataict = lastict = outerict = NULL;
  /* prepare stack */
  ilmb.ilmavl = BOS_SIZE;
} /* Begindata */

static void
Writedata(void)
{
  dinit(dataivl, dataict);
} /* Writedata */

static void
dataDo(void)
{
  VAR *ivl;
  if (!checkname("Do")) {
    fprintf(stderr, "ILM file line %d: Error in data Do record\ngot %s\n",
            ilmlinenum, line);
    ++errors;
    return;
  }
  ivl = (VAR *)getitem(5, sizeof(VAR));
  BZERO(ivl, VAR, 1);
  ivl->id = Dostart;
  ivl->u.dostart.indvar = getoperand("ilm", chilm);
  ivl->u.dostart.lowbd = getoperand("ilm", chilm);
  ivl->u.dostart.upbd = getoperand("ilm", chilm);
  ivl->u.dostart.step = getoperand("ilm", chilm);
  data_add_ivl(ivl);
  push((int *)ivl);
} /* dataDo */

static void
dataEnddo(void)
{
  VAR *ivl;
  if (!checkname("Enddo")) {
    fprintf(stderr, "ILM file line %d: Error in data Enddo record\ngot %s\n",
            ilmlinenum, line);
    ++errors;
    return;
  }
  ivl = (VAR *)getitem(5, sizeof(VAR));
  BZERO(ivl, VAR, 1);
  ivl->id = Doend;
  ivl->u.doend.dostart = (VAR *)pop();
  data_add_ivl(ivl);
} /* dataEnddo */

static void
dataReference(void)
{
  VAR *ivl;
  if (!checkname("Reference")) {
    fprintf(stderr,
            "ILM file line %d: Error in data Reference record\ngot %s\n",
            ilmlinenum, line);
    ++errors;
    return;
  }
  ivl = (VAR *)getitem(5, sizeof(VAR));
  BZERO(ivl, VAR, 1);
  ivl->id = Varref;
  ivl->u.varref.id = S_LVALUE;
  ivl->u.varref.ptr = getoperand("ilm", chilm);
  ivl->u.varref.dtype = getDtypeOperand("datatype", chdtype);
  ivl->u.varref.shape = 0;
  data_add_ivl(ivl);
} /* dataReference */

static void
dataVariable(void)
{
  VAR *ivl;
  if (!checkname("Variable")) {
    fprintf(stderr, "ILM file line %d: Error in data Variable record\ngot %s\n",
            ilmlinenum, line);
    ++errors;
    return;
  }
  ivl = (VAR *)getitem(5, sizeof(VAR));
  BZERO(ivl, VAR, 1);
  ivl->id = Varref;
  ivl->u.varref.id = S_IDENT;
  ivl->u.varref.ptr = getoperand("ilm", chilm);
  ivl->u.varref.dtype = getDtypeOperand("datatype", chdtype);
  ivl->u.varref.shape = 0;
  data_add_ivl(ivl);
} /* dataVariable */

static void
dataConstant(void)
{
  CONST *ict;

  if (!checkname("Constant")) {
    fprintf(stderr, "ILM file line %d: Error in data Constant record\ngot %s\n",
            ilmlinenum, line);
    ++errors;
    return;
  }

  skipwhitespace();
  switch (line[pos]) {
  case 'C':
    getval("CONSTANT");
    ict = (CONST *)getitem(4, sizeof(CONST));
    BZERO(ict, CONST, 1);
    ict->id = AC_CONST;
    ict->repeatc = getoperand("number", chnum);
    ict->dtype = getDtypeOperand("datatype", chdtype);
    ict->sptr = getSptrOperand("symbol", chsym);
    if (ict->sptr && DTY(DTYPEG(ict->sptr)) == TY_PTR) {
      /* ict->sptr != 0 ==> component initialization.  Assigning
       * something (0 from NULL()) to a pointer.
       * The type of the pointer was changed late in lower()
       * after this constant was written.  Change the type
       * to avoid errors in dinit */
      ict->dtype = DT_ADDR;
    }
    skipwhitespace();
    if (line[pos] == 'n') {
      ict->u1.conval = getoperand("number", chnum);
    } else if (line[pos] == 's') {
      ict->u1.conval = getoperand("symbol", chsym);
    } else {
      fprintf(
          stderr,
          "ILM file line %d: error in Constant line: unknown value\ngot %s\n",
          ilmlinenum, line);
      ++errors;
      return;
    }
    data_add_const(ict);
    break;
  case 'L':
    getval("LITRLINT");
    ict = (CONST *)getitem(4, sizeof(CONST));
    BZERO(ict, CONST, 1);
    ict->id = AC_CONST;
    ict->u1.conval = getoperand("number", chnum);
    ict->dtype = DT_INT;
    data_add_const(ict);
    break;
  case 'I':
    getval("ID");
    ict = (CONST *)getitem(4, sizeof(CONST));
    BZERO(ict, CONST, 1);
    ict->id = AC_IDENT;
    ict->repeatc = getoperand("number", chnum);
    ict->dtype = getDtypeOperand("datatype", chdtype);
    ict->sptr = getSptrOperand("symbol", chsym);
    if (STYPEG(ict->sptr) == ST_PARAM) {
      ict->sptr = SymConval1(ict->sptr);
    }
    ict->mbr = getSptrOperand("symbol", chsym);
    data_add_const(ict);
    break;
  case 'D':
    getval("DO");
    ict = (CONST *)getitem(4, sizeof(CONST));
    BZERO(ict, CONST, 1);
    ict->id = AC_IDO;
    ict->u1.ido.index_var = getSptrOperand("do index var", chsym);
    ict->repeatc = 1;
    data_add_const(ict);
    init_list_count++; /* need an place to do idx value */
    do_level++;
    break;
  case 'd':
    getval("doend");
    if (!do_level--) {
      fprintf(stderr, "ILM file line %d: error in Constant: unexpected doend\n",
              ilmlinenum);
      ++errors;
      return;
    }
    data_pop_const();
    break;
  case 'B':
    getval("BOUNDS");
    data_push_const();
    break;
  case 'b':
    getval("boundsend");
    data_pop_const();
    if (lastict->u1.ido.initval == 0) {
      lastict->u1.ido.initval = lastict->subc;
    } else if (lastict->u1.ido.limitval == 0) {
      lastict->u1.ido.limitval = lastict->subc;
    } else {
      lastict->u1.ido.stepval = lastict->subc;
      data_push_const();
    }
    break;
  case 'A':
    getval("ARRAY");
    in_array_ctor++;
    ict = (CONST *)getitem(4, sizeof(CONST));
    BZERO(ict, CONST, 1);
    ict->id = AC_ACONST;
    ict->sptr = getSptrOperand("symbol", chsym);
    ict->dtype = getDtypeOperand("datatype", chdtype);
    ict->repeatc = 1;
    data_add_const(ict);
    data_push_const();
    break;
  case 'a':
    getval("arrayend");
    if (--in_array_ctor < 0) {
      fprintf(stderr,
              "ILM file line %d: error in Constant: too many arrayends\n",
              ilmlinenum);
      ++errors;
      return;
    }
    data_pop_const();
    break;
  case 'E':
    getval("EXPR");
    ict = (CONST *)getitem(4, sizeof(CONST));
    BZERO(ict, CONST, 1);
    ict->id = AC_IEXPR;
    ict->repeatc = getoperand("number", chnum);
    ict->u1.expr.op = getoperand("expression operator", chnum);
    if (AC_UNARY_OP(ict->u1.expr)) {
      oprnd_cnt += 1;
    } else {
      oprnd_cnt += 2;
    }
    ict->sptr = getSptrOperand("symbol", chsym);
    ict->dtype = getDtypeOperand("datatype", chdtype);
    data_add_const(ict);
    break;
  case 'O':
    getval("OPERAND");
    if (!(oprnd_cnt)) {
      fprintf(stderr,
              "ILM file line %d: error in Constant: unexpected "
              "expression operand\n",
              ilmlinenum);
      ++errors;
      return;
    }
    data_push_const();
    break;
  case 'o':
    getval("operandend");
    if (!(oprnd_cnt--)) {
      fprintf(stderr,
              "ILM file line %d: error in Constant: unexpected "
              "expression operand end\n",
              ilmlinenum);
      ++errors;
      return;
    }
    data_pop_const();
    if (lastict->u1.expr.lop == 0) {
      lastict->u1.expr.lop = lastict->subc;
    } else {
      lastict->u1.expr.rop = lastict->subc;
    }
    lastict->subc = 0;
    break;
  default:
    fprintf(stderr,
            "ILM file line %d: error in Constant: unknown constant type\n",
            ilmlinenum);
    ++errors;
    return;
    break;
  }

} /* dataConstant */

static void
dataStructure(void)
{
  CONST *ict;
  if (!checkname("structure")) {
    fprintf(stderr,
            "ILM file line %d: Error in data structure record\ngot %s\n",
            ilmlinenum, line);
    ++errors;
    return;
  }
  ict = (CONST *)getitem(4, sizeof(CONST));
  BZERO(ict, CONST, 1);
  ict->id = AC_SCONST;
  ict->repeatc = getoperand("number", chnum);
  ict->dtype = getDtypeOperand("datatype", chdtype);
  ict->sptr = getSptrOperand("symbol", chsym);
  ict->no_dinitp = getoperand("number", chnum);
  data_add_const(ict);
  data_push_const();
} /* dataConstant */

/*
 * read file entries
 */
static void
read_fileentries(void)
{
  int fihx, tag, parent, flags, lineno, srcline, level, next;
  int dirlen, filelen, funclen, fullnlen;
  char *dirname, *filename, *funcname, *fullname;

  fihx = getval("fihx");
  tag = getlval("tag");
  parent = getval("parent");
  flags = getval("flags");
  lineno = getval("lineno");
  srcline = getval("srcline");
  level = getval("level");
  next = getval("next");

  dirlen = getnamelen();
  dirname = line + pos;
  pos += dirlen;

  filelen = getnamelen();
  filename = line + pos;
  pos += filelen;

  funclen = getnamelen();
  funcname = line + pos;
  pos += funclen;

  fullnlen = getnamelen();
  fullname = line + pos;
  pos += fullnlen;

  dirname[dirlen] = '\0';
  filename[filelen] = '\0';
  funcname[funclen] = '\0';
  fullname[fullnlen] = '\0';

  if (funclen == 0)
    funcname = NULL;

  if (fihx > 1) {
    addfile(fullname, funcname, tag, flags, lineno, srcline, level);
    FIH_PARENT(fihx) = parent;
  }
}

/*
 * read symbol for which GSCOPE must be set
 */
static void
read_global(void)
{
  int sptr;
  sptr = getval("global");
  sptr = symbolxref[sptr];
  if (sptr > NOSYM) {
    GSCOPEP(sptr, 1);
  }
} /* read_global */

/*
 * Read CCFF messages, save in the CCFF message database
 */
static int
read_CCFF(void)
{
  int endilmfile;
  int fihx;
  if (!checkname("CCFF")) {
    fprintf(stderr, "ILM file line %d: Expecting CCFF info, got %s\n",
            ilmlinenum, line);
    ++errors;
    return 0;
  }
  fihx = 1;
  do {
    /* CCFFinl
     * CCFFlni
     * CCFFmsg
     * CCFFarg
     * CCFFtxt
     * CCFFend */
    int seq, lineno, msgtype;
    char *symname, *msgid, *funcname;
    char *argname, *argval, *text;

    endilmfile = read_line();
    if (endilmfile)
      return endilmfile;
    if (strncmp(line, "CCFF", 4) != 0) {
      fprintf(stderr, "ILM file line %d: Expecting CCFF data, got %s\n",
              ilmlinenum, line);
      ++errors;
      return 0;
    }
    switch (line[4]) {
    case 'i': /* CCFFinl */
      pos = 8;
      break;
    case 'l': /* CCFFlni */
      pos = 8;
      break;
    case 'm': /* CCFFmsg */
      pos = 8;
      seq = getval("seq");
      lineno = getval("lineno");
      msgtype = getval("type");
      symname = getname();
      funcname = getname();
      msgid = getname();
      save_ccff_msg(msgtype, msgid, fihx, lineno, symname, funcname);
      break;
    case 'a': /* CCFFarg */
      pos = 8;
      argname = getname();
      argval = getname();
      save_ccff_arg(argname, argval);
      break;
    case 't': /* CCFFtxt */
      pos = 8;
      text = line + pos;
      save_ccff_text(text);
      break;
    case 'e': /* CCFFend */
      return 0;
      break;
    }
  } while (1);
} /* read_CCFF */

/*
 * read a host subprogram entry symbol
 */
static void
read_Entry(void)
{
  SPTR sptr;
  int outersub;
  sptr = getSptrVal("Entry");
  sptr = symbolxref[sptr];
  if (sptr > NOSYM && gbl.outersub) {
    outersub = symbolxref[gbl.outersub];
    if (SYMLKG(outersub) == 0) {
      SYMLKP(outersub, NOSYM);
    }
    SYMLKP(sptr, SYMLKG(outersub));
    SYMLKP(outersub, sptr);
  }
} /* read_Entry */

/*
 * read names of contained subprograms
 */
static void
read_contained(void)
{
  int namelen, hashid, sptr;
  char *ch;
  if (!checkname("contained")) {
    fprintf(stderr,
            "ILM file line %d: Expecting contained routine name, got %s\n",
            ilmlinenum, line);
    ++errors;
    return;
  }
  if (gbl.internal == 1)
    ++gbl.numcontained;
  namelen = getnamelen();
  ch = line + pos;
  ch[namelen] = '\0';
  HASH_ID(hashid, ch, namelen);
  if (hashid < 0)
    hashid = -hashid;
  /* look for the symbol */
  for (sptr = stb.hashtb[hashid]; sptr > NOSYM; sptr = HASHLKG(sptr)) {
    switch (STYPEG(sptr)) {
    case ST_PROC:
    case ST_ENTRY:
      if (!INMODULEG(sptr) && strcmp(ch, SYMNAME(sptr)) == 0) {
        CONTAINEDP(sptr, 1);
        return;
      }
      break;
    default:
      break;
    }
  }
  /* not found for this subprogram, must be no calls to it */
} /* read_contained */

/* Replicate prefix string a number of times */
static void
put_prefix(FILE *dfile, char *str, int cnt)
{
  int i;

  fprintf(dfile, "    ");
  for (i = 0; i < cnt; i++)
    fprintf(dfile, "%s", str);
}

void
dmp_const(CONST *acl, int indent)
{
  CONST *c_aclp;
  char two_spaces[3] = "  ";
  FILE *dfile;

  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;

  if (!acl) {
    return;
  }

  if (indent == 0)
    fprintf(dfile, "line %d:\n", gbl.lineno);

  for (c_aclp = acl; c_aclp; c_aclp = c_aclp->next) {
    switch (c_aclp->id) {
    case AC_IDENT:
      put_prefix(dfile, two_spaces, indent);
      fprintf(dfile, "AC_IDENT: '%s' (%d), repeatc=%ld\n",
              SYMNAME(c_aclp->sptr), c_aclp->sptr, c_aclp->repeatc);
      break;
    case AC_CONST:
      put_prefix(dfile, two_spaces, indent);
      fprintf(dfile, "AC_CONST: %d, sptr %d repeatc=%ld\n", c_aclp->u1.conval,
              c_aclp->sptr, c_aclp->repeatc);
      break;
    case AC_IEXPR:
      put_prefix(dfile, two_spaces, indent);
      fprintf(dfile, "AC_IEXPR: op %d, repeatc %ld\n", c_aclp->u1.expr.op,
              c_aclp->repeatc);
      dmp_const(c_aclp->u1.expr.lop, indent + 1);
      dmp_const(c_aclp->u1.expr.rop, indent + 1);
      break;
    case AC_IDO:
      put_prefix(dfile, two_spaces, indent);
      fprintf(dfile,
              "AC_IDO: sptr %d, index var sptr %d, init val %p, "
              "limit val %p, step val %p, repeatc %ld\n",
              c_aclp->sptr, c_aclp->u1.ido.index_var, c_aclp->u1.ido.initval,
              c_aclp->u1.ido.limitval, c_aclp->u1.ido.stepval, c_aclp->repeatc);
      put_prefix(dfile, two_spaces, indent);
      fprintf(dfile, " Initialization Values:\n");
      dmp_const(c_aclp->subc, indent + 1);
      break;
    case AC_ACONST:
      put_prefix(dfile, two_spaces, indent);
      fprintf(dfile, "AC_ACONST: sptr %d, repeatc %ld\n", c_aclp->sptr,
              c_aclp->repeatc);
      put_prefix(dfile, two_spaces, indent);
      fprintf(dfile, " Initialization Values:\n");
      dmp_const(c_aclp->subc, indent + 1);
      break;
    case AC_SCONST:
      put_prefix(dfile, two_spaces, indent);
      fprintf(dfile, "AC_SCONST: sptr %d, repeatc %ld\n", c_aclp->sptr,
              c_aclp->repeatc);
      put_prefix(dfile, two_spaces, indent);
      fprintf(dfile, " Initialization Values:\n");
      dmp_const(c_aclp->subc, indent + 1);
      break;
    default:
      put_prefix(dfile, two_spaces, indent);
      fprintf(dfile, "*** UNKNOWN/UNUSED CONST id %d\n", c_aclp->id);
      break;
    }
  }
}

/*
 * given string and some other information, produce the
 * external name that assem will produce
 *  (copied from ipasave.c)
 */
char *
getexnamestring(char *string, int sptr, int stype, int scg, int extraunderscore)
{
  static char *id = NULL;
  static int idsize = 0;
  char *s;
  s = string;
  if (idsize == 0) {
    idsize = 200;
    NEW(id, char, idsize);
  }
  if (s[0] == '.') {
    sprintf(id, "%s%d", s, sptr);
  } else {
    char *ss;
    int l, ll;
    int has_underscore;
    l = 0;
    switch (stype) {
    case ST_VAR:
    case ST_ARRAY:
    case ST_STRUCT:
    case ST_UNION:
      ll = strlen(s);
      /* l+ll+2 = ll for string, 1 for optional _, 1 for null */
      NEED(l + ll + 2, id, char, idsize, l + ll + 200);
      switch (scg) {
      case SC_EXTERN:
        id[l++] = '_';
        break;
      default:
        break;
      }
      strcpy(id + l, s);
      l += ll;
      break;
    case ST_ENTRY:
    case ST_PROC:
      if (gbl.internal >= 1 && CONTAINEDG(sptr)) {
        int m;
        m = INMODULEG(gbl.outersub);
        if (m) {
          ss = SYMNAME(m);
          ll = strlen(ss);
          NEED(l + ll + 1, id, char, idsize, l + ll + 200);
          for (; *ss; ++ss) {
            if (*ss == '$') {
              id[l++] = flg.dollar;
            } else {
              id[l++] = *ss;
            }
          }
          id[l++] = '_';
        }
        ss = SYMNAME(gbl.outersub);
        ll = strlen(ss);
        NEED(l + ll + 1, id, char, idsize, l + ll + 200);
        for (; *ss; ++ss) {
          if (*ss == '$') {
            id[l++] = flg.dollar;
          } else {
            id[l++] = *ss;
          }
        }
        id[l++] = '_';
        ss = SYMNAME(sptr);
        ll = strlen(ss);
        NEED(l + ll + 1, id, char, idsize, l + ll + 200);
        for (; *ss; ++ss) {
          if (*ss == '$') {
            id[l++] = flg.dollar;
          } else {
            id[l++] = *ss;
          }
        }
        id[l] = '\0';
      } else {
        int m;
        if (XBIT(119, 0x1000)) { /* add leading underscore */
          NEED(l + 1, id, char, idsize, l + 200);
          id[l++] = '_';
        }
        m = INMODULEG(sptr);
        if (m) {
          ss = SYMNAME(m);
          ll = strlen(ss);
          NEED(l + ll + 1, id, char, idsize, l + ll + 200);
          for (; *ss; ++ss) {
            if (*ss == '$') {
              id[l++] = flg.dollar;
            } else {
              id[l++] = *ss;
            }
          }
          id[l++] = '_';
        }
        has_underscore = 0;
        ll = strlen(s);
        /* l+ll+3 = ll for string, 2 for optional __, 1 for null */
        NEED(l + ll + 3, id, char, idsize, l + ll + 200);
        for (ss = s; *ss; ++ss) {
          if (*ss == '_') {
            id[l++] = *ss;
            has_underscore = 1;
          } else if (*ss == '$') {
            id[l++] = flg.dollar;
          } else {
            id[l++] = *ss;
          }
        }
        id[l] = '\0';
      }
      if (stype == ST_ENTRY || extraunderscore) {
        if (!XBIT(119, 0x01000000)) {
          id[l++] = '_';
          if (XBIT(119, 0x2000000) && has_underscore && !LIBSYMG(sptr)) {
            id[l++] = '_';
          }
        }
      }
      id[l] = '\0';
#if defined(TARGET_WIN_X86) && defined(PGFTN)
      if (STYPEG(sptr) == ST_CMBLK && !CCSYMG(sptr) && XBIT(119, 0x01000000))
        upcase_name(id);
      if ((STYPEG(sptr) == ST_ENTRY || STYPEG(sptr) == ST_PROC) &&
          MSCALLG(sptr) && !STDCALLG(sptr))
        upcase_name(id);
#endif
      break;
    case ST_CMBLK: /* just leading/trailing underscores */
      if (XBIT(119, 0x1000)) { /* add leading underscore */
        NEED(l + 1, id, char, idsize, l + 200);
        id[l++] = '_';
      }
      has_underscore = 0;
      ll = strlen(s);
      /* l+ll+3 = ll for string, 2 for optional __, 1 for null */
      NEED(l + ll + 1, id, char, idsize, l + ll + 200);
      for (ss = s; *ss; ++ss) {
        if (*ss == '_') {
          id[l++] = *ss;
          has_underscore = 1;
        } else if (*ss == '$') {
          id[l++] = flg.dollar;
        } else {
          id[l++] = *ss;
        }
      }
      id[l] = '\0';
      break;
    default:
      ll = strlen(s);
      NEED(l + ll + 1, id, char, idsize, l + ll + 200);
      strcpy(id + l, s);
      l += ll;
      break;
    }
  }
  return id;
} /* getexnamestring */

/*
 * find index for sptr, or add one
 */
static int
newindex(int sptr)
{
  int l, h, i, j;
  l = 0;
  h = ipab.indexavl - 1;
  while (l <= h) {
    i = (l + h) >> 1; /* (l+h)/2 */
    if (IPNDX_SPTR(i) > sptr) {
      h = i - 1;
    } else if (IPNDX_SPTR(i) < sptr) {
      l = i + 1;
    } else {
      break;
    }
  }
  if (l <= h) { /* found it */
    return i;
  }
  NEED(ipab.indexavl + 1, ipab.index, IPAindex, ipab.indexsize,
       ipab.indexsize + 100);
  i = h + 1; /* where to insert */
  for (j = ipab.indexavl - 1; j >= i; --j) {
    IPNDX_SPTR(j + 1) = IPNDX_SPTR(j);
    IPNDX_INFO(j + 1) = IPNDX_INFO(j);
  }
  ++ipab.indexavl;
  IPNDX_SPTR(i) = sptr;
  IPNDX_INFO(i) = 0;
  Trace(("add info index for symbol %d:%s at index %d of %d", sptr,
         SYMNAME(sptr), i, ipab.indexavl));
  return i;
} /* newindex */

/*
 * return new ipab.info index
 */
static int
newinfo(void)
{
  int i = ipab.infoavl;
  ++ipab.infoavl;
  NEED(ipab.infoavl, ipab.info, IPAinfo, ipab.infosize, ipab.infosize + 100);
  return i;
} /* newinfo */

/*
 * find index for sptr or return -1
 */
static int
findindex(int sptr)
{
  int l, h, i;
  l = 0;
  h = ipab.indexavl - 1;
  while (l <= h) {
    i = (l + h) >> 1; /* (l+h)/2 */
    if (IPNDX_SPTR(i) > sptr) {
      h = i - 1;
    } else if (IPNDX_SPTR(i) < sptr) {
      l = i + 1;
    } else {
      break;
    }
  }
  if (l <= h) { /* found it */
    return i;
  }
  return -1;
} /* findindex */

/**
 * return -1 if nme1/nme2 point to the same address;
 * return 0 if they point to different addresses;
 * return 1 if they may point to the same address
 */
int
IPA_nme_conflict(int nme1, int nme2)
{
  int t2, vnme1, sym1, sym2, i1, n1;

  if (!XBIT(89, 0x100) || XBIT(89, 0x80))
    return 1;

  /* nme1 must be an indirection; see if we have information about it */
  if (NME_TYPE(nme1) != NT_IND)
    return 1;
  /* single direction? */
  vnme1 = NME_NM(nme1);
  if (NME_TYPE(vnme1) != NT_VAR)
    return 1;
  sym1 = NME_SYM(vnme1);
  n1 = findindex(sym1);
  if (n1 < 0)
    return 1;
  i1 = IPNDX_INFO(n1);

  t2 = NME_TYPE(nme2);
  if (t2 == NT_VAR) {
    int j1, count;
    /* see if nme2 is in the list of symbols pointed to by nme1 */
    sym2 = NME_SYM(nme2);
    count = 0;
    for (j1 = i1; j1 > 0; j1 = IPNFO_NEXT(j1)) {
      switch (IPNFO_TYPE(j1)) {
      case INFO_LTARGET:
      case INFO_GTARGET:
        if (IPNFO_TARGET(j1) == sym2) {
          if (j1 == i1 && IPNFO_NEXT(j1) == 0 && IPNFO_INDIRECT(j1) == 0 &&
              IPNFO_IMPRECISE(j1) == 0) {
            /* the only target, no stars, not imprecise */
            return -1;
          }
          return 1;
        }
        ++count;
        break;
      case INFO_OGTARGET:
      case INFO_OTARGET:
        ++count;
        break;
      }
      /* if we have some targets, none of them are this symbol */
      if (count) {
        ++IPA_Pointer_Targets_Disambiguated;
        return 0;
      }
    }
  } else if (t2 == NT_IND) {
    int vnme2, i2, j1, count1, n2;
    /* t2 is an indirection, too; see if we have info about it! */
    /* single direction? */
    vnme2 = NME_NM(nme2);
    if (NME_TYPE(vnme2) != NT_VAR)
      return 1;
    sym2 = NME_SYM(vnme2);
    n2 = findindex(sym2);
    if (n2 < 0)
      return 1;
    i2 = IPNDX_INFO(n2);

    /* two pointers, we have information about both pointers;
     * they may point to the same item precisely: SAME
     * they may point to to different items: NO CONFLICT
     * otherwise: CONFLICT */
    if (IPNFO_NEXT(i1) == 0 && IPNFO_NEXT(i2) == 0) {
      /* Both the same type, both precise? */
      if (IPNFO_TYPE(i1) == IPNFO_TYPE(i2)) {
        switch (IPNFO_TYPE(i1)) {
        case INFO_LTARGET:
        case INFO_GTARGET:
          if (IPNFO_IMPRECISE(i1) == 0 && IPNFO_IMPRECISE(i2) == 0) {
            /* same symbol? */
            if (IPNFO_TARGET(i1) == IPNFO_TARGET(i2) &&
                IPNFO_TARGET(i1) != sym1 && IPNFO_TARGET(i1) != sym2 &&
                IPNFO_INDIRECT(i1) == IPNFO_INDIRECT(i2)) {
              /* only one target, same target */
              return -1;
            }
          }
          break;
        case INFO_OTARGET:
        case INFO_OGTARGET:
          if (IPNFO_IMPRECISE(i1) == 0 && IPNFO_IMPRECISE(i2) == 0) {
            /* same symbol? */
            if (IPNFO_TARGET(i1) == IPNFO_TARGET(i2) &&
                IPNFO_INDIRECT(i1) == IPNFO_INDIRECT(i2)) {
              /* only one target, same target */
              return -1;
            }
          }
          break;
        }
      }
    }
    count1 = 0;
    for (j1 = i1; j1 > 0; j1 = IPNFO_NEXT(j1)) {
      int j2;
      switch (IPNFO_TYPE(j1)) {
      case INFO_LTARGET:
        if (IPNFO_TARGET(j1) == sym1) {
          /* S1 -> *S1, ignore */
          continue;
        }
        if (IPNFO_TARGET(j1) == sym2) {
          /* probably S1 -> *S2, conflict */
          return 1;
        }
        FLANG_FALLTHROUGH;
      case INFO_GTARGET:
      case INFO_OGTARGET:
      case INFO_OTARGET:
        ++count1;
        /* look for this pointee in the i2 list */
        for (j2 = i2; j2 > 0; j2 = IPNFO_NEXT(j2)) {
          if (IPNFO_TYPE(j2) == IPNFO_TYPE(j1) &&
              IPNFO_TARGET(j2) == IPNFO_TARGET(j1)) {
            /* S2 -> Y and S1 -> Y */
            return 1;
          }
        }
      }
    }
    /* no shared targets, independent */
    if (count1) {
      ++IPA_Pointer_Targets_Disambiguated;
      return 0;
    }
  }
  return 1;
} /* IPA_nme_conflict */

/** \brief Detect Fortran 90 name conflicts.
 *
 * return -1 if nme1/nme2 point to the same address;
 * return 0 if they point to different addresses;
 * return 1 if they may point to the same address
 */
int
F90_nme_conflict(int nme1, int nme2)
{
  int t2, vnme1, sym1, sym2, i1, n1;

  /* special case:  see if at least one of these input pointers is a structure member */
  if (F90_struct_mbr_nme_conflict(nme1, nme2) == 0) {
    return 0;
  }
  /* nme1 must be an indirection; see if we have information about it */
  if (NME_TYPE(nme1) != NT_IND)
    return 1;
  /* single direction? */
  vnme1 = NME_NM(nme1);
  if (NME_TYPE(vnme1) != NT_VAR)
    return 1;
  sym1 = NME_SYM(vnme1);
  n1 = findindex(sym1);
  if (n1 < 0)
    return 1;
  i1 = IPNDX_INFO(n1);

  t2 = NME_TYPE(nme2);
  if (t2 == NT_VAR) {
    int j1, count;
    sym2 = NME_SYM(nme2);
    /* see if sym2 is in the list of symbols pointed to by nme1 */
    count = 0;
    for (j1 = i1; j1 > 0; j1 = IPNFO_NEXT(j1)) {
      switch (IPNFO_TYPE(j1)) {
      case INFO_FSTARGET:
        if (IPNFO_TARGET(j1) == sym2) {
          return 1;
        }
        ++count;
        break;
      case INFO_FLDYNTARGET:
      case INFO_FGDYNTARGET:
      case INFO_FOTARGET:
      case INFO_FOSTARGET:
        ++count;
        break;
      case INFO_FUNKTARGET:
        return 1;
        break;
      }
    }
    if (SCG(sym2) == SC_BASED) {
      int i2, n2, count1;
      /* see if the base pointer might conflict with this pointer */
      sym2 = MIDNUMG(sym2);
      n2 = findindex(sym2);
      if (n2 < 0)
        return 1;
      i2 = IPNDX_INFO(n2);
      /* two pointers, we have information about both pointers;
       * they may point to to different items: NO CONFLICT
       * otherwise: CONFLICT */
      count1 = 0;
      for (j1 = i1; j1 > 0; j1 = IPNFO_NEXT(j1)) {
        int j2;
        switch (IPNFO_TYPE(j1)) {
        case INFO_FSTARGET:
          if (IPNFO_TARGET(j1) == sym1) {
            /* S1 -> *S1, ignore */
            continue;
          }
          if (IPNFO_TARGET(j1) == sym2) {
            /* probably S1 -> *S2, conflict */
            return 1;
          }
          FLANG_FALLTHROUGH;
        case INFO_FLDYNTARGET:
        case INFO_FGDYNTARGET:
        case INFO_FOTARGET:
        case INFO_FOSTARGET:
          ++count1;
          /* look for this pointee in the i2 list */
          for (j2 = i2; j2 > 0; j2 = IPNFO_NEXT(j2)) {
            if (IPNFO_TYPE(j2) == IPNFO_TYPE(j1) &&
                IPNFO_TARGET(j2) == IPNFO_TARGET(j1)) {
              /* S2 -> Y and S1 -> Y */
              return 1;
            }
          }
          break;
        case INFO_FUNKTARGET:
          return 1;
          break;
        }
      }
      /* no shared targets, independent */
      if (count1) {
        return 0;
      }
    } else {
      /* if we have some targets, none of them are this symbol */
      if (count) {
        return 0;
      }
    }
  } else if (t2 == NT_IND) {
    int vnme2, i2, j1, count1, n2;
    /* t2 is an indirection, too; see if we have info about it! */
    /* single direction? */
    vnme2 = NME_NM(nme2);
    if (NME_TYPE(vnme2) != NT_VAR)
      return 1;
    sym2 = NME_SYM(vnme2);
    n2 = findindex(sym2);
    if (n2 < 0)
      return 1;
    i2 = IPNDX_INFO(n2);

    /* two pointers, we have information about both pointers;
     * they may point to to different items: NO CONFLICT
     * otherwise: CONFLICT */
    count1 = 0;
    for (j1 = i1; j1 > 0; j1 = IPNFO_NEXT(j1)) {
      int j2;
      switch (IPNFO_TYPE(j1)) {
      case INFO_FSTARGET:
        if (IPNFO_TARGET(j1) == sym1) {
          /* S1 -> *S1, ignore */
          continue;
        }
        if (IPNFO_TARGET(j1) == sym2) {
          /* probably S1 -> *S2, conflict */
          return 1;
        }
        FLANG_FALLTHROUGH;
      case INFO_FLDYNTARGET:
      case INFO_FGDYNTARGET:
      case INFO_FOTARGET:
      case INFO_FOSTARGET:
        ++count1;
        /* look for this pointee in the i2 list */
        for (j2 = i2; j2 > 0; j2 = IPNFO_NEXT(j2)) {
          if (IPNFO_TYPE(j2) == IPNFO_TYPE(j1) &&
              IPNFO_TARGET(j2) == IPNFO_TARGET(j1)) {
            /* S2 -> Y and S1 -> Y */
            return 1;
          }
        }
        break;
      case INFO_FUNKTARGET:
        return 1;
        break;
      }
    }
    /* no shared targets, independent */
    if (count1) {
      return 0;
    }
  }
  return 1;
} /* F90_nme_conflict */

/** \brief Detect Fortran 90 structure member name conflicts.
 *
 * return 0 if they point to different addresses;
 * return 1 otherwise
 */
int
F90_struct_mbr_nme_conflict(int nme1, int nme2)
{
  int mbr1, struct1, is_struct_mbr1, sptr1;
  int mbr2, struct2, is_struct_mbr2, sptr2;
  is_struct_mbr1 = 0;
  is_struct_mbr2 = 0;

  /* handles one level of struct%mbr only */

  /* input 1 */
  if (NME_TYPE(nme1) == NT_IND) {
    mbr1 = NME_NM(nme1);
    if (NME_TYPE(mbr1) == NT_MEM) {
      /* struct member */
      struct1 = NME_NM(mbr1);
      if (NME_TYPE(struct1) == NT_VAR) {
        sptr1 = NME_SYM(struct1);
        if (sptr1 > 0) {
          is_struct_mbr1 = 1;
        }
      }
    }
  }
  /* input 2 */
  if (NME_TYPE(nme2) == NT_IND) {
    mbr2 = NME_NM(nme2);
    if (NME_TYPE(mbr2) == NT_MEM) {
      /* struct member */
      struct2 = NME_NM(mbr2);
      if (NME_TYPE(struct2) == NT_VAR) {
        sptr2 = NME_SYM(struct2);
        if (sptr2 > 0) {
          is_struct_mbr2 = 1;
        }
      }
    }
  }
  if (is_struct_mbr1 && is_struct_mbr2) {
    /* both are structure member pointers */
    if (struct1 == struct2 && mbr1 == mbr2) {
      return 1; /* same */
    }
    if (NOCONFLICTG(sptr1) && NOCONFLICTG(sptr1)) {
      return 0;
    }
  }
  else if (is_struct_mbr1) {
    if (NME_TYPE(nme2) == NT_IND && NME_TYPE(NME_NM(nme2)) == NT_VAR) {
      /* first one is a structure member pointer, the other is not */
      sptr2 = NME_SYM(NME_NM(nme2));
      if (sptr2 > 0 && NOCONFLICTG(sptr2) && NOCONFLICTG(sptr1)) {
        return 0;
      }
    }
  }
  else if (is_struct_mbr2) {
    if (NME_TYPE(nme1) == NT_IND && NME_TYPE(NME_NM(nme1)) == NT_VAR) {
      /* second one is a structure member pointer, the other is not */
      sptr1 = NME_SYM(NME_NM(nme1));
      if (sptr1 > 0 && NOCONFLICTG(sptr1) && NOCONFLICTG(sptr2)) {
        return 0;
      }
    }
  }
  return 1; /* anything else */
} /* F90_struct_mbr_nme_conflict */

/**
 * \return 1 if sptr is a pointer which has its pointer targets identified,
 * and its pointer targets do not conflict with any other pointers in
 * the program, and do not conflict which any array used in the program.
 * return 0 otherwise
 */
int
IPA_pointer_safe(int nme)
{
  int vnme, sym, n, nme2;
  /* both -x 89 0x20000000 and -x 89 0x100 must be set */
  if (XBIT(89, 0x20000100) != 0x20000100 || XBIT(89, 0x80))
    return 0;
  if (NME_TYPE(nme) != NT_IND)
    return 0;
  /* single direction? */
  vnme = NME_NM(nme);
  if (NME_TYPE(vnme) != NT_VAR)
    return 0;
  sym = NME_SYM(vnme);
  n = findindex(sym);
  if (n < 0)
    return 0;

  /* go through other NMEs, see if nme conflicts with other nmes */
  for (nme2 = 2; nme2 < nmeb.stg_avail; ++nme2) {
    switch (NME_TYPE(nme2)) {
    case NT_VAR:
      /* don't compare against itself */
      if (nme2 != vnme) {
        if (IPA_nme_conflict(nme, nme2)) {
          /* nme conflicts with nme2, not safe */
          return 0;
        }
      }
      break;
    case NT_IND:
      /* don't compare against itself */
      if (NME_NM(nme2) != vnme) {
        if (IPA_nme_conflict(nme, nme2)) {
          /* nme conflicts with nme2, not safe */
          return 0;
        }
      }
      break;
    default:
      break;
    }
  }
  return 1;
} /* IPA_pointer_safe */

/**
 * \return 1 if sptr is known to be within a limited integer range
 * at the start of the function.
 * return 0 otherwise
 */
int
IPA_range(int sptr, int *plo, int *phi)
{
  int n, i;
  if (XBIT(89, 0x80))
    return 0;
  n = findindex(sptr);
  if (n < 0)
    return 0;
  for (i = IPNDX_INFO(n); i > 0; i = IPNFO_NEXT(i)) {
    if (IPNFO_TYPE(i) == INFO_RANGE) {
      ++IPA_Range_Propagated;
      *plo = IPNFO_LOW(i);
      *phi = IPNFO_HIGH(i);
      return 1;
    }
  }
  return 0;
} /* IPA_range */

/*
 * return 1 if sptr has never had its address taken.
 * return 0 otherwise
 */
int
IPA_noaddr(int sptr)
{
  int n, i;
  if (!XBIT(89, 0x20000) || XBIT(89, 0x80))
    return 0;
  n = findindex(sptr);
  if (n < 0)
    return 0;
  for (i = IPNDX_INFO(n); i > 0; i = IPNFO_NEXT(i)) {
    if (IPNFO_TYPE(i) == INFO_NOADDR) {
      ++IPA_Safe_Globals_Confirmed;
      return 1;
    }
  }
  return 0;
} /* IPA_noaddr */

/** \brief detect pure function from IPA standpoint
 *
 * \return 1 if function sptr is known to be 'pure'
 * that means it does not read or modify globals
 * or arguments or file statics.
 */
int
IPA_func_pure(int sptr)
{
  int n, i;
  if (!XBIT(66, 0x10000))
    return 0;
  n = findindex(sptr);
  if (n < 0)
    return 0;
  for (i = IPNDX_INFO(n); i > 0; i = IPNFO_NEXT(i)) {
    if (IPNFO_TYPE(i) == INFO_FUNC) {
      if (IPNFO_FUNCINFO(i))
        return 0;
      ++IPA_Func_Propagated;
      return 1;
    }
  }
  return 0;
} /* IPA_func_pure */

/** \brief detect "almost pure" function for IPA
 *
 * \return 1 if function sptr is known to be 'almost pure'
 * that means it does not read or modify globals that are
 * visible in the current file, and does not modify its arguments.
 */
int
IPA_func_almostpure(int sptr)
{
  int n, i;
  if (!XBIT(66, 0x10000))
    return 0;
  n = findindex(sptr);
  if (n < 0)
    return 0;
  for (i = IPNDX_INFO(n); i > 0; i = IPNFO_NEXT(i)) {
    if (IPNFO_TYPE(i) == INFO_FUNC) {
      if (IPNFO_FUNCINFO(i) &
          (FINFO_WRITEARG | FINFO_READGLOB | FINFO_WRITEGLOB))
        return 0;
      /* if defined in this file, have to pay attention to statics also */
      if (FUNCLINEG(sptr) &&
          (IPNFO_FUNCINFO(i) & (FINFO_READSTATIC | FINFO_WRITESTATIC)))
        return 0;
      ++IPA_Func_Propagated;
      return 1;
    }
  }
  return 0;
} /* IPA_func_almostpure */

/*
 * return stride for pointers
 */
long
IPA_pstride(int sptr)
{
  int n, i;
  if (!XBIT(66, 0x1000000))
    return 0;
  n = findindex(sptr);
  if (n < 0)
    return 0;
  for (i = IPNDX_INFO(n); i > 0; i = IPNFO_NEXT(i)) {
    if (IPNFO_TYPE(i) == INFO_PSTRIDE) {
      ++IPA_Pointer_Strides_Propagated;
      return IPNFO_PSTRIDE(i);
    }
  }
  return 0;
} /* IPA_pstride */

/*
 * return section stride for pointers
 */
long
IPA_sstride(int sptr)
{
  int n, i;
  if (!XBIT(66, 0x1000000))
    return 0;
  n = findindex(sptr);
  if (n < 0)
    return 0;
  for (i = IPNDX_INFO(n); i > 0; i = IPNFO_NEXT(i)) {
    if (IPNFO_TYPE(i) == INFO_SSTRIDE) {
      ++IPA_Pointer_Strides_Propagated;
      return IPNFO_SSTRIDE(i);
    }
  }
  return 0;
} /* IPA_sstride */

/*
 * return '1' if 'free' is never called anywhere in the application
 */
int
IPA_NoFree(void)
{
  return 0; /* until we know */
} /* IPA_NoFree */

/*
 * return 1 if sptr is a 'safe' symbol, not modified by any calls.
 * return 0 otherwise
 */
int
IPA_safe(int sptr)
{
  int n, i;
  if (!XBIT(89, 0x20000) || XBIT(89, 0x80))
    return 0;
  n = findindex(sptr);
  if (n < 0)
    return 0;
  for (i = IPNDX_INFO(n); i > 0; i = IPNFO_NEXT(i)) {
    if (IPNFO_TYPE(i) == INFO_SAFE) {
      ++IPA_Safe_Globals_Confirmed;
      return 1;
    }
  }
  return 0;
} /* IPA_safe */

/*
 * return 1 if sptr is 'safe' in a call to 'funcsptr',i
 * not modified by funcsptr or any calls within funcsptr
 * return 0 otherwise
 */
int
IPA_call_safe(int funcsptr, int sptr)
{
  int n, i;
  if (!XBIT(89, 0x20000) || XBIT(89, 0x80))
    return 0;
  n = findindex(sptr);
  if (n < 0)
    return 0;
  for (i = IPNDX_INFO(n); i > 0; i = IPNFO_NEXT(i)) {
    if (IPNFO_TYPE(i) == INFO_SAFE || IPNFO_TYPE(i) == INFO_ALLCALLSAFE) {
      ++IPA_Safe_Globals_Confirmed;
      return 1;
    }
    if (IPNFO_TYPE(i) == INFO_CALLSAFE && IPNFO_VAL(i) == funcsptr) {
      ++IPA_Safe_Globals_Confirmed;
      return 1;
    }
  }
  return 0;
} /* IPA_call_safe */

/*
 * return 1 if sptr is 'safe' in any call from this function
 * even if it is modified by this function itself
 */
int
IPA_allcall_safe(int sptr)
{
  int n, i;
  if (!XBIT(89, 0x20000) || XBIT(89, 0x80))
    return 0;
  n = findindex(sptr);
  if (n < 0)
    return 0;
  for (i = IPNDX_INFO(n); i > 0; i = IPNFO_NEXT(i)) {
    if (IPNFO_TYPE(i) == INFO_SAFE || IPNFO_TYPE(i) == INFO_ALLCALLSAFE) {
      ++IPA_Safe_Globals_Confirmed;
      return 1;
    }
  }
  return 0;
} /* IPA_allcall_safe */

static struct {
  bool smp;
  bool recursive;
  int profile;
  int x5;
  int x121;
  int x123;
} cusv;

void
cuda_emu_start(void)
{
  gbl.cudaemu = cudaemu;
  if (cudaemu) {
    cusv.smp = flg.smp;
    cusv.recursive = flg.recursive;
    cusv.profile = flg.profile;
    cusv.x5 = flg.x[5];
    cusv.x121 = flg.x[121];
    cusv.x123 = flg.x[123];
    flg.smp = false;
    flg.recursive = true;
    flg.profile = 0;
    flg.x[121] |= 0x1; /* -Mnoframe */
    if (flg.debug) {
      flg.x[5] |= 1;
      flg.x[123] |= 0x400;
    }
  }
}

void
cuda_emu_end(void)
{
  if (cudaemu) {
    flg.smp = cusv.smp;
    flg.recursive = cusv.recursive;
    flg.profile = cusv.profile;
    flg.x[5] = cusv.x5;
    flg.x[121] = cusv.x121;
    flg.x[123] = cusv.x123;
    cudaemu = 0;
    gbl.cudaemu = 0;
  }
}

#ifdef FLANG2_UPPER_UNUSED
/* get the size of STATICS/BSS - this has to be done after fix_datatype so that
   we can get the size of sptr if it is an array. AD_DPTR is done in
   fix_datatype.
 */
static void
do_llvm_sym_is_refd(void)
{
  SPTR sptr;
  for (sptr = stb.firstusym; sptr < stb.stg_avail; ++sptr) {
    switch (STYPEG(sptr)) {
    case ST_VAR:
    case ST_ARRAY:
    case ST_STRUCT:
    case ST_UNION:
    case ST_PLIST:
      if (REFG(sptr) == 0) {
        switch (SCG(sptr)) {
        case SC_LOCAL:
        case SC_STATIC:
          sym_is_refd(sptr);
          break;
        default:
          break;
        }
      }
      break;
    default:
      break;
    }
  }
}
#endif

/**
   \brief ...
 */
void
stb_upper_init(void)
{
  int end;
  end = read_line();
  while (line[0] == 'i') {
    char *name, *cname, *filename;
    int level, which, namelen, cnamelen, filenamelen, base, size;
    long offset, objoffset;
    /* an 'inline' line */
    level = getval("inline");
    offset = getlval("offset");
    which = getval("which");
    cnamelen = getnamelen();
    cname = line + pos;
    pos += cnamelen;
    namelen = getnamelen();
    name = line + pos;
    pos += namelen;
    filenamelen = getnamelen();
    filename = line + pos;
    pos += filenamelen;
    objoffset = getlval("objoffset");
    base = getval("base");
    size = getval("size");
    name[namelen] = '\0';
    cname[cnamelen] = '\0';
    filename[filenamelen] = '\0';
    end = read_line();
  }

} /* upper_init */

SPTR
llvm_get_uplevel_newsptr(int oldsptr)
{
  SPTR sptr = symbolxref[oldsptr];
  if (SCG(sptr) == SC_BASED)
    sptr = MIDNUMG(sptr);
  return sptr;
}

static void
build_agoto(void)
{
  extern void exp_build_agoto(int *, int); /* exp_rte.c */
  if (agotosz == 0)
    return;
  exp_build_agoto(agototab, agotomax);
  FREE(agototab);
  agotosz = 0;
}

const char *
lookup_modvar_alias(SPTR sptr)
{
  alias_syminfo *node = modvar_alias_list;
  while (node) {
    if (node->sptr == sptr) {
      return node->alias;
    }
    node = node->next;
  }
  return NULL;
}

SPTR get_symbol_start(void) { return (SPTR)(oldsymbolcount + 1); }

/**
   \brief Given a alias name of a mod var sptr, create a new alias_syminfo node
   and add it to the linked list for later lookup.
 */
static void
save_modvar_alias(SPTR sptr, char *alias_name)
{
  alias_syminfo *new_alias_info;
  if (!alias_name || lookup_modvar_alias(sptr))
    return;
  NEW(new_alias_info, alias_syminfo, 1);
  new_alias_info->sptr = sptr;
  new_alias_info->alias = alias_name;
  new_alias_info->next = modvar_alias_list;
  modvar_alias_list = new_alias_info;
}

/**
   \brief Release the memory space ocupied by the linked list of alias_symifo nodes.
 */
static void
free_modvar_alias_list()
{
  alias_syminfo *node;
  while (modvar_alias_list) {
    node = modvar_alias_list;
    modvar_alias_list = modvar_alias_list->next;
    FREE(node->alias);
    FREE(node);
  }
}

