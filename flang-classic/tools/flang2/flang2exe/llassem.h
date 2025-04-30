/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef LLASSEM_H_
#define LLASSEM_H_

#include "gbldefs.h"
#include "global.h"
#include "symtab.h"
#include "llutil.h"

typedef struct DTLIST {
  LL_Type *lltype;
  int byval;

  /* XXX: sptr needs to go away, since fortran sptrs are only relevant for
   * the function being compiled.  This is for homing
   * (process_formal_arguments) support.  Which should only be called when
   * this sptr data is valid.
   */
  SPTR sptr;
  struct DTLIST *tail;
  struct DTLIST *next;
} DTLIST;

typedef struct uplevelpair {
  int oldsptr;  /* sptr from ilm file */
  SPTR newsptr; /* newsptr - from symbolxref[oldsptr] */
  int newmem;   /* sptr member of struct for newsptr */
} UPLEVEL_PAIR;

#define STACK_CAN_BE_32_BYTE_ALIGNED (aux.curr_entry->flags & 0x200)
#define ENFORCE_32_BYTE_STACK_ALIGNMENT (aux.curr_entry->flags |= 0x400)

#define IS_STABS (XBIT(120, 0x20))
#define ASMFIL gbl.asmfil

/** general length suitable for creating names from a symbol name during
    assembly, e.g., 1 for null, 3 for extra '_' , * 4 for @### with mscall */
#define MXIDLN (3 * MAXIDLEN + 10)

/**
 * structure to represent items being dinit'd -- used to generate
 * a sorted list of dinit items for a given common block or local
 * variable.
 */
typedef struct DSRT {
  SPTR sptr;         ///< sym being init'd (could be structure)
  ISZ_T offset;      ///< byte offset of item init'd
  int sectionindex;  ///< Fortran - section index
  long filepos;      ///< Fortran dinit file position for item's dinit(s)
  int func_count;    ///< Fortran save/restore func_count
  DTYPE dtype;       ///< used for C
  int len;           ///< used for C - character
  ISZ_T conval;      ///< used for C
  struct DSRT *next; ///< next in list (sorted in ascending offset)
  struct DSRT *ladd; ///< item last added - not used in C
} DSRT;

char *get_local_overlap_var(void);
char *put_next_member(char *ptr);
ISZ_T put_skip(ISZ_T old, ISZ_T New, bool is_char);

/*
 * macros to get and put DSRT pointers in symbol table entry - this
 * uses the XREF field
 */
#define DSRTG(s) ((DSRT *)get_getitem_p(XREFLKG(s)))
#define DSRTP(s, v) XREFLKP(s, put_getitem_p(v))

#define GET_DSRT (DSRT *)getitem(2, sizeof(DSRT))

/* structures and routines to process assembler globals for the entire file */

#define AG_HASHSZ 19
#define AG_SIZE(s) agb.s_base[s].size
#define AG_ALIGN(s) agb.s_base[s].align
#define AG_DSIZE(s) agb.s_base[s].dsize
#define AG_SYMLK(s) agb.s_base[s].symlk
#define AG_HASHLK(s) agb.s_base[s].hashlk
#define AG_NMPTR(s) agb.s_base[s].nmptr
#define AG_TYPENMPTR(s) agb.s_base[s].type_nmptr
#define AG_OLDNMPTR(s) agb.s_base[s].old_nmptr
#define AG_TYPEDESC(s) agb.s_base[s].typedesc /* Boolean */
#define AG_STYPE(s) agb.s_base[s].stype
#define AG_RET_LLTYPE(s) agb.s_base[s].ret_lltype
#define AG_LLTYPE(s) agb.s_base[s].lltype
#define AG_DTYPE(s) agb.s_base[s].dtype
#define AG_DTYPESC(s) agb.s_base[s].dtypesc
#define AG_SC(s) agb.s_base[s].sc /* Storage class */
#define AG_ALLOC(s) agb.s_base[s].alloc
#define AG_REF(s) agb.s_base[s].ref
#define AG_DEFD(s) agb.s_base[s].defd
#define AG_DEVICE(s) agb.s_base[s].device
#define AG_ISMOD(s) agb.s_base[s].ismod
#define AG_ISTLS(s) agb.s_base[s].istls
#define AG_NEEDMOD(s) agb.s_base[s].needmod
#define AG_ISCTOR(s) agb.s_base[s].ctor
#define AG_ISIFACE(s) agb.s_base[s].iface
#define AG_FINAL(s) agb.s_base[s].final
#define AG_UPLEVEL_AVL(s) agb.s_base[s].uplevel_avl
#define AG_UPLEVEL_SZ(s) agb.s_base[s].uplevel_sz
#define AG_UPLEVELPTR(s) agb.s_base[s].uplist
#define AG_UPLEVEL_OLD(s, i) agb.s_base[s].uplist[i].oldsptr
#define AG_UPLEVEL_NEW(s, i) agb.s_base[s].uplist[i].newsptr
#define AG_UPLEVEL_MEM(s, i) agb.s_base[s].uplist[i].newmem
#define AG_DLL(s) agb.s_base[s].dll
#define AG_NAME(s) agb.n_base + agb.s_base[s].nmptr
#define AG_TYPENAME(s) agb.n_base + agb.s_base[s].type_nmptr
#define AG_OLDNAME(s) agb.n_base + agb.s_base[s].old_nmptr
#define AG_ARGDTLIST(s) agb.s_base[s].argdtlist
#define AG_ARGDTLIST_LENGTH(s) agb.s_base[s].n_argdtlist
#define AG_ARGDTLIST_IS_VALID(s) agb.s_base[s].argdtlist_is_set
#define AG_OBJTODBGLIST(s) agb.s_base[s].cmblk_mem_mdnode_list
#define AG_CMBLKINITDATA(s) agb.s_base[s].cmblk_init_data

#define FPTR_HASHLK(s) fptr_local.s_base[s].hashlk
#define FPTR_IFACENMPTR(s) fptr_local.s_base[s].ifacenmptr
#define FPTR_IFACENM(s) fptr_local.n_base + fptr_local.s_base[s].ifacenmptr
#define FPTR_NMPTR(s) fptr_local.s_base[s].nmptr
#define FPTR_NAME(s) fptr_local.n_base + fptr_local.s_base[s].nmptr
#define FPTR_SYMLK(s) fptr_local.s_base[s].symlk

LL_Value *gen_ptr_offset_val(int, LL_Type *, const char *);

/**
   \brief llassem global symbol table entries
 */
typedef struct {
  ISZ_T size;  /**< max size of common block in file
                  if entry/proc, 1 => defd, 0 => proc */
  ISZ_T dsize; /**< size of common block when init'd */
  INT nmptr;
  INT type_nmptr;  /**< Used for external function */
  INT farg_nmptr;  /**< make all function that is not defined in same file
                      vararg with first argument specified if any */
  INT old_nmptr;   /**< Used for interface to keep original function name */
  INT align;       /**< alignment for BIND(C) variables */
  int symlk;       /**< used to link ST_CMBLK and ST_PROC */
  SPTR hashlk;     /**< hash collision field */
  int dtype;       /**< used for keep track dtype which is created for static/
                      bss area (only for AGL ag-local) */
  int dtypesc;     /**< dtype scope */
  int n_argdtlist; /**< Number of items in argdtlist */
  bool argdtlist_is_set; /**< Argdtlist has built, perhaps with 0 args */
  char stype;            /**< ST_ of global */
  char sc;               /**< SC_ of global */
  char alloc;            /**< ALLOCATABLE flag */
  char dll;              /**< DLL_NONE, DLL_EXPORT, DLL_IMPORT */
  LL_Type *lltype;       /**< LLVM representation of the ag symbol */
  LL_Type *ret_lltype;   /**< If this is a func this is the return type */
  DTLIST *argdtlist;     /**< linked listed of argument lltype */
  LL_ObjToDbgList *cmblk_mem_mdnode_list; ///< linked listed of cmem mdnode
  char* cmblk_init_data; /**< llvm reference of common block initialization data*/
  int uplevel_avl;
  int uplevel_sz;
  UPLEVEL_PAIR *uplist; /**< uplevel list for internal procecure */
  unsigned ref : 1;     /**< ST_PROC is referenced */
  unsigned defd : 1;    /**< module ST_CMBLK is defined in file */
  unsigned device : 1;  /**< CUDA device routine */
  unsigned ismod : 1;
  unsigned needmod : 1;
  unsigned ctor : 1;     /**< set if this routine has attribute constructor */
  unsigned typedesc : 1; /**< set if this variable is a type descriptor */
  unsigned iface : 1;    /**< set if this is part of interface */
  unsigned final : 1;    /**< set if this is final table */
  unsigned istls : 1;    /**< set if this is TLS */
} AG;

/**
   \brief storage allocation structure for assem's symtab
 */
typedef struct AGB_t {
  AG *s_base;   /**< pointer to table of common block nodes */
  int s_size;   /**< size of CM table */
  int s_avl;    /**< currently available entry */
  char *n_base; /**< pointer to names space */
  int n_size;
  int n_avl;
  SPTR hashtb[AG_HASHSZ];
} AGB_t;

extern AGB_t agb;

/** similar to AG struct but smaller */
typedef struct {
  INT nmptr;
  INT ifacenmptr;
  int hashlk;
  int symlk;
} FPTRSYM;

/** storage for function pointer */
typedef struct fptr_local_t {
  FPTRSYM *s_base;
  int s_size;
  int s_avl;
  char *n_base; /* pointer to names space */
  int n_size;
  int n_avl;
  int hashtb[AG_HASHSZ];
} fptr_local_t;

extern fptr_local_t fptr_local;

extern DSRT *lcl_inits;          /* head list of DSRT's for local variables */
extern DSRT *section_inits;      /* head list of DSRT's for initialized
                                           variables in named sections */
extern DSRT *extern_inits;       /* head list of DSRT's for BIND(C) module
                                           variables */
extern char static_name[MXIDLN]; /* name of STATIC area for a function */
extern int first_data;

struct sec_t {
  const char *name;
  int align;
};

/* ag entries */
extern int ag_cmblks;  /* list of common blocks in file */
extern int ag_procs;   /* list of procs in file */
extern int ag_other;   /* list of other externals in file */
extern int ag_global;  /* list of symbols that need to be declared
                                 global */
extern int ag_typedef; /* list of derived type that need to be
                                 declared  */
extern int ag_static;  /* keep name and type of static */
extern int ag_intrin;  /* intrinsic list generated by the bridge and
                                 has no sptr */
extern int ag_local;   /* dummy arguments which is a subroutine -
                                 need its signature and type */
extern int ag_funcptr; /* list of function pointer - should be a member
                                 of user defined type. Currently keep both
                                 LOCAL(any?) and STATIC in same list */

void put_i32(int);
void put_string_n(const char *, ISZ_T, int);
void put_short(int);
void put_int4(INT);

#if defined(TARGET_LLVM_X8664) || defined(TARGET_LLVM_POWER) || defined(TARGET_LLVM_ARM64)
#define DIR_LONG_SIZE 64
#else
#define DIR_LONG_SIZE 32
#endif

#define MAXARGLEN 256

void ll_override_type_string(LL_Type *llt, const char *str);
int alignment(DTYPE);
int add_member_for_llvm(int, int, DTYPE, ISZ_T);
LL_Type *update_llvm_typedef(DTYPE dtype, int sptr, int rank);
int llvm_get_unique_sym(void);
void align_func(void);
void put_global(char *name);
void put_func_name(int sptr);
void put_type(int sptr);
void init_huge_tlb(void);
void init_flushz(void);
void init_daz(void);
void init_ktrap(void);
ISZ_T get_socptr_offset(int);
#if defined(PG0CL)
#define llassem_end_func(ignore1, arg2) assem_end_func(arg2)
#else
#define llassem_end_func(arg1, arg2) lldbg_function_end(arg1, arg2)
#endif

LL_Type *make_generic_dummy_lltype(void);
LL_Type *get_local_overlap_vartype(void);

#ifdef OMP_OFFLOAD_LLVM
/**
   \brief ...
 */
void ompaccel_write_sharedvars(void);
#endif

/**
   \brief ...
 */
bool get_byval_from_argdtlist(DTLIST *argdtlist);

/**
   \brief ...
 */
bool has_typedef_ag(int gblsym);

/**
   \brief ...
 */
bool is_llvmag_entry(int gblsym);

/**
   \brief ...
 */
bool is_llvmag_iface(int gblsym);

/**
   \brief ...
 */
bool is_typedesc_defd(SPTR sptr);

/**
   \brief ...
 */
char *getaddrdebug(SPTR sptr);

/**
   \brief ...
 */
char *get_ag_name(int gblsym);

/**
   \brief ...
 */
char *get_ag_typename(int gblsym);

/**
   \brief ...
 */
DTLIST *get_argdtlist(int gblsym);

/**
   \brief return external name for procedure
 */
char *getextfuncname(SPTR sptr);

/**
   \brief ...
 */
char *get_llvm_name(SPTR sptr);

/**
   \brief ...
 */
char *get_main_progname(void);

/**
   \brief ...
 */
DTLIST *get_next_argdtlist(DTLIST *argdtlist);

/**
   \brief ...
 */
char *getsname(SPTR sptr);

/**
   \brief ...
 */
char *get_string_constant(int sptr);

/**
   \brief ...
 */
DTYPE get_ftn_typedesc_dtype(SPTR sptr);

/**
   \brief ...
 */
int add_ag_typename(int gblsym, const char *typeName);

/**
   \brief ...
 */
SPTR find_ag(const char *ag_name);

/**
   \brief ...
 */
int find_funcptr_name(SPTR sptr);

/**
   \brief ...
 */
int get_ag_argdtlist_length(int gblsym);

/**
   \brief ...
 */
SPTR get_ag(SPTR sptr);

/**
   \brief ...
 */
int get_bss_addr(void);

/**
   \brief ...
 */
SPTR get_dummy_ag(SPTR sptr);

/**
   \brief ...
 */
int get_hollerith_size(int sptr);

/**
   \brief ...
 */
SPTR get_intrin_ag(char *ag_name, DTYPE dtype);

/**
   \brief ...
 */
SPTR get_llvm_funcptr_ag(SPTR sptr, const char *ag_name);

/**
   \brief ...
 */
int get_private_size(void);

/**
   \brief ...
 */
SPTR get_sptr_from_argdtlist(DTLIST *argdtlist);

/**
   \brief ...
 */
int get_sptr_uplevel_address(int sptr);

/**
   \brief ...
 */
int get_stack_size(void);

/**
   \brief ...
 */
SPTR get_typedef_ag(const char *ag_name, const char *typeName);

/**
   \brief ...
 */
int get_uplevel_address_size(void);

/**
   \brief ...
 */
int has_valid_ag_argdtlist(int gblsym);

/**
   \brief determine if the address represented by \p syma, an address constant,
   is cache aligned.
 */
int is_cache_aligned(SPTR syma);

/**
   \brief ...
 */
int ll_shallow_copy_uplevel(SPTR hostsptr, SPTR olsptr);

/**
   Return the AG number associated to the local sptr value:
   1) Search local-fnptr-table of function pointers
   2) Get the ag name from (1)
   3) Get the gblsym using the ag name from (2)
   4) Return the AG gblsym from (3)
 */
SPTR local_funcptr_sptr_to_gblsym(SPTR sptr);

/**
   \brief ...
 */
DTYPE make_uplevel_arg_struct(void);

/**
   \brief the 32-byte alignment of the address constant \p acon_sptr
   \param acon_sptr
   \return the alignment or -1 if it's unknown.
 */
int runtime_32_byte_alignment(SPTR acon_sptr);

/**
   \brief determine the alignment of the address represented by syma, an address
   constant, within a cache-aligned container

   \return -1 if unknown or the byte boundary of the address

   For example, given a single precision quantity and a container which is
   16-byte aligned the following values are possible:
   \li 0   aligned with the beginning of the container.
   \li 4   multiple of 4 bytes from the beginning of the container.
   \li 8   multiple of 8 bytes from the beginning of the container.
   \li 12  multiple of 12 bytes from the beginning of the container.
 */
int runtime_alignment(SPTR syma);

/**
   \brief ...
 */
LL_ObjToDbgList **llassem_get_objtodbg_list(SPTR sptr);

/**
   \brief ...
 */
LL_Type *get_ag_lltype(int gblsym);

/**
   \brief ...
 */
LL_Type *get_ag_return_lltype(int gblsym);

/**
   \brief ...
 */
LL_Type *get_lltype_from_argdtlist(DTLIST *argdtlist);

/**
   \brief ...
 */
unsigned align_of_var(SPTR sptr);

/**
   If arg_num in [1-n] where 1 is the first argument passed and the function
   contains n arguments.  If arg_num is 0, the function's return value.

   Called by build_routine_parameters()
 */
void addag_llvm_argdtlist(SPTR gblsym, int arg_num, SPTR arg_sptr,
                          LL_Type *lltype);

/**
   \brief ...
 */
void add_aguplevel_oldsptr(void);

/**
   \brief ...
 */
void _add_llvm_uplevel_symbol(int oldsptr);

/**
   \brief ...
 */
void add_uplevel_to_host(int *ptr, int cnt);

/**
   \brief ...
 */
void arg_is_refd(int sptr);

/**
   \brief ...
 */
void assem_begin_func(SPTR sptr);

/**
   \brief ...
 */
void assemble_end(void);

/**
   \brief ...
 */
void assemble_init(int argc, char *argv[], const char *cmdline);

/**
   \brief ...
 */
void assemble(void);

/**
   \brief ...
 */
void assem_data(void);

/**
   \brief ...
 */
void assem_dinit(void);

/**
   \brief ...
 */
void assem_emit_align(int n);

/**
   \brief ...
 */
void assem_emit_file_line(int findex, int lineno);

/**
   \brief ...
 */
void assem_emit_line(int findex, int lineno);

/**
   \brief ...
 */
void assem_end(void);

/**
   \brief ...
 */
void assem_init(void);

/**
   \brief ...
 */
void assem_put_linux_trace(int sptr);

/**
   \brief ...
 */
void create_static_base(int num);

/**
   \brief ...
 */
void create_static_name(char *name, int usestatic, int num);

/**
   \brief ...
 */
void deleteag_llvm_argdtlist(int gblsym);

/**
   \brief ...
 */
void fix_equiv_locals(SPTR loc_list, ISZ_T loc_addr);

/**
   \brief ...
 */
void fix_equiv_statics(SPTR loc_list, ISZ_T loc_addr, bool dinitflg);

/**
   \brief ...
 */
void fix_private_sym(int sptr);

/**
   \brief ...
 */
void _fixup_llvm_uplevel_symbol(void);

/**
   \brief ...
 */
void hostsym_is_refd(SPTR sptr);

/**
   \brief ...
 */
void llvm_funcptr_store(SPTR sptr, char *ag_name);

/**
   \brief ...
 */
void load_uplevel_addresses(SPTR display_temp);

/**
   \brief ...
 */
void put_fstr(SPTR sptr, int add_null);

/**
   \brief ...
 */
void put_section(int sect);

/**
   \brief ...
 */
void set_ag_argdtlist_is_valid(int gblsym);

/**
   \brief ...
 */
void set_ag_lltype(int gblsym, LL_Type *llt);

/**
   \brief ...
 */
void set_ag_return_lltype(int gblsym, LL_Type *llt);

/**
   \brief ...
 */
void set_bss_addr(int size);

/**
   \brief ...
 */
void set_llvmag_entry(int gblsym);

/**
   \brief ...
 */
void set_llvm_iface_oldname(int gblsym, char *nm);

/**
   \brief ...
 */
void set_private_size(ISZ_T sz);

/**
   \brief ...
 */
void sym_is_refd(SPTR sptr);

/**
   \brief Writes libomptarget related initialization.
 */
void write_libomtparget(void);

#endif
