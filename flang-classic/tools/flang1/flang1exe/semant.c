/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
    \file
    \brief This file contains part 1 of the compiler's semantic actions
    (also known as the semant1 phase).
*/

#include "gbldefs.h"
#include "gramsm.h"
#include "gramtk.h"
#include "error.h"
#include "global.h"
#include "symtab.h"
#include "symutl.h"
#include "dtypeutl.h"
#include "semant.h"
#include "scan.h"
#include "dinit.h"
#include "semstk.h"
#include "ast.h"
#include "pragma.h"
#include "rte.h"
#include "pd.h"
#include "interf.h"
#include "fdirect.h"
#include "fih.h"
#include "ccffinfo.h" /* for setfile */

#include "atomic_common.h"


static void gen_dinit(int, SST *);
static void pop_subprogram(void);

static void fix_proc_ptr_dummy_args();
static void set_len_attributes(SST *, int);
static void set_char_attributes(int, int *);
static void set_aclen(SST *, int, int);
static void copy_type_to_entry(int);
static void save_host(INTERF *);
static void restore_host(INTERF *, LOGICAL);
static void do_end_subprogram(SST *, RU_TYPE);
static void check_end_subprogram(RU_TYPE, int);
static const char *name_of_rutype(RU_TYPE);
static void convert_intrinsics_to_idents(void);
static int chk_intrinsic(int, LOGICAL, LOGICAL);
static int create_func_entry(int);
static int create_func_entry_result(int);
static int create_var(int);
static int chk_func_entry_result(int);
static void get_param_alias_const(SST *, int, int);
static void set_string_type_from_init(int, ACL *);
static void fixup_param_vars(SST *, SST *);
static void save_typedef_init(int, int);
static void symatterr(int, int, const char *);
static void fixup_function_return_type(int, int);
static void get_retval_KIND_value();
static void get_retval_LEN_value();
static void get_retval_derived_type();
static void init_allocatable_typedef_components(int);

static int chk_kind_parm(SST *);
static int get_kind_parm(int, int);
static int get_kind_parm_strict(int, int);
static int get_len_parm(int, int);
static int has_kind_parm_expr(int, int, int);
static void chk_initialization_with_kind_parm(int);
static void check_kind_type_param(int dtype);
static void defer_put_kind_type_param(int, int, char *, int, int, int);
static void replace_sdsc_in_bounds(int sdsc, ADSC *ad, int i);
static int replace_sdsc_in_ast(int sdsc, int ast);
static void chk_new_param_dt(int, int);
static int get_vtoff(int, DTYPE);
#ifdef FLANG_SEMANT_UNUSED
static int has_length_type_parameter(int);
#endif
static int get_highest_param_offset(int);
static ACL *dup_acl(ACL *src, int sptr);
static int match_memname(int sptr, int list);
static LOGICAL is_pdt_dtype(DTYPE dtype);
static int chk_asz_deferlen(int, int);

static int ident_host_sub = 0;
static void defer_ident_list(int ident, int proc);
static void clear_ident_list();
static void decr_ident_use(int ident, int proc);
static void check_duplicate(bool checker, const char * op);
#ifdef GSCOPEP
static void fixup_ident_bounds(int);
#endif

static int decl_procedure_sym(int sptr, int proc_interf_sptr, int attr);
static int setup_procedure_sym(int sptr, int proc_interf_sptr, int attr,
                               char access);
static LOGICAL ignore_common_decl(void);
static void record_func_result(int func_sptr, int func_result_sptr,
                               LOGICAL in_ENTRY);
static bool bindingNameRequiresOverloading(SPTR sptr);
static void clear_iface(int i, SPTR iface);
static bool do_fixup_param_vars_for_derived_arrays(bool, SPTR, int);
static void gen_unique_func_ast(int ast, SPTR sptr, SST *stkptr);

static IFACE *iface_base;
static int iface_avail;
static int iface_size;

static IDENT_LIST *ident_base[HASHSIZE];
static LOGICAL dirty_ident_base = FALSE;

static STSK *stsk; /* gen_dinit() defines, semant1() uses */
static LOGICAL seen_implicit;
static LOGICAL seen_parameter;
static LOGICAL craft_intrinsics;
static LOGICAL is_entry;
static LOGICAL is_exe_stmt;
static LOGICAL entry_seen;
static LOGICAL seen_options;
static struct {
  int kind;
  INT len;
  int propagated;
} lenspec[2];

#define _LEN_CONST 1
#define _LEN_ASSUM 2
#define _LEN_ZERO 3
#define _LEN_ADJ 4
#define _LEN_DEFER 5

/** \brief Subprogram prefix struct defintions for RECURESIVE, PURE,
           IMPURE, ELEMENTAL, and MODULE. 
 */
static struct subp_prefix_t {
  bool recursive;  /** processing RECURSIVE attribute */
  bool pure;       /** processing PURE attribute */
  bool impure;     /** processing IMPURE attribute */
  bool elemental;  /** processing ELEMENTAL attribute */
  bool module;     /** processing MODULE attribute */
} subp_prefix;

static void clear_subp_prefix_settings(struct subp_prefix_t *);
static void check_module_prefix();

static int mscall;
static int cref;
static int nomixedstrlen;
static int next_enum;

/* for non array parameters, default set by attributes of the function
 */
#define BYVALDEFAULT(ffunc) \
  (!(PASSBYREFG(ffunc)) &&  \
   (PASSBYVALG(ffunc) | STDCALLG(ffunc) | CFUNCG(ffunc)))

/* flag indicating the presence of a 'host' for contained subprograms. Values
 * are selected so that they can be used as a mask to determine when an
 * IMPLICIT NONE statement has already been specified:
 *
 * 0x02 - no host present (module or top level subprogram)
 * 0x04 - host present (within a module CONTAINed subprogram)
 * 0x08 - host present (within a CONTAINed subprogram in another subprogram)
 */
static int host_present;
static INTERF host_state;
static int end_of_host;

#define ERR310(s1, s2) error(310, 3, gbl.lineno, s1, s2)
/*
 * Declarations for processing the attributes specified in an entity type
 * declaration.  Note that some of the ET_ manifest constants, as well as
 * the entity_attr struct, are used in other processing, such as for PROCEDURE
 * attributes;  likewise, there are a few ET_ entries that aren't used
 * for declarations, but are for PROCEDURE attributes such as PASS/NOPASS.
 */
#define ET_ACCESS 0
#define ET_ALLOCATABLE 1
#define ET_DIMENSION 2
#define ET_EXTERNAL 3
#define ET_INTENT 4
#define ET_INTRINSIC 5
#define ET_OPTIONAL 6
#define ET_PARAMETER 7
#define ET_POINTER 8 
#define ET_SAVE 9
#define ET_TARGET 10
#define ET_AUTOMATIC 11
#define ET_STATIC 12
#define ET_BIND 13
#define ET_VALUE 14
#define ET_VOLATILE 15
#define ET_PASS 16
#define ET_NOPASS 17
#define ET_DEVICE 18
#define ET_PINNED 19
#define ET_SHARED 20
#define ET_CONSTANT 21
#define ET_PROTECTED 22
#define ET_ASYNCHRONOUS 23
#define ET_TEXTURE 24
#define ET_KIND 25
#define ET_LEN 26
#define ET_CONTIGUOUS 27
#define ET_MANAGED 28
#define ET_IMPL_MANAGED 29
#define ET_MAX 30

/* derive bit mask for each entity type */

#define ET_B(e) (1 << e)

#define SYMI_SPTR(i) aux.symi_base[i].sptr
#define SYMI_NEXT(i) aux.symi_base[i].next

/*
 * structure to record which attributes occurred for an entity type
 * declaration.
 */
static LOGICAL in_entity_typdcl; /* TRUE if processing an entity type decl */
static struct {
  int exist;     /* bit vector indicating which attributes exist */
  int dimension; /* TY_ARRAY DT record */
  char access;   /* 'u' => access public ; 'v' => access private */
  char intent;   /* bit vector formed from INTENT_... */
  char bounds[sizeof(sem.bounds)]; /* copy of sem.bounds[...] */
  char arrdim[sizeof(sem.arrdim)]; /* copy of sem.arrdim */
  int pass_arg;                    /* sptr of the ident in PASS ( <ident> ) */
} entity_attr;

static struct {
  const char *name;
  int no; /* bit vector of attributes which do not coexist */
} et[ET_MAX] = {
    {"access",
     ~(ET_B(ET_ALLOCATABLE) | ET_B(ET_DIMENSION) | ET_B(ET_EXTERNAL) |
       ET_B(ET_INTRINSIC) | ET_B(ET_PARAMETER) | ET_B(ET_POINTER) |
       ET_B(ET_SAVE) | ET_B(ET_TARGET) | ET_B(ET_BIND) | ET_B(ET_VALUE) |
       ET_B(ET_VOLATILE) | ET_B(ET_ASYNCHRONOUS) | ET_B(ET_PROTECTED) |
       ET_B(ET_DEVICE) | ET_B(ET_CONSTANT) | ET_B(ET_PINNED) |
       ET_B(ET_MANAGED) | ET_B(ET_IMPL_MANAGED) | ET_B(ET_CONTIGUOUS))},
    {"allocatable",
     ~(ET_B(ET_ACCESS) | ET_B(ET_DIMENSION) | ET_B(ET_SAVE) | ET_B(ET_TARGET) |
       ET_B(ET_INTENT) | ET_B(ET_OPTIONAL) | ET_B(ET_VOLATILE) |
       ET_B(ET_DEVICE) | ET_B(ET_PINNED) | ET_B(ET_ASYNCHRONOUS) |
       ET_B(ET_PROTECTED) | ET_B(ET_MANAGED) | ET_B(ET_IMPL_MANAGED) |
       ET_B(ET_CONTIGUOUS))},
    {"dimension",
     ~(ET_B(ET_ACCESS) | ET_B(ET_ALLOCATABLE) | ET_B(ET_INTENT) |
       ET_B(ET_OPTIONAL) | ET_B(ET_PARAMETER) | ET_B(ET_POINTER) |
       ET_B(ET_SAVE) | ET_B(ET_TARGET) | ET_B(ET_BIND) | ET_B(ET_VALUE) |
       ET_B(ET_VOLATILE) | ET_B(ET_DEVICE) | ET_B(ET_SHARED) | ET_B(ET_PINNED) |
       ET_B(ET_CONSTANT) | ET_B(ET_ASYNCHRONOUS) | ET_B(ET_PROTECTED) |
       ET_B(ET_TEXTURE) | ET_B(ET_CONTIGUOUS) | ET_B(ET_MANAGED) |
       ET_B(ET_IMPL_MANAGED))},
    {"external",
     ~(ET_B(ET_ACCESS) | ET_B(ET_OPTIONAL) | ET_B(ET_BIND) | ET_B(ET_VALUE) |
       ET_B(ET_POINTER))},
    {"intent",
     ~(ET_B(ET_DIMENSION) | ET_B(ET_OPTIONAL) | ET_B(ET_TARGET) |
       ET_B(ET_ALLOCATABLE) | ET_B(ET_BIND) | ET_B(ET_VALUE) |
       ET_B(ET_POINTER) | ET_B(ET_VOLATILE) | ET_B(ET_DEVICE) |
       ET_B(ET_CONSTANT) | ET_B(ET_PINNED) |
       ET_B(ET_SHARED | ET_B(ET_ASYNCHRONOUS) | ET_B(ET_PROTECTED)) |
       ET_B(ET_CONTIGUOUS) | ET_B(ET_TEXTURE) | ET_B(ET_MANAGED) |
       ET_B(ET_IMPL_MANAGED))},
    {"intrinsic", ~(ET_B(ET_ACCESS))},
    {"optional",
     ~(ET_B(ET_DIMENSION) | ET_B(ET_EXTERNAL) | ET_B(ET_INTENT) |
       ET_B(ET_POINTER) | ET_B(ET_SAVE) | ET_B(ET_TARGET) |
       ET_B(ET_ALLOCATABLE) | ET_B(ET_VOLATILE) | ET_B(ET_ASYNCHRONOUS) |
       ET_B(ET_PROTECTED) | ET_B(ET_CONTIGUOUS) | ET_B(ET_MANAGED) |
       ET_B(ET_VALUE) | ET_B(ET_IMPL_MANAGED) | ET_B(ET_DEVICE))},
    {"parameter",
     ~(ET_B(ET_ACCESS) | ET_B(ET_DIMENSION) | ET_B(ET_SAVE) | ET_B(ET_VALUE) |
       ET_B(ET_ASYNCHRONOUS) | ET_B(ET_CONSTANT))},
    {"pointer",
     ~(ET_B(ET_ACCESS) | ET_B(ET_DIMENSION) | ET_B(ET_OPTIONAL) |
       ET_B(ET_SAVE) | ET_B(ET_VALUE) | ET_B(ET_BIND) | ET_B(ET_INTENT) |
       ET_B(ET_VOLATILE) | ET_B(ET_ASYNCHRONOUS) | ET_B(ET_PROTECTED) |
       ET_B(ET_TEXTURE) | ET_B(ET_DEVICE) | ET_B(ET_CONTIGUOUS) |
       ET_B(ET_MANAGED) | ET_B(ET_EXTERNAL))},
    {"save",
     ~(ET_B(ET_ACCESS) | ET_B(ET_ALLOCATABLE) | ET_B(ET_DIMENSION) |
       ET_B(ET_PARAMETER) | ET_B(ET_POINTER) | ET_B(ET_TARGET) |
       ET_B(ET_VALUE) | ET_B(ET_VOLATILE) | ET_B(ET_SHARED) |
       ET_B(ET_ASYNCHRONOUS) | ET_B(ET_PROTECTED) | ET_B(ET_PINNED) |
       ET_B(ET_TEXTURE) | ET_B(ET_DEVICE) | ET_B(ET_MANAGED) |
       ET_B(ET_CONTIGUOUS) | ET_B(ET_IMPL_MANAGED))},
    {"target",
     ~(ET_B(ET_ACCESS) | ET_B(ET_ALLOCATABLE) | ET_B(ET_DIMENSION) |
       ET_B(ET_INTENT) | ET_B(ET_OPTIONAL) | ET_B(ET_SAVE) | ET_B(ET_VALUE) |
       ET_B(ET_BIND) | ET_B(ET_PINNED) | ET_B(ET_VOLATILE) |
       ET_B(ET_ASYNCHRONOUS) | ET_B(ET_PROTECTED) | ET_B(ET_CONTIGUOUS) |
       ET_B(ET_DEVICE) | ET_B(ET_MANAGED) | ET_B(ET_IMPL_MANAGED))},
    {"automatic",
     ~(ET_B(ET_ALLOCATABLE) | ET_B(ET_DIMENSION) | ET_B(ET_POINTER) |
       ET_B(ET_TARGET) | ET_B(ET_VALUE) | ET_B(ET_VOLATILE) |
       ET_B(ET_ASYNCHRONOUS) | ET_B(ET_PROTECTED))},
    {"static",
     ~(ET_B(ET_ACCESS) | ET_B(ET_ALLOCATABLE) | ET_B(ET_DIMENSION) |
       ET_B(ET_POINTER) | ET_B(ET_SAVE) | ET_B(ET_TARGET) | ET_B(ET_BIND) |
       ET_B(ET_VALUE) | ET_B(ET_VOLATILE) | ET_B(ET_ASYNCHRONOUS) |
       ET_B(ET_PROTECTED))},
    {"bind",
     ~(ET_B(ET_ACCESS) | ET_B(ET_DIMENSION) | ET_B(ET_EXTERNAL) |
       ET_B(ET_INTENT) | ET_B(ET_POINTER) | ET_B(ET_TARGET) | ET_B(ET_STATIC) |
       ET_B(ET_VOLATILE) | ET_B(ET_ASYNCHRONOUS) | ET_B(ET_PROTECTED) |
       ET_B(ET_CONTIGUOUS))},
    {"value",
     ~(ET_B(ET_ACCESS) | ET_B(ET_DIMENSION) | ET_B(ET_EXTERNAL) |
       ET_B(ET_INTENT) | ET_B(ET_PARAMETER) | ET_B(ET_POINTER) | ET_B(ET_SAVE) |
       ET_B(ET_TARGET) | ET_B(ET_STATIC) | ET_B(ET_ASYNCHRONOUS) |
       ET_B(ET_OPTIONAL) | ET_B(ET_PROTECTED) | ET_B(ET_CONTIGUOUS))},
    {"volatile",
     ~(ET_B(ET_ACCESS) | ET_B(ET_ALLOCATABLE) | ET_B(ET_DIMENSION) |
       ET_B(ET_INTENT) | ET_B(ET_OPTIONAL) | ET_B(ET_POINTER) | ET_B(ET_SAVE) |
       ET_B(ET_TARGET) | ET_B(ET_AUTOMATIC) | ET_B(ET_STATIC) | ET_B(ET_BIND) |
       ET_B(ET_ASYNCHRONOUS) | ET_B(ET_PROTECTED) | ET_B(ET_DEVICE) |
       ET_B(ET_SHARED) | ET_B(ET_CONTIGUOUS))},
    {"pass", ~(0)},
    {"nopass", ~(0)},
    {"device",
     ~(ET_B(ET_ALLOCATABLE) | ET_B(ET_DIMENSION) | ET_B(ET_INTENT) |
       ET_B(ET_VOLATILE) | ET_B(ET_ACCESS) | ET_B(ET_TARGET) |
       ET_B(ET_POINTER) | ET_B(ET_TEXTURE) | ET_B(ET_CONTIGUOUS) |
       ET_B(ET_OPTIONAL) | ET_B(ET_SAVE) | ET_B(ET_IMPL_MANAGED))},
    {"pinned",
     ~(ET_B(ET_ALLOCATABLE) | ET_B(ET_DIMENSION) | ET_B(ET_INTENT) |
       ET_B(ET_SAVE) | ET_B(ET_TARGET) | ET_B(ET_ACCESS) | ET_B(ET_CONTIGUOUS) |
       ET_B(ET_IMPL_MANAGED))},
    {"shared",
     ~(ET_B(ET_DIMENSION) | ET_B(ET_SAVE) | ET_B(ET_INTENT) |
       ET_B(ET_VOLATILE))},
    {"constant", ~(ET_B(ET_DIMENSION) | ET_B(ET_INTENT) | ET_B(ET_ACCESS) |
       ET_B(ET_PARAMETER))},
    {"protected",
     ~(ET_B(ET_ACCESS) | ET_B(ET_ALLOCATABLE) | ET_B(ET_DIMENSION) |
       ET_B(ET_INTENT) | ET_B(ET_OPTIONAL) | ET_B(ET_POINTER) | ET_B(ET_SAVE) |
       ET_B(ET_TARGET) | ET_B(ET_AUTOMATIC) | ET_B(ET_STATIC) | ET_B(ET_BIND) |
       ET_B(ET_VALUE) | ET_B(ET_VOLATILE) | ET_B(ET_ASYNCHRONOUS) |
       ET_B(ET_CONTIGUOUS) | ET_B(ET_IMPL_MANAGED))},
    {"asynchronous",
     ~(ET_B(ET_ACCESS) | ET_B(ET_ALLOCATABLE) | ET_B(ET_DIMENSION) |
       ET_B(ET_INTENT) | ET_B(ET_OPTIONAL) | ET_B(ET_PARAMETER) |
       ET_B(ET_POINTER) | ET_B(ET_SAVE) | ET_B(ET_TARGET) | ET_B(ET_AUTOMATIC) |
       ET_B(ET_STATIC) | ET_B(ET_BIND) | ET_B(ET_VALUE) | ET_B(ET_VOLATILE) |
       ET_B(ET_PROTECTED) | ET_B(ET_IMPL_MANAGED))},
    {"texture",
     ~(ET_B(ET_DIMENSION) | ET_B(ET_INTENT) | ET_B(ET_POINTER) |
       ET_B(ET_DEVICE) | ET_B(ET_SAVE))},
    {"kind", 0},       /* 'no' field not used, so make it 0 */
    {"len", 0},        /* 'no' field not used, so make it 0 */
    {"contiguous", 0}, /* 'no' field not used, so make it 0 */
    {"managed",
     ~(ET_B(ET_ALLOCATABLE) | ET_B(ET_DIMENSION) | ET_B(ET_INTENT) |
       ET_B(ET_SAVE) | ET_B(ET_TARGET) | ET_B(ET_ACCESS) | ET_B(ET_CONTIGUOUS) |
       ET_B(ET_OPTIONAL) | ET_B(ET_POINTER) | ET_B(ET_IMPL_MANAGED))},
    {"implicit-managed", 0}, /* 'no' field not used */
};
/*
 * Declarations for processing the attributes specified in a DEC ATTRIBUTES
 * declaration.
 */
#define DA_ALIAS 0
#define DA_C 1
#define DA_STDCALL 2
#define DA_DLLEXPORT 3
#define DA_DLLIMPORT 4
#define DA_VALUE 5
#define DA_REFERENCE 6
#define DA_DECORATE 7
#define DA_NOMIXEDSLA 8
#define DA_MAX 9

/* derive bit mask for each attribute type */

#define DA_B(e) (1 << e)

/*
 * structure to record which attributes occurred for a DEC ATTRIBUTES
 * and BIND declaration.
 */
struct dec_attr_t {
  int exist;   /* bit vector indicating which attributes exist */
  int altname; /* sptr to a character constant representing alias */
};

static struct dec_attr_t dec_attr;
static struct dec_attr_t bind_attr;

static struct {
  const char *name;
  int no; /* bit vector of attributes which do not coexist */
          /* unlike the et[...].no values, it's easier to explicitly
           * specify those which do not coexist as opposed to the
           * negation of those which can coexist.
           */
} da[DA_MAX] = {
    {"alias", 0},
    {"c", (DA_B(DA_STDCALL))},
    {"stdcall", (DA_B(DA_C))},
    {"dllexport", (0)},
    {"dllimport", (0)},
    {"value", (DA_B(DA_REFERENCE))},
    {"reference", (DA_B(DA_VALUE))},
    {"decorate", 0},
    {"nomixed_str_len_arg", 0},
};

static void process_bind(int);

static void defer_iface(int, int, int, int);
static void do_iface(int);
static void do_iface_module(void);
static void _do_iface(int, int);
static void fix_iface(int);
static void fix_iface0();

/** \brief Initialize semantic analyzer for new user subprogram unit.
 */
void
semant_init(int noparse)
{
  if (!noparse) {
    if (sem.doif_base == NULL) {
      sem.doif_size = 12;
      NEW(sem.doif_base, DOIF, sem.doif_size);
    }
    sem.doif_depth = 0;
    DI_ID(0) = -1;
    DI_NEST(0) = 0;
    DI_LINENO(0) = 0;
    if (sem.stsk_base == NULL) {
      sem.stsk_size = 12;
      NEW(sem.stsk_base, STSK, sem.stsk_size);
    }
    sem.block_scope = 0;
    sem.doconcurrent_symavl = SPTR_NULL;
    sem.doconcurrent_dtype = DT_NONE;
    sem.stsk_depth = 0;
    scopestack_init();
    sem.eqvlist = 0;
    sem.eqv_avail = 1;
    if (sem.eqv_size == 0) {
      sem.eqv_size = 20;
      NEW(sem.eqv_base, EQVV, sem.eqv_size);
    }
    sem.eqv_ss_avail = 1;
    if (sem.eqv_ss_size == 0) {
      sem.eqv_ss_size = 50;
      NEW(sem.eqv_ss_base, int, sem.eqv_ss_size);
    }
    EQV_NUMSS(0) = 0;
    sem.non_private_avail = 0;
    if (sem.non_private_size == 0) {
      sem.non_private_size = 50;
      NEW(sem.non_private_base, int, sem.non_private_size);
    }
    if (sem.typroc_base == NULL) {
      sem.typroc_size = 50;
      NEW(sem.typroc_base, int, sem.typroc_size);
    }
    sem.typroc_avail = 0;
    if (sem.iface_base == NULL) {
      sem.iface_size = 50;
      NEW(sem.iface_base, IFACE, sem.iface_size);
    }
    sem.iface_avail = 0;
    sem.pgphase = PHASE_INIT;
    sem.flabels = 0; /* not NOSYM - a sym's SYMLK is init'd to NOSYM. if
                      * its SYMLK is NOSYM, then it hasn't been added */
    sem.nml = NOSYM;
    sem.atemps = 0;
    sem.itemps = 0;
    sem.ptemps = 0;
    sem.savall = flg.save;
    sem.savloc = FALSE;
    sem.autoloc = FALSE;
    sem.psfunc = FALSE;
    sem.in_stfunc = FALSE;
    sem.dinit_error = FALSE;
    sem.dinit_data = FALSE;
    sem.equal_initializer = false;
    sem.proc_initializer = false;
    sem.dinit_nbr_inits = 0;
    sem.contiguous = XBIT(125, 0x80000); /* xbit is set for -Mcontiguous */
    seen_implicit = FALSE;
    symutl.none_implicit = sem.none_implicit = flg.dclchk;
    seen_parameter = FALSE;
  }

  flg.sequence = TRUE;
  flg.hpf = FALSE;

  if (!noparse) {
    sem.ignore_stmt = FALSE;
    sem.switch_avl = 0;
    if (switch_base == NULL) {
      sem.switch_size = 400;
      NEW(switch_base, SWEL, sem.switch_size);
    }
    sem.temps_reset = FALSE;
    seen_options = FALSE;
    sem.gdtype = -1;
    lenspec[0].kind = 0;
    sem.seql.type = 0;    /* [NO]SEQUENCE not yet seen */
    sem.seql.next = NULL; /* sequence list is empty */
    sem.dtemps = 0;
    sem.interface = 0;
    if (sem.interf_base == NULL) {
      sem.interf_size = 2;
      NEW(sem.interf_base, INTERF, sem.interf_size);
    }
    sem.p_dealloc = NULL;
    sem.p_dealloc_delete = NULL;
    sem.alloc_std = 0;
    clear_subp_prefix_settings(&subp_prefix);
    sem.accl.type = 0;    /* PUBLIC/PRIVATE statement not yet seen */
    sem.accl.next = NULL; /* access list is empty */
    sem.in_struct_constr = 0;
    sem.atomic[0] = sem.atomic[1] = sem.atomic[2] = FALSE;
    sem.master.cnt = 0;
    sem.critical.cnt = 0;
    sem.intent_list = NULL;
    sem.symmetric = FALSE;
    sem.mpaccatomic.seen = sem.mpaccatomic.pending = sem.mpaccatomic.apply =
        sem.mpaccatomic.is_acc = FALSE;
    sem.mpaccatomic.ast = 0;
    sem.mpaccatomic.action_type = ATOMIC_UNDEF;
    sem.mpaccatomic.mem_order = MO_UNDEF;
    sem.mpaccatomic.rmw_op = AOP_UNDEF;
    sem.mpaccatomic.accassignc = 0;
    sem.parallel = 0;
    sem.task = 0;
    sem.orph = 0;
    sem.target = 0;
    sem.teams = 0;
    sem.expect_do = FALSE;
    sem.expect_simd_do = FALSE;
    sem.expect_dist_do = FALSE;
    sem.expect_acc_do = 0;
    sem.collapsed_acc_do = 0;
    sem.seq_acc_do = 0;
    sem.expect_cuf_do = 0;
    sem.close_pdo = FALSE;
    sem.is_hpf = FALSE;
    sem.hpfdcl = 0;
    sem.ssa_area = 0;
    sem.use_etmps = FALSE;
    sem.etmp_list = NULL;
    sem.auto_dealloc = NULL;
    sem.blksymnum = 0;
    sem.ignore_default_none = FALSE;
    sem.in_enum = FALSE;
    sem.type_mode = 0;
    sem.seen_import = FALSE;
    sem.seen_end_module = FALSE;
    sem.tbp_arg = 0;
    sem.tbp_arg_cnt = 0;
    sem.tbp_access_stmt = 0;
    sem.generic_tbp = 0;
    sem.auto_finalize = NULL;
    sem.type_initialize = NULL;
    sem.alloc_mem_initialize = NULL;
    sem.select_type_seen = 0;
    sem.param_offset = 0;
    sem.kind_type_param = 0;
    sem.len_type_param = 0;
    sem.type_param_candidate = 0;
    sem.len_candidate = 0;
    sem.kind_candidate = 0;
    sem.type_param_sptr = 0;
    sem.param_struct_constr = 0;
    sem.new_param_dt = 0;
    sem.extends = 0;
    sem.param_assume_sz = 0;
    sem.param_defer_len = 0;
    sem.save_aconst = 0;
    sem.defined_io_type = 0;
    sem.defined_io_seen = 0;
    sem.use_seen = 0;
    sem.ieee_features = FALSE;
    sem.collapse = sem.collapse_depth = 0;
    sem.stats.allocs = 0;
    sem.stats.nodes = 0;
    sem.modhost_proc = 0;
    sem.modhost_entry = 0;
    sem.array_const_level = 0;
    sem.ac_std_range = NULL;
    sem.elp_stack = NULL;
    sem.parsing_operator = false;

    mscall = 0;
    cref = 0;
    nomixedstrlen = 0;
#if defined(TARGET_WIN)
    if (WINNT_CALL)
      mscall = 1;
    if (WINNT_CREF)
      cref = 1;
    if (WINNT_NOMIXEDSTRLEN)
      nomixedstrlen = 1;
#endif
  } else {
    /*
     * Needed for handling the 03 allocatable semantics in semutil2.c via
     * transform which might occur during the IPA recompile.
     */
    sem.p_dealloc = NULL;
    sem.p_dealloc_delete = NULL;
  }

  sem.sc = SC_LOCAL;
  stb.curr_scope = 0;
  ast_init();           /* ast.c */
  init_intrinsic_opr(); /* semgnr.c */
  import_init();        /* interf.c */
  if (!noparse) {
    if (IN_MODULE) {
      mod_init();
      host_present = 0x04;
      restore_implicit();
      save_implicit(TRUE);
    } else if (gbl.internal) { /* hasn't been incremented yet */
      host_present = 0x08;
      restore_implicit();
      save_implicit(TRUE);
    } else {
      host_present = 0x02;
    }
  }
  clean_struct_default_init(stb.stg_avail);
  use_init();    /* module.c */
  bblock_init(); /* bblock.c */

  if (!noparse) {
    craft_intrinsics = FALSE;

    if (XBIT(49, 0x1040000))
      /* T3D/T3E or C90 Cray targets */
      change_predefineds(ST_CRAY, FALSE);

    end_of_host = 0;
    if (gbl.internal && sem.which_pass)
      restore_host_state(2);
  } else {
    if (gbl.internal)
      restore_host_state(4);
  }
}

/* for each SC_DUMMY parameter that is passed by value,
   copy it to a local (reference ) of the same name.
   all lookups will subsequently find this local
 */
static void
reloc_byvalue_parameters()
{
  INT dpdsc;
  INT psptr;
  INT iarg;
  INT newsptr;
  ITEM *itemp; /* Pointers to items */
  int byval_default = 0;
  int thesub;

  if (STYPEG(gbl.currsub) == ST_MODULE)
    return;

  for (thesub = gbl.currsub; thesub > NOSYM; thesub = SYMLKG(thesub)) {
    dpdsc = DPDSCG(thesub);
    for (iarg = PARAMCTG(thesub); iarg > 0; dpdsc++, iarg--) {
      psptr = *(aux.dpdsc_base + dpdsc);

      /* copy all parameters passed by value to local stack.
         arrays are always passed by reference  unless specifically
         marked by value
       */
      /* disable array and struct parameters passed by value */
      if (((DTY(DTYPEG(psptr))) == TY_ARRAY) ||
          ((DTY(DTYPEG(psptr))) == TY_STRUCT)) {
        if (PASSBYVALG(thesub) || PASSBYVALG(psptr))
          error(84, 3, gbl.lineno, SYMNAME(psptr),
                "- VALUE derived types and arrays not yet supported");
      } else
        byval_default = BYVALDEFAULT(thesub);
      if (PASSBYVALG(psptr) && OPTARGG(psptr)) {
        /* an address is passed for optional value arguments as if call by
         * reference, but the address is of a temp
         */
        continue;
      }
      if ((byval_default || PASSBYVALG(psptr)) && (!PASSBYREFG(psptr)) &&
          (DTY(DTYPEG(psptr)) != TY_ARRAY) &&
          /* don't redo what we've already done */
          (strncmp(SYMNAME(psptr), "_V_", 3) != 0)) {

        /* declare a new variable _V_<orig_name> which subsumes the
         * original by value parameter.  The original variable becomes
         * SC_LOCAL and all further user code references will be to this
         * SC_LOCAL var.
         * The copy of the by-value _V_<name> parameter to this local
         * is done at expand time.
         */
        newsptr = lookupsymf("_V_%s", SYMNAME(psptr));
        if (newsptr > NOSYM) {
          /* already exists */
          *(aux.dpdsc_base + dpdsc) = newsptr; /* fix the DPDSC entry */
          return;
        }
        newsptr = getsymf("_V_%s", SYMNAME(psptr));
        dup_sym(newsptr, stb.stg_base + psptr); /* also _V_... is the dummy*/
        DCLDP(newsptr, TRUE);                 /* so DCLCHK is quiet */
        REFP(newsptr, TRUE);
        SCP(psptr, SC_LOCAL); /* make the original a local*/
        /* the byval flag on the original arg (psptr) is cleared in semfin */
        MIDNUMP(newsptr, psptr); /* link from new symbol to original symbol */
        *(aux.dpdsc_base + dpdsc) = newsptr; /* fix the DPDSC entry */
        for (itemp = sem.intent_list; itemp != NULL; itemp = itemp->next) {
          if (psptr == itemp->t.sptr) {
            itemp->t.sptr = newsptr;
            break;
          }
        }
        /*
         * The original symbol may not yet be classified as an object.
         * Take care of that here for the original symbol; semfin will
         * take of new symbol.
         */
        switch (STYPEG(psptr)) {
        case ST_UNKNOWN:
        case ST_IDENT:
          STYPEP(psptr, ST_VAR);
          break;
        default:;
        }
        if (sem.which_pass) {
          /* the back-end will always copy _V_<orig_name> to
           * <orig_name>; make sure that <orig_name> is referenced.
           */
          sym_is_refd(psptr);
        }

      } /* if pass by val */

      else if (thesub != gbl.currsub && SCG(psptr) == SC_LOCAL) {
        /* presumably, thesub is an ST_ENTRY and the parameter has
         * already been processed; make sure to fix the DPDSC entry.
         */
        newsptr = lookupsymf("_V_%s", SYMNAME(psptr));
        if (newsptr) {
          *(aux.dpdsc_base + dpdsc) = newsptr; /* fix the DPDSC entry */
        }
      }

    } /* for  all parameters */
  }
}

static void
end_subprogram_checks()
{
  if (sem.master.cnt)
    error(155, 3, sem.master.lineno, "Unterminated MASTER", CNULL);
  if (sem.critical.cnt)
    error(155, 3, sem.critical.lineno, "Unterminated CRITICAL", CNULL);
  sem_err104(sem.doif_depth, DI_LINENO(sem.doif_depth), "unterminated");
} /* end_subprogram_checks */

static int restored = 0;

/** \brief Semantic actions - part 1.
    \param rednum reduction number
    \param top    top of stack after reduction
 */
void
semant1(int rednum, SST *top)
{
  SPTR sptr, sptr1, sptr2, block_sptr, sptr_temp, lab;
  int dtype, dtypeset, ss, numss;
  int stype, stype1, i;
  int begin, end, count;
  int opc;
  int std;
  INT rhstop, rhsptr;
  LOGICAL inited;
  ITEM *itemp, /* Pointers to items */
      *itemp1;
  INT conval;
  int doif;
  int evp;
  ADSC *ad;
  char *np, *np2; /* char ptrs to symbol names area */
  int name_prefix_char;
  VAR *ivl;        /* Initializer Variable List */
  ACL *ict, *ict1; /* Initializer Constant Tree */
  int ast, alias;
  static int et_type; /* one of ET_...; '<attr>::=' passes up */
  int et_bitv;
  LOGICAL no_init; /* init not allowed for entity decl */
  int func_result; /* sptr of ident in result ( ident ) */
  ACL *aclp;
  ACCL *accessp;
  int gnr;
  LOGICAL is_array;
  LOGICAL is_member;
  INT val[2];
  int constarraysize; /* set to 1 if array bounds are constant */
  ISZ_T arraysize;    /* the actual array size; check for < 0 */
  static int da_type; /* one of DA_...; '<msattr>::=' passes up */
  PHASE_TYPE prevphase;
  INT id_name;
  INT result_name;
  int construct_name;
  SST *e1;
  static int proc_interf_sptr; /* <proc interf ::= <id> passed up */
  /* for deepcopy */
  int symi;

  switch (rednum) {

  /* ------------------------------------------------------------------ */
  /*
   *      <SYSTEM GOAL SYMBOL> ::=
   */
  case SYSTEM_GOAL_SYMBOL1:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<stmt> ::= <stbeg> <statement> <stend>
   */
  case STMT1:
    /*
     * `!DIR$ ALIGN alignment` pragma should only take effect within the
     * scope of the statement, so flang1 need to clear the flg.x[251] here.
     */
    flg.x[251] = 0;
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<stbeg> ::=
   */
  case STBEG1:
    if (sem.in_enum) {
      switch (scn.stmtyp) {
      case TK_ENUMERATOR:
      case TK_ENDENUM:
        break;
      default:
        error(155, 3, gbl.lineno, "ENUMERATOR statement expected", CNULL);
        sem.ignore_stmt = TRUE;
        break;
      }
    }
    sem.is_hpf = scn.is_hpf;
    sem.alloc_std = 0;
    sem.p_dealloc_delete = NULL;
    if (sem.pgphase == PHASE_USE) {
      switch (scn.stmtyp) {
      case TK_USE:
      case TK_INCLUDE:
        break;
      default:
        apply_use_stmts();
        if (sem.deferred_func_kind) {
          get_retval_KIND_value();
        }
        if (sem.deferred_func_len) {
          get_retval_LEN_value();
        }
        if (sem.deferred_dertype) {
          get_retval_derived_type();
        }
        break;
      }
    }
    if (sem.pgphase == 0 && sem.interface && gbl.currsub == 0) {
      if (scn.stmtyp == TK_USE) {
        error(155, 3, gbl.lineno, "USE", "is not in a correct position.");
        sem.ignore_stmt = TRUE;
      }
    }
    if (sem.deferred_func_kind && (sem.pgphase > PHASE_USE || is_exe_stmt)) {
      get_retval_KIND_value();
    }
    if (sem.deferred_func_len && (sem.pgphase > PHASE_USE || is_exe_stmt)) {
      get_retval_LEN_value();
    }
    if (sem.deferred_dertype && (sem.pgphase > PHASE_USE || is_exe_stmt)) {
      get_retval_derived_type();
    }

    if (!sem.interface && sem.pgphase < PHASE_EXEC &&
        (is_exe_stmt = is_executable(sem.tkntyp)) && !sem.block_scope) {

      if (!IN_MODULE) 
        do_iface(0); 
      else
        do_iface_module();

      reloc_byvalue_parameters();
      if (sem.which_pass == 1 && restored == 0) {
        restore_internal_subprograms();
        restored = 1;
      }
    }
    if (sem.expect_do || sem.expect_acc_do || sem.expect_simd_do ||
        sem.expect_dist_do || (sem.expect_cuf_do && XBIT(137, 0x20000))) {
      int stt;
      stt = sem.tkntyp;
      if (stt == TK_NAMED_CONSTRUCT)
        stt = get_named_stmtyp();
      if (stt != TK_DO) {
        const char *p;
        switch (DI_ID(sem.doif_depth)) {
        case DI_ACCDO:
          sem.doif_depth--; /* remove from stack */
          p = "ACC DO";
          break;
        case DI_ACCLOOP:
          sem.doif_depth--; /* remove from stack */
          p = "ACC LOOP";
          break;
        case DI_ACCREGDO:
          sem.doif_depth--; /* remove from stack */
          p = "ACC REGION DO";
          break;
        case DI_ACCREGLOOP:
          sem.doif_depth--; /* remove from stack */
          p = "ACC REGION LOOP";
          break;
        case DI_ACCKERNELSDO:
          sem.doif_depth--; /* remove from stack */
          p = "ACC KERNELS DO";
          break;
        case DI_ACCKERNELSLOOP:
          sem.doif_depth--; /* remove from stack */
          p = "ACC KERNELS LOOP";
          break;
        case DI_ACCPARALLELDO:
          sem.doif_depth--; /* remove from stack */
          p = "ACC PARALLEL DO";
          break;
        case DI_ACCPARALLELLOOP:
          sem.doif_depth--; /* remove from stack */
          p = "ACC PARALLEL LOOP";
          break;
        case DI_ACCSERIALLOOP:
          sem.doif_depth--; /* remove from stack */
          p = "ACC SERIAL LOOP";
          break;
        case DI_CUFKERNEL:
          sem.doif_depth--; /* remove from stack */
          p = "CUDA KERNEL DO";
          break;
        case DI_PDO:
          if (DI_ISSIMD(sem.doif_depth))
            p = "OMP DO SIMD";
          else
            p = "OMP DO";
          sem.doif_depth--; /* remove PDO from stack */
          par_pop_scope();
          break;
        case DI_TARGETSIMD:
          sem.doif_depth--; /* remove from TARGET SIMD stack */
          p = "OMP TARGET SIMD";
          par_pop_scope();
          break;
        case DI_SIMD:
          sem.doif_depth--; /* remove from SIMD stack */
          p = "OMP SIMD";
          par_pop_scope();
          break;

        case DI_DISTRIBUTE:
          sem.doif_depth--; /* remove from DISTRIBUTE stack */
          p = "OMP DISTRIBUTE";
          par_pop_scope();
          break;
        case DI_TARGPARDO:
          sem.doif_depth--; /* remove from TARGET PARALLEL DO stack */
          p = "OMP TARGET PARALLEL DO";
          par_pop_scope();
          break;
        case DI_DISTPARDO:
          sem.doif_depth--; /* remove from stack */
          p = "OMP DISTRIBUTE PARALLEL DO";
          par_pop_scope();

          if (scn.stmtyp == TK_MP_ENDTEAMS) {
            /* distribute parallel do */
            break;
          } else if (scn.stmtyp == TK_MP_ENDTARGET) {
            /* teams distribute parallel do */
            par_pop_scope();
          } else if (DI_ID(sem.doif_depth) == DI_TEAMS) {
            /* if the previous stack id is DI_TEAMS
             * and scn.stmtyp != TK_MP_ENDTEAMS, then
             * this is target teams distribute parallel do
             * construct: pop teams and target as we manually
             * add stack for those.
             */
            par_pop_scope();
            par_pop_scope();
          }

          break;
        case DI_DOACROSS:
          p = "DOACROSS";
          goto reset_st;
        case DI_PARDO:
          if (DI_ISSIMD(sem.doif_depth))
            p = "PARALLEL DO SIMD";
          else
            p = "PARALLEL DO";
        reset_st:
          sem.doif_depth--; /* remove from stack */
          /* restore symbol table state */
          par_pop_scope();
          break;
        case DI_TASKLOOP:
          sem.doif_depth--; /* remove from stack */
          p = "OMP TASKLOOP";
          par_pop_scope();
          break;
        default:
          p = "???";
          break;
        }
        error(155, 3, gbl.lineno, "DO loop expected after", p);
        sem.expect_do = FALSE;
        sem.expect_simd_do = FALSE;
        sem.expect_dist_do = FALSE;
        sem.expect_acc_do = 0;
        sem.collapsed_acc_do = 0;
        sem.seq_acc_do = 0;
        sem.expect_cuf_do = 0;
        sem.collapse = sem.collapse_depth = 0;
      }
    } else if (sem.collapse_depth) {
      int stt;
      stt = sem.tkntyp;
      if (stt == TK_NAMED_CONSTRUCT)
        stt = get_named_stmtyp();
      if (stt != TK_DO) {
        /*
         * The collapse value is larger than the number of loops;
         * this needs to be a fatal error since the DOIF stack
         * is probably inconsistent wrt matching ENDDOs etc.
         */
        error(155, 4, gbl.lineno, "DO loop expected after", "COLLAPSE");
        sem.collapse = sem.collapse_depth = 0;
      }
    }
    if (sem.close_pdo) {
      sem.close_pdo = FALSE;
      switch (DI_ID(sem.doif_depth)) {
      case DI_PDO:
        if (scn.stmtyp != TK_MP_ENDPDO) {
          if (A_TYPEG(STD_AST(STD_PREV(0))) != A_MP_BARRIER)
            (void)add_stmt(mk_stmt(A_MP_BARRIER, 0));
          sem.doif_depth--; /* pop DOIF stack */
        }
        /* else ENDPDO pops the stack */
        break;
      case DI_DISTRIBUTE:
        if (scn.stmtyp != TK_MP_ENDDISTRIBUTE) {
          sem.doif_depth--; /* pop DOIF stack */
        }
        /* else ENDDISTRIBUTE pops the stack */
        break;
      case DI_TEAMSDIST:
        if (scn.stmtyp != TK_MP_ENDTEAMSDIST) {
          sem.doif_depth--; /* pop DOIF stack */
          end_teams();
        }
        /* else ENDTEAMSDIST pops the stack */
        break;
      case DI_TARGTEAMSDIST:
        if (scn.stmtyp != TK_MP_ENDTARGTEAMSDIST) {
          sem.doif_depth--; /* pop DOIF stack */
          end_teams();
          end_target();
        }
        /* else ENDTEAMSDIST pops the stack */
        break;
      case DI_TARGPARDO:
        if (scn.stmtyp != TK_MP_ENDTARGPARDO) {
          (void)add_stmt(mk_stmt(A_MP_BARRIER, 0));
          sem.doif_depth--; /* pop DOIF stack */
          end_target();
        }
        /* else ENDTARGPARDO[SIMD] pops the stack */
        break;

      case DI_TEAMSDISTPARDO:
        if (scn.stmtyp != TK_MP_ENDTEAMSDISTPARDO &&
            scn.stmtyp != TK_MP_ENDTEAMSDISTPARDOSIMD) {
          sem.doif_depth--; /* pop DOIF stack */
          end_teams();
        }
        /* else ENDTEAMSDISTPARDO[SIMD] pops the stack */
        break;
      case DI_TARGTEAMSDISTPARDO:
        if (scn.stmtyp != TK_MP_ENDTARGTEAMSDISTPARDO &&
            scn.stmtyp != TK_MP_ENDTARGTEAMSDISTPARDOSIMD) {
          sem.doif_depth--; /* pop DOIF stack */
          end_teams();
          end_target();
        }
        /* else ENDTARGTEAMSDISTPARDO[SIMD] pops the stack */
        break;
      case DI_DISTPARDO:
        if (scn.stmtyp != TK_MP_ENDDISTPARDO &&
            scn.stmtyp != TK_MP_ENDDISTPARDOSIMD) {
          sem.doif_depth--; /* pop DOIF stack */
        }
        break;
      case DI_TARGETSIMD:
        if (scn.stmtyp != TK_MP_ENDTARGSIMD) {
          sem.doif_depth--; /* pop DOIF stack */
          end_target();
        }
        /* else ENDTARGETSIMD pops the stack */
        break;
      case DI_SIMD:
        if (scn.stmtyp != TK_MP_ENDSIMD) {
          sem.doif_depth--; /* pop DOIF stack */
        }
        /* else ENDSIMD pops the stack */
        break;
      case DI_DOACROSS:
        /* the DOIF stack could have been popped when the
         * DO loop was closed, but it's done here with
         * the other DO directives.  */
        sem.doif_depth--; /* pop DOIF stack */
        break;
      case DI_PARDO:
        if (scn.stmtyp != TK_MP_ENDPARDO) {
          sem.doif_depth--; /* pop DOIF stack */
          /* else ENDPARDO pops the stack */
        }
        break;
      case DI_TASKLOOP:
        if (scn.stmtyp != TK_MP_ENDTASKLOOP) {
          sem.doif_depth--; /* pop DOIF stack */
          /* else ENDTASKLOOP pops the stack */
        }
        break;
      default:
        break;
      }
    }
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<stend> ::=
   */
  case STEND1:
    if (sem.pgphase >= PHASE_EXEC) {
     if (sem.atomic[0]) {
        sem.atomic[0] = sem.atomic[1] = sem.atomic[2] = FALSE;
        error(155, 3, gbl.lineno,
              "Statement after ATOMIC UPDATE is not an assignment", CNULL);
      } else {
        sem.atomic[0] = sem.atomic[1];
        sem.atomic[1] = FALSE;
      }
      if (sem.mpaccatomic.pending &&
          sem.mpaccatomic.action_type != ATOMIC_CAPTURE) {
        error(155, 3, gbl.lineno,
              "Statement after ATOMIC UPDATE is not an assignment", CNULL);
      }
      if (sem.mpaccatomic.seen &&
          sem.mpaccatomic.action_type != ATOMIC_CAPTURE) {
        if ((!sem.mpaccatomic.is_acc && use_opt_atomic(sem.doif_depth))) {
         ;
        } else {
          if (sem.mpaccatomic.is_acc)
            sem.mpaccatomic.seen = FALSE;
          sem.mpaccatomic.pending = TRUE;
        }
      }
    }
    freearea(0); /* free ITEM list areas */
    sem.new_param_dt = 0;
    sem.param_offset = 0;
    sem.kind_type_param = 0;
    sem.len_type_param = 0;
    sem.type_param_candidate = 0;
    sem.len_candidate = 0;
    sem.kind_candidate = 0;
    sem.type_param_sptr = 0;
    sem.param_struct_constr = 0;
    sem.save_aconst = 0;
    sem.tbp_arg = 0;
    sem.tbp_arg_cnt = 0;
    sem.extends = 0;
    if (sem.select_type_seen > 1) {
      error(155, 3, gbl.lineno,
            "Only a CLASS IS, TYPE IS, CLASS DEFAULT, or END SELECT"
            " statement may follow a SELECT TYPE statement",
            CNULL);
    } else if (sem.select_type_seen == 1) {
      sem.select_type_seen = 2;
    } else {
      sem.select_type_seen = 0;
    }
    if (flg.smp && sem.doif_base && sem.doif_depth &&
        DI_ID(sem.doif_depth) != DI_SELECT_TYPE)
      check_no_scope_sptr();
    entity_attr.access = ' '; /* Need to reset entity access */
    sem.parsing_operator = false;
    sem.equal_initializer = false;
    sem.proc_initializer = false;
    sem.array_const_level = 0;
    sem.ac_std_range = NULL;
    sem.elp_stack = NULL;
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <statement> ::= <prog title>  |
   */
  case STATEMENT1:
    prevphase = sem.pgphase;
    sem.gdtype = -1;
    lenspec[0].kind = 0;
    /*if( sem.which_pass == 1 )
        restore_internal_subprograms();*/
    restored = 0;
    goto statement_shared;
  /*
   *      <statement> ::= <nii> <nim> <entry statement> |
   */
  case STATEMENT2:
    prevphase = sem.pgphase;
    SST_ASTP(LHS, SST_ASTG(RHS(3)));
    goto statement_shared;
  /*
   *      <statement> ::= <declaration> |
   */
  case STATEMENT3:
    sem.class = 0;
    prevphase = sem.pgphase;
    if (scn.stmtyp == TK_IMPLICIT) {
      if (sem.pgphase > PHASE_IMPLICIT)
        errsev(70);
      else
        sem.pgphase = PHASE_IMPLICIT;
    } else if (scn.stmtyp == TK_DATA || scn.stmtyp == TK_NAMELIST) {
      if (sem.pgphase > PHASE_EXEC)
        errsev(70);
      else if (sem.pgphase < PHASE_SPEC)
        sem.pgphase = PHASE_SPEC;
    } else if (scn.stmtyp == TK_INTERFACE || scn.stmtyp == TK_ABSTRACT) {
      sem.pgphase = PHASE_INIT;
      prevphase = PHASE_INIT;
    } else if (scn.stmtyp == TK_PARAMETER) {
      if (sem.pgphase > PHASE_SPEC)
        errsev(70);
      else if (sem.pgphase < PHASE_IMPLICIT)
        sem.pgphase = PHASE_IMPLICIT;
    } else if (scn.stmtyp == TK_USE) {
      if (sem.pgphase > PHASE_USE)
        errsev(70);
      else if (sem.pgphase < PHASE_USE)
        sem.pgphase = PHASE_USE;
    } else if (scn.stmtyp == TK_IMPORT) {
      if (sem.pgphase > PHASE_IMPORT)
        errsev(70);
      else if (sem.pgphase < PHASE_IMPORT)
        sem.pgphase = PHASE_IMPORT;
    } else {
      if (sem.pgphase > PHASE_SPEC)
        errsev(70);
/* allow for routine before a use statement */
      /* allow for attributes before a use statement */
      else if (scn.stmtyp != TK_ATTRIBUTES && scn.stmtyp != TK_MP_DECLARESIMD)
        sem.pgphase = PHASE_SPEC;
    }
    sem.gdtype = -1;
    lenspec[0].kind = 0;
    goto statement_shared;
  /*
   *      <statement> ::= <nii> <nim> <simple stmt> |
   */
  case STATEMENT4:
    prevphase = sem.pgphase;
    SST_ASTP(LHS, SST_ASTG(RHS(3)));
    goto statement_shared;
  /*
   *      <statement> ::= <nii> <nim> <GOTO stmt>   |
   */
  case STATEMENT5:
    prevphase = sem.pgphase;
    SST_ASTP(LHS, SST_ASTG(RHS(3)));
    goto executable_shared;
  /*
   *      <statement> ::= <nii> <nim> <control stmt> |
   */
  case STATEMENT6:
    prevphase = sem.pgphase;
    SST_ASTP(LHS, SST_ASTG(RHS(3)));
    goto executable_shared;
  /*
   *      <statement> ::= <nii> <nim> <block stmt> |
   */
  case STATEMENT7:
    prevphase = sem.pgphase;
    goto statement_end;
  /*
   *      <statement> ::= <nii> <nim> <format stmt>  |
   */
  case STATEMENT8:
    prevphase = sem.pgphase;
    if (sem.pgphase == PHASE_INIT)
      sem.pgphase = PHASE_HEADER;
    /*
     * Allow semant ccsym vars allocated by get_temp to be re-used for
     * the next statement, if necessary:
     */
    sem.temps_reset = FALSE;
    SST_ASTP(LHS, SST_ASTG(RHS(3)));
    if (SST_ASTG(LHS)) /* TBD: delete this and next stmt */
      (void)add_stmt((int)SST_ASTG(LHS));
    goto statement_end;
  /*
   *	<statement> ::= <null stmt> |
   */
  case STATEMENT9:
    prevphase = sem.pgphase;
    if (scn.currlab) {
      errlabel(18, 3, gbl.lineno, SYMNAME(scn.currlab),
               "- must be followed by a keyword or an identifier");
      ast = mk_stmt(A_CONTINUE, 0);
      SST_ASTP(LHS, ast);
      DEFDP(scn.currlab, 1);
      goto executable_shared;
    }
    SST_ASTP(LHS, 0); /* don't change sem.pgphase */
    break;
  /*
   *      <statement> ::= <end> <end stmt>     |
   */
  case STATEMENT10:
    /*
     * Initialize AST field since an A_END is not generated for the end
     * of a host subprogram containing internal procedures
     */
    prevphase = sem.pgphase;
    if (!sem.interface && sem.pgphase < PHASE_EXEC) {
      reloc_byvalue_parameters();
      if (sem.which_pass == 1 && restored == 0) {
        restore_internal_subprograms();
        restored = 1;
      }
    }
    SST_ASTP(LHS, 0);
    if (sem.interface) {
      if ((gnr = sem.interf_base[sem.interface - 1].generic)) {
        if (GTYPEG(gnr) && gbl.rutype == RU_SUBR) {
          error(155, 3, gbl.lineno, "Generic INTERFACE with the same name as a "
                                    "derived type may only contain functions -",
                SYMNAME(gbl.currsub));
          GTYPEP(gnr, 0);
        }
        if (GNCNTG(gnr) == 0)
          sem.interf_base[sem.interface - 1].gnr_rutype = gbl.rutype;
        else if (sem.interf_base[sem.interface - 1].gnr_rutype &&
                 sem.interf_base[sem.interface - 1].gnr_rutype != gbl.rutype) {

           errWithSrc(155, 3, SST_LINENOG(RHS(2)),
                   "Generic INTERFACE may not mix functions and subroutines",
                   CNULL, SST_COLUMNG(RHS(2)), 0, false, CNULL);
        }

        if (gbl.currsub)
          add_overload(gnr, gbl.currsub);
      } else if ((gnr = sem.interf_base[sem.interface - 1].operator)) {
        if (sem.interf_base[sem.interface - 1].opval == OP_ST) {
          if (gbl.rutype != RU_SUBR)
            error(155, 3, gbl.lineno,
                  "Assignment INTERFACE requires subroutines -",
                  SYMNAME(gbl.currsub));
          else if (PARAMCTG(gbl.currsub) != 2)
            error(155, 3, gbl.lineno,
                  "Assignment INTERFACE requires subroutines 2 arguments -",
                  SYMNAME(gbl.currsub));
        } else {
          if (gbl.rutype != RU_FUNC)
            error(155, 3, gbl.lineno, "Operator INTERFACE requires functions -",
                  SYMNAME(gbl.currsub));
          else if (PARAMCTG(gbl.currsub) != 1 && PARAMCTG(gbl.currsub) != 2)
            error(
                155, 3, gbl.lineno,
                "Operator INTERFACE requires functions with 1 or 2 arguments -",
                SYMNAME(gbl.currsub));
        }
        add_overload(gnr, gbl.currsub);
      }
      if (gbl.currsub)
        pop_subprogram();
      break;
    }

    if (gbl.rutype == RU_BDATA) {
      /* error if executable statements in block data: */
      if (sem.pgphase > PHASE_SPEC)
        errsev(71);
    } else if (!end_of_host && SST_IDG(RHS(2))) {
      chk_adjarr(); /* any extra code for adjustable arrays */
      end_subprogram_checks();
    }
    /*
     * The END statement may be for a module or subprogram.  If a
     * subprogram, the end AST is generated and semfin() is called.
     * If the end of a module, there are two cases:
     * 1.  only specifications were seen (i.e., no contained subprograms);
     *     since the module blockdata will be output, the end AST needs
     *     to be generated, however, semfin() can't be called.
     * 2.  module subprograms were present; the module blockdata was
     *     already written when the CONTAINS was seen; no END ast is
     *     necessary; semfin() still can't be called.
     */
    if (gbl.currsub || gbl.rutype == RU_PROG)
      SST_ASTP(LHS, mk_stmt(A_END, 0));
    if (SST_IDG(RHS(2))) /* end of subprogram */
      sem.pgphase = PHASE_END;
    else
      sem.pgphase = PHASE_END_MODULE; /* end of module */
    goto statement_shared;
  /*
   *      <statement> ::= <empty file>
   */
  case STATEMENT11:
    prevphase = sem.pgphase;
    goto statement_end;
  /*
   *	<statement> ::= INCLUDE <quoted string>
   */
  case STATEMENT12:
    prevphase = sem.pgphase;
    sptr = SST_SYMG(RHS(2));
    scan_include(stb.n_base + CONVAL1G(sptr));
    goto statement_end;
  /*
   *	<statement> ::= <nii> <nim> OPTIONS |
   *           [stuff that follows OPTIONS is not parsed - hidden by scanner]
   */
  case STATEMENT13:
    prevphase = sem.pgphase;
    if (flg.standard)
      error(171, 2, gbl.lineno, "OPTIONS", CNULL);
    if (sem.pgphase != PHASE_INIT || seen_options)
      errsev(70);
    else {
      scan_options();
      seen_options = TRUE;
    }
    goto statement_end;
  /*
   *	<statement> ::= <nis> <nii> CONTAINS |
   */
  case STATEMENT14:
    prevphase = sem.pgphase;
    SST_ASTP(LHS, 0);
    /*do_iface(0);*/
    reloc_byvalue_parameters();
    if (sem.pgphase >= PHASE_CONTAIN)
      errsev(70);
    sem.pgphase = PHASE_CONTAIN;
    if (gbl.currsub) {
      /* internal subprogram context */
      if (gbl.rutype == RU_BDATA) {
        errsev(70);
        goto executable_shared;
      }
      if (gbl.internal) {
        error(155, 3, gbl.lineno, "Internal subprograms may not be nested",
              CNULL);
        goto executable_shared;
      }
      convert_intrinsics_to_idents();
      save_host(&host_state);
      gbl.internal = 1;
      if (sem.which_pass == 0)
        gbl.empty_contains = FALSE;
      restore_host(&host_state, TRUE);
      if (sem.which_pass == 0) {
        /*
         * when first processing an internal procedure within a module
         * subprogram, need to save the state of the host which will be
         * restored for subsequent internal procedures within the same
         * module subprogram.  Note that the scanner ensures that the
         * end statement of the internal procedure in this context
         * (processing a module the first time) does not terminate
         * compilation (scn.end_program_unit is FALSE).
         */
        save_host_state(0x3);
        sem.pgphase = PHASE_INIT;
        SST_ASTP(LHS, 0);
      } else {
        chk_adjarr(); /* any extra code for adjustable arrays */
        end_subprogram_checks();
        fix_class_args(gbl.currsub);
        save_host_state(0x11);
        /*
         * When the CONTAINS is seen, ensure that an END ast is
         * generated for the host subprogram.
         * Note that scan has set 'scn.end_program_unit to TRUE'.
         */
        if (sem.end_host_labno && sem.which_pass) {
          int labsym = getsymf(".L%05ld", (long)sem.end_host_labno);
          /*
           * If a label was present on the end statement of the
           * host subprogram, need to define & emit the label now.
           */
          int lab = declref(labsym, ST_LABEL, 'd');
          if (DEFDG(lab))
            errlabel(97, 3, 0, SYMNAME(labsym), CNULL);
          else
            scn.currlab = lab;
          L3FP(lab, 1); /* HACK - disable errorcheck in scan.c*/
        }
        SST_ASTP(LHS, mk_stmt(A_END, 0));
      }
      sem.end_host_labno = 0;
      goto statement_shared;
    }
    if (IN_MODULE) {
      if (ANCESTORG(gbl.currmod) && !HAS_SMP_DECG(ANCESTORG(gbl.currmod)))
        error(1210, ERR_Severe, gbl.lineno, 
              SYMNAME(ANCESTORG(gbl.currmod)), CNULL); 
      fe_save_state();
      begin_contains();
      sem.pgphase = PHASE_INIT;
      /*
       * When the CONTAINS is seen, emit a blockdata just in case any
       * data statements are seen; ensure that an END ast is generated.
       * Note that scan has set 'scn.end_program_unit to TRUE'.
       */
      SST_ASTP(LHS, mk_stmt(A_END, 0));
      goto statement_shared;
    }
    errsev(70);
    goto executable_shared;
  /*
   *	<statement> ::= <directive>
   */
  case STATEMENT15:
    prevphase = sem.pgphase;
    if (sem.interface == 0) {
      ast = mk_comstr(scn.directive);
      (void)add_stmt(ast);
    }
    goto statement_end;

  executable_shared:
    sem.pgphase = PHASE_EXEC;
    sem.temps_reset = FALSE;
  /* fall thru to 'statement_shared' */

  statement_shared:

    if ((ast = SST_ASTG(LHS))) {
      (void)add_stmt(ast);
      SST_ASTG(LHS) = 0;
    }
    sem.dinit_error = FALSE;
    gen_deallocate_arrays();

   if (sem.atomic[2]) {
      ast = mk_stmt(A_ENDATOMIC, 0);
      (void)add_stmt(ast);
      sem.atomic[0] = sem.atomic[2] = FALSE;
    }
    if (sem.mpaccatomic.apply &&
        sem.mpaccatomic.action_type != ATOMIC_CAPTURE) {
      int ecs;
      sem.mpaccatomic.apply = FALSE;
      if (!sem.mpaccatomic.is_acc) {
        if (use_opt_atomic(sem.doif_depth)) {
          ecs = mk_stmt(A_MP_ENDATOMIC, 0);
          add_stmt(ecs);
        } else {
          ecs = emit_bcs_ecs(A_MP_ENDCRITICAL);
          /* point to each other */
          A_LOPP(ecs, sem.mpaccatomic.ast);
          A_LOPP(sem.mpaccatomic.ast, ecs);
        }
        sem.mpaccatomic.ast = 0;
      } else {
        int ast_atomic;
        ast_atomic = mk_stmt(A_ENDATOMIC, 0);
        add_stmt(ast_atomic);
        A_LOPP(ast_atomic, sem.mpaccatomic.ast);
        A_LOPP(sem.mpaccatomic.ast, ast_atomic);
        sem.mpaccatomic.ast = 0;
      }
    }
    /*
     * If the current statement is labeled and we are inside a DO [WHILE|
     * CONCURRENT] loop, search to see if this statement ends the loop.
     *
     * OpenMP ARB interpretations version 1.0:
     * If a do loop nest which shares the same termination statement is
     * followed by an ENDDO or ENDPARALLEL, the DO or PARALLEL DO can
     * only be specified for the outermost DO.
     */
    if (scn.currlab != 0 && sem.doif_depth > 0) {
      int par_type = 0; /* nonzero => par do needs to be closed */
      for (doif = sem.doif_depth; doif > 0; --doif) {
        if ((DI_ID(doif) == DI_DO || DI_ID(doif) == DI_DOWHILE ||
             DI_ID(doif) == DI_DOCONCURRENT) &&
             DI_DO_LABEL(doif) == scn.currlab) {
          switch (par_type) {
          /*
           * If a parallel do appears between two do loops sharing the
           * same termination statement, close the parallel do now.
           * (The innermost do loop is the parallel do.)
           */
          case DI_PDO:
          case DI_TARGETSIMD:
          case DI_SIMD:
          case DI_DISTRIBUTE:
          case DI_DISTPARDO:
          case DI_DOACROSS:
          case DI_PARDO:
          case DI_TASKLOOP:
          case DI_ACCDO:
          case DI_ACCLOOP:
          case DI_CUFKERNEL:
            sem.close_pdo = FALSE;
            --sem.doif_depth;
            par_type = 0;
          }
          do_end(DI_DOINFO(doif));
          if (sem.which_pass)
            direct_loop_end(DI_LINENO(doif), gbl.lineno);
          par_type = DI_ID(sem.doif_depth);
        }
      }
    }

    /* For END statements clean up end of program unit. */
    if (sem.pgphase == PHASE_END) {
      if (!end_of_host) {
        semfin();
        if (IN_MODULE && sem.interface == 0)
          mod_end_subprogram_two();
        if (sem.which_pass != 0 || gbl.internal == 0)
          semfin_free_memory();
        if (sem.which_pass == 0) {
          /* CONTAINS clause has an empty body without any internal subprograms */
          if (gbl.internal == 1) {
            /* even if it CONTAINS no internal routine, still need to change 
               the entry points of the containing */
            if (STYPEG(gbl.currsub) == ST_ENTRY)
              STYPEP(gbl.currsub, ST_PROC);

            gbl.currsub = 0;
            gbl.internal = 0;
            gbl.empty_contains = TRUE;
            gbl.p_adjarr = NOSYM;
            gbl.p_adjstr = NOSYM;
          } else if (gbl.internal > 1) {
            /*
             * we're at the end of an internal procedure within a
             * a module during the first pass over the module.
             * The scanner does not set scn.end_program_unit to TRUE
             * in this context.  So now, need to reinitialize for the
             * next internal subprogram if it appears.
             */
            restore_host_state(1);
            restore_host(&host_state, TRUE);
            gbl.currsub = 0;
            sem.pgphase = PHASE_INIT;
            gbl.p_adjarr = NOSYM;
            gbl.p_adjstr = NOSYM;
          }
        }
    } else {
        if (IN_MODULE && sem.interface == 0) {
          gbl.currsub = end_of_host;
          mod_end_subprogram_two();
          gbl.currsub = 0;
        }
        semfin_free_memory();
      }
    } else if (sem.pgphase == PHASE_END_MODULE) { /* end of module */
      sem.pgphase = PHASE_INIT;
      /*
       * For a module containing just specifications, end_module() calls
       * semfin() in which case sem.doif_base is NULL.
       * For a module with contained subprograms, semfin() isn't called
       * after the last END statement.
       */
      semfin_free_memory();
      if (sem.which_pass) {
        gbl.currmod = 0;
      }
    } else if (sem.pgphase == PHASE_CONTAIN && gbl.internal && sem.which_pass) {
      /* end of host subprogram*/
      semfin();
      if (sem.mod_sym && sem.interface == 0)
        mod_end_subprogram_two();
      if (sem.which_pass != 0 || gbl.internal == 0)
        semfin_free_memory();
    }
    /*
     * Allow semant ccsym vars allocated by get_temp to be re-used for
     * the next statement, if necessary:
     */
    sem.temps_reset = FALSE;
  /* fall thru to 'statement_end' */

  statement_end: /* Processing for all <statement>s terminates here */
    if (STYPEG(gbl.currsub) == ST_ENTRY && FVALG(gbl.currsub) &&
        prevphase <= PHASE_USE && sem.pgphase > PHASE_USE) {
      int retdtype = DTYPEG(FVALG(gbl.currsub));
      int dtsptr = DTY(retdtype + 3);
      if (DTY(retdtype) == TY_DERIVED && dtsptr > NOSYM && !DCLDG(dtsptr)) {
        fixup_function_return_type(retdtype, dtsptr);
      }
    }

    sem.last_std = STD_PREV(0);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<iii> ::=
   */
  case III1:
    if (sem.interface) {
      error(155, 1, gbl.lineno, "Statement is redundant in an INTERFACE block",
            CNULL);
      sem.ignore_stmt = TRUE;
    }
    /* check whether we have entered a program as yet */
    if (sem.scope_level == 0) {
      dummy_program();
      restored = 0;
    }
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<nii> ::=
   */
  case NII1:
    if (sem.interface) {
      errsev(195);
      sem.ignore_stmt = TRUE;
    }
    /* check whether we have entered a program as yet */
    if (sem.scope_level == 0) {
      dummy_program();
      restored = 0;
    }
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<nim> ::=
   */
  case NIM1:
    if (IN_MODULE_SPEC) {
      ERR310("Illegal statement in the specification part of a MODULE", CNULL);
      sem.ignore_stmt = TRUE;
    }
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<pgm> ::=
   */
  case PGM1:
    /* check that we have entered a program as yet */
    if (sem.scope_level == 0)
      dummy_program();
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<end> ::=
   */
  case END1:
    if (!sem.interface && sem.pgphase < PHASE_EXEC) {
      if (gbl.currsub && !sem.which_pass) {
        do_iface(0);
      }
      if (!IN_MODULE)
        do_iface(1);
      else
        do_iface_module();
    } else if (sem.which_pass && !IN_MODULE && gbl.internal <= 1) {
        do_iface(1);
    }
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <prog title> ::= <routine id>    |
   */
  case PROG_TITLE1:
    itemp = ITEM_END;
    func_result = 0;
    goto prog_title;
  /*
   *      <prog title> ::= <routine id> ( ) <func suffix> |
   */
  case PROG_TITLE2:
    itemp = ITEM_END;
    func_result = SST_SYMG(RHS(4));
    goto prog_title;
  /*
   *      <prog title> ::= <routine id> ( <formal list> ) <func suffix>  |
   */
  case PROG_TITLE3:
    itemp = SST_BEGG(RHS(3));
    func_result = SST_SYMG(RHS(5));
  prog_title:
    /* no parameters allowed for programs */
    if (gbl.rutype == RU_PROG && itemp != ITEM_END)
      errsev(41);
    if (!sem.interface)
      gbl.funcline = gbl.lineno;

    if (gbl.rutype == RU_FUNC) {
      /* reserve one extra space in case this a function requires an
       * extra argument - a new argument may be inserted at the
       * beginning of the list.
       */
      NEED(aux.dpdsc_avl + 1, aux.dpdsc_base, int, aux.dpdsc_size,
           aux.dpdsc_size + 100);
      *(aux.dpdsc_base + (aux.dpdsc_avl++)) = 0;
    }

    DPDSCP(gbl.currsub, aux.dpdsc_avl);
    NEED(aux.dpdsc_avl + 1, aux.dpdsc_base, int, aux.dpdsc_size,
         aux.dpdsc_size + 100);
    *(aux.dpdsc_base + (aux.dpdsc_avl)) = 0;
    count = 0;
    for (; itemp != ITEM_END; itemp = itemp->next) {
      sptr = itemp->t.sptr;
      if (sptr == 0) { /* alternate return designator (i.e. *) */
        if (gbl.rutype != RU_SUBR)
          errsev(49);
        else if (!sem.interface)
          gbl.arets = TRUE;
      } else {
        if ((sptr < gbl.currsub) && IN_MODULE) {
          sptr = insert_sym(sptr);
        }
        sptr = declsym(sptr, ST_IDENT, TRUE);
        if (SCG(sptr) != SC_NONE)
          error(42, 3, gbl.lineno, SYMNAME(sptr), CNULL);
        SCP(sptr, SC_DUMMY);
        if (sem.interface) {
          NODESCP(sptr, 1);
          IGNOREP(sptr, TRUE);
        }
      }
      count++;
      NEED(aux.dpdsc_avl + 1, aux.dpdsc_base, int, aux.dpdsc_size,
           aux.dpdsc_size + 100);
      *(aux.dpdsc_base + (aux.dpdsc_avl++)) = sptr;
    }
    /* Set parameter count
     *
     * For procedure pointer symbols it should go into dtype, for old style
     * procedure symbols use PARAMCT attribute.
     *
     * FIXME this might need to go into a function
     */
    if (is_procedure_ptr(gbl.currsub)) {
      set_proc_ptr_param_count_dtype(DTYPEG(gbl.currsub), count);
    } else {
      PARAMCTP(gbl.currsub, count);
    }
    SST_ASTP(LHS, 0);

    if (IN_MODULE && sem.interface == 0)
      gbl.currsub = mod_add_subprogram(gbl.currsub);
    record_func_result(gbl.currsub, func_result, FALSE /* not in ENTRY */);
    if (bind_attr.exist != -1) {
      process_bind(gbl.currsub);
      bind_attr.exist = -1;
      bind_attr.altname = 0;
    }
    break;

  /*
   *      <prog title> ::= BLOCKDATA   |
   */
  case PROG_TITLE4:
    rhstop = 1;
    gbl.rutype = RU_BDATA;
    sem.module_procedure = false;
    SST_SYMP(RHS(rhstop), getsymbol(".blockdata."));
    CCSYMP(SST_SYMG(RHS(rhstop)), 1);
    if (IN_MODULE)
      ERR310("BLOCKDATA may not appear in a MODULE", CNULL);
    goto routine_id;
  /*
   *      <prog title> ::= BLOCKDATA <id> |
   */
  case PROG_TITLE5:
    rhstop = 2;
    gbl.rutype = RU_BDATA;
    sem.module_procedure = false;
    if (IN_MODULE)
      ERR310("BLOCKDATA may not appear in a MODULE", CNULL);
    goto routine_id;
  /*
   *	<prog title> ::= MODULE <id> |
   */
  case PROG_TITLE6:
    sem.submod_sym = 0;
    sptr = begin_module(SST_SYMG(RHS(2)));
    sptr1 = NOSYM;
    goto module_shared;
  /*
   *	<prog title> ::= SUBMODULE ( <id> ) <id> |
   */
  case PROG_TITLE7:
    sem.submod_sym = SST_SYMG(RHS(5));
    sptr = begin_submodule(sem.submod_sym, SST_SYMG(RHS(3)), NOSYM, &sptr1);
    STYPEP(sem.submod_sym, ST_MODULE);
    goto module_shared;
  /*
   *	<prog title> ::= SUBMODULE ( <id> : <id> ) <id> |
   */
  case PROG_TITLE8:
    sem.submod_sym = SST_SYMG(RHS(7));
    sptr = begin_submodule(sem.submod_sym, SST_SYMG(RHS(3)), SST_SYMG(RHS(5)),
                           &sptr1);
    goto module_shared;
  /*
   *   <prog title> ::= <module procedure stmt>
   */
  case PROG_TITLE9:
    break;
  module_shared:
    gbl.prog_file_name = (char *)getitem(15, strlen(gbl.curr_file) + 1);
    strcpy(gbl.prog_file_name, gbl.curr_file);
    if (sem.pgphase != PHASE_INIT) {
      errsev(70);
      break;
    }
    if (sem.mod_sym) {
      if (sem.mod_cnt == 1)
        /* issue error during first pass */
        ERR310("MODULEs may not be nested", CNULL);
      break;
    }
    sem.mod_cnt++;
    sem.pgphase = PHASE_HEADER;
    sem.mod_sym = sptr;
    setfile(1, SYMNAME(sem.mod_sym), 0);
    gbl.currmod = sem.mod_sym;
    push_scope_level(sem.mod_sym, SCOPE_NORMAL);
    push_scope_level(sem.mod_sym, SCOPE_MODULE);
    SST_ASTP(LHS, 0);
    clear_subp_prefix_settings(&subp_prefix); 

    /* SUBMODULEs work as if they are hosted within their immediate parents. */
    if (sptr1 > NOSYM) {
      sem.use_seen = TRUE;
      sem.pgphase = PHASE_USE;
      init_use_stmts();
      open_module(sptr1);
      add_submodule_use();
      close_module();
    }
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<ident> ::= <id>
   */
  case IDENT1:
    sptr = SST_SYMG(RHS(1));
    if (STYPEG(sptr) == ST_ALIAS) {
      /*SST_SYMP(LHS, SYMLKG(sptr));*/
      SST_ALIASP(LHS, 1);
    } else
      SST_ALIASP(LHS, 0);
    SST_IDP(LHS, S_IDENT);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<id> ::= <id name>
   */
  case ID1:
    np = scn.id.name + SST_CVALG(RHS(1));
    sptr = getsymbol(np);
    if (sem.in_dim && sem.type_mode && !KINDG(sptr) &&
        STYPEG(sptr) != ST_MEMBER) {
      /* possible use of a type parameter in the dimension field
       * of an array type component declaration
       */
      KINDP(sptr, -1);
    }
    SST_SYMP(LHS, sptr);
    SST_ACLP(LHS, 0);
    if (sem.arrdim.assumedrank && SCG(sptr) == SC_DUMMY) {
      IGNORE_TKRP(sptr, IGNORE_R);
    }
#ifdef GSCOPEP
    if (!sem.which_pass && gbl.internal <= 1 && gbl.currsub) {
      ident_host_sub = gbl.currsub;
    } else if (!sem.which_pass && gbl.internal > 1 && gbl.currsub
               /* && STYPEG(sptr)*/) {
      defer_ident_list(sptr, ident_host_sub);
    } else if (sem.which_pass && gbl.internal <= 1 &&
               internal_proc_has_ident(sptr, gbl.currsub)) {
      if (STYPEG(sptr) == ST_ENTRY || STYPEG(sptr) == ST_PROC) {
        if (FVALG(sptr))
          GSCOPEP(FVALG(sptr), 1);
      } else if (STYPEG(sptr) == ST_UNKNOWN || STYPEG(sptr) == ST_IDENT ||
                 ST_ISVAR(STYPEG(sptr))) {
        GSCOPEP(sptr, 1);
      }
    }
#endif
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<func suffix> ::=  |
   */
  case FUNC_SUFFIX1:
    SST_SYMP(LHS, 0);
    break;
  /*
   *      <func suffix> ::= BIND  <bind attr> <id name> ( <id name> ) |
   */
  case FUNC_SUFFIX2:
    result_name = SST_CVALG(RHS(3));
    id_name = SST_CVALG(RHS(5));
    goto result_shared;
  /*
   *      <func suffix> ::= BIND <bind attr> |
   */
  case FUNC_SUFFIX3:

    /* pass nothing */
    SST_SYMP(LHS, 0);
    break;

  /*
   *      <func suffix> ::= <id name> ( <id name> )  BIND <bind attr>
   */
  case FUNC_SUFFIX4:
  /* do nothing */
  /* fall through */
  /*
   *	<func suffix> ::= <id name> ( <id name> )
   */
  case FUNC_SUFFIX5:

    result_name = SST_CVALG(RHS(1));
    id_name = SST_CVALG(RHS(3));
  result_shared:
    sptr = 0;
    np = scn.id.name + result_name;
    if (sem_strcmp(np, "result") == 0) {
      np2 = scn.id.name + id_name;
      sptr2 = getsymbol(np2);

      sptr = chk_intrinsic(sptr2, FALSE, FALSE);
      if (scn.stmtyp == TK_ENTRY && gbl.rutype == RU_FUNC) {
        /* have a function entry - create its result variable */
        sptr = create_func_entry_result(sptr);
      } else {
        sptr = declsym(sptr, ST_IDENT, TRUE);
        SCP(sptr, SC_DUMMY);
      }
      if (sem.interface) {
        NODESCP(sptr, 1);
        IGNOREP(sptr, TRUE);
      }
    } else
      error(34, 3, gbl.lineno, np, CNULL);
    SST_SYMP(LHS, sptr);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <entry statement> ::= <entry id> |
   */
  case ENTRY_STATEMENT1:
    itemp = ITEM_END;
    func_result = 0;
    goto entry_statement;
  /*
   *      <entry statement> ::= <entry id> ( ) <func suffix> |
   */
  case ENTRY_STATEMENT2:
    itemp = ITEM_END;
    func_result = SST_SYMG(RHS(4));
    goto entry_statement;
  /*
   *      <entry statement> ::= <entry id> ( <formal list> ) <func suffix>
   */
  case ENTRY_STATEMENT3:
    itemp = SST_BEGG(RHS(3));
    func_result = SST_SYMG(RHS(5));
  entry_statement:
    if (flg.standard) {
      error(535, 2, gbl.lineno, "ENTRY statement", "FORTRAN 2008");
    }

    entry_seen = TRUE;
    sptr2 = SST_SYMG(RHS(1));
    if (sptr2 == 0) {
      /* an error was detected in <entry id> */
      SST_ASTP(LHS, 0);
      break;
    }

    /* write out ENTRY */
    sptr1 = getlab();
    RFCNTP(sptr1, 1);

    /* reserve one extra space in case this is an array-valued function -
     * a new argument may be inserted at the beginning of the list.
     */
    if (gbl.rutype == RU_FUNC) {
      NEED(aux.dpdsc_avl + 1, aux.dpdsc_base, int, aux.dpdsc_size,
           aux.dpdsc_size + 100);
      *(aux.dpdsc_base + (aux.dpdsc_avl++)) = 0;
    } else
      DTYPEP(sptr2, 0);
    DPDSCP(sptr2, aux.dpdsc_avl);
    NEED(aux.dpdsc_avl + 1, aux.dpdsc_base, int, aux.dpdsc_size,
         aux.dpdsc_size + 100);
    *(aux.dpdsc_base + (aux.dpdsc_avl)) = 0;
    count = 0;
    for (; itemp != ITEM_END; itemp = itemp->next) {
      sptr = itemp->t.sptr;
      if (sptr == 0) { /* alternate return designator (i.e. *) */
        if (gbl.rutype != RU_SUBR)
          errsev(49);
        else
          gbl.arets = TRUE;
      } else {
        sptr = ref_ident(sptr);
        stype = STYPEG(sptr);
        if (stype == ST_ENTRY) {
          error(43, 3, gbl.lineno, "symbol", SYMNAME(sptr));
          sptr = insert_sym(sptr);
          SCP(sptr, SC_DUMMY);
        } else if (SCG(sptr) == SC_NONE) {
          if (stype != ST_UNKNOWN && stype != ST_IDENT && stype != ST_ARRAY &&
              stype != ST_STRUCT && stype != ST_PROC && stype != ST_VAR) {
            error(43, 3, gbl.lineno, "symbol", SYMNAME(sptr));
          }
          SCP(sptr, SC_DUMMY);
        } else if (SCG(sptr) == SC_LOCAL && !SAVEG(sptr))
          /*
           * watch out for the case where an <ident> is seen
           * as a use in a declaration (e.g., in an adj. array
           * expression).  NOTE that if it's dinit'd, dinit will
           * issue error.
           */
          SCP(sptr, SC_DUMMY);
        else if (SCG(sptr) != SC_DUMMY)
          error(43, 3, gbl.lineno, "symbol", SYMNAME(sptr));
      }
      NEED(aux.dpdsc_avl + 1, aux.dpdsc_base, int, aux.dpdsc_size,
           aux.dpdsc_size + 100);
      *(aux.dpdsc_base + (aux.dpdsc_avl++)) = sptr;
      count++;
    }

    PARAMCTP(sptr2, count);
    ast = mk_stmt(A_ENTRY, 0);
    A_SPTRP(ast, sptr2);
    SST_ASTP(LHS, ast);
    record_func_result(sptr2, func_result, TRUE /* in ENTRY */);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <routine id> ::= <subr prefix> SUBROUTINE <id>   |
   */
  case ROUTINE_ID1:
    rhstop = 3;
    gbl.rutype = RU_SUBR;
    sem.module_procedure = false;
    goto routine_id;
  /*
   *      <routine id> ::= <subr prefix> FUNCTION <id>  |
   */
  case ROUTINE_ID2:
    rhstop = 3;
    gbl.rutype = RU_FUNC;
    sem.module_procedure = false;
    /* data type of function not specified */
    lenspec[1].len = sem.gdtype = -1;
    lenspec[1].propagated = 0;
    goto routine_id;
  /*
   *      <routine id> ::= <func prefix> FUNCTION <fcn name> |
   */
  case ROUTINE_ID3:
    rhstop = 3;
    gbl.rutype = RU_FUNC;
    if (!(sem.deferred_func_kind || sem.deferred_func_len)) {
      /*
         The KIND was an unresolved ident (e.g., ident from an unprocessed
         module),
         skip the mod_type until after USE stmt processing
       */
      sem.gdtype =
          mod_type(sem.gdtype, sem.gty, lenspec[1].kind, lenspec[1].len,
                   lenspec[1].propagated, (int)SST_SYMG(RHS(3)));
    }
    goto routine_id;
  /*
   *      <routine id> ::= PROGRAM <id>
   */
  case ROUTINE_ID4:
    gbl.rutype = RU_PROG;
    sem.module_procedure = false;
    rhstop = 2;
    if (IN_MODULE)
      ERR310("PROGRAM may not appear in a MODULE", CNULL);

  routine_id:
    is_entry = FALSE;
    if (sem.interface && gbl.currsub) {
      error(303, 2, gbl.lineno, SYMNAME(gbl.currsub), CNULL);
      pop_subprogram();
      pop_scope_level(SCOPE_NORMAL);
    }
    if (gbl.empty_contains && sem.pgphase == PHASE_END && sem.which_pass == 0) {
      /* empty CONTAINS body with no internal subprograms */
      gbl.internal = 0;
      sem.pgphase = PHASE_INIT;
    }
    if (sem.pgphase != PHASE_INIT && !sem.interface) {
      if (IN_MODULE && !have_module_state()) {
        /* terminate -- ow, reset_module_state() will issue
         * an ICE because modstate_file is NULL; could say
         * something about CONTAINS, but we currently cannot
         * detect the missing CONTAINS of a module after the
         * first.
         */
        error(70, 0, gbl.lineno, CNULL, CNULL);
      }
      errsev(70);
    }
    /* C1548: checking MODULE prefix for subprograms that were 
              declared as separate module procedures */
    if (!sem.interface && subp_prefix.module) {
      sptr_temp = SST_SYMG(RHS(rhstop));
      if (!SEPARATEMPG(sptr_temp) && !find_explicit_interface(sptr_temp))
        error(1056, ERR_Severe, gbl.lineno, NULL, NULL);  
    }

    /* First internal subprogram after CONTAINS, semfin may have altered the
     * symbol table
     * (esp. INVOBJ) for the host subprogram processing. Restore the state to
     * what it was
     * before semfin. (FS 20415)
     */
    if (sem.which_pass && sem.pgphase == PHASE_CONTAIN && gbl.internal == 1) {
      restore_host_state(2);
    }

    if (!sem.interface && sem.mod_cnt == 0) {
      gbl.prog_file_name = (char *)getitem(15, strlen(gbl.curr_file) + 1);
      strcpy(gbl.prog_file_name, gbl.curr_file);
    }
    entry_seen = FALSE;
    if (sem.interface) {
      /* Open the interface scope. */
      sem.scope_stack[sem.scope_level].closed = FALSE;
      /* set curr_scope to parent's scope */
      stb.curr_scope = sem.scope_stack[sem.scope_level - 1].sptr;
      queue_tbp(SST_SYMG(RHS(rhstop)), 0, 0, 0, TBP_IFACE);
    }
    sptr = block_local_sym(refsym_inscope(SST_SYMG(RHS(rhstop)), OC_OTHER));
    if (STYPEG(sptr) == ST_ENTRY
        /* Call insert_sym() if there's a type bound
         * procedure that is in scope
         */
        || (STYPEG(sptr) == ST_PROC && CLASSG(sptr) && VTOFFG(sptr))) {
      /* this must be the enclosing routine */
      sptr = insert_sym(sptr);

    } else if (STYPEG(sptr) == ST_PROC && IN_MODULE_SPEC &&
               get_seen_contains() && !sem.which_pass &&
              /* separate module procedure is allowed to be declared &
                 defined within the same module
               */
               !IS_INTERFACEG(sptr)) {
      LOGICAL err = TYPDG(sptr) && SCOPEG(sptr) != stb.curr_scope;
      if (!err) {
        int dpdsc = 0;
        proc_arginfo(sptr, 0, &dpdsc, 0);
        err = dpdsc != 0;
      }
      if (err) {

        errWithSrc(155, 3, SST_LINENOG(RHS(rhstop)),
                   "Redefinition of", SYMNAME(sptr),
                   SST_COLUMNG(RHS(rhstop)), 0, false, CNULL);
      }
    }
    if (subp_prefix.pure && subp_prefix.impure) {
      error(545, 3, gbl.lineno, NULL, NULL);
    }
    sptr = declsym(sptr, ST_ENTRY, TRUE);

    if (sem.interface) {
      /* Re-close the interface scope. */
      sem.scope_stack[sem.scope_level].closed = TRUE;
      /* curr_scope will be reset by push_scope_level */
    }
    gbl.currsub = sptr;
    push_scope_level(sptr, SCOPE_NORMAL);
    if (sem.interface) {
      /* For submodules, don't close the scope_stack in order to make 
       * sure entities defined in parent modules are visible in
       * descendant submodules
       */
      if (!subp_prefix.module)
        /* Close the 'normal' scope. */
        sem.scope_stack[sem.scope_level].closed = TRUE;
    }
    push_scope_level(sptr, SCOPE_SUBPROGRAM);
    sem.pgphase = PHASE_HEADER;
    /* Set the storage class; if it's already dummy, then this subprogram
     * is an argument for which there is an interface.
     */
    if (SCG(sptr) != SC_DUMMY) {
      if (!sem.interface || !sem.interf_base[sem.interface - 1].abstract)
        SCP(sptr, SC_EXTERN);
      else {
        SCP(sptr, SC_NONE);
        if (sem.interf_base[sem.interface - 1].abstract) {
          ABSTRACTP(sptr, 1);
          INMODULEP(sptr, IN_MODULE);
        }
      }
    }
    PUREP(sptr, subp_prefix.pure);
    RECURP(sptr, subp_prefix.recursive);
    IMPUREP(sptr, subp_prefix.impure);
    ELEMENTALP(sptr, subp_prefix.elemental);
    if (subp_prefix.module) {
      if (!IN_MODULE && !INMODULEG(sptr)) {
        ERR310("MODULE prefix allowed only within a module or submodule", CNULL);
      } else if (sem.interface) {
        /* Use SEPARATEMPP to mark this is submod related subroutines, 
         * functions, procdures to differentiate regular module. The 
         * SEPARATEMPP field is overloaded with ISSUBMODULEP field 
         * ISSUBMODULEP is used for name mangling. 
         */
        SEPARATEMPP(sptr, TRUE);
        HAS_SMP_DECP(SCOPEG(sptr), TRUE);
        if (IN_MODULE)
          INMODULEP(sptr, TRUE);
        if (SST_FIRSTG(RHS(rhstop))) {
          TBP_BOUND_TO_SMPP(sptr, TRUE);
          /* We also set the HAS_TBP_BOUND_TO_SMP flag on the separate module 
           * procedure's module. This indicates that the module contains a 
           * separate module procedure declaration to which at least one TBP
           * has been bound.
           */
          HAS_TBP_BOUND_TO_SMPP(SCOPEG(sptr), TRUE);
        }
      } else {
        SEPARATEMPP(sptr, TRUE);

        /* check definition vs. declared interface */
        /*  F2008 [12.6.2.5]
            The characteristics and binding label of a procedure are fixed, but the 
            remainder of the interface may differ in differing contexts, except that 
            for a separate module procedure body.
         */
        if (sem.which_pass) {
          SPTR def = find_explicit_interface(sptr);
          /* Make sure this def is not from the contains of ancestor module*/
          if (def > NOSYM) {
            sptr_temp = SYMLKG(sptr) ? SYMLKG(sptr) : sptr;
            /* Check Characteristics of procedures matches for definition vs. declaration*/
            if (!cmp_interfaces_strict(def, sptr_temp, CMP_IFACE_NAMES | 
                                                       CMP_SUBMOD_IFACE))
              ;
          }
        }
      }
    } else {
      if (sem.interface && SYMIG(sptr) && INMODULEG(sptr)) {
        for (symi = SYMIG(sptr); symi; symi = SYMI_NEXT(symi)) {
          if (STYPEG(SYMI_SPTR(symi)) == ST_OPERATOR || 
              STYPEG(SYMI_SPTR(symi)) == ST_USERGENERIC)
            error(1212, ERR_Severe, gbl.lineno, SYMNAME(sptr), NULL);
        }
      } 
    }
    clear_subp_prefix_settings(&subp_prefix); 
    if (gbl.rutype == RU_FUNC) {
      /* for a FUNCTION (including ENTRY's), compiler created
       * symbols are created to represent the return values and
       * are stored in the FVAL field of the ENTRY's.
       * In the worst case, each entry will have its own ccsym.
       * As references occur (and in semfin), an attempt will be
       * made to share the temporaries.  Also, at these times,
       * the dtype of the temporary will have to be set properly.
       * semfin adjusts the storage class if necessary.
       */
      if (sem.gdtype != -1) {
        /* data type of function was specified */
        DCLDP(sptr, TRUE);
        DTYPEP(sptr, sem.gdtype);
        set_char_attributes(sptr, &sem.gdtype);
      }
    } else {
      DTYPEP(sptr, 0);
    }
    SYMLKP(sptr, NOSYM);
    FUNCLINEP(sptr, gbl.lineno);
    if (gbl.rutype != RU_PROG) {
      MSCALLP(sptr, mscall);
#ifdef CREFP
      CREFP(sptr, cref);
      NOMIXEDSTRLENP(sptr, nomixedstrlen);
#endif
    }
    SST_ASTP(LHS, 0);
    if (sem.interface) {
      init_implicit();
    } else if (IN_MODULE) {
    } else if (gbl.internal) {
      gbl.internal++;
      host_present = 0x8;
      symutl.none_implicit = sem.none_implicit &= ~host_present;
      SCP(sptr, SC_STATIC); 
    }
    seen_implicit = FALSE;
    seen_parameter = FALSE;
    if (sem.interface && gbl.internal <= 1) {
      /* INTERNAL flag might have gotten set in getsym()
       * for this symbol even though it is an interface. An interface
       * body should never contain a procedure defined by a subprogram,
       * so this flag should never be set for an interface. Because
       * getsym() does not have access to sem.interface, we reset the
       * INTERNAL flag here.
       */
      INTERNALP(sptr, 0);
    }
    IS_INTERFACEP(sptr, sem.interface);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<subr prefix> ::=  |
   */
  case SUBR_PREFIX1:
  /* fall through */
  /*
   *	<subr prefix> ::= <prefix spec>
   */
  case SUBR_PREFIX2:
    check_module_prefix();
    if (sem.interface) {
      /* set curr_scope to parent's scope, so subprogram ID
       * gets scope of parent */
      stb.curr_scope = sem.scope_stack[sem.scope_level - 1].sptr;
    }
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<prefix spec> ::= <prefix spec> <prefix> |
   */
  case PREFIX_SPEC1:
    break;
  /*
   *	<prefix spec> ::= <prefix>
   */
  case PREFIX_SPEC2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<prefix> ::= RECURSIVE |
   */
  case PREFIX1:
    check_duplicate(subp_prefix.recursive, "RECURSIVE");
    subp_prefix.recursive = TRUE;
    if (subp_prefix.elemental) {
      errsev(460);
    }
    break;
  /*
   *	<prefix> ::= PURE |
   */
  case PREFIX2:
    check_duplicate(subp_prefix.pure, "PURE");
    subp_prefix.pure = TRUE;
    break;
  /*
   *	<prefix> ::= ELEMENTAL |
   */
  case PREFIX3:
    check_duplicate(subp_prefix.elemental, "ELEMENTAL");
    subp_prefix.elemental = TRUE;
    if (subp_prefix.recursive) {
      errsev(460);
    }
    break;
  /*
   *	<prefix> ::= ATTRIBUTES ( <id name list> )
   */
  case PREFIX4:
    if (!cuda_enabled("attributes"))
      break;
    break;

  /*
   *      <prefix> ::= IMPURE
   */
  case PREFIX5:
    check_duplicate(subp_prefix.impure, "IMPURE");
    subp_prefix.impure = TRUE;
    break;

  /*
   *      <prefix> ::= MODULE
   */
  case PREFIX6:
    check_duplicate(subp_prefix.module, "MODULE");
    subp_prefix.module = TRUE;
    break;

  /*
   *	<prefix> ::= LAUNCHBOUNDS ( <launchbound> ) |
   */
  case PREFIX7:
    break;

  /*
   *	<prefix> ::= LAUNCHBOUNDS ( <launchbound> , <launchbound> )
   */
  case PREFIX8:
    break;


  /* ------------------------------------------------------------------ */
  /*
   *	<launchbound> ::= <integer>
   */
  case LAUNCHBOUND1:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<id name list> ::= <id name list> , <id name> |
   */
  case ID_NAME_LIST1:
    rhstop = 3;
    goto add_name_to_list;
    break;
  /*
   *	<id name list> ::= <id name>
   */
  case ID_NAME_LIST2:
    rhstop = 1;
  add_name_to_list:
    itemp = (ITEM *)getitem(0, sizeof(ITEM));
    itemp->next = ITEM_END;
    itemp->t.conval = SST_CVALG(RHS(rhstop));
    if (rhstop == 1)
      /* adding first item to list */
      SST_BEGP(LHS, itemp);
    else
      /* adding subsequent items to list */
      SST_ENDG(RHS(1))->next = itemp;
    SST_ENDP(LHS, itemp);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<func prefix> ::= <data type> |
   */
  case FUNC_PREFIX1:
  /* fall through */
  /*
   *	<func prefix> ::= <data type> <prefix spec> |
   */
  case FUNC_PREFIX2:
  /* fall through */
  /*
   *	<func prefix> ::= <prefix spec> <data type>
   */
  case FUNC_PREFIX3:
  /* fall through */
  /*
   *	<func prefix> ::= <prefix spec> <data type> <prefix spec>
   */
  case FUNC_PREFIX4:
    check_module_prefix();
    if (sem.interface) {
      /* set curr_scope to parent's scope, so subprogram ID
       * gets scope of parent */
      stb.curr_scope = sem.scope_stack[sem.scope_level - 1].sptr;
    }
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <entry id> ::= ENTRY <id>
   */
  case ENTRY_ID1:
    sptr = SST_SYMG(RHS(2));
    if (gbl.internal > 1) {
      error(155, 3, gbl.lineno, SYMNAME(sptr),
            "- The ENTRY statement is not allowed in an internal procedure");
      SST_SYMP(LHS, 0);
      break;
    }
    if (sem.doif_depth > 0) {
      /* Inside DO, IF, WHERE block; ignore statement */
      errsev(118);
      SST_SYMP(LHS, 0);
      break;
    }
    if (INSIDE_STRUCT) {
      error(117, 3, gbl.lineno,
            STSK_ENT(0).type == 's' ? "STRUCTURE" : "derived type", CNULL);
      SST_SYMP(LHS, 0);
      break;
    }
    if (gbl.rutype == RU_PROG || gbl.rutype == RU_BDATA || sem.interface) {
      errsev(70);
      SST_SYMP(LHS, 0);
      break;
    }
    if (gbl.rutype == RU_FUNC)
      /* have a function entry; create its ST_ENTRY symbol */
      sptr = create_func_entry(sptr);
    else
      sptr = declsym(sptr, ST_ENTRY, TRUE);

    if (IN_MODULE && sem.interface == 0)
      sptr = mod_add_subprogram(sptr);
    SST_SYMP(LHS, sptr);

    SYMLKP(sptr, SYMLKG(gbl.currsub));
    SYMLKP(gbl.currsub, sptr);
    FUNCLINEP(sptr, gbl.lineno);
    MSCALLP(sptr, mscall);
    if (sptr != gbl.currsub) {
      CFUNCP(sptr, CFUNCG(gbl.currsub));
    }
#ifdef CREFP
    CREFP(sptr, cref);
    NOMIXEDSTRLENP(sptr, nomixedstrlen);
#endif
    is_entry = TRUE;
    PUREP(sptr, PUREG(gbl.currsub));
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <fcn name> ::= <id> <opt len spec>
   */
  case FCN_NAME1:
    set_len_attributes(RHS(2), 1);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<formal list> ::= <formal list> , <formal> |
   */
  case FORMAL_LIST1:
    rhstop = 3;
    goto add_sym_to_list;
  /*
   *	<formal list> ::= <formal>
   */
  case FORMAL_LIST2:
    rhstop = 1;
    goto add_sym_to_list;

  /* ------------------------------------------------------------------ */
  /*
   *	<formal> ::= <id> |
   */
  case FORMAL1:
    /* scan sets SST_SYMP with sym pointer */
    sptr = chk_intrinsic(SST_SYMG(RHS(1)), TRUE, is_entry);
    SST_SYMP(LHS, sptr);
    break;
  /*
   *	<formal> ::= *
   */
  case FORMAL2:
    SST_SYMP(LHS, 0);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<ident list> ::= <ident list> , <ident> |
   */
  case IDENT_LIST1:
    rhstop = 3;
    goto add_sym_to_list;
  /*
   *	<ident list> ::= <ident>
   */
  case IDENT_LIST2:
    rhstop = 1;
  add_sym_to_list:
    itemp = (ITEM *)getitem(0, sizeof(ITEM));
    itemp->next = ITEM_END;
    itemp->t.sptr = SST_SYMG(RHS(rhstop));
    itemp->ast = SST_ASTG(RHS(rhstop)); /* copied for <access> rules */
    if (rhstop == 1)
      /* adding first item to list */
      SST_BEGP(LHS, itemp);
    else
      /* adding subsequent items to list */
      SST_ENDG(RHS(1))->next = itemp;
    SST_ENDP(LHS, itemp);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<end stmt> ::= <END stmt>    |
   */
  case END_STMT1:
    if (gbl.rutype == RU_SUBR || gbl.rutype == RU_FUNC)
      defer_arg_chk(SPTR_NULL, SPTR_NULL, SPTR_NULL, 0, 0, true);
    if (sem.interface && !gbl.rutype)
        error(310, 3, gbl.lineno, "Missing ENDINTERFACE statement", CNULL);
    else if (sem.which_pass)
      fix_class_args(gbl.currsub);

    dummy_program();
    if (IN_MODULE_SPEC && gbl.internal == 0)
      goto end_of_module;
    if (gbl.currsub == 0 && sem.pgphase == PHASE_INIT && gbl.internal)
      check_end_subprogram(host_state.rutype, 0);
    else if (gbl.internal > 1) {
      if (gbl.rutype == RU_PROG || gbl.rutype == RU_BDATA) {
        error(302, 3, gbl.lineno, name_of_rutype(gbl.rutype), CNULL);
        gbl.internal = 0;
      }
    } else {
      if (0 && sem.which_pass && !sem.mod_cnt && gbl.internal == 0 &&
          !sem.interface) {
        fprintf(stderr, "OPROC %s:", gbl.src_file);
        fprintf(stderr, "%s\n", SYMNAME(gbl.currsub));
      }
      enforce_denorm();
    }
    SST_IDP(LHS, 1); /* mark as end of subprogram unit */
    if (IN_MODULE && sem.interface == 0)
      mod_end_subprogram();
    pop_scope_level(SCOPE_NORMAL);
    if (!IN_MODULE && !sem.interface) {
      queue_tbp(0, 0, 0, 0, TBP_CLEAR);
      check_defined_io();
    }
    defer_pt_decl(0, 0);
    break;
  /*
   *	<end stmt> ::= ENDBLOCKDATA  <opt ident> |
   */
  case END_STMT2:
    if (gbl.currsub == 0 || gbl.rutype != RU_BDATA)
      error(302, 3, gbl.lineno, "BLOCKDATA", CNULL);
    else if (SST_SYMG(RHS(2)) &&
             strcmp(SYMNAME(gbl.currsub), SYMNAME(SST_SYMG(RHS(2)))) != 0)
      error(309, 3, gbl.lineno, SYMNAME(SST_SYMG(RHS(2))), CNULL);

    SST_IDP(LHS, 1); /* mark as end of subprogram unit */
    pop_scope_level(SCOPE_NORMAL);
    break;
  /*
   *	<end stmt> ::= ENDFUNCTION   <opt ident> |
   */
  case END_STMT3:
    defer_arg_chk(SPTR_NULL, SPTR_NULL, SPTR_NULL, 0, 0, true);
  submod_proc_endfunc:
    fix_iface(gbl.currsub);
    if (sem.which_pass && !sem.interface) {
      fix_class_args(gbl.currsub);
    }
    if (/*!IN_MODULE*/ !sem.mod_cnt && !sem.interface) {
      queue_tbp(0, 0, 0, 0, TBP_COMPLETE_END);
      queue_tbp(0, 0, 0, 0, TBP_CLEAR);
    }
    defer_pt_decl(0, 0);
    dummy_program();
    check_end_subprogram(RU_FUNC, SST_SYMG(RHS(2)));

    SST_IDP(LHS, 1); /* mark as end of subprogram unit */
    pop_scope_level(SCOPE_NORMAL);
    if (sem.interface) {
      if (DTYPEG(gbl.currsub) == DT_ASSCHAR) {
        error(
            155, 3, FUNCLINEG(gbl.currsub),
            "FUNCTION may not be declared character*(*) when in an INTERFACE -",
            SYMNAME(gbl.currsub));
      }
      if (IN_MODULE) {
        do_iface_module();
      }
    }
    if (IN_MODULE && sem.interface == 0)
      mod_end_subprogram();
    check_defined_io();
    if (!IN_MODULE && !sem.interface)
      clear_ident_list();
    fix_proc_ptr_dummy_args();
    sem.seen_import = FALSE;
    break;
  /*
   *	<end stmt> ::= ENDMODULE     <opt ident> |
   */
  case END_STMT4:
    sem.seen_end_module = TRUE;
    if (sem.mod_sym == 0) {
      error(302, 3, gbl.lineno, "MODULE", CNULL);
      gbl.internal = 0;
      break;
    }
    if (sem.interface) {
      error(310, 3, gbl.lineno, "Missing ENDINTERFACE statement", CNULL);
      sem.interface = 0;
    }
    if (gbl.currsub) {
      error(310, 3, gbl.lineno, "Missing END statement", SYMNAME(gbl.currsub));
    }
    if (SST_SYMG(RHS(2)) &&
        strcmp(SYMNAME(sem.mod_sym), SYMNAME(SST_SYMG(RHS(2)))) != 0)
      error(309, 3, gbl.lineno, SYMNAME(SST_SYMG(RHS(2))), CNULL);
  end_of_module:
    queue_tbp(0, 0, 0, 0, TBP_COMPLETE_ENDMODULE);
    queue_tbp(0, 0, 0, 0, TBP_CLEAR);
    do_iface(sem.which_pass);
    fix_iface0();
    end_module();
    SST_IDP(LHS, 0); /* mark as end of module */
    if (sem.mod_cnt == 1) {
      sem.mod_cnt++;
      /*fe_restart();*/
    } else {
      sem.mod_cnt = 0;
      sem.mod_sym = 0;
      sem.submod_sym = 0;
    }
    check_defined_io();
    clear_ident_list();
    sem.seen_end_module = FALSE;
    break;
  /*
   *	<end stmt> ::= ENDPROGRAM    <opt ident> |
   */
  case END_STMT5:
    queue_tbp(0, 0, 0, 0, TBP_COMPLETE_END);
    queue_tbp(0, 0, 0, 0, TBP_CLEAR);
    defer_pt_decl(0, 0);
    dummy_program();
    check_end_subprogram(RU_PROG, SST_SYMG(RHS(2)));

    SST_IDP(LHS, 1); /* mark as end of subprogram unit */
    pop_scope_level(SCOPE_NORMAL);
    check_defined_io();
    break;
  /*
   *	<end stmt> ::= ENDSUBROUTINE <opt ident> |
   */
  case END_STMT6:
    defer_arg_chk(SPTR_NULL, SPTR_NULL, SPTR_NULL, 0, 0, true);
    fix_iface(gbl.currsub);
    if (sem.which_pass && !sem.interface) {
      fix_class_args(gbl.currsub);
    }
    if (/*!IN_MODULE*/ !sem.mod_cnt && !sem.interface) {
      queue_tbp(0, 0, 0, 0, TBP_COMPLETE_END);
      queue_tbp(0, 0, 0, 0, TBP_CLEAR);
    }
    defer_pt_decl(0, 0);
    dummy_program();
    check_end_subprogram(RU_SUBR, SST_SYMG(RHS(2)));

    SST_IDP(LHS, 1); /* mark as end of subprogram unit */
    pop_scope_level(SCOPE_NORMAL);
    if (sem.interface && IN_MODULE) {
      do_iface_module();
    }
    if (IN_MODULE && sem.interface == 0)
      mod_end_subprogram();
    check_defined_io();
    if (!IN_MODULE && !sem.interface)
      clear_ident_list();
    fix_proc_ptr_dummy_args();
    sem.seen_import = FALSE;
    break;
  /*
   *	<end stmt> ::= ENDSUBMODULE <opt ident>
   */
  case END_STMT7:
    sem.seen_end_module = TRUE;
    if (sem.submod_sym <= NOSYM) {
      error(302, 3, gbl.lineno, "SUBMODULE", CNULL);
      gbl.internal = 0;
      break;
    }
    if (SST_SYMG(RHS(2)) &&
        strcmp(SYMNAME(sem.submod_sym), SYMNAME(SST_SYMG(RHS(2)))) != 0) {
      error(309, 3, gbl.lineno, SYMNAME(SST_SYMG(RHS(2))), CNULL);
    }
    goto end_of_module;
  /*
   *	<end stmt> ::= ENDPROCEDURE <opt ident>
   */
  case END_STMT8:
    if (gbl.currsub == 0 || !sem.module_procedure) {
      ERR310("unexpected END PROCEDURE", CNULL);
      break;
    }
    if (gbl.rutype == RU_FUNC)
       goto submod_proc_endfunc;
    /* For sub-module procedure points to a subroutine of another module,
       we need to take cares of the dummy arguments and process differently
       from the general ENDPROCEDURE.
     */
    if (gbl.rutype == RU_SUBR) {
      dummy_program();
      enforce_denorm();
      SST_IDP(LHS, 1); /* mark as end of subprogram unit */ 
      pop_scope_level(SCOPE_SUBPROGRAM);
      defer_pt_decl(0, 0);
      sem.seen_import = FALSE;
      do_end_subprogram(top, gbl.rutype);
      break;
    }
    SST_IDP(LHS, 1); /* mark as end of subprogram unit */
    mod_end_subprogram();
    sem.seen_import = FALSE;
    do_end_subprogram(top, gbl.rutype);
    gbl.currsub = 0;
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<opt ident> ::= |
   */
  case OPT_IDENT1:
    SST_SYMP(LHS, 0);
    break;
  /*
   *	<opt ident> ::= <ident>
   */
  case OPT_IDENT2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <block stmt> ::= BLOCK |
   *      <block stmt> ::= <check construct> : BLOCK
   *
   *   Generate block code with the form:
   *     continue           -- first block std (labeled)
   *       block prolog     -- allocate/init/array_check code
   *     comment (continue) -- prolog end == body begin boundary marker
   *       block body       -- user code
   *       block epilog     -- finalize/deallocate code
   *     continue           -- last block std (labeled)
   *
   *   Each block has an ST_BLOCK sptr where:
   *    - STARTLAB(sptr) is the top-of-block label
   *    - ENDLAB(sptr) is the end-of-block label
   *
   *   For any sptr local to a block, the block entry, end-of-prolog, and exit
   *   stds that are needed for inserting prolog and epilog code are accessible
   *   via macros defined in symutl.h:
   *    - BLOCK_ENTRY_STD(sptr)
   *    - BLOCK_ENDPROLOG_STD(sptr)
   *    - BLOCK_EXIT_STD(sptr)
   *   Prolog code can be inserted at the top of the prolog via BLOCK_ENTRY_STD,
   *   and at the end of the prolog via BLOCK_ENDPROLOG_STD.  Epilog code can
   *   be inserted at the end of the epilog via BLOCK_EXIT_STD.  There is no
   *   known need to insert code at the top of the epilog, so there is no
   *   marker std between body and epilog code.
   */
  case BLOCK_STMT1:
    set_construct_name(0);
    FLANG_FALLTHROUGH;
  case BLOCK_STMT2:
    if (DI_NEST(sem.doif_depth) >= DI_B(DI_FIRST_DIRECTIVE) && !XBIT(59,8))
      error(1219, ERR_Severe, gbl.lineno,
            "BLOCK construct in the scope of a parallel directive", CNULL);
    sptr = sem.scope_stack[sem.scope_level].sptr;
    push_scope_level(sptr, SCOPE_NORMAL);
    push_scope_level(sptr, SCOPE_BLOCK);
    block_sptr = getccsym('b', sem.blksymnum++, ST_BLOCK);
    ENCLFUNCP(block_sptr,
              sem.construct_sptr ? sem.construct_sptr : gbl.currsub);
    sem.construct_sptr = block_sptr;
    if (sem.which_pass) {
      lab = scn.currlab ? scn.currlab : getlab();
      RFCNTI(lab);
      // Setting VOL on this block entry label and the exit label just below
      // prohibits the back end from deleting them.  This is necessary to
      // support parallelization and debugging.  However, this can cause the
      // back end at -O2 and above to generate dead code during unrolling,
      // which causes control flow analysis prior to vectorization to fail.
      // Pending a more complete fix for this problem, only set this flag at
      // low opt levels (and prohibit parallelization of code containing a
      // block).
      VOLP(lab, flg.opt < 2 && flg.debug && !XBIT(123, 0x400));
      ENCLFUNCP(lab, block_sptr);
      std = add_stmt(mk_stmt(A_CONTINUE, 0));
      STARTLINEP(block_sptr, gbl.lineno);
      STARTLABP(block_sptr, lab);
      LABSTDP(lab, std);
      STD_LABEL(std) = lab;
      std = add_stmt(mk_stmt(A_CONTINUE, 0));
      ast_to_comment(STD_AST(std));
      ENTSTDP(block_sptr, std);
    }
    NEED_DOIF(doif, DI_BLOCK);
    DI_NAME(doif) = get_construct_name();
    DI_ENCL_BLOCK_SCOPE(doif) = sem.block_scope;
    sem.block_scope = sem.scope_level;
    sem.pgphase = PHASE_HEADER;
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <block stmt> ::= ENDBLOCK <construct name>
   */
  case BLOCK_STMT3:
    doif = sem.doif_depth;
    if (sem.doif_depth <= 0) {
      error(104, ERR_Severe, gbl.lineno, "- mismatched END BLOCK", CNULL);
      break;
    }
    construct_name = get_construct_name();
    if (DI_NAME(doif) != construct_name)
      err307("BLOCK and ENDBLOCK", DI_NAME(doif), construct_name);
    if (sem.which_pass) {
      if (scn.currlab)
        add_stmt(mk_stmt(A_CONTINUE, 0));
      if (DI_EXIT_LABEL(doif)) {
        std = add_stmt(mk_stmt(A_CONTINUE, 0));
        STD_LABEL(std) = DI_EXIT_LABEL(doif);
      }
      block_sptr = sem.construct_sptr;
      lab = getlab();
      RFCNTI(lab);
      // See the comment just above about the entry label VOL flag.
      VOLP(lab, flg.opt < 2 && flg.debug && !XBIT(123, 0x400));
      ENCLFUNCP(lab, block_sptr);
      std = add_stmt(mk_stmt(A_CONTINUE, 0));
      ENDLINEP(block_sptr, gbl.lineno);
      ENDLABP(block_sptr, lab);
      LABSTDP(lab, std);
      STD_LABEL(std) = lab;
    }
    --sem.doif_depth;
    sem.block_scope = DI_ENCL_BLOCK_SCOPE(doif);
    sem.construct_sptr = ENCLFUNCG(sem.construct_sptr);
    if (STYPEG(sem.construct_sptr) != ST_BLOCK)
      sem.construct_sptr = 0; // not in a construct
    pop_scope_level(SCOPE_NORMAL);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <declaration> ::= <data type> <optional comma> <pgm> <typdcl list> |
   */
  case DECLARATION1:
    if (sem.class && sem.type_mode) {
      error(155, 3, gbl.lineno, "CLASS components must be pointer or"
                                " allocatable",
            CNULL);
    }
    SST_ASTP(LHS, 0);
    break;
  /*
   *      <declaration> ::= <dimkeyword> <opt attr> <pgm> <dcl id list>    |
   */
  case DECLARATION2:
    SST_ASTP(LHS, 0);
    break;
  /*
   *      <declaration> ::= <nis> IMPLICIT <pgm> <implicit type>   |
   */
  case DECLARATION3:
    if (sem.block_scope)
      error(1218, ERR_Severe, gbl.lineno, "An IMPLICIT", CNULL);
    SST_ASTP(LHS, 0);
    break;
  /*
   *      <declaration> ::= <nis> COMMON <pgm> <common list>   |
   */
  case DECLARATION4:
    if (sem.block_scope)
      error(1218, ERR_Severe, gbl.lineno, "A COMMON", CNULL);
    SST_ASTP(LHS, 0);
    break;
  /*
   *      <declaration> ::= <nis> EXTERNAL <opt attr> <pgm> <ident list>      |
   */
  case DECLARATION5:
    for (itemp = SST_BEGG(RHS(5)); itemp != ITEM_END; itemp = itemp->next) {
      /* Produce a procedure symbol */
      if (POINTERG(itemp->t.sptr)) {
        LOGICAL was_declared = DCLDG(itemp->t.sptr);
        /* External pointer should come out the same as procedure(T) pointer */
        sptr = decl_procedure_sym(itemp->t.sptr, proc_interf_sptr,
                                  (entity_attr.exist | ET_B(ET_POINTER)));
        sptr = setup_procedure_sym(itemp->t.sptr, proc_interf_sptr,
                                   (entity_attr.exist | ET_B(ET_POINTER)),
                                   entity_attr.access);
        DCLDP(sptr, was_declared);
      } else {
        /* Use simple approach when we can't argue that this needs to be a
         * procedure pointer */
        sptr = declsym(itemp->t.sptr, ST_PROC, FALSE);
      }

      if (!TYPDG(sptr)) {
        TYPDP(sptr, 1);
      }
      if (SCG(sptr) == SC_DUMMY) {
        IS_PROC_DUMMYP(sptr, 1);
      } 
    }
    SST_ASTP(LHS, 0);
    break;
  /*
   *      <declaration> ::= <nis> INTRINSIC <opt attr> <pgm> <ident list>     |
   */
  case DECLARATION6:
    for (itemp = SST_BEGG(RHS(5)); itemp != ITEM_END; itemp = itemp->next) {
      sptr = refsym(itemp->t.sptr, OC_OTHER);
      stype = STYPEG(sptr);
      if (!IS_INTRINSIC(sptr)) {
        /* Not an intrinsic. So, try finding it */
        sptr2 = findByNameStypeScope(SYMNAME(sptr), ST_PD, 0);
        if (!sptr2) {
          sptr2 = findByNameStypeScope(SYMNAME(sptr), ST_INTRIN, 0);
          if (!sptr2) {
            sptr2 = findByNameStypeScope(SYMNAME(sptr), ST_GENERIC, 0);
          }
        }
        if (sptr2) {
          sptr = sptr2;
          stype = STYPEG(sptr);
          sptr2 = insert_sym(sptr);
          STYPEP(sptr2, ST_ALIAS);
          SYMLKP(sptr2, sptr);
        }
      }
      if (IS_INTRINSIC(stype)) {
        EXPSTP(sptr, 1); /* Freeze as an intrinsic */
        TYPDP(sptr, 1);  /* appeared in INTRINSIC statement */
        if (stype == ST_GENERIC) {
          sptr2 = select_gsame(sptr);
          if (sptr2)
            /* no need to
             * EXPSTP(sptr2, 1);
             * symbol is always begins with a .
             */
            ;
          else if (IN_MODULE) {
            /* Predefined symbols such as generics are
             * not exported into mod files.  A statement such as
             * use m, ren => max
             * will produce a "not public entity" message unless
             * we make a symbol that will be exported.
             */
            sptr2 = insert_sym(sptr);
            STYPEP(sptr2, ST_ALIAS);
            SYMLKP(sptr2, sptr);
          }
        }
      } else
        error(126, 3, gbl.lineno, SYMNAME(sptr), CNULL);
    }
    SST_ASTP(LHS, 0);
    break;
  /*
   *      <declaration> ::= <iii> <nis> SAVE <opt attr> <save list> |
   */
  case DECLARATION7:
    SST_ASTP(LHS, 0);
    break;
  /*
   *      <declaration> ::= <iii> <nis> SAVE                       |
   */
  case DECLARATION8:
    SST_ASTP(LHS, 0);
    if (sem.construct_sptr)
      SAVEP(sem.construct_sptr, true);
    else
      sem.savall = TRUE;
    sem.savloc = TRUE;
    break;
  /*
   *      <declaration> ::= PARAMETER <pgm> ( <ideqc list> ) |
   */
  case DECLARATION9:
    seen_parameter = TRUE;
    SST_ASTP(LHS, 0);
    if (sem.interface == 0)
      end_param();
    break;
  /*
   *      <declaration> ::= <nis> EQUIVALENCE <pgm> <equiv groups> |
   */
  case DECLARATION10:
    if (sem.block_scope)
      error(1218, ERR_Severe, gbl.lineno, "An EQUIVALENCE", CNULL);
    SST_ASTP(LHS, 0);
    break;
  /*
   *      <declaration> ::= <iii> <nis> DATA <dinit list>          |
   */
  case DECLARATION11:
    SST_ASTP(LHS, 0);
    break;
  /*
   *      <declaration> ::= PARAMETER  <pgm> <vxeqc list>    |
   */
  case DECLARATION12:
    if (flg.standard)
      error(171, 2, gbl.lineno, "PARAMETER", CNULL);
    seen_parameter = TRUE;
    SST_ASTP(LHS, 0);
    if (sem.interface == 0)
      end_param();
    break;
  /*
   *      <declaration> ::= <iii> <nis> NAMELIST <namelist groups> |
   */
  case DECLARATION13:
    if (sem.block_scope)
      error(1218, ERR_Severe, gbl.lineno, "A NAMELIST", CNULL);
    SST_ASTP(LHS, 0);
    break;
  /*
   *      <declaration> ::= STRUCTURE <pgm> <struct begin1> <struct begin2> |
   */
  case DECLARATION14:
    if (flg.standard)
      error(171, 2, gbl.lineno, "STRUCTURE", CNULL);
    if (INSIDE_STRUCT && STSK_ENT(0).type != 's' && STSK_ENT(0).type != 'm') {
      error(70, 2, gbl.lineno, "(STRUCTURE ignored)", CNULL);
      break;
    }
    /* Get a structure stack entry */
    sem.stsk_depth++;
    NEED(sem.stsk_depth, sem.stsk_base, STSK, sem.stsk_size,
         sem.stsk_depth + 12);
    stsk = &STSK_ENT(0);

    dtype = sem.stag_dtype;

    /* Save structure information in structure stack */
    stsk->type = 's';
    stsk->mem_access = 0;
    stsk->dtype = dtype;
    stsk->sptr = SST_RNG2G(RHS(4)); /* sym ptr to field name list */
    stsk->last = NOSYM;
    stsk->ict_beg = stsk->ict_end = NULL;

    /* Handle the field-namelist field */

    sptr = stsk->sptr;
    if (sptr == NOSYM) {
      if (sem.stsk_depth != 1)
        error(137, 2, gbl.lineno, CNULL, CNULL);
    } else {
      if (sem.stsk_depth == 1) {
        error(136, 2, gbl.lineno, CNULL, CNULL);
      } else {
        /* link field-namelist into member list at this level */
        stsk = &STSK_ENT(1);
        link_members(stsk, sptr);
      }
    }
    SST_ASTP(LHS, 0);
    break;
  /*
   *      <declaration> ::= ENDSTRUCTURE               |
   */
  case DECLARATION15:
    if (flg.standard)
      error(171, 2, gbl.lineno, "ENDSTRUCTURE", CNULL);
    if (INSIDE_STRUCT) {

      /* Check out structure, get its length */
      stsk = &STSK_ENT(0);
      if (stsk->type != 's') {
        errsev(160);
        break;
      }
      dtype = stsk->dtype;
      sptr = stsk->sptr;
      chkstruct(dtype);

      /* Save initializer constant tree (ict) for this structure */
      if (sem.stsk_depth == 1 && stsk->ict_beg != NULL) {
        /* This is top structure, fix up top subc ict entry */
        ict = GET_ACL(15);
        ict->id = AC_VMSSTRUCT;
        ict->next = NULL;
        ict->subc = stsk->ict_beg;
        ict->u1.ast = 0;
        ict->repeatc = astb.i1;
        ict->sptr = 0;
        ict->dtype = dtype;
        stsk->ict_beg = ict;
      }
      DTY(dtype + 5) = put_getitem_p(stsk->ict_beg);
      if (DTY(dtype + 3))
        DCLDP(DTY(dtype + 3), TRUE); /* "complete" tag declaration */

      /* Pop out to parent structure (if any) */
      sem.stsk_depth--;
      stsk = &STSK_ENT(0);

      /* For each member in parent structure (if any), having this
       * ict generate a substructure (subc) ict entry.  These are then
       * linked to the parent's ict.
       */
      if (INSIDE_STRUCT && DTY(dtype + 5) != 0) {
        for (; sptr != NOSYM; sptr = SYMLKG(sptr)) {
          ict = GET_ACL(15);
          ict->id = AC_VMSSTRUCT;
          ict->next = NULL;
          if (stsk->ict_end)
            stsk->ict_end->next = ict;
          else
            stsk->ict_beg = ict;
          stsk->ict_end = ict;
          ict->subc = get_getitem_p(DTY(dtype + 5));
          ict->u1.ast = 0;
          if (DTY(DTYPEG(sptr)) == TY_ARRAY)
            ict->repeatc = AD_NUMELM(AD_PTR(sptr));
          else
            ict->repeatc = astb.i1;
          ict->sptr = sptr;
          ict->dtype = dtype;
        }
      }
    } else
      error(70, 2, gbl.lineno, "(ENDSTRUCTURE ignored)", CNULL);
    SST_ASTP(LHS, 0);
    break;
  /*
   *      <declaration> ::= RECORD <pgm> <record list>
   */
  case DECLARATION16:
    if (flg.standard)
      error(171, 2, gbl.lineno, "RECORD", CNULL);
    break;
  /*
   *      <declaration> ::= UNION
   */
  case DECLARATION17:
    if (flg.standard)
      error(171, 2, gbl.lineno, "UNION", CNULL);
    if (!INSIDE_STRUCT) {
      error(70, 2, gbl.lineno, "(UNION ignored)", CNULL);
      break;
    }
    stsk = &STSK_ENT(0);
    if (stsk->type != 's' && stsk->type != 'm') {
      error(70, 2, gbl.lineno, "(UNION ignored)", CNULL);
      break;
    }
    dtype = get_type(6, TY_UNION, NOSYM);
    name_prefix_char = 'u';
    goto union_map;
  /*
   *      <declaration> ::= ENDUNION
   */
  case DECLARATION18:
    if (flg.standard)
      error(171, 2, gbl.lineno, "ENDUNION", CNULL);
    if (!INSIDE_STRUCT) {
      error(70, 2, gbl.lineno, "(ENDUNION ignored)", CNULL);
      break;
    }
    stsk = &STSK_ENT(0);
    if (stsk->type != 'u') {
      errsev(160);
      break;
    }
    dtype = stsk->dtype;
    sptr = stsk->sptr;
    chkstruct(dtype);
    STSK_ENT(1).last = stsk->last;
    DTY(dtype + 5) = put_getitem_p(stsk->ict_beg);
    if (stsk->ict_beg != NULL) {
      STSK *pstsk = &STSK_ENT(1); /* parent (a struct) of the union */
#if DEBUG
      assert(pstsk->type == 's', "ENDUNION:union not in struct", sptr, 3);
#endif
      /*
       * create a set node of the union which contains all of the
       * initializers for the union's maps.  This set node is added
       * to the structure stack of the union's parent (a structure).
       */
      ict = GET_ACL(15);
      ict->id = AC_VMSUNION;
      ict->next = NULL;
      ict->subc = stsk->ict_beg;
      ict->u1.ast = 0;
      ict->repeatc = astb.i1;
      ict->sptr = sptr;
      ict->dtype = dtype;
      if (pstsk->ict_beg == NULL)
        pstsk->ict_beg = ict;
      else
        pstsk->ict_end->next = ict;
      pstsk->ict_end = ict;
    }
    sem.stsk_depth--;
    SST_ASTP(LHS, 0);
    break;
  /*
   *      <declaration> ::= MAP
   */
  case DECLARATION19:
    if (flg.standard)
      error(171, 2, gbl.lineno, "MAP", CNULL);
    if (!INSIDE_STRUCT) {
      error(70, 2, gbl.lineno, "(MAP ignored)", CNULL);
      break;
    }
    stsk = &STSK_ENT(0);
    if (stsk->type != 'u') {
      error(70, 2, gbl.lineno, "(MAP ignored)", CNULL);
      break;
    }
    dtype = get_type(6, TY_STRUCT, NOSYM);
    name_prefix_char = 'm';
  union_map:
    stype = ST_MEMBER;
    sptr =
        declref(getsymf("%c@%05ld", name_prefix_char, (long)dtype), stype, 'r');
#if DEBUG
    assert(STYPEG(sptr) == stype,
           scn.stmtyp == TK_UNION ? "UNION: bad stype" : "MAP: bad stype", sptr,
           3);
#endif
    CCSYMP(sptr, 1);
    SYMLKP(sptr, NOSYM);
    DTYPEP(sptr, dtype); /* must be done before link members */
    DTY(dtype + 3) = 0;  /* no tag */
    /* link the union or map (structure) into the current structure */
    link_members(stsk, sptr);

    /* Save union information in structure stack */
    sem.stsk_depth++;
    NEED(sem.stsk_depth, sem.stsk_base, STSK, sem.stsk_size,
         sem.stsk_depth + 12);
    stsk = &STSK_ENT(0);
    stsk->type = scn.stmtyp == TK_UNION ? 'u' : 'm';
    stsk->mem_access = 0;
    stsk->dtype = dtype;
    stsk->sptr = sptr; /* sym ptr union */
    stsk->last = STSK_ENT(1).last;
    stsk->ict_beg = stsk->ict_end = NULL;
    SST_ASTP(LHS, 0);
    break;
  /*
   *      <declaration> ::= ENDMAP |
   */
  case DECLARATION20:
    if (flg.standard)
      error(171, 2, gbl.lineno, "ENDMAP", CNULL);
    if (!INSIDE_STRUCT) {
      error(70, 2, gbl.lineno, "(ENDMAP ignored)", CNULL);
      break;
    }
    stsk = &STSK_ENT(0);
    if (stsk->type != 'm') {
      errsev(160);
      break;
    }
    dtype = stsk->dtype;
    sptr = stsk->sptr;
    chkstruct(dtype);
    STSK_ENT(1).last = stsk->last;
    DTY(dtype + 5) = put_getitem_p(stsk->ict_beg);
    if (stsk->ict_beg != NULL) {
      STSK *pstsk = &STSK_ENT(1); /* parent (a union) of the map */
#if DEBUG
      assert(pstsk->type == 'u', "ENDMAP: map not in union", sptr, 3);
#endif
      /*
       * add the map's initializer trees to the union's (its parent)
       * structure stack.
       */
      if (pstsk->ict_beg == NULL)
        pstsk->ict_beg = stsk->ict_beg;
      else
        pstsk->ict_end->next = stsk->ict_beg;
      pstsk->ict_end = stsk->ict_end;
    }
    sem.stsk_depth--;
    SST_ASTP(LHS, 0);
    break;
  /*
   *      <declaration> ::= TYPE <opt type spec> <opt attr> <pgm> <id> <opt
   * tpsl> |
   */
  case DECLARATION21:
    sptr = block_local_sym(SST_SYMG(RHS(5)));
    np = SYMNAME(sptr);
    if (strcmp(np, "integer") == 0 || strcmp(np, "logical") == 0 ||
        strcmp(np, "real") == 0 || strcmp(np, "doubleprecision") == 0 ||
        strcmp(np, "complex") == 0 || strcmp(np, "character") == 0) {
      error(155, 3, gbl.lineno, "A derived type type-name must not be the same "
                                "as the name of the intrinsic type",
            np);
      if (IS_INTRINSIC(STYPEG(sptr)))
        sptr = insert_sym(sptr);
    } else if (RESULTG(sptr)) {
      error(155, 3, gbl.lineno, "A derived type type-name conflicts with"
                                " function result -",
            np);
      sptr = insert_sym(sptr);
    } else
      sptr = getocsym(sptr, OC_OTHER, TRUE);
    if (STYPEG(sptr) == ST_TYPEDEF && DTY(DTYPEG(sptr) + 2) == 0) {
      /* This declaration will fill in an empty TYPEDEF created in
       * an implicit statement.
       */
      dtype = sem.stag_dtype = DTYPEG(sptr);
      DTY(sem.stag_dtype + 2) = 1; /* size */
    } else {
      if (STYPEG(sptr) == ST_USERGENERIC) {
        int origSym = sptr;
        sptr = insert_sym(sptr);
        STYPEP(sptr, ST_TYPEDEF);
        GTYPEP(origSym, sptr);
      } else {
        sptr = declsym(sptr, ST_TYPEDEF, TRUE);
      }
      dtype = sem.stag_dtype = get_type(6, TY_DERIVED, NOSYM);
      DTYPEP(sptr, sem.stag_dtype);
      DTY(sem.stag_dtype + 2) = 1; /* size */
      DTY(sem.stag_dtype + 3) = sptr;
      DTY(sem.stag_dtype + 5) = 0;
    }
#if defined(PARENTP)
    if (SST_CVALG(RHS(2)) & 0x4) {
      int sym = SST_LSYMG(RHS(2));
      int dtype2 = DTYPEG(sym);
      /* type extension */
      if (CFUNCG(sym)) {
        error(155, 3, gbl.lineno, "Cannot EXTEND BIND(C) derived type",
              SYMNAME(sym));
      } else if (DTY(dtype2) == TY_DERIVED && SEQG(DTY(dtype2 + 3))) {
        error(155, 3, gbl.lineno, "Cannot EXTEND SEQUENCE derived type",
              SYMNAME(sym));
      } else if (SST_CVALG(RHS(2)) & 0x1) {
        error(155, 3, gbl.lineno, "EXTENDS may not be used with BIND(C) "
                                  "derived type",
              SYMNAME(sym));
      }
      PARENTP(sptr, sym);
    } else {
      /* type extension */
      PARENTP(sptr, 0);
    }
    if (SST_CVALG(RHS(2)) & 0x8) {
      /* abstract type */
      ABSTRACTP(sptr, 1);
    }
#endif
    if (SST_CVALG(RHS(2)) & 0x1)
      /* BIND present? */
      CFUNCP(sptr, 1);
    if (entity_attr.access == 'v') {
      /* we can set the private bit immediately here,
       * since it doesn't get overwritten later */
      PRIVATEP(sptr, 1);
    } else if (entity_attr.access == 'u') {
      /* if the default access mode for the module is private,
       * the private bit gets overwritten in semfin.do_access()
       * We need to remember to reset this to public */
      accessp = (ACCL *)getitem(3, sizeof(ACCL));
      accessp->sptr = sptr;
      accessp->type = entity_attr.access;
      accessp->oper = ' ';
      accessp->next = sem.accl.next;
      sem.accl.next = accessp;
    }

    if (INSIDE_STRUCT)
      error(117, 3, gbl.lineno,
            STSK_ENT(0).type == 'd' ? "derived type" : "STRUCTURE", CNULL);

    /* Get a structure stack entry */
    sem.stsk_depth++;
    NEED(sem.stsk_depth, sem.stsk_base, STSK, sem.stsk_size,
         sem.stsk_depth + 12);
    stsk = &STSK_ENT(0);
    /* Save structure information in structure stack */
    stsk->type = 'd';
    stsk->mem_access = 0;
    stsk->dtype = dtype;
    stsk->sptr = sptr;
    stsk->last = NOSYM;
    stsk->ict_beg = stsk->ict_end = NULL;
    sem.type_mode = 1;
    SST_ASTP(LHS, 0);
    link_parents(stsk, PARENTG(sptr));
    break;
  /*
   *      <declaration> ::= ENDTYPE <opt ident> |
   */
  case DECLARATION22:
    if (INSIDE_STRUCT) {
      /* Check out structure, get its length */
      stsk = &STSK_ENT(0);
      if (stsk->type != 'd') {
        errsev(160);
        break;
      }
      dtype = stsk->dtype;
      sptr = stsk->sptr;
      chkstruct(dtype);
      if (dtype && SST_SYMG(RHS(2)) && DTY(dtype + 3) &&
          strcmp(SYMNAME(DTY(dtype + 3)), SYMNAME(SST_SYMG(RHS(2)))) != 0) {
        error(155, 3, gbl.lineno, "Name on END TYPE statement does not"
                                  " match name on corresponding TYPE statement",
              CNULL);
      }
      if (PARENTG(DTY(dtype + 1)) && DINITG(DTY(dtype + 1))) {
        /* Type extension - make sure we initialize any parent components
         * that require initialization.
         */
        build_typedef_init_tree(DTY(dtype + 1), DDTG(DTYPEG(DTY(dtype + 1))));
      }
      if (ALLOCFLDG(sptr)) {
        init_allocatable_typedef_components(sptr);
      }
      save_typedef_init(sptr, dtype);
      build_typedef_init_tree(sptr, dtype);

      queue_type_param(0, dtype, 0, 2);
      put_default_kind_type_param(dtype, 0, 1);
      queue_type_param(0, 0, 0, 0);

      queue_tbp(sptr, 0, 0, dtype, TBP_INHERIT);
      queue_tbp(sptr, 0, 0, dtype, TBP_ADD_TO_DTYPE);
      if (!IN_MODULE)
        queue_tbp(0, 0, 0, 0, TBP_COMPLETE_ENDTYPE);
      /* Call get_static_type_descriptor() to ensure creation of type
       * descriptor at its definition point. This is especially important
       * for derived types defined in a subprogram and referenced in a
       * contains subprogram.
       */
      if (gbl.internal <= 1)
        get_static_type_descriptor(sptr);
      if (0 && size_of(dtype) == 0 && DTY(dtype + 1) <= NOSYM) {
        int oldsptr, tag;
        tag = DTY(DTYPEG(sptr) + 3);
        if (!UNLPOLYG(tag)) {
          /* Create "empty" typedef. */
          oldsptr = sptr;
          get_static_type_descriptor(sptr);
          sptr = insert_sym(sptr);
          sptr = declsym(sptr, ST_TYPEDEF, TRUE);
          dtype = get_type(6, TY_DERIVED, NOSYM);
          DTYPEP(sptr, dtype);
          DTY(dtype + 1) = NOSYM;
          DTY(dtype + 2) = 0; /* will be filled in */
          DTY(dtype + 3) = sptr;
          DTY(dtype + 5) = 0;
          SDSCP(sptr, SDSCG(oldsptr));
          DCLDP(sptr, DCLDG(oldsptr));
          chkstruct(dtype);
        }
      }
      chk_initialization_with_kind_parm(dtype);
    } else
      error(70, 2, gbl.lineno, "(END TYPE ignored)", CNULL);
    sem.type_mode = 0;
    sem.tbp_access_stmt = 0;
    entity_attr.access = ' '; /* Reset access spec of types */
    SST_ASTP(LHS, 0);
    break;
  /*
   *      <declaration> ::= VOLATILE <opt attr> <pgm> <vol list> |
   */
  case DECLARATION23:
    SST_ASTP(LHS, 0);
    break;
  /*
   *      <declaration> ::= <nis> POINTER <opt attr> <pgm> <ptr list>
   */
  case DECLARATION24:
    SST_ASTP(LHS, 0);
    break;
  /*
   *      <declaration> ::= <nis> ALLOCATABLE <opt attr> <pgm> <alloc id list>
   */
  case DECLARATION25:
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<declaration> ::= <data type> <opt attr list> :: <pgm> <entity decl
   *list> |
   */
  case DECLARATION26:
    if (entity_attr.exist & ET_B(ET_PARAMETER)) {
      seen_parameter = TRUE;
      if (sem.interface == 0)
        end_param();
    }
    SST_ASTP(LHS, 0);
    in_entity_typdcl = FALSE;
    entity_attr.exist = 0;
    entity_attr.access = ' ';
    bind_attr.exist = -1;
    bind_attr.altname = 0;
    break;
  /*
   *	<declaration> ::= <intent> <opt attr> <pgm> <ident list> |
   */
  case DECLARATION27:
    if (sem.block_scope)
      error(1218, ERR_Severe, gbl.lineno, "An INTENT", CNULL);
    count = 0;
    for (itemp = SST_BEGG(RHS(4)); itemp != ITEM_END; itemp = itemp->next) {
      sptr = refsym(itemp->t.sptr, OC_OTHER);
      INTENTP(sptr, entity_attr.intent);
      if (sem.interface) {
        if (SCG(sptr) != SC_DUMMY)
          error(134, 3, gbl.lineno, "- intent specified for nondummy argument",
                SYMNAME(sptr));
      } else {
        /* defer checking of storage class until semfin */
        itemp1 = (ITEM *)getitem(3, sizeof(ITEM));
        itemp1->next = sem.intent_list;
        sem.intent_list = itemp1;
        itemp1->t.sptr = sptr;
        itemp1->ast = gbl.lineno;
      }
      if (bind_attr.altname && (++count > 1))
        error(280, 2, gbl.lineno, "BIND: allowed only in module", 0);
      if (bind_attr.exist != -1) {
        process_bind(sptr);
      }
    }
    bind_attr.exist = -1;
    bind_attr.altname = 0;
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<declaration> ::= <access spec> <opt attr> <pgm> <access list> |
   */
  case DECLARATION28:
    count = 0;
    for (itemp = SST_BEGG(RHS(4)); itemp != ITEM_END; itemp = itemp->next) {
      sptr1 = sptr = itemp->t.sptr;
      if (STYPEG(sptr) != ST_OPERATOR && STYPEG(sptr) != ST_USERGENERIC)
        sptr = refsym(sptr, OC_OTHER);
      if (STYPEG(sptr) == ST_ARRAY && ADJARRG(sptr))
        error(84, 3, gbl.lineno, SYMNAME(sptr),
              "- must not be an automatic array");
      else {
        accessp = (ACCL *)getitem(3, sizeof(ACCL));
        accessp->sptr = sptr1;
        accessp->type = entity_attr.access;
        accessp->next = sem.accl.next;
        accessp->oper = ' ';
        if (itemp->ast == 1)
          accessp->oper = 'o';
        sem.accl.next = accessp;
      }
      if (bind_attr.altname && (++count > 1))
        error(84, 3, gbl.lineno, SYMNAME(bind_attr.altname),
              "- too many variables bound to name");
      if (bind_attr.exist != -1) {
        if (!IN_MODULE)
          error(280, 2, gbl.lineno, "BIND: allowed only in module", 0);
        process_bind(sptr);
      }
    }
    entity_attr.access = ' ';
    bind_attr.exist = -1;
    bind_attr.altname = 0;
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<declaration> ::= OPTIONAL <opt attr> <pgm> <ident list> |
   */
  case DECLARATION29:
    if (sem.block_scope)
      error(1218, ERR_Severe, gbl.lineno, "An OPTIONAL", CNULL);
    for (itemp = SST_BEGG(RHS(4)); itemp != ITEM_END; itemp = itemp->next) {
      sptr = refsym(itemp->t.sptr, OC_OTHER);
      OPTARGP(sptr, 1);
    }
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<declaration> ::= TARGET <opt attr> <pgm> <target list> |
   */
  case DECLARATION30:
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<declaration> ::= <nis> <interface> |
   */
  case DECLARATION31:
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<declaration> ::= <nis> <end interface> |
   */
  case DECLARATION32:
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<declaration> ::= <nis> <pgm> USE <use>
   */
  case DECLARATION33:
    close_module();
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<declaration> ::= <access spec> |
   */
  case DECLARATION34:
    if (INSIDE_STRUCT) {
      if (STSK_ENT(0).type != 'd') {
        error(155, 3, gbl.lineno,
              "PUBLIC/PRIVATE may only be used in derived types", "");
        break;
      }
      if (entity_attr.access == 'u') {
        ERR310("PUBLIC may not appear in a derived type definition", CNULL);
      } else {
        stsk = &STSK_ENT(0);
        sptr = DTY(stsk->dtype + 3); /* tag sptr */
        if (stsk->last != NOSYM) {
          if (sem.type_mode == 2 && IN_MODULE_SPEC) {
            if (queue_tbp(0, 0, 0, stsk->dtype, TBP_STATUS)) {
              error(155, 3, gbl.lineno,
                    "Incorrect sequence of PRIVATE and type bound "
                    "procedures in",
                    SYMNAME(sptr));
            }
            if (sem.tbp_access_stmt) {
              error(155, 3, gbl.lineno,
                    "Redundant PRIVATE statement in type bound "
                    "procedure section of",
                    SYMNAME(sptr));
            } else {
              sem.tbp_access_stmt = 1;
            }
          } else if (!PARENTG(stsk->last) || PARENTG(stsk->last) != stsk->last)
            /* error - private statement appears after member */
            error(155, 3, gbl.lineno, "PRIVATE statement must appear before "
                                      "components of derived type",
                  SYMNAME(sptr));
        } else {
          if (sem.type_mode == 2 && IN_MODULE_SPEC) {
            if (sem.tbp_access_stmt) {
              error(155, 3, gbl.lineno,
                    "Redundant PRIVATE statement in type bound "
                    "procedure section of",
                    SYMNAME(sptr));
            } else {
              sem.tbp_access_stmt = 1;
            }
          } else
          if (stsk->mem_access) {
            error(155, 3, gbl.lineno,
                  "Redundant PRIVATE statement in derived type", SYMNAME(sptr));
          }
          /* set PUBLIC/PRIVATE here.  link_members() will apply it to
             the components of this derived type. */
          stsk->mem_access = entity_attr.access;
        }
      }
    } else { /* not INSIDE_STRUCT */
      if (sem.accl.type) {
        if (sem.accl.type == entity_attr.access)
          error(155, 2, gbl.lineno, "Redundant PUBLIC/PRIVATE statement",
                CNULL);
        else
          error(155, 3, gbl.lineno, "Conflicting PUBLIC/PRIVATE statement",
                CNULL);
      } else
        sem.accl.type = entity_attr.access;
    }
    SST_ASTP(LHS, 0);
    break;

  /*
   *	<declaration> ::= <procedure stmt> |
   */
  case DECLARATION35:
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<declaration> ::= <mp threadprivate> ( <tp list> ) |
   */
  case DECLARATION36:
    for (itemp = SST_BEGG(RHS(3)); itemp != ITEM_END; itemp = itemp->next) {
      sptr = itemp->t.sptr; /* ST_CMBLK */
      if (sptr == 0)
        continue;
      THREADP(sptr, 1);

      if (STYPEG(sptr) != ST_CMBLK && !DCLDG(sptr) && !SAVEG(sptr) &&
          !in_save_scope(sptr)) {
        error(38, 3, gbl.lineno, SYMNAME(sptr), CNULL);
      } else if (STYPEG(sptr) != ST_CMBLK && ALLOCATTRG(sptr)) {
        get_static_descriptor(sptr);
        get_all_descriptors(sptr);
      }
    }
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<declaration> ::= <dec declaration>
   */
  case DECLARATION37:
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<declaration> ::= <pragma declaration> |
   */
  case DECLARATION38:
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<declaration> ::= <nis> AUTOMATIC <opt attr> <pgm> <ident list>     |
   */
  case DECLARATION39:
    uf("AUTOMATIC");
    break;
  /*
   *	<declaration> ::= <nis> STATIC <opt attr> <pgm> <ident list>
   */
  case DECLARATION40:
    uf("STATIC");
    break;
  /*
   *      <declaration> ::= BIND <bind attr> <opt attr> <bind list> |
   */
  case DECLARATION41: {
    int ii;
    ii = 1;
    count = 0;
    /* go through ths bind list and call process_bind for each */
    if (bind_attr.exist != -1) {
      for (itemp = SST_BEGG(RHS(4)); itemp != ITEM_END; itemp = itemp->next) {
        if (bind_attr.altname && (++count > 1))
          error(84, 3, gbl.lineno, SYMNAME(bind_attr.altname),
                "- too many variables bound to name");
        if (!IN_MODULE)
          error(84, 2, gbl.lineno, "BIND: allowed only in module", 0);
        process_bind(itemp->t.sptr);
      }
      bind_attr.exist = -1;
      bind_attr.altname = 0;
    }
  }
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<declaration> ::= <nis> <pgm> <import> <opt import> |
   */
  case DECLARATION42:
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<declaration> ::= <nis> <pgm> ENUM , BIND ( <id name> ) |
   */
  case DECLARATION43:
    np = scn.id.name + SST_CVALG(RHS(7));
    if (sem_strcmp(np, "c") == 0) {
      sem.in_enum = TRUE;
    } else
      error(4, 3, gbl.lineno, "Illegal BIND -", np);
    next_enum = 0;
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<declaration> ::= <nis> ENUMERATOR <opt attr> <enums> |
   */
  case DECLARATION44:
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<declaration> ::= <nis> ENDENUM |
   */
  case DECLARATION45:
    sem.in_enum = FALSE;
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<declaration> ::= <procedure declaration> |
   */
  case DECLARATION46:
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<declaration> ::= <type bound procedure> |
   */
  case DECLARATION47:
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<declaration> ::= ATTRIBUTES ( <id name list> ) <opt attr> <pgm> <ident
   *list> |
   */
  case DECLARATION48:
    if (!cuda_enabled("attributes")) {
      SST_ASTP(LHS, 0);
      break;
    }
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<declaration> ::= TCONTAINS |
   */
  case DECLARATION49:
    dtype = stsk->dtype;
    if (DTY(dtype) == TY_DERIVED) {
      int tag = DTY(dtype + 3);
      if (SEQG(tag)) {
        error(155, 3, gbl.lineno, "Type bound procedure part not allowed "
                                  "for SEQUENCE type",
              SYMNAME(tag));
      }
      if (CFUNCG(tag)) {
        error(155, 3, gbl.lineno, "Type bound procedure part not allowed "
                                  "for BIND(C) type",
              SYMNAME(tag));
      }
    }
    sem.type_mode = 2;
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<declaration> ::= <nis> PROTECTED <opt attr> <pgm> <ident list>
   */
  case DECLARATION50:
    if (!IN_MODULE_SPEC) {
      error(155, 3, gbl.lineno,
            "PROTECTED may only appear in the specification part of a MODULE",
            CNULL);
    }
    for (itemp = SST_BEGG(RHS(5)); itemp != ITEM_END; itemp = itemp->next) {
      sptr = ref_ident_inscope(itemp->t.sptr);
      PROTECTEDP(sptr, 1);
    }
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<declaration> ::= <nis> ASYNCHRONOUS <opt attr> <pgm> <ident list>
   */
  case DECLARATION51:
    for (itemp = SST_BEGG(RHS(5)); itemp != ITEM_END; itemp = itemp->next) {
      sptr = ref_ident_inscope(itemp->t.sptr);
      if (sem.block_scope && sptr < sem.scope_stack[sem.block_scope].symavl &&
          !ASYNCG(sptr))
        error(1219, ERR_Severe, gbl.lineno,
              "ASYNCHRONOUS statement in a BLOCK construct", CNULL);
      ASYNCP(sptr, true);
    }
    SST_ASTP(LHS, 0);
    break;

  /*
   *	<declaration> ::= <nis> <accel decl begin> ACCDECL <accel decl list>
   */
  case DECLARATION52:
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<declaration> ::= <nis> <accel decl begin> DECLARE <accel decl list> |
   */
  case DECLARATION53:
    SST_ASTP(LHS, 0);
    break;
  /*
   *      <declaration> ::= <generic type procedure> |
   */
  case DECLARATION54:
    break;
  /*
   *	<declaration> ::= <final subroutines> |
   */
  case DECLARATION55:
    break;
  /*
   *	<declaration> ::= <nis> CONTIGUOUS <opt attr> <pgm> <ident list>
   */
  case DECLARATION56:
    break;
  /*
   *	<declaration> ::= <nis> <accel decl begin> ROUTINE <accel routine list>
   */
  case DECLARATION57:
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<declaration> ::= <nis> <accel decl begin> ROUTINE
   *           ( <routine id list> ) <accel routine list> |
   */
  case DECLARATION58:
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<declaration> ::= <seq> <pgm> |
   */
  case DECLARATION59:
    if (INSIDE_STRUCT && STSK_ENT(0).type == 'd' && SST_CVALG(RHS(1)) == 's') {
      stsk = &STSK_ENT(0);
      sptr = DTY(stsk->dtype + 3); /* tag sptr */
      if (stsk->last != NOSYM) {
        /* error - SEQUENCE statement appears after member */
        error(
            155, 3, gbl.lineno,
            "SEQUENCE statement must appear before components of derived type",
            SYMNAME(sptr));
      } else {
        if (SEQG(sptr)) {
          error(155, 3, gbl.lineno,
                "Redundant SEQUENCE statement in derived type", SYMNAME(sptr));
        }
        SEQP(sptr, 1); /* set SEQ on the tag of derived type */
        if (PARENTG(sptr)) {
          error(155, 3, gbl.lineno,
                "SEQUENCE may not appear in a derived type with "
                "EXTENDS keyword",
                CNULL);
        }
      }
    }
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<declaration> ::= <nis> <mp decl begin> <mp decl> |
   */
  case DECLARATION60:
    break;
  /*
   *	<declaration> ::= <nis> VALUE <opt attr> <pgm> <ident list>
   */
  case DECLARATION61:
    if (sem.block_scope)
      error(1218, ERR_Severe, gbl.lineno, "A VALUE", CNULL);
    for (itemp = SST_BEGG(RHS(5)); itemp != ITEM_END; itemp = itemp->next) {
      sptr = ref_ident_inscope(itemp->t.sptr);
      PASSBYVALP(sptr, 1);
      PASSBYREFP(sptr, 0);
    }
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<declaration> ::= <accel begin> <accel dp stmts>
   */
  case DECLARATION62:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<accel dp stmts> ::= <accel shape declstmt> |
   */
  case ACCEL_DP_STMTS1:
    break;
  /*
   *	<accel dp stmts> ::= <accel policy declstmt>
   */
  case ACCEL_DP_STMTS2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<accel shape declstmt> ::= ACCSHAPE <accel shape dir>
   */
  case ACCEL_SHAPE_DECLSTMT1:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<accel shape dir> ::= ( <accel dpvarlist> ) |
   */
  case ACCEL_SHAPE_DIR1:
  /*
   *	<accel shape dir> ::= ( <accel dpvarlist> ) <accel shape attrs> |
   */
  case ACCEL_SHAPE_DIR2:
    break;
  /*
   *	<accel shape dir> ::= '<' <ident> '>' ( <accel dpvarlist> ) |
   */
  case ACCEL_SHAPE_DIR3:
  /*
   *	<accel shape dir> ::= '<' <ident> '>' ( <accel dpvarlist> ) <accel shape attrs>
   */
  case ACCEL_SHAPE_DIR4:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<accel shape attrs> ::= <accel shape attrs> <accel shape attr> |
   */
  case ACCEL_SHAPE_ATTRS1:
    break;
  /*
   *	<accel shape attrs> ::= <accel shape attr>
   */
  case ACCEL_SHAPE_ATTRS2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<accel shape attr> ::= <accel dpdefault attr> |
   */
  case ACCEL_SHAPE_ATTR1:
    break;
  /*
   *	<accel shape attr> ::= <accel dpinit_needed attr> |
   */
  case ACCEL_SHAPE_ATTR2:
    break;
  /*
   *	<accel shape attr> ::= <accel dptype attr>
   */
  case ACCEL_SHAPE_ATTR3:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<accel dpdefault attr> ::= DEFAULT ( <ident> ) 
   */
  case ACCEL_DPDEFAULT_ATTR1:
    break;


  /* ------------------------------------------------------------------ */
  /*
   *	<accel dpinit_needed attr> ::= INIT_NEEDED ( <accel dpinitvar list> )
   */
  case ACCEL_DPINIT_NEEDED_ATTR1:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<accel dpinitvar list> ::= <accel dpinitvar list>, <ident> |
   */
  case ACCEL_DPINITVAR_LIST1:
  /*
   *	<accel dpinitvar list> ::= <ident>
   */
  case ACCEL_DPINITVAR_LIST2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<accel dptype attr> ::= TYPE ( <ident> )
   */
  case ACCEL_DPTYPE_ATTR1:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<accel policy declstmt> ::= ACCPOLICY <accel policy name> <accel policy dir>
   */
  case ACCEL_POLICY_DECLSTMT1:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<accel policy name> ::= '<' <ident> '>' |
   */
  case ACCEL_POLICY_NAME1:
  /*
   *	<accel policy name> ::= '<' <ident> : <ident> '>'
   */
  case ACCEL_POLICY_NAME2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<accel policy dir> ::= <accel policy attr list>
   */
  case ACCEL_POLICY_DIR1:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<accel policy attr list> ::= <accel policy attr list> <accel policy attr> |
   */
  case ACCEL_POLICY_ATTR_LIST1:
    break;
  /*
   *	<accel policy attr list> ::= <accel policy attr>
   */
  case ACCEL_POLICY_ATTR_LIST2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<accel policy attr> ::= CREATE ( <accel dpvarlist> ) |
   */
  case ACCEL_POLICY_ATTR1:
    break;
  /*
   *	<accel policy attr> ::= NO_CREATE ( <accel dpvarlist> ) |
   */
  case ACCEL_POLICY_ATTR2:
    break;
  /*
   *	<accel policy attr> ::= COPYIN ( <accel dpvarlist> ) |
   */
  case ACCEL_POLICY_ATTR3:
    break;
  /*
   *	<accel policy attr> ::= COPYOUT ( <accel dpvarlist> ) |
   */
  case ACCEL_POLICY_ATTR4:
    break;
  /*
   *	<accel policy attr> ::= COPY ( <accel dpvarlist> ) |
   */
  case ACCEL_POLICY_ATTR5:
    break;
  /*
   *	<accel policy attr> ::= UPDATE ( <accel dpvarlist> ) |
   */
  case ACCEL_POLICY_ATTR6:
    break;
  /*
   *	<accel policy attr> ::= DEVICEPTR ( <accel dpvarlist> ) |
   */
  case ACCEL_POLICY_ATTR7:
    break;
  /*
   *	<accel policy attr> ::= <accel dpdefault attr> |
   */
  case ACCEL_POLICY_ATTR8:
    break;
  /*
   *	<accel policy attr> ::= <accel dptype attr>
   */
  case ACCEL_POLICY_ATTR9:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<accel dpvarlist> ::= <accel dpvarlist> <accel dpvar> |
   */
  case ACCEL_DPVARLIST1:
    break;
  /*
   *	<accel dpvarlist> ::= <accel dpvar>
   */
  case ACCEL_DPVARLIST2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<accel dpvar> ::= <ident> |
   */
  case ACCEL_DPVAR1:
  /*
   *	<accel dpvar> ::= <ident> '<' <ident> '>' |
   */
  case ACCEL_DPVAR2:
  /*
   *	<accel dpvar> ::= <ident> ( <accel dpvar bnds> ) |
   */
  case ACCEL_DPVAR3:
  /*
   *	<accel dpvar> ::= <ident> '<' <ident> '>' ( <accel dpvar bnds> )
   */
  case ACCEL_DPVAR4:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<accel dpvar bnds> ::= <accel dpvar bnds> , <accel dpvar bnd> |
   */
  case ACCEL_DPVAR_BNDS1:
    break;
  /*
   *	<accel dpvar bnds> ::= <accel dpvar bnd>
   */
  case ACCEL_DPVAR_BNDS2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<accel dpvar bnd> ::= <accel dp bnd> : <accel dp bnd> |
   */
  case ACCEL_DPVAR_BND1:
    break;
  /*
   *	<accel dpvar bnd> ::= <accel dp bnd>
   */
  case ACCEL_DPVAR_BND2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<accel dp bnd> ::= <accel dp sbnd> 
   */
  case ACCEL_DP_BND1:
    break;
  /*
   *	<accel dp bnd> ::= <accel dp bndexp> |
   */
  case ACCEL_DP_BND2:
    break;
  /*
   *	<accel dp bnd> ::= <accel dp bndexp1>
   */
  case ACCEL_DP_BND3:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<accel dp bndexp> ::= <accel dp addexp> |
   */
  case ACCEL_DP_BNDEXP1:
    break;
  /*
   *	<accel dp bndexp> ::= <accel dp mulexp>
   */
  case ACCEL_DP_BNDEXP2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<accel dp addexp> ::= <accel dp sbnd> <accel add opr> <accel dp sbnd>
   */
  case ACCEL_DP_ADDEXP1:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<accel dp mulexp> ::= <accel dp sbnd> <accel mul opr> <accel dp sbnd>
   */
  case ACCEL_DP_MULEXP1:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<accel add opr> ::= + |
   */
  case ACCEL_ADD_OPR1:
    break;
  /*
   *	<accel add opr> ::= -
   */
  case ACCEL_ADD_OPR2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<accel mul opr> ::= * |
   */
  case ACCEL_MUL_OPR1:
    break;
  /*
   *	<accel mul opr> ::= /
   */
  case ACCEL_MUL_OPR2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<accel dp bndexp1> ::= <accel dp mulexp> <accel add opr> <accel dp sbnd>
   */
  case ACCEL_DP_BNDEXP11:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<accel dp sbnd> ::= <constant> |
   */
  case ACCEL_DP_SBND1:
    break;
  /*
   *	<accel dp sbnd> ::= <ident> 
   */
  case ACCEL_DP_SBND2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<routine id list> ::= <ident> |
   */
  case ROUTINE_ID_LIST1:
    itemp = (ITEM *)getitem(0, sizeof(ITEM));
    itemp->next = ITEM_END;
    itemp->t.sptr = SST_SYMG(RHS(1));
    SST_BEGP(LHS, itemp);
    SST_ENDP(LHS, itemp);
    break;

  /*
   *	<routine id list> ::= <routine id list> , <ident>
   */
  case ROUTINE_ID_LIST2:
    itemp = (ITEM *)getitem(0, sizeof(ITEM));
    itemp->next = ITEM_END;
    itemp->t.sptr = SST_SYMG(RHS(3));
    SST_ENDG(RHS(1))->next = itemp;
    SST_ENDP(LHS, itemp);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <final> ::= <id>
   */
  case FINAL1:
    break;
  /* ------------------------------------------------------------------ */
  /*
   *      <opt tpsl> ::= |
   */
  case OPT_TPSL1:
    break;
  /*
   *      <opt tpsl> ::= ( <type param spec list> )
   */
  case OPT_TPSL2:
    sem.param_offset = 0;
    break;
  /* ------------------------------------------------------------------ */
  /*
   *      <type param spec list> ::= <type param spec list> , <id> |
   */
  case TYPE_PARAM_SPEC_LIST1:
    rhstop = 3;
    goto tpsl_shared;
  /*
   *      <type param spec list> ::= <id>
   */
  case TYPE_PARAM_SPEC_LIST2:
    rhstop = 1;
  tpsl_shared:
    sptr = SST_SYMG(RHS(rhstop));
    if (sem.extends && sem.param_offset == 0) {
      sem.param_offset = get_highest_param_offset(DTYPEG(sem.extends));
    }
    sem.param_offset += 1;
    queue_type_param(sptr, 0, sem.param_offset, 1);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <opt derived type spec> ::= |
   */
  case OPT_DERIVED_TYPE_SPEC1:
  /* fall thru */
  /*
   *      <opt derived type spec> ::= ( <type param decl list> )
   */
  case OPT_DERIVED_TYPE_SPEC2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <type param decl list> ::= <type param value> |
   */
  case TYPE_PARAM_DECL_LIST1:
    break;
  /*
   *      <type param decl list> ::= <type param decl list> , <type param value>
   */
  case TYPE_PARAM_DECL_LIST2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <type param value> ::= * |
   */
  case TYPE_PARAM_VALUE5:
    sem.param_assume_sz = 1;
    sem.param_defer_len = 0;
    goto param_comm;

  /*
   *      <type param value> ::= : |
   */
  case TYPE_PARAM_VALUE3:
    sem.param_assume_sz = 0;
    sem.param_defer_len = 1;
    goto param_comm;

  /*
   *      <type param value> ::= <expression> |
   */
  case TYPE_PARAM_VALUE1:
    sem.param_assume_sz = 0;
    sem.param_defer_len = 0;
  param_comm:
    if (sem.param_offset < 0) {
      error(155, 3, gbl.lineno, "A non keyword = type parameter specifier "
                                "cannot follow a keyword = type parameter "
                                "specifier",
            NULL);
    } else {
      sem.param_offset += 1;
      if (!sem.param_assume_sz && !sem.param_defer_len) {
        mkexpr(RHS(1));
        ast = SST_ASTG(RHS(1));
      } else {
        ast = 0;
      }
      if (A_TYPEG(ast) == A_CNST) {
        defer_put_kind_type_param(sem.param_offset, SST_CVALG(RHS(1)), NULL, 0,
                                  ast, 1);
      } else {
        defer_put_kind_type_param(sem.param_offset, -1, NULL, 0, ast, 1);
      }
    }
    break;
  /*
   *      <type param value> ::= <id name> = *
   */
  case TYPE_PARAM_VALUE6:
    sem.param_assume_sz = 1;
    sem.param_defer_len = 0;
    goto param_kwd_comm;
  /*
   *      <type param value> ::= <id name> = :
   */
  case TYPE_PARAM_VALUE4:
    sem.param_assume_sz = 0;
    sem.param_defer_len = 1;
    goto param_kwd_comm;

  /*
   *      <type param value> ::= <id name> = <expression>
   */
  case TYPE_PARAM_VALUE2:
    sem.param_assume_sz = 0;
    sem.param_defer_len = 0;
  param_kwd_comm:
    np = scn.id.name + SST_CVALG(RHS(1));
    sem.param_offset = -1;
    if (!sem.param_assume_sz && !sem.param_defer_len) {
      mkexpr(RHS(3));
      ast = SST_ASTG(RHS(3));
    } else {
      ast = 0;
    }
    if (A_TYPEG(ast) == A_CNST) {
      defer_put_kind_type_param(sem.param_offset, SST_CVALG(RHS(3)), np, 0, ast,
                                1);
    } else {
      defer_put_kind_type_param(sem.param_offset, -1, np, 0, ast, 1);
    }
    FLANG_FALLTHROUGH;
  /* ------------------------------------------------------------------ */
  /*
   *	<opt comma> ::= |
   */
  case OPT_COMMA1:
    break;
  /*
   *	<opt comma> ::= ,
   */
  case OPT_COMMA2:
    break;

    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<dimkeyword> ::= DIMENSION |
   */
  case DIMKEYWORD1:
    /* disallow DIMENSION in a structure */
    if (INSIDE_STRUCT &&
        (STSK_ENT(0).type != 'd' || scn.stmtyp != TK_SEQUENCE)) {
      error(117, 3, gbl.lineno,
            STSK_ENT(0).type == 's' ? "STRUCTURE" : "derived type", CNULL);
      sem.ignore_stmt = TRUE;
    }
    break;
  /*
   *	<dimkeyword> ::= <dimattr>
   */
  case DIMKEYWORD2:
    /* disallow DIMENSION in a structure */
    if (INSIDE_STRUCT &&
        (STSK_ENT(0).type != 'd' || scn.stmtyp != TK_SEQUENCE)) {
      error(117, 3, gbl.lineno,
            STSK_ENT(0).type == 's' ? "STRUCTURE" : "derived type", CNULL);
      sem.ignore_stmt = TRUE;
    }
    scn.stmtyp = TK_DIMENSION;
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <nis> ::=
   */
  case NIS1:
    /* "not inside structure" test; if inside a structure emit error
     * message and set flag to tell parser to skip over the current
     * statement.
     */
    /* need to allow SEQUENCE (a hpf spec) in derived types */
    if (INSIDE_STRUCT &&
        (STSK_ENT(0).type != 'd' || scn.stmtyp != TK_SEQUENCE)) {
      error(117, 3, gbl.lineno,
            STSK_ENT(0).type == 's' ? "STRUCTURE" : "derived type", CNULL);
      sem.ignore_stmt = TRUE;
    }
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <data type> ::= <base type> <opt len spec> |
   */
  case DATA_TYPE1:
    rhstop = 2;
    goto data_type_shared;
  /*
   *	  <data type> ::= <base type> ( <len kind> ) |
   */
  case DATA_TYPE2:
    rhstop = 3;
  data_type_shared:
    if (sem.deferred_func_len || sem.deferred_func_kind) {
      /* probably defined in a USEd module, wait until USE stmts have been
       * processed */
      break;
    }

    set_len_attributes(RHS(rhstop), 0);
    sem.gdtype =
        mod_type(sem.gdtype, sem.gty, lenspec[0].kind, lenspec[0].len, 0, 0);
    break;
  /*
   *      <data type> ::= CLASS <pgm> ( * )
   */
  case DATA_TYPE5:
    sptr = get_unl_poly_sym(0);
#if DEBUG
    assert(DTY(DTYPEG(sptr)) == TY_DERIVED && UNLPOLYG(DTY(DTYPEG(sptr) + 3)),
           "semant1: Invalid dtype for CLASS(*)", 0, 3);
#endif
    sem.class = 1;
    goto type_common;

  /*
   *      <data type> ::= CLASS  <pgm> ( <id> <opt derived type spec> )
   */
  case DATA_TYPE4:
    sptr = refsym((int)SST_SYMG(RHS(4)), OC_OTHER);
    sem.class = 1;
    goto type_common;
  /*
   *	<data type> ::= TYPE ( <id> <opt derived type spec> )
   */
  case DATA_TYPE3:
    sptr = refsym((int)SST_SYMG(RHS(3)), OC_OTHER);
  type_common:
    if (STYPEG(sptr) != ST_TYPEDEF) {
      if (STYPEG(sptr) == ST_USERGENERIC && GTYPEG(sptr)) {
        sptr = GTYPEG(sptr);
      } else if (STYPEG(sptr) == ST_UNKNOWN && sem.pgphase == PHASE_INIT) {
        sem.deferred_dertype = sptr;
        sem.deferred_kind_len_lineno = gbl.lineno;
        sptr = declsym(sptr, ST_TYPEDEF, TRUE);
      } else if (STYPEG(sptr) == ST_UNKNOWN &&
                 (scn.stmtyp == TK_IMPLICIT ||
                  (INSIDE_STRUCT && STSK_ENT(0).type == 'd'))) {
        /* assume a forward reference -- legal if the type
         * appears in an implicit statement or is a component
         * declaration with the POINTER attribute or if phase is
         * PHASE_INIT (in which case it could be a function return
         * type).
         */
        sptr = declsym(sptr, ST_TYPEDEF, TRUE);
        dtype = get_type(6, TY_DERIVED, NOSYM);
        DTYPEP(sptr, dtype);
        DTY(dtype + 2) = 0; /* will be filled in */
        DTY(dtype + 3) = sptr;
        DTY(dtype + 5) = 0;
      } else {
        /* recover by creating an empty typedef */
        error(155, 3, gbl.lineno, "Derived type has not been declared -",
              SYMNAME(sptr));
        sptr = insert_sym(sptr);
        sptr = declsym(sptr, ST_TYPEDEF, TRUE);
        dtype = get_type(6, TY_DERIVED, NOSYM);
        DTYPEP(sptr, dtype);
        DTY(dtype + 2) = 0; /* will be filled in */
        DTY(dtype + 3) = sptr;
        DTY(dtype + 5) = 0;
      }
    }

    else if (DTY(DTYPEG(sptr) + 1) <= NOSYM &&
             (!INSIDE_STRUCT || STSK_ENT(0).type != 'd')) {
      int tag;
      tag = DTY(DTYPEG(sptr) + 3);
    } else if (!sem.class && ABSTRACTG(sptr)) {
      error(155, 3, gbl.lineno, "illegal use of abstract type", SYMNAME(sptr));
    }
    if (!sem.type_mode || sem.stag_dtype != DTYPEG(sptr)) {
/* Do not call defer_put_kind_type_param() if this declaration
 * is part of a recursively typed component. The
 * defer_put_kind_type_param() call below processes all type parameters.
 * But in this case, the type has not yet been fully defined. So, we
 * need to handle this later.
 */
      sem.stag_dtype = DTYPEG(sptr);
      sem.gdtype = sem.ogdtype = sem.stag_dtype;
      defer_put_kind_type_param(0, 0, NULL, sem.stag_dtype, 0, 2);
    } else {
      sem.stag_dtype = DTYPEG(sptr);
      sem.gdtype = sem.ogdtype = sem.stag_dtype;
    }
    defer_put_kind_type_param(0, 0, NULL, 0, 0, 0);
    if (!sem.new_param_dt && has_type_parameter(sem.stag_dtype) &&
        defer_pt_decl(0, 2)) {
      /* In this case we're using just the default type
       * of a parameterized derived type. We need to make sure we
       * create another instance of it so we don't pollute the
       * default type that's shared across all instances of the type
       * (e.g., type(t) :: x may pollute type(t(5)) :: y ).
       */
      sem.new_param_dt = create_parameterized_dt(sem.stag_dtype, 0);
    }
    put_default_kind_type_param(
        (sem.new_param_dt) ? sem.new_param_dt : sem.stag_dtype, 0, 0);
    put_length_type_param(
        (sem.new_param_dt) ? sem.new_param_dt : sem.stag_dtype, 0);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<type spec> ::= <intrinsic type>
   */
  case TYPE_SPEC1:
    break;
  /*
   *	<type spec> ::= <ident>
   */
  case TYPE_SPEC2:
    SST_DTYPEP(LHS, sem.gdtype);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<intrinsic type> ::= <base type> <opt len spec> |
   */
  case INTRINSIC_TYPE1:
    rhstop = 2;
    if (sem.gdtype == DT_CHAR || sem.gdtype == DT_NCHAR) {
      if (SST_IDG(RHS(2)) == 0) {
        if (SST_ASTG(RHS(2)))
          sem.gcvlen = SST_ASTG(RHS(2));
        else if (SST_SYMG(RHS(2)) == -2 || SST_SYMG(RHS(2)) == -1)
          sem.gcvlen = astb.i1;
        else
          sem.gcvlen = mk_cval(SST_SYMG(RHS(2)), DT_INT4);

      } else {
        sem.gcvlen = SST_ASTG(RHS(2));
      }
    }
    goto intrinsic_type_shared;
  /*
   *	<intrinsic type> ::= <base type> ( <len kind> )
   */
  case INTRINSIC_TYPE2:
    rhstop = 3;
    if (sem.gdtype == DT_CHAR || sem.gdtype == DT_NCHAR) {
      if (SST_IDG(RHS(3)) == 0) {
        if (SST_ASTG(RHS(3)))
          sem.gcvlen = SST_ASTG(RHS(3));
        else if (SST_SYMG(RHS(3)) == -2 || SST_SYMG(RHS(3)) == -1)
          sem.gcvlen = astb.i1;
        else
          sem.gcvlen = mk_cval(SST_SYMG(RHS(3)), DT_INT4);

      } else {
        sem.gcvlen = SST_ASTG(RHS(3));
      }
    }

  intrinsic_type_shared:
    if (is_exe_stmt && sem.which_pass == 0)
      break;
    if (sem.deferred_func_len) {
      /* probably defined in a USEd module, wait USE stmts have been processed
       */
      break;
    }
    set_aclen(RHS(rhstop), 1, 1);
    sem.gdtype = mod_type(sem.gdtype, sem.gty, lenspec[1].kind, lenspec[1].len,
                          lenspec[1].propagated, 0);
    SST_DTYPEP(LHS, sem.gdtype);
    set_aclen(RHS(rhstop), 1, 0);
    SST_IDP(LHS, 0);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <base type> ::= INTEGER  |
   */
  case BASE_TYPE1:
    sem.gdtype = sem.ogdtype = stb.user.dt_int;
    sem.gty = TY_INT;
    break;
  /*
   *      <base type> ::= REAL     |
   */
  case BASE_TYPE2:
    sem.gdtype = sem.ogdtype = stb.user.dt_real;
    sem.gty = TY_REAL;
    break;
  /*
   *      <base type> ::= DOUBLEPRECISION |
   */
  case BASE_TYPE3:
    sem.gdtype = sem.ogdtype = DT_DBLE;
    sem.gty = TY_DBLE;
    if (XBIT(57, 0x10) && DTY(sem.gdtype) == TY_QUAD) {
      error(437, 2, gbl.lineno, "DOUBLE PRECISION", "REAL");
      sem.gdtype = DT_REAL;
    }
    break;
  /*
   *      <base type> ::= COMPLEX |
   */
  case BASE_TYPE4:
    sem.gdtype = sem.ogdtype = stb.user.dt_cmplx;
    sem.gty = TY_CMPLX;
    break;
  /*
   *      <base type> ::= DOUBLECOMPLEX   |
   */
  case BASE_TYPE5:
    if (flg.standard)
      error(171, 2, gbl.lineno, "DOUBLECOMPLEX", CNULL);
    sem.gdtype = sem.ogdtype = DT_DCMPLX;
    sem.gty = TY_DCMPLX;
    if (XBIT(57, 0x10) && DTY(sem.gdtype) == TY_DCMPLX) {
      error(437, 2, gbl.lineno, "DOUBLE COMPLEX", "COMPLEX");
      sem.gdtype = DT_CMPLX;
    }
    break;
  /*
   *      <base type> ::= LOGICAL  |
   */
  case BASE_TYPE6:
    sem.gdtype = sem.ogdtype = stb.user.dt_log;
    sem.gty = TY_LOG;
    break;
  /*
   *      <base type> ::= CHARACTER |
   */
  case BASE_TYPE7:
    sem.gdtype = sem.ogdtype = DT_CHAR;
    sem.gty = TY_CHAR;
    break;
  /*
   *      <base type> ::= NCHARACTER |
   */
  case BASE_TYPE8:
    if (flg.standard)
      error(171, 2, gbl.lineno, "NCHARACTER", CNULL);
    sem.gdtype = sem.ogdtype = DT_NCHAR;
    sem.gty = TY_NCHAR;
    break;
  /*
   *      <base type> ::= BYTE
   */
  case BASE_TYPE9:
    if (flg.standard)
      error(171, 2, gbl.lineno, "BYTE", CNULL);
    sem.gdtype = sem.ogdtype = DT_BINT;
    sem.gty = TY_BINT;
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <opt len spec> ::= |
   */
  case OPT_LEN_SPEC1:
    SST_IDP(LHS, 0);
    SST_SYMP(LHS, -1);
    SST_ASTP(LHS, 0);
    SST_DTYPEP(LHS, sem.gdtype);
    break;
  /*
   *      <opt len spec> ::= * <len spec>
   */
  case OPT_LEN_SPEC2:
    *LHS = *RHS(2);
    if (sem.ogdtype != DT_CHAR && flg.standard)
      errwarn(173);
    break;

  /*
   *      <opt len spec> ::= : <len spec>
   */
  case OPT_LEN_SPEC3:
    *LHS = *RHS(2);
    if (sem.ogdtype != DT_CHAR && flg.standard)
      errwarn(173);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <len spec> ::= <integer>  |
   */
  case LEN_SPEC1:    /* constant value set by scan */
    SST_IDP(LHS, 0); /* flag that an expression was seen */
    SST_ASTP(LHS, 0);
    goto len_spec;
  /*
   *      <len spec> ::= ( <tpv> ) |
   */
  case LEN_SPEC2:
    *LHS = *RHS(2);
  char_len_spec:
    if (sem.ogdtype != DT_CHAR && sem.ogdtype != DT_NCHAR)
      SST_SYMP(LHS, 0);
  len_spec:
    if (is_exe_stmt && sem.which_pass == 0)
      break;
    if (sem.ogdtype == DT_CHAR || sem.ogdtype == DT_NCHAR) {
      if (SST_IDG(LHS) == 0) {
        if (SST_CVALG(LHS) <= 0) {
          /* zero-size character - set flag */
          SST_SYMP(LHS, -2);
        }
      }
      break;
    }
    if (SST_IDG(LHS) == 0 && SST_SYMG(LHS) <= 0) {
      /* Cause error message to print later when context is known,
       * ensure that illegal value -1 doesn't map to internal
       * flag -1 for no length spec.
       */
      SST_SYMP(LHS, 99); /* cause error message displayed later */
    }
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<tpv> ::= <expression> |
   */
  case TPV1:
    if (is_exe_stmt && sem.which_pass == 0)
      break;
    if (chk_kind_parm(RHS(1))) {
      mkexpr(RHS(1)); /* Needed for type parameter */
      ast = SST_ASTG(RHS(1));
      switch (A_TYPEG(ast)) {
      case A_ID:
      case A_LABEL:
      case A_ENTRY:
      case A_SUBSCR:
      case A_SUBSTR:
      case A_MEM:
        /* Mark possible use of type parameter */
        sptr = sym_of_ast(ast);
        KINDP(sptr, -1);
        break;
      }
    }
    rhstop = 5;
    if (sem.ogdtype != DT_CHAR && sem.ogdtype != DT_NCHAR) {
      int offset;
      if (sem.pgphase <= PHASE_USE) {
        if (SST_IDG(top) == S_IDENT && STYPEG(SST_SYMG(top)) == ST_UNKNOWN) {
          /* probably defined in a USEd module, wait until USE stmts
           * have been processed */
          ast = SST_ASTG(RHS(1));
          if (!ast) {
            ast = mk_id(SST_SYMG(top));
          }
          sem.deferred_func_kind = ast;
          sem.deferred_kind_len_lineno = gbl.lineno;
          break;
        } else if (SST_IDG(top) == S_EXPR) {
          sem.deferred_func_kind = SST_ASTG(RHS(1));
          sem.deferred_kind_len_lineno = gbl.lineno;
          break;
        }
      }
      offset = chk_kind_parm(RHS(1));
      if (offset) {
        /* TO DO: Save length expression candidate like in DT_CHAR case */
        sem.type_param_candidate = offset;
        SST_SYMP(LHS, 4); /* place holder */
        sem.kind_candidate = (ITEM *)getitem(0, sizeof(ITEM));
        sem.kind_candidate->t.stkp = (SST *)getitem(0, sizeof(SST));
        *(sem.kind_candidate->t.stkp) = *RHS(1);
      } else
        SST_SYMP(LHS, chkcon(RHS(1), DT_INT4, TRUE));
    } else {
      int offset;
      offset = chk_kind_parm(RHS(1));
      if (offset) {
        sem.type_param_candidate = offset;
        sem.len_candidate = (ITEM *)getitem(0, sizeof(ITEM));
        sem.len_candidate->t.stkp = (SST *)getitem(0, sizeof(SST));
        *(sem.len_candidate->t.stkp) = *RHS(1);
        SST_SYMP(LHS, 1); /* place holder */
        SST_IDP(LHS, 0);  /* flag that a constant was seen */
        SST_ASTP(LHS, 0); /* not expression */
        break;
      }
      sem.len_candidate = 0;
      constant_lvalue(RHS(1));
      if (SST_IDG(RHS(1)) == S_CONST) {
        SST_SYMP(LHS, chkcon(RHS(1), DT_INT4, TRUE));
      } else {
        (void)chktyp(RHS(1), DT_INT, TRUE);
        ast = SST_ASTG(RHS(1));
        /* flag that an expression was seen: id field is 1, sym field
         * is non-zero, and ast field is the ast of the expression.
         */
        if (sem.pgphase == PHASE_INIT) {
          if (SST_IDG(top) == S_IDENT && STYPEG(SST_SYMG(top)) == ST_UNKNOWN) {
            /* probably defined in a USEd module,
             * wait until USE stmts have been processed */
            if (!ast) {
              ast = mk_id(SST_SYMG(top));
            }
            sem.deferred_func_len = ast;
            sem.deferred_kind_len_lineno = gbl.lineno;
            break;
          } else if (SST_IDG(top) == S_EXPR) {
            sem.deferred_func_len = SST_ASTG(RHS(1));
            sem.deferred_kind_len_lineno = gbl.lineno;
            break;
          }
        }

        SST_IDP(LHS, 1);
        SST_SYMP(LHS, _INF_CLEN);
        SST_ASTP(LHS, SST_ASTG(RHS(1)));
        break;
      }
    }

    SST_IDP(LHS, 0);  /* flag that a constant was seen */
    SST_ASTP(LHS, 0); /* not expression */
    break;
  /*
   *	<tpv> ::= *
   */
  case TPV2:
    /* flag that a '*' was seen: id field is 1, sym field is zero. */
    SST_IDP(LHS, 1);
    SST_SYMP(LHS, 0);
    SST_ASTP(LHS, 0); /* not expression */
    break;
  /*
   *	<tpv> ::= :
   */
  case TPV3:
    /* flag that a ':' was seen: id field is 1, sym field is -1. */
    SST_IDP(LHS, 1);
    SST_SYMP(LHS, -1);
    SST_ASTP(LHS, 0); /* not expression */
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<len kind> ::= <tpv> |
   */
  case LEN_KIND1:
    if (is_exe_stmt && sem.which_pass == 0)
      break;
    if (sem.deferred_func_kind) {
      /* probably defined in a USEd module, wait USE stmts have been processed
       */
      break;
    }

    if (sem.gdtype != DT_CHAR && sem.gdtype != DT_NCHAR) {
      sem.gdtype = select_kind(sem.gdtype, sem.gty, (INT)SST_SYMG(RHS(1)));
      SST_SYMP(LHS, -1);
      break;
    }
    goto len_spec;
  /*
   *	<len kind> ::= <len kind spec> |
   */
  case LEN_KIND2:
    if (is_exe_stmt && sem.which_pass == 0)
      break;
    switch (SST_FLAGG(RHS(1))) {
    case 0: /* error */
      break;
    case 1: /* LEN = */
      if (sem.ogdtype == DT_CHAR)
        goto char_len_spec;
      error(81, 3, gbl.lineno,
            "- LEN = cannot be specified with non-character type", CNULL);
      break;
    case 2: /* KIND = */
      sem.gdtype = select_kind(sem.gdtype, sem.gty, (INT)SST_SYMG(RHS(1)));
      break;
    }
    SST_SYMP(LHS, -1);
    break;
  /*
   *	<len kind> ::= <tpv> , <len kind spec>|
   */
  case LEN_KIND3: /* len, kind = ... */
    if (is_exe_stmt && sem.which_pass == 0)
      break;
    if (sem.ogdtype != DT_CHAR) {
      error(81, 3, gbl.lineno, "- LEN and KIND with non-character type", CNULL);
      SST_SYMP(LHS, -1); /* an error occurred - null processing */
      break;
    }
    switch (SST_FLAGG(RHS(3))) {
    case 0: /* error */
      break;
    case 1: /* LEN = */
      error(81, 3, gbl.lineno, "- Repeated LEN", CNULL);
      break;
    case 2: /* KIND = */
      sem.gdtype = select_kind(sem.gdtype, sem.gty, (INT)SST_SYMG(RHS(3)));
      break;
    }
    goto char_len_spec;
  /*
   *	<len kind> ::= <tpv> , <tpv> |
   */
  case LEN_KIND4: /* len, kind */
    if (is_exe_stmt && sem.which_pass == 0)
      break;
    if (sem.ogdtype != DT_CHAR) {
      error(81, 3, gbl.lineno, "- LEN and KIND with non-character type", CNULL);
      SST_SYMP(LHS, -1); /* an error occurred - null processing */
      break;
    }
    sem.gdtype = select_kind(sem.gdtype, sem.gty, (INT)SST_SYMG(RHS(3)));
    goto char_len_spec;
  /*
   *	<len kind> ::= <len kind spec> , <len kind spec>
   */
  case LEN_KIND5: /* len = .., kind = ... or kind = ..., len = ... */
    if (is_exe_stmt && sem.which_pass == 0)
      break;
    if (sem.ogdtype != DT_CHAR) {
      error(81, 3, gbl.lineno, "- LEN and KIND with non-character type", CNULL);
      SST_SYMP(LHS, -1); /* an error occurred - null processing */
      break;
    }
    switch (SST_FLAGG(RHS(1))) {
    default: /* error */
      break;
    case 1: /* LEN = */
      switch (SST_FLAGG(RHS(3))) {
      case 0: /* error */
        break;
      case 1: /* LEN = */
        error(81, 3, gbl.lineno, "- Repeated LEN =", CNULL);
        break;
      case 2: /* KIND = */
        sem.gdtype = select_kind(sem.gdtype, sem.gty, (INT)SST_SYMG(RHS(3)));
        goto char_len_spec;
      }
      break;
    case 2: /* KIND = */
      switch (SST_FLAGG(RHS(3))) {
      case 0: /* error */
        break;
      case 1: /* LEN = */
        sem.gdtype = select_kind(sem.gdtype, sem.gty, (INT)SST_SYMG(RHS(1)));
        *LHS = *RHS(3);
        goto char_len_spec;
      case 2: /* KIND = */
        error(81, 3, gbl.lineno, "- Repeated KIND =", CNULL);
        break;
      }
      break;
    }
    SST_SYMP(LHS, -1); /* an error occurred - null processing */
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<len kind spec> ::= <id name> = <tpv>
   */
  case LEN_KIND_SPEC1:
    np = scn.id.name + SST_CVALG(RHS(1));
    *LHS = *RHS(3);
    if (is_exe_stmt && sem.which_pass == 0)
      break;
    SST_FLAGP(LHS, 0);
    if (sem_strcmp(np, "len") == 0) {
      SST_FLAGP(LHS, 1);
      if (sem.type_param_candidate && sem.len_candidate) {
        sem.len_type_param = sem.type_param_candidate;
        sem.type_param_candidate = 0;
        mkexpr(sem.len_candidate->t.stkp);
        ast = SST_ASTG(sem.len_candidate->t.stkp);
        if (A_TYPEG(ast) != A_CNST) {
          /* set ignore flag on any len type parameters to prevent
           * "implicit none" errors
           */
          chk_len_parm_expr(ast, 0, 1);
        }
      }
    } else if (sem_strcmp(np, "kind") == 0) {
      sem.kind_type_param = sem.type_param_candidate;
      sem.type_param_candidate = 0;
      if (!sem.deferred_func_kind) {
        if (SST_IDG(RHS(3))) {
          if (SST_ASTG(RHS(3)))
            errsev(87);
          else
            error(81, 3, gbl.lineno, "- KIND = *", CNULL);
        } else
          SST_FLAGP(LHS, 2);
      }
    } else {
      error(34, 3, gbl.lineno, np, CNULL);
    }
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <optional comma> ::= |
   */
  case OPTIONAL_COMMA1:
    break;
  /*
   *      <optional comma> ::= ,
   */
  case OPTIONAL_COMMA2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<opt attr> ::=   |
   */
  case OPT_ATTR1:
    break;
  /*
   *	<opt attr> ::= ::
   */
  case OPT_ATTR2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <typdcl list> ::= <typdcl list> , <typdcl item> |
   */
  case TYPDCL_LIST1:
    break;
  /*
   *      <typdcl list> ::= <typdcl item>
   */
  case TYPDCL_LIST2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <typdcl item> ::= <dcl id> / <dinit const list> / |
   */
  case TYPDCL_ITEM1:
    if (flg.standard)
      errwarn(174);
    inited = TRUE;
    goto typ_dcl_item;
  /*
   *      <typdcl item> ::= <dcl id>
   */
  case TYPDCL_ITEM2:
    inited = FALSE;
  typ_dcl_item:
    sptr = SST_SYMG(RHS(1));
    if (flg.xref)
      xrefput(sptr, 'd');
    dtype = mod_type(sem.gdtype, sem.gty, lenspec[1].kind, lenspec[1].len,
                     lenspec[1].propagated, sptr);
    if (!DCLDG(sptr)) {
      switch (STYPEG(sptr)) {
      /* any cases for which a type must be identical to the variable's
       * implicit type.
       */
      case ST_PARAM:
        if (DTYPEG(sptr) != dtype)
          error(37, 3, gbl.lineno, SYMNAME(sptr), CNULL);
        break;
      default:
        break;
      }
    }
  common_typespecs:
    if (DCLDG(sptr)) {
      switch (STYPEG(sptr)) {
      /*  any cases for which a data type does not apply */
      case ST_MODULE:
      case ST_NML:
        error(44, 3, gbl.lineno, SYMNAME(sptr), CNULL);
        break;
      default:
        /* data type for ident has already been specified */
        if (DDTG(DTYPEG(sptr)) == dtype)
          error(119, 2, gbl.lineno, SYMNAME(sptr), CNULL);
        else if (DTY(DTYPEG(sptr)) == TY_PTR &&
                 DTY(DTY(DTYPEG(sptr) + 1)) == TY_PROC &&
                 DTY(DTY(DTYPEG(sptr) + 1) + 1) == DT_NONE &&
                 DTY(DTY(DTYPEG(sptr) + 1) + 2) == 0) {
          /* ptr to procedure, return dtype is DT_NONE, no interface; just
           * update the return dtype (no longer assume it's a pointer to a
           * subroutine).
           */
          DTY(DTY(DTYPEG(sptr) + 1) + 1) = dtype;
        } else {
          error(37, 3, gbl.lineno, SYMNAME(sptr), CNULL);
        }
      }
      break; /* to avoid setting symbol table entry's stype field */
    }

    DCLDP(sptr, TRUE);

    /* Procedure pointer without a declared type (combination of "external" and
     * "pointer" attributes) */
    if (is_procedure_ptr_dtype(DTYPEG(sptr))) {
      set_proc_ptr_result_dtype(DTYPEG(sptr), dtype);
      /* Avoid the rest */
      break;
    }

    /* Procedure without a type ("external" attribute) */
    if (is_procedure_dtype(DTYPEG(sptr))) {
      set_proc_result_dtype(DTYPEG(sptr), dtype);
      /* Avoid the rest */
      break;
    }

    set_char_attributes(sptr, &dtype);
    if (DTY(DTYPEG(sptr)) == TY_ARRAY) {
      DTY(DTYPEG(sptr) + 1) = dtype;
      if (DTY(dtype) == TY_DERIVED && DTY(dtype + 3) &&
          DISTMEMG(DTY(dtype + 3))) {
        error(451, 3, gbl.lineno, SYMNAME(sptr), CNULL);
      }
    } else {
      DTYPEP(sptr, dtype);
    }
    if (STYPEG(sptr) == ST_ENTRY && FVALG(sptr)) {
#if DEBUG
      interr("semant1: data type set for ST_ENTRY with FVAL", sptr, 3);
#endif
      DCLDP(FVALG(sptr), TRUE);
      DTYPEP(FVALG(sptr), DTYPEG(sptr));
      set_char_attributes(FVALG(sptr), &dtype);
    }
    if (STYPEG(sptr) != ST_ENTRY && STYPEG(sptr) != ST_MEMBER &&
        RESULTG(sptr)) {
      /* set the type for the entry point as well */
      copy_type_to_entry(sptr);
    }
    if (inited) { /* check if symbol is data initialized */
      gen_dinit(sptr, RHS(3));
    } else if (DTY(DDTG(dtype)) == TY_DERIVED && !POINTERG(sptr) &&
               !ADJARRG(sptr) && !ALLOCG(sptr) && SCG(sptr) != SC_DUMMY) {
      int dt_dtype = DDTG(dtype);
      if (INSIDE_STRUCT) {
        /* Uninitialized declaration of a derived type data item.
         * Check for and handle any component intializations defined
         * for this derived type */
        build_typedef_init_tree(sptr, dt_dtype);
      } else if (DTY(dt_dtype + 5) && SCOPEG(sptr) &&
                 SCOPEG(sptr) == stb.curr_scope &&
                 STYPEG(stb.curr_scope) == ST_MODULE) {
        /*
         * a derived type module variable has component initializers,
         * so generate inits.
         */
        build_typedef_init_tree(sptr, dt_dtype);
      }
    }

    break;

  /*
   *      <typdcl item> ::= %FILL
   */
  case TYPDCL_ITEM3:
    if (flg.standard)
      error(176, 2, gbl.lineno, "%FILL", CNULL);
    if (sem.stsk_depth == 0)
      errwarn(145);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <dcl id list> ::= <dcl id list> , <dcl id> |
   */
  case DCL_ID_LIST1:
    rhstop = 3;
    goto dcl_id_list;
  /*
   *      <dcl id list> ::= <dcl id> |
   */
  case DCL_ID_LIST2:
    rhstop = 1;
  /* Shared by DIMENSION and COMMON statements */
  dcl_id_list:
    sptr = SST_SYMG(RHS(rhstop));
    if (lenspec[1].kind)
      error(32, 2, gbl.lineno, SYMNAME(sptr), CNULL);
    if (flg.xref)
      xrefput(sptr, 'd');
    if (scn.stmtyp == TK_COMMON) {
      /* COMMON block defn: link symbol into list */
      {
        itemp = (ITEM *)getitem(0, sizeof(ITEM));
        itemp->next = ITEM_END;
        itemp->t.sptr = sptr;
        if (rhstop == 1)
          /* adding first common block item to list: */
          SST_BEGP(LHS, itemp);
        else
          SST_ENDG(RHS(1))->next = itemp;
      }
      SST_ENDP(LHS, itemp);
    } else {
#if DEBUG
      assert(scn.stmtyp == TK_DIMENSION, "semant:unexp.stmt-dcl_id_lis",
             scn.stmtyp, 3);
#endif
    }
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <dcl id> ::= <ident> <opt len spec>  |
   */
  case DCL_ID1:
    set_len_attributes(RHS(2), 1);
    stype = ST_IDENT;
    sptr = SST_SYMG(RHS(1));
    if (STYPEG(sptr) == ST_ENTRY && FVALG(sptr))
      sptr = FVALG(sptr);
    if (test_scope(sptr) == sem.scope_level && STYPEG(sptr) != ST_MEMBER) {
      dtype = DTYPEG(sptr);
    } else {
      dtype = 0;
    }
    sem.dinit_count = 1;
    goto dcl_shared;
  /*
   *      <dcl id> ::= <ident> <opt len spec> <dim beg> <dimension list> ) <opt
   * len spec>
   */
  case DCL_ID2:
    /* Send len spec up with ident on semantic stack */
    if (SST_SYMG(RHS(6)) != -1) {
      if (SST_SYMG(RHS(2)) != -1)
        errsev(46);
      set_len_attributes(RHS(6), 1);
    } else
      set_len_attributes(RHS(2), 1);
    stype = ST_ARRAY;
    dtype = SST_DTYPEG(RHS(4));
    ad = AD_DPTR(dtype);
    if (AD_ASSUMSZ(ad) || AD_ADJARR(ad) || AD_DEFER(ad) || sem.interface)
      sem.dinit_count = -1;
    else
      sem.dinit_count = ad_val_of(sym_of_ast(AD_NUMELM(AD_DPTR(dtype))));
  dcl_shared:
    sptr = SST_SYMG(RHS(1));
    if (!(entity_attr.exist & ET_B(ET_BIND))) {
      sptr = block_local_sym(SST_SYMG(RHS(1)));
      SST_SYMP(RHS(1), sptr);
    }
    if (!sem.which_pass && gbl.internal > 1) {
      decr_ident_use(sptr, ident_host_sub);
    }
    if (!sem.kind_type_param && !sem.len_type_param &&
        sem.type_param_candidate) {
      sem.kind_type_param = sem.type_param_candidate;
      sem.type_param_candidate = 0;
    }
    if (INSIDE_STRUCT) {
      if (STYPEG(sptr) != ST_UNKNOWN)
        SST_SYMP(LHS, (sptr = insert_sym(sptr)));
      if (sem.kind_type_param) {
        USEKINDP(sptr, 1);
        KINDP(sptr, sem.kind_type_param);
        if (sem.kind_candidate) {
          /* Save kind expression in component */
          mkexpr(sem.kind_candidate->t.stkp);
          KINDASTP(sptr, SST_ASTG(sem.kind_candidate->t.stkp));
        }
      }
      if (sem.len_type_param) {
        USELENP(sptr, 1);
        LENP(sptr, sem.len_type_param);
      }
      SYMLKP(sptr, NOSYM);
      STYPEP(sptr, ST_MEMBER);
      /* if the dtype was determined from the symbol table entry then it
       * is incorrect (because we got a new symbol entry above).
       */
      if (stype == ST_IDENT)
        dtype = sem.gdtype;

      if (sem.gdtype != -1 && DTY(sem.gdtype) == TY_DERIVED &&
          (STSK_ENT(0).type == 'd')) {
        stsk = &STSK_ENT(0);
        /* if outer derived type has SEQUENCE then nested one should */
        if (SEQG(DTY(stsk->dtype + 3)) && !SEQG(DTY(sem.gdtype + 3))) {
          error(155, 3, gbl.lineno,
                "SEQUENCE must be set for nested derived type",
                SYMNAME(DTY(sem.gdtype + 3)));
        }
        if (DTY(stsk->dtype + 3) == DTY(sem.gdtype + 3)) {
          error(155, 3, gbl.lineno,
                "Derived type component must have the POINTER attribute -",
                SYMNAME(sptr));
        } else if (!DCLDG(DTY(sem.gdtype + 3)))
          error(155, 3, gbl.lineno, "Derived type has not been declared -",
                SYMNAME(DTY(sem.gdtype + 3)));
      }

      DTYPEP(sptr, dtype); /* must be done before link members */
      /* link field-namelist into member list at this level */
      stsk = &STSK_ENT(0);
      link_members(stsk, sptr);
      if (stype == ST_ARRAY && STSK_ENT(0).type != 'd' &&
          (AD_ASSUMSZ(ad) || AD_ADJARR(ad) || AD_DEFER(ad)))
        error(50, 3, gbl.lineno, SYMNAME(sptr), CNULL);
      if (DTY(dtype) == TY_ARRAY) {
        int d;
        d = DTY(dtype + 1);
        if (DTY(d) == TY_DERIVED && DTY(d + 3) && DISTMEMG(DTY(d + 3))) {
          error(451, 3, gbl.lineno, SYMNAME(sptr), CNULL);
        }
        if (AD_ASSUMSZ(ad) || AD_ADJARR(ad) || AD_DEFER(ad)) {
          if (!ALLOCG(sptr) && AD_ADJARR(ad)) {
            int bndast, badArray;
            int numdim = AD_NUMDIM(ad);
            for (badArray = i = 0; i < numdim; i++) {
              bndast = AD_LWAST(ad, i);
              badArray = !chk_len_parm_expr(bndast, ENCLDTYPEG(sptr), 0);
              if (!badArray) {
                bndast = AD_UPAST(ad, i);
                badArray = !chk_len_parm_expr(bndast, ENCLDTYPEG(sptr), 0);
                if (!badArray) {
                  ADJARRP(sptr, 1);
                  USELENP(sptr, 1);
                  break;
                }
              }
            }
            if (badArray) {
              for (badArray = i = 0; i < numdim; i++) {
                bndast = AD_LWAST(ad, i);
                badArray = !chk_kind_parm_expr(bndast, ENCLDTYPEG(sptr), 1, 0);
                if (badArray) {
                  badArray = !chk_len_parm_expr(bndast, ENCLDTYPEG(sptr), 1);
                  if (!badArray) {
                    ADJARRP(sptr, 1);
                    USELENP(sptr, 1);
                    break;
                  }
                }
                if (badArray)
                  goto illegal_array_member;
                bndast = AD_UPAST(ad, i);
                badArray = !chk_kind_parm_expr(bndast, ENCLDTYPEG(sptr), 1, 0);
                if (badArray) {
                  badArray = !chk_len_parm_expr(bndast, ENCLDTYPEG(sptr), 1);
                  if (!badArray) {
                    ADJARRP(sptr, 1);
                    USELENP(sptr, 1);
                    break;
                  }
                } else if (A_TYPEG(bndast) != A_ID &&
                           A_TYPEG(bndast) != A_CNST) {

                  ADJARRP(sptr, 1);
                  USELENP(sptr, 1);
                  if (!chk_len_parm_expr(bndast, ENCLDTYPEG(sptr), 1)) {
                    USEKINDP(sptr, 1);
                  }
                  break;
                }
                if (badArray)
                  goto illegal_array_member;
              }
            }
          } else if (!ALLOCG(sptr)) {
          illegal_array_member:
            error(134, 3, gbl.lineno,
                  "- deferred shape array must have the POINTER "
                  "attribute in a derived type",
                  SYMNAME(sptr));
            ALLOCP(sptr, 1);
          }
        }
      }
      if (XBIT(58, 0x10000) && !F90POINTERG(sptr)) {
        /* we are processing a member, and we must handle all pointers
         * do we need descriptors for this member? */
        if (POINTERG(sptr) || ALLOCG(sptr) ||
#ifdef USELENG
            USELENG(sptr) ||
#endif
            (STYPEG(sptr) != ST_MEMBER && (ADJARRG(sptr) || RUNTIMEG(sptr)))) {
          get_static_descriptor(sptr);
          get_all_descriptors(sptr);
          SCP(sptr, SC_BASED);
        }
      }
    } else {
      sptr = create_var(sptr);
      SST_SYMP(LHS, sptr);
      stype1 = STYPEG(sptr);
      if (sem.kind_type_param) {
        USEKINDP(sptr, 1);
        KINDP(sptr, sem.kind_type_param);
      }
      if (sem.len_type_param) {
        USELENP(sptr, 1);
        LENP(sptr, sem.len_type_param);
      }

      if (DTY(sem.stag_dtype) == TY_DERIVED && sem.class) {
        /* TBD - Probably need to fix this condition when we
         * support unlimited polymorphic entities.
         */
        if (SCG(sptr) == SC_DUMMY || POINTERG(sptr) || ALLOCG(sptr)) {
          CLASSP(sptr, 1); /* mark polymorphic variable */
          if (PASSBYVALG(sptr)) {
            error(155, 3, gbl.lineno, "Polymorphic variable cannot have VALUE "
                                      "attribute -",
                  SYMNAME(sptr));
          }
          if (DTY(sem.stag_dtype) == TY_DERIVED) {
            int tag = DTY(sem.stag_dtype + 3);
            if (CFUNCG(tag)) {
              error(155, 3, gbl.lineno,
                    "Polymorphic variable cannot be declared "
                    "with a BIND(C) derived type - ",
                    SYMNAME(sptr));
            }
            if (SEQG(tag)) {
              error(155, 3, gbl.lineno,
                    "Polymorphic variable cannot be declared "
                    "with a SEQUENCE derived type - ",
                    SYMNAME(sptr));
            }
          }

        } else {
          error(155, 3, gbl.lineno, "Polymorphic variable must be a pointer, "
                                    "allocatable, or dummy object - ",
                SYMNAME(sptr));
        }
      }
      if (DTY(sem.stag_dtype) == TY_DERIVED && sem.which_pass &&
          !(entity_attr.exist & (ET_B(ET_POINTER) | ET_B(ET_ALLOCATABLE))) &&
          SCG(sptr) != SC_DUMMY && !FVALG(sptr) &&
          (gbl.rutype != RU_PROG || CONSTRUCTSYMG(sptr))) {
        add_auto_finalize(sptr);
      }
      if (dtype == 0)
        dtype = DTYPEG(sptr);
      /* Assertion:
       *  stype  = stype we want to make symbol {ARRAY,STRUCT,or IDENT}
       *	stype1 = symbol's current stype
       */
      if (stype == ST_ARRAY) {
        if (IS_INTRINSIC(stype1)) {
          /* Changing intrinsic symbol to ARRAY */
          if ((sptr = newsym(sptr)) == 0)
            /* Symbol frozen as an intrinsic, ignore type decl */
            break;
          SST_SYMP(LHS, sptr);
          /* Cause STYPE and DTYPE to change AFTER fixing dtype */
          stype1 = ST_UNKNOWN;
        } else
          switch (stype1) {
          case ST_UNKNOWN:
          case ST_IDENT:
          case ST_VAR:
          case ST_STRUCT:
            break;
          case ST_ENTRY:
            if (DTY(DTYPEG(sptr)) != TY_ARRAY)
              break;
            error(43, 3, gbl.lineno, "symbol", SYMNAME(sptr));
            goto dcl_shared_end;
          case ST_ARRAY: {
            /* if symbol is already an array, check if the
             * dimension specifiers are identical.
             */
            ADSC *ad1, *ad2;
            int ndim;

            ad1 = AD_DPTR(DTYPEG(sptr));
            ad2 = AD_DPTR(dtype);
            ndim = AD_NUMDIM(ad1);
            if (ndim != AD_NUMDIM(ad2)) {
              error(43, 3, gbl.lineno, "symbol", SYMNAME(sptr));
              goto dcl_shared_end;
            }
            for (i = 0; i < ndim; i++)
              if (AD_LWBD(ad1, i) != AD_LWBD(ad2, i) ||
                  AD_UPBD(ad1, i) != AD_UPBD(ad2, i))
                break;
            if (i < ndim) {
              error(43, 3, gbl.lineno, "symbol", SYMNAME(sptr));
              goto dcl_shared_end;
            }
          }
            error(119, 2, gbl.lineno, SYMNAME(sptr), CNULL);
            break;
          default:
            error(43, 3, gbl.lineno, "symbol", SYMNAME(sptr));
            goto dcl_shared_end;
          }
        DTY(dtype + 1) = DTYPEG(sptr);
      } else if (stype == ST_STRUCT) {
        if (IS_INTRINSIC(stype1)) {
          /* Changing intrinsic symbol to STRUCT */
          if ((sptr = newsym(sptr)) == 0)
            /* Symbol frozen as an intrinsic, ignore type decl */
            break;
          SST_SYMP(LHS, sptr);
          /* Cause STYPE and DTYPE to change AFTER fixing dtype */
          stype1 = ST_UNKNOWN;
        } else if (stype1 == ST_ARRAY && DCLDG(sptr) == 0) {
          /* this case is OK */
        } else if (stype1 != ST_UNKNOWN && stype1 != ST_IDENT) {
          error(43, 3, gbl.lineno, "symbol", SYMNAME(sptr));
          break;
        }
      } else if ((scn.stmtyp == TK_COMMON || scn.stmtyp == TK_POINTER) &&
                 IS_INTRINSIC(stype1)) {
        /* Changing intrinsic symbol to IDENT in COMMON/POINTER */
        if ((sptr = newsym(sptr)) == 0)
          /* Symbol frozen as an intrinsic, ignore in COMMON */
          break;
        SST_SYMP(LHS, sptr);
        /* Cause STYPE and DTYPE to change AFTER fixing dtype */
        stype1 = ST_UNKNOWN;
        dtype = DTYPEG(sptr);
      } else if (IN_MODULE_SPEC && !sem.interface && IS_INTRINSIC(stype1)) {
        /* Changing intrinsic symbol to IDENT in module specification */
        if ((sptr = newsym(sptr)) == 0)
          /* Symbol frozen as an intrinsic, ignore in COMMON */
          break;
        SST_SYMP(LHS, sptr);
        /* Cause STYPE and DTYPE to change AFTER fixing dtype */
        stype1 = ST_UNKNOWN;
        dtype = DTYPEG(sptr);
      }
      /*
       * The symbol's stype and data type can only be changed if
       * it is new or if the type is changing from an identifier or
       * structure to an array.  The latter can occur because of the
       * separation of type/record declarations from DIMENSION/COMMON
       * statements.  If the symbol is a record, its stype can change
       * only if it's an identifier; note, that its dtype will be
       * set (and checked) by the semantic actions for record.
       */
      if (stype1 == ST_UNKNOWN ||
          (stype == ST_ARRAY &&
           (stype1 == ST_IDENT || stype1 == ST_VAR || stype1 == ST_STRUCT))) {
        STYPEP(sptr, stype);
        DTYPEP(sptr, dtype);
        if (DTY(dtype) == TY_ARRAY) {
          int d;
          d = DTY(dtype + 1);
          if (DTY(d) == TY_DERIVED && DTY(d + 3) && DISTMEMG(DTY(d + 3))) {
            error(451, 3, gbl.lineno, SYMNAME(sptr), CNULL);
          }
        }
        if (stype == ST_ARRAY) {
          if (POINTERG(sptr)) {
            if (!AD_DEFER(ad) || AD_ASSUMSHP(ad))
              error(196, 3, gbl.lineno, SYMNAME(sptr), CNULL);
            if (SCG(sptr) != SC_DUMMY)
              ALLOCP(sptr, 1);
            if (!F90POINTERG(sptr)) {
              get_static_descriptor(sptr);
              get_all_descriptors(sptr);
            }
          } else if (AD_ASSUMSZ(ad)) {
            if (SCG(sptr) != SC_NONE && SCG(sptr) != SC_DUMMY &&
                SCG(sptr) != SC_BASED)
              error(50, 3, gbl.lineno, SYMNAME(sptr), CNULL);
            ASUMSZP(sptr, 1);
            SEQP(sptr, 1);
          }
          if (AD_ADJARR(ad)) {
            ADJARRP(sptr, 1);
            /*
             * mark the adjustable array if the declaration
             * occurs after an ENTRY statement.
             */
            if (entry_seen)
              AFTENTP(sptr, 1);
          } else if (!POINTERG(sptr) && AD_DEFER(ad)) {
            if (SCG(sptr) == SC_CMBLK)
              error(43, 3, gbl.lineno, "deferred shape array", SYMNAME(sptr));
            if (SCG(sptr) == SC_DUMMY) {
              mk_assumed_shape(sptr);
              ASSUMSHPP(sptr, 1);
              if (!XBIT(54, 2) && !(XBIT(58, 0x400000) && TARGETG(sptr)))
                SDSCS1P(sptr, 1);
            } else {
              if (AD_ASSUMSHP(ad)) {
                /* this is an error if it isn't a dummy; the
                 * declaration could occur before its entry, so
                 * the check needs to be performed in semfin.
                 */
                ASSUMSHPP(sptr, 1);
                if (!XBIT(54, 2))
                  SDSCS1P(sptr, 1);
              }
              ALLOCP(sptr, 1);
              mk_defer_shape(sptr);
            }
          }
        }
      } else if (sem.gdtype != -1 && DTY(sem.gdtype) == TY_DERIVED) {
        if (stype1 == ST_ENTRY) {
          if (FVALG(sptr)) {
/* should not reach this point */
#if DEBUG
            interr("semant1: trying to set data type of ST_ENTRY", sptr, 3);
#endif
            sptr = FVALG(sptr);
          } else {
            error(43, 3, gbl.lineno, "subprogram or entry", SYMNAME(sptr));
            sptr = insert_sym(sptr);
          }
        }
        if (stype == ST_ARRAY && RESULTG(sptr)) {
          DTYPEP(sptr, dtype);
          if (POINTERG(sptr)) {
            if (!AD_DEFER(ad) || AD_ASSUMSHP(ad))
              error(196, 3, gbl.lineno, SYMNAME(sptr), CNULL);
          } else if (AD_ASSUMSZ(ad)) {
            ASUMSZP(sptr, 1);
            SEQP(sptr, 1);
          } else if (AD_ADJARR(ad))
            ADJARRP(sptr, 1);
          else if (AD_DEFER(ad)) {
            mk_assumed_shape(sptr);
            ASSUMSHPP(sptr, 1);
            if (!XBIT(54, 2) && !(XBIT(58, 0x400000) && TARGETG(sptr)))
              SDSCS1P(sptr, 1);
            AD_ASSUMSHP(ad) = 1;
          }
          copy_type_to_entry(sptr);
        }
      } else if (stype == ST_STRUCT && stype1 == ST_IDENT)
        STYPEP(sptr, ST_STRUCT);
      else if (stype == ST_ARRAY) {
        if (stype1 == ST_ENTRY) {
          if (FVALG(sptr)) {
/* should not reach this point */
#if DEBUG
            interr("semant1: trying to set data type of ST_ENTRY", sptr, 3);
#endif
            sptr = FVALG(sptr);
          } else {
            error(43, 3, gbl.lineno, "subprogram or entry", SYMNAME(sptr));
            sptr = insert_sym(sptr);
          }
        }
        if (RESULTG(sptr)) {
          DTYPEP(sptr, dtype);
          if (POINTERG(sptr)) {
            if (!AD_DEFER(ad) || AD_ASSUMSHP(ad))
              error(196, 3, gbl.lineno, SYMNAME(sptr), CNULL);
          } else if (AD_ASSUMSZ(ad)) {
            ASUMSZP(sptr, 1);
            SEQP(sptr, 1);
          } else if (AD_ADJARR(ad))
            ADJARRP(sptr, 1);
          else if (AD_DEFER(ad)) {
            mk_assumed_shape(sptr);
            ASSUMSHPP(sptr, 1);
            if (!XBIT(54, 2) && !(XBIT(58, 0x400000) && TARGETG(sptr)))
              SDSCS1P(sptr, 1);
            AD_ASSUMSHP(ad) = 1;
          }
          copy_type_to_entry(sptr);
        }
      }
    }
  dcl_shared_end:
    if (STYPEG(sptr) != ST_ENTRY && STYPEG(sptr) != ST_MEMBER &&
        RESULTG(sptr)) {
      /* set the type for the entry point as well */
      copy_type_to_entry(sptr);
    }
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <dim beg> ::= (
   */
  case DIM_BEG1:
    sem.in_dim = 1;
    sem.arrdim.ndim = 0;
    sem.arrdim.ndefer = 0;
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <dimension list> ::= <dim list>
   */
  case DIMENSION_LIST1:

    sem.in_dim = 0;
    dtype = mk_arrdsc(); /* semutil2.c */
    SST_DTYPEP(LHS, dtype);
    break;
  /* ------------------------------------------------------------------ */
  /*
   *      <dim list> ::= <dim list> , <dim spec> |
   */
  case DIM_LIST1:
    break;
  /*
   *      <dim list> ::= <dim spec>
   */
  case DIM_LIST2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <dim spec> ::= <explicit shape> |
   */
  case DIM_SPEC1:
    break;
  /*
   *      <dim spec> ::= <expression> : *  |
   */
  case DIM_SPEC2:
    rhstop = 3;
    SST_IDP(RHS(3), S_STAR);
    goto dim_spec;
  /*
   *      <dim spec> ::= *
   */
  case DIM_SPEC3:
    rhstop = 1;
    SST_IDP(RHS(1), S_STAR);
  dim_spec:
    if (sem.arrdim.ndim >= MAXDIMS) {
      error(47, 3, gbl.lineno, CNULL, CNULL);
      break;
    }

    /* check upper bound expression */
    constarraysize = 1;
    arraysize = 0;

    constant_lvalue(RHS(rhstop));
    if (SST_IDG(RHS(rhstop)) == S_CONST) {
      sem.bounds[sem.arrdim.ndim].uptype = S_CONST;
      if (flg.standard) {
        int uptyp;
        uptyp = SST_DTYPEG(RHS(rhstop));
        if (!DT_ISINT(uptyp)) {
          error(170, 2, gbl.lineno, "array upper bound", "is not integer");
        }
      }
      arraysize = sem.bounds[sem.arrdim.ndim].upb =
          chkcon_to_isz(RHS(rhstop), FALSE);
      sem.bounds[sem.arrdim.ndim].upast = mk_bnd_int(SST_ASTG(RHS(rhstop)));
    } else if (SST_IDG(RHS(rhstop)) == S_STAR) {
      constarraysize = 0;
      sem.bounds[sem.arrdim.ndim].uptype = S_STAR;
      sem.bounds[sem.arrdim.ndim].upb = 0;
      sem.bounds[sem.arrdim.ndim].upast = 0;
      SST_LSYMP(RHS(rhstop), 0);
      SST_DTYPEP(RHS(rhstop), DT_INT);
    } else {
      constarraysize = 0;
      sem.bounds[sem.arrdim.ndim].uptype = S_EXPR;
      sem.bounds[sem.arrdim.ndim].upb =
          chk_arr_extent(RHS(rhstop), "array upper bound");
      ast = SST_ASTG(RHS(rhstop));
      if (A_ALIASG(ast)) {
        ast = mk_bnd_int(A_ALIASG(ast));
        sem.bounds[sem.arrdim.ndim].uptype = S_CONST;
        sem.bounds[sem.arrdim.ndim].upb = get_isz_cval(A_SPTRG(ast));
      } else {
        /* When we have an AST with A_CONV, we want to skip the type 
           conversion AST in order to process the real intrinsic-call AST.*/
        if (A_TYPEG(ast) == A_CONV) {
          if (A_LOPG(ast) && A_TYPEG(A_LOPG(ast)) == A_INTR)
            ast = A_LOPG(ast);
        }
        if (*astb.atypes[A_TYPEG(ast)] == 'i' &&   
          DT_ISINT(A_DTYPEG(ast)) && ast_isparam(ast)) {
          INT conval;
          ACL *acl = construct_acl_from_ast(ast, A_DTYPEG(ast), 0);
          if (acl) {
            acl = eval_init_expr(acl);
            conval = cngcon(acl->conval, acl->dtype, A_DTYPEG(ast));
            ast = mk_cval1(conval, (int)A_DTYPEG(ast));
            SST_IDP(RHS(1), S_CONST);
            SST_LSYMP(RHS(1), 0);
            SST_ASTP(RHS(1), ast);
            SST_ACLP(RHS(1), 0);
            if (DT_ISWORD(A_DTYPEG(ast)))
              SST_SYMP(RHS(1), CONVAL2G(A_SPTRG(ast)));
            else
              SST_SYMP(RHS(1), A_SPTRG(ast));
          }
        }
      }
      sem.bounds[sem.arrdim.ndim].upast = ast;
    }

    /* check lower bound expression */

    if (rhstop == 1) { /* set default lower bound */
      sem.bounds[sem.arrdim.ndim].lowtype = S_CONST;
      sem.bounds[sem.arrdim.ndim].lowb = 1;
      sem.bounds[sem.arrdim.ndim].lwast = 0;
    } else {
      constant_lvalue(RHS(1));
      if (SST_IDG(RHS(1)) == S_CONST) {
        sem.bounds[sem.arrdim.ndim].lowtype = S_CONST;
        if (flg.standard) {
          int lowtyp;
          lowtyp = SST_DTYPEG(RHS(1));
          if (!DT_ISINT(lowtyp)) {
            error(170, 2, gbl.lineno, "array lower bound", "is not integer");
          }
        }
        sem.bounds[sem.arrdim.ndim].lowb = chkcon_to_isz(RHS(1), FALSE);
        if (constarraysize)
          arraysize -= (sem.bounds[sem.arrdim.ndim].lowb - 1);
        sem.bounds[sem.arrdim.ndim].lwast = mk_bnd_int(SST_ASTG(RHS(1)));
      } else {
        constarraysize = 0;
        sem.bounds[sem.arrdim.ndim].lowtype = S_EXPR;
        sem.bounds[sem.arrdim.ndim].lowb =
            chk_arr_extent(RHS(1), "array lower bound");
        ast = SST_ASTG(RHS(1));
        if (A_ALIASG(ast)) {
          ast = mk_bnd_int(A_ALIASG(ast));
          sem.bounds[sem.arrdim.ndim].lowtype = S_CONST;
          sem.bounds[sem.arrdim.ndim].lowb = get_isz_cval(A_SPTRG(ast));
        }
        sem.bounds[sem.arrdim.ndim].lwast = ast;
      }
    }
    if (constarraysize && arraysize < 0) {
      error(435, 2, gbl.lineno, "", CNULL);
      if (arraysize < 0) {
        /*
         * fix the upper bound to be lowb-1 so that the extent
         * evaluates to 0 so that the relatively new error #219,
         * 'Array too large' produced by dtypeutl.c:size_of_sym()
         * is avoided.
         */
        sem.bounds[sem.arrdim.ndim].upb = sem.bounds[sem.arrdim.ndim].lowb - 1;
        sem.bounds[sem.arrdim.ndim].upast =
            mk_isz_cval(sem.bounds[sem.arrdim.ndim].upb, astb.bnd.dtype);
      }
    }
    sem.arrdim.ndim++;
    break;
  /*
   *      <dim spec> ::= : |
   */
  case DIM_SPEC4:
    if (sem.arrdim.ndim >= MAXDIMS) {
      error(47, 3, gbl.lineno, CNULL, CNULL);
      break;
    }
    sem.bounds[sem.arrdim.ndim].lowtype = 0;
    sem.arrdim.ndim++;
    sem.arrdim.ndefer++;
    break;
  /*
   *      <dim spec> ::= <expression> : |
   */
  case DIM_SPEC5:
    if (sem.arrdim.ndim >= MAXDIMS) {
      error(47, 3, gbl.lineno, CNULL, CNULL);
      break;
    }
    sem.bounds[sem.arrdim.ndim].lowtype = S_EXPR;
    (void)chk_scalartyp(RHS(1), astb.bnd.dtype, FALSE);
    sem.bounds[sem.arrdim.ndim].lwast = SST_ASTG(RHS(1));
    sem.arrdim.ndim++;
    sem.arrdim.ndefer++;
    break;
  /*
   *    <dim spec> ::= ..
   */
  case DIM_SPEC6:
    sem.arrdim.ndim++;
    sem.arrdim.ndefer++;
    sem.arrdim.assumedrank = TRUE;
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<explicit shape> ::= <expression> : <expression> |
   */
  case EXPLICIT_SHAPE1:
    rhstop = 3;
    goto dim_spec;
  /*
   *	<explicit shape> ::= <expression>
   */
  case EXPLICIT_SHAPE2:
    rhstop = 1;
    goto dim_spec;

  /* ------------------------------------------------------------------ */
  /*
   *      <implicit type> ::= <implicit list> |
   */
  case IMPLICIT_TYPE1:
    break;
  /*
   *      <implicit type> ::= NONE
   */
  case IMPLICIT_TYPE2:
    if (sem.none_implicit & host_present)
      errwarn(55);
    if (seen_implicit || seen_parameter)
      error(70, 3, gbl.lineno, ": implicit none", CNULL);
    else
      symutl.none_implicit = sem.none_implicit |= host_present;
    newimplicitnone();
    if (sem.interface == 0) {
      ast_implicit(0, 0, 0);
      if (IN_MODULE_SPEC)
        mod_implicit(0, 0, 0);
    }
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <implicit list> ::= <implicit list> , <data type> <implp> <range list>
   * ) |
   */
  case IMPLICIT_LIST1:
  /*
   *      <implicit list> ::= <data type> <implp> <range list> )
   */
  case IMPLICIT_LIST2:
    if (sem.none_implicit & host_present)
      errwarn(56);
    seen_implicit = TRUE;
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <range list> ::= <range list> , <range> |
   */
  case RANGE_LIST1:
    rhstop = 3;
    goto range_list;
  /*
   *      <range list> ::= <range>
   */
  case RANGE_LIST2:
    rhstop = 1;
  range_list:
    begin = SST_RNG1G(RHS(rhstop));
    end = SST_RNG2G(RHS(rhstop));
    if (begin > end) {
      errwarn(36);
      end = begin;
    }
    if (flg.standard && (begin == '$' || begin == '_' || end == 0))
      errwarn(175);
    newimplicit(begin, end, sem.gdtype);
    if (sem.interface == 0) {
      ast_implicit(begin, end, sem.gdtype);
      if (IN_MODULE_SPEC)
        mod_implicit(begin, end, sem.gdtype);
    }

    /* adjust dtype of function and dummy arguments if necessary */

    for (sptr = gbl.currsub; sptr && sptr != NOSYM; sptr = SYMLKG(sptr)) {
      if (gbl.rutype == RU_FUNC) {
        if (FVALG(sptr) && !DCLDG(FVALG(sptr))) {
          setimplicit(FVALG(sptr));
          copy_type_to_entry(FVALG(sptr));
        }
      }

      count = PARAMCTG(sptr);
      i = DPDSCG(sptr);
      while (count--) {
        sptr2 = *(aux.dpdsc_base + i + count);
        if (!DCLDG(sptr2))
          setimplicit(sptr2);
      }
    }
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <range> ::= <letter> - <letter> |
   */
  case RANGE1:
    begin = SST_RNG1G(RHS(1));
    end = SST_RNG1G(RHS(3));
    if (begin == '$' || begin == '_' || end == '$' || end == '_') {
      /* cause an error and no action at the next production up */
      end = 0;
    }
    SST_RNG2P(LHS, end);
    break;
  /*
   *      <range> ::= <letter>
   */
  case RANGE2:
    SST_RNG2P(LHS, SST_RNG1G(RHS(1)));
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <common list> ::= <common list> <com dcl> |
   */
  case COMMON_LIST1:
    break;
  /*
   *      <common list> ::= <init com dcl>
   */
  case COMMON_LIST2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <init com dcl> ::= <dcl id list> |
   */
  case INIT_COM_DCL1:
  /*
   *      <init com dcl> ::= <dcl id list> , |
   */
  case INIT_COM_DCL2:
    rhsptr = 1;
    goto blank_common;
  /*
   *      <init com dcl> ::= <com dcl>
   */
  case INIT_COM_DCL3:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <com dcl> ::= '//' <dcl id list> <optional comma>   |
   */
  case COM_DCL1:
    rhsptr = 2;
    goto blank_common;
  /*
   *	<com dcl> ::= / / <dcl id list> <optional comma>   |
   */
  case COM_DCL2:
    rhsptr = 3;
    goto blank_common;
  blank_common:
    if (ignore_common_decl()) {
      break;
    }
    sptr = getsymbol("_BLNK_");
    sptr = refsym_inscope(sptr, OC_CMBLK);
    if (flg.xref)
      xrefput(sptr, 'd');
    if (STYPEG(sptr) == ST_UNKNOWN) {
      STYPEP(sptr, ST_CMBLK);
      SCOPEP(sptr, stb.curr_scope);
      SAVEP(sptr, 1);
      BLANKCP(sptr, 1);
    }
    goto com_dcl;
  /*
   *      <com dcl> ::= <common> <dcl id list> <optional comma>
   */
  case COM_DCL3:
    if (ignore_common_decl()) {
      break;
    }
    rhsptr = 2;
    sptr = SST_SYMG(RHS(1));
  com_dcl:
    if (CMEMFG(sptr) == 0) {
      /* first definition of this common block */
      {
        SYMLKP(sptr, gbl.cmblks); /* link into list of common blocks */
        gbl.cmblks = sptr;
      }
      i = 0;
      CMEMFP(sptr, NOSYM);
      CMEMLP(sptr, NOSYM);
    } else
      i = CMEMLG(sptr); /* last element of common block so far */

    /* loop thru dcl id list linking together symbol table entries */
    for (itemp = SST_BEGG(RHS(rhsptr)); itemp != ITEM_END;
         itemp = itemp->next) {
      sptr2 = itemp->t.sptr;
      stype = STYPEG(sptr2);
      if (IS_INTRINSIC(stype)) {
        /*
         * an intrinsic which can be changed due to its appearance in a
         * COMMON statement has already been processed in dcl_shared.
         * Getting here implies that the intrinsic is frozen, and
         * therefore, it will be ignored in the COMMON stmt.
         */
        error(40, 3, gbl.lineno, SYMNAME(sptr2), CNULL);
        break;
      } else if (stype != ST_UNKNOWN && stype != ST_IDENT && stype != ST_VAR &&
                 stype != ST_ARRAY && stype != ST_STRUCT &&
                 (!POINTERG(sptr2))) {
        error(40, 3, gbl.lineno, SYMNAME(sptr2), CNULL);
        reinit_sym(sptr2);
        STYPEP(sptr2, ST_VAR);
        DTYPEP(sptr2, DT_INT);
        SCP(sptr2, SC_LOCAL);
      }
      if (SCG(sptr2) == SC_CMBLK || SCG(sptr2) == SC_DUMMY)
        error(43, 3, gbl.lineno, "symbol", SYMNAME(sptr2));
      else if (stype == ST_ARRAY && (ASUMSZG(sptr2) || ADJARRG(sptr2)))
        error(50, 3, gbl.lineno, SYMNAME(sptr2), CNULL);
      else if (SAVEG(sptr2)) {
        error(39, 2, gbl.lineno, SYMNAME(sptr2), " and a COMMON statement");
        SAVEP(sptr2, 0);
      } else {
        SCP(sptr2, SC_CMBLK);
        CMBLKP(sptr2, sptr);
        if (i == 0)
          CMEMFP(sptr, sptr2);
        else
          SYMLKP(i, sptr2);
        SYMLKP(sptr2, NOSYM);
      }
      i = sptr2;
    }
    CMEMLP(sptr, i); /* point to last element of common block */
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <common> ::= / <ident> /
   */
  case COMMON1:
    if (ignore_common_decl()) {
      SST_SYMP(LHS, 0);
      break;
    }
    sptr = refsym_inscope((int)SST_SYMG(RHS(2)), OC_CMBLK);
    if (STYPEG(sptr) == ST_UNKNOWN) {
      STYPEP(sptr, ST_CMBLK);
      SCOPEP(sptr, stb.curr_scope);
    }
    SST_SYMP(LHS, sptr);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <save list> ::= <save list> , <save id> |
   */
  case SAVE_LIST1:
    if (flg.xref)
      xrefput((int)SST_SYMG(RHS(3)), 'd');
    break;
  /*
   *      <save list> ::= <save id>
   */
  case SAVE_LIST2:
    if (flg.xref)
      xrefput((int)SST_SYMG(RHS(1)), 'd');
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <save id> ::= <common>
   */
  case SAVE_ID1:
    sptr = SST_SYMG(RHS(1));
    if (sem.block_scope) {
      error(39, 3, gbl.lineno, SYMNAME(sptr), CNULL);
      break;
    }
    SAVEP(sptr, 1);
    break;
  /*
   *      <save id> ::= <ident>
   */
  case SAVE_ID2:
    sptr = block_local_sym(ref_ident_inscope((int)SST_SYMG(RHS(1))));
    stype = STYPEG(sptr);

    /* <ident> must be a variable or an array; it cannot be a dummy
     * argument or common block member.
     */
    if (stype == ST_ARRAY && (ASUMSZG(sptr) || ADJARRG(sptr))) {
      if (ASUMSZG(sptr))
        error(155, 3, gbl.lineno,
              "An assumed-size array cannot have the SAVE attribute -",
              SYMNAME(sptr));
      else if (SCG(sptr) == SC_DUMMY)
        error(155, 3, gbl.lineno,
              "An adjustable array cannot have the SAVE attribute -",
              SYMNAME(sptr));
      else
        error(155, 3, gbl.lineno,
              "An automatic array cannot have the SAVE attribute -",
              SYMNAME(sptr));
    } else if ((SCG(sptr) == SC_NONE || SCG(sptr) == SC_LOCAL ||
                SCG(sptr) == SC_BASED) &&
               (stype == ST_VAR || stype == ST_ARRAY || stype == ST_STRUCT ||
                stype == ST_IDENT)) {
      sem.savloc = TRUE;
      SAVEP(sptr, 1);
      /* SCP(sptr, SC_LOCAL);
       * SAVE is now an attribute and may appear allocatable; the
       * appearance of a variable in a SAVE statement is no longer
       * sufficient to define the variable's storage class.
       */
    } else
      error(39, 2, gbl.lineno, SYMNAME(sptr), CNULL);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <ideqc list> ::= <ideqc list> , <ident> <init beg> <expression> |
   */
  case IDEQC_LIST1:
    rhstop = 5;
    goto common_ideqc;
  /*
   *      <ideqc list> ::= <ident> <init beg> <expression>
   */
  case IDEQC_LIST2:
    rhstop = 3;
  common_ideqc:
    SST_IDP(RHS(rhstop - 2), S_IDENT);
    sptr = SST_SYMG(RHS(rhstop - 2));

    fixup_param_vars(RHS(rhstop - 2), RHS(rhstop));
    if (DTY(DTYPEG(sptr)) == TY_ARRAY || DTY(DTYPEG(sptr)) == TY_DERIVED) {
      sptr1 = CONVAL1G(sptr);

      construct_acl_for_sst(RHS(rhstop), DTYPEG(sptr1));
      if (!SST_ACLG(RHS(rhstop))) {
        goto end_ideqc;
      }
      CONVAL2P(sptr, put_getitem_p(save_acl(SST_ACLG(RHS(rhstop)))));

      ast = mk_id(sptr1);
      SST_ASTP(RHS(rhstop - 2), ast);
      SST_DTYPEP(RHS(rhstop - 2), DTYPEG(sptr1));
      SST_SHAPEP(RHS(rhstop - 2), A_SHAPEG(ast));
      ivl = dinit_varref(RHS(rhstop - 2));

      dinit(ivl, SST_ACLG(RHS(rhstop)));
    }

  end_ideqc:
    if (flg.xref)
      xrefput(sptr, 'i');
    sem.dinit_data = FALSE;
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<init beg> ::= =
   */
  case INIT_BEG1:
    sem.dinit_data = TRUE;
    sem.equal_initializer = true;
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <vxeqc list> ::= <vxeqc list> , <ident> = <expression> |
   */
  case VXEQC_LIST1:
    rhstop = 5;
    goto common_vxeqc;
  /*
   *      <vxeqc list> ::= <ident> = <expression>
   */
  case VXEQC_LIST2:
    rhstop = 3;
  common_vxeqc:
    sptr = declsym((int)SST_SYMG(RHS(rhstop - 2)), ST_PARAM, TRUE);
    dtype = SST_DTYPEG(RHS(rhstop));
    if (DCLDG(sptr) && dtype != DTYPEG(sptr))
      error(43, 3, gbl.lineno, "symbol", SYMNAME(sptr));

    if (SCG(sptr) != SC_NONE) {
      error(43, 3, gbl.lineno, "symbol", SYMNAME(sptr));
      break;
    }

    constant_lvalue(RHS(rhstop));
    if (SST_IDG(RHS(rhstop)) == S_CONST)
      conval = SST_CVALG(RHS(rhstop));
    else {
      errsev(87);
      dtype = DT_INT;
      conval = 1;
    }
    TYPDP(sptr, DCLDG(sptr));              /* appeared in a type statement */
    CONVAL2P(sptr, SST_ASTG(RHS(rhstop))); /* ast of <expression> */
    DTYPEP(sptr, dtype);
    DCLDP(sptr, TRUE);
    CONVAL1P(sptr, conval);
    VAXP(sptr, 1); /* vax-style parameter */
    if (sem.interface == 0)
      add_param(sptr);
    /* create an ast for the parameter; set the alias field of the ast
     * so that we don't have to set the alias field whenever the parameter
     * is referenced.
     */
    ast = mk_id(sptr);
    alias = mk_cval1(CONVAL1G(sptr), (int)DTYPEG(sptr));
    A_ALIASP(ast, alias);
    if (flg.xref)
      xrefput(sptr, 'i');
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<enums> ::= <enums> , <enum> |
   */
  case ENUMS1:
    break;
  /*
   *	<enums> ::= <enum>
   */
  case ENUMS2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<enum> ::= <ident> = <expression> |
   */
  case ENUM1:
    rhstop = 3;
    constant_lvalue(RHS(rhstop));
    conval = chkcon(RHS(rhstop), DT_INT4, TRUE);
    goto common_enum;
  /*
   *	<enum> ::= <ident>
   */
  case ENUM2:
    conval = next_enum;
  common_enum:
    dtype = DT_INT4;
    ast = mk_cval(conval, dtype);
    sptr = declsym(block_local_sym((int)SST_SYMG(RHS(1))), ST_PARAM, TRUE);
    if (DCLDG(sptr) || SCG(sptr) != SC_NONE) {
      error(43, 3, gbl.lineno, "symbol", SYMNAME(sptr));
      break;
    }
    TYPDP(sptr, DCLDG(sptr)); /* appeared in a type statement */
    CONVAL2P(sptr, ast);      /* ast of <expression> */
    DTYPEP(sptr, dtype);
    DCLDP(sptr, TRUE);
    CONVAL1P(sptr, conval);
    ast = mk_id(sptr);
    alias = mk_cval1(CONVAL1G(sptr), (int)DTYPEG(sptr));
    A_ALIASP(ast, alias);
    next_enum = conval + 1;
    if (flg.xref)
      xrefput(sptr, 'i');
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <equiv groups> ::= <equiv groups> , <equiv group> |
   */
  case EQUIV_GROUPS1:
    break;
  /*
   *      <equiv groups> ::= <equiv group>
   */
  case EQUIV_GROUPS2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <equiv group> ::= ( <equiv list> )
   */
  case EQUIV_GROUP1:
    /*
     * equivalence groups are linked together using the same field
     * used to link equivalence items within a single group.
     * A single equiv group is defined by the list beginning with an
     * EQVV item with a non-zero line number and ending with the item
     * preceding the next EQVV item with a non-zero line number (or
     * ending with the last item in the list).  The remaining
     * members in the group have line number fields which are zero.
     */
    if (sem.interface) /* HACK - throw away if in an interface block*/
      break;
    EQV(SST_NMLENDG(RHS(2))).next = sem.eqvlist;
    sem.eqvlist = SST_NMLBEGG(RHS(2));
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <equiv list> ::= <equiv list> , <equiv var> |
   */
  case EQUIV_LIST1:
    rhstop = 3;
    goto common_equiv;
  /*
   *      <equiv list> ::= <equiv var>
   */
  case EQUIV_LIST2:
    rhstop = 1;
  common_equiv:
    if (sem.interface) /* HACK - throw away if in an interface block*/
      break;
    evp = sem.eqv_avail;
    ++sem.eqv_avail;
    NEED(sem.eqv_avail, sem.eqv_base, EQVV, sem.eqv_size, sem.eqv_size + 20);
    EQV(evp).sptr = SST_SYMG(RHS(rhstop));
    EQV(evp).subscripts = SST_SUBSCRIPTG(RHS(rhstop));
    EQV(evp).substring = SST_SUBSTRINGG(RHS(rhstop));
    EQV(evp).byte_offset = SST_OFFSETG(RHS(rhstop));
    EQV(evp).next = 0;
    /* SEQP(evp->sptr, 1); -- SEQ flag set in semfin.c */
    if (flg.xref)
      xrefput(EQV(evp).sptr, 'e');
    if (rhstop == 1) {
      EQV(evp).lineno = gbl.lineno;
      EQV(evp).is_first = 1;
      SST_NMLBEGP(LHS, evp);
    } else {
      EQV(evp).lineno = 0;
      EQV(evp).is_first = 0;
      EQV(SST_NMLENDG(RHS(1))).next = evp;
    }
    SST_NMLENDP(LHS, evp);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <equiv var> ::= <ident> |
   */
  case EQUIV_VAR1:
    sptr = ref_ident_inscope((int)SST_SYMG(RHS(1)));
    SST_SYMP(LHS, sptr);
    SST_SUBSCRIPTP(LHS, 0); /* No subscripting */
    SST_OFFSETP(LHS, 0);    /* No substringing */
    SST_SUBSTRINGP(LHS, 0); /* No substringing - ast */
    break;
  /*
   *      <equiv var> ::= <equiv var> ( <ssa list> ) |
   */
  case EQUIV_VAR2:
    /* Validate that the subscripts are constant expressions, and build
     * an item list of them in long term (until end of program) storage.
     */
    sptr = SST_SYMG(RHS(1));
    itemp = SST_BEGG(RHS(3));
    if (itemp->next == ITEM_END && SST_IDG(itemp->t.stkp) == S_TRIPLE) {
      if (SST_IDG(SST_E3G(itemp->t.stkp)) == S_NULL) {
        /* This is a possible form of a substring.  Vector triplet
         * notation is illegal in any form.
         */
        if (SST_OFFSETG(RHS(1)))
          error(144, 3, gbl.lineno, "Ugly equivalence ", "1");
        if (SST_IDG(SST_E1G(itemp->t.stkp)) == S_NULL) {
          i = 1;
          SST_SUBSTRINGP(LHS, 0);
        } else {
          i = chkcon(SST_E1G(itemp->t.stkp), DT_INT4, TRUE);
          if (i <= 0) {
            error(82, 3, gbl.lineno, SYMNAME(sptr), CNULL);
            i = 0;
          }
          SST_SUBSTRINGP(LHS, SST_ASTG(SST_E1G(itemp->t.stkp)));
        }
        SST_OFFSETP(LHS, i);
        break;
      }
    }

    if (SST_SUBSCRIPTG(RHS(1)) != 0) {
      error(144, 3, gbl.lineno, "Ugly equivalence 3", CNULL);
      break;
    }
    ss = 0;
    numss = 0;
    for (itemp = SST_BEGG(RHS(3)); itemp != ITEM_END; itemp = itemp->next) {
      if (ss == 0) {
        numss = 1;
        ss = sem.eqv_ss_avail;
        sem.eqv_ss_avail += 2;
        NEED(sem.eqv_ss_avail, sem.eqv_ss_base, int, sem.eqv_ss_size,
             sem.eqv_ss_size + 50);
        SST_SUBSCRIPTP(LHS, ss); /* Save begin of subscript list */
        EQV_NUMSS(ss) = numss;
      } else {
        ++sem.eqv_ss_avail;
        NEED(sem.eqv_ss_avail, sem.eqv_ss_base, int, sem.eqv_ss_size,
             sem.eqv_ss_size + 50);
        ++numss;
        EQV_NUMSS(ss) = numss;
      }
      if (SST_IDG(itemp->t.stkp) == S_KEYWORD) {
        /* <ident> = <expr> is illegal just use <expr> part */
        errsev(79);
        SST_SUBSCRIPTP(LHS, 0);
      } else if (SST_IDG(itemp->t.stkp) == S_TRIPLE) {
        /* Legally this can only mean character substringing.  Vector
         * triplet notation is not allowable in equivalencing.
         */
        error(155, 3, gbl.lineno,
              "Subscript triplet not allowed in EQUIVALENCE -", SYMNAME(sptr));
        SST_SUBSCRIPTP(LHS, 0);
      } else {
        (void)chkcon_to_isz(itemp->t.stkp, TRUE);
        EQV_SS(ss, numss - 1) = SST_ASTG(itemp->t.stkp);
      }
    }
    break;
  /*
   *      <equiv var> ::= <equiv var> . <ident>
   */
  case EQUIV_VAR3:
    SST_IDP(LHS, S_IDENT);
    SST_SYMP(LHS, 0);
    SST_SUBSCRIPTP(LHS, 0); /* No subscripting */
    SST_OFFSETP(LHS, 0);    /* No substringing */
    SST_SUBSTRINGP(LHS, 0); /* No substringing - ast */
    error(155, 3, gbl.lineno, "Member cannot be equivalenced -",
          SYMNAME(SST_SYMG(RHS(3))));
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <namelist groups> ::= <namelist groups> <namelist group> |
   */
  case NAMELIST_GROUPS1:
    break;
  /*
   *      <namelist groups> ::= <namelist group>
   */
  case NAMELIST_GROUPS2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <namelist group> ::= / <ident> / <namelist list>
   */
  case NAMELIST_GROUP1:
    sptr = declref((int)SST_SYMG(RHS(2)), ST_NML, 'd');
    if (DCLDG(sptr))
      NML_NEXT(CMEMLG(sptr)) = SST_NMLBEGG(RHS(4));
    else {
      SYMLKP(sptr, sem.nml);
      sem.nml = sptr;
      CMEMFP(sptr, SST_NMLBEGG(RHS(4)));
      DCLDP(sptr, TRUE);
      /* create the array representing the namelist group */
      (void)get_nml_array(sptr);
    }
    CMEMLP(sptr, SST_NMLENDG(RHS(4)));
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <namelist list> ::= <namelist list> <namelist var> |
   */
  case NAMELIST_LIST1:
    rhstop = 2;
    goto nml_list;
  /*
   *      <namelist list> ::= <namelist var>
   */
  case NAMELIST_LIST2:
    rhstop = 1;
  nml_list:
    i = aux.nml_avl++;
    NEED(aux.nml_avl, aux.nml_base, NMLDSC, aux.nml_size, aux.nml_size + 100);
    NML_SPTR(i) = SST_SYMG(RHS(rhstop));
    NML_NEXT(i) = 0;
    NML_LINENO(i) = gbl.lineno;
    if (rhstop == 1) /* first item in the list */
      SST_NMLBEGP(LHS, i);
    else /* add item to the end of the list */
      NML_NEXT(SST_NMLENDG(RHS(1))) = i;
    SST_NMLENDP(LHS, i); /* item is now the end of the list */
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <namelist var> ::= <ident> <optional comma>
   */
  case NAMELIST_VAR1:
    sptr = ref_ident((int)SST_SYMG(RHS(1)));
    SST_SYMP(LHS, sptr);
    /* equivalence processing is done before the namelist processing;
     * this order is necessary to accomodate adding members to a
     * common block by equivalencing.  For SC_LOCALs the namelist
     * processing switches the storage class to SC_STATIC; therefore,
     * the equivalence processor needs to know that a variable appeared
     * as a namelist item.
     */
    NMLP(sptr, 1);
    break;

  /* ------------------------------------------------------------------ */
  /*
   * <struct begin1> ::= |
   */
  case STRUCT_BEGIN11:
    sem.stag_dtype = get_type(6, TY_STRUCT, NOSYM);
    DTY(sem.stag_dtype + 3) = 0; /* no tag */
    if (sem.stsk_depth == 0)
      error(135, 2, gbl.lineno, CNULL, CNULL);
    break;
  /*
   *      <struct begin1> ::= / <ident> /
   */
  case STRUCT_BEGIN12:
    sptr = declsym((int)SST_SYMG(RHS(2)), ST_STAG, TRUE);
    sem.stag_dtype = get_type(6, TY_STRUCT, NOSYM);
    DTYPEP(sptr, sem.stag_dtype);   /* give tag its dtype */
    DTY(sem.stag_dtype + 3) = sptr; /* give dtype its tag */
    DTY(sem.stag_dtype + 5) = 0;    /* ict pointer */
    NESTP(sptr, INSIDE_STRUCT);     /* nested structure */
    /* NOTE: we don't set DCLD here; see ENDSTRUCTURE */
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <struct begin2> ::= |
   */
  case STRUCT_BEGIN21:
    SST_RNG2P(LHS, NOSYM);
    break;
  /*
   *      <struct begin2> ::= <field namelist>
   */
  case STRUCT_BEGIN22:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <field namelist> ::= <field namelist> , <field name> |
   */
  case FIELD_NAMELIST1:
    SYMLKP(SST_SYMG(RHS(1)), SST_SYMG(RHS(3)));
    SST_SYMP(LHS, SST_SYMG(RHS(3)));
    break;
  /*
   *      <field namelist> ::= <field name>
   */
  case FIELD_NAMELIST2:
    /* Save ptr to 1st field name in field namelist */
    SST_RNG2P(LHS, SST_SYMG(RHS(1)));
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <field name> ::= <ident> |
   */
  case FIELD_NAME1:
    dtype = sem.stag_dtype;
    goto field_name;
  /*
   *      <field name> ::= <ident> <dim beg> <dimension list> )
   */
  case FIELD_NAME2:
    dtype = SST_DTYPEG(RHS(3));
    ad = AD_DPTR(dtype);
    if (AD_ASSUMSZ(ad) || AD_ADJARR(ad) || AD_DEFER(ad))
      error(50, 3, gbl.lineno, SYMNAME(SST_SYMG(RHS(1))), CNULL);
  field_name:
    stype = ST_MEMBER;
    sptr = SST_SYMG(RHS(1));
    if (STYPEG(sptr) != ST_UNKNOWN)
      SST_SYMP(LHS, (sptr = insert_sym(sptr)));
    SYMLKP(sptr, NOSYM);
    if (DTY(dtype) == TY_ARRAY)
      DTY(dtype + 1) = sem.stag_dtype;
    STYPEP(sptr, stype);
    DTYPEP(sptr, dtype);
    FNMLP(sptr, 1); /* declaration due to field name */
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <record list> ::= <record list> <record> |
   */
  case RECORD_LIST1:
    break;
  /*
   *      <record list> ::= <record>
   */
  case RECORD_LIST2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <record> ::= / <struct name> / <record namelist>
   */
  case RECORD1:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <struct name> ::= <ident>
   */
  case STRUCT_NAME1:
    /* Make sure sym ptr on stack is to a structure tag */
    SST_SYMP(LHS, (sptr = declref((int)SST_SYMG(RHS(1)), ST_STAG, 'r')));
    if (!DCLDG(sptr)) {
      error(139, 3, gbl.lineno, SYMNAME(sptr), CNULL);
      dtype = get_type(6, TY_STRUCT, NOSYM);
      DTY(dtype + 2) = 1;    /* size */
      DTY(dtype + 3) = sptr; /* tag */
      DTY(dtype + 5) = 0;    /* ict pointer */
      DTYPEP(sptr, dtype);
      DCLDP(sptr, TRUE);
    }
    sem.stag_dtype = DTYPEG(sptr);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <record namelist> ::= <record namelist> <record dcl> |
   */
  case RECORD_NAMELIST1:
    sptr = SST_SYMG(RHS(2));
    goto record_dcl;
  /*
   *      <record namelist> ::= <record dcl>
   */
  case RECORD_NAMELIST2:
    sptr = SST_SYMG(RHS(1));
  record_dcl:
    dtype = sem.stag_dtype;
    inited = FALSE;
    ict1 = (ACL *)get_getitem_p(DTY(dtype + 5));
    if (ict1) {
      /* Need to build an initializer constant tree */
      ict = GET_ACL(15);
      *ict = *ict1;
      ict->sptr = sptr;
      if (DTY(DTYPEG(sptr)) == TY_ARRAY)
        ict->repeatc = AD_NUMELM(AD_PTR(sptr));
      else
        ict->repeatc = astb.i1;
      if (INSIDE_STRUCT) {
        if (stsk->ict_end)
          stsk->ict_end->next = ict;
        else
          stsk->ict_beg = ict;
        stsk->ict_end = ict;
      } else if (SCG(sptr) != SC_DUMMY) {
        /*
         * NOTE: it's legal to use a STRUCTURE which contains
         * dinits to declare a dummy argument
         */
        dinit((VAR *)NULL, ict);
      }
    }
    goto common_typespecs;

  /* ------------------------------------------------------------------ */
  /*
   *      <record dcl> ::= <ident> <optional comma> |
   */
  case RECORD_DCL1:
    stype = ST_STRUCT;
    dtype = sem.stag_dtype;
    goto dcl_shared;
  /*
   *      <record dcl> ::= <ident> <dim beg> <dimension list> ) <optional comma>
   */
  case RECORD_DCL2:
    stype = ST_ARRAY;
    dtype = SST_DTYPEG(RHS(3));
    ad = AD_DPTR(dtype);
    goto dcl_shared;

  /* ------------------------------------------------------------------ */
  /*
   *      <vol list> ::= <vol list> , <vol id> |
   */
  case VOL_LIST1:
    break;
  /*
   *      <vol list> ::= <vol id>
   */
  case VOL_LIST2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <vol id> ::= <common> |
   */
  case VOL_ID1:
    sptr = SST_SYMG(RHS(1));
    VOLP(sptr, 1);
    break;
  /*
   *      <vol id> ::= <ident>
   */
  case VOL_ID2:
    sptr = ref_ident_inscope((int)SST_SYMG(RHS(1)));
    if (sem.block_scope && sptr < sem.scope_stack[sem.block_scope].symavl &&
        !VOLG(sptr))
      error(1219, ERR_Severe, gbl.lineno,
            "VOLATILE statement in a BLOCK construct", CNULL);
    VOLP(sptr, true);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <dinit list> ::= <dinit list> <optional comma> <dinit> |
   */
  case DINIT_LIST1:
    break;
  /*
   *      <dinit list> ::= <dinit>
   */
  case DINIT_LIST2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <dinit> ::= <dinit var list> / <dinit const list> /
   */
  case DINIT1:
    /* call dinit to write data initialization records */
    if (!sem.dinit_error) {
      SST_CLBEGP(RHS(3),
                 rewrite_acl(SST_CLBEGG(RHS(3)), SST_CLBEGG(RHS(3))->dtype, 0));
      dinit(SST_VLBEGG(RHS(1)), SST_CLBEGG(RHS(3)));
    }
    sem.dinit_error = FALSE;
    sem.dinit_data = FALSE;
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <dinit var list> ::= <dinit var list> , <dinit var> |
   */
  case DINIT_VAR_LIST1:
    /* append entry to end of dinit var list */
    ((SST_VLENDG(RHS(1))))->next = SST_VLBEGG(RHS(3));
    SST_VLENDP(LHS, SST_VLENDG(RHS(3)));
    break;
  /*
   *      <dinit var list> ::= <dinit var>
   */
  case DINIT_VAR_LIST2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <dinit var> ::= <dvar ref> |
   */
  case DINIT_VAR1:
    (void)mklvalue(RHS(1), 2); /* ILM pointer of var ref */
    dtype = SST_DTYPEG(RHS(1));
    {
      /* build an element for the dinit var list */
      ivl = dinit_varref(RHS(1));
      if (ivl == NULL) {
        /* an array section was initialized -- dinit_varref()
         * transforms this <data var> into an implied do or a nested
         * implied do.
         */
        break;
      }
    }
    sem.dinit_data = TRUE;
    if (ivl->u.varref.id == S_LVALUE && SCG(SST_LSYMG(RHS(1))) == SC_BASED) {
      error(116, 3, gbl.lineno, SYMNAME(SST_LSYMG(RHS(1))), "(DATA)");
      sem.dinit_error = TRUE;
    }
    SST_VLBEGP(LHS, SST_VLENDP(LHS, ivl));
    break;
  /*
   *      <dinit var> ::= ( <dinit var list> , <ident> = <expression> ,
   * <expression> <e3> )
   */
  case DINIT_VAR2:
    (void)chk_scalartyp(RHS((9)), DT_INT, TRUE);
    /* build a doend element for the dinit var list */
    ivl = (VAR *)getitem(15, sizeof(VAR));
    SST_VLENDP(LHS, ivl);
    SST_VLENDG(RHS(2))->next = ivl;
    ivl->id = Doend;
    ivl->next = NULL;

    /* Create the dostart element, link it to the doend element, and
     * link all in the order dostart, <dinit var list>, then doend
     */
    ivl->u.doend.dostart = (VAR *)getitem(15, sizeof(VAR));
    ivl = ivl->u.doend.dostart;
    ivl->id = Dostart;
    sptr = refsym((int)SST_SYMG(RHS(4)), OC_OTHER);
    if (!DCLDG(sptr))
      IGNOREP(sptr, TRUE);
    SST_SYMP(RHS(4), sptr);
    (void)chktyp(RHS(4), DT_INT, TRUE);
    ivl->u.dostart.indvar = SST_ASTG(RHS(4));
    (void)chk_scalartyp(RHS(6), DT_INT, TRUE);
    ivl->u.dostart.lowbd = SST_ASTG(RHS(6));
    (void)chk_scalartyp(RHS(8), DT_INT, TRUE);
    ivl->u.dostart.upbd = SST_ASTG(RHS(8));
    ivl->u.dostart.step = SST_ASTG(RHS(9));
    ivl->next = SST_VLBEGG(RHS(2));
    SST_VLBEGP(LHS, ivl);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <e3> ::=   |
   */
  case E31:
    SST_IDP(LHS, S_CONST);
    SST_CVALP(LHS, 1);
    SST_DTYPEP(LHS, DT_INT);
    SST_ASTP(LHS, 0);
    break;
  /*
   *      <e3> ::= , <expression>
   */
  case E32:
    *LHS = *RHS(2);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <dinit const list> ::= <dinit const list> , <data item> |
   */
  case DINIT_CONST_LIST1:
    if (SST_CLBEGG(RHS(3)) != NULL) {
      SST_CLENDG(RHS(1))->next = SST_CLBEGG(RHS(3));
      SST_CLENDP(LHS, SST_CLENDG(RHS(3)));
    }
    break;
  /*
   *      <dinit const list> ::= <data item>
   */
  case DINIT_CONST_LIST2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <data item> ::= <data constant> |
   */
  case DATA_ITEM1:
    conval = 1; /* default repeat count */
    ast = 0;
    goto common_data_item;
  /*
   *      <data item> ::= <data rpt> * <data constant>
   */
  case DATA_ITEM2:
    ast = SST_ASTG(RHS(1));
    conval = SST_CVALG(RHS(1));
    *RHS(1) = *RHS(3);
  common_data_item:
    /*
     * Check for too many constant initializers here!  Why here and not in
     * dinit?  Because for structures and type decl stmts we want the error
     * flagged on the structure stmt not the record stmt which may occur
     * many times and much later.
     */
    if (!sem.dinit_data) { /* Don't do this if in DATA stmt */
      if (sem.dinit_count < conval) {
        if (sem.dinit_count >= 0)
          errsev(67);
        if (sem.dinit_count <= 0) { /* Error already handled */
          SST_CLBEGP(LHS, SST_CLENDP(LHS, NULL));
          break;
        }
        conval = sem.dinit_count; /* Put out as many as possible */
        sem.dinit_count = -1;     /* Prevent further error msgs */
      }
      sem.dinit_count -= conval;
    }
    if (SST_IDG(RHS(1)) == S_SCONST) {
      ict = dinit_struct_vals(SST_ACLG(RHS(1)), SST_DTYPEG(RHS(1)), NOSYM);
      if (!ict) {
        break;
      }
      ict->repeatc = ast;
      SST_CLBEGP(LHS, SST_CLENDP(LHS, ict));
      break;
    }

    /* allocate and init an Initializer Constant Tree entry */
    ict = GET_ACL(15);
    ict->id = AC_AST;
    ict->next = NULL;
    ict->subc = NULL;
    ict->u1.ast = SST_ASTG(RHS(1));
    ict->repeatc = ast;
    ict->sptr = 0;
    ict->dtype = SST_DTYPEG(RHS(1));
    SST_CLBEGP(LHS, SST_CLENDP(LHS, ict));
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<data rpt> ::= <integer> |
   */
  case DATA_RPT1:
    conval = SST_CVALG(RHS(1));
    ast = mk_cval(SST_CVALG(RHS(1)), DT_INT4);
    goto common_rpt;
  /*
   *	<data rpt> ::= <int kind const> |
   */
  case DATA_RPT2:
    /* token value of <int kind const> is an ST_CONST entry */
    conval = get_int_cval(SST_CVALG(RHS(1)));
    ast = mk_cnst(SST_CVALG(RHS(1)));
    goto common_rpt;
  /*
   *	<data rpt> ::= <ident constant>
   */
  case DATA_RPT3:
    dtype = SST_DTYPEG(RHS(1));
    if (dtype == DT_INT8 || dtype == DT_LOG8)
      conval = get_int_cval(SST_CVALG(RHS(1)));
    else
      conval = SST_CVALG(RHS(1));
    ast = SST_ASTG(RHS(1));
  common_rpt:
    if (conval < 0) {
      errsev(65);
      conval = 0;
    }
    SST_CVALP(LHS, conval);
    SST_ASTP(LHS, ast);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <data constant> ::= <constant> |
   */
  case DATA_CONSTANT1:
    SST_IDP(LHS, S_CONST);
    break;
  /*
   *      <data constant> ::= <addop> <constant>  |
   */
  case DATA_CONSTANT2:
    SST_IDP(RHS(2), S_CONST);
    goto addop_data_constant;
  /*
   *      <data constant> ::= <ident constant> |
   */
  case DATA_CONSTANT3:
    break;
  /*
   *      <data constant> ::= <addop> <ident constant> |
   */
  case DATA_CONSTANT4:
  addop_data_constant:
    opc = SST_OPTYPEG(RHS(1));
    *LHS = *RHS(2);
    if (opc == OP_SUB) {
      SST_CVALP(LHS, negate_const(SST_CVALG(RHS(2)), (int)SST_DTYPEG(RHS(2))));
      ast = mk_unop(OP_SUB, SST_ASTG(RHS(2)), SST_DTYPEG(LHS));
      SST_ASTP(LHS, ast);
      mk_alias(ast, mk_cval1(SST_CVALG(LHS), (int)SST_DTYPEG(LHS)));
    }
    break;
  /*
   *	<data constant> ::= <ident ssa> ( <ssa list> ) |
   */
  case DATA_CONSTANT5:
    sptr = SST_SYMG(RHS(1));
    dtype = SST_DTYPEG(RHS(1));
    if (sem.in_struct_constr) {
      /* create head AC_SCONST for element list */
      aclp = GET_ACL(15);
      aclp->id = AC_SCONST;
      aclp->next = NULL;
      aclp->subc = (ACL *)SST_BEGG(RHS(3));
      aclp->dtype = dtype = DTYPEG(sem.in_struct_constr);
      SST_IDP(LHS, S_SCONST);
      SST_DTYPEP(LHS, dtype);
      SST_ACLP(LHS, aclp);
      if (is_empty_typedef(dtype)) {
        error(155, 3, gbl.lineno, "Structure constructor specified"
                                  " for empty derived type",
              SYMNAME(sptr));
      } else
        chk_struct_constructor(aclp);
      SST_SYMP(LHS, sem.in_struct_constr);  /* use tag as SYM */
      sem.in_struct_constr = SST_TMPG(LHS); /*restore old value */
      break;
    }
    sem.in_struct_constr = SST_TMPG(LHS); /* restore old value */

    if (STYPEG(sptr) == ST_PARAM && DTY(dtype) == TY_NCHAR) {
      SST *sp;

      itemp = SST_BEGG(RHS(3));
      sp = itemp->t.stkp;
      if (SST_IDG(sp) != S_TRIPLE || SST_IDG(SST_E3G(sp)) != S_NULL ||
          itemp->next != ITEM_END) {
        error(82, 3, gbl.lineno, SYMNAME(sptr), CNULL);
        SST_DTYPEP(LHS, DT_NCHAR);
        val[0] = getstring(" ", 1);
        val[1] = 0;
        SST_CVALP(LHS, getcon(val, DT_NCHAR));
        SST_ASTP(LHS, mk_cnst(SST_CVALG(LHS)));
        SST_SHAPEP(LHS, 0);
        break;
      }
      SST_IDP(LHS, S_CONST);
      SST_CVALP(LHS, CONVAL1G(sptr)); /* get constant sptr */
      SST_DTYPEP(LHS, dtype);
      SST_ASTP(LHS, CONVAL2G(sptr)); /* constant's ast */
      SST_SHAPEP(LHS, 0);
      SST_ERRSYMP(LHS, sptr); /* save for error tracing */
      ch_substring(LHS, SST_E1G(sp), SST_E2G(sp));
      goto check_data_substring;
    }
    if (STYPEG(sptr) == ST_PARAM && DTY(dtype) == TY_CHAR) {
      SST *sp;

      itemp = SST_BEGG(RHS(3));
      sp = itemp->t.stkp;
      if (SST_IDG(sp) != S_TRIPLE || SST_IDG(SST_E3G(sp)) != S_NULL ||
          itemp->next != ITEM_END) {
        error(82, 3, gbl.lineno, SYMNAME(sptr), CNULL);
        SST_DTYPEP(LHS, DT_CHAR);
        SST_CVALP(LHS, getstring(" ", 1));
        SST_ASTP(LHS, mk_cnst(SST_CVALG(LHS)));
        SST_SHAPEP(LHS, 0);
        break;
      }
      SST_IDP(LHS, S_CONST);
      SST_CVALP(LHS, CONVAL1G(sptr)); /* get constant sptr */
      SST_DTYPEP(LHS, dtype);
      SST_ASTP(LHS, CONVAL2G(sptr)); /* constant's ast */
      SST_SHAPEP(LHS, 0);
      SST_ERRSYMP(LHS, sptr); /* save for error tracing */
      ch_substring(LHS, SST_E1G(sp), SST_E2G(sp));
      goto check_data_substring;
    } else {
      errsev(87);
      sem.dinit_error = TRUE;
    }
    break;
  /*
   *	<data constant> ::= <ident ssa> ( ) |
   */
  case DATA_CONSTANT6:
    if (STYPEG(SST_SYMG(RHS(1))) != ST_PD ||
        PDNUMG(SST_SYMG(RHS(1))) != PD_null) {
      dtype = SST_DTYPEG(RHS(1));
      if (sem.in_struct_constr && is_empty_typedef(dtype)) {
        /* Ignore empty struct constructor for an
         * empty typedef
         */
        sem.dinit_error = TRUE;
        break;
      }
      errsev(87);
      sem.dinit_error = TRUE;
      break;
    }
    SST_IDP(RHS(1), S_IDENT);
    (void)mkvarref(RHS(1), ITEM_END);
    break;

  /*
   *	<data constant> ::= <substring>
   */
  case DATA_CONSTANT7:
    dtype = SST_DTYPEG(RHS(1));
  check_data_substring:
    constant_lvalue(RHS(1));
    if (SST_IDG(RHS(1)) != S_CONST) {
      errsev(87);
      sem.dinit_error = TRUE;
      if (DTY(dtype) == TY_NCHAR) {
        SST_DTYPEP(LHS, DT_NCHAR);
        val[0] = getstring(" ", 1);
        val[1] = 0;
        SST_CVALP(LHS, getcon(val, DT_NCHAR));
        SST_ASTP(LHS, mk_cnst(SST_CVALG(LHS)));
        SST_SHAPEP(LHS, 0);
        break;
      }
      SST_DTYPEP(LHS, DT_CHAR);
      SST_CVALP(LHS, getstring(" ", 1));
      SST_ASTP(LHS, mk_cnst(SST_CVALG(LHS)));
      SST_SHAPEP(LHS, 0);
    }
    break;
  /*
   *      <ident ssa> ::= <ident>
   */
  case IDENT_SSA1:
    sptr = refsym((int)SST_SYMG(RHS(1)), OC_OTHER);
    dtype = DTYPEG(sptr);
    SST_SYMP(LHS, sptr);
    SST_DTYPEP(LHS, dtype);
    SST_TMPP(LHS, sem.in_struct_constr); /* save old value */
    /* set a flag for ssa list processing */
    if (STYPEG(sptr) == ST_TYPEDEF && DTY(dtype) == TY_DERIVED) {
      sem.in_struct_constr = sptr;
    } else
      sem.in_struct_constr = 0;
    break;

  /*
   *      <ident constant> ::= <ident>
   */
  case IDENT_CONSTANT1:
    sptr = refsym((int)SST_SYMG(RHS(1)), OC_OTHER);
    SST_IDP(LHS, S_CONST);
    if (STYPEG(sptr) == ST_PARAM) {
      /* resolve constant */
      SST_DTYPEP(LHS, DTYPEG(sptr));
      SST_CVALP(LHS, CONVAL1G(sptr));
      ast = mk_id(sptr);
      if (!XBIT(49, 0x10)) /* preserve PARAMETER? */
        ast = A_ALIASG(ast);
    } else if (flg.standard)
      goto ident_constant_error;
    else {
      np = SYMNAME(sptr);
      if (*np == 't') {
        if (DTY(stb.user.dt_log) == TY_LOG8) {
          if (gbl.ftn_true == -1)
            val[0] = val[1] = -1;
          else {
            val[0] = 0;
            val[1] = 1;
          }
          SST_CVALP(LHS, getcon(val, DT_LOG8));
          ast = mk_cval1(SST_CVALG(LHS), DT_LOG);
        } else {
          SST_CVALP(LHS, SCFTN_TRUE);
          ast = mk_cval(SCFTN_TRUE, DT_LOG);
        }
        SST_DTYPEP(LHS, DT_LOG);
      } else if (*np == 'f') {
        if (DTY(stb.user.dt_log) == TY_LOG8) {
          val[0] = val[1] = 0;
          SST_CVALP(LHS, getcon(val, DT_LOG8));
          ast = mk_cval1(SST_CVALG(LHS), DT_LOG);
        } else {
          SST_CVALP(LHS, SCFTN_FALSE);
          ast = mk_cval(SCFTN_FALSE, DT_LOG);
        }
        SST_DTYPEP(LHS, DT_LOG);
      } else
        goto ident_constant_error;
    }
    SST_ASTP(LHS, ast);
    break;
  ident_constant_error:
    errsev(87);
    SST_CVALP(LHS, stb.i0);
    SST_DTYPEP(LHS, DT_INT4);
    ast = mk_id(sptr);
    sem.dinit_error = TRUE;
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <ptr list> ::= <ptr list> , <ptr assoc> |
   */
  case PTR_LIST1:
    break;
  /*
   *      <ptr list> ::= <ptr assoc>
   */
  case PTR_LIST2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <ptr assoc> ::= ( <ident> , <dcl id> ) |
   */
  case PTR_ASSOC1:
    sptr = declsym((int)SST_SYMG(RHS(2)), ST_VAR, FALSE);
    if (flg.standard)
      error(171, 2, gbl.lineno, "- Cray POINTER statement", CNULL);
    if (XBIT(124, 0x10)) {
      /* -i8 */
      if (DCLDG(sptr) && DTYPEG(sptr) != DT_INT8)
        error(37, 3, gbl.lineno, SYMNAME(sptr), CNULL);
      DTYPEP(sptr, DT_INT8);
    } else {
      if (DCLDG(sptr) && DTYPEG(sptr) != DT_PTR)
        error(37, 3, gbl.lineno, SYMNAME(sptr), CNULL);
      DTYPEP(sptr, DT_PTR);
    }
    DCLDP(sptr, TRUE);
    PTRVP(sptr, 1);
    sptr1 = SST_SYMG(RHS(4));
    if (VOLG(sptr1) || SCG(sptr1) != SC_NONE) {
      error(43, 3, gbl.lineno, "symbol", SYMNAME(sptr1));
      break;
    }
    SCP(sptr1, SC_BASED);
    MIDNUMP(sptr1, sptr);
    if (SAVEG(sptr1))
      error(39, 2, gbl.lineno, SYMNAME(sptr1), CNULL);
    if (STYPEG(sptr1) == ST_ARRAY) {
      if (ADJARRG(sptr1) || RUNTIMEG(sptr1)) {
        if (entry_seen)
          AFTENTP(sptr1, 1);
      }
    }
    while (TRUE) {
      if (SCG(sptr) == SC_BASED) {
        if (sptr == sptr1) {
          error(155, 3, gbl.lineno, "Recursive POINTER declaration of",
                SYMNAME(sptr1));
          MIDNUMP(sptr1, 0);
          SCP(sptr1, SC_NONE);
          break;
        }
        sptr = MIDNUMG(sptr);
      } else
        break;
    }
    break;
  /*
   *	<ptr assoc> ::= <alloc id>
   */
  case PTR_ASSOC2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<alloc id list> ::= <alloc id list> , <alloc id> |
   */
  case ALLOC_ID_LIST1:
    break;
  /*
   *	<alloc id list> ::= <alloc id>
   */
  case ALLOC_ID_LIST2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<alloc id> ::= <ident> |
   */
  case ALLOC_ID1:
    sptr = SST_SYMG(RHS(1));
    sptr = create_var(sptr);
    SST_SYMP(LHS, sptr);
    if (STYPEG(sptr) == ST_UNKNOWN)
      STYPEP(sptr, ST_IDENT);
    stype1 = STYPEG(sptr);
    if (IS_INTRINSIC(stype1)) {
      /* Changing intrinsic symbol to ARRAY */
      if ((sptr = newsym(sptr)) == 0)
        /* Symbol frozen as an intrinsic, ignore type decl */
        break;
      SST_SYMP(LHS, sptr);
      /* Cause STYPE and DTYPE to change AFTER fixing dtype */
      stype1 = ST_UNKNOWN;
    } else if (stype1 == ST_ENTRY) {
      if (DTY(DTYPEG(sptr)) == TY_ARRAY) {
        error(43, 3, gbl.lineno, "symbol", SYMNAME(sptr));
        break;
      }
    } else if (stype1 != ST_UNKNOWN && stype1 != ST_IDENT && stype1 != ST_VAR &&
               stype1 != ST_ARRAY) {
      /* Add special handling for procedure pointers
       *
       * The only two ways we can get here is either through pointer or through
       * allocatable declaration. Pointer attribute can be applied to
       * procedures, but not allocatable attribute.
       */
      if ((scn.stmtyp != TK_POINTER) || (stype1 != ST_PROC)) {
        error(43, 3, gbl.lineno, "symbol", SYMNAME(sptr));
        break;
      }
    }

    if (scn.stmtyp == TK_POINTER) {
      POINTERP(sptr, TRUE);
      if (STYPEG(sptr) == ST_PROC) {
        LOGICAL declared;
        sptr = SST_SYMG(RHS(1));
        /* Save "declared" flag to preserve implicit types */
        declared = DCLDG(sptr);
        /* Generate proper procedure symbol */
        sptr = insert_sym(sptr);
        sptr = setup_procedure_sym(sptr, proc_interf_sptr, ET_B(ET_POINTER),
                                   entity_attr.access);
        SST_SYMP(RHS(1), sptr);
        /* Restore "declared" flag */
        DCLDP(sptr, declared);
      }
      if (sem.contiguous)
        CONTIGATTRP(sptr, 1);
      if (DTYG(DTYPEG(sptr)) == TY_DERIVED && XBIT(58, 0x40000)) {
        F90POINTERP(sptr, TRUE);
      }
      if (DTY(DTYPEG(sptr)) == TY_ARRAY) {
        dtype = DTYPEG(sptr);
        ad = AD_DPTR(dtype);
        if (SCG(sptr) != SC_DUMMY) {
          if (!AD_DEFER(ad) || AD_ASSUMSHP(ad))
            error(196, 3, gbl.lineno, SYMNAME(sptr), CNULL);
          ALLOCP(sptr, 1);
        } else {
          if (!AD_DEFER(ad))
            error(196, 3, gbl.lineno, SYMNAME(sptr), CNULL);
          /* may have assumed the array was assumed-shape;
           * now we know better, it's an array pointer */
          ASSUMSHPP(sptr, 0);
          SDSCS1P(sptr, 0);
          AD_ASSUMSHP(ad) = 0;
        }
        if (!F90POINTERG(sptr)) {
          get_static_descriptor(sptr);
          get_all_descriptors(sptr);
        }
      }
    } else if ((stype1 != ST_ARRAY && stype1 != ST_IDENT
                /* Allow ST_IDENT here.  It happens when an
                 * ALLOCATABLE statement precedes the DIMENSION statement.
                 * If the allocatable is still an ST_IDENT in semfin.c,
                 * we'll call it an error at that time.
                 */
                ) ||
               (!ALLOCG(sptr) && stype1 != ST_IDENT) || SCG(sptr) != SC_NONE)
      error(84, 3, gbl.lineno, SYMNAME(sptr),
            "- must be a deferred shape array");
    else
      ALLOCATTRP(sptr, 1);

    if (RESULTG(sptr)) {
      /* set the type for the entry point as well */
      copy_type_to_entry(sptr);
    }
    break;
  /*
   *	<alloc id> ::= <ident> <dim beg> <dimension list> )
   */
  case ALLOC_ID2:
    sptr = SST_SYMG(RHS(1));
    sptr = create_var(sptr);
    SST_SYMP(LHS, sptr);
    if (STYPEG(sptr) == ST_UNKNOWN)
      STYPEP(sptr, ST_IDENT);
    stype1 = STYPEG(sptr);
    if (IS_INTRINSIC(stype1)) {
      /* Changing intrinsic symbol to ARRAY */
      if ((sptr = newsym(sptr)) == 0)
        /* Symbol frozen as an intrinsic, ignore type decl */
        break;
      SST_SYMP(LHS, sptr);
      /* Cause STYPE and DTYPE to change AFTER fixing dtype */
      stype1 = ST_UNKNOWN;
    } else if (stype1 == ST_ENTRY) {
      if (DTY(DTYPEG(sptr)) == TY_ARRAY) {
        error(43, 3, gbl.lineno, "symbol", SYMNAME(sptr));
        break;
      }
    } else if (stype1 != ST_UNKNOWN && stype1 != ST_IDENT && stype1 != ST_VAR) {
      error(43, 3, gbl.lineno, "symbol", SYMNAME(sptr));
      break;
    }

    STYPEP(sptr, ST_ARRAY);
    dtype = SST_DTYPEG(RHS(3));
    ad = AD_DPTR(dtype);
    DTY(dtype + 1) = DTYPEG(sptr);
    DTYPEP(sptr, dtype);
    if (DTY(dtype) == TY_ARRAY) {
      int d;
      d = DTY(dtype + 1);
      if (DTY(d) == TY_DERIVED && DTY(d + 3) && DISTMEMG(DTY(d + 3))) {
        error(451, 3, gbl.lineno, SYMNAME(sptr), CNULL);
      }
    }
    if (scn.stmtyp == TK_POINTER) {
      if (!AD_DEFER(ad) || AD_ASSUMSHP(ad))
        error(196, 3, gbl.lineno, SYMNAME(sptr), CNULL);
      if (SCG(sptr) != SC_DUMMY)
        ALLOCP(sptr, 1);
      POINTERP(sptr, TRUE);
      if (DTYG(DTYPEG(sptr)) == TY_DERIVED && XBIT(58, 0x40000)) {
        F90POINTERP(sptr, TRUE);
      }
      if (SDSCG(sptr) == 0 && !F90POINTERG(sptr)) {
        get_static_descriptor(sptr);
        get_all_descriptors(sptr);
      }
    } else if (AD_DEFER(ad) == 0)
      error(84, 3, gbl.lineno, SYMNAME(sptr),
            "- must be a deferred shape array");
    else {
      ALLOCP(sptr, 1);
      ALLOCATTRP(sptr, 1);
      if (DTYG(DTYPEG(sptr)) == TY_DERIVED && XBIT(58, 0x40000)) {
        F90POINTERP(sptr, TRUE);
      }
    }
    if (RESULTG(sptr)) {
      /* set the type for the entry point as well */
      copy_type_to_entry(sptr);
    }
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<opt attr list> ::=  |
   */
  case OPT_ATTR_LIST1:
  /*
   *	<opt attr list> ::= , <attr list>
   */
  case OPT_ATTR_LIST2:
    in_entity_typdcl = TRUE;
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<attr list> ::= <attr list> , <attr> |
   */
  case ATTR_LIST1:
  /* fall thru */
  /*
   *	<attr list> ::= <attr>
   */
  case ATTR_LIST2:
    if (INSIDE_STRUCT && (STSK_ENT(0).type == 'd')) {
      if (!(et_type == ET_DIMENSION || et_type == ET_POINTER
            || et_type == ET_ACCESS || et_type == ET_ALLOCATABLE ||
            et_type == ET_CONTIGUOUS || et_type == ET_KIND ||
            et_type == ET_LEN))
        error(134, 3, gbl.lineno, et[et_type].name,
              "for derived type component");
    }
    if (entity_attr.exist & ET_B(et_type))
      error(134, 3, gbl.lineno, "- duplicate", et[et_type].name);
    else if (entity_attr.exist & et[et_type].no)
      error(134, 3, gbl.lineno, "- conflict with", et[et_type].name);
    else {
      entity_attr.exist |= ET_B(et_type);
    }
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<attr> ::= PARAMETER     |
   */
  case ATTR1:
    et_type = ET_PARAMETER;
    break;
  /*
   *	<attr> ::= <access spec> |
   */
  case ATTR2:
    et_type = ET_ACCESS;
    break;
  /*
   *	<attr> ::= ALLOCATABLE   |
   */
  case ATTR3:
    et_type = ET_ALLOCATABLE;
    break;
  /*
   *	<attr> ::= <dimattr> <dim beg> <dimension list> ) |
   */
  case ATTR4:
    et_type = ET_DIMENSION;
    entity_attr.dimension = SST_DTYPEG(RHS(3));
    /* save bounds information just in case the dimension attribute
     * is used more than once
     */
    BCOPY(entity_attr.bounds, sem.bounds, char, sizeof(sem.bounds));
    BCOPY(entity_attr.arrdim, &sem.arrdim, char, sizeof(sem.arrdim));
    break;
  /*
   *	<attr> ::= EXTERNAL      |
   */
  case ATTR5:
    et_type = ET_EXTERNAL;
    break;
  /*
   *	<attr> ::= <intent> |
   */
  case ATTR6:
    et_type = ET_INTENT;
    break;
  /*
   *	<attr> ::= INTRINSIC     |
   */
  case ATTR7:
    et_type = ET_INTRINSIC;
    break;
  /*
   *	<attr> ::= OPTIONAL      |
   */
  case ATTR8:
    et_type = ET_OPTIONAL;
    break;
  /*
   *	<attr> ::= POINTER       |
   */
  case ATTR9:
    et_type = ET_POINTER;
    break;
  /*
   *	<attr> ::= SAVE          |
   */
  case ATTR10:
    et_type = ET_SAVE;
    break;
  /*
   *	<attr> ::= TARGET        |
   */
  case ATTR11:
    et_type = ET_TARGET;
    break;
  /*
   *	<attr> ::= AUTOMATIC     |
   */
  case ATTR12:
    et_type = ET_AUTOMATIC;
    break;
  /*
   *	<attr> ::= STATIC        |
   */
  case ATTR13:
    et_type = ET_STATIC;
    break;
  /*
   *      <attr> ::= BIND <bind attr>        |
   */
  case ATTR14:
    et_type = ET_BIND;
    break;
  /*
   *      <attr> ::= VALUE        |
   */
  case ATTR15:
    et_type = ET_VALUE;
    break;
  /*
   *      <attr> ::= VOLATILE     |
   */
  case ATTR16:
    et_type = ET_VOLATILE;
    break;
  /*
   *	<attr> ::= DEVICE        |
   */
  case ATTR17:
    if (cuda_enabled("device"))
      et_type = ET_DEVICE;
    else
      et_type = 0;
    break;
  /*
   *	<attr> ::= PINNED        |
   */
  case ATTR18:
    if (cuda_enabled("pinned"))
      et_type = ET_PINNED;
    else
      et_type = 0;
    break;
  /*
   *	<attr> ::= SHARED        |
   */
  case ATTR19:
    et_type = 0;
#ifdef CUDAG
    if (cuda_enabled("shared")) {
      if ((gbl.currsub && CUDAG(gbl.currsub) &&
           !(CUDAG(gbl.currsub) & CUDA_HOST)) ||
          (gbl.currmod && !gbl.currsub)) {
        /* device routine, or module declaration part */
        et_type = ET_SHARED;
      } else {
        error(134, 3, gbl.lineno, et[ET_SHARED].name,
              "not allowed in host subprograms");
      }
    }
#endif
    break;
  /*
   *	<attr> ::= CONSTANT |
   */
  case ATTR20:
    et_type = 0;
#ifdef CUDAG
    if (cuda_enabled("constant")) {
      if ((gbl.currsub && CUDAG(gbl.currsub) &&
           !(CUDAG(gbl.currsub) & CUDA_HOST)) ||
          (gbl.currmod && !gbl.currsub)) {
        /* device routine, or module declaration part */
        et_type = ET_CONSTANT;
      } else {
        error(134, 3, gbl.lineno, et[ET_CONSTANT].name,
              "not allowed in host subprograms");
      }
    }
#endif
    break;
  /*
   *	<attr> ::= PROTECTED |
   */
  case ATTR21:
    et_type = ET_PROTECTED;
    if (!IN_MODULE_SPEC) {
      error(155, 3, gbl.lineno,
            "PROTECTED may only appear in the specification part of a MODULE",
            CNULL);
    }
    break;
  /*
   *	<attr> ::= ASYNCHRONOUS
   */
  case ATTR22:
    et_type = ET_ASYNCHRONOUS;
    break;
  /*
   *	<attr> ::= ABSTRACT |
   */
  case ATTR23:
    /* anything here? */
    break;
  /*
   *	<attr> ::= TEXTURE
   */
  case ATTR24:
    if (cuda_enabled("texture"))
      et_type = ET_TEXTURE;
    else
      et_type = 0;
    break;

  /*
   *      <attr> ::= KIND |
   */
  case ATTR25:
    et_type = ET_KIND;
    break;
  /*
   *      <attr> ::= LEN |
   */
  case ATTR26:
    et_type = ET_LEN;
    break;
  /*
   *	<attr> ::= CONTIGUOUS |
   */
  case ATTR27:
    et_type = ET_CONTIGUOUS;
    break;
  /*
   *	<attr> ::= MANAGED |
   */
  case ATTR28:
    et_type = 0;
    if (cuda_enabled("managed")) {
#if defined(TARGET_OSX)
      /* not supported */
      error(538, 3, gbl.lineno, CNULL, CNULL);
#else
      /* supported */
      et_type = ET_MANAGED;
#endif
    }
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <bind attr> ::= ( <id name> ) |
   */
  case BIND_ATTR1:
    /* see also FUNC_SUFFIX2 for a copy of this processing */

    bind_attr.exist = -1;
    bind_attr.altname = 0;

    np = scn.id.name + SST_CVALG(RHS(2));
    if (sem_strcmp(np, "c") != 0) {
      error(4, 3, gbl.lineno, "Illegal BIND -", np);
    } else {
      bind_attr.exist = DA_B(DA_C);
    }

    break;
  /*
   *      <bind attr> ::=  ( <id name> , <id name> = <quoted string> )
   */
  case BIND_ATTR2:
    np = scn.id.name + SST_CVALG(RHS(4));
    if (sem_strcmp(np, "name") != 0) {
      error(4, 3, gbl.lineno, "Illegal BIND syntax. Expecting: NAME Got:", np);
    }

    bind_attr.exist = -1;
    bind_attr.altname = 0;

    np = scn.id.name + SST_CVALG(RHS(2));
    if (sem_strcmp(np, "c") != 0) {
      error(4, 3, gbl.lineno, "Illegal BIND -", np);
    } else {
      bind_attr.exist = DA_B(DA_C) | DA_B(DA_ALIAS);
      bind_attr.altname = SST_SYMG(RHS(6)); // altname may be ""
    }

    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <bind list> ::=  <bind list> , <bind entry> |
   */
  case BIND_LIST1:
    rhstop = 3;
    goto add_sym_to_bind_list;
    break;
  /*
   *      <bind list> ::= <bind entry>
   */
  case BIND_LIST2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <bind entry> ::= <common> |
   */
  case BIND_ENTRY1:
  /* fall through */
  /*
   *      <bind entry> ::= <id>
   */
  case BIND_ENTRY2:
    rhstop = 1;
  add_sym_to_bind_list:
    itemp = (ITEM *)getitem(0, sizeof(ITEM));
    itemp->next = ITEM_END;
    itemp->t.sptr = SST_SYMG(RHS(rhstop));
    itemp->ast = SST_ASTG(RHS(rhstop)); /* copied for <access> rules */
    if (rhstop == 1)
      /* adding first item to list */
      SST_BEGP(LHS, itemp);
    else
      /* adding subsequent items to list */
      SST_ENDG(RHS(1))->next = itemp;
    SST_ENDP(LHS, itemp);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<opt type spec> ::= |
   */
  case OPT_TYPE_SPEC1:
    entity_attr.access = ' ';
    SST_CVALP(LHS, 0);
    break;
  /*
   *	<opt type spec> ::= , <type attr list>
   */
  case OPT_TYPE_SPEC2:
    SST_CVALP(LHS, SST_CVALG(RHS(2)));
    SST_LSYMP(LHS, SST_LSYMG(RHS(2)));
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<type attr list> ::= <type attr list> , <type attr> |
   */
  case TYPE_ATTR_LIST1:
    switch (SST_CVALG(RHS(1)) & SST_CVALG(RHS(3))) {
    case 0x1:
      error(134, 3, gbl.lineno, "- duplicate", et[ET_BIND].name);
      SST_CVALP(RHS(3), 0);
      break;
    case 0x2:
      error(134, 3, gbl.lineno, "- duplicate", et[ET_ACCESS].name);
      SST_CVALP(RHS(3), 0);
      break;
    case 0x4: /* type extension */
      error(134, 3, gbl.lineno, "- duplicate", et[ET_ACCESS].name);
      SST_CVALP(RHS(3), 0);
      break;
    }
    SST_CVALP(LHS, SST_CVALG(RHS(1)) | SST_CVALG(RHS(3)));
    if (SST_CVALG(RHS(3)) & 0x4)
      SST_LSYMP(LHS, SST_LSYMG(RHS(3)));
    break;
  /*
   *	<type attr list> ::= <type attr>
   */
  case TYPE_ATTR_LIST2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<type attr> ::= BIND <bind attr> |
   */
  case TYPE_ATTR1:
    /* struct types are already properly aligned for C compatibility;
     * pass up presence of BIND so that the type can be marked as
     * BIND(C) with the flag CFUNC.
     */
    SST_CVALP(LHS, 0x1);
    break;
  /*
   *	<type attr> ::= <access spec>
   */
  case TYPE_ATTR2:
    SST_CVALP(LHS, 0x2);
    break;
  /*
   *      <type attr> ::= EXTENDS ( <id> ) |
   */
  case TYPE_ATTR3:
    /* type extension */
    SST_CVALP(LHS, 0x4);
    sptr = SST_SYMG(RHS(3));
    while (STYPEG(sptr) == ST_ALIAS)
      sptr = SYMLKG(sptr);
    if (STYPEG(sptr) == ST_USERGENERIC && GTYPEG(sptr)) {
      sptr = GTYPEG(sptr);
    }
    if (sptr > NOSYM && STYPEG(sptr) != ST_TYPEDEF) {
      int sym = findByNameStypeScope(SYMNAME(sptr), ST_TYPEDEF, -1);
      if (sym > NOSYM)
        sptr = sym;
    }
    if (DTY(DTYPEG(sptr)) != TY_DERIVED) {
      error(155, 4, gbl.lineno, "Invalid type extension", NULL);
    } else {
      /* Check for private type extension */

      int tag = DTY(DTYPEG(sptr) + 3);
      int tag_scope = SCOPEG(tag);
      int host_scope = stb.curr_scope;

      if (PRIVATEG(tag)) {
        if (STYPEG(tag_scope) == ST_MODULE && STYPEG(host_scope) != ST_MODULE)
          host_scope = SCOPEG(host_scope);
        if (tag_scope != host_scope)
          error(155, 3, gbl.lineno,
                "Cannot extend type with PRIVATE attribute -", SYMNAME(tag));
      }
    }
    sem.extends = sptr;
    SST_LSYMP(LHS, sptr);
    break;
  /*
   *	<type attr> ::= ABSTRACT |
   */
  case TYPE_ATTR4:
    SST_CVALP(LHS, 0x8);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<access spec> ::= PUBLIC  |
   */
  case ACCESS_SPEC1:
    entity_attr.access = 'u';
    if (!IN_MODULE_SPEC)
      ERR310("PUBLIC/PRIVATE may only appear in a MODULE scoping unit", CNULL);
    break;
  /*
   *	<access spec> ::= PRIVATE
   */
  case ACCESS_SPEC2:
    if (sem.type_mode == 2 && IN_MODULE_SPEC) {
      /* private seen in type bound procedure "contains" section */
      entity_attr.access = '0';
    } else
      entity_attr.access = 'v';
    if (!IN_MODULE_SPEC)
      ERR310("PUBLIC/PRIVATE may only appear in a MODULE scoping unit", CNULL);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<access list> ::= <access list>, <access> |
   */
  case ACCESS_LIST1:
    rhstop = 3;
    goto add_sym_to_list;
  /*
   *	<access list> ::= <access>
   */
  case ACCESS_LIST2:
    rhstop = 1;
    goto add_sym_to_list;

  /* ------------------------------------------------------------------ */
  /*
   *	<access> ::= <ident> |
   */
  case ACCESS1:
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<access> ::= <id name> ( <operator> ) |
   */
  case ACCESS2:
    np = scn.id.name + SST_CVALG(RHS(1));
    if (sem_strcmp(np, "operator") == 0)
      SST_SYMP(LHS, SST_LSYMG(RHS(3)));
    else {
      error(34, 3, gbl.lineno, np, CNULL);
      SST_SYMP(LHS, getsymbol(".34"));
    }
    SST_ASTP(LHS, 1); /* mark this as being from OPERATOR stmt */
    break;
  /*
   *	<access> ::= <id name> ( = )
   */
  case ACCESS3:
    np = scn.id.name + SST_CVALG(RHS(1));
    if (sem_strcmp(np, "assignment") == 0) {
      sptr = get_intrinsic_opr(OP_ST, 0);
      SST_SYMP(LHS, sptr);
    } else {
      error(34, 3, gbl.lineno, np, CNULL);
      SST_SYMP(LHS, getsymbol(".34"));
    }
    SST_ASTP(LHS, 1); /* treat as if from OPERATOR stmt */
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<seq> ::= SEQUENCE |
   */
  case SEQ1:
    if (!INSIDE_STRUCT || STSK_ENT(0).type != 'd') {
      error(155, 3, gbl.lineno,
            "SEQUENCE must appear in a derived type definition", CNULL);
    }
    SST_CVALP(LHS, 's');
    break;
  /*
   *	<seq> ::= NOSEQUENCE
   */
  case SEQ2:
    error(34, 3, gbl.lineno, "NOSEQUENCE", CNULL);
    SST_CVALP(LHS, 'n');
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<intent> ::= INTENT ( <id name> ) |
   */
  case INTENT1:
    np = scn.id.name + SST_CVALG(RHS(3));
    if (sem_strcmp(np, "in") == 0)
      entity_attr.intent = INTENT_IN;
    else if (sem_strcmp(np, "out") == 0)
      entity_attr.intent = INTENT_OUT;
    else if (sem_strcmp(np, "inout") == 0)
      entity_attr.intent = INTENT_INOUT;
    else {
      error(81, 3, gbl.lineno, "- illegal intent", np);
      entity_attr.intent = INTENT_DFLT;
    }
    break;
  /*
   *	<intent> ::= INTENT ( <id name> <id name> )
   */
  case INTENT2:
    np = scn.id.name + SST_CVALG(RHS(3));
    if (sem_strcmp(np, "in") == 0) {
      np = scn.id.name + SST_CVALG(RHS(4));
      if (sem_strcmp(np, "out") == 0)
        entity_attr.intent = INTENT_INOUT;
      else {
        error(81, 3, gbl.lineno, "- illegal intent in", np);
        entity_attr.intent = INTENT_DFLT;
      }
    } else {
      error(81, 3, gbl.lineno, "- illegal intent", np);
      entity_attr.intent = INTENT_DFLT;
    }
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<entity decl list> ::= <entity decl list> , <entity decl> |
   */
  case ENTITY_DECL_LIST1:
    rhstop = 3;
    goto add_entity_to_list;
  /*
   *	<entity decl list> ::= <entity decl>
   */
  case ENTITY_DECL_LIST2:
    rhstop = 1;
  add_entity_to_list:
    if (in_entity_typdcl) { /* only pass up list if hpf decls */
      SST_BEGP(LHS, ITEM_END);
      break;
    }
    itemp = (ITEM *)getitem(0, sizeof(ITEM));
    itemp->next = ITEM_END;
    itemp->t.sptr = SST_SYMG(RHS(rhstop));
    if (rhstop == 1)
      /* adding first item to list */
      SST_BEGP(LHS, itemp);
    else
      /* adding subsequent items to list */
      SST_ENDG(RHS(1))->next = itemp;
    SST_ENDP(LHS, itemp);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<entity decl> ::= <entity id> |
   */
  case ENTITY_DECL1:
    /* only pass up sym if hpf decls */
    if (!in_entity_typdcl)
      break;

    inited = FALSE;
    goto entity_decl_shared;
  /*
   *	<entity decl> ::= <entity id> <init beg> <expression> |
   */
  case ENTITY_DECL2:
    if (!in_entity_typdcl) {
      error(114, 3, gbl.lineno, SYMNAME(SST_SYMG(RHS(1))), CNULL);
      break;
    }
    sptr = SST_SYMG(RHS(1));
    stype1 = STYPEG(sptr);
    if (IS_INTRINSIC(stype1)) {
      if ((sptr = newsym(sptr)) == 0)
        /* Symbol frozen as an intrinsic, ignore in COMMON */
        break;
      SST_SYMP(LHS, sptr);
    }
    inited = TRUE;
    sem.dinit_data = FALSE;
    goto entity_decl_shared;
  /*
   *	<entity decl> ::= <entity id> '=>' <id> ( )
   */
  case ENTITY_DECL3:
    if (!in_entity_typdcl) {
      error(114, 3, gbl.lineno, SYMNAME(SST_SYMG(RHS(1))), CNULL);
      break;
    }
    sptr = SST_SYMG(RHS(1));
    stype1 = STYPEG(sptr);
    if (IS_INTRINSIC(stype1)) {
      if ((sptr = newsym(sptr)) == 0)
        /* Symbol frozen as an intrinsic, ignore in COMMON */
        break;
      SST_SYMP(LHS, sptr);
    }
    sptr = SST_SYMG(RHS(3));
    sptr = refsym(sptr, OC_OTHER);
    SST_SYMP(RHS(3), sptr);
    SST_IDP(RHS(3), S_IDENT);
    sem.dinit_data = TRUE;
    (void)mkvarref(RHS(3), ITEM_END);
    sem.dinit_data = FALSE;
    inited = TRUE;

  entity_decl_shared:
    sptr = SST_SYMG(RHS(1));
    if (!(entity_attr.exist & ET_B(ET_BIND))) {
      sptr = block_local_sym(SST_SYMG(RHS(1)));
      SST_SYMP(RHS(1), sptr);
    }
    SST_SYMP(RHS(1), sptr);
    if (sem.new_param_dt) {
      dtype = DTYPEG(sptr);
      if (DTY(dtype) == TY_ARRAY) {
        DTY(dtype + 1) = sem.new_param_dt;
      } else {
        DTYPEP(sptr, sem.new_param_dt);
      }
      fix_type_param_members(sptr, sem.new_param_dt);
    }

    if (!sem.interface)
      add_type_param_initialize(sptr);

    if (sem.class && sem.type_mode &&
        !(entity_attr.exist & (ET_B(ET_ALLOCATABLE) | ET_B(ET_POINTER)))) {
      error(155, 3, gbl.lineno, "CLASS component must be "
                                "allocatable or pointer -",
            SYMNAME(sptr));
    }
    sem.gdtype = SST_GDTYPEG(RHS(1));
    sem.gty = SST_GTYG(RHS(1));
    if (flg.xref)
      xrefput(sptr, 'd');
    dtype = mod_type(sem.gdtype, sem.gty, lenspec[1].kind, lenspec[1].len,
                     lenspec[1].propagated, sptr);
    if (DCLDG(sptr) && !RESULTG(sptr) && !IS_INTRINSIC(STYPEG(sptr))) {
      switch (STYPEG(sptr)) {
      /*  any cases for which a data type does not apply */
      case ST_MODULE:
      case ST_NML:
        error(44, 3, gbl.lineno, SYMNAME(sptr), CNULL);
        break;
      default:
        /* data type for ident has already been specified */
        if (DDTG(DTYPEG(sptr)) == dtype)
          error(119, 2, gbl.lineno, SYMNAME(sptr), CNULL);
        else
          error(37, 3, gbl.lineno, SYMNAME(sptr), CNULL);
      }
      /* to avoid setting symbol table entry's stype field */
      goto entity_decl_end;
    } else {
      switch (STYPEG(sptr)) {
      /* any cases for which a type must be identical to the variable's
       * implicit type.
       */
      case ST_PARAM:
        if (!(entity_attr.exist & ET_B(ET_PARAMETER)) && DTYPEG(sptr) != dtype)
          error(37, 3, gbl.lineno, SYMNAME(sptr), CNULL);
        break;
      default:
        break;
      }
    }
    /*
     * Finalize the dtype of the variable.
     * Determine the tentative stype we want give to the variable if
     * it's still ST_UNKNOWN or ST_IDENT.
     */
    DCLDP(sptr, TRUE);
    set_char_attributes(sptr, &dtype);

    if (DTY(DTYPEG(sptr)) == TY_ARRAY) {
      int dims, idx, lbast;
      if (sem.new_param_dt && has_type_parameter(DTY(DTYPEG(sptr) + 1))) {
        /* Make sure we use the new parameterized dtype */
        dtype = sem.new_param_dt;
      }
      DTY(DTYPEG(sptr) + 1) = dtype;
      if (DTY(dtype) == TY_DERIVED && DTY(dtype + 3) &&
          DISTMEMG(DTY(dtype + 3))) {
        error(451, 3, gbl.lineno, SYMNAME(sptr), CNULL);
      }
      dtype = DTYPEG(sptr);
      if (AD_ASSUMSZ(AD_DPTR(dtype)) && DTY(dtype + 1) == TY_INT &&
          SCG(sptr) != SC_DUMMY && !(entity_attr.exist & ET_B(ET_PARAMETER))) {
        error(155, 3, gbl.lineno,
              "Implied-shape array must have the PARAMETER attribute -",
              SYMNAME(sptr));
        goto entity_decl_end;
      }
      dims = AD_NUMDIM(AD_DPTR(dtype));
      for (idx = 0; idx < dims; idx++) {
        lbast = AD_LWAST(AD_DPTR(dtype), idx);
        if (AD_ASSUMSZ(AD_DPTR(dtype)) && DTY(dtype + 1) == TY_INT &&
            SCG(sptr) != SC_DUMMY && A_TYPEG(lbast) != A_CNST) {
          error(155, 3, gbl.lineno,
                "Implied-shape array lower bound is not constant -",
                SYMNAME(sptr));
          goto entity_decl_end;
        }
      }
    } else if (DTY(DTYPEG(sptr)) == TY_PTR &&
               DTY(DTY(DTYPEG(sptr) + 1)) == TY_PROC) {
      /* ptr to a function, set the func return value and the pointer flag */
      int func_dtype = DTY(DTYPEG(sptr) + 1);
      DTY(func_dtype + 5) = dtype;
    } else if (!USELENG(sptr) && !LENG(sptr)) {
      /* parameterized derived type TBD: array case???? */
      DTYPEP(sptr, (!sem.new_param_dt) ? dtype : sem.new_param_dt);
      if (SCG(sptr) == SC_DUMMY) {
        put_length_type_param(DTYPEG(sptr), 3);
      }
    }
    if (DTY(dtype) == TY_ARRAY)
      is_array = TRUE;
    else
      is_array = FALSE;
    is_member = FALSE;
    stype = STYPEG(sptr);
    if (stype == ST_MEMBER) {
      stype = 0;
      is_member = TRUE;
    } else if (stype == ST_ENTRY)
      stype = 0;
    else if (is_array)
      stype = ST_ARRAY;
    else if (DTY(dtype) == TY_STRUCT)
      stype = ST_STRUCT;

    no_init = FALSE;
    et_type = 0;
    et_bitv = entity_attr.exist;
    /* Loop through all assigned attributes */
    for (; et_bitv; et_bitv >>= 1, et_type++) {
      if ((et_bitv & 0x0001) == 0)
        continue;
      switch (et_type) {
      default:
        continue;
      case ET_ACCESS:
        if (sptr == ST_ARRAY && ADJARRG(sptr))
          error(84, 3, gbl.lineno, SYMNAME(sptr),
                "- must not be an automatic array");
        else if (is_member) {
          if (entity_attr.access == 'v')
            PRIVATEP(sptr, 1);
          else
            PRIVATEP(sptr, 0);
        } else {
          accessp = (ACCL *)getitem(3, sizeof(ACCL));
          accessp->sptr = sptr;
          accessp->type = entity_attr.access;
          accessp->next = sem.accl.next;
          accessp->oper = ' ';
          sem.accl.next = accessp;
        }
        break;
      case ET_ALLOCATABLE:
        if (is_array) {
          ad = AD_DPTR(dtype);
          if (AD_DEFER(ad) == 0)
            error(84, 3, gbl.lineno, SYMNAME(sptr),
                  "- must be a deferred shape array");
          else {
            if (AD_ASSUMSHP(ad)) {
              /* this is an error if it isn't a dummy; the
               * declaration could occur before its entry, so
               * the check needs to be performed in semfin.
               */
              ASSUMSHPP(sptr, 1);
              if (!XBIT(54, 2) && !(XBIT(58, 0x400000) && TARGETG(sptr)))
                SDSCS1P(sptr, 1);
            }
            mk_defer_shape(sptr);
          }
        }
        ALLOCP(sptr, 1);
        ALLOCATTRP(sptr, 1);
        if (STYPEG(sptr) == ST_MEMBER) {
          ALLOCFLDP(DTY(ENCLDTYPEG(sptr) + 3), 1);
        }

        dtype = DTYPEG(sptr);
        if (DTY(dtype) == TY_ARRAY) {
          dtype = DTY(dtype + 1);
          if (sem.class)
            CLASSP(sptr, 1);
        }
        if (STYPEG(sptr) == ST_MEMBER && DTY(dtype) == TY_DERIVED &&
            has_finalized_component(sptr)) {
          FINALIZEDP(sptr, 1);
        }
        if (!(DTY(DTYPEG(sptr)) == TY_ARRAY && STYPEG(sptr) == ST_MEMBER) &&
            DTY(dtype) == TY_DERIVED) {
          /* Note: Do not execute this case for array
           * components since they already have a full array descriptor
           * embedded in the derived type.
           */
          if (sem.class)
            CLASSP(sptr, 1);
          set_descriptor_rank(TRUE);
          get_static_descriptor(sptr);
          set_descriptor_rank(FALSE);

          get_all_descriptors(sptr);

          if (SCG(sptr) != SC_DUMMY)
            SCP(sptr, SC_BASED);
          ALLOCDESCP(sptr, TRUE);
        } else if (SCG(sptr) == SC_DUMMY) {
          get_static_descriptor(sptr);
          get_all_descriptors(sptr);
        } else if (!INSIDE_STRUCT && SDSCG(sptr) == 0 &&
                   (DDTG(DTYPEG(sptr)) == DT_DEFERCHAR ||
                    DDTG(DTYPEG(sptr)) == DT_DEFERNCHAR)) {
          if (SCG(sptr) != SC_DUMMY)
            SCP(sptr, SC_BASED); /* Don't change dummy */
          get_static_descriptor(sptr);
          get_all_descriptors(sptr);
          ALLOCDESCP(sptr, TRUE);
        } else {
          SCP(sptr, SC_BASED);
        }
        no_init = TRUE;
        break;
      case ET_CONTIGUOUS:
#ifdef CONTIGATTRP
        CONTIGATTRP(sptr, 1);
#endif
        break;
      case ET_DIMENSION:
        break;
      case ET_EXTERNAL:
        if (DTY(DTYPEG(sptr)) == TY_ARRAY) {
          /* conflict with EXTERNAL */
          error(134, 3, gbl.lineno, "- array bounds not allowed with external",
                SYMNAME(sptr));
        }
        /* Produce procedure symbol based on attributes */
        sptr = decl_procedure_sym(sptr, 0, entity_attr.exist);
        sptr =
            setup_procedure_sym(sptr, 0, entity_attr.exist, entity_attr.access);
        if (!TYPDG(sptr)) {
          TYPDP(sptr, 1);
          if (SCG(sptr) == SC_DUMMY) {
            IS_PROC_DUMMYP(sptr, 1);
          }
        }
        stype = 0;
        no_init = TRUE;
        break;
      case ET_INTENT:
        INTENTP(sptr, entity_attr.intent);
        if (sem.interface) {
          if (SCG(sptr) != SC_DUMMY) {
            error(134, 3, gbl.lineno,
                  "- intent specified for nondummy argument", SYMNAME(sptr));
          } else if (POINTERG(sptr)) {
            error(134, 3, gbl.lineno, "- intent specified for pointer argument",
                  SYMNAME(sptr));
          } else if (STYPEG(sptr) == ST_PROC) {
            error(134, 3, gbl.lineno,
                  "- intent specified for dummy subprogram argument",
                  SYMNAME(sptr));
          }
        } else {
          /* defer checking of storage class until semfin */
          itemp1 = (ITEM *)getitem(3, sizeof(ITEM));
          itemp1->next = sem.intent_list;
          sem.intent_list = itemp1;
          itemp1->t.sptr = sptr;
          itemp1->ast = gbl.lineno;
        }
        break;
      case ET_INTRINSIC:
        stype = STYPEG(sptr);
        if (IS_INTRINSIC(stype)) {
          EXPSTP(sptr, 1); /* Freeze as an intrinsic */
          TYPDP(sptr, 1);  /* appeared in INTRINSIC statement */
        } else
          error(126, 3, gbl.lineno, SYMNAME(sptr), CNULL);
        stype = 0;
        no_init = TRUE;
        break;
      case ET_OPTIONAL:
        OPTARGP(sptr, 1);
        break;
      case ET_PARAMETER:
        break; /* handle after scanning all attributes */
      case ET_POINTER:
        POINTERP(sptr, TRUE);
        if (sem.contiguous)
          CONTIGATTRP(sptr, 1);
        if (DTYG(DTYPEG(sptr)) == TY_DERIVED && XBIT(58, 0x40000)) {
          F90POINTERP(sptr, TRUE);
        }
        if (is_array) {
          ad = AD_DPTR(dtype);
          if (AD_DEFER(ad) == 0)
            error(84, 3, gbl.lineno, SYMNAME(sptr),
                  "- must be a deferred shape array");
        }
        dtype = DTYPEG(sptr);
        if (DTY(dtype) == TY_ARRAY) {
          dtype = DTY(dtype + 1);
          if (sem.class)
            CLASSP(sptr, 1);
        }
        if (STYPEG(sptr) == ST_MEMBER && DTY(dtype) == TY_DERIVED &&
            has_finalized_component(sptr)) {
          FINALIZEDP(sptr, 1);
        }
        if (!(DTY(DTYPEG(sptr)) == TY_ARRAY && STYPEG(sptr) == ST_MEMBER) &&
            DTY(dtype) == TY_DERIVED) {
          int sav_sc;
          if (sem.class)
            CLASSP(sptr, TRUE);
          set_descriptor_rank(TRUE);
          sav_sc = 0;
          if (IN_MODULE && in_save_scope(sptr)) {
            /* SAVE is set, so we need to set our descriptor
             * to SC_STATIC here instead of later (in do_save() of
             * semfin.c). Otherwise, we may get unresolved symbol
             * link errors because we save descriptor early on in
             * the module.
             */
            /* Note: The SC_STATIC fix is only required for polymorphic
             * objects. For non-polymorphic objects, we can safely use
             * SC_LOCAL since the type does not mutate.
             */
            sav_sc = get_descriptor_sc();
            set_descriptor_sc(sem.class ? SC_STATIC : SC_LOCAL);
          }
          if (sem.class || has_tbp_or_final(dtype) ||
              STYPEG(sptr) == ST_MEMBER || DTY(DTYPEG(sptr)) == TY_ARRAY) {
            ALLOCDESCP(sptr, TRUE);
          }
          get_static_descriptor(sptr);
          set_descriptor_rank(FALSE);
          if (IN_MODULE && in_save_scope(sptr)) {
            set_descriptor_sc(sav_sc);
          }
          if (!sem.class)
            CCSYMP(SDSCG(sptr), TRUE);
        } else if (!INSIDE_STRUCT && SDSCG(sptr) == 0 &&
                   (DDTG(DTYPEG(sptr)) == DT_DEFERCHAR ||
                    DDTG(DTYPEG(sptr)) == DT_DEFERNCHAR)) {
          if (SCG(sptr) != SC_DUMMY) /* Can't change dummy */
            SCP(sptr, SC_BASED);
          get_static_descriptor(sptr);
          get_all_descriptors(sptr);
        }
        break;
      case ET_SAVE:
/* <ident> must be a variable or an array; it cannot be a dummy
 * argument or common block member.
 */
        if (stype == 0)
          error(50, 3, gbl.lineno, SYMNAME(sptr), CNULL);
        else if (stype == ST_ARRAY && (ASUMSZG(sptr) || ADJARRG(sptr))) {
          if (ASUMSZG(sptr))
            error(155, 3, gbl.lineno,
                  "An assumed-size array cannot have the SAVE attribute -",
                  SYMNAME(sptr));
          else if (SCG(sptr) == SC_DUMMY)
            error(155, 3, gbl.lineno,
                  "An adjustable array cannot have the SAVE attribute -",
                  SYMNAME(sptr));
          else
            error(155, 3, gbl.lineno,
                  "An automatic array cannot have the SAVE attribute -",
                  SYMNAME(sptr));
        } else if (flg.standard && gbl.currsub && !is_impure(gbl.currsub)) {
          error(170, 2, gbl.lineno,
                sem.block_scope ?
                  "SAVE attribute for a BLOCK variable of a PURE subroutine" :
                  "SAVE attribute for a local variable of a PURE subroutine",
                CNULL);
        } else if ((SCG(sptr) == SC_NONE || SCG(sptr) == SC_LOCAL ||
                    SCG(sptr) == SC_BASED) &&
                   (stype == ST_VAR || stype == ST_ARRAY ||
                    stype == ST_STRUCT || stype == ST_IDENT)) {
          sem.savloc = TRUE;
          SAVEP(sptr, 1);
          /* SCP(sptr, SC_LOCAL);
           * SAVE is now an attribute and may appear allocatable; the
           * appearance of a variable in a SAVE statement is no longer
           * sufficient to define the variable's storage class.
           */
        } else
          error(39, 2, gbl.lineno, SYMNAME(sptr), CNULL);
        break;
      case ET_TARGET:
        TARGETP(sptr, 1);
        if( XBIT(58, 0x400000) && SCG(sptr) == SC_DUMMY && ASSUMSHPG(sptr) )
             SDSCS1P(sptr,0);
        break;
      case ET_AUTOMATIC:
        /* <ident> must be a variable or an array; it cannot be a dummy
         * argument or common block member.
         */
        if (stype == 0)
          error(50, 3, gbl.lineno, SYMNAME(sptr), CNULL);
        else if (stype == ST_ARRAY && (ASUMSZG(sptr) || ADJARRG(sptr)))
          error(50, 3, gbl.lineno, SYMNAME(sptr), CNULL);
        else if (flg.standard)
          error(171, 2, gbl.lineno, "AUTOMATIC", CNULL);
        else if ((SCG(sptr) == SC_NONE || SCG(sptr) == SC_LOCAL ||
                  SCG(sptr) == SC_BASED) &&
                 (stype == ST_VAR || stype == ST_ARRAY || stype == ST_STRUCT ||
                  stype == ST_IDENT)) {
          if (SCG(sptr) == SC_BASED && MIDNUMG(sptr))
            symatterr(2, sptr, "AUTOMATIC");
          else if (gbl.rutype != RU_PROG || CONSTRUCTSYMG(sptr)) {
            sem.autoloc = TRUE;
            /* TBD -- need to resolve SC_BASED vs SC_LOCAL & SCFXD
             * DON'T FORGET the AUTOMATIC & STATIC statements.
             */
            SCP(sptr, SC_LOCAL);
            SCFXDP(sptr, 1);
          }
        } else
          symatterr(2, sptr, "AUTOMATIC");
        break;
      case ET_STATIC:
        /* <ident> must be a variable or an array; it cannot be a dummy
         * argument or common block member.
         */
        if (stype == 0)
          error(50, 3, gbl.lineno, SYMNAME(sptr), CNULL);
        else if (stype == ST_ARRAY && (ASUMSZG(sptr) || ADJARRG(sptr)))
          error(50, 3, gbl.lineno, SYMNAME(sptr), CNULL);
        else if (flg.standard)
          error(171, 2, gbl.lineno, "STATIC", CNULL);
        else if ((SCG(sptr) == SC_NONE || SCG(sptr) == SC_LOCAL ||
                  SCG(sptr) == SC_BASED) &&
                 (stype == ST_VAR || stype == ST_ARRAY || stype == ST_STRUCT ||
                  stype == ST_IDENT)) {
          if (SCG(sptr) == SC_BASED && MIDNUMG(sptr))
            symatterr(2, sptr, "STATIC");
          /* just use the save semantics */
          sem.savloc = TRUE;
          SAVEP(sptr, 1);
        } else
          symatterr(2, sptr, "STATIC");
        break;
      case ET_BIND:
        if (!IN_MODULE)
          error(280, 2, gbl.lineno, "BIND: allowed only in module", 0);
        process_bind(sptr);
        break;
      case ET_VALUE:
        if (CLASSG(sptr)) {
          error(155, 3, gbl.lineno, "Polymorphic variable"
                                    " cannot have VALUE attribute -",
                SYMNAME(sptr));
        }
        if ((DTY(DTYPEG(sptr)) == TY_CHAR || DTY(DTYPEG(sptr)) == TY_NCHAR) &&
            string_length(DTYPEG(sptr)) != 1) {
          error(155, 3, gbl.lineno,
                "Multi-CHARACTER strings can not have the VALUE attribue - ",
                SYMNAME(sptr));
        }
        PASSBYVALP(sptr, 1);
        PASSBYREFP(sptr, 0);
        break;
      case ET_VOLATILE:
        VOLP(sptr, 1);
        break;
      case ET_ASYNCHRONOUS:
/*
 * do we need a specific flag set a flag? OR, just hit it
 * with VOLP?  Wait until it really matters.
 */
#ifdef ASYNCP
        /* Yes, flag is needed so we can check
         * characteristics of dummy arguments for type bound
         * procedures.
         */
        ASYNCP(sptr, 1);
#endif
        break;
      case ET_PROTECTED:
        PROTECTEDP(sptr, 1);
        break;
      case ET_KIND:
#ifdef KINDP
        if (!DT_ISINT(DTYPEG(sptr))) {
          error(155, 3, gbl.lineno,
                "derived type parameter must be an INTEGER -", SYMNAME(sptr));
        }
        KINDP(sptr, -1);
#endif
        break;
      case ET_LEN:
#ifdef KINDP
        if (!DT_ISINT(DTYPEG(sptr))) {
          error(155, 3, gbl.lineno,
                "derived type parameter must be an INTEGER -", SYMNAME(sptr));
        }
        KINDP(sptr, -1);
        LENPARMP(sptr, 1);
#endif
        break;
      }
    }
    if (sem.new_param_dt)
      chk_new_param_dt(sptr, sem.new_param_dt);
    if ((DTYPEG(sptr) == DT_DEFERCHAR || DTYPEG(sptr) == DT_DEFERNCHAR) &&
        (!POINTERG(sptr) && !ALLOCATTRG(sptr))) {
      error(155, 3, gbl.lineno, "Object with deferred character length"
                                " (:) must be a pointer or an allocatable -",
            SYMNAME(sptr));
    }

    if (RESULTG(sptr) && STYPEG(sptr) != ST_ENTRY &&
        (entity_attr.exist & ET_B(ET_PARAMETER))) {
      error(155, ERR_Severe, gbl.lineno, "Function result cannot have the"
                                         " PARAMETER attribute -",
            SYMNAME(sptr));
      goto entity_decl_end;
    }
    if ((entity_attr.exist & ET_B(ET_PARAMETER)) || 
        do_fixup_param_vars_for_derived_arrays(inited, sptr, 
                                               SST_IDG(RHS(3)))) {
      if (inited) {
        if (DTY(dtype) == TY_ARRAY && AD_ASSUMSZ(AD_DPTR(dtype)) &&
            DTY(SST_DTYPEG(RHS(3))) != TY_ARRAY) {
          error(155, 3, gbl.lineno, "Implied-shape array must be initialized "
                "with a constant array -", SYMNAME(sptr));
          goto entity_decl_end;
        }
        fixup_param_vars(top, RHS(3));
        /* Don't build ACLS for scalar or unknown data type array parameters. */
        if (((DTY(dtype) != TY_DERIVED) && (DTY(dtype) != TY_ARRAY)) ||
            ((DTY(dtype) == TY_ARRAY) && (DTY(dtype + 1) == DT_NONE))) {
          goto entity_decl_end;
        }
      } else {
        error(143, 3, gbl.lineno, SYMNAME(sptr), CNULL);
        goto entity_decl_end;
      }
    }

    if (RESULTG(sptr) && STYPEG(sptr) != ST_ENTRY) {
      if (inited) {
        error(155, ERR_Severe, gbl.lineno, "Function result cannot have"
                                           " an initializer -",
              SYMNAME(sptr));
        goto entity_decl_end;
      }
      /* set the type for the entry point as well */
      copy_type_to_entry(sptr);
    }
    if (stype) {
      if (stype != STYPEG(sptr) && STYPEG(sptr) != ST_PARAM) {
        if (STYPEG(sptr) == ST_VAR && stype == ST_ARRAY) {
          /* HACK: if the item being defined has an initializer
           * that contains an intrinsic call that uses the item
           * as an argument, then the argument handling may have
           * changed the item's STYPE to ST_VAR.  If the item is
           * an array, change its STYPE to ST_IDENT so declsym
           * will function correctly.
           */
          STYPEP(sptr, ST_IDENT);
        }
        sptr = declsym(sptr, stype, TRUE);
      }
      if (stype == ST_ARRAY && !F90POINTERG(sptr)) {
        if (POINTERG(sptr) || MDALLOCG(sptr) ||
            (ALLOCATTRG(sptr) && STYPEG(sptr) == ST_MEMBER)) {
          int dty = DTYPEG(sptr);
          get_static_descriptor(sptr); 
          get_all_descriptors(sptr);
          if (DTY(dty) == TY_ARRAY) {
            dty = DTY(dty + 1);
          }
          if (DTY(dty) == TY_DERIVED && SCG(sptr) != SC_DUMMY) {  
            /* initialize the type field in the descriptor */
            int astnew, type;
            type = get_static_type_descriptor(DTY(dty + 3));
            astnew = mk_set_type_call(mk_id(SDSCG(sptr)), mk_id(type), FALSE);
            add_stmt(astnew);
          }
        }
      }
    }
    if (INSIDE_STRUCT && XBIT(58, 0x10000) && !F90POINTERG(sptr)) {
      /* we are processing a member, and we must handle all pointers */
      /* do we need descriptors for this member? */
      if (POINTERG(sptr) || ALLOCG(sptr) || ADJARRG(sptr) || RUNTIMEG(sptr)) {
        set_preserve_descriptor(ALLOCDESCG(sptr));
        get_static_descriptor(sptr);
        get_all_descriptors(sptr);
        SCP(sptr, SC_BASED);
        set_preserve_descriptor(0);
      }
    }
    if (inited) { /* check if symbol is data initialized */
      if (INSIDE_STRUCT && (STSK_ENT(0).type == 'd')) {
        if (no_init) {
          error(114, 3, gbl.lineno, SYMNAME(sptr), CNULL);
          goto entity_decl_end;
        }
        stsk = &STSK_ENT(0);
        if (SST_IDG(RHS(3)) == S_LVALUE || SST_IDG(RHS(3)) == S_EXPR ||
            SST_IDG(RHS(3)) == S_IDENT || SST_IDG(RHS(3)) == S_CONST) {
          mkexpr(RHS(3));
          ast = SST_ASTG(RHS(3));
          if (has_kind_parm_expr(ast, stsk->dtype, 1)) {
            if (chk_kind_parm_expr(ast, stsk->dtype, 1, 1)) {
              INITKINDP(sptr, 1);
              PARMINITP(sptr, ast);
            }
          } else if (DTY(DTYPEG(sptr)) == TY_ARRAY) {
            int dim;
            ad = AD_DPTR(DTYPEG(sptr));
            for (dim = 0; dim < AD_NUMDIM(ad); ++dim) {
              int lb = AD_LWAST(ad, dim);
              int ub = AD_UPAST(ad, dim);
              if (has_kind_parm_expr(lb, stsk->dtype, 1) ||
                  has_kind_parm_expr(ub, stsk->dtype, 1)) {
                INITKINDP(sptr, 1);
                PARMINITP(sptr, ast);
                break;
              }
            }
          }
        }
        if (!INITKINDG(sptr))
          construct_acl_for_sst(RHS(3), DTYPEG(SST_SYMG(RHS(1))));
        if (!SST_ACLG(RHS(3))) {
          goto entity_decl_end;
        }

        ict = SST_ACLG(RHS(3));
        ict->sptr = sptr; /* field/component sptr */
        save_struct_init(ict);
        stsk = &STSK_ENT(0);
        if (stsk->ict_beg) {
          (stsk->ict_end)->next = SST_ACLG(RHS(3));
          stsk->ict_end = SST_ACLG(RHS(3));
        } else {
          stsk->ict_beg = SST_ACLG(RHS(3));
          stsk->ict_end = SST_ACLG(RHS(3));
        }
      } else {
        /* Data item (not TYPE component) initialization */
        if (no_init) {
          error(114, 3, gbl.lineno, SYMNAME(sptr), CNULL);
          goto entity_decl_end;
        }

        if (DTY(DTYPEG(sptr)) == TY_ARRAY && !POINTERG(sptr)) {
          if (ADD_DEFER(DTYPEG(sptr)) || ADD_NOBOUNDS(DTYPEG(sptr))) {
            error(155, 3, gbl.lineno, "Cannot initialize deferred-shape array",
                  SYMNAME(sptr));
            goto entity_decl_end;
          }
        }
        if (POINTERG(sptr)) {
          /* have
           *   ... :: <ptr> => NULL()
           * <ptr>$p, <ptr>$o, <ptr>$sd  will be needed */
          dtype = DTYPEG(sptr);
          if (DTY(dtype) == TY_ARRAY) {
            dtype = DTY(dtype + 1);
          }
          if ((DTY(DTYPEG(sptr)) != TY_ARRAY || STYPEG(sptr) != ST_MEMBER) &&
              DTY(dtype) == TY_DERIVED &&
              (sem.class || has_tbp_or_final(dtype) ||
               STYPEG(sptr) == ST_MEMBER || DTY(DTYPEG(sptr)) == TY_ARRAY))
            set_descriptor_rank(1);
          get_static_descriptor(sptr);

          if ((DTY(DTYPEG(sptr)) != TY_ARRAY || STYPEG(sptr) != ST_MEMBER) &&
              DTY(dtype) == TY_DERIVED &&
              (sem.class || has_tbp_or_final(dtype) ||
               STYPEG(sptr) == ST_MEMBER || DTY(DTYPEG(sptr)) == TY_ARRAY))
            set_descriptor_rank(0);
          get_all_descriptors(sptr);
        }

        if (SST_IDG(RHS(3)) == S_ACONST) {
          if (DTY(DTYPEG(sptr)) == TY_ARRAY) {
            if (AD_NUMDIM(AD_DPTR(DTYPEG(sptr))) !=
                AD_NUMDIM(AD_DPTR(SST_DTYPEG(RHS(3))))) {
              if (size_of_array(DTYPEG(sptr)) == 0 &&
                  DTY(SST_DTYPEG(RHS(3))) != TY_ARRAY) {
                /* i.e., a(0) == (/integer::/) */
                goto entity_decl_end;
              }
              error(155, 3, gbl.lineno,
                    "Shape of initializer does not match shape of",
                    SYMNAME(sptr));
              goto entity_decl_end;
            }
          } else if (POINTERG(sptr) || ALLOCATTRG(sptr)) {
            errsev(457);
            goto entity_decl_end;
          }
        }
        construct_acl_for_sst(RHS(3), DTYPEG(SST_SYMG(RHS(1))));
        if (!SST_ACLG(RHS(3))) {
          goto entity_decl_end;
        }

        dtype = DTYPEG(sptr);
        if (STYPEG(sptr) == ST_PARAM) {
          if (DTY(dtype) == TY_ARRAY || DTY(dtype) == TY_DERIVED) {
            CONVAL2P(sptr, put_getitem_p(save_acl(SST_ACLG(RHS(3)))));
            sptr = CONVAL1G(sptr);
          }
        } else if (DTY(dtype) == TY_DERIVED && !POINTERG(sptr)) {
          /* This used to be done in dinit_struct_constr. It is necessary */
          /* to get ADDRESS (i.e., offset into STATICS) set */
          if (STYPEG(sptr) == ST_IDENT || STYPEG(sptr) == ST_UNKNOWN) {
            STYPEP(sptr, ST_VAR);
          }
          if (SCG(sptr) == SC_NONE)
            SCP(sptr, SC_LOCAL);
          DINITP(sptr, 1);
          sym_is_refd(sptr);
        }

        ast = mk_id(sptr);
        SST_ASTP(RHS(1), ast);
        SST_DTYPEP(RHS(1), DTYPEG(SST_SYMG(RHS(1))));
        SST_SHAPEP(RHS(1), A_SHAPEG(ast));
        ivl = dinit_varref(RHS(1));
        dinit(ivl, SST_ACLG(RHS(3)));
      }
    } else if (DTY(DDTG(dtype)) == TY_DERIVED && !POINTERG(sptr) &&
               !ALLOCG(sptr) && !ADJARRG(sptr)) {
      int dt_dtype = DDTG(dtype);

      if (INSIDE_STRUCT) {
        /* Uninitialized declaration of a derived type data item.
         * Check for and handle any component intializations defined
         * for this derived type */
        build_typedef_init_tree(sptr, dt_dtype);
      } else if (DTY(dt_dtype + 5) && SCOPEG(sptr) &&
                 SCOPEG(sptr) == stb.curr_scope &&
                 STYPEG(stb.curr_scope) == ST_MODULE) {
        /*
         * a derived type module variable has component initializers,
         * so generate inits.
         */
        build_typedef_init_tree(sptr, dt_dtype);
      }
    } else {
      if (POINTERG(sptr)) {

        /* have
         *   ... :: <ptr>
         * <ptr>$p, <ptr>$o, <ptr>$sd  will be needed */
        if (!SDSCG(sptr))
          get_static_descriptor(sptr);

        if (!PTROFFG(sptr))
          get_all_descriptors(sptr);
      }
    }

  entity_decl_end:
    sem.dinit_error = FALSE;
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<entity id> ::= <ident> <opt len spec>  |
   */
  case ENTITY_ID1:
    set_len_attributes(RHS(2), 1);
    stype = ST_IDENT;
    dtype = -1;
    dtypeset = 0;
    sem.dinit_count = 1;
    if (entity_attr.exist & ET_B(ET_DIMENSION)) {
      if (entity_attr.dimension) {
        /* allow just one use of this data type record */
        dtype = entity_attr.dimension;
        dtypeset = 1;
        entity_attr.dimension = 0;
      } else {
        /* create a new array dtype record from the bounds information
         * saved earlier
         */
        BCOPY(sem.bounds, entity_attr.bounds, char, sizeof(sem.bounds));
        BCOPY(&sem.arrdim, entity_attr.arrdim, char, sizeof(sem.arrdim));
        dtype = mk_arrdsc();
        dtypeset = 1;
      }
      ad = AD_DPTR(dtype);
      if (AD_ASSUMSZ(ad) || AD_ADJARR(ad) || AD_DEFER(ad))
        sem.dinit_count = -1;
      stype = ST_ARRAY;
    } else
      ad = NULL;
    goto entity_id_shared;
  /*
   *	<entity id> ::= <ident> <opt len spec> <dim beg> <dimension list> ) <opt
   *len spec>
   */
  case ENTITY_ID2:
    /* Send len spec up with ident on semantic stack */
    if (SST_SYMG(RHS(6)) != -1) {
      if (SST_SYMG(RHS(2)) != -1)
        errsev(46);
      set_len_attributes(RHS(6), 1);
    } else
      set_len_attributes(RHS(2), 1);
    stype = ST_ARRAY;
    dtype = SST_DTYPEG(RHS(4));
    dtypeset = 1;
    ad = AD_DPTR(dtype);
    if (AD_ASSUMSZ(ad) || AD_ADJARR(ad) || AD_DEFER(ad) || sem.interface)
      sem.dinit_count = -1;
    else
      sem.dinit_count = ad_val_of(sym_of_ast(AD_NUMELM(ad)));
  entity_id_shared:
    sptr = SST_SYMG(RHS(1));
    if (!(entity_attr.exist & ET_B(ET_BIND))) {
      sptr = block_local_sym(SST_SYMG(RHS(1)));
      SST_SYMP(RHS(1), sptr);
    }
    if (!sem.kind_type_param && !sem.len_type_param &&
        sem.type_param_candidate) {
      sem.kind_type_param = sem.type_param_candidate;
      sem.type_param_candidate = 0;
    }
    if (INSIDE_STRUCT) {
      /* this may be an HPF directive in a derived type */
      stsk = &STSK_ENT(0);
      if (sem.is_hpf && STYPEG(sptr) == ST_MEMBER &&
          ENCLDTYPEG(sptr) == stsk->dtype) {
        /* do nothing */
      } else {
        if (STYPEG(sptr) != ST_UNKNOWN)
          SST_SYMP(LHS, (sptr = insert_sym(sptr)));
        SYMLKP(sptr, NOSYM);
        STYPEP(sptr, ST_MEMBER);
        if (!dtypeset)
          dtype = sem.gdtype;
        DTYPEP(sptr, dtype); /* must be done before link members */
        if (sem.kind_type_param) {
          USEKINDP(sptr, 1);
          if (sem.kind_candidate) {
            /* Save kind expression in component */
            mkexpr(sem.kind_candidate->t.stkp);
            KINDASTP(sptr, SST_ASTG(sem.kind_candidate->t.stkp));
          }
          KINDP(sptr, sem.kind_type_param);
        }
        if (sem.len_type_param) {
          USELENP(sptr, 1);
          LENP(sptr, sem.len_type_param);
        }
        if (sem.len_candidate) {
          int ty = DTY(DTYPEG(sptr));
          if (ty == TY_CHAR || ty == TY_NCHAR)
          {
            ast = SST_ASTG((SST *)sem.len_candidate->t.stkp);
            ty = get_type(2, ty, ast);
            DTYPEP(sptr, ty);
            USELENP(sptr, 1);
            sem.len_candidate = 0;
            chk_len_parm_expr(ast, stsk->dtype, 1);
          }
        }
        if (DTY(dtype) == TY_ARRAY) {
          int d;
          d = DTY(dtype + 1);
          if (DTY(d) == TY_DERIVED && DTY(d + 3) && DISTMEMG(DTY(d + 3))) {
            error(451, 3, gbl.lineno, SYMNAME(sptr), CNULL);
          }
        }
        /* link field-namelist into member list at this level */
        link_members(stsk, sptr);
        if (stype == ST_ARRAY && STSK_ENT(0).type != 'd' &&
            (AD_ASSUMSZ(ad) || AD_ADJARR(ad) || AD_DEFER(ad))) {
          error(50, 3, gbl.lineno, SYMNAME(sptr), CNULL);
        }
        if (stype == ST_ARRAY) {
          if (entity_attr.exist & (ET_B(ET_POINTER) | ET_B(ET_ALLOCATABLE))) {
            ALLOCP(sptr, 1);
          } else if (STSK_ENT(0).type == 'd') {
            /* error message wasn't issued above for derived type.
             * issue one now
             */
            if (AD_DEFER(ad)) {
              error(84, 3, gbl.lineno, SYMNAME(sptr),
                    "- deferred shape array must have the POINTER "
                    "or ALLOCATABLE attribute in a derived type");
              entity_attr.exist |= ET_B(ET_POINTER);
            } else if (AD_ASSUMSZ(ad) || AD_ADJARR(ad)) {
              if (AD_ADJARR(ad)) {
                int bndast, badArray;
                int numdim = AD_NUMDIM(ad);
                for (badArray = i = 0; i < numdim; i++) {
                  bndast = AD_LWAST(ad, i);
                  badArray = !chk_len_parm_expr(bndast, ENCLDTYPEG(sptr), 0);
                  if (!badArray) {
                    bndast = AD_UPAST(ad, i);
                    badArray = !chk_len_parm_expr(bndast, ENCLDTYPEG(sptr), 0);
                    if (!badArray) {
                      ADJARRP(sptr, 1);
                      USELENP(sptr, 1);
                      break;
                    }
                  }
                }
                if (badArray) {
                  for (badArray = i = 0; i < numdim; i++) {
                    bndast = AD_LWAST(ad, i);
                    badArray =
                        !chk_kind_parm_expr(bndast, ENCLDTYPEG(sptr), 1, 0);
                    if (badArray) {
                      badArray =
                          !chk_len_parm_expr(bndast, ENCLDTYPEG(sptr), 1);
                      if (!badArray) {
                        ADJARRP(sptr, 1);
                        USELENP(sptr, 1);
                        break;
                      }
                    }
                    if (badArray) {
                      goto illegal_array;
                    }
                    bndast = AD_UPAST(ad, i);
                    badArray =
                        !chk_kind_parm_expr(bndast, ENCLDTYPEG(sptr), 1, 0);
                    if (badArray) {
                      badArray =
                          !chk_len_parm_expr(bndast, ENCLDTYPEG(sptr), 1);
                      if (!badArray) {
                        ADJARRP(sptr, 1);
                        USELENP(sptr, 1);
                        break;
                      }
                    } else if (A_TYPEG(bndast) != A_ID &&
                               A_TYPEG(bndast) != A_CNST) {

                      ADJARRP(sptr, 1);
                      USELENP(sptr, 1);
                      if (chk_kind_parm_expr(bndast, ENCLDTYPEG(sptr), 1, 0)) {
                        USEKINDP(sptr, 1);
                      }
                      break;
                    }
                    if (badArray) {
                      goto illegal_array;
                    }
                  }
                }
              } else {
              illegal_array:
                error(84, 3, gbl.lineno, SYMNAME(sptr),
                      "- array must have constant bounds "
                      "in a derived type");
                entity_attr.exist |= ET_B(ET_POINTER);
              }
            }
          }
          if (DTY(dtype) == TY_ARRAY) {
            int d;
            d = DTY(dtype + 1);
            if (DTY(d) == TY_DERIVED && DTY(d + 3) && DISTMEMG(DTY(d + 3))) {
              error(451, 3, gbl.lineno, SYMNAME(sptr), CNULL);
            }
          }
        }
        if (DTY(sem.gdtype) == TY_DERIVED && (stsk->type == 'd')) {
          /* outer derived type has SEQUENCE, nested one should too */

          if (SEQG(DTY(stsk->dtype + 3)) && DCLDG(DTY(sem.gdtype + 3)) &&
              !SEQG(DTY(sem.gdtype + 3))) {
            error(155, 3, gbl.lineno,
                  "SEQUENCE must be set for nested derived type",
                  SYMNAME(DTY(sem.gdtype + 3)));
          }
          if (DTY(stsk->dtype + 3) == DTY(sem.gdtype + 3)) {
            if ((entity_attr.exist & ET_B(ET_POINTER)) == 0) {
              error(155, 3, gbl.lineno, "Derived type component must "
                                        "have the POINTER attribute -",
                    SYMNAME(sptr));
            }
          } else if ((entity_attr.exist & ET_B(ET_POINTER)) == 0 &&
                     !DCLDG(DTY(sem.gdtype + 3)))
            error(155, 4, gbl.lineno, "Derived type has not been declared -",
                  SYMNAME(DTY(sem.gdtype + 3)));
        }
      }

    } else {
      sptr = create_var(sptr);
      if (sem.kind_type_param) {
        USEKINDP(sptr, 1);
        KINDP(sptr, sem.kind_type_param);
      }
      if (sem.len_type_param) {
        USELENP(sptr, 1);
        LENP(sptr, sem.len_type_param);
      }
      if (DTY(sem.stag_dtype) == TY_DERIVED && sem.class) {
        /* TBD - Probably need to fix this condition when we
         * support unlimited polymorphic entities.
         */
        if (SCG(sptr) == SC_DUMMY ||
            entity_attr.exist & (ET_B(ET_POINTER) | ET_B(ET_ALLOCATABLE))) {
          CLASSP(sptr, 1); /* mark polymorphic variable */
          if (PASSBYVALG(sptr)) {
            error(155, 3, gbl.lineno, "Polymorphic variable cannot have VALUE"
                                      " attribute -",
                  SYMNAME(sptr));
          }
          if (DTY(sem.stag_dtype) == TY_DERIVED) {
            int tag = DTY(sem.stag_dtype + 3);
            if (CFUNCG(tag)) {
              error(155, 3, gbl.lineno,
                    "Polymorphic variable cannot be declared "
                    "with a BIND(C) derived type - ",
                    SYMNAME(sptr));
            }
            if (SEQG(tag)) {
              error(155, 3, gbl.lineno,
                    "Polymorphic variable cannot be declared "
                    "with a SEQUENCE derived type - ",
                    SYMNAME(sptr));
            }
          }

        } else {
          error(155, 3, gbl.lineno, "Polymorphic variable must be a pointer, "
                                    "allocatable, or dummy object - ",
                SYMNAME(sptr));
        }
      }
      if (DTY(sem.stag_dtype) == TY_DERIVED && sem.which_pass &&
          !(entity_attr.exist & (ET_B(ET_POINTER) | ET_B(ET_ALLOCATABLE))) &&
          SCG(sptr) != SC_DUMMY && !FVALG(sptr) &&
          (gbl.rutype != RU_PROG || CONSTRUCTSYMG(sptr))) {
        add_auto_finalize(sptr);
      }
      if (STYPEG(sptr) == ST_PROC && SCOPEG(sptr) &&
          SCOPEG(sptr) == stb.curr_scope && sem.which_pass &&
          gbl.rutype == RU_FUNC) {
        /* sptr is the ST_PROC for an ENTRY statement to appear later.
         * make a new sptr */
        sptr = insert_sym(sptr);
      }
      SST_SYMP(LHS, sptr);
      stype1 = STYPEG(sptr);
      /* Assertion:
       *  stype  = stype we want to make symbol {ARRAY or IDENT}
       *	stype1 = symbol's current stype
       */
      if (stype == ST_ARRAY) {
        if (IS_INTRINSIC(stype1)) {
          /* Changing intrinsic symbol to ARRAY */
          if ((sptr = newsym(sptr)) == 0)
            /* Symbol frozen as an intrinsic, ignore type decl */
            break;
          SST_SYMP(LHS, sptr);
          /* Cause STYPE and DTYPE to change AFTER fixing dtype */
          stype1 = ST_UNKNOWN;
        } else if (stype1 == ST_ENTRY) {
          if (DTY(DTYPEG(sptr)) == TY_ARRAY) {
            error(43, 3, gbl.lineno, "symbol", SYMNAME(sptr));
            break;
          }
        } else if (stype1 == ST_ARRAY) {
          /* if symbol is already an array, check if the dimension
           * specifiers are identical.
           */
          ADSC *ad1, *ad2;
          int ndim;

          ad1 = AD_DPTR(DTYPEG(sptr));
          /* dtype must be set */
          assert(dtypeset, "semant: dtype was not set", dtype, 3);
          ad2 = AD_DPTR(dtype);
          ndim = AD_NUMDIM(ad1);
          if (ndim != AD_NUMDIM(ad2)) {
            error(43, 3, gbl.lineno, "symbol", SYMNAME(sptr));
            break;
          }
          for (i = 0; i < ndim; i++)
            if (AD_LWBD(ad1, i) != AD_LWBD(ad2, i) ||
                AD_UPBD(ad1, i) != AD_UPBD(ad2, i))
              break;
          if (i < ndim) {
            error(43, 3, gbl.lineno, "symbol", SYMNAME(sptr));
            break;
          }
          error(119, 2, gbl.lineno, SYMNAME(sptr), CNULL);
        } else if (stype1 != ST_UNKNOWN && stype1 != ST_IDENT &&
                   stype1 != ST_VAR) {
          error(43, 3, gbl.lineno, "symbol", SYMNAME(sptr));
          break;
        }
        DTY(dtype + 1) = DTYPEG(sptr);
      } else if (IS_INTRINSIC(stype1) &&
                 (entity_attr.exist & ET_B(ET_INTRINSIC)) == 0) {
        /* Changing intrinsic symbol to IDENT in COMMON */
        if (IN_MODULE_SPEC || entity_attr.exist || sem.interface) {
          if ((sptr = newsym(sptr)) == 0)
            /* Symbol frozen as an intrinsic, ignore in COMMON */
            break;
          SST_SYMP(LHS, sptr);
          /* Cause STYPE and DTYPE to change AFTER fixing dtype */
          stype1 = ST_UNKNOWN;
          dtype = DTYPEG(sptr);
          dtypeset = 1;
        }
      }
      /*
       * The symbol's stype and data type can only be changed if
       * it is new or if the type is changing from an identifier or
       * structure to an array.  The latter can occur because of the
       * separation of type/record declarations from DIMENSION/COMMON
       * statements.  If the symbol is a record, its stype can change
       * only if it's an identifier; note, that its dtype will be
       * set (and checked) by the semantic actions for record.
       */
      if (stype1 == ST_UNKNOWN ||
          (stype == ST_ARRAY && (stype1 == ST_IDENT || stype1 == ST_VAR))) {
        if (in_entity_typdcl)
          STYPEP(sptr, ST_IDENT); /* stype will be filled in later*/
        /* ...else stype will be set by the actions for <combined> */

        if (!dtypeset)
          dtype = sem.gdtype;
        if (dtype > 0)
          DTYPEP(sptr, dtype);
        if (stype == ST_ARRAY) {
          if ((entity_attr.exist & ET_B(ET_POINTER)) || POINTERG(sptr)) {
            if (AD_ASSUMSHP(ad))
              error(196, 3, gbl.lineno, SYMNAME(sptr), CNULL);
            if (SCG(sptr) != SC_DUMMY)
              ALLOCP(sptr, 1);
          } else if (AD_ASSUMSZ(ad)) {
            if (SCG(sptr) != SC_NONE && SCG(sptr) != SC_DUMMY &&
                SCG(sptr) != SC_BASED)
              error(50, 3, gbl.lineno, SYMNAME(sptr), CNULL);
            ASUMSZP(sptr, 1);
            SEQP(sptr, 1);
          }
          if (AD_ADJARR(ad)) {
            ADJARRP(sptr, 1);
            if (SCG(sptr) != SC_NONE && SCG(sptr) != SC_DUMMY &&
                SCG(sptr) != SC_BASED)
              error(50, 3, gbl.lineno, SYMNAME(sptr), CNULL);
            else {
              /*
               * mark the adjustable array if the declaration
               * occurs after an ENTRY statement.
               */
              if (entry_seen)
                AFTENTP(sptr, 1);
            }
          } else if (!(entity_attr.exist &
                       (ET_B(ET_POINTER) | ET_B(ET_ALLOCATABLE))) &&
                     AD_DEFER(ad)) {
            if (SCG(sptr) == SC_CMBLK)
              error(43, 3, gbl.lineno, "deferred shape array", SYMNAME(sptr));
            if (SCG(sptr) == SC_DUMMY) {
              mk_assumed_shape(sptr);
              ASSUMSHPP(sptr, 1);
              if (sem.arrdim.assumedrank) {
                ASSUMRANKP(sptr, 1);
              }
              if (!XBIT(54, 2) && !(XBIT(58, 0x400000) && TARGETG(sptr)))
                SDSCS1P(sptr, 1);
            } else {
              if (AD_ASSUMSHP(ad)) {
                /* this is an error if it isn't a dummy; the
                 * declaration could occur before its entry, so
                 * the check needs to be performed in semfin.
                 */
                ASSUMSHPP(sptr, 1);
                if (!XBIT(54, 2) && !(XBIT(58, 0x400000) && TARGETG(sptr)))
                  SDSCS1P(sptr, 1);
              }
              ALLOCP(sptr, 1);
              mk_defer_shape(sptr);
            }
          }
        }
      } else if (stype == ST_ARRAY) {
        if (stype1 == ST_ENTRY) {
          if (FVALG(sptr)) {
#if DEBUG
            interr("semant1: trying to set data type of ST_ENTRY", sptr, 3);
#endif
            sptr = FVALG(sptr);
          } else {
            error(43, 3, gbl.lineno, "subprogram or entry", SYMNAME(sptr));
            sptr = insert_sym(sptr);
          }
        }
        if (RESULTG(sptr)) {
          assert(dtypeset, "semant: dtype was not set (2)", dtype, 3);
          DTYPEP(sptr, dtype);
          if ((entity_attr.exist & ET_B(ET_POINTER)) || POINTERG(sptr)) {
            if (!AD_DEFER(ad) || AD_ASSUMSHP(ad))
              error(196, 3, gbl.lineno, SYMNAME(sptr), CNULL);
          } else if (AD_ASSUMSZ(ad)) {
            ASUMSZP(sptr, 1);
            SEQP(sptr, 1);
          } else if (AD_ADJARR(ad))
            ADJARRP(sptr, 1);
          else if (AD_DEFER(ad)) {
            mk_assumed_shape(sptr);
            ASSUMSHPP(sptr, 1);
            if (!XBIT(54, 2) && !(XBIT(58, 0x400000) && TARGETG(sptr)))
              SDSCS1P(sptr, 1);
            AD_ASSUMSHP(ad) = 1;
          }
          copy_type_to_entry(sptr);
        }
      }
    }
    if (RESULTG(sptr) && STYPEG(sptr) != ST_ENTRY) {
      /* set the type for the entry point as well */
      copy_type_to_entry(sptr);
    }

    /* store gdtype, gty so that we can retrieve later to get
     * dtype for each declared variable, sem.gdtype an sem.gty
     * may get overwritten if variable is initialized with f2003
     * feature.
     */
    SST_GDTYPEP(RHS(1), sem.gdtype);
    SST_GTYP(RHS(1), sem.gty);

    /*
     * When declaring a variable's symbol, flang1 should store
     * the alignment from `!DIR$ ALIGN alignment` pragma to
     * the symbol.
     */
    PALIGNP(sptr, flg.x[251]);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<target list> ::= <target list> , <target> |
   */
  case TARGET_LIST1:
    break;
  /*
   *	<target list> ::= <target>
   */
  case TARGET_LIST2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<target> ::= <dcl id>
   */
  case TARGET1:
    TARGETP(SST_SYMG(RHS(1)), 1);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <interface> ::= <begininterface> |
   */
  case INTERFACE1:
    push_scope_level(sem.next_unnamed_scope++, SCOPE_INTERFACE);
    break;
  /*
   *	<interface> ::= <begininterface> <generic spec>
   */
  case INTERFACE2:
    push_scope_level(sem.next_unnamed_scope++, SCOPE_INTERFACE);
    if (sem.interf_base[sem.interface - 1].abstract) {
      error(155, 3, gbl.lineno, "A generic specifier cannot be present in an",
            "ABSTRACT INTERFACE");
    }
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <begininterface> ::= <pgm> INTERFACE |
   */
  case BEGININTERFACE1:
    i = 0;
    goto begininterf;
  /*
   *	<begininterface> ::= <pgm> ABSTRACT INTERFACE
   */
  case BEGININTERFACE2:
    i = 1;
  begininterf:
    if (IN_MODULE_SPEC && get_seen_contains()) {
      error(155, 3, gbl.lineno,
            "Interface-block may not appear in a"
            " module after the CONTAINS statement unless it is inside"
            " a module subprogram",
            CNULL);
    }
    NEED(sem.interface + 1, sem.interf_base, INTERF, sem.interf_size,
         sem.interf_size + 2);
    save_host(&sem.interf_base[sem.interface]);
    sem.interf_base[sem.interface].generic = 0;
    sem.interf_base[sem.interface].operator= 0;
    sem.interf_base[sem.interface].opval = 0;
    sem.interf_base[sem.interface].abstract = i;
    sem.interf_base[sem.interface].hpfdcl = sem.hpfdcl;
    sem.interface++;
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<generic spec> ::= <generic name> |
   */
  case GENERIC_SPEC1:
    if (scn.stmtyp != TK_ENDINTERFACE) {
      /* If we have a previously defined symbol with
       * same name as a generic type bound procedure, delay declaring
       * the generic type bound procedure until we process the entire
       * module (see queue_tbp() function, flag == 3 case for the
       * call to declsym).
       */
      int oldsptr;
      sptr = (int)SST_SYMG(RHS(1));
      oldsptr = sptr;
      if (STYPEG(sptr) == ST_TYPEDEF) {
        sptr = insert_sym(sptr); /* Overloaded type */
      }
      if (!sem.generic_tbp || !STYPEG(sptr) || SCOPEG(sptr) != stb.curr_scope) {
        if (STYPEG(sptr) == ST_PROC && VTOFFG(sptr) && !sem.generic_tbp) {
          /* Type bound procedure and generic interface can co-exist */
          sptr = insert_sym(sptr);
        } else if (STYPEG(sptr) && STYPEG(sptr) != ST_USERGENERIC) {
          sptr = insert_sym(sptr);
        } else if (STYPEG(sptr) == ST_USERGENERIC && IS_TBP(sptr)) {
          sptr = insert_sym(sptr);
        }
        sptr = declsym(sptr, ST_USERGENERIC, FALSE);
        if (STYPEG(oldsptr) != ST_TYPEDEF) {
          /* Check for the case where we overload the
           * type-name with a binding-name in a type bound procedure.
           */
          int oldsptr2 = oldsptr;
          for (; STYPEG(oldsptr2) == ST_ALIAS; oldsptr2 = SYMLKG(oldsptr2))
            ;
          if (STYPEG(oldsptr2) == ST_PROC && CLASSG(oldsptr2) &&
              VTOFFG(oldsptr2)) {
            oldsptr2 = findByNameStypeScope(SYMNAME(oldsptr2), ST_TYPEDEF,
                                            SCOPEG(oldsptr2));
          }
          if (STYPEG(oldsptr2) == ST_TYPEDEF)
            oldsptr = oldsptr2;
        }
        if (STYPEG(oldsptr) == ST_TYPEDEF) {
          GTYPEP(sptr, oldsptr); /* Store overloaded type */
        } else {
          /* Check for overloaded type in scope */
          oldsptr =
              findByNameStypeScope(SYMNAME(oldsptr), ST_TYPEDEF, SCOPEG(sptr));
          if (oldsptr)
            GTYPEP(sptr, oldsptr);
        }
      }
      if (SCOPEG(sptr) != stb.curr_scope) {
        int oldsptr = sptr;
        sptr = insert_sym(sptr);
        STYPEP(sptr, ST_USERGENERIC);
        SCOPEP(sptr, stb.curr_scope);
        copy_specifics(oldsptr, sptr);
        IGNOREP(oldsptr, TRUE);
      }
      EXPSTP(sptr, 1);
      sem.interf_base[sem.interface - 1].generic = sptr;
    }
    /*else
     * SST_SYMP(LHS, SST_SYMG(RHS(1)));
     */
    break;
  /*
   *	<generic spec> ::= OPERATOR ( <operator> )
   */
  case GENERIC_SPEC2:
    if (scn.stmtyp != TK_ENDINTERFACE) {
      sem.interf_base[sem.interface - 1].operator= SST_LSYMG(RHS(3));
      sem.interf_base[sem.interface - 1].opval = SST_OPTYPEG(RHS(3));
    } else {
      SST_SYMP(LHS, SST_LSYMG(RHS(3)));
    }
    break;
  /*
   *	<generic spec> ::= ASSIGNMENT ( = )
   */
  case GENERIC_SPEC3:
    if (scn.stmtyp != TK_ENDINTERFACE) {
      sptr = get_intrinsic_opr(OP_ST, 0);
      sem.interf_base[sem.interface - 1].operator= sptr;
      sem.interf_base[sem.interface - 1].opval = OP_ST;
    } else {
      sptr = get_intrinsic_oprsym(OP_ST, 0);
      SST_SYMP(LHS, sptr);
    }
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<generic name> ::= <ident> |
   */
  case GENERIC_NAME1:
    break;
  /*
   *	<generic name> ::= OPERATOR |
   */
  case GENERIC_NAME2:
    sptr = getsymbol("operator");
    SST_SYMP(LHS, sptr);
    break;
  /*
   *	<generic name> ::= ASSIGNMENT
   */
  case GENERIC_NAME3:
    sptr = getsymbol("assignment");
    SST_SYMP(LHS, sptr);
    break;

  /*
   *      <generic name> ::= <ident> ( <ident> )
   */
  case GENERIC_NAME4:
    i = sem.defined_io_type;
    if (strcmp(SYMNAME(SST_SYMG(RHS(1))), "read") == 0) {
      if (strcmp(SYMNAME(SST_SYMG(RHS(3))), "formatted") == 0) {
        sem.defined_io_type = 1;
      } else if (strcmp(SYMNAME(SST_SYMG(RHS(3))), "unformatted") == 0) {
        sem.defined_io_type = 2;
      } else {
        error(155, 3, gbl.lineno, "(FORMATTED) or (UNFORMATTED) "
                                  "must follow defined READ",
              CNULL);
        sem.defined_io_type = 0;
      }
    } else if (strcmp(SYMNAME(SST_SYMG(RHS(1))), "write") == 0) {
      if (strcmp(SYMNAME(SST_SYMG(RHS(3))), "formatted") == 0) {
        sem.defined_io_type = 3;
      } else if (strcmp(SYMNAME(SST_SYMG(RHS(3))), "unformatted") == 0) {
        sem.defined_io_type = 4;
      } else {
        error(155, 3, gbl.lineno, "(FORMATTED) or (UNFORMATTED) "
                                  "follow defined WRITE",
              CNULL);
        sem.defined_io_type = 0;
      }
    } else {
      error(155, 3, gbl.lineno, "Invalid generic specification -",
            SYMNAME(SST_SYMG(RHS(1))));
      sem.defined_io_type = 0;
    }
    if (i && sem.defined_io_type && i != sem.defined_io_type) {
      char *name_cpy;
      name_cpy = getitem(0,
                         strlen(SYMNAME(SST_SYMG(RHS(1)))) +
                             strlen(SYMNAME(SST_SYMG(RHS(3)))) + 1);
      sprintf(name_cpy, "%s(%s)", SYMNAME(SST_SYMG(RHS(1))),
              SYMNAME(SST_SYMG(RHS(3))));
      error(155, 3, gbl.lineno,
            "Generic name for INTERFACE statement "
            "does not match generic name for END INTERFACE ",
            name_cpy);
    } else if (!i && sem.defined_io_type) {
      sptr = getsymf(".%s", SYMNAME(SST_SYMG(RHS(1))));
      IGNOREP(sptr, TRUE);
      SST_SYMP(LHS, sptr);
    }
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<operator> ::= <intrinsic op> |
   */
  case OPERATOR1:
    if (scn.stmtyp != TK_ENDINTERFACE)
      sptr1 = get_intrinsic_opr(SST_OPTYPEG(RHS(1)), SST_IDG(RHS(1)));
    else
      sptr1 = get_intrinsic_oprsym(SST_OPTYPEG(RHS(1)), SST_IDG(RHS(1)));
    sptr = block_local_sym(sptr1);
    STYPEP(sptr, STYPEG(sptr1));
    SST_IDP(LHS, 1);
    SST_LSYMP(LHS, sptr);
    break;
  /*
   *	<operator> ::= . <ident> .
   */
  case OPERATOR2:
    sptr = block_local_sym(SST_SYMG(RHS(2)));
    STYPEP(sptr, STYPEG(SST_SYMG(RHS(2))));
    if (!sem.generic_tbp || !STYPEG(sptr) || SCOPEG(sptr) != stb.curr_scope) {
      if (STYPEG(sptr) == ST_PROC && VTOFFG(sptr) && !sem.generic_tbp) {
        /* Type bound procedure and generic operator can co-exist */
        sptr = insert_sym(sptr);
      }
      sptr = declsym(sptr, ST_OPERATOR, FALSE);
    }
    SST_IDP(LHS, 1);
    SST_LSYMP(LHS, sptr);
    if (scn.stmtyp == TK_INTERFACE) {
      const char *anm;
      anm = NULL;
      if (strcmp(SYMNAME(sptr), "x") == 0)
        anm = ".x.";
      else if (strcmp(SYMNAME(sptr), "xor") == 0)
        anm = ".xor.";
      else if (strcmp(SYMNAME(sptr), "o") == 0)
        anm = ".o.";
      else if (strcmp(SYMNAME(sptr), "n") == 0)
        anm = ".n.";
      if (anm) {
        error(155, 1, gbl.lineno,
              "Predefined intrinsic operator loses intrinsic property -", anm);
      }
    }
    break;
  /*
   *	<operator> ::= <defined op>
   */
  case OPERATOR3:
    sptr = block_local_sym(SST_SYMG(RHS(1)));
    STYPEP(sptr, STYPEG(SST_SYMG(RHS(1))));
    SST_IDP(LHS, 1);
    SST_LSYMP(LHS, sptr);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<intrinsic op> ::= <addop>   |
   */
  case INTRINSIC_OP1:
    SST_IDP(LHS, 0);
    SST_LSYMP(LHS, 3); /* unary and binary */
    break;
  /*
   *	<intrinsic op> ::= <mult op> |
   */
  case INTRINSIC_OP2:
    SST_IDP(LHS, 0);
    SST_LSYMP(LHS, 2); /* binary */
    break;
  /*
   *	<intrinsic op> ::= **        |
   */
  case INTRINSIC_OP3:
    SST_IDP(LHS, 0);
    SST_OPTYPEP(LHS, OP_XTOI);
    SST_LSYMP(LHS, 2); /* binary */
    break;
  /*
   *	<intrinsic op> ::= <n eqv op> |
   */
  case INTRINSIC_OP4:
    break;
  /*
   *	<intrinsic op> ::= .OR.      |
   */
  case INTRINSIC_OP5:
    SST_IDP(LHS, 0);
    SST_OPTYPEP(LHS, OP_LOR);
    SST_LSYMP(LHS, 2); /* binary */
    break;
  /*
   *    <intrinsic op> ::= .O.       |
   */
  case INTRINSIC_OP6:
    SST_IDP(LHS, TK_ORX);
    SST_OPTYPEP(LHS, OP_LOR);
    SST_LSYMP(LHS, 2); /* binary */
    break;
  /*
   *	<intrinsic op> ::= .AND.     |
   */
  case INTRINSIC_OP7:
    SST_IDP(LHS, 0);
    SST_OPTYPEP(LHS, OP_LAND);
    SST_LSYMP(LHS, 2); /* binary */
    break;
  /*
   *	<intrinsic op> ::= .NOT.     |
   */
  case INTRINSIC_OP8:
    SST_IDP(LHS, 0);
    SST_OPTYPEP(LHS, OP_LNOT);
    SST_LSYMP(LHS, 1); /* unary */
    break;
  /*
   *    <intrinsic op> ::= .N.       |
   */
  case INTRINSIC_OP9:
    SST_IDP(LHS, TK_NOTX);
    SST_OPTYPEP(LHS, OP_LNOT);
    SST_LSYMP(LHS, 1); /* unary */
    break;
  /*
   *	<intrinsic op> ::= <relop>   |
   */
  case INTRINSIC_OP10:
    SST_IDP(LHS, 0);
    SST_LSYMP(LHS, 2); /* binary */
    break;
  /*
   *	<intrinsic op> ::= '//'
   */
  case INTRINSIC_OP11:
    SST_IDP(LHS, 0);
    SST_OPTYPEP(LHS, OP_CAT);
    SST_LSYMP(LHS, 2); /* binary */
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<n eqv op> ::= .EQV. |
   */
  case N_EQV_OP1:
    SST_IDP(LHS, 0);
    SST_OPTYPEP(LHS, OP_LEQV);
    SST_LSYMP(LHS, 2); /* binary */
    break;
  /*
   *	<n eqv op> ::= .NEQV. |
   */
  case N_EQV_OP2:
    SST_IDP(LHS, 0);
    SST_OPTYPEP(LHS, OP_LNEQV);
    SST_LSYMP(LHS, 2); /* binary */
    break;
  /*
   *	<n eqv op> ::= .X. |
   */
  case N_EQV_OP3:
    SST_IDP(LHS, TK_XORX);
    SST_OPTYPEP(LHS, OP_LNEQV);
    SST_LSYMP(LHS, 2); /* binary */
    break;
  /*
   *	<n eqv op> ::= .XOR.
   */
  case N_EQV_OP4:
    SST_IDP(LHS, TK_XOR);
    SST_OPTYPEP(LHS, OP_LNEQV);
    SST_LSYMP(LHS, 2); /* binary */
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <end interface> ::= ENDINTERFACE |
   */
  case END_INTERFACE1:
    rhstop = 1;
    goto end_interface_shared;
  /*
   *	<end interface> ::= ENDINTERFACE <generic spec>
   */
  case END_INTERFACE2:
    rhstop = 2;
  end_interface_shared:
    if (sem.interface == 0) {
      error(302, 3, gbl.lineno, "INTERFACE", CNULL);
      SST_ASTP(LHS, 0);
      break;
    }
    if (gbl.currsub) {
      error(303, 2, gbl.lineno, SYMNAME(gbl.currsub), CNULL);
      pop_subprogram();
      pop_scope_level(SCOPE_NORMAL);
    }
    sem.interface--;
    restore_host(&sem.interf_base[sem.interface], FALSE);
    sptr = sem.interf_base[sem.interface].generic;
    if (sptr)
      check_generic(sptr);
    else if ((sptr = sem.interf_base[sem.interface].operator))
      check_generic(sptr);
    if (sem.scope_stack[sem.scope_level].kind == SCOPE_INTERFACE) {
      pop_scope_level(SCOPE_INTERFACE);
    }
    if (sptr && rhstop == 2 && !sem.defined_io_type) {
      sptr1 = SST_SYMG(RHS(2));
      if (strcmp(SYMNAME(sptr), SYMNAME(sptr1)))
        error(309, 3, gbl.lineno, SYMNAME(sptr1), CNULL);
    }
    sem.defined_io_type = 0;
    break;
  /*
   *	<module procedure stmt> ::= MODULE PROCEDURE <ident list> |
   *	                            MODULE PROCEDURE :: <ident list>
   */
  case MODULE_PROCEDURE_STMT1:
    rhstop = 3;
    goto module_procedure_stmt;
  case MODULE_PROCEDURE_STMT2:
    rhstop = 4;
module_procedure_stmt:
    if (IN_MODULE &&
        !sem.interface &&
        (itemp = SST_BEGG(RHS(rhstop))) != ITEM_END &&
        itemp->next == ITEM_END) {
      /* MODULE PROCEDURE <id> - begin separate module subprogram */
      sptr = itemp->t.sptr;
      
      /* C1548: checking MODULE prefix for subprograms that were
              declared as separate module procedures */
      if (!sem.interface && 
          !SEPARATEMPG(sptr) && !SEPARATEMPG(ref_ident(sptr))) {
        error(1056, ERR_Severe, gbl.lineno, NULL, NULL);  
        DCLDP(sptr, true);
      }
     
      gbl.currsub = instantiate_interface(sptr);
      sem.module_procedure = TRUE;
      gbl.rutype = FVALG(sptr) > NOSYM ? RU_FUNC : RU_SUBR;
      push_scope_level(sptr, SCOPE_NORMAL);
      push_scope_level(sptr, SCOPE_SUBPROGRAM);
      sem.pgphase = PHASE_HEADER;
      SST_ASTP(LHS, 0);
      break;
    }
    gnr = sem.interf_base[sem.interface - 1].generic;
    if (gnr == 0) {
      gnr = sem.interf_base[sem.interface - 1].operator;
      if (gnr == 0) {
        error(195, 3, gbl.lineno,
              "- MODULE PROCEDURE requires a generic INTERFACE", CNULL);
        break;
      }
    }
    count = 0;
    for (itemp = SST_BEGG(RHS(rhstop)); itemp != ITEM_END; itemp = itemp->next) {
      sptr = itemp->t.sptr;
      /* Temporarily open the interface scope. */
      sem.scope_stack[sem.scope_level].closed = FALSE;
      if (!IN_MODULE) {
        sptr = refsym(sptr, OC_OTHER);
        if (STYPEG(sptr) != ST_PROC)
          error(195, 3, gbl.lineno, "- Unable to access module procedure",
                CNULL);
        if (ENCLFUNCG(sptr) == 0 || STYPEG(ENCLFUNCG(sptr)) != ST_MODULE) {
          error(454, 3, gbl.lineno, SYMNAME(sptr), CNULL);
        }
      } else {
        if (STYPEG(sptr) == ST_PROC && !sem.which_pass && !INMODULEG(sptr)) {
          error(454, 3, gbl.lineno, SYMNAME(sptr), CNULL);
        }
        sptr = declsym(sptr, ST_MODPROC, FALSE);
        if (SYMLKG(sptr) == NOSYM)
          SYMLKP(sptr, 0);
        /* rescope modproc to 'module' scope */
        SCOPEP(sptr, sem.scope_stack[sem.scope_level - 1].sptr);
        i = add_symitem(gnr, SYMIG(sptr));
        SYMIP(sptr, i);
      }
      /* Reclose the interface scope. */
      sem.scope_stack[sem.scope_level].closed = TRUE;
      add_overload(gnr, sptr);
      if (STYPEG(SCOPEG(sptr)) == ST_MODULE) {
        /* make sure we include module name when generating
         * the symbol name.
         */
        INMODULEP(sptr, 1);
      }
            if (bind_attr.altname && (++count > 1))
                error(280, 2, gbl.lineno, "BIND: allowed only in module", 0);
        if (bind_attr.exist != -1) {
          process_bind(sptr);
        }
    }
    bind_attr.exist = -1;
    bind_attr.altname = 0;
    break;
  /*
   *      <procedure stmt> ::= PROCEDURE <ident list> |
   *                           PROCEDURE :: <ident list>
   */
  case PROCEDURE_STMT1:
    rhstop = 2;
    goto procedure_stmt;
  case PROCEDURE_STMT2:
    rhstop = 3;
procedure_stmt:
    if (sem.interface == 0) {
      error(155, 3, gbl.lineno, "PROCEDURE must appear in an INTERFACE", CNULL);
      break;
    }
    gnr = sem.interf_base[sem.interface - 1].generic;
    if (gnr == 0) {
      gnr = sem.interf_base[sem.interface - 1].operator;
      if (gnr == 0) {
        error(195, 3, gbl.lineno, "- PROCEDURE requires a generic INTERFACE",
              CNULL);
        break;
      }
    }
    count = 0;
    for (itemp = SST_BEGG(RHS(rhstop)); itemp != ITEM_END; itemp = itemp->next) {
      sptr = itemp->t.sptr;
      /* Temporarily open the interface scope. */
      sem.scope_stack[sem.scope_level].closed = FALSE;
      sptr = refsym(sptr, OC_OTHER);
      if (STYPEG(sptr) != ST_PROC) {
        if (STYPEG(sptr) == ST_USERGENERIC) {
          sptr = insert_sym(sptr);
        }
        sptr = declsym(sptr, ST_PROC, FALSE);
        if (SYMLKG(sptr) == NOSYM)
          SYMLKP(sptr, 0);
        /* rescope proc to 'host' scope */
        SCOPEP(sptr, sem.scope_stack[sem.scope_level - 1].sptr);
        i = add_symitem(gnr, SYMIG(sptr));
        SYMIP(sptr, i);
      }
      /* Reclose the interface scope. */
      sem.scope_stack[sem.scope_level].closed = TRUE;
      add_overload(gnr, sptr);
    }
    bind_attr.exist = -1;
    bind_attr.altname = 0;
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<use> ::= <get module> |
   */
  case USE1:
    add_use_stmt();
    break;
  /*
   *	<use> ::= <get module> , <rename list> |
   */
  case USE2:
    break;
  /*
   *	<use> ::= <get module> , <id name> : <only list> |
   */
  case USE3:
  /*  fall thru  */
  /*
   *      <use> ::= <get module> , <id name> :
   */
  case USE4:
    np = scn.id.name + SST_CVALG(RHS(3));
    if (sem_strcmp(np, "only") != 0)
      error(34, 3, gbl.lineno, np, CNULL);
    break;

  /* ------------------------------------------------------------------  */
  /*
   *
   *      <get module> ::= , <module nature> :: <id> |
   */
  case GET_MODULE2:

    sptr = SST_SYMG(RHS(4));

    /* Undo context sensitive scanner confusion.  This is a
       use statement, even though it contains a TK_INTRINSIC token
       This allows  us to move into PHASE_USE.
     */
    if ((scn.stmtyp == TK_INTRINSIC) || (scn.stmtyp == TK_NON_INTRINSIC))
      scn.stmtyp = TK_USE;

    /* check and enable ISO_C_BINDING INTRINSICS HERE? */
    if (SST_IDG(RHS(2))) {
/* use, intrinsic :: the only one we support is
   iso_c_binding
*/

    } else {
      if (strcmp(SYMNAME(sptr), "iso_c_binding") == 0)
        error(4, 3, gbl.lineno, "invalid non-intrinsic module", SYMNAME(sptr));
    }
    goto common_module;
    break;
  /*
   *      <get module> ::= :: <id>
   */
  case GET_MODULE3:
    sptr = SST_SYMG(RHS(2));
    goto common_module;
    break;
  /*
   *	<get module> ::= <id> |
   */
  case GET_MODULE1:
    sptr = SST_SYMG(RHS(1));
  common_module:
    sem.use_seen = 1;
    init_use_stmts();
    if (XBIT(68, 0x1)) {
      /* Append "_la" to the names of some modules. */
      static const char *names[] = {"ieee_exceptions", "ieee_arithmetic",
                                    "cudafor",         "openacc",
                                    "accel_lib",       NULL};
      int j;
      for (j = 0; names[j]; ++j) {
        if (strcmp(SYMNAME(sptr), names[j]) == 0) {
          sptr = getsymf("%s", SYMNAME(sptr));
          break;
        }
      }
    }
    if (IN_MODULE && strcmp(SYMNAME(sem.mod_sym), SYMNAME(sptr)) == 0) {
      error(4, 3, gbl.lineno, "MODULE cannot contain USE of itself -",
            SYMNAME(sptr));
      break;
    }
    if (sptr >= stb.firstusym && STYPEG(sptr) != ST_UNKNOWN &&
        STYPEG(sptr) != ST_MODULE) {
      int nsptr;
      /* see if this is really an error, or just an overloaded symbol */
      nsptr = sym_in_scope(sptr, stb.ovclass[ST_MODULE], NULL, NULL, 0);
      if (nsptr > 0 && (nsptr < stb.firstusym || STYPEG(nsptr) == ST_UNKNOWN ||
                        STYPEG(nsptr) == ST_MODULE)) {
        sptr = nsptr;
      } else {
        sptr = insert_sym(sptr);
      }
    }
    open_module(sptr);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <module nature> ::= INTRINSIC |
   */
  case MODULE_NATURE1:
    SST_IDP(LHS, 1);
    break;
  /*
   *      <module nature> ::= NON_INTRINSIC
   */
  case MODULE_NATURE2:
    SST_IDP(LHS, 0);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<rename list> ::= <rename list> , <rename> |
   */
  case RENAME_LIST1:
    break;
  /*
   *	<rename list> ::= <rename>
   */
  case RENAME_LIST2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<rename> ::= <ident> '=>' <ident> |
   */
  case RENAME1:
    add_use_stmt();
    sptr = sptr1 = SST_SYMG(RHS(3));
    if (test_scope(sptr) == -1) {
      // If symbol not in scope search for an in-scope symbol with same name.
      for (sptr1 = first_hash(sptr); sptr1 > NOSYM; sptr1 = HASHLKG(sptr1)) {
        if (sptr1 == sptr || NMPTRG(sptr) != NMPTRG(sptr1))
          continue;
        if (test_scope(sptr1) != -1) {
          sptr = sptr1;
          break; // Found it.
        }
      }
    }
    sptr = add_use_rename((int)SST_SYMG(RHS(1)), sptr, 0);
    SST_SYMP(RHS(3), sptr);
    break;
  /*
   *	<rename> ::= <id name> ( <rename operator> ) '=>' <id name> ( <rename
   *operator> )
   */
  case RENAME2:
    add_use_stmt();
    np = scn.id.name + SST_CVALG(RHS(1));
    if (sem_strcmp(np, "operator") == 0) {
      np = scn.id.name + SST_CVALG(RHS(6));
      if (sem_strcmp(np, "operator")) {
        error(34, 3, gbl.lineno, np, CNULL);
        break;
      }
    } else {
      error(34, 3, gbl.lineno, np, CNULL);
      break;
    }
    /* local (RHS(3)) => global (RHS(8)) */
    sptr = add_use_rename(SST_SYMG(RHS(3)), SST_SYMG(RHS(8)), 1);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<rename operator> ::= . <ident> .    |
   */
  case RENAME_OPERATOR1:
    SST_SYMP(LHS, SST_SYMG(RHS(2)));
    break;
  /*
   *	<rename operator> ::= <defined op>
   */
  case RENAME_OPERATOR2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<only list> ::= <only list> , <only> |
   */
  case ONLY_LIST1:
    break;
  /*
   *	<only list> ::= <only>
   */
  case ONLY_LIST2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<only> ::= <ident> |
   */
  case ONLY1:
    sptr = SST_SYMG(RHS(1));
    sptr = add_use_rename(0, sptr, 0);
    SST_SYMP(RHS(1), sptr);
    break;
  /*
   *	<only> ::= <ident> '=>' <ident> |
   */
  case ONLY2:
    sptr = SST_SYMG(RHS(3));
    sptr = add_use_rename((int)SST_SYMG(RHS(1)), sptr, 0);
    SST_SYMP(RHS(3), sptr);
    break;
  /*
   *	<only> ::= <id name> ( <only operator> ) |
   */
  case ONLY3:
    np = scn.id.name + SST_CVALG(RHS(1));
    if (sem_strcmp(np, "operator") == 0) {
      sptr = add_use_rename(0, SST_SYMG(RHS(3)), 1);
      SST_SYMP(RHS(3), sptr);
    } else
      error(34, 3, gbl.lineno, np, CNULL);
    break;
  /*
   *	<only> ::= <id name> ( = )
   */
  case ONLY4:
    np = scn.id.name + SST_CVALG(RHS(1));
    if (sem_strcmp(np, "assignment") == 0) {
      sptr = get_intrinsic_oprsym(OP_ST, 0);
      add_use_rename(0, sptr, 1);
    } else
      error(34, 3, gbl.lineno, np, CNULL);
    break;
  /*
   *   <only> ::= <id name> ( <only operator> ) '=>' <id name> ( <only operator
   *   > )
   */
  case ONLY5:
    np = scn.id.name + SST_CVALG(RHS(1));
    if (sem_strcmp(np, "operator") == 0) {
      np = scn.id.name + SST_CVALG(RHS(6));
      if (sem_strcmp(np, "operator")) {
        error(34, 3, gbl.lineno, np, CNULL);
        break;
      }
    } else {
      error(34, 3, gbl.lineno, np, CNULL);
      break;
    }
    sptr = SST_SYMG(RHS(3));
    if (STYPEG(sptr) == ST_OPERATOR && INKINDG(sptr)) {
      error(34, 3, gbl.lineno, SYMNAME(sptr), CNULL);
      break;
    }
    sptr = SST_SYMG(RHS(8));
    if (STYPEG(sptr) == ST_OPERATOR && INKINDG(sptr)) {
      error(34, 3, gbl.lineno, SYMNAME(sptr), CNULL);
      break;
    }
    /* local (RHS(3)) => global (RHS(8)) */
    (void)add_use_rename(SST_SYMG(RHS(3)), SST_SYMG(RHS(8)), 1);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<only operator> ::= <intrinsic op> |
   */
  case ONLY_OPERATOR1:
    sptr = get_intrinsic_oprsym(SST_OPTYPEG(RHS(1)), SST_IDG(RHS(1)));
    SST_SYMP(LHS, sptr);
    break;
  /*
   *	<only operator> ::= . <ident> .    |
   */
  case ONLY_OPERATOR2:
    SST_SYMP(LHS, SST_SYMG(RHS(2)));
    break;
  /*
   *	<only operator> ::= <defined op>
   */
  case ONLY_OPERATOR3:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<tp list> ::= <tp list> , <tp item> |
   */
  case TP_LIST1:
    rhstop = 3;
    goto add_tp_to_list;
  /*
   *	<tp list> ::= <tp item>
   */
  case TP_LIST2:
    rhstop = 1;
  add_tp_to_list:
    itemp = (ITEM *)getitem(0, sizeof(ITEM));
    itemp->next = ITEM_END;
    itemp->t.sptr = SST_SYMG(RHS(rhstop));
    if (rhstop == 1)
      /* adding first item to list */
      SST_BEGP(LHS, itemp);
    else
      /* adding subsequent items to list */
      SST_ENDG(RHS(1))->next = itemp;
    SST_ENDP(LHS, itemp);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<tp item> ::= <common> |
   */
  case TP_ITEM1:
    break;
  /*
   *	<tp item> ::= <ident>
   */
  case TP_ITEM2:
    sptr = refsym(SST_SYMG(RHS(1)), OC_OTHER);
    SST_SYMP(LHS, sptr);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<dec declaration> ::= ATTRIBUTES <msattr list> :: <cmn ident list> |
   */
  case DEC_DECLARATION1:
    for (itemp = SST_BEGG(RHS(4)); itemp != ITEM_END; itemp = itemp->next) {
      int da_bitv;
      sptr = itemp->t.sptr;
      if (sptr == 0)
        continue;
      if (STYPEG(sptr) != ST_CMBLK)
        sptr = refsym_inscope(sptr, OC_OTHER);
      da_type = 0;
      for (da_bitv = dec_attr.exist; da_bitv; da_bitv >>= 1, da_type++) {
        if ((da_bitv & 1) == 0)
          continue;
        switch (da_type) {
        case DA_ALIAS:

#if defined(TARGET_WIN)
          /* silently disallow ALIAS of winmain : it conflicts
             with our crt0.obj glue
           */
          if (strcmp(SYMNAME(sptr), "winmain") == 0)
            break;
#endif
          ALTNAMEP(sptr, dec_attr.altname);
          goto global_attrs;
        case DA_C:
          CFUNCP(sptr, 1);
          STDCALLP(sptr, 1); /* args must be passed by value */
          if (STYPEG(sptr) == ST_PROC || STYPEG(sptr) == ST_ENTRY) {
            MSCALLP(sptr, 0);
          }
          goto global_attrs;
        case DA_STDCALL:
          STDCALLP(sptr, 1);
#ifdef CREFP
          CREFP(sptr, 0);
          MSCALLP(sptr, 1);
#endif
          goto global_attrs;
        case DA_REFERENCE:
          if ((STYPEG(sptr) == ST_ENTRY) || (STYPEG(sptr) == ST_PROC))
            ss = sptr;
          else
            ss = gbl.currsub;
          PASSBYVALP(sptr, 0);
          PASSBYREFP(sptr, 1);
#ifdef CREFP
          if (CFUNCG(sptr)) {
            MSCALLP(sptr, 0);
            CREFP(sptr, 1);
          }
#endif
          goto global_attrs;

        case DA_VALUE:
          if ((STYPEG(sptr) == ST_ENTRY) || (STYPEG(sptr) == ST_PROC))
            ss = sptr;
          else
            ss = gbl.currsub;
          PASSBYVALP(sptr, 1);
          PASSBYREFP(sptr, 0);
          goto global_attrs;

        case DA_DLLEXPORT:
          if (IN_MODULE && sem.interface == 0 && STYPEG(sptr) != ST_CMBLK) {
            sem.mod_dllexport = TRUE;
            if (sptr == gbl.currmod)
              break;
          } else {
            DLLP(sptr, DLL_EXPORT);
          }
          goto global_attrs;
        case DA_DLLIMPORT:
          DLLP(sptr, DLL_IMPORT);
          goto global_attrs;
        case DA_DECORATE:
          DECORATEP(sptr, 1);
          goto global_attrs;
        case DA_NOMIXEDSLA:
#ifdef CREFP
          NOMIXEDSTRLENP(sptr, 1);
#endif
        /*  fall thru  */
        global_attrs:
          switch (STYPEG(sptr)) {
          case ST_CMBLK:
          case ST_ENTRY:
          case ST_PROC:
          case ST_UNKNOWN: /* allow undeclared identifiers */
            break;
          case ST_IDENT:
          case ST_VAR:
          case ST_ARRAY:
          case ST_STRUCT:
            if (da_type == DA_DLLEXPORT) {
              if (IN_MODULE && sem.interface == 0) {
                if ((SCG(sptr) == SC_CMBLK && !HCCSYMG(CMBLKG(sptr))) ||
                    SCOPEG(sptr) != gbl.currmod) {
                  error(84, 3, gbl.lineno, SYMNAME(sptr),
                        "- ATTRIBUTES items must be global");
                }
                break;
              }
            } else if ((da_type == DA_VALUE) || (da_type == DA_REFERENCE)) {
              break;
            }
            FLANG_FALLTHROUGH;
          default:
            error(84, 3, gbl.lineno, SYMNAME(sptr),
                  "- must be defined for ATTRIBUTES");
          }
          break;
        default:
          break;
        }
      }
    }
    dec_attr.exist = 0;
    break;
  /*
   *	<dec declaration> ::= ALIAS <ident> , <alt name>
   */
  case DEC_DECLARATION2:
  /*
   *	<dec declaration> ::= ALIAS <ident> : <alt name>
   */
  case DEC_DECLARATION3:
    sptr = refsym_inscope((int)SST_SYMG(RHS(2)), OC_OTHER);
    ALTNAMEP(sptr, SST_SYMG(RHS(4)));
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<msattr list> ::= <msattr list> , <msattr> |
   */
  case MSATTR_LIST1:
  /* fall thru */
  /*
   *	<msattr list> ::= <msattr>
   */
  case MSATTR_LIST2:
    if (da_type == -1)
      break;
    if (dec_attr.exist & DA_B(da_type))
      error(134, 3, gbl.lineno, "- duplicate", da[da_type].name);
    else if (dec_attr.exist & da[da_type].no)
      error(134, 3, gbl.lineno, "- conflict with", da[da_type].name);
    else
      dec_attr.exist |= DA_B(da_type);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<msattr> ::= <id name> |
   */
  case MSATTR1:
    da_type = -1;
    np = scn.id.name + SST_CVALG(RHS(1));
    if (strcmp(np, "alias") == 0) {
      error(155, 2, gbl.lineno, "Unrecognized directive: ATTRIBUTES", np);
    } else if (strcmp(np, "c") == 0)
      da_type = DA_C;
    else if (strcmp(np, "stdcall") == 0)
      da_type = DA_STDCALL;
    else if (sem_strcmp(np, "dllexport") == 0)
      da_type = DA_DLLEXPORT;
    else if (sem_strcmp(np, "dllimport") == 0)
      da_type = DA_DLLIMPORT;
    else if (sem_strcmp(np, "value") == 0)
      da_type = DA_VALUE;
    else if (sem_strcmp(np, "reference") == 0)
      da_type = DA_REFERENCE;
    else if (sem_strcmp(np, "decorate") == 0)
      da_type = DA_DECORATE;
    else if (sem_strcmp(np, "nomixed_str_len_arg") == 0)
      da_type = DA_NOMIXEDSLA;
    else
      error(155, 2, gbl.lineno, "Unrecognized directive: ATTRIBUTES", np);
    break;
  /*
   *	<msattr> ::= <id name> : <alt name>
   */
  case MSATTR2:
    da_type = -1;
    np = scn.id.name + SST_CVALG(RHS(1));
    if (strcmp(np, "alias") == 0) {
      da_type = DA_ALIAS;
      dec_attr.altname = SST_SYMG(RHS(3));
    } else
      error(155, 2, gbl.lineno, "Unrecognized directive: ATTRIBUTES", np);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<alt name> ::= <quoted string> |
   */
  case ALT_NAME1:
    break;
  /*
   *	<alt name> ::= <id name>
   */
  case ALT_NAME2:
    /* NEED TO UPCASE the name */
    for (np = scn.id.name + SST_CVALG(RHS(1)); (i = *np); np++) {
      if (i >= 'a' && i <= 'z')
        *np = i + ('A' - 'a');
    }
    np = scn.id.name + SST_CVALG(RHS(1));
    sptr = getstring(np, strlen(np));
    SST_SYMP(LHS, sptr);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<cmn ident list> ::= <cmn ident list> , <cmn ident> |
   */
  case CMN_IDENT_LIST1:
    rhstop = 3;
    goto add_cmn_to_list;
  /*
   *	<cmn ident list> ::= <cmn ident>
   */
  case CMN_IDENT_LIST2:
    rhstop = 1;
  add_cmn_to_list:
    itemp = (ITEM *)getitem(0, sizeof(ITEM));
    itemp->next = ITEM_END;
    itemp->t.sptr = SST_SYMG(RHS(rhstop));
    if (rhstop == 1)
      /* adding first item to list */
      SST_BEGP(LHS, itemp);
    else
      /* adding subsequent items to list */
      SST_ENDG(RHS(1))->next = itemp;
    SST_ENDP(LHS, itemp);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<cmn ident> ::= <common> |
   */
  case CMN_IDENT1:
    sptr = SST_SYMG(RHS(1));
    if (sem.which_pass && CMEMFG(sptr) == 0)
      error(38, 3, gbl.lineno, SYMNAME(sptr), CNULL);
    break;
  /*
   *	<cmn ident> ::= <ident>
   */
  case CMN_IDENT2:
    sptr = SST_SYMG(RHS(1));
    if (STYPEG(sptr) == ST_CMBLK) {
      sptr = refsym(sptr, OC_OTHER);
      SST_SYMP(LHS, sptr);
    }
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<pragma declaration> ::= <nis> LOCAL ( <ident list> ) |
   */
  case PRAGMA_DECLARATION1:
    break;
  /*
   *	<pragma declaration> ::= <nis> <ignore tkr> |
   */
  case PRAGMA_DECLARATION2:
    if (!sem.interface && !(IN_MODULE && gbl.currsub)) {
      error(155, 3, gbl.lineno,
            "IGNORE_TKR can only appear in an interface body"
            " or a module procedure",
            CNULL);
    }
    break;
  /*
   *	<pragma declaration> ::= <nis> DEFAULTKIND <dflt> |
   */
  case PRAGMA_DECLARATION3:
    break;
  /*
   *	<pragma declaration> ::= <nis> MOVEDESC <id name>
   */
  case PRAGMA_DECLARATION4:
#if defined(MVDESCP)
    np = scn.id.name + SST_CVALG(RHS(3));
    if (gbl.currsub && sem_strcmp(np, SYMNAME(gbl.currsub)) == 0) {
      MVDESCP(gbl.currsub, 1);
    }
#endif
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<ignore tkr> ::= IGNORE_TKR |
   */
  case IGNORE_TKR1:
    if (sem.interface || (IN_MODULE && gbl.currsub)) {
      /* must be in interface -- if not, an error will be reported* later */
      count = PARAMCTG(gbl.currsub);
      i = DPDSCG(gbl.currsub);
      while (count--) {
        sptr = *(aux.dpdsc_base + i + count);
        /* IGNORE_TKR_ALL includes all of the IGNORE_ values plus
         * an indicater except for IGNORE_C
         */
        IGNORE_TKRP(sptr, IGNORE_TKR_ALL);
      }
    }
    break;
  /*
   *	<ignore tkr> ::= IGNORE_TKR <tkr id list>
   */
  case IGNORE_TKR2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<tkr id list> ::= <tkr id list> , <tkr id> |
   */
  case TKR_ID_LIST1:
    break;
  /*
   *	<tkr id list> ::= <tkr id>
   */
  case TKR_ID_LIST2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<tkr id> ::= <tkr spec> <ident>
   */
  case TKR_ID1:
    sptr = refsym(SST_SYMG(RHS(2)), OC_OTHER);
    if (sem.interface || (IN_MODULE && gbl.currsub)) {
      /* must be in interface -- if not, an error will be reported* later */
      if (SCG(sptr) == SC_DUMMY)
        IGNORE_TKRP(sptr, IGNORE_TKRG(sptr) | SST_CVALG(RHS(1)));
      else
        error(134, 3, gbl.lineno,
              "- IGNORE_TKR specified for nondummy argument", SYMNAME(sptr));
    }
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<tkr spec> ::= |
   */
  case TKR_SPEC1:
    /*  NOT IGNORE_C  */
    SST_CVALP(LHS, IGNORE_T | IGNORE_K | IGNORE_R | IGNORE_D | IGNORE_M);
    break;
  /*
   *	<tkr spec> ::= ( <id name> )
   */
  case TKR_SPEC2:
    np = scn.id.name + SST_CVALG(RHS(2));
    conval = 0;
    count = strlen(np);
    for (i = 0; i < count; i++) {
      switch (np[i]) {
      case 't':
      case 'T':
        conval |= IGNORE_T;
        break;
      case 'k':
      case 'K':
        conval |= IGNORE_K;
        break;
      case 'r':
      case 'R':
        conval |= IGNORE_R;
        break;
      case 'a':
      case 'A':
        conval |= IGNORE_TKR_ALL;
        break;
      case 'd':
      case 'D':
        conval |= IGNORE_D;
        break;
      case 'm':
      case 'M':
        conval |= IGNORE_M;
        break;
      case 'c':
      case 'C':
        conval |= IGNORE_C;
        break;
      default:
        error(155, 3, gbl.lineno, "Illegal IGNORE_TKR specifier", CNULL);
        conval = 0;
        goto end_tkr_spec;
      }
    }
  end_tkr_spec:
    SST_CVALP(LHS, conval);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<dflt> ::= |
   */
  case DFLT1:
#ifdef DFLTP
    if (gbl.currsub) {
      DFLTP(gbl.currsub, 1);
    }
#endif
    break;
  /*
   *	<dflt> ::= ( <ident list> )
   */
  case DFLT2:
#ifdef DFLTP
    for (itemp = SST_BEGG(RHS(2)); itemp != ITEM_END; itemp = itemp->next) {
      sptr = getocsym(itemp->t.sptr, OC_OTHER, FALSE);
      if (STYPEG(sptr) == ST_ENTRY || STYPEG(sptr) == ST_PROC) {
        DFLTP(sptr, 1);
      }
    }
#endif
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<import> ::= IMPORT |
   */
  case IMPORT1:
    if (!sem.interface) {
      error(155, 3, gbl.lineno, "IMPORT can only appear in an interface body",
            CNULL);
    } else {
      sem.seen_import = TRUE;
    }
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<opt import> ::= |
   */
  case OPT_IMPORT1:
    if (sem.interface) {
      /*
       * The current context is:
       * interface
       *    ...
       *    subroutine/function  ...
       *       IMPORT
       *    end subroutine/function
       *    ...
       * end interface
       *
       * There should be three scope entries corresponding to this
       * context:
       *
       * scope_level-2 : SCOPE_INTERFACE
       * scope_level-1 : SCOPE_NORMAL
       * scope_level   : SCOPE_SUBPROGRAM
       *
       * For IMPORT without a list, open the SCOPE_NORMAL to make host
       * symbols visible.
       */
      for (i = sem.scope_level - 1; i >= 4; i--) {
        if (sem.scope_stack[i].kind == SCOPE_NORMAL) {
          sem.scope_stack[i].closed = FALSE;
          break;
        }
      }
    }
    break;
  /*
   *	<opt import> ::= <opt attr> <import name list>
   */
  case OPT_IMPORT2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<import name list> ::= <import name list> , <import name> |
   */
  case IMPORT_NAME_LIST1:
    break;
  /*
   *	<import name list> ::= <import name>
   */
  case IMPORT_NAME_LIST2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<import name> ::= <ident>
   */
  case IMPORT_NAME1:
    if (sem.interface) {
      /*
       * The current context is:
       * interface
       *    ...
       *    subroutine/function  ...
       *        IMPORT xxxx
       *    end subroutine/function
       *    ...
       * end interface
       *
       * There should be three scope entries corresponding to this
       * context:
       *
       * scope_level-2 : SCOPE_INTERFACE
       * scope_level-1 : SCOPE_NORMAL
       * scope_level   : SCOPE_SUBPROGRAM
       *
       * add the host-associcated symbols to the import list of
       * the SCOPE_NORMAL entry.
       */
      sem_import_sym(SST_SYMG(RHS(1)));
    }
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<procedure declaration> ::= <procedure> <opt attr> <proc dcl list>
   */
  case PROCEDURE_DECLARATION1:
    entity_attr.exist = 0;
    bind_attr.exist = -1;
    bind_attr.altname = 0;
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<procedure> ::= PROCEDURE ( <proc interf> ) <opt proc attr>
   */
  case PROCEDURE1:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<proc interf> ::= |
   */
  case PROC_INTERF1:
    sem.gdtype = -1;
    proc_interf_sptr = 0;
    break;
  /*
   *	<proc interf> ::= <id> |
   */
  case PROC_INTERF2:
    proc_interf_sptr = resolve_sym_aliases(SST_SYMG(RHS(1)));
    break;
  /*
   *	<proc interf> ::= <data type>
   */
  case PROC_INTERF3:
    proc_interf_sptr = 0;
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<opt proc attr> ::= |
   */
  case OPT_PROC_ATTR1:
    break;
  /*
   *	<opt proc attr> ::= , <proc attr list>
   */
  case OPT_PROC_ATTR2:
    if ((entity_attr.exist & ET_B(ET_PROTECTED)) &&
        !(entity_attr.exist & ET_B(ET_POINTER)))
      error(134, 3, gbl.lineno, et[ET_PROTECTED].name, "for procedure");
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<proc attr list> ::= <proc attr list> , <proc attr> |
   */
  case PROC_ATTR_LIST1:
  /*
   *	<proc attr list> ::= <proc attr>
   */
  case PROC_ATTR_LIST2:
    if (entity_attr.exist & ET_B(et_type))
      error(134, 3, gbl.lineno, "- duplicate", et[et_type].name);
    if (INSIDE_STRUCT && (STSK_ENT(0).type == 'd')) {
      if (ET_B(et_type) &
          ~(ET_B(ET_POINTER) | ET_B(ET_PASS) | ET_B(ET_NOPASS) |
            ET_B(ET_ACCESS))) {
        error(134, 3, gbl.lineno, et[et_type].name, "for procedure component");
      } else
        entity_attr.exist |= ET_B(et_type);
    } else {
      if (ET_B(et_type) &
          ~(ET_B(ET_ACCESS) | ET_B(ET_BIND) | ET_B(ET_INTENT) |
            ET_B(ET_OPTIONAL) | ET_B(ET_POINTER) | ET_B(ET_SAVE) |
            ET_B(ET_PROTECTED)))
        error(134, 3, gbl.lineno, et[et_type].name, "for procedure");
      else
        entity_attr.exist |= ET_B(et_type);
    }
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<proc attr> ::= <access spec> |
   */
  case PROC_ATTR1:
    et_type = ET_ACCESS;
    break;
  /*
   *	<proc attr> ::= BIND <bind attr> |
   */
  case PROC_ATTR2:
    et_type = ET_BIND;
    break;
  /*
   *	<proc attr> ::= <intent> |
   */
  case PROC_ATTR3:
    et_type = ET_INTENT;
    break;
  /*
   *	<proc attr> ::= OPTIONAL |
   */
  case PROC_ATTR4:
    et_type = ET_OPTIONAL;
    break;
  /*
   *	<proc attr> ::= POINTER |
   */
  case PROC_ATTR5:
    et_type = ET_POINTER;
    break;
  /*
   *	<proc attr> ::= SAVE |
   */
  case PROC_ATTR6:
    et_type = ET_SAVE;
    break;
  /*
   *	<proc attr> ::= PASS |
   */
  case PROC_ATTR7:
    et_type = ET_PASS;
    entity_attr.pass_arg = 0; /* PASS without argname */
    break;
  /*
   *	<proc attr> ::= PASS ( <ident> ) |
   */
  case PROC_ATTR8:
    et_type = ET_PASS;
    entity_attr.pass_arg = SST_SYMG(RHS(3)); /* PASS with argname */
    break;
  /*
   *	<proc attr> ::= NOPASS |
   */
  case PROC_ATTR9:
    et_type = ET_NOPASS;
    break;
  /*
   *	<proc attr> ::= PROTECTED
   */
  case PROC_ATTR10:
    et_type = ET_PROTECTED;
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<proc dcl list> ::= <proc dcl list> , <proc dcl> |
   */
  case PROC_DCL_LIST1:
    break;
  /*
   *	<proc dcl list> ::= <proc dcl>
   */
  case PROC_DCL_LIST2:
    break;
  /*
   *    <proc dcl> ::= <ident> '=>' <id>
   */
  case PROC_DCL3:
    sptr = SST_SYMG(RHS(3));
    sem.proc_initializer = true;
    goto proc_dcl_init;


  /* ------------------------------------------------------------------ */
  /*
   *	<proc dcl> ::= <ident> |
   */
  case PROC_DCL1:
    inited = FALSE;
    goto proc_dcl_shared;
  /*
   *	<proc dcl> ::= <ident> '=>' <id> ( )
   */
  case PROC_DCL2:
    sptr = SST_SYMG(RHS(3));
    if (sptr <= NOSYM || strcmp(SYMNAME(sptr),"null") != 0) {
      errsev(87);
    }
proc_dcl_init:
    sptr = refsym(sptr, OC_OTHER);
    SST_SYMP(RHS(3), sptr);
    SST_IDP(RHS(3), S_IDENT);
    sem.dinit_data = TRUE;
    (void)mkvarref(RHS(3), ITEM_END);
    sem.dinit_data = FALSE;
    inited = TRUE;

  proc_dcl_shared:
    sptr = SST_SYMG(RHS(1));
    {
      /* Hide, so we can modify attribute list without exposing it */
      int attr = entity_attr.exist;
      if (!POINTERG(sptr) && !(attr & ET_B(ET_POINTER)) &&
          proc_interf_sptr > NOSYM && SCG(sptr) != SC_DUMMY) {
        /* Check to see if we have a dummy argument with a name that overloads 
         * another symbol name (such as a procedure name).
         */
        SPTR sym;
        get_next_hash_link(sptr, 0);
        while ((sym = get_next_hash_link(sptr, 2)) > NOSYM) {
          if (!POINTERG(sym) && SCG(sym) == SC_DUMMY && 
              SCOPEG(sym) == stb.curr_scope) {
            sptr = sym;
            break;
          }
        }
      }
      if (!POINTERG(sptr) && !(attr & ET_B(ET_POINTER)) &&
          proc_interf_sptr > NOSYM && SCG(sptr) == SC_DUMMY) {
        IS_PROC_DUMMYP(sptr, 1);
      }
      if (POINTERG(sptr)) {
        attr |= ET_B(ET_POINTER);
      } 
    if (!IS_PROC_DUMMYG(sptr) && IS_INTERFACEG(proc_interf_sptr) &&
        !IS_PROC_PTR_IFACEG(proc_interf_sptr)) {
      /* Create a unique symbol for the interface so it does not conflict with
       * an external procedure symbol. For non-procedure dummy arguments,
       * we need a unique symbol for the interface in order to preserve
       * the interface flag (IS_PROC_PTR_IFACE). We need the interface flag in 
       * the back-end so we properly generate the procedure descriptor
       * actual arguments on the call-site (when we call the procedure pointer).
       * This is only needed  by the LLVM back-end because the bridge uses the 
       * interface to generate the LLVM IR for the actual arguments. 
       */
      char * buf;
      int len;
      SPTR sym;
    
      /* First, let's see if we aleady have a unique interface symbol */ 
      len = strlen(SYMNAME(proc_interf_sptr)) + strlen("iface") + 1;
      buf = getitem(0, len);
      sprintf(buf,"%s$iface",SYMNAME(proc_interf_sptr));
      sym = findByNameStypeScope(buf, ST_PROC, 0);
      if (sym > NOSYM && !cmp_interfaces_strict(sym, proc_interf_sptr, 0)) { 
        /* The interface is not compatible. We will now try to find one that
         * is compatible in the symbol table.
         */
        SPTR sym2 = sym;
        get_next_hash_link(sym2, 0);
        while ((sym2=get_next_hash_link(sym2, 1)) > NOSYM) {
          if (cmp_interfaces_strict(sym2, proc_interf_sptr, 0)) {
            break;
          }
        }
        sym = sym2;
      }
      if (sym <= NOSYM) {  
        /* We don't yet have a unique interface symbol, so create it now */
        sym  = get_next_sym(SYMNAME(proc_interf_sptr), "iface");
        /* Propagate flags from the original symbol to the new symbol */
        copy_sym_flags(sym, proc_interf_sptr);
        HCCSYMP(sym, 1);
        IS_PROC_PTR_IFACEP(sym, 1);
      }
      proc_interf_sptr = sym;
    }
      sptr = decl_procedure_sym(sptr, proc_interf_sptr, attr);
      sptr =
          setup_procedure_sym(sptr, proc_interf_sptr, attr, entity_attr.access);
    }

    /* Error while creating proc symbol */
    if (sptr == 0)
      break;

    SST_SYMP(RHS(1), sptr);

    stype = STYPEG(sptr);

    if (inited) { /* check if symbol is data initialized */
      if (stype == ST_PROC) {
        error(114, 3, gbl.lineno, SYMNAME(SST_SYMG(RHS(1))), CNULL);
        goto proc_decl_end;
      }
      if (INSIDE_STRUCT && (STSK_ENT(0).type == 'd')) {
        get_static_descriptor(sptr);
        get_all_descriptors(sptr);
        SCP(sptr, SC_BASED);
        ast = SST_ASTG(RHS(3));
        if (A_TYPEG(ast) == A_FUNC) {
          gen_unique_func_ast(ast, sptr, RHS(3)); 
        }
        construct_acl_for_sst(RHS(3), DTYPEG(SST_SYMG(RHS(1))));
        if (!SST_ACLG(RHS(3))) {
          goto proc_decl_end;
        }

        ict = SST_ACLG(RHS(3));
        ict->sptr = sptr; /* field/component sptr */
        save_struct_init(ict);
        stsk = &STSK_ENT(0);
        if (stsk->ict_beg) {
          (stsk->ict_end)->next = SST_ACLG(RHS(3));
          stsk->ict_end = SST_ACLG(RHS(3));
        } else {
          stsk->ict_beg = SST_ACLG(RHS(3));
          stsk->ict_end = SST_ACLG(RHS(3));
        }
      } else {
        /* Data item (not TYPE component) initialization */
        /* have
         *   ... :: <ptr> => NULL()
         * <ptr>$p, <ptr>$o, <ptr>$sd  will be needed */
        get_static_descriptor(sptr);
        get_all_descriptors(sptr);
        ast = SST_ASTG(RHS(3));
        if (A_TYPEG(ast) == A_FUNC) {
          gen_unique_func_ast(ast, sptr, RHS(3)); 
        }
        construct_acl_for_sst(RHS(3), DTYPEG(SST_SYMG(RHS(1))));
        if (!SST_ACLG(RHS(3))) {
          goto proc_decl_end;
        }
        ast = mk_id(sptr);
        SST_ASTP(RHS(1), ast);
        SST_DTYPEP(RHS(1), DTYPEG(SST_SYMG(RHS(1))));
        SST_SHAPEP(RHS(1), 0);
        ivl = dinit_varref(RHS(1));

        dinit(ivl, SST_ACLG(RHS(3)));
      } 
    } else if (POINTERG(sptr)) {
        get_static_descriptor(sptr);
        get_all_descriptors(sptr);
    }

  proc_decl_end:

    if (STYPEG(sptr) != ST_ENTRY && STYPEG(sptr) != ST_MEMBER &&
        RESULTG(sptr)) {
      /* set the type for the entry point as well */
      copy_type_to_entry(sptr);
    }
    sem.dinit_error = FALSE;

    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<type bound procedure> ::= <tprocedure> <opt attr> <binding name list>
   */
  case TYPE_BOUND_PROCEDURE1:
    dtype = /*sem.stag_dtype*/ stsk->dtype;
    if (SST_FIRSTG(RHS(1)) & 0x2) { /* nopass */
      queue_tbp(0, SST_SYMG(RHS(3)), 0, dtype, TBP_NOPASS);
    }
    if (SST_FIRSTG(RHS(1)) & 0x4) { /* non_overridable */
      queue_tbp(0, SST_SYMG(RHS(3)), 0, dtype, TBP_NONOVERRIDABLE);
    }
    if (SST_FIRSTG(RHS(1)) & 0x8) { /* deferred */
      if (!ABSTRACTG(DTY(dtype + 3))) {
        error(155, 3, gbl.lineno,
              "Specifying a deferred type bound procedure in "
              "non-abstract type",
              SYMNAME(DTY(dtype + 3)));
      }
      if (!sem.tbp_interface) {
        error(155, 3, gbl.lineno,
              "Specifying a deferred type bound procedure without"
              " an interface-name in",
              SYMNAME(DTY(dtype + 3)));
      }
      queue_tbp(sem.tbp_interface, SST_SYMG(RHS(3)), 0, dtype, TBP_DEFERRED);
    }
    if (SST_FIRSTG(RHS(1)) & 0x10) { /* private */
      queue_tbp(0, SST_SYMG(RHS(3)), 0, dtype, TBP_PRIVATE);
    } else if (SST_FIRSTG(RHS(1)) & 0x20) { /* public */
      queue_tbp(0, SST_SYMG(RHS(3)), 0, dtype, TBP_PUBLIC);
    }
    if (SST_FIRSTG(RHS(1)) & 0x1) {
      sptr = SST_LSYMG(RHS(1));
      if (sptr) { /* pass */
        sptr = getsym(LOCAL_SYMNAME(sptr), strlen(SYMNAME(sptr)));
        if (STYPEG(sptr) != ST_IDENT || DTYPEG(sptr) != dtype) {
          sptr = insert_sym(sptr);
          sptr = declsym(sptr, ST_IDENT, TRUE);
          DTYPEP(sptr, dtype);
          SCP(sptr, SC_DUMMY);
          IGNOREP(sptr, TRUE);
        }
        queue_tbp(sptr, SST_SYMG(RHS(3)), 0, dtype, TBP_PASS);
      }
    }
    sem.tbp_interface = 0;
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<tprocedure> ::= TPROCEDURE <opt interface name> <opt binding attr list>
   */
  case TPROCEDURE1:
    SST_FIRSTP(LHS, SST_FIRSTG(RHS(3)));
    if (SST_FIRSTG(RHS(3)) & 0x1)
      SST_LSYMP(LHS, SST_LSYMG(RHS(3)));
    SST_ASTP(LHS, 0);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<opt interface name> ::= |
   */
  case OPT_INTERFACE_NAME1:
    break;
  /*
   *	<opt interface name> ::= ( <id> )
   */
  case OPT_INTERFACE_NAME2:
    sem.tbp_interface = SST_SYMG(RHS(2));
    dtype = /*sem.stag_dtype*/ stsk->dtype;
    queue_tbp(SST_SYMG(RHS(2)), 0, 0, dtype, TBP_ADD_INTERFACE);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<opt binding attr list> ::= |
   */
  case OPT_BINDING_ATTR_LIST1:
    SST_FIRSTP(LHS, 0);
    SST_LSYMP(LHS, 0);
    break;
  /*
   *	<opt binding attr list> ::= , <binding attr list>
   */
  case OPT_BINDING_ATTR_LIST2:
    SST_FIRSTP(LHS, SST_FIRSTG(RHS(2)));
    if (SST_FIRSTG(RHS(2)) & 0x1) {
      SST_LSYMP(LHS, SST_LSYMG(RHS(2)));
    }
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<binding attr list> ::= <binding attr list> , <binding attr> |
   */
  case BINDING_ATTR_LIST1:
    switch (SST_FIRSTG(RHS(1)) & SST_FIRSTG(RHS(3))) {
    case 0x1:
      error(134, 3, gbl.lineno, "- duplicate", "PASS");
      break;
    case 0x2:
      error(134, 3, gbl.lineno, "- duplicate", "NOPASS");
      break;
    case 0x4:
      error(134, 3, gbl.lineno, "- duplicate", "NON_OVERRIDABLE");
      break;
    case 0x8:
      error(134, 3, gbl.lineno, "- duplicate", "DEFERRED");
      break;
    case 0x10:
      error(134, 3, gbl.lineno, "- duplicate", "PRIVATE");
      break;
    case 0x20:
      error(134, 3, gbl.lineno, "- duplicate", "PUBLIC");
      break;
    }

    if (((SST_FIRSTG(RHS(1)) | SST_FIRSTG(RHS(3))) & 0x1) &&
        ((SST_FIRSTG(RHS(1)) | SST_FIRSTG(RHS(3))) & 0x2)) {

      error(155, 3, gbl.lineno, "PASS and NOPASS may not appear "
                                "in same type bound procedure",
            CNULL);
    } else if (((SST_FIRSTG(RHS(1)) | SST_FIRSTG(RHS(3))) & 0x4) &&
               ((SST_FIRSTG(RHS(1)) | SST_FIRSTG(RHS(3))) & 0x8)) {
      error(155, 3, gbl.lineno, "DEFERRED and NON_OVERRIDABLE "
                                "may not appear in same type bound procedure",
            CNULL);
    } else if (((SST_FIRSTG(RHS(1)) | SST_FIRSTG(RHS(3))) & 0x10) &&
               ((SST_FIRSTG(RHS(1)) | SST_FIRSTG(RHS(3))) & 0x20)) {
      error(155, 3, gbl.lineno, "PRIVATE and PUBLIC "
                                "may not appear in same type bound procedure",
            CNULL);
    }

    SST_FIRSTP(LHS, SST_FIRSTG(RHS(1)) | SST_FIRSTG(RHS(3)));

    if (SST_FIRSTG(RHS(3)) & 0x1 && SST_LSYMG(RHS(3)))
      SST_LSYMP(RHS(1), SST_LSYMG(RHS(3)));
    FLANG_FALLTHROUGH;
  /*
   *	<binding attr list> ::= <binding attr>
   */
  case BINDING_ATTR_LIST2:
    if (SST_FIRSTG(RHS(1)) & 0x1)
      SST_LSYMP(LHS, SST_LSYMG(RHS(1)));
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<binding attr> ::= <id name> |
   */
  case BINDING_ATTR1:
    /*
     * Not using keywords to enumerate the attributes; <id name> may be:
     * PASS NOPASS NON_OVERRIDABLE DEFERRED PRIVATE PUBLIC
     */
    SST_LSYMP(LHS, 0);
    np = scn.id.name + SST_CVALG(RHS(1));
    if (sem_strcmp(np, "pass") == 0) {
      SST_FIRSTP(LHS, 0x1);
    } else if (sem_strcmp(np, "nopass") == 0) {
      SST_FIRSTP(LHS, 0x2);
    } else if (sem_strcmp(np, "non_overridable") == 0) {
      SST_FIRSTP(LHS, 0x4);
    } else if (sem_strcmp(np, "deferred") == 0) {
      SST_FIRSTP(LHS, 0x8);
    } else if (sem_strcmp(np, "private") == 0) {
      SST_FIRSTP(LHS, 0x10);
    } else if (sem_strcmp(np, "public") == 0) {
      SST_FIRSTP(LHS, 0x20);
    } else {
      error(34, 3, gbl.lineno, np, CNULL);
    }
    break;
  /*
   *	<binding attr> ::= <id name> ( <id> )
   */
  case BINDING_ATTR2:
    /*
     * Not using keywords to enumerate the attributes; this must be
     *    PASS ( arg-name )
     */
    np = scn.id.name + SST_CVALG(RHS(1));
    if (sem_strcmp(np, "pass") == 0) {
      SST_FIRSTP(LHS, 0x1);
      SST_LSYMP(LHS, SST_SYMG(RHS(3)));
    } else {
      error(34, 3, gbl.lineno, np, CNULL);
    }
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<binding name list> ::= <binding name list> , <binding name> |
   */
  case BINDING_NAME_LIST1:
    break;
  /*
   *	<binding name list> ::= <binding name>
   */
  case BINDING_NAME_LIST2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<binding name> ::=  <id> |
   */
  case BINDING_NAME1:
    rhstop = 1;
    goto binding_name_common;
  /*
   *	<binding name> ::= <id> '=>' <id>
   */
  case BINDING_NAME2: {
    SPTR tag, sptr3, sptr2, orig_sptr;
    char *name, *name_cpy, *name_cpy2;
    DTYPE parent;
    SPTR sym;
    int vtoff, len;

    if (strcmp(SYMNAME(SST_SYMG(RHS(1))), SYMNAME(SST_SYMG(RHS(3)))) == 0) {
      rhstop = 1;
    } else {
      rhstop = 3;
    }

  binding_name_common:

    tag = DTY(stsk->dtype + 3);
    orig_sptr = sptr = SST_SYMG(RHS(1));
    if (sem.tbp_interface > NOSYM) {
      sptr2 = sem.tbp_interface;
    } else {
      sptr2 = refsym(SST_SYMG(RHS(rhstop)), OC_OTHER);
    }
  
    if (SEPARATEMPG(sptr2))
      TBP_BOUND_TO_SMPP(sptr2, TRUE);

    if (bindingNameRequiresOverloading(sptr)) {
      sptr = insert_sym(sptr);
    }

    parent = DTYPEG(PARENTG(tag));
    vtoff = 0;
    for (sym = get_struct_members(parent); sym > NOSYM; sym = SYMLKG(sym)) {
      if (is_tbp(sym)) {
        len = strlen(SYMNAME(BINDG(sym))) + 1;
        name_cpy = getitem(0, len);
        strcpy(name_cpy, SYMNAME(BINDG(sym)));
        name = strstr(name_cpy, "$tbp");
        if (name)
          *name = '\0';
        if (strcmp(name_cpy, SYMNAME(sptr)) == 0) {
          vtoff = VTOFFG(BINDG(sym));
          VTOFFP(sptr, vtoff);
          break;
        }
      }
    }
    if (rhstop == 1) {
      if (STYPEG(sptr2) && STYPEG(sptr2) != ST_PROC) {
        sptr2 = insert_sym(sptr2);
      }
      sptr = getsymf("%s$tbp", SYMNAME(sptr));
      if (STYPEG(sptr) > 0) {
        sptr = insert_sym(sptr);
      }
    }

    if (TBPLNKG(sptr) && !eq_dtype2(TBPLNKG(sptr), stsk->dtype, 1)) {
      sptr3 = insert_sym(sptr);
      STYPEP(sptr3, STYPEG(sptr));
      IGNOREP(sptr3, IGNOREG(sptr));
      sptr = sptr3;
      parent = DTYPEG(PARENTG(tag));
      sym = DTY(parent + 1);
      vtoff = 0;
      for (sym = get_struct_members(parent); sym > NOSYM; sym = SYMLKG(sym)) {
        if (CCSYMG(sym) && BINDG(sym)) {

          len = strlen(SYMNAME(BINDG(sym))) + 1;
          name_cpy = getitem(0, len);
          strcpy(name_cpy, SYMNAME(BINDG(sym)));
          name = strstr(name_cpy, "$tbp");
          if (name)
            *name = '\0';

          len = strlen(SYMNAME(sptr)) + 1;
          name_cpy2 = getitem(0, len);
          strcpy(name_cpy2, SYMNAME(sptr));
          name = strstr(name_cpy2, "$tbp");
          if (name)
            *name = '\0';

          if (strcmp(name_cpy, name_cpy2) == 0) {
            vtoff = VTOFFG(BINDG(sym));
            VTOFFP(sptr, vtoff);
            break;
          }
        }
      }
    }
    /* Ignore temporary binding name only if we're overloading
     * a binding name with a derived type name or if stype is 0.
     */

    if (STYPEG(orig_sptr) != ST_PD && STYPEG(sptr) != ST_PROC) {
      /* when found a binding name has a parameter attribute, don't ignore it 
       * as we need to export this sptr into a *.mod file.
       */
      if (STYPEG(orig_sptr) != ST_PARAM)
        IGNOREP(sptr, TRUE);
      sptr = insert_sym(sptr);
      sptr = declsym(sptr, ST_PROC, FALSE);
      IGNOREP(sptr, TRUE); /* Needed for overloading */
    }

    if (vtoff) {
      VTOFFP(sptr, vtoff);
    }

    if (!VTOFFG(tag) && PARENTG(tag) && VTOFFG(PARENTG(tag))) {
      VTOFFP(tag, VTOFFG(PARENTG(tag))); /*initialize offset*/
    }
    if (!VTOFFG(sptr) && !VTOFFG(tag) &&
        (vtoff = get_vtoff(0, stsk->dtype)) > 0) {
      /* Set vtable offset based on dtype and its parents */
      VTOFFP(sptr, vtoff + 1);
      VTOFFP(tag, vtoff + 1);
      CLASSP(sptr, 1);
    }
    if (!VTOFFG(sptr)) {
      /* Give this type bound procedure (tbp) an offset by incrementing
       * the tag's offset count and storing it in the tbp's PARENT field.
       */
      VTOFFP(tag, VTOFFG(tag) + 1);
      VTOFFP(sptr, VTOFFG(tag));
      CLASSP(sptr, 1);
    }

    /* keep track of pass object type in tbp by storing the "least extended"
     * type extension in TBPLNK field.
     */
    if (!TBPLNKG(sptr)) {
      TBPLNKP(sptr, /*sem.stag_dtype*/ stsk->dtype);
    } else if (eq_dtype2(/*DTYPEG*/ (TBPLNKG(sptr)),
                         /*sem.stag_dtype*/ stsk->dtype, 1)) {
      TBPLNKP(sptr, /*sem.stag_dtype*/ stsk->dtype);
    }
    queue_tbp(sptr2, sptr, VTOFFG(sptr), /*sem.stag_dtype*/ stsk->dtype,
              (rhstop == 1) ? TBP_ADD_SIMPLE : TBP_ADD_IMPL);

    /* If we pushed the binding name into the symbol table,
     * we might have to remove it now, as it might be masking
     * a previous name (e.g., a parameter).
     */
    if (!STYPEG(sptr) ||
        (orig_sptr > NOSYM &&
         HASHLKG(sptr) == orig_sptr &&
         STYPEG(orig_sptr))) {
      pop_sym(sptr);
    }
  } break;
  /* ------------------------------------------------------------------ */
  /*
   *      <accel decl begin> ::=
   */
  case ACCEL_DECL_BEGIN1:
    parstuff_init();
    break;
  /* ------------------------------------------------------------------ */
  /*
   *	<accel decl list> ::= <accel decl list> <opt comma> <accel decl attr> |
   */
  case ACCEL_DECL_LIST1:
    break;
  /*
   *	<accel decl list> ::= <accel decl attr>
   */
  case ACCEL_DECL_LIST2:
    break;
  /* ------------------------------------------------------------------ */
  /*
   *	<accel decl attr> ::= COPYIN ( <accel decl data list> ) |
   */
  case ACCEL_DECL_ATTR1:
    break;
  /*
   *	<accel decl attr> ::= COPYOUT ( <accel decl data list> ) |
   */
  case ACCEL_DECL_ATTR2:
    break;
  /*
   *	<accel decl attr> ::= LOCAL ( <accel decl data list> ) |
   */
  case ACCEL_DECL_ATTR3:
    break;
  /*
   *	<accel decl attr> ::= COPY ( <accel decl data list> ) |
   */
  case ACCEL_DECL_ATTR4:
    break;
  /*
   *	<accel decl attr> ::= MIRROR ( <accel mdecl data list> ) |
   */
  case ACCEL_DECL_ATTR5:
    break;
  /*
   *	<accel decl attr> ::= REFLECTED ( <accel mdecl data list> ) |
   */
  case ACCEL_DECL_ATTR6:
    break;
  /*
   *	<accel decl attr> ::= CREATE ( <accel decl data list> ) |
   */
  case ACCEL_DECL_ATTR7:
    break;
  /*
   *	<accel decl attr> ::= PRESENT ( <accel decl data list> ) |
   */
  case ACCEL_DECL_ATTR8:
    break;
  /*
   *	<accel decl attr> ::= PCOPY ( <accel decl data list> ) |
   */
  case ACCEL_DECL_ATTR9:
    break;
  /*
   *	<accel decl attr> ::= PCOPYIN ( <accel decl data list> ) |
   */
  case ACCEL_DECL_ATTR10:
    break;
  /*
   *	<accel decl attr> ::= PCOPYOUT ( <accel decl data list> ) |
   */
  case ACCEL_DECL_ATTR11:
    break;
  /*
   *	<accel decl attr> ::= PLOCAL ( <accel decl data list> ) |
   */
  case ACCEL_DECL_ATTR12:
    break;
  /*
   *	<accel decl attr> ::= PCREATE ( <accel decl data list> ) |
   */
  case ACCEL_DECL_ATTR13:
    break;
  /*
   *	<accel decl attr> ::= DEVICEPTR ( <accel decl data list> ) |
   */
  case ACCEL_DECL_ATTR14:
    break;
  /*
   *	<accel decl attr> ::= DEVICE_RESIDENT ( <accel decl data list> ) |
   */
  case ACCEL_DECL_ATTR15:
    break;
  /*
   *	<accel decl attr> ::= LINK ( <accel decl data list> ) |
   */
  case ACCEL_DECL_ATTR16:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<accel decl data list> ::= <accel decl data list> , <accel decl data> |
   */
  case ACCEL_DECL_DATA_LIST1:
  accel_decl_data_list1:
    itemp = (ITEM *)getitem(0, sizeof(ITEM));
    itemp->next = ITEM_END;
    itemp->ast = SST_ASTG(RHS(3));
    SST_ENDG(RHS(1))->next = itemp;
    SST_ENDP(LHS, itemp);
    break;
  /*
   *	<accel decl data list> ::= <accel decl data>
   */
  case ACCEL_DECL_DATA_LIST2:
  accel_decl_data_list2:
    itemp = (ITEM *)getitem(0, sizeof(ITEM));
    itemp->next = ITEM_END;
    itemp->ast = SST_ASTG(RHS(1));
    SST_BEGP(LHS, itemp);
    SST_ENDP(LHS, itemp);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<accel decl data> ::= <accel decl data name> ( <accel decl sub list> ) |
   */
  case ACCEL_DECL_DATA1:
  /*###*/
  accel_decl_data1:
    sptr = refsym((int)SST_SYMG(RHS(1)), OC_OTHER);
    switch (STYPEG(sptr)) {
    case ST_ARRAY:
      itemp = SST_BEGG(RHS(3));
      (void)mkvarref(RHS(1), itemp);
      SST_PARENP(LHS, 0); /* ? */
      break;
    default:
      error(155, 3, gbl.lineno, "Unknown symbol used in data clause -",
            SYMNAME(sptr));
      break;
    }
    break;
  /*
   *	<accel decl data> ::= <accel decl data name> |
   */
  /*###*/
  case ACCEL_DECL_DATA2:
  accel_decl_data2:
    sptr = refsym((int)SST_SYMG(RHS(1)), OC_OTHER);
    mkident(LHS);
    SST_SYMP(LHS, sptr);
    SST_DTYPEP(LHS, DTYPEG(sptr));
    SST_ASTP(LHS, mk_id(sptr));
    break;
  /*
   *	<accel decl data> ::= <constant> |
   */
  case ACCEL_DECL_DATA3:
    /*###*/
    break;
  /*
   *	<accel decl data> ::= <common>
   */
  case ACCEL_DECL_DATA4:
    sptr = SST_SYMG(RHS(1));
    SST_SYMP(LHS, sptr);
    SST_DTYPEP(LHS, 0);
    SST_ASTP(LHS, mk_id(sptr));
    break;
  /* ------------------------------------------------------------------ */
  /*
   *	<accel mdecl data> ::= <accel mdecl data name> ( <accel decl sub list> )
   *|
   */
  case ACCEL_MDECL_DATA1:
    goto accel_decl_data1;
  /*
   *	<accel mdecl data> ::= <accel mdecl data name>
   */
  case ACCEL_MDECL_DATA2:
    goto accel_decl_data2;
  /*
   *	<accel mdecl data> ::= <constant>
   */
  case ACCEL_MDECL_DATA3:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<accel mdecl data list> ::= <accel mdecl data list> , <accel mdecl data>
   *|
   */
  case ACCEL_MDECL_DATA_LIST1:
    goto accel_decl_data_list1;
  /*
   *	<accel mdecl data list> ::= <accel mdecl data>
   */
  case ACCEL_MDECL_DATA_LIST2:
    goto accel_decl_data_list2;

  /* ------------------------------------------------------------------ */
  /*
   *	<accel decl sub list> ::= <accel decl sub list> , <accel decl sub> |
   */
  case ACCEL_DECL_SUB_LIST1:
    itemp = (ITEM *)getitem(0, sizeof(ITEM));
    itemp->next = ITEM_END;
    itemp->t.stkp = SST_E1G(RHS(3));
    SST_ENDG(RHS(1))->next = itemp;
    SST_ENDP(LHS, itemp);
    break;
  /*
   *	<accel decl sub list> ::= <accel decl sub>
   */
  case ACCEL_DECL_SUB_LIST2:
    itemp = (ITEM *)getitem(0, sizeof(ITEM));
    itemp->next = ITEM_END;
    itemp->t.stkp = SST_E1G(RHS(1));
    SST_BEGP(LHS, itemp);
    SST_ENDP(LHS, itemp);
    break;
  /* ------------------------------------------------------------------ */
  /*
   *      <generic type procedure> ::=  GENERIC <opt gen access spec> ::
   * <generic binding>
   */
  case GENERIC_TYPE_PROCEDURE1:
    sptr = sem.interf_base[sem.interface - 1].generic;
    if (!sptr) {
      sptr = sem.interf_base[sem.interface - 1].operator;
      sem.generic_tbp = ST_OPERATOR;
    } else {
      sem.generic_tbp = ST_USERGENERIC;
    }

    switch (SST_FIRSTG(RHS(2))) {
    case 0x10:
      i = TBP_CHECK_PRIVATE; /* private */
      break;
    case 0x20:
      i = TBP_CHECK_PUBLIC; /* public */
      break;
    case 0x0:
    default:
      i = TBP_CHECK_CHILD;
    }
    for (itemp = SST_BEGG(RHS(4)); itemp != ITEM_END; itemp = itemp->next) {
      int tag;
      dtype = stsk->dtype;
      tag = DTY(dtype + 3);

      if (!VTOFFG(sptr)) {
        int vt = VTOFFG(tag);
        if (!vt && PARENTG(tag) && VTOFFG(PARENTG(tag))) {
          /* Seed the vtable offset field of derived type tag with its parent's
           * vtable offset. It will get updated in
           * <binding name> ::= <id> '=>' <id> production.
           */
          vt = VTOFFG(PARENTG(tag));
          VTOFFP(tag, vt);
        }
        /* Set offset of binding name to next offset. */
        VTOFFP(sptr, vt + 1);
        if (STYPEG(sptr) == ST_OPERATOR) {
/* Set CLASS flag so we can properly handle its
 * access in semfin.c do_access(). We don't set it for
 * ST_USERGENERIC here because a USERGENERIC can overload
 * a type name (including the type name of the type defining
 * the generic tbp).
 */
          CLASSP(sptr, 1);
        }
      }
      /* offset needs to be same as overloaded tbp */
      queue_tbp(itemp->t.sptr, sptr, VTOFFG(sptr), stsk->dtype, i);
    }
    sem.interface--;
    sem.generic_tbp = 0;
    sem.defined_io_type = 0;
    break;

  /*
   *      <opt gen access spec> ::= |
   */
  case OPT_GEN_ACCESS_SPEC1:
    SST_FIRSTP(LHS, 0x0);
    goto gen_access_spec_common;
  /*
   *      <opt gen access spec> ::= , <gen access spec>
   */
  case OPT_GEN_ACCESS_SPEC2:
    SST_FIRSTP(LHS, SST_FIRSTG(RHS(2)));
  gen_access_spec_common:
    sem.generic_tbp = 1;
    NEED(sem.interface + 1, sem.interf_base, INTERF, sem.interf_size,
         sem.interf_size + 2);
    sem.interf_base[sem.interface].generic = 0;
    sem.interf_base[sem.interface].operator= 0;
    sem.interf_base[sem.interface].opval = 0;
    sem.interf_base[sem.interface].abstract = 0;
    sem.interf_base[sem.interface].hpfdcl = sem.hpfdcl;
    sem.interface++;
    break;

  /*
   *      <gen access spec> ::= <id name>
   */
  case GEN_ACCESS_SPEC1:
    np = scn.id.name + SST_CVALG(RHS(1));
    sptr = getsymbol(np);
    if (strcmp(SYMNAME(sptr), "private") == 0)
      SST_FIRSTP(LHS, 0x10);
    else if (strcmp(SYMNAME(sptr), "public") == 0)
      SST_FIRSTP(LHS, 0x20);
    else
      error(155, 3, gbl.lineno, "Invalid access specifier in generic"
                                " type bound procedure",
            CNULL);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<accel decl sub> ::= <opt sub> : <opt sub> |
   */
  case ACCEL_DECL_SUB1:
    e1 = (SST *)getitem(sem.ssa_area, sizeof(SST));
    SST_IDP(e1, S_TRIPLE);
    SST_E1P(e1, (SST *)getitem(sem.ssa_area, sizeof(SST)));
    *(SST_E1G(e1)) = *RHS(1);
    SST_E2P(e1, (SST *)getitem(sem.ssa_area, sizeof(SST)));
    *(SST_E2G(e1)) = *RHS(3);
    SST_E3P(e1, (SST *)getitem(sem.ssa_area, sizeof(SST)));
    SST_IDP(SST_E3G(e1), S_NULL);
    SST_E1P(LHS, e1);
    SST_E2P(LHS, 0);
    break;
  /*
   *	<accel decl sub> ::= <expression>
   */
  case ACCEL_DECL_SUB2:
    e1 = (SST *)getitem(sem.ssa_area, sizeof(SST));
    *e1 = *RHS(1);
    SST_E1P(LHS, e1);
    SST_E2P(LHS, 0);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<accel routine list> ::= |
   */
  case ACCEL_ROUTINE_LIST1:
  break;
  /*
   *	<accel routine list> ::= <accel routine list> <opt comma> GANG |
   */
  case ACCEL_ROUTINE_LIST2:
  break;
  /*
   *	<accel routine list> ::= <accel routine list> <opt comma> WORKER |
   */
  case ACCEL_ROUTINE_LIST3:
  break;
  /*
   *	<accel routine list> ::= <accel routine list> <opt comma> VECTOR |
   */
  case ACCEL_ROUTINE_LIST4:
  break;
  /*
   *	<accel routine list> ::= <accel routine list> <opt comma> SEQ |
   */
  case ACCEL_ROUTINE_LIST5:
  break;
  /*
   *	<accel routine list> ::= <accel routine list> <opt comma> NOHOST |
   */
  case ACCEL_ROUTINE_LIST6:
  break;
  /*
   *	<accel routine list> ::= <accel routine list> <opt comma> BIND ( <ident>
   *) |
   */
  case ACCEL_ROUTINE_LIST7:
  break;
  /*
   *	<accel routine list> ::= <accel routine list> <opt comma> BIND ( <quoted
   *string> ) |
   */
  case ACCEL_ROUTINE_LIST8:
  break;
  /*
   *	<accel routine list> ::= <accel routine list> <opt comma> DEVICE_TYPE (
   *<devtype list> )
   */
  case ACCEL_ROUTINE_LIST9:
  break;
  /*
   *	<accel routine list> ::= <accel routine list> <opt comma> GANG ( <ident>
   *: <expression> )
   */
  case ACCEL_ROUTINE_LIST10:
  break;
  /*
   *	<accel routine list> ::= <accel routine list> <opt comma> EXCLUDE
   */
  case ACCEL_ROUTINE_LIST11:
  break;

  /* ------------------------------------------------------------------ */
  /*
   *	<devtype list> ::= <devtype list> , <devtype attr> |
   */
  case DEVTYPE_LIST1:
  break;
  /*
   *	<devtype list> ::= <devtype attr>
   */
  case DEVTYPE_LIST2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<devtype attr> ::= * |
   */
  case DEVTYPE_ATTR1:
    break;
  /*
   *	<devtype attr> ::= <ident>
   */
  case DEVTYPE_ATTR2:
  break;

  /* ------------------------------------------------------------------ */
  /*
   *      <generic binding> ::= <generic spec> '=>' <generic binding list>
   */
  case GENERIC_BINDING1:
    sptr = sem.interf_base[sem.interface - 1].generic;
    if (!sptr) {
      sptr = sem.interf_base[sem.interface - 1].operator;
    }
    TBPLNKP(sptr, stsk->dtype);
    SST_BEGP(LHS, SST_BEGG(RHS(3)));
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <generic binding name> ::= <id>
   */
  case GENERIC_BINDING_NAME1:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <generic binding list> ::= <generic binding name> |
   */
  case GENERIC_BINDING_LIST1:
    rhstop = 1;
    goto shared_generic_binding;
  /*
   *      <generic binding list> ::= <generic binding list>, <generic binding
   * name>
   */
  case GENERIC_BINDING_LIST2:
    rhstop = 3;
  shared_generic_binding:
    sptr = SST_SYMG(RHS(rhstop));
    itemp = (ITEM *)getitem(0, sizeof(ITEM));
    itemp->next = ITEM_END;
    itemp->t.sptr = sptr;
    if (rhstop == 1)
      /* adding first item to list */
      SST_BEGP(LHS, itemp);
    else
      /* adding subsequent items to list */
      SST_ENDG(RHS(1))->next = itemp;
    SST_ENDP(LHS, itemp);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<final subroutines> ::= FINAL <opt attr> <final list>
   */
  case FINAL_SUBROUTINES1:
    if (sem.type_mode < 2) {
      error(155, 3, gbl.lineno,
            "a FINAL subroutine statement can only appear"
            " within the type bound procedure part of a derived type",
            CNULL);
    }
    for (itemp = SST_BEGG(RHS(3)); itemp != ITEM_END; itemp = itemp->next) {
      dtype = stsk->dtype;
      sptr = itemp->t.sptr;
      queue_tbp(sptr, 0, 0, dtype, TBP_ADD_FINAL);
      /*queue_tbp(sptr, 0, 0, dtype, TBP_ADD_TO_DTYPE);*/
    }
    break;
  /* ------------------------------------------------------------------ */
  /*
   *      <final list> ::= <final>
   */
  case FINAL_LIST2:
    rhstop = 1;
    goto shared_final_sub;

  /* ------------------------------------------------------------------ */
  /*
   *	<final list> ::= <final list> , <final> |
   */
  case FINAL_LIST1:
    rhstop = 3;
  shared_final_sub:
    sptr = SST_SYMG(RHS(rhstop));
    itemp = (ITEM *)getitem(0, sizeof(ITEM));
    itemp->next = ITEM_END;
    itemp->t.sptr = sptr;
    if (rhstop == 1)
      /* adding first item to list */
      SST_BEGP(LHS, itemp);
    else
      /* adding subsequent items to list */
      SST_ENDG(RHS(1))->next = itemp;
    SST_ENDP(LHS, itemp);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<mp decl begin> ::=
   */
  case MP_DECL_BEGIN1:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<mp decl> ::= <mp declaresimd> <declare simd> |
   */
  case MP_DECL1:
#ifdef OMP_OFFLOAD_LLVM
    if(flg.omptarget) {
      error(1200, ERR_Severe, gbl.lineno, "declare simd",
            NULL);
    }
#endif
    break;
  /*
   *	<mp decl> ::= <declare target> <opt par list> |
   */
  case MP_DECL2:
#ifdef OMP_OFFLOAD_LLVM
    if(flg.omptarget) {
      error(1200, ERR_Severe, gbl.lineno, "declare target",
            NULL);
    }
#endif
    break;
  /*
   *	<mp decl> ::= <declarered begin> <declare reduction>
   */
  case MP_DECL3:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<declarered begin> ::= <mp declarereduction>
   */
  case DECLARERED_BEGIN1:
    if (sem.which_pass == 0)
      error(155, 2, gbl.lineno, "Unimplemented feature - DECLARE REDUCTION",
            NULL);
    sem.ignore_stmt = TRUE;
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<declare reduction> ::= ( <reduc op> : <type list> : <red comb> ) <opt
   *red init>
   */
  case DECLARE_REDUCTION1:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<type list> ::= <type list> , <red type> |
   */
  case TYPE_LIST1:
    break;
  /*
   *	<type list> ::= <red type>
   */
  case TYPE_LIST2:
    break;

  /* ------------------------------------------------------------------ */
  default:
    interr("semant1:bad rednum", rednum, 3);
    break;
  }
}

/** Make a unique func ast with a unique sptr (and name) so we can 
    set its associated pointer field. The unique sptr is a placeholder
    for the pointer target so it does not conflict with the 
    original ST_PROC symbol. We hold the original pointer target
    in the PTR_TARGET field.
*/
static void
gen_unique_func_ast(int ast, SPTR sptr, SST *stkptr)

{
  SPTR sym, orig_sym = sym_of_ast(ast);

  sym = get_next_sym(SYMNAME(orig_sym), "tgt");
  STYPEP(sym, STYPEG(orig_sym));
  SCP(sym, SCG(orig_sym));
  SCOPEP(sym, SC_NONE);
  ASSOC_PTRP(sym, sptr);
  PTR_TARGETP(sym, orig_sym); 
  PTR_TARGETP(sptr, orig_sym);
  DINITP(sym, 1);
  ast = replace_memsym_of_ast(ast, sym);
  SST_ASTP(stkptr, ast);
  if (STYPEG(SCOPEG(orig_sym)) == ST_MODULE) {
    INMODULEP(orig_sym, 1);
  }
}

static void
gen_dinit(int sptr, SST *stkptr)
{
  switch (STYPEG(sptr)) { /* change symbol type if necessary */
  case ST_UNKNOWN:
  case ST_IDENT:
    STYPEP(sptr, ST_VAR);
    FLANG_FALLTHROUGH;
  case ST_VAR:
  case ST_ARRAY:
    if (SCG(sptr) == SC_NONE)
      SCP(sptr, SC_LOCAL);
    if (!dinit_ok(sptr))
      return;
    break;
  case ST_STAG:
  case ST_STRUCT:
  case ST_MEMBER:
    break;
  case ST_GENERIC:
  case ST_INTRIN:
  case ST_PD:
    if ((sptr = newsym(sptr)) == 0)
      /* Symbol frozen as an intrinsic, ignore data initialization */
      return;
    break;
  default:
    error(84, 3, gbl.lineno, SYMNAME(sptr), CNULL);
    return;
  }

  if (flg.xref)
    xrefput(sptr, 'i');

  if (SCG(sptr) == SC_DUMMY) {
    /* Dummy variables may not be initialized */
    error(41, 3, gbl.lineno, SYMNAME(sptr), CNULL);
    sem.dinit_error = TRUE;
  }

  if (sem.dinit_count > 0) {
    errsev(66);
    sem.dinit_error = TRUE;
  }

  /* Call dinit to generate dinit records */
  if (INSIDE_STRUCT) {
    /* In structure so accumulate Initializer Constant Tree
     * in the structure stack.
     */
    /* Set first constant to point to variable needing init'd */
    (SST_CLBEGG(stkptr))->sptr = sptr;
    stsk = &STSK_ENT(0);
    if (stsk->ict_beg) {
      (stsk->ict_end)->next = SST_CLBEGG(stkptr);
      stsk->ict_end = SST_CLENDG(stkptr);
    } else {
      stsk->ict_beg = SST_CLBEGG(stkptr);
      stsk->ict_end = SST_CLENDG(stkptr);
    }
  } else {
    /* Not in structure so generate dinit records */
    if (!sem.dinit_error) {
      SST tmpsst;
      VAR *ivl;
      mkident(&tmpsst);
      SST_SYMP(&tmpsst, sptr);
      SST_DTYPEP(&tmpsst, DTYPEG(sptr));
      SST_SHAPEP(&tmpsst, 0);
      SST_ASTP(&tmpsst, mk_id(sptr));
      SST_SHAPEP(&tmpsst, A_SHAPEG(SST_ASTG(&tmpsst)));
      ivl = dinit_varref(&tmpsst);
      dinit(ivl, SST_CLBEGG(stkptr));
    }
    sem.dinit_error = FALSE;
  }
}

static void
pop_subprogram(void)
{
  int scope;
  if (sem.none_implicit) {
    int i, arg;
    int *dscptr;

    dscptr = aux.dpdsc_base + DPDSCG(gbl.currsub);
    for (i = PARAMCTG(gbl.currsub); i > 0; i--)
      if ((arg = *dscptr++)) {
        /* any implicit typing needs to be explicit */
        switch (STYPEG(arg)) {
        case ST_VAR:
        case ST_ARRAY:
          DCLCHK(arg);
          DCLDP(arg, TRUE);
          break;
        case ST_PROC:
          if (FUNCG(arg)) {
            DCLCHK(arg);
            DCLDP(arg, TRUE);
          }
          break;
        default:
          break;
        }
      }
  }
  if (gbl.rutype == RU_FUNC) {
    DCLCHK(gbl.currsub);
    DCLDP(gbl.currsub, TRUE); /* any implicit typing needs to be explicit */
  }

  STYPEP(gbl.currsub, ST_PROC);
  if (sem.interface && SCG(gbl.currsub) == SC_DUMMY) {
    /* if this is a interface block definition of a subprogram
     * for a dummy argument, force it to appear in an external statement */
    TYPDP(gbl.currsub, 1);
    IS_PROC_DUMMYP(gbl.currsub, 1);
  }
  /* if this is an interface block for the program we are compiling,
   * ignore this symbol henceforth */
  scope = SCOPEG(gbl.currsub);
  if (scope && NMPTRG(gbl.currsub) == NMPTRG(scope)) {
    IGNOREP(gbl.currsub, TRUE);
    pop_sym(gbl.currsub);
  }
  gbl.currsub = 0;
  gbl.rutype = 0;
  sem.module_procedure = FALSE;
  sem.pgphase = PHASE_INIT;
  symutl.none_implicit = sem.none_implicit = flg.dclchk;
  seen_implicit = FALSE;
  seen_parameter = FALSE;
}

static void
set_len_attributes(SST *stkptr, int lvl)
{
  /* lenspec[].kind */ /* 0 - length not present
                        * 1 - constant length
                        * 2 - length is '*'
                        * 3 - length is zero
                        * 4 - length is adjustable
                        * 5 - length is ':'
                        */
  /* lenspec[].len */  /* -1 if length not present;
                        * -2 if zero length;
                        * -3 if ':';
                        * 0 if '*';
                        * constant value if length is constant;
                        * ast of adjustable length expression.
                        */
  if (SST_IDG(stkptr) == 0) {
    lenspec[lvl].len = SST_SYMG(stkptr);
    switch (lenspec[lvl].len) {
    case -1:
      lenspec[lvl].kind = 0;
      break;
    case -2:
      lenspec[lvl].kind = _LEN_ZERO;
      break;
    default:
      lenspec[lvl].kind = _LEN_CONST;
    }
  } else {
    lenspec[lvl].len = SST_ASTG(stkptr);
    if (lenspec[lvl].len == 0 && SST_SYMG(stkptr) == -1) {
      lenspec[lvl].kind = _LEN_DEFER;
    } else if (lenspec[lvl].len == 0)
      lenspec[lvl].kind = _LEN_ASSUM;
    else
      lenspec[lvl].kind = _LEN_ADJ;
  }
  if (lvl == 0 || (lenspec[1].kind == 0 && lenspec[0].kind)) {
    /* propagate the global length attributes if:
     * 1.  the global attributes are being set, or
     * 2.  the augmented attributes were not present and the global
     *     attributes were present.
     */
    lenspec[1] = lenspec[0];
    lenspec[1].propagated = 1;
  } else {
    lenspec[lvl].propagated = 0;
  }
}

static void
set_char_attributes(int sptr, int *pdtype)
{
  int dtype;
  dtype = *pdtype;
  if (DTY(dtype) != TY_CHAR && DTY(dtype) != TY_NCHAR)
    return;
  if (lenspec[1].kind == _LEN_ADJ) {
    ADJLENP(sptr, 1);
  }
  if (lenspec[1].kind == _LEN_ASSUM) {
    ASSUMLENP(sptr, 1);
  }
}

static void
set_aclen(SST *stkptr, int ivl, int flag)
{
  static int kind0, kind1, propagate0, propagate1;
  static INT len0, len1;

  if (flag) {
    len0 = lenspec[0].len;
    kind0 = lenspec[0].kind;
    propagate0 = lenspec[0].propagated;
    len1 = lenspec[1].len;
    kind1 = lenspec[1].kind;
    propagate1 = lenspec[1].propagated;
    lenspec[0].len = 0;
    lenspec[0].kind = 0;
    lenspec[0].propagated = 0;
    lenspec[1].len = 0;
    lenspec[1].kind = 0;
    lenspec[1].propagated = 0;

    set_len_attributes(stkptr, ivl);
  } else {
    lenspec[0].len = len0;
    lenspec[0].kind = kind0;
    lenspec[0].propagated = propagate0;
    lenspec[1].len = len1;
    lenspec[1].kind = kind1;
    lenspec[1].propagated = propagate1;
  }
}

#ifdef FLANG_SEMANT_UNUSED
static int
get_actype(SST *stkptr, int ivl)
{
  sem.gdtype = mod_type(sem.gdtype, sem.gty, lenspec[ivl].kind,
                        lenspec[ivl].len, lenspec[ivl].propagated, 0);
  return sem.gdtype;
}
#endif

static void
ctte(int entry, int sptr)
{
  int dtype;
  ADJARRP(entry, ADJARRG(sptr));
  ADJLENP(entry, ADJLENG(sptr));
  ALLOCP(entry, ALLOCG(sptr));
  ASSUMSHPP(entry, ASSUMSHPG(sptr));
  ASUMSZP(entry, ASUMSZG(sptr));
  DCLDP(entry, DCLDG(sptr));
  DTYPEP(entry, DTYPEG(sptr));
  POINTERP(entry, POINTERG(sptr));
  F90POINTERP(entry, F90POINTERG(sptr));
  SEQP(entry, SEQG(sptr));
  /* check that the datatype is a legal function datatype */
  dtype = DTYPEG(sptr);
  if (POINTERG(sptr)) {
    /* cannot be a character(len=*) */
    if (dtype == DT_ASSCHAR || dtype == DT_ASSNCHAR) {
      error(155, 3, gbl.lineno,
            "Function result cannot be assumed-length character pointer -",
            SYMNAME(sptr));
      POINTERP(sptr, FALSE);
      POINTERP(entry, FALSE);
    }
  }
  if (DTY(dtype) == TY_ARRAY) {
    /* cannot be a character(len=*) */
    if (DTY(dtype + 1) == DT_ASSCHAR || DTY(dtype + 1) == DT_ASSNCHAR) {
      error(155, 3, gbl.lineno,
            "Function result cannot be assumed-length character array -",
            SYMNAME(sptr));
      DTYPEP(sptr, DTY(dtype + 1));
      DTYPEP(entry, DTY(dtype + 1));
      dtype = DTY(dtype + 1);
    }
  }
} /* ctte */

static void
copy_type_to_entry(int sptr)
{
  if (RESULTG(sptr)) {
    if (sem.interface) {
      /* find the entry symbol in the interface block */
      int sl, e;
      for (sl = sem.scope_level; sl > 0; --sl) {
        e = sem.scope_stack[sl].sptr;
        if (STYPEG(e) == ST_ENTRY || STYPEG(e) == ST_PROC) {
          if (FVALG(e) == sptr)
            ctte(e, sptr);
        }
        if (sem.scope_stack[sl].kind == SCOPE_INTERFACE)
          break;
      }
      for (e = sem.scope_stack[sl].symavl; e < stb.stg_avail; ++e) {
        if (STYPEG(e) == ST_ENTRY || STYPEG(e) == ST_PROC) {
          if (FVALG(e) == sptr)
            ctte(e, sptr);
        }
      }
    } else {
      int e;
      /*  scan all entries. NOTE: gbl.entries not yet set  */
      for (e = gbl.currsub; e > NOSYM; e = SYMLKG(e)) {
        if (FVALG(e) == sptr)
          ctte(e, sptr);
      }
    }
  }
} /* copy_type_to_entry */

static void
save_host(INTERF *state)
{
  state->currsub = gbl.currsub;
  state->rutype = gbl.rutype;
  state->module_procedure = sem.module_procedure;
  state->pgphase = sem.pgphase;
  state->none_implicit = sem.none_implicit;
  state->seen_implicit = seen_implicit;
  state->seen_parameter = seen_parameter;
  state->gnr_rutype = 0;
  state->nml = sem.nml;

  gbl.currsub = 0;
  gbl.rutype = 0;
  sem.module_procedure = false;
  sem.pgphase = PHASE_INIT;
  symutl.none_implicit = sem.none_implicit = flg.dclchk;
  seen_implicit = FALSE;
  seen_parameter = FALSE;
  save_implicit(FALSE); /* save host's implicit state */
}

static void
restore_host(INTERF *state, LOGICAL keep_implicit)
{
  gbl.currsub = state->currsub;
  gbl.rutype = state->rutype;
  sem.module_procedure = state->module_procedure;
  sem.pgphase = state->pgphase;
  symutl.none_implicit = sem.none_implicit = state->none_implicit;
  seen_implicit = state->seen_implicit;
  seen_parameter = state->seen_parameter;
  sem.nml = state->nml;
  restore_implicit(); /* restore host's implicit state */
  if (keep_implicit) {
    save_implicit(TRUE);
    /* in a contained subprogram, ignore host's implicit/parameter stmts */
    seen_implicit = FALSE;
    seen_parameter = FALSE;
  }
}

/* return TRUE if the name on the end is different from the name
 * of the routine */
static LOGICAL
wrong_name(SPTR endname)
{
  if (endname == 0)
    return FALSE;
  if (UNAMEG(gbl.currsub)) {
    /* compare to the original name */
    char *uname = stb.n_base + UNAMEG(gbl.currsub);
    return strcmp(uname, SYMNAME(endname)) != 0;
  }
  return strcmp(SYMNAME(gbl.currsub), SYMNAME(endname)) != 0;
} /* wrong_name */

/** Reset scopes and related set ups after processing and subroutine 
 */
static void
do_end_subprogram(SST *top, RU_TYPE rutype)
{
  fix_iface(gbl.currsub);
  if (sem.interface && IN_MODULE) {
    do_iface_module();
  }
  if (sem.which_pass && !sem.interface) {
    fix_class_args(gbl.currsub);
  }
  if (/*!IN_MODULE*/ !sem.mod_cnt && !sem.interface) {
    queue_tbp(0, 0, 0, 0, TBP_COMPLETE_END);
    queue_tbp(0, 0, 0, 0, TBP_CLEAR);
  }
  defer_pt_decl(0, 0);
  dummy_program();
  check_end_subprogram(rutype, SST_SYMG(RHS(2)));

  SST_IDP(LHS, 1); /* mark as end of subprogram unit */
  if (IN_MODULE && sem.interface == 0)
    mod_end_subprogram();
  pop_scope_level(SCOPE_NORMAL);
  check_defined_io();
  if (!IN_MODULE && !sem.interface)
    clear_ident_list();
  fix_proc_ptr_dummy_args();
  sem.seen_import = FALSE;
}

static void
check_end_subprogram(RU_TYPE rutype, int sym)
{
  if (gbl.currsub == 0) {
    if (sem.pgphase == PHASE_INIT && gbl.internal) {
      /* end of subprogram containing internal subprograms */
      restore_host(&host_state, TRUE);
      gbl.internal = 0;
      check_end_subprogram(rutype, sym);
      end_of_host = gbl.currsub;
      gbl.currsub = 0;
      if (sem.which_pass)
        end_contained();
      if (scn.currlab && sem.which_pass == 0)
        /* The end statement of the host subprogram is labeled.
         * Save its number for when the host's CONTAINS statement is
         * processed during the second pass.
         */
        sem.end_host_labno = scn.labno;
      return;
    }
    if (gbl.internal && sem.pgphase == PHASE_END && sem.which_pass == 0) {
      /* end of module subprogram containing internal subprograms */
      restore_host(&host_state, TRUE);
      gbl.internal = 0;
      sem.pgphase = PHASE_INIT;
      return;
    }
    error(302, 3, gbl.lineno, name_of_rutype(rutype), CNULL);
    gbl.internal = 0;
  } else if (gbl.rutype != rutype) {
    error(302, 3, gbl.lineno, name_of_rutype(rutype), CNULL);
  } else if (sym && wrong_name(sym))
    error(309, 3, gbl.lineno, SYMNAME(sym), CNULL);

  enforce_denorm();
}

static const char *
name_of_rutype(RU_TYPE rutype)
{
  switch (rutype) {
  case RU_SUBR:
    return "SUBROUTINE";
  case RU_FUNC:
    return "FUNCTION";
  case RU_PROC:
    return "PROCEDURE";
  case RU_PROG:
    return "PROGRAM";
  case RU_BDATA:
    return "BLOCKDATA";
  }
  return "";
}

/* If an intrinsic is declared in a host subprogram and not otherwise used,
 * convert it to an identifier for the internal subprograms to share.
 */
static void
convert_intrinsics_to_idents()
{
  SPTR sptr;
  assert(gbl.currsub && gbl.internal == 0,
         "only applicable for non-internal subprogram", 0, ERR_Severe);
  for (sptr = NOSYM + 1; sptr < stb.firstusym; ++sptr) {
    if (DCLDG(sptr) && !EXPSTG(sptr) && IS_INTRINSIC(STYPEG(sptr))) {
      SPTR new_sptr = newsym(sptr);
      STYPEP(new_sptr, ST_IDENT);
    }
  }
}

/*
 * In certain contexts, a new symbol must be created immediately
 * if the identifier is an intrinsic rather than relying on newsym().
 * For example, calling newsym() on a formal argument in an interface
 * block creates a new symbol as expected, but the effects of the
 * appearance of the intrinsic name in a type statement in an outer
 * scope are applied to the new symbol:
 *      integer cos	<- sets the DCLD flag of the generic
 *      interface
 *          subroutine sub(cos)
 *          integer cos <- newsym, but generic's DCLD flag is applied
 *          endsubroutine
 *      endinterface
 *      call sub(cos)   <- the first type statement no longer applies
 */
static int
chk_intrinsic(int first, LOGICAL now, LOGICAL settype)
{
  int oldsptr;
  int sptr;

  sptr = getocsym(first, OC_OTHER, FALSE);
  if (IS_INTRINSIC(STYPEG(sptr))) {
    if ((sem.interface && DCLDG(sptr)) || now) {
      error(35, 1, gbl.lineno, SYMNAME(sptr), CNULL);
      oldsptr = sptr;
      sptr = insert_sym(sptr);
      if (now && settype && DCLDG(oldsptr)) {
        DTYPEP(sptr, DTYPEG(oldsptr));
        DCLDP(sptr, TRUE);
      }
    }
  }
  return sptr;
}

/*
 * Create a ST_ENTRY for a function ENTRY.  Must be aware of the situation
 * where a variable named the same as the entry already exists.
 */
static int
create_func_entry(int sptr)
{
  int func_result = chk_func_entry_result(sptr);
  if (func_result > NOSYM) {
    sptr = 0;
    if (sem.which_pass && IN_MODULE) {
      /* if in a module, we have already seen the ENTRY during
       * which_pass == 0; get THAT symbol */
      for (sptr = first_hash(func_result); sptr > NOSYM; sptr = HASHLKG(sptr)) {
        if (NMPTRG(sptr) == NMPTRG(func_result) && STYPEG(sptr) == ST_PROC &&
            FVALG(sptr) == func_result) {
          break;
        }
        if (NMPTRG(sptr) == NMPTRG(func_result) && STYPEG(sptr) == ST_ALIAS &&
            STYPEG(SYMLKG(sptr)) == ST_PROC &&
            SCOPEG(SYMLKG(sptr)) == SCOPEG(func_result)) {
          break;
        }
      }
    }
    /* sptr is the old symbol for the entry point, now an ST_PROC */
    if (sptr) {
      int fval;
      if (STYPEG(sptr) == ST_ALIAS) {
        fval = FVALG(SYMLKG(sptr));
      } else {
        fval = FVALG(sptr);
      }
      if (fval) {
        STYPEP(fval, ST_UNKNOWN);
        IGNOREP(fval, TRUE);
        HIDDENP(fval, TRUE);
        FVALP(sptr, 0);
      }
    } else {
      /* A variable is already defined in the same scope of
       * the entry and assume that the variable's declaration
       * is for the entry.  Create a new symbol as the
       * ST_ENTRY; make the variable found by chk_func_entry_result
       * the function result of the ST_ENTRY.
       */
      sptr = insert_sym(func_result);
    }

    SCP(func_result, SC_DUMMY);
    RESULTP(func_result, TRUE);
    pop_sym(func_result);
    sptr = declsym(sptr, ST_ENTRY, TRUE);
    DTYPEP(sptr, DTYPEG(func_result));
    ADJLENP(sptr, ADJLENG(func_result));
    DCLDP(sptr, DCLDG(func_result));
    FVALP(sptr, func_result);
    return sptr;
  }
  sptr = declsym(sptr, ST_ENTRY, TRUE);
  if (SCG(sptr) != SC_NONE)
    error(43, 3, gbl.lineno, SYMNAME(sptr), CNULL);
  return sptr;
}

/*
 * Create the result variable for a function ENTRY.  Must be aware of the
 * situation where a variable named the same as the 'result' already exists.
 */
static int
create_func_entry_result(int sptr)
{
  int func_result = chk_func_entry_result(sptr);
  if (func_result > NOSYM) {
    /* A variable is already defined in the same scope of
     * the entry and assume that the variable's declaration
     * is for the entry.  Just use the variable as the
     * result of the entry.
     */
    SCP(func_result, SC_DUMMY);
    RESULTP(func_result, TRUE);
    return func_result;
  }
  sptr = declsym(sptr, ST_IDENT, TRUE);
  SCP(sptr, SC_DUMMY);
  return sptr;
}

/*
 * Retrieve/create a variable in the current scope.  Must be aware of
 * the situation where a variable is a function in which case, its
 * result variable must be used.
 */
static int
create_var(int sym)
{
  int sptr;
  sptr = refsym_inscope(sym, OC_OTHER);
  switch (STYPEG(sptr)) {
  case ST_ENTRY:
    if (gbl.rutype != RU_FUNC) {
      error(43, 3, gbl.lineno, "subprogram or entry name", SYMNAME(sptr));
      sptr = insert_sym(sptr);
    } else {
      /* should we specify the RESULT name? */
      if (RESULTG(sptr)) {
        error(43, 3, gbl.lineno, SYMNAME(sptr),
              "- you must specify the RESULT name");
      }
      sptr = FVALG(sptr);
    }
    break;
  case ST_MODULE:
    if (!DCLDG(sptr)) {
      /*
       * if the module is indirectly USEd (DCLD is not set)
       * it's ok to create a new symbol when used.
       * Otherwise, the module name is stll visible.
       */
      sptr = insert_sym(sptr);
    }
    break;
  default:;
  }
  return sptr;
}

/*
 * For entries, the variable specified in the result clause or
 * the variable implied by the entry name may have already been
 * declared in the same scope; also, the variable may have already
 * been referenced.  Determine if a variable has already been declared
 * whose name is the same as the entry or the result variable.
 */
static int
chk_func_entry_result(int sptr)
{
  int sptr2;

  sptr = refsym(sptr, OC_OTHER);
  switch (STYPEG(sptr)) {
  case ST_IDENT:
  case ST_VAR:
  case ST_ARRAY:
    switch (SCG(sptr)) {
    case SC_NONE:
    case SC_LOCAL:
      sptr2 = SCOPEG(sptr);
      if (sptr2 == 0)
        break;
      if (STYPEG(sptr2) == ST_ALIAS)
        sptr2 = SYMLKG(sptr2);
      if (sptr2 == gbl.currsub) {
        /* A variable is already defined in the same scope of
         * the entry and assume that the variable's declaration
         * is for the entry or the result.
         */
        return sptr;
      }
      break;
    default:;
    }
    break;
  default:;
  }
  /* a variable with the same name doesn't exist in the same scope: */
  return 0;
}

static void
get_param_alias_const(SST *stkp, int param_sptr, int dtype)
{
  int ast;
  int alias;
  INT conval;
  ACL *aclp;

  if (SST_IDG(stkp) == S_EXPR) {
    aclp = construct_acl_from_ast(SST_ASTG(stkp), dtype, 0);
    if (sem.dinit_error || !aclp) {
      return;
    }
    aclp = eval_init_expr(aclp);
    conval = cngcon(aclp->conval, aclp->dtype, dtype);
  } else if (SST_IDG(stkp) == S_LVALUE && stkp->value.cnval.acl) {
    construct_acl_for_sst(stkp, dtype);
    aclp = SST_ACLG(stkp);
    if (sem.dinit_error || !aclp) {
      return;
    }
    aclp = eval_init_expr(aclp);
    conval = cngcon(aclp->conval, aclp->dtype, dtype);
  } else {
    conval = chkcon(stkp, dtype, FALSE);
  }
  CONVAL1P(param_sptr, conval);
  if (dtype == DT_ASSCHAR || dtype == DT_ASSNCHAR || dtype == DT_DEFERCHAR ||
      dtype == DT_DEFERNCHAR)
    DTYPEP(param_sptr, DTYPEG(CONVAL1G(param_sptr)));
  alias = mk_cval1(conval, (int)DTYPEG(param_sptr));
  CONVAL2P(param_sptr, alias); /* ast of <expression> */
  if (sem.interface == 0)
    add_param(param_sptr);
  /* create an ast for the parameter; set the alias field of the ast
   * so that we don't have to set the alias field whenever the
   */
  ast = mk_id(param_sptr);
  A_ALIASP(ast, alias);
}

/* get the char length from the initialization expression */
static void
set_string_type_from_init(int sptr, ACL *init_acl)
{
  int sdtype = DTYPEG(sptr);
  int ndtype = init_acl->dtype;

  if (DTY(ndtype) == TY_ARRAY)
    ndtype = DTY(ndtype + 1);
  /* get the new char length */
  if (DTY(sdtype) == TY_ARRAY) {
    /* make array type with new char subtype, same bounds */
    ndtype = get_type(3, TY_ARRAY, ndtype);
    DTY(ndtype + 2) = DTY(sdtype + 2);
  }
  DTYPEP(sptr, ndtype);
}

static void
fixup_param_vars(SST *var, SST *init)
{
  int sptr;
  int sptr1;
  int dtype;
  ADSC *ad;
  int sdtype;
  ACL *aclp;

  sptr = SST_SYMG(var);
  PARAMP(sptr, 1);

  if (SST_IDG(init) == S_EXPR && A_TYPEG(SST_ASTG(init)) == A_INTR &&
      DTY(SST_DTYPEG(init)) == TY_ARRAY) {
    aclp = construct_acl_from_ast(SST_ASTG(init), SST_DTYPEG(init), 0);
    dinit_struct_param(sptr, aclp, SST_DTYPEG(init));

    sdtype = DTYPEG(sptr);
    if (DDTG(sdtype) == DT_ASSCHAR || DDTG(sdtype) == DT_ASSNCHAR ||
        DDTG(sdtype) == DT_DEFERCHAR || DDTG(sdtype) == DT_DEFERNCHAR) {
      set_string_type_from_init(sptr, aclp);
    }
  } else if (SST_IDG(init) == S_SCONST) {
    construct_acl_for_sst(init, SST_DTYPEG(init));
    dinit_struct_param(sptr, SST_ACLG(init), SST_DTYPEG(init));

    sdtype = DTYPEG(sptr);
    if (DDTG(sdtype) == DT_ASSCHAR || DDTG(sdtype) == DT_ASSNCHAR ||
        DDTG(sdtype) == DT_DEFERCHAR || DDTG(sdtype) == DT_DEFERNCHAR) {
      set_string_type_from_init(sptr, SST_ACLG(init));
    }
  } else if (SST_IDG(init) == S_ACONST ||
             (SST_IDG(init) == S_IDENT &&
              (STYPEG(SST_SYMG(init)) == ST_PARAM || PARAMG(SST_SYMG(init))))) {
    sdtype = DTYPEG(sptr);
    if (DDTG(sdtype) == DT_ASSCHAR || DDTG(sdtype) == DT_ASSNCHAR ||
        DDTG(sdtype) == DT_DEFERCHAR || DDTG(sdtype) == DT_DEFERCHAR) {
      set_string_type_from_init(sptr, SST_ACLG(init));
    }

    dinit_struct_param(sptr, SST_ACLG(init), DTYPEG(sptr));
  } else if (DTY(DTYPEG(sptr)) == TY_ARRAY && SST_IDG(init) == S_CONST &&
             (DDTG(DTYPEG(sptr)) == DT_ASSCHAR ||
              DDTG(DTYPEG(sptr)) == DT_ASSNCHAR)) {
    aclp = construct_acl_from_ast(SST_ASTG(init), SST_DTYPEG(init), 0);
    set_string_type_from_init(sptr, aclp);
  } else if (DTY(DTYPEG(sptr)) == TY_ARRAY && SST_IDG(init) == S_CONST) {
    aclp = construct_acl_from_ast(SST_ASTG(init), SST_DTYPEG(init), 0);
    dinit_struct_param(sptr, aclp, SST_DTYPEG(init));
  }

  if ((STYPEG(sptr) == ST_ARRAY) && SCG(sptr) == SC_NONE &&
      SCOPEG(sptr) == stb.curr_scope) {
    STYPEP(sptr, ST_PARAM);
    if (flg.xref)
      xrefput(sptr, 'd');
  } else if (STYPEG(sptr) == ST_VAR && DTY(DTYPEG(sptr)) == TY_ARRAY &&
             SCOPEG(sptr) == stb.curr_scope) {
/* HACK: if the named constant being defined has an initializer
 * that contains an intrinsic call that uses the named constant
 * as an argument, then the argument handling may have
 * changed the item's STYPE to ST_VAR when array. Change it back to
 * an ST_PARAM.
 */
    STYPEP(sptr, ST_PARAM);
    if (flg.xref)
      xrefput(sptr, 'd');

  } else if (STYPEG(sptr) == ST_VAR && SCOPEG(sptr) == stb.curr_scope &&
             KINDG(sptr)) {
    /* Overloaded type parameter */
    STYPEP(sptr, ST_PARAM);
    if (flg.xref)
      xrefput(sptr, 'd');

  } else if (STYPEG(sptr) == ST_IDENT && SCOPEG(sptr) == stb.curr_scope) {
    STYPEP(sptr, ST_PARAM);
    if (flg.xref)
      xrefput(sptr, 'd');

  } else {
    sptr = declsym(sptr, ST_PARAM, TRUE);
    if (SCG(sptr) != SC_NONE) {
      error(43, 3, gbl.lineno, "symbol", SYMNAME(sptr));
      return;
    }
  }

  dtype = DTYPEG(sptr);
  if (DTY(dtype) == TY_DERIVED) {
    sptr1 = get_param_alias_var(sptr, dtype);
  } else if (DTY(dtype) == TY_ARRAY) {
    ad = AD_DPTR(dtype);
    if (AD_ADJARR(ad) || AD_DEFER(ad)) {
      error(84, 3, gbl.lineno, SYMNAME(sptr),
            "- a named constant array must have constant extents");
      /* recover as zero sized array */
      int i;
      int ndim = AD_NUMDIM(ad);
      for (i = 0; i < ndim; i++) {
        AD_LWBD(ad, i) = AD_LWAST(ad, i) = astb.bnd.one;
        AD_UPBD(ad, i) = AD_UPAST(ad, i) = astb.bnd.zero;
        AD_EXTNTAST(ad, i) = astb.bnd.zero;
        AD_MLPYR(ad, i) = astb.bnd.zero;
      }
      AD_ZBASE(ad) = astb.bnd.one;
      AD_ADJARR(ad) = AD_DEFER(ad) = AD_NOBOUNDS(ad) = 0;
      goto param_alias;
    }
    if (AD_ASSUMSZ(ad)) {
      int i, dtype2;
      dtype2 = SST_DTYPEG(init);
      ADSC *ad2 = AD_DPTR(dtype2);
      int ndim1 = AD_NUMDIM(ad);
      int ndim2 = AD_NUMDIM(ad2);
      int lb1, ub1, lb2, ub2, zbase;

      if (ndim1 != ndim2) {
        error(155, 3, gbl.lineno, "Implied-shape array must be initialized "
              "with an array of the same rank -", SYMNAME(sptr));
        DTY(dtype + 1) = DT_NONE;
        return ;
      }
      zbase = 0;
      for (i = 0; i < ndim1; i++) {
        lb2 = ad_val_of(sym_of_ast(AD_LWAST(ad2, i)));
        ub2 = ad_val_of(sym_of_ast(AD_UPAST(ad2, i)));
        lb1 = ad_val_of(sym_of_ast(AD_LWAST(ad, i)));
        if (ADD_LWAST(dtype2, i) == ADD_LWAST(dtype, i)) {
          AD_UPBD(ad, i) = AD_UPAST(ad, i) = AD_UPAST(ad2, i);
          AD_EXTNTAST(ad, i) = AD_EXTNTAST(ad2, i);
        } else {
          ub1 = ub2 - lb2  + lb1;
          AD_UPBD(ad, i) = AD_UPAST(ad, i) =
            mk_bnd_int(mk_isz_cval(ub1, astb.bnd.dtype));
          AD_EXTNTAST(ad, i) =
            mk_shared_extent(AD_LWAST(ad, i), AD_UPAST(ad, i), i);
        }
        if (i == 0)
          zbase = zbase + lb1;
        else
          zbase = zbase + lb1 * (ub2 - lb2 + 1);
        AD_MLPYR(ad, i) = AD_MLPYR(ad2, i);
      }
      AD_ZBASE(ad) = mk_isz_cval(zbase, astb.bnd.dtype);
      if (i == ndim1)
        AD_MLPYR(ad, i) = AD_MLPYR(ad2, i);
      AD_ASSUMSZ(ad) = 0;
    }
  param_alias:
    sptr1 = get_param_alias_var(sptr, dtype);
    STYPEP(sptr1, ST_ARRAY);
    if (sem.interface == 0) {
      init_named_array_constant(sptr, gbl.currsub);
    }
  } else {
    get_param_alias_const(init, sptr, dtype);

    sdtype = DTYPEG(sptr);
    if (DDTG(sdtype) == DT_ASSCHAR || DDTG(sdtype) == DT_ASSNCHAR ||
        DDTG(sdtype) == DT_DEFERCHAR || DDTG(sdtype) == DT_DEFERNCHAR) {
      set_string_type_from_init(sptr, SST_ACLG(init));
    }
  }
}

static void
save_typedef_init(int sptr, int dtype)
{
  ACL *ict = NULL;

  if (!stsk->ict_beg) {
    DCLDP(DTY(dtype + 3), TRUE); /* "complete" tag declaration */
    /* Pop out to parent structure (if any) */
    sem.stsk_depth--;
    stsk = &STSK_ENT(0);
    return;
  }

  if (sem.stsk_depth == 1 && stsk->ict_beg != NULL) {
    /* This is the outer most structure, fix up top subc ict entry */
    ict = GET_ACL(15);
    ict->id = AC_TYPEINIT;
    ict->next = NULL;
    ict->subc = stsk->ict_beg;
    ict->repeatc = astb.i1;
    ict->sptr = sptr;
    ict->dtype = dtype;
    stsk->ict_beg = ict;
  }
  df_dinit(NULL, ict);
  DTY(dtype + 5) = put_getitem_p(stsk->ict_beg);

  DCLDP(DTY(dtype + 3), TRUE); /* "complete" tag declaration */

  /* Pop out to parent structure (if any) */
  sem.stsk_depth--;
  stsk = &STSK_ENT(0);

}

void
build_typedef_init_tree(int sptr, int dtype)
{
  ACL *ict;
  ACL *ict1;
  int td_dtype;

  td_dtype = DDTG(dtype);

  ict1 = (ACL *)get_getitem_p(DTY(td_dtype + 5));
  if (ict1) {
    /* Need to build an initializer constant tree */
    ict = GET_ACL(15);
    *ict = *ict1;
    ict->sptr = sptr;
    if (DTY(DTYPEG(sptr)) == TY_ARRAY)
      ict->repeatc = AD_NUMELM(AD_PTR(sptr));
    else
      ict->repeatc = astb.i1;
    if (ict->sptr)
      save_struct_init(ict);
    if (INSIDE_STRUCT) {
      if (stsk->ict_end)
        stsk->ict_end->next = ict;
      else
        stsk->ict_beg = ict;
      stsk->ict_end = ict;
    } else {
      /* For initialized sptr, don't create init list */
      if (DINITG(sptr) && DTY(td_dtype) == TY_DERIVED && !SAVEG(sptr))
        return;

      dinit_no_dinitp((VAR *)NULL, ict);
    }
  }
}

static void
init_allocatable_typedef_components(SPTR td_sptr)
{
  DTYPE td_dtype = DTYPEG(td_sptr);
  SPTR sptr = 0;
  SPTR fld_sptr;
  ACL *td_aclp;
  ACL **aclpp;
  int init_ict = get_struct_initialization_tree(td_dtype);

  if (init_ict) {
    td_aclp = get_getitem_p(init_ict);
  } else {
    td_aclp = GET_ACL(15);
    td_aclp->id = AC_TYPEINIT;
    td_aclp->sptr = td_sptr;
    td_aclp->dtype = td_dtype;
  }
  aclpp = &td_aclp->subc;

  for (fld_sptr = DTY(td_dtype + 1); fld_sptr > NOSYM;
       fld_sptr = SYMLKG(fld_sptr)) {
    ACL *aclp = NULL;
    DTYPE fld_dtype = DTYPEG(fld_sptr);
    if (is_array_dtype(fld_dtype))
      fld_dtype = array_element_dtype(fld_dtype);

    /* position the init list ptr */
    if (*aclpp) {
      for (sptr = td_sptr;
           sptr > NOSYM && sptr != fld_sptr && sptr != (*aclpp)->sptr;
           sptr = SYMLKG(sptr))
        continue;
      if (sptr == (*aclpp)->sptr) {
        /* this field already has an initializer */
        aclpp = &(*aclpp)->next;
        continue;
      }
    }

    if (DTY(fld_dtype) == TY_DERIVED && ALLOCFLDG(sptr)) {
      init_allocatable_typedef_components(fld_sptr);
      aclp = get_getitem_p(get_struct_initialization_tree(fld_dtype));
    } else if (ALLOCATTRG(fld_sptr)) {
      aclp = mk_init_intrinsic(AC_I_null);
    }
    if (aclp) {
      aclp->sptr = MIDNUMG(fld_sptr);
      aclp->next = *aclpp;
      *aclpp = aclp;
      aclpp = &aclp->next;
    }
  }

  df_dinit(NULL, td_aclp);
  if (!init_ict) { /* this is the "initialization tree" field */
    DTY(td_dtype + 5) = put_getitem_p(td_aclp);
  }
}

static void
symatterr(int sev, int sptr, const char *att)
{
  char buf[100];
  snprintf(buf, sizeof buf, "Attribute '%s' cannot be applied to symbol", att);
  buf[sizeof buf - 1] = '\0'; /* Windows snprintf bug workaround */
  error(155, sev, gbl.lineno, buf, SYMNAME(sptr));
}

static void
fixup_function_return_type(int retdtype, int dtsptr)
{
  dtsptr = lookupsymbol(SYMNAME(dtsptr));
  if (dtsptr && dtsptr != DTY(retdtype + 3)) {
    DTYPEP(gbl.currsub, DTYPEG(dtsptr));
    DTYPEP(FVALG(gbl.currsub), DTYPEG(dtsptr));
  } else if (sem.pgphase > PHASE_SPEC) {
    error(4, 3, FUNCLINEG(gbl.currsub),
          "Function return type has not been declared", CNULL);
    DTYPEP(gbl.currsub, DTYPEG(dtsptr));
    DTYPEP(FVALG(gbl.currsub), DTYPEG(dtsptr));
  }
}

static int
fixup_KIND_expr(int ast)
{
  int newast = ast;
  int tmp_ast1 = 0;
  int tmp_ast2 = 0;
  int sptr;
  int newsptr;
  int ndim;
  int subs[MAXRANK];
  int argt;
  int i;
  int changed;

  switch (A_TYPEG(ast)) {
  case A_CNST:
    if (DT_ISREAL(A_DTYPEG(ast))) {
      newast = mk_convert(ast, DT_INT);
    }
    break;
  case A_SUBSCR: /* NECESSARY? */
    sptr = A_SPTRG(A_LOPG(ast));
    ndim = ADD_NUMDIM(DTYPEG(sptr));
    argt = A_ARGSG(ast);
    changed = tmp_ast1 = fixup_KIND_expr(A_LOPG(ast));
    tmp_ast1 = tmp_ast1 ? tmp_ast1 : A_LOPG(ast);
    for (i = 0; i < ndim; i++) {
      changed |= tmp_ast2 = fixup_KIND_expr(ARGT_ARG(argt, i));
      subs[i] = tmp_ast2 ? tmp_ast2 : ARGT_ARG(argt, i);
    }
    if (changed) {
      newast = mk_subscr(tmp_ast1, subs, ndim, A_DTYPEG(ast));
    }
    break;
  case A_MEM: /* NECESSARY? */
    tmp_ast1 = fixup_KIND_expr(A_PARENTG(ast));
    tmp_ast2 = fixup_KIND_expr(A_MEMG(ast));
    if (tmp_ast1 || tmp_ast2) {
      tmp_ast1 = tmp_ast1 ? tmp_ast1 : A_LOPG(ast);
      tmp_ast2 = tmp_ast2 ? tmp_ast2 : A_ROPG(ast);
      newast = mk_member(tmp_ast1, tmp_ast2, A_DTYPEG(ast));
    }
    break;
  case A_UNOP:
    tmp_ast1 = fixup_KIND_expr(A_LOPG(ast));
    if (tmp_ast1) {
      newast = mk_unop(A_OPTYPEG(ast), tmp_ast1, A_DTYPEG(ast));
    }
    break;
  case A_BINOP:
    tmp_ast1 = fixup_KIND_expr(A_LOPG(ast));
    tmp_ast2 = fixup_KIND_expr(A_ROPG(ast));
    if (tmp_ast1 || tmp_ast2) {
      tmp_ast1 = tmp_ast1 ? tmp_ast1 : A_LOPG(ast);
      tmp_ast2 = tmp_ast2 ? tmp_ast2 : A_ROPG(ast);
      newast = mk_binop(A_OPTYPEG(ast), tmp_ast1, tmp_ast2, DT_INT);
    }
    break;
  case A_FUNC:
    /* could be an subscripted array expr */
    sptr = findByNameStypeScope(SYMNAME(A_SPTRG(A_LOPG(ast))), ST_PARAM, 0);
    if (sptr && DTY(DTYPEG(sptr)) == TY_ARRAY) {
      tmp_ast1 = mk_id(CONVAL1G(sptr));
      ndim = ADD_NUMDIM(DTYPEG(sptr));
      if (ndim != A_ARGCNTG(ast))
        break;
      argt = A_ARGSG(ast);
      for (i = 0; i < ndim; i++) {
        subs[i] = ARGT_ARG(argt, i);
      }
      newast = mk_subscr(tmp_ast1, subs, ndim, DTYPEG(sptr));
    }
    break;
  case A_ID:
    sptr = A_SPTRG(ast);
    if (!SCOPEG(sptr) || sem.pgphase == PHASE_USE) {
      newsptr = findByNameStypeScope(SYMNAME(A_SPTRG(ast)), ST_PARAM, 0);
      if (newsptr != sptr) {
        if (STYPEG(newsptr) == ST_CONST) {
          /* MORE can this happen, A_ID&ST_CONST */
          newast = mk_cnst(newsptr);
        } else if (STYPEG(newsptr) == ST_PARAM) {
          newast = CONVAL2G(newsptr);
        } else {
          newast = 0;
        }
      }
    }
    break;
  }
  return newast;
}

static int
eval_KIND_expr(int ast, int *val, int *dtyp)
{
  int val1;
  int val2;
  int tmp_ast1;
  int success = 0;

  if (!ast)
    return 0;

  if (A_ALIASG(ast)) {
    *dtyp = A_DTYPEG(ast);
    ast = A_ALIASG(ast);
  }

  switch (A_TYPEG(ast)) {
  case A_CNST:
    *dtyp = A_DTYPEG(ast);
    *val = CONVAL2G(A_SPTRG(ast));
    success = 1;
    break;
  case A_UNOP:
    if (eval_KIND_expr(A_LOPG(ast), &val1, dtyp)) {
      if (A_OPTYPEG(ast) == OP_SUB)
        *val = negate_const(val1, A_DTYPEG(ast));
      if (A_OPTYPEG(ast) == OP_LNOT)
        *val = ~(val1);
      *dtyp = A_DTYPEG(ast);
      success = 1;
    }
    break;
  case A_BINOP:
    if (eval_KIND_expr(A_LOPG(ast), &val1, dtyp) &&
        eval_KIND_expr(A_ROPG(ast), &val2, dtyp)) {
      *val = const_fold(A_OPTYPEG(ast), val1, val2, A_DTYPEG(ast));
      *dtyp = A_DTYPEG(ast);
      success = 1;
    }
    break;
  case A_SUBSCR:
  case A_MEM:
    tmp_ast1 = complex_alias(ast);
    if (eval_KIND_expr(tmp_ast1, &val1, dtyp)) {
      *val = val1;
      success = 1;
    }
    break;
  }

  return success;
}

static void
get_retval_KIND_value()
{
  int sptr;
  int sav_gbl_lineno = gbl.lineno;
  int val = -1;
  int dtyp;
  int l_ast1;

  gbl.lineno = sem.deferred_kind_len_lineno;

  /* Handle deferred KIND spec */
  if (A_TYPEG(sem.deferred_func_kind) == A_ID) {
    sptr = findByNameStypeScope(SYMNAME(A_SPTRG(sem.deferred_func_kind)),
                                ST_PARAM, 0);
    if (sptr) {
      dtyp = DTYPEG(sptr);
      val = CONVAL1G(sptr);
      if (STYPEG(A_SPTRG(sem.deferred_func_kind)) == ST_UNKNOWN) {
        IGNOREP(A_SPTRG(sem.deferred_func_kind), TRUE);
        HIDDENP(A_SPTRG(sem.deferred_func_kind), TRUE);
      }
    }
  } else if (A_ISEXPR(A_TYPEG(sem.deferred_func_kind))) {
    l_ast1 = fixup_KIND_expr(sem.deferred_func_kind);
    if (!eval_KIND_expr(l_ast1, &val, &dtyp)) {
      val = -1;
    }
  }

  if (val < 0) {
    errsev(87);
    goto exit;
  }

  if (dtyp != DT_INT4) {
    errwarn(91);
    goto exit;
  }

  if ((dtyp =
           select_kind(DTYPEG(gbl.currsub), DTY(DTYPEG(gbl.currsub)), val))) {
    DTYPEP(gbl.currsub, dtyp);
    DTYPEP(FVALG(gbl.currsub), dtyp);
    if ((sptr = findByNameStypeScope(SYMNAME(gbl.currsub), ST_ALIAS, 0))) {
      DTYPEP(sptr, dtyp);
    }
  }

exit:
  gbl.lineno = sav_gbl_lineno;
  sem.deferred_func_kind = 0;
  if (!sem.deferred_func_len)
    sem.deferred_kind_len_lineno = 0;
}

static void
get_retval_LEN_value()
{
  int sav_gbl_lineno = gbl.lineno;
  int dtyp = 0;
  int l_ast1;

  gbl.lineno = sem.deferred_kind_len_lineno;

  /* Handle deferred LEN spec */
  l_ast1 = fixup_KIND_expr(sem.deferred_func_len);
  if (A_TYPEG(l_ast1) == A_CNST) {
    dtyp = mod_type(sem.ogdtype, DTY(sem.ogdtype), 1, CONVAL2G(A_SPTRG(l_ast1)),
                    0, gbl.currsub);
    if (dtyp) {
      DTYPEP(gbl.currsub, dtyp);
      DTYPEP(FVALG(gbl.currsub), dtyp);
    }
  } else {
    dtyp = mod_type(sem.ogdtype, DTY(sem.ogdtype), 4, l_ast1, 0, gbl.currsub);
    if (dtyp) {
      DTYPEP(gbl.currsub, dtyp);
      ADJLENP(gbl.currsub, 1);
      DTYPEP(FVALG(gbl.currsub), dtyp);
      ADJLENP(FVALG(gbl.currsub), 1);
    }
  }

  gbl.lineno = sav_gbl_lineno;
  sem.deferred_func_len = 0;
  sem.deferred_kind_len_lineno = 0;
}

static void
get_retval_derived_type()
{
  int sptr;
  LOGICAL found = FALSE;

  if (gbl.rutype == RU_FUNC) {
    for (sptr = first_hash(sem.deferred_dertype); sptr > NOSYM;
         sptr = HASHLKG(sptr)) {
      if (sptr == sem.deferred_dertype)
        continue;
      if (STYPEG(sptr) == ST_TYPEDEF && STYPEG(SCOPEG(sptr)) == ST_MODULE) {
        pop_sym(sem.deferred_dertype);
        found = TRUE;
        break;
      }
    }
  }

  if (found) {
    DTYPEP(gbl.currsub, DTYPEG(sptr));
    DTYPEP(FVALG(gbl.currsub), DTYPEG(sptr));
  } else {
    error(155, 3, sem.deferred_kind_len_lineno,
          "Derived type has not been declared -",
          SYMNAME(sem.deferred_dertype));
  }

  sem.deferred_dertype = 0;
  sem.deferred_kind_len_lineno = 0;
}

static void
process_bind(int sptr)
{
  int b_type;
  int b_bitv;
  int need_altname = 0;
  char *np;

  /* A module routine without an explicit C name uses the routine name. */
  if (!XBIT(58,0x200000)) {
    if ((bind_attr.exist & DA_B(DA_C)) &&
        !bind_attr.altname && INMODULEG(sptr) &&
        (STYPEG(sptr) == ST_PROC || STYPEG(sptr) == ST_ENTRY)) {
      char *np = SYMNAME(sptr);
      bind_attr.exist |= DA_B(DA_ALIAS);
      bind_attr.altname = getstring(np, strlen(np));
    }
  }

  b_type = 0;
  for (b_bitv = bind_attr.exist; b_bitv; b_bitv >>= 1, b_type++) {

    if ((b_bitv & 1) == 0)
      continue;

    switch (b_type) {
    case DA_ALIAS:
      /* An altname can't be empty.  Exit early to use a "normal" mangled
       * variant of the primary symbol name. */
      np = stb.n_base + CONVAL1G(bind_attr.altname);
      if (!*np)
        return;
      ALTNAMEP(sptr, bind_attr.altname);
      break;
    case DA_C:

#if defined(TARGET_OSX)
      /* add underscore to OSX common block names */
      if (STYPEG(sptr) == ST_CMBLK)
        need_altname = 1;
#endif
      /* NEW CFUNCP and REFERENCEP */
      CFUNCP(sptr, 1);
      if ((STYPEG(sptr) == ST_PROC) || (STYPEG(sptr) == ST_ENTRY)) {
        PASSBYREFP(sptr, 1);
        MSCALLP(sptr, 0);
      }

      break;
    } /* end switch */

  } /* end for */

  if ((need_altname) && ALTNAMEG(sptr) == 0) {
    /* set default altname, so that no underbar gets added */
    ALTNAMEP(sptr, getstring(SYMNAME(sptr), strlen(SYMNAME(sptr))));
  }
} /* process_bind */

static void
clear_ident_list()
{
  IDENT_LIST *curr, *curr_next;
  IDENT_PROC_LIST *curr_proc, *curr_proc_next;
  long hashval;

  if (!sem.which_pass || !dirty_ident_base || gbl.internal > 1) {
    return;
  }

  for (hashval = 0; hashval < HASHSIZE; ++hashval) {
    for (curr = ident_base[hashval]; curr;) {
      for (curr_proc = curr->proc_list; curr_proc;) {
        curr_proc_next = curr_proc->next;
        FREE(curr_proc);
        curr_proc = curr_proc_next;
      }
      curr->proc_list = 0;
      curr_next = curr->next;
      FREE(curr);
      curr = curr_next;
    }
    ident_base[hashval] = 0;
  }

  dirty_ident_base = FALSE;
}

/** \brief Emit a warning if a duplicate subproblem prefix is used.
 */
static void
check_duplicate(bool checker, const char *op)
{
  if (checker)
   error(1054, ERR_Warning, gbl.lineno, op, NULL); 
}

/** \brief Reset subprogram prefixes to zeroes
 */
static void 
clear_subp_prefix_settings(struct subp_prefix_t *subp)
{
  BZERO(subp, struct subp_prefix_t, 1);
}

/** \brief MODULE prefix checking for subprograms
           C1547: cannot be inside a an abstract interface 
 */
static void
check_module_prefix()
{
  if (sem.interface && subp_prefix.module && 
      sem.interf_base[sem.interface - 1].abstract)
    error(1055, ERR_Severe, gbl.lineno, NULL, NULL);
}

static void
decr_ident_use(int ident, int proc)
{
  long hashval;
  IDENT_LIST *curr;
  IDENT_PROC_LIST *curr_proc;

  if (sem.which_pass || !dirty_ident_base || gbl.internal <= 1) {
    return;
  }
  HASH_STR(hashval, SYMNAME(ident), strlen(SYMNAME(ident)))
  for (curr = ident_base[hashval]; curr; curr = curr->next) {
    if (strcmp(curr->ident, SYMNAME(ident)) == 0) {
      for (curr_proc = curr->proc_list; curr_proc;
           curr_proc = curr_proc->next) {
        if (strcmp(SYMNAME(proc), curr_proc->proc_name) == 0) {
          curr_proc->usecnt -= 1;
        }
      }
    }
  }
}

static void
defer_ident_list(int ident, int proc)
{

  long hashval;
  IDENT_LIST *curr;
  IDENT_PROC_LIST *curr_proc;

  if (STYPEG(ident) && SCOPEG(ident) == gbl.currsub && SCOPEG(ident) != proc) {
    /* Note: if STYPEG(ident) == 0, then this is an implicitly defined symbol */
    proc = SCOPEG(ident);
  }
  HASH_STR(hashval, SYMNAME(ident), strlen(SYMNAME(ident)));
  for (curr = ident_base[hashval]; curr; curr = curr->next) {
    if (strcmp(curr->ident, SYMNAME(ident)) != 0)
      continue;
    for (curr_proc = curr->proc_list; curr_proc; curr_proc = curr_proc->next) {
      if (strcmp(SYMNAME(proc), curr_proc->proc_name) == 0) {
        curr_proc->usecnt += 1;
        return; /* identifier and procedure already added */
      }
    }
    /* add procedure name */
    dirty_ident_base = TRUE;
    NEW(curr_proc, IDENT_PROC_LIST, 1);
    NEW(curr_proc->proc_name, char, strlen(SYMNAME(proc)) + 1);
    strcpy(curr_proc->proc_name, SYMNAME(proc));
    curr_proc->next = curr->proc_list;
    curr->proc_list = curr_proc;
    curr_proc->usecnt = 1;
    return;
  }
  /* add identifier and create new procedure list */
  NEW(curr, IDENT_LIST, 1);
  NEW(curr->ident, char, strlen(SYMNAME(ident)) + 1);
  strcpy(curr->ident, SYMNAME(ident));
  NEW(curr_proc, IDENT_PROC_LIST, 1);
  NEW(curr_proc->proc_name, char, strlen(SYMNAME(proc)) + 1);
  strcpy(curr_proc->proc_name, SYMNAME(proc));
  curr->proc_list = curr_proc;
  curr_proc->next = 0;
  curr_proc->usecnt = 1;
  curr->next = ident_base[hashval];
  ident_base[hashval] = curr;
  dirty_ident_base = TRUE;
}

int
internal_proc_has_ident(int ident, int proc)
{
  long hashval;
  IDENT_LIST *curr;
  IDENT_PROC_LIST *curr_proc;

  if (!dirty_ident_base)
    return 0;

  HASH_STR(hashval, SYMNAME(ident), strlen(SYMNAME(ident)));
  for (curr = ident_base[hashval]; curr; curr = curr->next) {
    if (strcmp(curr->ident, SYMNAME(ident)) == 0) {
      for (curr_proc = curr->proc_list; curr_proc;
           curr_proc = curr_proc->next) {
        if (strcmp(curr_proc->proc_name, SYMNAME(proc)) == 0 &&
            curr_proc->usecnt > 0) {
          return 1;
        }
      }
    }
  }
  return 0;
}

#ifdef GSCOPEP
static void
prop_reqgs(int ast)
{
  switch (A_TYPEG(ast)) {
  case A_ID:
    GSCOPEP(A_SPTRG(ast), 1);
    break;
  case A_SUBSCR:
  case A_SUBSTR:
  case A_UNOP:
    prop_reqgs(A_LOPG(ast));
    break;
  case A_MEM:
    prop_reqgs(A_PARENTG(ast));
    break;
  case A_BINOP:
    prop_reqgs(A_LOPG(ast));
    prop_reqgs(A_ROPG(ast));
    break;
  }
}

static void
fixup_ident_bounds(int sptr)
{
  int dtype, numdim, i;
  ADSC *ad;

  if (GSCOPEG(sptr)) {
    dtype = DTYPEG(sptr);
    if (DTY(dtype) != TY_ARRAY)
      return;
    ad = AD_DPTR(dtype);
    numdim = AD_NUMDIM(ad);
    prop_reqgs(AD_NUMELM(ad));
    prop_reqgs(AD_ZBASE(ad));
    for (i = 0; i < numdim; ++i) {
      prop_reqgs(AD_LWAST(ad, i));
      prop_reqgs(AD_UPAST(ad, i));
      prop_reqgs(AD_EXTNTAST(ad, i));
      prop_reqgs(AD_MLPYR(ad, i));
    }
  }
}

void
fixup_reqgs_ident(int sptr)
{
  if (GSCOPEG(sptr)) {
    if (SDSCG(sptr)) { 
      GSCOPEP(SDSCG(sptr), 1);
    }
    if (PTRVG(sptr)) {
      GSCOPEP(PTRVG(sptr), 1);
    }
    if (MIDNUMG(sptr)) {
      GSCOPEP(MIDNUMG(sptr), 1);
    }
    if (DTY(DTYPEG(sptr)) == TY_ARRAY) {
      fixup_ident_bounds(sptr);
    }
  }
}

#endif

static void
defer_iface(int iface, int dtype, int proc, int mem)
{
  int pass, len;
  iface_avail++;
  NEED(iface_avail, iface_base, IFACE, iface_size, iface_avail + 50);
  iface_base[iface_avail - 1].iface = iface;
  iface_base[iface_avail - 1].dtype = dtype;
  iface_base[iface_avail - 1].proc = proc;
  iface_base[iface_avail - 1].scope = SCOPEG(mem);
  iface_base[iface_avail - 1].internal = gbl.internal;
  /* Need to save which sem pass created this iface */
  iface_base[iface_avail - 1].sem_pass = sem.which_pass;

  len = strlen(SYMNAME(iface)) + 1;
  NEW(iface_base[iface_avail - 1].iface_name, char, len);
  strcpy(iface_base[iface_avail - 1].iface_name, SYMNAME(iface));

  iface_base[iface_avail - 1].tag_name = 0;
  iface_base[iface_avail - 1].pass_class = 0;

  if (mem && STYPEG(mem) == ST_MEMBER) {
    iface_base[iface_avail - 1].mem = mem;
    pass = PASSG(mem);
    if (pass && DTYPEG(pass) != stsk->dtype) {
      /* assume dtype of pass argument is same as enclosed dtype.
       * We do this since PASS will get written to a module before
       * we can fix it after we've seen the procedure/interface.
       * If the pass argument differs from enclosed dtype, we will
       * catch it in do_iface().
       */
      DTYPEP(pass, stsk->dtype);
    }

  } else {
    iface_base[iface_avail - 1].mem = 0;
  }
  
  iface_base[iface_avail - 1].proc_var = mem;
  iface_base[iface_avail - 1].lineno = gbl.lineno;
}

/** \brief This routine sets the PASS field in a procedure pointer for
  * semantic pass 0 prior to call to end_module().
  *
  * This is needed, otherwise we may incorrectly write the procedure pointer
  * module info without PASS set.
  */
static void
fix_iface0()
{
  int i, iface, mem;
  char *name;

  if (sem.which_pass)
    return;

  for (i = 0; i < iface_avail; i++) {
    mem = iface_base[i].mem;
    name = iface_base[i].iface_name;

    if (!name || !mem)
      continue;
    iface = findByNameStypeScope(name, ST_PROC, 0);
    iface_base[i].stype = STYPEG(iface); /* need to save stype */
    if (iface && !PASSG(mem) && !NOPASSG(mem)) {
      int arg_sptr = aux.dpdsc_base[DPDSCG(iface)];
      PASSP(mem, arg_sptr);
    }
  }
}

static void
fix_iface(int sptr)
{
  int len, tag, i, iface, proc, mem, dtype;
  int *dscptr;
  char *name;

  for (i = 0; i < iface_avail; i++) {
    iface = iface_base[i].iface;
    proc = iface_base[i].proc;
    mem = iface_base[i].mem;
    name = iface_base[i].iface_name;
    dtype = iface_base[i].dtype;
    if (!iface && mem && dtype && !NOPASSG(mem) &&
        strcmp(name, SYMNAME(sptr)) == 0) {
      iface = sptr;
      iface_base[i].iface = sptr;
    }
    if (iface && sptr && strcmp(name, SYMNAME(sptr)) == 0) {
      iface_base[i].iface = sptr;
      if (!PASSG(mem) && !NOPASSG(mem)) {
        dscptr = aux.dpdsc_base + DPDSCG(iface);
        PASSP(mem, *dscptr);
      } else if (PASSG(mem)) {
        int j = find_dummy_position(iface, PASSG(mem));
        if (j > 0)
          PASSP(mem, aux.dpdsc_base[DPDSCG(iface) + j - 1]);
      }
#ifdef CLASSG
      if (CLASSG(PASSG(mem))) {
        iface_base[i].pass_class = 1;

        tag = DTYPEG(PASSG(mem));
        tag = DTY(tag + 3);

        len = strlen(SYMNAME(tag)) + 1;
        NEW(iface_base[iface_avail - 1].tag_name, char, len);
        strcpy(iface_base[iface_avail - 1].tag_name, SYMNAME(tag));
      }
#endif
    }
  }
}

/* Called during sem pass 0 at the end of the subroutine/function. We attempt
 * to share compatible procedure pointer dtypes found in argument descriptors.
 * This fixes a problem exhibited in the Whizard code where we perform an
 * argument check on a call to a forward referenced internal procedure. In
 * this case, the argument's DT_PROC dtype has not yet been seen.
 */
static void
fix_proc_ptr_dummy_args()
{

  int paramct, dpdsc, i;

  if (sem.which_pass)
    return;
  proc_arginfo(gbl.currsub, &paramct, &dpdsc, NULL);
  for (i = 0; i < paramct; ++i) {
    int sptr = aux.dpdsc_base[dpdsc + i];
    if (is_procedure_ptr(sptr) && SCG(sptr) == SC_DUMMY) {
      char *symname = SYMNAME(sptr);
      int len = strlen(symname);
      int hash, hptr;
      HASH_ID(hash, symname, len);
      for (hptr = stb.hashtb[hash]; hptr; hptr = HASHLKG(hptr)) {
        if (is_procedure_ptr(hptr) && strcmp(symname, SYMNAME(hptr)) == 0) {
          if (hptr != sptr && test_scope(hptr) >= 0) {
            DTYPE d1 = DTYPEG(sptr);
            DTYPE d2 = DTYPEG(hptr);
            if (cmp_interfaces(DTY(d1 + 2), DTY(d2 + 2), TRUE)) {
              DTYPEP(sptr, d2);
              break;
            }
          }
        }
      }
    }
  }
}

static void
do_iface(int iface_state)
{
  int i;
  for (i = 0; i < iface_avail; i++) {
    _do_iface(iface_state, i);
  }
  if (iface_state) {
    iface_avail = 0;
  } 
}

static void
do_iface_module(void)
{
  /*
   * processing interfaces while in a module-contained subprogram;
   * need to process those interfaces which are not module procedures.
   */
  int i;
  int iface;
  assert(IN_MODULE, "must be in module", 0, ERR_Fatal);
  if (sem.interface && !get_seen_contains()) {
    /* in an interface block in a module specification, if the iface is from
     * this module, defer until the end of the module
     */
    for (i = 0; i < iface_avail; i++) {
      iface = iface_base[i].iface;
      if ((!iface || STYPEG(iface) == ST_UNKNOWN) && !sem.which_pass)
        continue;
      _do_iface(/*1*/ sem.which_pass, i);
      iface_base[i].iface = 0;
    }
  } else {
    if (!gbl.currsub) {
      /* IN_MODULE_SPEC */
      for (i = 0; i < iface_avail; i++) {
        iface = iface_base[i].iface;
        if (!iface)
          continue;
        switch (STYPEG(iface)) {
        case ST_UNKNOWN:
        case ST_MODPROC:
          continue;
        case ST_ALIAS:
          if (SCOPEG(iface) == gbl.currmod)
            continue;
          break;
        default:;
        }
        _do_iface(/*1*/ sem.which_pass, i);
        iface_base[i].iface = 0;
      }
    }
    for (i = 0; i < iface_avail; i++) {
      iface = iface_base[i].iface;
      if (iface) {
        int scp;
        scp = SCOPEG(iface);
        if (scp && (scp == gbl.currsub || scp == SCOPEG(gbl.currsub)) &&
            !INMODULEG(iface)) {
          _do_iface(1, i);
          iface_base[i].iface = 0;
        } else if (sem.which_pass) {
          switch (STYPEG(iface)) {
          case ST_MODPROC:
          case ST_ALIAS:
            break;
          default:
            if (scp == gbl.currmod) {
              _do_iface(sem.which_pass, i);
              iface_base[i].iface = 0;
            } else if (scp != gbl.currmod && NEEDMODG(scp)) {
              _do_iface(sem.which_pass, i);
              iface_base[i].iface = 0;
            }
          }
        } else if (gbl.currsub && scp &&
                   (!INMODULEG(iface) || ABSTRACTG(iface))) {
          switch (STYPEG(iface)) {
          case ST_MODPROC:
          case ST_ALIAS:
            break;
          default:
            if (scp == ENCLFUNCG(gbl.currsub)) {
              _do_iface(1, i);
              iface_base[i].iface = 0;
            } else if (scp != SCOPEG(gbl.currsub)) {
              _do_iface(1, i);
              iface_base[i].iface = 0;
            }
          }
        }
      }
    }
  }
}

/**
 * Called by _do_iface() as part of error clean-up. We need to clear the
 * next attempt to use an erroneous interface specified in the iface argument
 * starting at the "i + 1" element in iface_base.
 */
static void
clear_iface(int i, SPTR iface)
{
    int j;

    for (j = i + 1; j < iface_avail; j++) {
      if (iface_base[j].iface &&
          sem_strcmp(SYMNAME(iface), SYMNAME(iface_base[j].iface)) == 0) {
        /* inhibit the next attempt to use the same interface */
        iface_base[j].iface = 0;
      }
    }
}

static void
_do_iface(int iface_state, int i)
{
  SPTR sptr, orig, fval;
  int dpdsc, paramct = 0;
  LOGICAL pass_notfound;
  SPTR passed_object; /* passed-object dummy argument */
  SPTR iface = iface_base[i].iface;
  SPTR ptr_scope = iface_base[i].scope;
  const char *name = iface_base[i].iface_name;
  DTYPE dtype = iface_base[i].dtype;
  SPTR proc = iface_base[i].proc;
  SPTR mem = iface_base[i].mem;
  int lineno = iface_base[i].lineno;
  LOGICAL class = iface_base[i].pass_class;
  const char *dt_name = iface_base[i].tag_name;
  SPTR proc_var = iface_base[i].proc_var;
  int internal = iface_base[i].internal;

  if (!iface) {
    return;
  }
 
  if (dtype > 0) { 
    if (DTY(dtype) == TY_ARRAY) {
      dtype = DTY(dtype + 1);
    }
    if (DTY(dtype) == TY_PTR) {
      dtype = DTY(dtype+1);
    }
    if (DTY(dtype) != TY_PROC) {
      return;
    }
  }

  if (ptr_scope && STYPEG(ptr_scope) != ST_MODULE &&
      ptr_scope != stb.curr_scope &&
      (gbl.internal <= 1 || (gbl.internal > 1 && gbl.outersub != ptr_scope))) {
    /* This procedure pointer is not in scope. So, we skip it to avoid
     * overwriting another dtype.
     */
    return;
  }

  if (internal > 1 && gbl.internal != internal) {
    /* This procedure variable/pointer was declared in an internal procedure 
     * that differs from the current procedure. So, skip it to avoid
     * overwriting it with another dtype.
     */
    return;
  }

  if (proc) {
    DTYPEP(proc, DTYPEG(iface));
  }
  if (!STYPEG(iface)) {
    if (sem.which_pass) {
      SPTR hptr;
      char *symname = SYMNAME(iface);
      int len = strlen(symname);
      int hash;
      HASH_ID(hash, symname, len);
      for (hptr = stb.hashtb[hash]; hptr; hptr = HASHLKG(hptr)) {
        if (STYPEG(hptr) == ST_PROC && strcmp(symname, SYMNAME(hptr)) == 0) {
          int scope = test_scope(hptr);
		
          if (scope && scope <= test_scope(iface)) {
            iface = hptr;
            break;
          }
        }
      }
      if (!STYPEG(iface)) {
        /* Check to see if we saw this iface in the first pass.
         * If so, do not generate an error.
         */
        int j;
        for (j = 0; j < iface_avail; j++) {
          if (iface_base[j].sem_pass == 0 &&
              strcmp(iface_base[j].iface_name, name) == 0 &&
              iface_base[j].stype == ST_PROC) {
            return;
          }
        }
        orig = iface;
        goto iface_err;
      }
    }
    if (proc <= NOSYM)
      return; 
  }
  if (strcmp(SYMNAME(iface), name) != 0)
    iface = getsymbol(name);
  if (sem.interface <= 1) {
    sptr = refsym(iface, OC_OTHER); 
  } else {
    sptr = refsym_inscope(iface, OC_OTHER);
  }
  if (DTY(dtype) == TY_PROC && STYPEG(DTY(dtype + 2)) == ST_MEMBER) {
    iface = sptr;
    DTY(dtype + 2) = iface;
  }
  if ((!sem.which_pass || STYPEG(sptr)) &&
      (STYPEG(iface) != ST_ENTRY || sptr != FVALG(iface))) {
    iface = sptr;
  }
  orig = iface;
  switch (STYPEG(iface)) {
  case ST_IDENT:
    if (RESULTG(iface)) /* Interface not seen yet */
      return;
    goto iface_err;
  case ST_GENERIC:
    iface = GSAMEG(iface);
    FLANG_FALLTHROUGH;
  case ST_INTRIN:
  case ST_PD:
    iface = iface_intrinsic(iface);
    if (!iface) {
      goto iface_err;
    }
    FLANG_FALLTHROUGH;
  case ST_ENTRY:
  case ST_PROC:
    paramct = PARAMCTG(iface);
    dpdsc = DPDSCG(iface);
    break;
  case ST_MEMBER:
    if (DTY(DTYPEG(iface)) == TY_PTR) {
      /* Procedure pointer that's a component of a derived type. */
      break;
    }
    goto iface_err;
  default:
  iface_err:
    if (!STYPEG(iface) &&
        (!sem.which_pass || iface_state == 0 ||
        (IN_MODULE && !sem.seen_end_module))) {
/* Do not generate error on semantic pass 0. May not have seen the
 * entire module yet. Return only if we have seen an IMPORT stmt.
 */
      return;
    }
    error(155, 3, lineno, "Illegal procedure interface -", SYMNAME(orig));
    clear_iface(i, orig);
    return;
  }
  if (ELEMENTALG(orig) && !IS_INTRINSIC(STYPEG(orig)) &&
      POINTERG(proc_var)) {
    error(1010, ERR_Severe, lineno, SYMNAME(proc_var), CNULL);
    clear_iface(i, orig);
  } 
  passed_object = 0;
  pass_notfound = mem && PASSG(mem);
  fval = FVALG(iface);
  if (paramct || fval) {
    SPTR *dscptr;
    int j;
    if (fval)
      dpdsc = ++aux.dpdsc_avl;
    else
      dpdsc = aux.dpdsc_avl;
    NEED(aux.dpdsc_avl + paramct, aux.dpdsc_base, int, aux.dpdsc_size,
         aux.dpdsc_size + paramct + 100);
    dscptr = aux.dpdsc_base + DPDSCG(iface);
    if (paramct && mem && !NOPASSG(mem) && !PASSG(mem)) {
      passed_object = *dscptr; /* passed-object default */
    }
    for (j = 0; j < paramct; j++) {
      SPTR arg = *dscptr++;
      aux.dpdsc_base[dpdsc + j] = arg;
      if (pass_notfound && sem_strcmp(SYMNAME(arg), SYMNAME(PASSG(mem))) == 0) {
        pass_notfound = FALSE;
        passed_object = arg;
      }
    }
    if (fval) {
      aux.dpdsc_base[dpdsc - 1] = fval;
      FUNCP(mem, TRUE);
    }
    aux.dpdsc_avl += paramct;
  } else {
    dpdsc = 0;
  }
  if (proc) {
    DTYPEP(proc, DTYPEG(iface));
    PARAMCTP(proc, paramct);
    DPDSCP(proc, dpdsc);
    FVALP(proc, fval);
    PUREP(proc, PUREG(iface));
    ELEMENTALP(proc, ELEMENTALG(iface));
    CFUNCP(proc, CFUNCG(iface)); 
  } else {
    /*  dtype locates the TY_PROC data type record  */
    if (mem && paramct == 0 && !NOPASSG(mem)) {
      error(155, 3, lineno, "NOPASS attribute must be present for",
            SYMNAME(mem));
      NOPASSP(mem, TRUE);
      passed_object = 0;
    }
    DTY(dtype + 1) = DTYPEG(iface);
    DTY(dtype + 2) = iface;
    DTY(dtype + 3) = paramct;
    DTY(dtype + 4) = dpdsc;
    DTY(dtype + 5) = fval;
    if (pass_notfound) {
      error(155, 3, lineno, "Passed-object dummy argument not found -",
            SYMNAME(PASSG(mem)));
    }
    if (passed_object && iface_state) {
      DTYPE dt;
      if (dt_name) {
        dt = DTYPEG(getsymbol(dt_name));
      } else
        dt = DTYPEG(passed_object);
      if (DTY(dt) != TY_DERIVED || DTY(dt + 3) == 0) {
        error(155, 3, lineno,
              "Passed-object dummy argument must be a derived type scalar -",
              SYMNAME(passed_object));
      } else {
        SPTR tdf = DTY(dt + 3);
        if (dt != ENCLDTYPEG(mem)) {
          error(155, 3, lineno,
                "Incompatible passed-object dummy argument for ",
                SYMNAME(iface));
        } else if (!SEQG(tdf) && !class) {
          error(155, 3, lineno,
                "Passed-object dummy argument is not polymorphic -",
                SYMNAME(passed_object));
        }
        if (POINTERG(passed_object) || ALLOCATTRG(passed_object))
          error(155, 3, lineno, "Passed-object dummy argument must not be "
                                "POINTER or ALLOCATABLE -",
                SYMNAME(passed_object));
      }
      PASSP(mem, passed_object); /* default or specified */
    }
  }
}

/** \brief Sets up type parameters used in parameterized derived types (PDTs)
  */
int
queue_type_param(int sptr, int dtype, int offset, int flag)
{

  /* linked list of type parameters for a particular derived type */
  typedef struct tp {
    char *name;      /* name of parameter */
    int dtype;       /* derived type holding this type parameter */
    int offset;      /* parameter's position in list parm list */
    struct tp *next; /* next record */
  } TP;

  static TP *tp_queue = 0;
  TP *prev, *curr, *new_tp;
  char *c;
  int tag, parent, mem, i;
  int prevmem, firstuse, parentuse;

  if (flag == 0) {
    /* init/clear entries */
    for (prev = curr = tp_queue; curr;) {
      FREE(curr->name);
      prev = curr;
      curr = curr->next;
      FREE(prev);
    }
    tp_queue = 0;
    return 1;
  } else if (flag == 1) {
    /* add entry */
    c = SYMNAME(sptr);

    /* step 1 - check for duplicate type parameter in this type */
    for (curr = tp_queue; curr; curr = curr->next) {
      if (curr->dtype == dtype && strcmp(curr->name, c) == 0) {
        error(155, 3, gbl.lineno, "Duplicate type parameter -", c);
        return 0;
      }
    }
    /* step 2 - add type parameter to queue */
    NEW(new_tp, TP, 1);
    BZERO(new_tp, TP, 1);

    NEW(new_tp->name, char, strlen(c) + 1);
    strcpy(new_tp->name, c);
    new_tp->dtype = dtype;
    new_tp->offset = offset;
    new_tp->next = tp_queue;
    tp_queue = new_tp;
    return 1;
  } else if (flag == 3) {
    tag = DTY(dtype + 3);
    parent = DTYPEG(PARENTG(tag));

    if (parent) {
      i = queue_type_param(sptr, parent, offset, 3);
      if (i)
        return i;
    }
    for (curr = tp_queue; curr; curr = curr->next) {
      if (curr->dtype == dtype) {
        c = curr->name;
        if (strcmp(c, SYMNAME(sptr)) == 0)
          return curr->offset;
      }
    }
    return 0;
  } else if (flag == 2) {
    /* fill in dtype into type param fields, check parent type params,
     * check to make sure defined params have corresponding components
     * in the the dtype, and reorder (if necessary) params.
     */
    for (curr = tp_queue; curr; curr = curr->next) {
      if (curr->dtype == 0)
        curr->dtype = dtype;
    }

    tag = DTY(dtype + 3);
    parent = DTYPEG(PARENTG(tag));

    if (parent) {
      for (curr = tp_queue; curr; curr = curr->next) {
        if (curr->dtype == dtype) {
          c = curr->name;
          for (mem = DTY(parent + 1); mem > NOSYM; mem = SYMLKG(mem)) {
            if (!USEKINDG(mem) && KINDG(mem) && strcmp(SYMNAME(mem), c) == 0) {
              error(155, 3, gbl.lineno, "Duplicate type parameter "
                                        "(in parent type) -",
                    c);
            }
          }
        }
      }
    }

    for (curr = tp_queue; curr; curr = curr->next) {
      if (curr->dtype == dtype) {
        c = curr->name;
        for (mem = DTY(dtype + 1); mem > NOSYM; mem = SYMLKG(mem)) {
          if (!USEKINDG(mem) && KINDG(mem) && strcmp(SYMNAME(mem), c) == 0) {
            KINDP(mem, curr->offset);
            break;
          }
        }
        if (mem <= NOSYM) {
          error(155, 3, gbl.lineno, "Missing type parameter specification -",
                c);
        }
      }
    }

    /* check for extraneous kind type parameters */

    for (mem = DTY(dtype + 1); mem > NOSYM; mem = SYMLKG(mem)) {
      if (!USEKINDG(mem) && KINDG(mem) == -1) {
        error(155, 3, gbl.lineno, "Kind type parameter component does not have "
                                  "a corresponding type parameter specifier -",
              SYMNAME(mem));
      }
    }

/* For now, place length type parameters at the beginning of the dtype
 * to improve processing of them later.
 * Also fix up recursively typed components.
 */

    firstuse = parentuse = 0;
    for (prevmem = mem = DTY(dtype + 1); mem > NOSYM;) {
      int bt;
      bt = DTYPEG(mem);
      if ((POINTERG(mem) || ALLOCATTRG(mem)) && DTY(bt) == TY_DERIVED) {
        bt = DTY(bt + 3);
        bt = BASETYPEG(bt);
        if (bt && bt == ENCLDTYPEG(mem)) {
          /* This is a recursively typed component. We need to set
           * this component's type to the enclosed type since this component
           * was added before the enclosed type was fully defined. Otherwise,
           * this component's type is incomplete and may not have all of its
           * components. Recursively typed components must have POINTER
           * attribute in F2003. In F2008, they can have POINTER or
           * ALLOCTABLE attribute.
           */
          DTYPEP(mem, bt);
        }
      }
      if (PARENTG(mem)) {
        parentuse = mem;
      } else if (!firstuse && !LENPARMG(mem) && USELENG(mem)) {
        firstuse = mem;
      } else if (firstuse && LENPARMG(mem)) {
        SYMLKP(prevmem, SYMLKG(mem));
        if (!parentuse) {
          SYMLKP(mem, DTY(dtype + 1));
          DTY(dtype + 1) = mem;
        } else {
          SYMLKP(mem, SYMLKG(parentuse));
          SYMLKP(parentuse, mem);
        }
        mem = SYMLKG(prevmem);
        continue;
      }
      prevmem = mem;
      mem = SYMLKG(mem);
    }

    /* ditto with kind type parameters */

    firstuse = parentuse = 0;
    for (prevmem = mem = DTY(dtype + 1); mem > NOSYM;) {
      if (PARENTG(mem)) {
        parentuse = mem;
      } else if (!firstuse && !LENPARMG(mem) && USEKINDG(mem) &&
                 A_TYPEG(KINDASTG(mem)) != A_CNST &&
                 A_TYPEG(KINDASTG(mem)) != A_ID) {
        firstuse = mem;
      } else if (firstuse && KINDG(mem) && !USEKINDG(mem) && !KINDASTG(mem)) {
        SYMLKP(prevmem, SYMLKG(mem));
        if (!parentuse) {
          SYMLKP(mem, DTY(dtype + 1));
          DTY(dtype + 1) = mem;
        } else {
          SYMLKP(mem, SYMLKG(parentuse));
          SYMLKP(parentuse, mem);
        }
        mem = SYMLKG(prevmem);
        continue;
      }
      prevmem = mem;
      mem = SYMLKG(mem);
    }

    return 1;
  }

  return 0;
}

static void
search_kind(int ast, int *offset)
{

  int sptr, rslt;

  if (!offset || *offset)
    return;
  if (A_TYPEG(ast) == A_ID) {
    sptr = A_SPTRG(ast);
    if (sptr) {
      rslt = queue_type_param(sptr, 0, 0, 3);
      if (!rslt && sem.stsk_depth && stsk == &STSK_ENT(0)) {
        rslt = get_kind_parm(sptr, stsk->dtype);
      }
      if (rslt) {
        *offset = rslt;
        return;
      }
    }
  }
}

static int
chk_kind_parm(SST *stkp)
{
  int offset;
  int sptr;
  int ast;

  sptr = 0;
  switch (SST_IDG(stkp)) {
  case S_IDENT:
    sptr = SST_SYMG(stkp);
    break;
  case S_LVALUE:
    sptr = SST_LSYMG(stkp);
    break;
  case S_EXPR:
    ast = SST_ASTG(stkp);
    offset = 0;
    ast_visit(1, 1);
    ast_traverse(ast, NULL, search_kind, &offset);
    ast_unvisit();
    return offset;
  }
  if (!sptr)
    return 0;
  /* Check to see if this is a kind type parameter */
  offset = queue_type_param(sptr, 0, 0, 3);
  if (!offset && INSIDE_STRUCT && stsk == &STSK_ENT(0) && stsk->type == 'd') {
    offset = get_kind_parm(sptr, stsk->dtype);
  }
  if (offset)
    IGNOREP(sptr, TRUE); /* needed for "implicit none" */
  return offset;
}

static int
get_kind_parm(int sptr, int dtype)
{
  int rslt, mem;

  if (DTY(dtype) != TY_DERIVED)
    return 0;

  for (mem = DTY(dtype + 1); mem > NOSYM; mem = SYMLKG(mem)) {
    if (PARENTG(mem)) {
      rslt = get_kind_parm(sptr, DTYPEG(mem));
      if (rslt)
        return rslt;
    }
    if (!USEKINDG(mem) && KINDG(mem) &&
        strcmp(SYMNAME(mem), SYMNAME(sptr)) == 0)
      return KINDG(mem);
  }

  return 0;
}

static int
get_kind_parm_strict(int sptr, int dtype)
{
  int rslt, mem;

  if (DTY(dtype) != TY_DERIVED)
    return 0;

  for (mem = DTY(dtype + 1); mem > NOSYM; mem = SYMLKG(mem)) {
    if (PARENTG(mem)) {
      rslt = get_kind_parm(sptr, DTYPEG(mem));
      if (rslt)
        return rslt;
    }
    if (!USEKINDG(mem) && !LENPARMG(mem) && KINDG(mem) &&
        strcmp(SYMNAME(mem), SYMNAME(sptr)) == 0) {
      return KINDG(mem);
    }
  }

  return 0;
}

/** \brief search a derived type for a kind type parameter with a specified
  *        name.
  *
  * \param np is the name we're search for
  * \param dtype is the derived type record that we are searching
  *
  * \return integer > 0 for the parameter number, else 0 if not found.
  */
int
get_kind_parm_by_name(char *np, int dtype)
{
  int rslt, mem;

  if (DTY(dtype) != TY_DERIVED)
    return 0;

  for (mem = DTY(dtype + 1); mem > NOSYM; mem = SYMLKG(mem)) {
    if (PARENTG(mem)) {
      rslt = get_kind_parm_by_name(np, DTYPEG(mem));
      if (rslt)
        return rslt;
    }
    if (!USEKINDG(mem) && KINDG(mem) && strcmp(SYMNAME(mem), np) == 0)
      return KINDG(mem);
  }

  return 0;
}

/** \brief search derived type for a type parameter in the same position as
  *        specified by offset.
  *
  * \param offset is the desired parameter position
  * \param dtype is the derived type record to search in
  *
  * \return symbol table pointer of the parameter component in the derived
  *         type; else 0 if not found.
  */
int
get_parm_by_number(int offset, int dtype)
{
  int rslt, mem;

  if (DTY(dtype) != TY_DERIVED)
    return 0;

  for (mem = DTY(dtype + 1); mem > NOSYM; mem = SYMLKG(mem)) {
    if (PARENTG(mem)) {
      rslt = get_parm_by_number(offset, DTYPEG(mem));
      if (rslt)
        return rslt;
    }
    if (!USEKINDG(mem) && KINDG(mem) == offset)
      return mem;
  }
  return 0;
}

/** \brief search a derived type for a kind or length type parameter with a
  *        specified name.
  *
  * \param np is the name we're search for
  * \param dtype is the derived type record that we are searching
  *
  * \return integer > 0 for the parameter number, else 0 if not found.
  */
int
get_parm_by_name(char *np, int dtype)
{
  int rslt, mem;

  if (DTY(dtype) != TY_DERIVED)
    return 0;

  for (mem = DTY(dtype + 1); mem > NOSYM; mem = SYMLKG(mem)) {
    if (PARENTG(mem)) {
      rslt = get_parm_by_name(np, DTYPEG(mem));
      if (rslt)
        return rslt;
    }
    if (!USEKINDG(mem) && KINDG(mem) && strcmp(np, SYMNAME(mem)) == 0)
      return mem;
  }
  return 0;
}

/** Should be called when we parse ENDTYPE. This function goes
  * through a derived type's members and makes sure there are
  * no length type parameters in the initialization part of a
  * member.
  */
static void
chk_initialization_with_kind_parm(int dtype)
{
  int mem;

  if (DTY(dtype) == TY_ARRAY)
    dtype = DTY(dtype + 1);

  if (DTY(dtype) != TY_DERIVED)
    return;

  for (mem = DTY(dtype + 1); mem > NOSYM; mem = SYMLKG(mem)) {
    if (PARENTG(mem)) {
      chk_initialization_with_kind_parm(DTYPEG(mem));
    }
    if (INITKINDG(mem) && PARMINITG(mem) &&
        !chk_kind_parm_expr(PARMINITG(mem), dtype, 0, 1)) {
      error(155, 3, gbl.lineno, "Initialization must be a constant"
                                " expression for component",
            SYMNAME(mem));
    }
  }
}

int
chk_kind_parm_expr(int ast, int dtype, int flag, int strict_flag)
{
  int sptr, offset, rslt, i;

  if (!ast)
    return 0;

  switch (A_TYPEG(ast)) {
  case A_INTR:
    switch (A_OPTYPEG(ast)) {
    case I_INT1:
    case I_INT2:
    case I_INT4:
    case I_INT8:
    case I_INT:
      i = A_ARGSG(ast);
      return chk_kind_parm_expr(ARGT_ARG(i, 0), dtype, flag, strict_flag);
    }
    break;
  case A_CONV:
    return chk_kind_parm_expr(A_LOPG(ast), dtype, flag, strict_flag);
  case A_CNST:
    return 1;
  case A_ID:
    sptr = A_SPTRG(ast);
    offset = (!strict_flag) ? get_kind_parm(sptr, dtype)
                            : get_kind_parm_strict(sptr, dtype);
    if (flag && !offset && (!strict_flag || !get_kind_parm(sptr, dtype))) {
      /* we might be in the middle of a derived type definition, so see if
       * there's a match in the type parameter queue.
       */
      offset = queue_type_param(sptr, 0, 0, 3);
    }
    if (!offset)
      return 0;
    IGNOREP(sptr, TRUE); /* prevent "implicit none" errors */
    KINDP(sptr, offset);
    return offset;
  case A_UNOP:
    return chk_kind_parm_expr(A_LOPG(ast), dtype, flag, strict_flag);
  case A_BINOP:
    rslt = chk_kind_parm_expr(A_LOPG(ast), dtype, flag, strict_flag);
    if (!rslt)
      return 0;
    rslt = chk_kind_parm_expr(A_ROPG(ast), dtype, flag, strict_flag);
    if (!rslt)
      return 0;
    return rslt;
  }

  return 0;
}

static int
has_kind_parm_expr(int ast, int dtype, int flag)
{

  int sptr, offset, rslt, i;

  if (!ast)
    return 0;

  switch (A_TYPEG(ast)) {
  case A_INTR:
    switch (A_OPTYPEG(ast)) {
    case I_INT1:
    case I_INT2:
    case I_INT4:
    case I_INT8:
    case I_INT:
      i = A_ARGSG(ast);
      return has_kind_parm_expr(ARGT_ARG(i, 0), dtype, flag);
    }
    break;
  case A_CONV:
    return has_kind_parm_expr(A_LOPG(ast), dtype, flag);
  case A_CNST:
    return 0;
  case A_ID:
    sptr = A_SPTRG(ast);
    offset = get_kind_parm_strict(sptr, dtype);
    if (flag && !offset) {
      /* we might be in the middle of a derived type definition, so see if
       * there's a match in the type parameter queue.
       */
      offset = queue_type_param(sptr, 0, 0, 3);
    }
    if (!offset)
      return 0;
    IGNOREP(sptr, TRUE); /* prevent "implicit none" errors */
    KINDP(sptr, offset);
    return offset;
  case A_UNOP:
    return has_kind_parm_expr(A_LOPG(ast), dtype, flag);
  case A_BINOP:
    rslt = has_kind_parm_expr(A_LOPG(ast), dtype, flag);
    if (rslt)
      return rslt;
    rslt = has_kind_parm_expr(A_ROPG(ast), dtype, flag);
    return rslt;
  }

  return 0;
}

static int
chk_asz_deferlen(int ast, int dtype)
{

  int sptr, mem, rslt;

  if (!ast)
    return 0;

  switch (A_TYPEG(ast)) {
  case A_ID:
    sptr = A_SPTRG(ast);
    rslt = 0;
    for (mem = DTY(dtype + 1); mem > NOSYM; mem = SYMLKG(mem)) {
      if (PARENTG(mem)) {
        rslt = chk_asz_deferlen(ast, DTYPEG(mem));
        if (rslt < 0)
          return rslt;
        continue;
      }
      if (strcmp(SYMNAME(mem), SYMNAME(sptr)) == 0) {
        rslt = sptr = mem;
        break;
      }
    }
    if (rslt) {
      if (DEFERLENG(sptr))
        return -1;
      else if (ASZG(sptr))
        return -2;
    }
    break;
  case A_BINOP:
    rslt = chk_asz_deferlen(A_LOPG(ast), dtype);
    if (rslt != 0) {
      return rslt;
    }
    rslt = chk_asz_deferlen(A_ROPG(ast), dtype);
    if (rslt != 0) {
      return rslt;
    }
  }
  return 0;
}

static int
get_len_parm(int sptr, int dtype)
{
  int rslt, mem;

  if (DTY(dtype) != TY_DERIVED)
    return 0;

  for (mem = DTY(dtype + 1); mem > NOSYM; mem = SYMLKG(mem)) {
    if (PARENTG(mem)) {
      rslt = get_len_parm(sptr, DTYPEG(mem));
      if (rslt)
        return rslt;
    }
    if (LENPARMG(mem) && !USEKINDG(mem) && KINDG(mem) &&
        strcmp(SYMNAME(mem), SYMNAME(sptr)) == 0)
      return KINDG(mem);
  }

  return 0;
}

int
chk_len_parm_expr(int ast, int dtype, int flag)
{
  int sptr, offset, rslt;

  if (!ast)
    return 0;

  switch (A_TYPEG(ast)) {

  case A_CNST:
    return 1;
  case A_ID:
    sptr = A_SPTRG(ast);
    offset = get_len_parm(sptr, dtype);
    if (flag && !offset) {
      /* we might be in the middle of a derived type definition, so see if
       * there's a match in the type parameter queue.
       */
      offset = queue_type_param(sptr, 0, 0, 3);
    }
    if (offset) {
      IGNOREP(sptr, TRUE); /* prevent "implicit none" errors */
      if (ST_ISVAR(STYPEG(sptr)) || STYPEG(sptr) == ST_IDENT) {
        /* This symbol is a len parameter place holder. */
        LENPHP(sptr, 1);
      }
    }
    return offset;
  case A_UNOP:
    return chk_len_parm_expr(A_LOPG(ast), dtype, flag);
  case A_BINOP:
    rslt = chk_len_parm_expr(A_LOPG(ast), dtype, flag);
    if (!rslt)
      return 0;
    rslt = chk_len_parm_expr(A_ROPG(ast), dtype, flag);
    if (!rslt)
      return 0;
    return rslt;
  }

  return 0;
}

#ifdef FLANG_SEMANT_UNUSED
static int
fix_kind_parm_expr(int ast, int dtype, int offset, int value)
{
  int sptr, newast;

  switch (A_TYPEG(ast)) {

  case A_CNST:
    break;
  case A_ID:
    sptr = A_SPTRG(ast);
    if (KINDG(sptr) == offset) {
      ast = mk_cval1(value, DT_INT);
    }
    break;
  case A_UNOP:
    newast = fix_kind_parm_expr(A_LOPG(ast), dtype, offset, value);
    A_LOPP(ast, newast);
    break;
  case A_BINOP:
    newast = fix_kind_parm_expr(A_LOPG(ast), dtype, offset, value);
    A_LOPP(ast, newast);
    newast = fix_kind_parm_expr(A_ROPG(ast), dtype, offset, value);
    A_ROPP(ast, newast);
    break;
  }

  return ast;
}
#endif

int
get_len_set_parm_by_name(char *np, int dtype, int *val)
{
  int rslt, mem;

  if (DTY(dtype) != TY_DERIVED)
    return 0;

  for (mem = DTY(dtype + 1); mem > NOSYM; mem = SYMLKG(mem)) {
    if (PARENTG(mem)) {
      rslt = get_len_set_parm_by_name(np, DTYPEG(mem), val);
      if (rslt)
        return rslt;
    }
    if (LENPARMG(mem) && SETKINDG(mem) && !USEKINDG(mem) && KINDG(mem) &&
        strcmp(SYMNAME(mem), np) == 0) {
      *val = LENG(mem);
      return KINDG(mem);
    }
  }

  return 0;
}

int
cmp_len_parms(int ast1, int ast2)
{

  int sptr1, sptr2;
  int rslt;

  if (A_TYPEG(ast1) != A_TYPEG(ast2))
    return 0;

  switch (A_TYPEG(ast1)) {

  case A_CNST:
    if (CONVAL2G(A_SPTRG(ast1)) == CONVAL2G(A_SPTRG(ast2)))
      return 1;
    return 0;
  case A_ID:
    sptr1 = A_SPTRG(ast1);
    sptr2 = A_SPTRG(ast2);
    return sptr1 == sptr2;
  case A_UNOP:
    if (A_OPTYPEG(ast1) != A_OPTYPEG(ast2))
      return 0;
    return cmp_len_parms(A_LOPG(ast1), A_LOPG(ast2));
  case A_BINOP:
    if (A_OPTYPEG(ast1) != A_OPTYPEG(ast2))
      return 0;
    rslt = cmp_len_parms(A_LOPG(ast1), A_LOPG(ast2));
    if (!rslt)
      return 0;
    rslt = cmp_len_parms(A_ROPG(ast1), A_ROPG(ast2));
    if (!rslt)
      return 0;
    return 1;
  }

  return 0;
}

/** \brief Store dtypes of parameterized derived types in which a parameter was
           explicitly declared (as opposed to using just the default values).
 */
int
defer_pt_decl(int dtype, int flag)
{
  typedef struct ptList {
    int dtype;
    struct ptList *next;
  } PL;

  static PL *pl = NULL;
  PL *curr, *newpl, *prev;
  int rslt;

  rslt = 0;
  if (flag == 0 && !sem.interface && sem.which_pass) {
    /* delete all entries from list */
    for (curr = pl; curr;) {
      prev = curr;
      curr = curr->next;
      FREE(prev);
      rslt = 1;
    }
    pl = NULL;
  } else if (flag == 1 && !sem.which_pass) {
    /* add entry */
    NEW(newpl, PL, 1);
    newpl->dtype = dtype;
    newpl->next = pl;
    pl = newpl;
    rslt = 1;
  } else if (flag == 2 && sem.which_pass) {
    /* is this list non-empty? */
    rslt = (pl != NULL);
  }

  return rslt;
}

static void
defer_put_kind_type_param(int offset, int value, char *name, int dtype, int ast,
                          int flag)
{
  typedef struct parmList {
    int offset;
    int value;
    char *name;
    int ast;
    int is_defer_len;
    int is_assume_sz;
    struct parmList *next;
  } PL;

  static PL *pl = NULL;
  PL *curr, *newpl, *prev;
  int i;
  int rslt;
  int flag2;

  rslt = 0;
  if (flag == 0) {
    /* delete all entries from list */
    for (curr = pl; curr;) {
      prev = curr;
      curr = curr->next;
      FREE(prev);
      rslt = 1;
    }
    pl = NULL;
  } else if (flag == 1) {
    /* add entry */
    NEW(newpl, PL, 1);
    newpl->offset = offset;
    newpl->value = value;
    newpl->name = name;
    newpl->ast = ast;
    newpl->is_defer_len = sem.param_defer_len;
    newpl->is_assume_sz = sem.param_assume_sz;
    newpl->next = pl;
    pl = newpl;
    rslt = 1;
  } else if (flag == 2) {
    /* process type params */
    if (DTY(dtype) != TY_DERIVED) {
      return;
    }
    for (curr = pl; curr; curr = curr->next) {
      rslt = 1;
      if (sem.new_param_dt == 0) {
        sem.new_param_dt = create_parameterized_dt(dtype, 0);
      }
      if (curr->is_defer_len) {
        flag2 = -1;
      } else if (curr->is_assume_sz) {
        flag2 = -2;
      } else
        flag2 = 0;
      if (!curr->name) {
        i = put_kind_type_param(sem.new_param_dt, curr->offset, curr->value,
                                curr->ast, flag2);
        if (!i) {
          error(155, 3, gbl.lineno, "Too many type parameter specifiers", NULL);
        }
      } else {
        i = get_kind_parm_by_name(curr->name, sem.new_param_dt);
        if (i) {
          put_kind_type_param(sem.new_param_dt, i, curr->value, curr->ast,
                              flag2);
        } else {
          error(155, 3, gbl.lineno, "Undefined type parameter", curr->name);
        }
      }
    }
    check_kind_type_param(sem.new_param_dt);
  }
}

void
put_default_kind_type_param(int dtype, int flag, int flag2)
{

  typedef struct dtyList {
    int dtype;
    struct dtyList *next;
  } DL;

  static DL *dl = NULL;
  DL *curr, *newdl, *prev;

  int mem_dtype, offset, val, mem;

  if (DTY(dtype) != TY_DERIVED || !has_type_parameter(dtype))
    return;
  for (mem = DTY(dtype + 1); mem > NOSYM; mem = SYMLKG(mem)) {
    mem_dtype = DTYPEG(mem);
    if (PARENTG(mem)) {
      NEW(newdl, DL, 1);
      BZERO(newdl, DL, 1);
      newdl->dtype = dtype;
      newdl->next = dl;
      dl = newdl;
      put_default_kind_type_param(mem_dtype, 1, flag2);
    } else if (!SETKINDG(mem) && !USEKINDG(mem) && (offset = KINDG(mem)) &&
               (val = PARMINITG(mem))) {
      put_kind_type_param(dtype, offset, val, 0, flag2);
      for (curr = dl; curr; curr = curr->next) {
        put_kind_type_param(curr->dtype, offset, val, 0, flag2);
      }
    }
  }
  if (!flag) {
    for (curr = dl; curr;) {
      prev = curr;
      curr = curr->next;
      FREE(prev);
    }
    dl = NULL;
  }
  chkstruct(dtype);
}

void
put_length_type_param(DTYPE dtype, int flag)
{

  typedef struct dtyList {
    DTYPE dtype;
    struct dtyList *next;
  } DL;

  typedef struct char_info {
    DTYPE dtype;
    int situation;
    int ast;
    struct char_info *next;
  } CL;

  static DL *dl = NULL;
  DL *curr, *newdl, *prev;

  static CL *cl = NULL;
  CL *ccl, *newcl, *pcl;

  int mem;

  if (flag == 2) {
    for (pcl = ccl = cl; ccl;) {
      ccl = ccl->next;
      FREE(pcl);
      pcl = ccl;
    }
    cl = NULL;
    return;
  }

  if (DTY(dtype) != TY_DERIVED || !has_type_parameter(dtype))
    return;

  if (!sem.new_param_dt) {
    dtype = sem.new_param_dt = create_parameterized_dt(dtype, 0);
  }

  for (mem = DTY(dtype + 1); mem > NOSYM; mem = SYMLKG(mem)) {
    DTYPE mem_dtype = DTYPEG(mem);
    if (PTRVG(mem) || DESCARRAYG(mem)) {
      continue;
    }
    if (PARENTG(mem)) {
      NEW(newdl, DL, 1);
      BZERO(newdl, DL, 1);
      newdl->dtype = dtype;
      newdl->next = dl;
      dl = newdl;
      put_length_type_param(mem_dtype, flag + 1);
    }

    if (DTY(mem_dtype) == TY_CHAR || DTY(mem_dtype) == TY_NCHAR)
    {
      int ast = DTY(mem_dtype + 1);
      if (flag >= 3)
        continue;
      for (ccl = cl; ccl; ccl = ccl->next) {
        if (ccl->dtype == mem_dtype && ccl->situation) {
          goto do_assume_sz;
        }
      }
      if (A_TYPEG(ast) != A_CNST) {
        int i = chk_kind_parm_set_expr(ast, dtype);
        if (i > 0) {
          if (A_TYPEG(i) == A_CNST) {
            int con = CONVAL2G(A_SPTRG(i));
            if (con < 0) {
              i = chk_asz_deferlen(ast, dtype);
              if (i == -1 || i == -2) {
                i = sym_get_scalar(SYMNAME(mem), "len", DT_INT);
                DTY(mem_dtype + 1) = mk_id(i);
              do_assume_sz:
                LENP(mem, ast);
                NEW(newcl, CL, 1);
                newcl->dtype = mem_dtype;
                newcl->situation = 2;
                newcl->ast = LENG(mem);
                newcl->next = cl;
                cl = newcl;
                ALLOCATTRP(mem, 1);
                TPALLOCP(mem, 1);
                goto shared_alloc_char;
              } else {
                interr("put_length_type_param: unexpected len type param", 0,
                       3);
                LENP(mem, astb.i0);
                DTY(mem_dtype + 1) = astb.i0;
              }

            } else {
              DTY(mem_dtype + 1) = i;
            }
          } else if (A_TYPEG(i) != A_CNST) {
            DTY(mem_dtype + 1) = i;
            LENP(mem, i);

          shared_alloc_char:
            if (!ALLOCG(mem) && !ALLOCATTRG(mem) && !POINTERG(mem))
              TPALLOCP(mem, 1);
            ALLOCP(mem, TRUE);
            USELENP(mem, TRUE);

            DTYPEP(mem,
                   (DTY(mem_dtype) == TY_CHAR) ? DT_DEFERCHAR : DT_DEFERNCHAR);
            if (SDSCG(mem) || STYPEG(SDSCG(mem)) != ST_MEMBER) {
              ENCLDTYPEP(mem, dtype);
              SDSCP(mem, sym_get_sdescr(mem, 0));
              get_all_descriptors(mem);
            }
            ALLOCDESCP(mem, TRUE);
          } else
            DTY(mem_dtype + 1) = i;
        }
      }
    }

    if (DTY(mem_dtype) == TY_ARRAY && !DESCARRAYG(mem)) {
      int numdim, i, num_ast;
      ADSC *ad;

      mem_dtype = dup_array_dtype(mem_dtype);
      DTYPEP(mem, mem_dtype);

      ad = AD_DPTR(mem_dtype);
      numdim = AD_NUMDIM(ad);
      num_ast = 0;

      for (i = 0; i < numdim; i++) {
        int lb, ub, bndast, con;

        if (SDSCG(mem) != 0) {
          /* replace the descriptor in the bounds expressions with the
             descriptor created for mem in get_parameterized_dt() */
          replace_sdsc_in_bounds(SDSCG(mem), ad, i);
        }

        lb = bndast = AD_LWAST(ad, i);
        if (bndast != 0 && A_ALIASG(bndast) == 0) {
          int ast = chk_kind_parm_set_expr(bndast, dtype);
          if (ast > 0) {
            lb = AD_LWAST(ad, i) = ast;
            if (A_TYPEG(ast) != A_CNST) {
              if (!ALLOCG(mem) && !ALLOCATTRG(mem) && !POINTERG(mem))
                TPALLOCP(mem, TRUE);
              ALLOCP(mem, TRUE);
              USELENP(mem, TRUE);
              ADJARRP(mem, TRUE);
              if (!SDSCG(mem)) {
                ENCLDTYPEP(mem, dtype);
                get_static_descriptor(mem);
                get_all_descriptors(mem);
              }
            }
          }
        }

        ub = bndast = AD_UPAST(ad, i);
        con = USEDEFERG(mem) && A_TYPEG(ub) == A_BINOP
                  ? 0
                  : chk_asz_deferlen(bndast, dtype);
        if (con == -1) {
          USEDEFERP(mem, TRUE);
          if (A_TYPEG(ub) == A_BINOP && flag < 3) {
            continue;
          }
        }
        if (!USEDEFERG(mem) && A_TYPEG(ub) == A_BINOP) {
          ub = mk_stmt(A_BINOP, 0);
          A_OPTYPEP(ub, A_OPTYPEG(bndast));
          A_LOPP(ub, A_LOPG(bndast));
          A_ROPP(ub, A_ROPG(bndast));
          A_DTYPEP(ub, A_DTYPEG(bndast));
          bndast = AD_UPAST(ad, i) = ub;
        }
        if (bndast != 0 && A_ALIASG(bndast) == 0) {
          int ast = chk_kind_parm_set_expr(bndast, dtype);
          if (ast <= 0 || A_TYPEG(ast) == A_CNST) {
            int con2 = ast <= 0 ? ast : CONVAL2G(A_SPTRG(ast));
            if (con2 <= 0 && (con == -1 || con == -2))
              ast = bndast;
          }

          if (ast > 0) {
            ub = AD_UPAST(ad, i) = ast;
            if (USELENG(mem)) {
              if (!ALLOCG(mem) && !ALLOCATTRG(mem) && !POINTERG(mem))
                TPALLOCP(mem, TRUE);
              ALLOCP(mem, TRUE);
              USELENP(mem, TRUE);
              ADJARRP(mem, TRUE);
              if (!SDSCG(mem) || STYPEG(SDSCG(mem)) != ST_MEMBER) {
                ENCLDTYPEP(mem, dtype);
                get_static_descriptor(mem);
                get_all_descriptors(mem);
              }

              if (USEDEFERG(mem)) {
                int mem2, mem3;
                int mem1 = SYMLKG(mem);
                int sdsc_mem = mem1;
                if (sdsc_mem == MIDNUMG(mem) || PTRVG(sdsc_mem)) {
                  sdsc_mem = mem2 = SYMLKG(sdsc_mem);
                }
                if (PTRVG(sdsc_mem) || !DESCARRAYG(sdsc_mem)) {
                  sdsc_mem = mem3 = SYMLKG(sdsc_mem);
                }

                if (DESCARRAYG(sdsc_mem)) {
                  if (mem1 > NOSYM)
                    USEDEFERP(mem1, TRUE);
                  if (mem2 > NOSYM)
                    USEDEFERP(mem2, TRUE);
                  if (mem3 > NOSYM)
                    USEDEFERP(mem3, TRUE);
                }
              }
            }
          }
        }
        AD_LWAST(ad, i) = mk_bnd_int(lb);
        AD_UPAST(ad, i) = mk_bnd_int(ub);
        bndast =
            mk_binop(OP_SUB, AD_UPAST(ad, i), AD_LWAST(ad, i), astb.bnd.dtype);
        bndast = mk_binop(OP_ADD, bndast, mk_isz_cval(1, astb.bnd.dtype),
                          astb.bnd.dtype);

        if (!SDSCG(mem)) {
          AD_EXTNTAST(ad, i) = bndast;
        } else {
          AD_EXTNTAST(ad, i) = get_extent(SDSCG(mem), i);
          AD_MLPYR(ad, i) = get_local_multiplier(SDSCG(mem), i);
        }

        if (!num_ast) {
          num_ast = bndast;
        } else {
          num_ast = mk_binop(OP_MUL, num_ast, bndast, astb.bnd.dtype);
        }
      }
      if (num_ast) {
        ADD_NUMELM(mem_dtype) = num_ast;
      }
    }
  }
  if (flag > 0) {
    for (curr = dl; curr;) {
      prev = curr;
      curr = curr->next;
      FREE(prev);
    }
    dl = NULL;
  }
  chkstruct(dtype);
}

/* Replace sdsc in the ASTs for each bound */
static void
replace_sdsc_in_bounds(int sdsc, ADSC *ad, int i)
{
  int ast = replace_sdsc_in_ast(sdsc, AD_LWAST(ad, i));
  if (ast != 0) {
    AD_LWAST(ad, i) = ast;
  }
  ast = replace_sdsc_in_ast(sdsc, AD_LWBD(ad, i));
  if (ast != 0) {
    AD_LWBD(ad, i) = ast;
  }
  ast = replace_sdsc_in_ast(sdsc, AD_UPAST(ad, i));
  if (ast != 0) {
    AD_UPAST(ad, i) = ast;
  }
  ast = replace_sdsc_in_ast(sdsc, AD_UPBD(ad, i));
  if (ast != 0) {
    AD_UPBD(ad, i) = ast;
  }
}

/* If there is an ID node in the ast tree that matches the name of this
   descriptor,
   replace it with the sdsc symbol.  Return 0 if unchanged.
 */
static int
replace_sdsc_in_ast(int sdsc, int ast)
{
  int lop, rop, sptr;
  switch (A_TYPEG(ast)) {
  case A_ID:
    sptr = A_SPTRG(ast);
    if (DESCARRAYG(sptr) && sdsc != sptr &&
        strcmp(SYMNAME(sdsc), SYMNAME(sptr)) == 0) {
      return mk_id(sdsc);
    }
    break;
  case A_BINOP:
    lop = replace_sdsc_in_ast(sdsc, A_LOPG(ast));
    rop = replace_sdsc_in_ast(sdsc, A_ROPG(ast));
    if (lop != 0 || rop != 0) {
      return mk_binop(A_OPTYPEG(ast), lop != 0 ? lop : A_LOPG(ast),
                      rop != 0 ? rop : A_ROPG(ast), A_DTYPEG(ast));
    }
    break;
  case A_SUBSCR:
    lop = replace_sdsc_in_ast(sdsc, A_LOPG(ast));
    if (lop != 0) {
      return mk_subscr_copy(lop, A_ASDG(ast), A_DTYPEG(ast));
    }
    break;
  }
  return 0;
}

int
get_len_parm_by_number(int num, int dtype, int flag)
{
  int mem, i;

  if (DTY(dtype) != TY_DERIVED)
    return 0;

  for (mem = DTY(dtype + 1); mem > NOSYM; mem = SYMLKG(mem)) {
    if (PARENTG(mem)) {
      i = get_len_parm_by_number(num, DTYPEG(PARENTG(mem)), flag);
      if (i)
        return i;
    }
    if (LENPARMG(mem) == num) {
      if (!flag || DEFERLENG(mem) || ASZG(mem)) {
        return mk_id(mem);
      } else {
        INT val[2];
        val[0] = 0;
        val[1] = PARMINITG(mem);
        return mk_cnst(getcon(val, DT_INT));
      }
    }
  }

  return 0;
}

/** \brief Return 0 if there's at least one length type parameter that is not
           assumed. Otherwise return 1.
 */
int
all_len_parms_assumed(int dtype)
{

  int i, mem;

  if (DTY(dtype) != TY_DERIVED)
    return 0;

  for (mem = DTY(dtype + 1); mem > NOSYM; mem = SYMLKG(mem)) {
    if (PARENTG(mem)) {
      i = all_len_parms_assumed(DTYPEG(PARENTG(mem)));
      if (!i)
        return 0;
    }
    if (LENPARMG(mem) && !ASZG(mem))
      return 0;
  }
  return 1;
}

static void
check_kind_type_param(int dtype)
{
  int mem, mem_dtype;

  for (mem = DTY(dtype + 1); mem > NOSYM; mem = SYMLKG(mem)) {
    mem_dtype = DTYPEG(mem);
    if (PARENTG(mem)) {
      check_kind_type_param(mem_dtype);
    }
    if (!SETKINDG(mem) && !USEKINDG(mem) && KINDG(mem) && !LENPARMG(mem) &&
        !PARMINITG(mem)) {
      error(155, 3, gbl.lineno,
            "Missing constant value for kind type parameter", SYMNAME(mem));
    }
  }
}

LOGICAL
put_kind_type_param(DTYPE dtype, int offset, int value, int expr, int flag)
{
  int mem;
  LOGICAL found = FALSE;

  if (DTY(dtype) != TY_DERIVED) {
    return FALSE;
  }

  for (mem = DTY(dtype + 1); mem > NOSYM; mem = SYMLKG(mem)) {
    DTYPE mem_dtype = DTYPEG(mem);
    if (PARENTG(mem)) {
      if (is_pdt_dtype(mem_dtype)) {
        found = put_kind_type_param(mem_dtype, offset, value, expr, flag);
      }
    } else if (USEKINDG(mem) && KINDG(mem) == offset) {
      if (expr && A_TYPEG(expr) != A_CNST) {
        error(155, ERR_Severe, gbl.lineno,
              "Kind type parameter value must be a compile-time constant"
              " for component",
              SYMNAME(mem));
      }
      if (DTY(mem_dtype) != TY_ARRAY) {
        int ast;
        DTYPE out_dtype;
        int ty;
        if (DT_ISINT(mem_dtype))
          ty = TY_INT;
        else if (DT_ISREAL(mem_dtype))
          ty = TY_REAL;
        else if (DT_ISCMPLX(mem_dtype))
          ty = TY_CMPLX;
        else
          ty = DTY(mem_dtype);
        /* Evaluate the kind expression. If we're processing the
         * default dtype, then ast is -1.
         */
        ast = chk_kind_parm_set_expr(KINDASTG(mem), dtype);
        if (ast > 0 && A_TYPEG(ast) == A_CNST) {
          value = CONVAL2G(A_SPTRG(ast));
        } else if (ast > 0) {
          error(155, ERR_Severe, gbl.lineno,
                "Kind type parameter value must be a compile-time constant"
                " for component",
                SYMNAME(mem));
        }
        if (ast > 0 || value == 1 || value == 2 || value == 4 || value == 8) {
          out_dtype = select_kind(mem_dtype, ty, value);
        } else {
          out_dtype = mem_dtype;
        }
        ty = DTY(out_dtype);
        if (ty == TY_CHAR || ty == TY_NCHAR)
        {
          int sym;

          out_dtype = get_type(2, ty, DTY(mem_dtype + 1));

          ast = DTY(mem_dtype + 1);
          switch (A_TYPEG(ast)) {
          case A_ID:
          case A_LABEL:
          case A_ENTRY:
          case A_SUBSCR:
          case A_SUBSTR:
          case A_MEM:
            sym = sym_of_ast(ast);
            break;
          default:
            sym = 0;
          }
          if (!get_len_parm(sym, dtype) && LENG(mem) && USELENG(mem)) {
            ast = get_len_parm_by_number(LENG(mem), dtype,
                                         sem.type_mode || sem.new_param_dt);
          }
        } else {
          ast = 0;
        }
        if (ast)
          DTY(mem_dtype + 1) = ast;
        DTYPEP(mem, out_dtype);
      } else {
        int ast;
        DTYPE out_dtype;
        DTYPE base_dtype = DTY(mem_dtype + 1);
        int ty;
        if (DT_ISINT(base_dtype))
          ty = TY_INT;
        else if (DT_ISREAL(base_dtype))
          ty = TY_REAL;
        else if (DT_ISCMPLX(base_dtype))
          ty = TY_CMPLX;
        else
          ty = DTY(base_dtype);
        out_dtype = select_kind(base_dtype, ty, value);
        if (ty == TY_CHAR || ty == TY_NCHAR)
        {
          out_dtype = get_type(2, ty, DTY(base_dtype + 1));
          ast = DTY(base_dtype + 1);
        } else {
          ast = 0;
        }

        DTY(mem_dtype + 1) = out_dtype;

        if (ast)
          DTY(base_dtype + 1) = ast;
      }
      found = TRUE;
    } else if (flag <= 0 && !SETKINDG(mem) && !USEKINDG(mem) &&
               KINDG(mem) == offset) {
      if (flag == -1)
        DEFERLENP(mem, TRUE);
      if (flag == -2) {
        ASZP(mem, TRUE);
      }
      KINDP(mem, value);
      SETKINDP(mem, TRUE);
      if (LENPARMG(mem)) {
        LENP(mem, expr);
      }
      if (flag == 0 && !LENPARMG(mem) && expr &&
          !chk_kind_parm_expr(expr, dtype, 0, 1)) {
        error(155, 3, gbl.lineno, "Constant expression required for KIND type"
                                  " parameter",
              SYMNAME(mem));
      }
      found = TRUE;
    }
  }
  return found;
}

static void
chk_new_param_dt(int sptr, int dtype)
{
  int mem;

  if (DTY(dtype) != TY_DERIVED)
    return;

  for (mem = DTY(dtype + 1); mem > NOSYM; mem = SYMLKG(mem)) {
    if (DEFERLENG(mem)) {
      if (!LENPARMG(mem)) {
        error(155, 3, gbl.lineno,
              "Deferred type parameter (:) cannot be used with non-length type "
              "parameter",
              SYMNAME(mem));
      }
      if (!ALLOCATTRG(sptr) && !POINTERG(sptr)) {
        error(155, 3, gbl.lineno,
              "A deferred type parameter (:) must be used with "
              " an allocatable or pointer object",
              SYMNAME(sptr));
      }
    }
    if (ASZG(mem)) {
      if (!LENPARMG(mem)) {
        error(155, 3, gbl.lineno,
              "Assumed type parameter (*) cannot be used with non-length type "
              "parameter",
              SYMNAME(mem));
      }
      if (SCG(sptr) != SC_DUMMY) {
        error(155, 3, gbl.lineno,
              "An assumed type parameter (*) cannot be used with non-dummy "
              "argument",
              SYMNAME(sptr));
      }
    }
  }
}

static int
get_vtoff(int vtoff, DTYPE dtype)
{
  SPTR sym = get_struct_members(dtype);

  for (; sym > NOSYM; sym = SYMLKG(sym)) {
    if (PARENTG(sym)) {
      int parent_vtoff = VTOFFG(get_struct_tag_sptr(DTYPEG(sym)));
      if (parent_vtoff > vtoff) {
        vtoff = parent_vtoff;
      }
      vtoff = get_vtoff(vtoff, DTYPEG(sym));
    }
    if (is_tbp(sym)) {
      if (VTOFFG(BINDG(sym)) > vtoff) {
        vtoff = VTOFFG(BINDG(sym));
      }
    }
  }
  return vtoff;
}

int
get_unl_poly_sym(int mem_dtype)
{
  int mem, dtype;
  int sptr = getsymf("_f03_unl_poly$%d", mem_dtype);

  if (STYPEG(sptr) == ST_UNKNOWN) {
    sptr = declsym(sptr, ST_TYPEDEF, TRUE);
    CCSYMP(sptr, 1);
    dtype = get_type(6, TY_DERIVED, NOSYM);
    DTYPEP(sptr, dtype);
    DTY(dtype + 1) = NOSYM;
    DTY(dtype + 2) = 0; /* will be filled in */
    DTY(dtype + 3) = sptr;
    DTY(dtype + 5) = 0;
    UNLPOLYP(sptr, 1);
    DCLDP(sptr, TRUE);
    if (!sem.interface)
      get_static_type_descriptor(sptr);
    if (mem_dtype) {
      mem = getccsym_sc('d', sem.dtemps++, ST_MEMBER, SC_NONE);
      DTYPEP(mem, mem_dtype);
      SYMLKP(mem, DTY(dtype + 1));
      DTY(dtype + 1) = mem;
    }
  } else {
    dtype = DTYPEG(sptr);
    if (DTY(dtype) == TY_DERIVED) {
      DTY(dtype + 3) = sptr;
      UNLPOLYP(sptr, 1);
      CCSYMP(sptr, 1);
      get_static_type_descriptor(sptr);
    }
  }
  return sptr;
}

/** \brief Returns true if dtype is a derived type that has a type parameter or
  * if it has a component that has a type parameter.
  *
  * This function also takes into account recursive components.
  */
static int
has_type_parameter2(int dtype, int visit_flag)
{
  typedef struct visitDty {
    int dty;
    struct visitDty *next;
  } VISITDTY;

  static VISITDTY *visit_list = 0;
  VISITDTY *curr, *new_visit, *prev;

  int rslt;
  int dty = dtype;
  int member;

  if (DTY(dty) == TY_ARRAY)
    dty = DTY(dty + 1);

  if (DTY(dty) != TY_DERIVED) {
    return 0;
  }

  if (visit_list) {
    for (curr = visit_list; curr; curr = curr->next) {
      if (curr->dty == dty) {
        return 0;
      }
    }
  }

  NEW(new_visit, VISITDTY, 1);
  new_visit->dty = dty;
  new_visit->next = visit_list;
  visit_list = new_visit;

  for (rslt = 0, member = DTY(dty + 1); member > NOSYM;
       member = SYMLKG(member)) {
    if (!USEKINDG(member) && KINDG(member)) {
      rslt = 1;
      break;
    }
    if (has_type_parameter2(DTYPEG(member), 1)) {
      rslt = 1;
      break;
    }
  }

  if (!visit_flag && visit_list) {
    for (prev = curr = visit_list; curr;) {

      curr = curr->next;
      FREE(prev);
      prev = curr;
    }
    visit_list = 0;
  }

  return rslt;
}

/** \brief checks to see if derived type record, dtype, has any type
  * parameters (kind or length type parameters).
  *
  * \param dtype is the derived type record we're searching
  *
  * \return integer > 0 if dtype has type parameters; else 0.
  */
int
has_type_parameter(int dtype)
{
  return has_type_parameter2(dtype, 0);
}

#ifdef FLANG_SEMANT_UNUSED
static int
has_length_type_parameter(int dtype)
{

  int mem;

  if (DTY(dtype) != TY_DERIVED)
    return 0;
  for (mem = DTY(dtype + 1); mem > NOSYM; mem = SYMLKG(mem)) {
    if (PARENTG(mem) && has_length_type_parameter(DTYPEG(mem)))
      return 1;
    if (!USEKINDG(mem) && KINDG(mem) && LENPARMG(mem)) {
      return 1;
    }
  }

  return 0;
}
#endif

int
has_length_type_parameter_use(int dtype)
{
  int mem;

  if (DTY(dtype) == TY_ARRAY)
    dtype = DTY(dtype + 1);

  if (DTY(dtype) != TY_DERIVED)
    return 0;
  for (mem = DTY(dtype + 1); mem > NOSYM; mem = SYMLKG(mem)) {
    if (PARENTG(mem) && has_length_type_parameter_use(DTYPEG(mem)))
      return 1;
    if (USELENG(mem)) {
      return 1;
    }
  }
  return 0;
}

static int
get_highest_param_offset(int dtype)
{

  int mem, start, p;

  if (DTY(dtype) != TY_DERIVED)
    return -1;

  for (start = 0, mem = DTY(dtype + 1); mem > NOSYM; mem = SYMLKG(mem)) {
    if (PARENTG(mem)) {
      start = get_highest_param_offset(DTYPEG(PARENTG(mem)));
    }
    if (!USEKINDG(mem) && (p = KINDG(mem))) {
      if (p > start)
        start = p;
    }
  }

  return start;
}

/** \brief Create a parameterized derived type based on dtype.
           If force is not set and dtype is already a PDT, return DT_NONE. */
DTYPE
create_parameterized_dt(DTYPE dtype, LOGICAL force)
{
  int mem;
  int prev_mem;

  if (!has_type_parameter(dtype)) {
    return DT_NONE;
  }
  if (!force && is_pdt_dtype(dtype)) {
    return DT_NONE;
  }
  dtype = get_parameterized_dt(dtype);
  prev_mem = NOSYM;
  for (mem = DTY(dtype + 1); mem > NOSYM; mem = SYMLKG(mem)) {
    if (PARENTG(mem)) {
      DTYPE new_dtype = create_parameterized_dt(DTYPEG(mem), force);
      if (new_dtype) {
        int new_mem = insert_dup_sym(mem);
        DTYPEP(new_mem, new_dtype);
        if (prev_mem == NOSYM) {
          DTY(dtype + 1) = new_mem;
        } else {
          SYMLKP(prev_mem, new_mem);
        }
      }
      break;
    }
    prev_mem = mem;
  }

  return dtype;
}

/** \brief Duplicate \a dtype by creating a new derived type with a $pt suffix.
           For use with processing parameterized derived type.
 */
DTYPE
get_parameterized_dt(DTYPE dtype)
{
  int tag, mem, sptr;
  int first_mem = NOSYM;
  int curr_mem = NOSYM;
  DTYPE new_dtype;
  ACL *ict;

  assert(DTY(dtype) == TY_DERIVED, "expected TY_DERIVED", DTY(dtype),
         ERR_Fatal);

  tag = DTY(dtype + 3);
  sptr = get_next_sym(SYMNAME(tag), "pt");
  DINITP(sptr, DINITG(tag));

  sptr = declsym(sptr, ST_TYPEDEF, TRUE);
  BASETYPEP(sptr, dtype);
  CCSYMP(sptr, 1);
  new_dtype = get_type(6, TY_DERIVED, NOSYM);
  DTYPEP(sptr, new_dtype);
  DTY(new_dtype + 2) = 0; /* will be filled in */
  DTY(new_dtype + 3) = sptr;

  for (mem = DTY(dtype + 1); mem > NOSYM; mem = SYMLKG(mem)) {
    int new_mem = insert_dup_sym(mem);
    VARIANTP(new_mem, curr_mem);
    ENCLDTYPEP(new_mem, new_dtype);
    ADDRESSP(new_mem, 0);
    if (first_mem == NOSYM) {
      first_mem = new_mem;
    } else {
      SYMLKP(curr_mem, new_mem);
    }
    curr_mem = new_mem;
  }
  DTY(new_dtype + 1) = first_mem;
  for (mem = first_mem; mem > NOSYM; mem = SYMLKG(mem)) {
    int descr;
    if (MIDNUMG(mem) && STYPEG(MIDNUMG(mem)) == ST_MEMBER) {
      int mid_mem = SYMLKG(mem);
      if (PTRVG(mid_mem) &&
          strcmp(SYMNAME(mid_mem), SYMNAME(MIDNUMG(mem))) == 0) {
        int off_mem;
        MIDNUMP(mem, mid_mem);
        off_mem = SYMLKG(mid_mem);
        if (PTROFFG(mem) && STYPEG(PTROFFG(mem)) == ST_MEMBER)
          PTROFFP(mem, off_mem);
      }
    }

    if (SDSCG(mem) && STYPEG(mem) == ST_MEMBER) {
      /* Always dup the component descriptor's array dtype */
      int sdsc_mem = get_member_descriptor(mem);
      if (sdsc_mem > NOSYM && DESCARRAYG(sdsc_mem)) {
        DTYPEP(sdsc_mem, dup_array_dtype(DTYPEG(sdsc_mem)));
        SDSCP(mem, sdsc_mem);
      }
    }
    descr = DESCRG(mem);
    if (descr != 0) {
      /* duplicate the descr */
      int new_descr = insert_dup_sym(descr);
      DESCRP(mem, new_descr);
      SECDSCP(new_descr, match_memname(SECDSCG(new_descr), first_mem));
      ARRAYP(new_descr, match_memname(ARRAYG(new_descr), first_mem));
    }
  }

  ict = get_getitem_p(DTY(dtype + 5));
  if (ict != 0) {
    ACL *newict = dup_acl(ict, sptr);
    DTY(new_dtype + 5) = put_getitem_p(newict);
  } else {
    DTY(new_dtype + 5) = 0;
  }

  chkstruct(new_dtype);
  return new_dtype;
}

static ACL *
dup_acl(ACL *src, int sptr)
{
  ACL *subc = src->subc;
  ACL *next = src->next;
  ACL *dst = GET_ACL(15);
  *dst = *src;
  dst->sptr = sptr;
  if (DTY(src->dtype) == TY_DERIVED) {
    dst->dtype = DTYPEG(sptr);
  }
  if (subc != 0) {
    dst->subc = dup_acl(subc, match_memname(subc->sptr, DTY(DTYPEG(sptr) + 1)));
  }
  if (next != 0) {
    dst->next = dup_acl(next, match_memname(next->sptr, SYMLKG(sptr)));
  }
  return dst;
}

/* Return the symbol in mem list (linked through SYMLK) whose name matches sptr.
   Return sptr if there is none. */
static int
match_memname(int sptr, int mem)
{
  for (; mem > NOSYM; mem = SYMLKG(mem)) {
    if (NMPTRG(sptr) == NMPTRG(mem)) {
      return mem;
    }
  }
  return sptr;
}

/* Return TRUE if dtype represents a parameterized derived type. */
static LOGICAL
is_pdt_dtype(DTYPE dtype)
{
  return DTY(dtype) == TY_DERIVED &&
         strstr(SYMNAME(DTY(dtype + 3)), "$pt") != 0;
}

/** \brief allow other source files to check whether we're processing a
  * parameter construct.
  */
int
is_parameter_context()
{
  return (entity_attr.exist & ET_B(ET_PARAMETER));
}

static LOGICAL
ignore_common_decl(void)
{
  if (sem.which_pass == 0) {
    if (sem.mod_cnt && gbl.currsub) {
      /*
       * Do not process the common declaration if in a module subroutine
       */
      return TRUE;
    }
  }
  return FALSE;
}

/** \brief Return the predicate: current entity has the INTRINSIC attribute. */
bool
in_intrinsic_decl(void)
{
  return (entity_attr.exist & ET_B(ET_INTRINSIC)) != 0;
}

/** \brief provide the current entity's access to other source files. */
int
get_entity_access()
{
  return entity_attr.access;
}

/** \brief provide mscall variable state to other source files. */
int
getMscall()
{
  return mscall;
}

/** \brief provide cref variable state to other source files. */
int
getCref()
{
  return cref;
}

/** \brief Determine procedure symbol type for a set of attributes
 *
 *  \param attr attribute mask
 *
 *  \return symbol type index, zero on error
 */
static int
get_procedure_stype(int attr)
{
  if (attr & ET_B(ET_POINTER)) {
    if (!INSIDE_STRUCT) {
      return ST_VAR;
    }

    return ST_MEMBER;
  }

  if (INSIDE_STRUCT) {
    return 0;
  }

  return ST_PROC;
}

/** \brief Declare a procedure symbol
 *
 * Perform check nesessary for a declaration or procedure and produce a new
 * symbol that matches procedure interface and attributes
 *
 *  \param sptr symbol table index for the symbol
 *  \param proc_interf_sptr symbol table entry for procedure interface
 *  \param attr attributes (bit vector), same as entity_attr.exist
 *
 *  \return symbol table index for created symbol
 */
static int
decl_procedure_sym(int sptr, int proc_interf_sptr, int attr)
{
  /* First get expected symbol type */
  int stype = get_procedure_stype(attr);

  if (!stype) {
    /* TODO better place for this error message? */
    error(155, 3, gbl.lineno,
          "PROCEDURE component must have the POINTER attribute -",
          SYMNAME(sptr));
    return 0;
  }

  /* Create a new symbol */
  if (stype != ST_MEMBER) {
    sptr = declsym(sptr, stype, FALSE);
  } else {
    if (STYPEG(sptr) != ST_UNKNOWN)
      sptr = insert_sym(sptr);
    SYMLKP(sptr, NOSYM);
    STYPEP(sptr, ST_MEMBER);
    if (attr & ET_B(ET_NOPASS)) {
      NOPASSP(sptr, 1);
    } else {
      if (!proc_interf_sptr) {
        error(155, 3, gbl.lineno, "The NOPASS attribute must be present for",
              SYMNAME(sptr));
      }
      if (attr & ET_B(ET_PASS)) {
        PASSP(sptr, entity_attr.pass_arg);
        if (IN_MODULE_SPEC) {
          /* Pop the pass arg so it does not pollute
           * other dummy arguments with same name in module.
           * That's because we do not rewrite the pass arg when
           * it's encountered in the contains subroutine. We only
           * write out new symbols. The pass arg does not get its
           * STYPE and CLASS fields, for example, set until we
           * process the contains subroutine. Later, when we use
           * the module, we pull in the uninitialized pass argument
           * which leads to problems if arg is declared CLASS and
           * it does not have CLASS set.
           */
          pop_sym(entity_attr.pass_arg);
        }
      }
    }
  }

  return sptr;
}

/** \brief Process procedure declaration
 *
 * Modify symbol table entry for a procedure declaration, producing the right
 * datatype for procedure pointers or members.
 *
 *  \param sptr symbol table index for the symbol
 *  \param proc_interf_sptr symbol table entry for procedure interface
 *  \param attr attributes (bit vector), same as entity_attr.exist
 *  \param access access level, same as entity_attr.access
 *
 *  \return index of produced symbol table entry, 0 if error
 *
 */
static int
setup_procedure_sym(int sptr, int proc_interf_sptr, int attr, char access)
{
  int stype;
  int dtype;

  /* ********** Determine symbol type ********** */
  stype = get_procedure_stype(attr);

  /*
   * Check for required attributes
   */
  if (!stype) {
    /* TODO better place for this error message? */
    error(155, 3, gbl.lineno,
          "PROCEDURE component must have the POINTER attribute -",
          SYMNAME(sptr));
    return 0;
  }

  if ((stype != ST_MEMBER) && (attr & (ET_B(ET_SAVE) | ET_B(ET_INTENT)))) {
    if (!(attr & ET_B(ET_POINTER))) {
      error(155, 3, gbl.lineno, "The POINTER attribute must be present for",
            SYMNAME(sptr));
      return sptr;
    }
  }

  STYPEP(sptr, stype);

  if (sem.gdtype != -1) {
    dtype = sem.gdtype;
  } else if (proc_interf_sptr) {
    dtype = DTYPEG(proc_interf_sptr);
  } else {
    dtype = DTYPEG(sptr);
  }
  DCLDP(sptr, TRUE);
  if (stype == ST_PROC) {
    if (proc_interf_sptr && (!gbl.currsub || SCG(sptr))) {
      defer_iface(proc_interf_sptr, 0, sptr, 0);
    } else if (scn.stmtyp == TK_PROCEDURE)
      /* have a procedure without an interface, i.e.,
       *   procedure() [...] :: foo
       * Assume 'subroutine'
       */
      dtype = DT_NONE;
  } else {
    /* stype == ST_MEMBER or ST_VAR => have an entity-style declaration with
     * the POINTER attribute
     */
    dtype = get_type(6, TY_PROC, dtype);
    DTY(dtype + 2) = 0; /* interface */
    DTY(dtype + 3) = 0; /* PARAMCT */
    DTY(dtype + 4) = 0; /* DPDSC */
    DTY(dtype + 5) = 0; /* FVAL */

    if (proc_interf_sptr) {
      DTY(dtype + 2) = proc_interf_sptr; /* Set interface */
      defer_iface(proc_interf_sptr, dtype, 0, sptr);
    } else if (sem.gdtype == -1)
      /*
       * Have procedure( ), pointer [...] :: foo k
       * If a type appears as the interface name, sem.gdtype will be set to
       * that type.
       */
      DTY(dtype + 1) = DT_NONE;

    dtype = get_type(2, TY_PTR, dtype);
    if (STYPEG(sptr) != ST_VAR || !IS_PROC_DUMMYG(sptr))
      POINTERP(sptr, TRUE);

    if (access == 'v' || (sem.accl.type == 'v' && access != 'u')) {
      /* Set PRIVATE here for procedure pointers. */
      PRIVATEP(sptr, 1);
    }
  }
  DTYPEP(sptr, dtype);

  /* ********** Add any additional attributes and return ********** */
  if (stype == ST_MEMBER) {
    stsk = &STSK_ENT(0);
    /* link field-namelist into member list at this level */
    link_members(stsk, sptr);
  }

  if (attr & ET_B(ET_SAVE))
    SAVEP(sptr, 1);
  if (attr & ET_B(ET_OPTIONAL))
    OPTARGP(sptr, 1);
  if (attr & ET_B(ET_PROTECTED))
    PROTECTEDP(sptr, 1);
  if (attr & ET_B(ET_BIND))
    process_bind(sptr);

  return sptr;
}

static void
record_func_result(int func_sptr, int func_result_sptr, LOGICAL in_ENTRY)
{
  if (gbl.rutype != RU_FUNC)
    return; /* can't have a RESULT clause unless a function */
  if (in_ENTRY && FVALG(func_sptr) != 0) {
    if (func_result_sptr)
      error(155, 3, gbl.lineno, "The ENTRY cannot have a result name -",
            SYMNAME(func_sptr));
    return;
  }
  if (func_result_sptr != 0) {
    /* result variable from RESULT(func_result_sptr) clause */
    RESULTP(func_sptr, TRUE);
    if (in_ENTRY)
      DCLDP(func_sptr, TRUE);
  } else {
    /* insert a dummy variable with the name of the function */
    func_result_sptr = insert_sym(func_sptr);
    pop_sym(func_result_sptr);
    STYPEP(func_result_sptr, ST_IDENT);
    SCOPEP(func_result_sptr, stb.curr_scope);
    SCP(func_result_sptr, SC_DUMMY);
    if (!in_ENTRY && sem.interface) {
      NODESCP(func_result_sptr, TRUE);
      IGNOREP(func_result_sptr, TRUE);
    }
  }
  if (in_ENTRY && RESULTG(func_result_sptr) != 0) {
    /* create_func_entry_result() discovered that a variable
     * named the same as the result-name was already declared.
     * transfer data type to entry
     */
    DTYPEP(func_sptr, DTYPEG(func_result_sptr));
  } else {
    if (DTYPEG(func_sptr)) {
      /* transfer data type from FUNCTION statement to func_result_sptr */
      DTYPEP(func_result_sptr, DTYPEG(func_sptr));
      ADJLENP(func_result_sptr, ADJLENG(func_sptr));
    }
    RESULTP(func_result_sptr, TRUE);
  }
  FVALP(func_sptr, func_result_sptr);
  if (DCLDG(func_sptr))
    DCLDP(func_result_sptr, TRUE);
}

/** \brief Determine if a type bound procedure (tbp) binding name requires
 * overloading.
 *
 * This is called by the <binding name> ::= <id> '=>' <id> production
 * above. After the tbp is set up, we perform additional overloading checks
 * in resolveBind() of semtbp.c.
 *
 * \pararm sptr is the binding name that we are checking.
 *
 * \return true if it is an overloaded binding name, else false.
 */
static bool
bindingNameRequiresOverloading(SPTR sptr)
{
  if (STYPEG(sptr) == ST_PD) {
    /* Overloaded intrinsic with same name. */
    return true;
  }

  if (STYPEG(sptr) == ST_PROC) {

    if (SCOPEG(sptr) != stb.curr_scope) {
      /* Another use associated symbol with same name. */
      return true;
    }

    if (IN_MODULE_SPEC && TBPLNKG(sptr) == 0) {
      /* Another symbol in module specification section with same name and
       * same scope.
       * This is possibly a procedure with the same name declared in an
       * interface block.
       */
      return true;
    }
  }
  return false;
}

const char *
sem_pgphase_name()
{
  switch (sem.pgphase) {
  case PHASE_END_MODULE:
    return "END_MODULE";
  case PHASE_INIT:
    return "INIT";
  case PHASE_HEADER:
    return "HEADER";
  case PHASE_USE:
    return "USE";
  case PHASE_IMPORT:
    return "IMPORT";
  case PHASE_IMPLICIT:
    return "IMPLICIT";
  case PHASE_SPEC:
    return "SPEC";
  case PHASE_EXEC:
    return "EXEC";
  case PHASE_CONTAIN:
    return "CONTAIN";
  case PHASE_INTERNAL:
    return "INTERNAL";
  case PHASE_END:
    return "END";
  }
}

/** \brief To re-initialize an array of derived types when found the 
 *         following conditions are satisfied:
           1. the element of the array is a derived type.
           2. the array has been initialized before and needs to be 
              re-initialized.
           3. none of any entity attributes used for array definition.
 */
static bool
do_fixup_param_vars_for_derived_arrays(bool inited, SPTR sptr, int sst_idg)
{
  return sem.dinit_count > 0 && inited && !entity_attr.exist &&
         STYPEG(sptr) == ST_IDENT && sst_idg == S_ACONST && 
         DTY(DTYPEG(sptr)) == TY_ARRAY && DTYG(DTYPEG(sptr)) == TY_DERIVED && 
         /* found the tag has been initialized already with a valid sptr*/
         DINITG(DTY(DTY(DTYPEG(sptr)+1)+3));
}
