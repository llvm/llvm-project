/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/* scatter.h - parameters for scatter/gather routines */

/* operation codes */

typedef enum { __SCATTER, __GATHER } gathscat_dir;

/* gather-scatter parameters (not included in schedule) */

typedef struct xstuff xstuff;
typedef struct gathscat_dim gathscat_dim;
typedef struct gathscat_parm gathscat_parm;

struct xstuff {
  xstuff *next;      /* next xlist entry */
  DECL_HDR_PTRS(xd); /* index array descriptor */
  DECL_F90_DIM_PTR(xdd);
  DECL_DIST_DIM_PTR(xdd); /* index array descriptor dimension */
  __INT_T cn;            /* number of cycles (blocks) */
  __INT_T cl;            /* lower cycle loop bound (current index) */
  __INT_T cs;            /* cycle loop stride */
  __INT_T clof;          /* cyclic global->local index offset */
  __INT_T clos;          /* cyclic global->local offset stride */
  __INT_T bn;            /* block element count */
  int str;               /* element stride */
  __INT_T off0;          /* saved element offset */
  int vx;                /* vectored array axis */
  int xx;                /* index array axis */
};

struct gathscat_dim {
  __INT_T *xb;       /* index base address */
  DECL_HDR_PTRS(xd); /* index descriptor */
  __INT_T *xmap;     /* map index axis -> unvectored axis */
};

typedef void (*gatherfn_t)(int, void *, void *, int *);
typedef void (*gathscatfn_t)(int, void *, int *, void *, int *);
typedef void (*scatterfn_t)(int, void *, int *, void *);

struct gathscat_parm {
  const char *what;           /* "GATHER"/"XXX_SCATTER" */
  void (*xfer_request)(struct chdr *, int, void *, long, long, int,
                       long); /* scatter: __fort_sendl; gather: __fort_recvl */
  void (*xfer_respond)(struct chdr *, int, void *, long, long, int,
                       long); /* scatter: __fort_recvl; gather: __fort_sendl */
  gathscatfn_t gathscatfn; /* local gather-scatter-reduction function */
  scatterfn_t scatterfn;   /* local scatter-reduction function */
  char *rb, *ab, *mb;      /* base addresses */
  char *ub, *vb;
  DECL_HDR_PTRS(rd);
  DECL_HDR_PTRS(ad);
  DECL_HDR_PTRS(md); /* descriptors */
  DECL_HDR_PTRS(ud);
  DECL_HDR_PTRS(vd);
  int *counts;      /* request-response counts per cpu */
  int *head;        /* head of linked list for each target */
  int *next;        /* next linked list pointer */
  int *roff;        /* offsets in remote vectored array */
  int *loff;        /* offsets in local unvectored array */
  gathscat_dir dir; /* transfer direction code */

  /* masks with bits selected by vectored array dim... */
  int indirect;    /* dims with indirect indexes */
  int permuted;    /* dims with permuted axes */
  int conform_x_u; /* index conforms with unvectored */
  int aligned_x_u; /* index aligned with unvectored */
  int aligned_v_u; /* vectored aligned with unvectored */
  int aligned_u_v; /* unvectored aligned with vectored */

  int communicate;  /* nonzero if need to communicate */
  int replicate;    /* nonzero if need to broadcast result */
  int group_offset; /* my offset within replication group */
  int outgoing;     /* total number of elements to transfer */

  int ui[MAXDIMS]; /* unvectored array index */

  /* each unvectored array axis has it own list identifying the
     vectored axis/index axis pairs which it indexes. */

  xstuff *xfree;          /* next available xlist entry */
  xstuff *xhead[MAXDIMS]; /* unvectored axis -> first xlist entry */
  xstuff xlist[MAXDIMS * (MAXDIMS + 1)];

  repl_t r_repl; /* result replication descriptor */

  gathscat_dim dim[MAXDIMS]; /* per vectored-dimension parameters */
};

/* prototypes */

void __fort_gathscat_abort(char *what, char *msg);

sked *I8(__fort_gathscat)(gathscat_parm *z);

void *ENTFTN(COMM_START, comm_start)(sked **, void *, F90_Desc *, void *, F90_Desc *);
void ENTFTN(COMM_FINISH, comm_finish)(void *);
void ENTFTN(COMM_FREE, comm_free)(__INT_T *ns, ...);

void *I8(__fort_adjust_index_array)(const char *what, char *idx_array,
                                    char *src, int dim, F90_Desc *is,
                                    F90_Desc *bs);

void *I8(__fort_create_conforming_index_array)(const char *what, char *ab,
                                               void *ib, F90_Desc *as,
                                               F90_Desc *is, F90_Desc *new_is);

void local_gathscat_WRAPPER(int n, void *dst, int *sv, void *src, int *gv, __INT_T kind);

void local_gather_WRAPPER(int n, void *dst, void *src, int *gv, __INT_T kind);

void local_scatter_WRAPPER(int n, void *dst, int *sv, void *src, __INT_T kind);
