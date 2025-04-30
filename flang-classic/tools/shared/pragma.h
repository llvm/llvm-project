/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef PRAGMA_H_
#define PRAGMA_H_

/*
 * pragma scopes
 */
typedef enum {
  PR_NOSCOPE = 0,
  PR_GLOBAL = 1,
  PR_ROUTINE = 2,
  PR_LOOP = 3,
  PR_LINE = 4,
} PR_SCOPE;

/*
 * PR_ACCREDUCTION operators
 */
typedef enum {
  PR_ACCREDUCT_OP_ADD = 0,
  PR_ACCREDUCT_OP_MUL = 1,
  PR_ACCREDUCT_OP_MAX = 2,
  PR_ACCREDUCT_OP_MIN = 3,
  PR_ACCREDUCT_OP_BITAND = 4,
  PR_ACCREDUCT_OP_BITIOR = 5,
  PR_ACCREDUCT_OP_BITEOR = 6,
  PR_ACCREDUCT_OP_LOGAND = 7,
  PR_ACCREDUCT_OP_LOGOR = 8,
  PR_ACCREDUCT_OP_EQV = 9,
  PR_ACCREDUCT_OP_NEQV = 10,
} PR_ACCREDUCT_OP;

/*
 * pragma values
 */
typedef enum {
  PR_NONE = 0,
  PR_INLININGON = 1,
  PR_INLININGOFF = 2,
  PR_ALWAYSINLINE = 3,
  PR_MAYINLINE = 4,
  PR_NEVERINLINE = 5,
  PR_ACCEL = 6,    /* accelerator region directive */
  PR_ENDACCEL = 7, /* accelerator end region directive */
  PR_INLINEONLY = 8,
  PR_INLINETYPE = 9,
  PR_INLINEAS = 10,
  PR_INLINEALIGN = 11,
  PR_ACCCOPYIN = 12,        /* accelerator copyin clause */
  PR_ACCCOPYOUT = 13,       /* accelerator copyout clause */
  PR_ACCLOCAL = 14,         /* accelerator local clause */
  PR_ACCELLP = 15,          /* accelerator loop directive */
  PR_ACCVECTOR = 16,        /* accelerator vector clause */
  PR_ACCPARALLEL = 17,      /* accelerator parallel clause */
  PR_ACCSEQ = 18,           /* accelerator seq clause */
  PR_ACCHOST = 19,          /* accelerator host clause */
  PR_ACCPRIVATE = 20,       /* accelerator private clause */
  PR_ACCCACHE = 21,         /* accelerator cache clause */
  PR_ACCSHORTLOOP = 22,     /* accelerator shortloop clause */
  PR_ACCBEGINDIR = 23,      /* accelerator begin directive */
  PR_ACCIF = 24,            /* accelerator if clause */
  PR_ACCUNROLL = 25,        /* accelerator unroll clause */
  PR_ACCKERNEL = 26,        /* accelerator kernel clause */
  PR_ACCCOPY = 27,          /* accelerator copy clause */
  PR_ACCDATAREG = 28,       /* accelerator data region */
  PR_ACCENDDATAREG = 29,    /* accelerator end data region */
  PR_ACCUPDATEHOST = 30,    /* accelerator update host data */
  PR_ACCUPDATEDEVICE = 31,  /* accelerator update device data */
  PR_ACCUPDATE = 32,        /* update directive */
  PR_ACCINDEPENDENT = 33,   /* loop is independent */
  PR_ACCWAIT = 34,          /* wait clause for kernels */
  PR_ACCNOWAIT = 35,        /* don't wait clause, launch asynchronously */
  PR_ACCIMPDATAREG = 36,    /* implicit accelerator data region */
  PR_ACCENDIMPDATAREG = 37, /* implicit accelerator data region */
  PR_ACCMIRROR = 38,        /* accelerator mirror clause */
  PR_ACCREFLECT = 39,       /* accelerator reflected clause */
  PR_KERNELBEGIN = 40,      /* kernel begin directive */
  PR_KERNEL = 41,           /* kernel */
  PR_ENDKERNEL = 42,        /* end kernel */
  PR_KERNELTILE = 43,       /* kernel tile size */
  PR_ACCDEVSYM = 44,      /* For communicating from F90 front-end to back-end */
  PR_ACCIMPDATAREGX = 45, /* necessary implicit data region */
  PR_KERNEL_NEST = 46,    /* kernel */
  PR_KERNEL_GRID = 47,    /* kernel */
  PR_KERNEL_BLOCK = 48,   /* kernel */
  PR_ACCDEVICEPTR = 49,   /* C device pointer */
  PR_ACCPARUNROLL = 50,   /* unroll parallel loop */
  PR_ACCVECUNROLL = 51,   /* unroll vector loop */
  PR_ACCSEQUNROLL = 52,   /* unroll sequential loop */
  PR_ACCCUDACALL = 53,    /* call cuda kernel directly */
  PR_ACCSCALARREG = 54,   /* scalar region */
  PR_ACCENDSCALARREG = 55,     /* end scalar region */
  PR_ACCPARCONSTRUCT = 56,     /* accelerator parallel construct */
  PR_ACCENDPARCONSTRUCT = 57,  /* end accelerator parallel construct */
  PR_ACCKERNELS = 58,          /* accelerator kernels construct */
  PR_ACCENDKERNELS = 59,       /* end accelerator kernels construct */
  PR_ACCCREATE = 60,           /* create clause, same as local clause */
  PR_ACCPRESENT = 61,          /* present clause */
  PR_ACCPCOPY = 62,            /* present_or_copy clause */
  PR_ACCPCOPYIN = 63,          /* present_or_copyin clause */
  PR_ACCPCOPYOUT = 64,         /* present_or_copyout clause */
  PR_ACCPCREATE = 65,          /* present_or_create clause */
  PR_ACCASYNC = 66,            /* async clause */
  PR_KERNEL_STREAM = 67,       /* kernel stream argument */
  PR_KERNEL_DEVICE = 68,       /* kernel device argument */
  PR_ACCWAITDIR = 69,          /* wait directive */
  PR_ACCKLOOP = 70,            /* loop in accelerator kernels region */
  PR_ACCPLOOP = 71,            /* loop in accelerator parallel region */
  PR_ACCGANG = 72,             /* accelerator gang clause */
  PR_ACCWORKER = 73,           /* accelerator worker clause */
  PR_ACCFIRSTPRIVATE = 74,     /* accelerator firstprivate clause */
  PR_ACCNUMGANGS = 75,         /* accelerator num_gangs clause */
  PR_ACCNUMWORKERS = 76,       /* accelerator num_workers clause */
  PR_ACCVLENGTH = 77,          /* accelerator vector_length clause */
  PR_ACCWAITARG = 78,          /* wait directive argument */
  PR_ACCREDUCTION = 79,        /* reduction clause */
  PR_ACCREDUCTOP = 80,         /* reduction operator */
  PR_ACCCACHEDIR = 81,         /* cache directive */
  PR_ACCCACHEARG = 82,         /* cache directive argument */
  PR_ACCHOSTDATA = 83,         /* host_data directive */
  PR_ACCENDHOSTDATA = 84,      /* end host_data */
  PR_ACCUSEDEVICE = 85,        /* use_device clause */
  PR_ACCCOLLAPSE = 86,         /* collapse clause */
  PR_ACCDEVICERES = 87,        /* device_resident clause */
  PR_ACCDEVICEID = 88,         /* deviceid clause */
  PR_ACCDELETE = 89,           /* delete clause on exit data */
  PR_ACCPDELETE = 90,          /* present_or_delete clause on exit data */
  PR_ACCENTERDATA = 91,        /* enter data directive */
  PR_ACCEXITDATA = 92,         /* exit data directive */
  PR_ACCLOOPPRIVATE = 93,      /* implicitly loop-private symbol */
  PR_ACCUPDATESELF = 94,       /* accelerator update self data */
  PR_ACCLINK = 95,             /* 'link' data clause */
  PR_ACCTILE = 96,             /* 'tile' loop clause */
  PR_ACCAUTO = 97,             /* 'auto' loop clause */
  PR_ACCGANGCHUNK = 98,        /* chunk size for scheduling gang loop */
  PR_ACCDEFNONE = 99,          /* default(none) clause */
  PR_ACCPNOT = 100,            /* present_or_not clause */
  PR_ACCNUMGANGS2 = 101,       /* accelerator num_gangs clause, 2nd dimension */
  PR_ACCNUMGANGS3 = 102,       /* accelerator num_gangs clause, 3rd dimension */
  PR_ACCGANGDIM = 103,         /* accelerator gang(dim:) clause value */
  PR_ACCDEFPRESENT = 104,      /* default(present) clause */
  PR_ACCFORCECOLLAPSE = 105,   /* collapse(force) clause */
  PR_CUFLOOPPRIVATE = 106,     /* implicitly loop-private symbol */
  PR_ACCCACHEREADONLY = 107,   /* cache(readonly:*) */
  PR_ACCFINALEXITDATA = 108,   /* exit data directive with finalize clause */
  PR_ACCUPDATEHOSTIFP = 109,   /* accelerator update host data if present */
  PR_ACCUPDATEDEVICEIFP = 110, /* accelerator update device data if present */
  PR_ACCUPDATESELFIFP = 111,   /* accelerator update self data if present */
  PR_ACCINITDIR = 112,         /* accelerator init directive */
  PR_ACCSHUTDOWNDIR = 113,     /* accelerator shutdown directive */
  PR_ACCDEFAULT_ASYNC = 114,   /* accelerator default_async clause */
  PR_ACCDEVICE_NUM = 115,      /* accelerator device_num clause */
  PR_ACCSETDIR = 116,          /* accelerator set directive */
  PR_ACCUSEDEVICEIFP = 117,    /* accelerator use device clause combined with if present */
  PR_ACCNO_CREATE = 118,       /* no_create clause */
  PR_ACCSERIAL = 119,          /* accelerator serial construct */
  PR_ACCENDSERIAL = 120,       /* end accelerator serial construct */
  PR_ACCSLOOP = 121,           /* loop in accelerator serial region */
  PR_ACCATTACH = 122,          /* attach clause */
  PR_ACCDETACH = 123,          /* detach clause */
  PR_ACCTKLOOP = 124,          /* tightly-nested outer loop in kernels construct */
  PR_ACCTPLOOP = 125,          /* tightly-nested outer loop in parallel construct */
  PR_ACCTSLOOP = 126,          /* tightly-nested outer loop in serial construct */
  PR_ACCCOMPARE = 127,         /* pragma that backends into acc_compare (__pgi_uacc_usercompare) */
  PR_PGICOMPARE = 128,         /* pragma that backends into pgi_compare */
  PR_PCASTCOMPARE = 129,       /* generic PCAST compare directive */
  PR_MAPALLOC = 130,           /* OpenMP clause map(alloc: */
  PR_MAPDELETE = 131,          /* OpenMP clause map(delete: */
  PR_MAPFROM = 132,            /* OpenMP clause map(from: */
  PR_MAPRELEASE = 133,          /* OpenMP clause map(release: */
  PR_MAPTO = 134,              /* OpenMP clause map(to: */
  PR_MAPTOFROM = 135,          /* OpenMP clause map(tofrom: */
  PR_UPDATEFROM = 136,         /* OpenMP clause target update from( */
  PR_UPDATETO = 137           /* OpenMP clause target update to( */
} PR_PRAGMA;

/* Ignore data movement pragmas */
#define ACC_DATAMOVEMENT_DISABLED XBIT(195, 0x400)

/**
   \brief ...
 */
void apply_nodepchk(int dir_lineno, int dir_scope);

/**
   \brief ...
 */
void apply_simdlen(int dir_lineno, int dir_scope, int simdlen);


/**
   \brief ...
 */
void p_pragma(char *pg, int pline);

/**
   \brief ...
 */
void push_lpprg(int beg_line);

/**
   \brief ...
 */
void rouprg_enter(void);

#endif
