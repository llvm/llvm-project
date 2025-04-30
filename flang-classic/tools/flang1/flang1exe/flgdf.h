/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* flgdf.h - data definitions for FTN compiler flags */

FLG flg = {
    FALSE,      /* asm = -noasm */
    FALSE,      /* list = -nolist  */
    TRUE,       /* object = -object */
    FALSE,      /* xref =   -show noxref */
    FALSE,      /* code =   -show nocode */
    FALSE,      /* include = -show noinclude */
    FALSE,      /* output = -nooutput */
    0,          /* debug = -nodebug */
    1,          /* opt  = -opt 1    */
    TRUE,       /* depchk = -depchk on */
    FALSE,      /* depwarn = -depchk warn */
    FALSE,      /* dclchk = -nodclchk */
    FALSE,      /* locchk = -nolocchk  */
    FALSE,      /* onetrip = -noonetrip */
    FALSE,      /* save =  -nosave     */
    1,          /* inform = -inform informational */
    0xFFFFFFFF, /* xoff */
    0x00000000, /* xon  */
    FALSE,      /* ucase = -noucase */
    NULL,       /* idir == empty list */
    FALSE,      /* dlines = -nodlines */
    72,         /* extend_source = -noextend_source */
    TRUE,       /* i4 = -i4 */
    FALSE,      /* line = -noline */
    FALSE,      /* symbol = -nosymbol */
    0,          /* profile = no profiling */
    FALSE,      /* standard = don't flag non-F77 uses */
    {0},        /* dbg[]  */
    TRUE,       /*  align doubles on double word boundary */
    0,          /*  default target assembler syntax:  1-coff, 0-elf */
    FALSE,      /* recursive = -norecursive */
    0,     /* ieee: 0 == none:   num == bit value for item (fdiv==1,ddiv==2) */
    0,     /* inliner: 0 == none */
    0,     /* vect: 0 = none:    num == vect item */
    0,     /* little endian */
    FALSE, /* not terse for summary, etc. */
    '_',   /* default is to change '$' to '_' */
    {0},   /*  x flags  */
    FALSE, /*  don't quad align "unconstrained objects" ; use natural
              alignment */
    FALSE, /* anno - default to no annotation of .s files */
    FALSE, /*  qa = -noqa */
    FALSE, /* es = -noes */
    FALSE, /* p = preprocessor does not emit # lines in its output */
    0,     /*  def ptr */
    0, /*  undef ptr */
    NULL,  /*  search the standard include */
    TRUE,  /* -hpf */
    FALSE, /* -nofreeform */
    FALSE, /* -nosequence */
    0,     /* no ipa */
    TRUE,  /* craft features are supported (if licensed) */
    TRUE,  /* generate the .prelink.f file */
    FALSE, /* generate ilm, not fortran, output */
    TRUE,  /* -hpf */
    FALSE, /* -nosequence */
    25,    /* errorlimit */
    FALSE, /* don't allow smp directives */
    FALSE,      /* omptarget - don't allow OpenMP Offload directives */
    0,                                     /* tpcount */
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },      /* tpvalue */
    0,     /* accmp */
    "",    /* cmdline */
    FALSE, /* qp */
    FALSE, /* list-macros */
};
