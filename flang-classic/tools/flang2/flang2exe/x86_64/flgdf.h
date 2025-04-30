/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
 * \brief Data definitions for FTN compiler flags
 */

FLG flg = {
    false,      /* asm = -noasm */
    false,      /* list = -nolist  */
    true,       /* object = -object */
    false,      /* xref =   -show noxref */
    false,      /* code =   -show nocode */
    false,      /* include = -show noinclude */
    0,          /* debug = -nodebug */
    1,          /* opt  = -opt 1    */
    true,       /* depchk = -depchk on */
    false,      /* depwarn = -depchk warn */
    false,      /* dclchk = -nodclchk */
    false,      /* locchk = -nolocchk  */
    false,      /* onetrip = -noonetrip */
    false,      /* save =  -nosave     */
    1,          /* inform = -inform informational */
    0xFFFFFFFF, /* xoff */
    0x00000000, /* xon  */
    false,      /* ucase = -noucase */
    NULL,       /* idir == empty list */
    NULL,       /* linker_directives == empty list */
    NULL,       /* llvm_target_triple == empty ptr */
    NULL,       /* target_features == empty ptr */
    0,          /* vscale_range_min = -vscale_range_min 0 */
    0,          /* vscale_range_max = -vscale_range_max 0 */
    false,      /* dlines = -nodlines */
    72,         /* extend_source = -noextend_source */
    true,       /* i4 = -i4 */
    false,      /* line = -noline */
    false,      /* symbol = -nosymbol */
    0,          /* profile = no profiling */
    false,      /* standard = don't flag non-F77 uses */
    {0},        /* dbg[]  */
    true,       /* align doubles on doubleword boundary */
    0,          /* astype - assembler syntax - 0-elf, 1-coff */
    false,      /* recursive = -norecursive */
    0,          /* ieee: 0 == none:   num == bit value for
                        item (fdiv==1,ddiv==2) */
    0,          /* inline: 0 == none: num == max # ilms */
    0,          /* autoinline */
    0,          /* vect: 0 = none:    num == vect item */
    0,          /* little endian */
    false,      /* not terse for summary, etc. */
    '_',        /* default is to change '$' to '_' */
    {0},        /*  x flags  */
    false,      /*  don't quad align "unconstrained objects";
                        use natural alignment */
    false,      /* anno - don't annotate asm file */
    false,      /*  qa = -noqa */
    false,      /* es = -noes */
    false,      /* p = preprocessor does not emit # lines in its output */
    0,          /*  def ptr */
    NULL,       /*  search the standard include */
    false,      /* don't allow smp directives */
    false,      /* omptarget - don't allow OpenMP Offload directives */
    25,         /* errorlimit */
    false,      /* trans_inv */
    0,                                      /* tpcount */
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, /* tpvalue */
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    "",         /* cmdline */
    false,      /* qp */
};
