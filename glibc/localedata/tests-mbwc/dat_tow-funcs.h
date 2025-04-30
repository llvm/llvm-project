/*
 *  TEST SUITE FOR MB/WC FUNCTIONS IN C LIBRARY
 *
 *       FILE:  dat_tow-funcs.h
 *
 *       ISW*:  int tow*( wint_t wc );
 */

#include <errno.h>
#include <stdlib.h>
#include <wctype.h>
#include "tst_types.h"
#include "tgn_locdef.h"

#define TST_TOW_LOC(FUNC, func) \
        TST_TOW## FUNC    tst_tow## func ##_loc[]

#define TST_TOW_REC(locale, func) \
        {  Ttow## func,    TST_LOC_## locale },

/*
 *  NOTE:
 *        need more test data!
 */
