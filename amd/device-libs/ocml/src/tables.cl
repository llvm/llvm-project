/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "ocml.h"
#include "tables.h"

#ifdef USE_TABLESTRUCT

#define DECLARE_TABLE(TYPE,NAME,LENGTH) {

#define END_TABLE() },

__attribute__((visibility("protected"))) __constant struct __tbl_mem_s __tbl_mem = {

#else

#define DECLARE_TABLE(TYPE,NAME,LENGTH) \
__attribute__((visibility("protected"))) __constant TYPE TABLE_MANGLE(NAME) [ LENGTH ] = {

#define END_TABLE() };

#endif

#if 0
#include "atan2D_table.h"
#endif

#ifdef TABLE_BASED_ATAN2
#include "atan2F_table.h"
#endif

#include "cbrtD_table.h"
#include "cbrtF_table.h"
#include "expD_table.h"
#include "expF_table.h"
#include "logD_table.h"
#include "logF_table.h"
#include "pibitsD.h"
#include "powD_table.h"
#include "sinhcoshD_table.h"
#include "sinhcoshF_table.h"
#include "rsqrtF_table.h"
#include "rsqrtD_table.h"
#include "erfF_table.h"
#include "erfD_table.h"
#include "rcbrtF_table.h"
#include "rcbrtD_table.h"
#include "besselF_table.h"
#include "besselD_table.h"

#ifdef USE_TABLESTRUCT
};
#endif

