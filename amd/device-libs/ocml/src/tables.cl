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

#include "besselF_table.h"
#include "besselD_table.h"

#ifdef USE_TABLESTRUCT
};
#endif

