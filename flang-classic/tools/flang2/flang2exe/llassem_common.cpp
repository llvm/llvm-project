/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
   \file llassem_common.c
   Some various functions that emit LLVM IR.
 */

#include "llassem_common.h"
#include "dinit.h"
#include "dinitutl.h"
#include "version.h"
#include "machreg.h"
#include "assem.h"
#include "mach.h"
#include "llutil.h"
#include "cgllvm.h"
#include "cgmain.h"
#include "cg.h"
#include "llassem.h"

union {
  unsigned short i8;
  unsigned char byte[2];
} i8bit;

union {
  unsigned short i16;
  unsigned char byte[2];
} i16bit;

union {
  unsigned int i32;
  float r4;
  unsigned char byte[4];
} i32bit;

#include "dtypeutl.h"

AGB_t agb;

DSRT *lcl_inits;
DSRT *section_inits;
DSRT *extern_inits;
char static_name[MXIDLN];
int first_data;
int ag_cmblks;
int ag_procs;
int ag_other;
int ag_global;
int ag_typedef;
int ag_static;
int ag_intrin;
int ag_local;
int ag_funcptr;

static void put_ncharstring_n(char *, ISZ_T, int);
static void put_zeroes(ISZ_T);
static void put_cmplx_n(int, int);
static void put_dcmplx_n(int, int);
#ifdef TARGET_SUPPORTS_QUADFP
static void put_qcmplx_n(int, int);
#endif
static void put_i8(int);
static void put_i16(int);
static void put_r4(INT);
static void put_r8(int, int);
#ifdef TARGET_SUPPORTS_QUADFP
static void put_r16(int, int);
#endif
static void put_cmplx_n(int, int);
static void add_ctor(const char *);
static void write_proc_pointer(SPTR sptr);

static void
add_ctor(const char *constructor)
{
  LL_Type *ret = ll_create_basic_type(cpu_llvm_module, LL_VOID, 0);
  LL_Function *fn;
  char buff[128];
  snprintf(buff, sizeof(buff), "declare void @%s()", constructor);
  fn = ll_create_function(cpu_llvm_module, buff + 13, ret, 0, 0, 0, "",
                          LL_NO_LINKAGE);
  llvm_ctor_add(constructor);
  ll_proto_add(fn->name, NULL);
  ll_proto_set_intrinsic(fn->name, buff);
}

char *
put_next_member(char *ptr)
{
  if (!ptr)
    return NULL;
  if (*ptr == ',')
    ptr++;
  while (*ptr != ',' && *ptr != '\0') {
    fprintf(ASMFIL, "%c", *ptr);
    ptr++;
  }
  fprintf(ASMFIL, " ");
  return ptr;
}

ISZ_T
put_skip(ISZ_T old, ISZ_T New, bool is_char)
{
  ISZ_T amt;
  const char *s = "i8 0";
  const char *str = "i8 0,i8 0,i8 0,i8 0,i8 0,i8 0,i8 0,i8 0,i8 0,i8 0,i8 0,"
                    "i8 0,i8 0,i8 0,i8 0,i8 0,i8 0,i8 0,i8 0,i8 0,i8 0,i8 0,"
                    "i8 0,i8 0,i8 0,i8 0,i8 0,i8 0,i8 0,i8 0,i8 0,i8 0";

  if (is_char) {
    s = "i8 32";
    str = "i8 32,i8 32,i8 32,i8 32,i8 32,i8 32,i8 32,i8 32,i8 32,i8 32,i8 32,"
          "i8 32,i8 32,i8 32,i8 32,i8 32,i8 32,i8 32,i8 32,i8 32,i8 32,i8 32,"
          "i8 32,i8 32,i8 32,i8 32,i8 32,i8 32,i8 32,i8 32,i8 32,i8 32";
  }
  if ((amt = New - old) > 0) {
    INT i;
    i = amt;
    while (i > 32) {
      fprintf(ASMFIL, str);
      i -= 32;
      if (i)
        fprintf(ASMFIL, ",");
    }
    if (i) {
      while (1) {
        fprintf(ASMFIL, s);
        i--;
        if (i == 0)
          break;
        fprintf(ASMFIL, ",");
      }
    }
  } else {
    assert(amt == 0, "assem.c-put_skip old,new not in sync", New, ERR_Severe);
  }
  return amt;
}

static void
write_proc_pointer(SPTR sptr)
{
  if (PTR_INITIALIZERG(sptr) && PTR_TARGETG(sptr)) {
    sptr = (SPTR) PTR_TARGETG(sptr);
  }
  fprintf(ASMFIL, "ptr @%s", getsname(sptr));
}

void
emit_init(DTYPE tdtype, ISZ_T tconval, ISZ_T *addr, ISZ_T *repeat_cnt,
          ISZ_T loc_base, ISZ_T *i8cnt, int *ptrcnt, char **cptr, bool is_char)
{
  ISZ_T al;
  int area;
  int size_of_item;
  int putval;
  INT skip_size;
  char str[32];
  area = LLVM_LONGTERM_AREA;
  const ISZ_T orig_tconval = tconval;
  char *oldptr;

  switch ((int)tdtype) {
  case 0: /* alignment record */
          /*  word or halfword alignment required: */
#if DEBUG
    assert(tconval == 7 || tconval == 3 || tconval == 1 || tconval == 0,
           "emit_init:bad align", (int)tconval, ERR_Severe);
#endif
    skip_size = ALIGN(*addr, tconval) - *addr;
    if (skip_size == 0) {
      *addr = ALIGN(*addr, tconval);
      break;
    }
    if (DBGBIT(5, 32)) {
      fprintf(gbl.dbgfil, "emit_init:0 first_data:%d i8cnt:%ld ptrcnt:%d\n",
              first_data, *i8cnt, *ptrcnt);
    }
    if (*ptrcnt) {
      if (!first_data)
        fprintf(ASMFIL, ", ");
      *cptr = put_next_member(*cptr);
      fprintf(ASMFIL, "[" /*]*/);
      *ptrcnt = 0;
    } else if (!(*i8cnt)) {
      if (!first_data)
        fprintf(ASMFIL, ", ");
      *cptr = put_next_member(*cptr);
      fprintf(ASMFIL, "[" /*]*/);
    } else if (*i8cnt) {
      if (!first_data)
        fprintf(ASMFIL, ", ");
    }
    *i8cnt = *i8cnt + put_skip(*addr, ALIGN(*addr, tconval), is_char);
    *addr = ALIGN(*addr, tconval);
    first_data = 0;
    break;
  case DINIT_ZEROES:
    if (!tconval) {
      *addr += tconval;
      break;
    }
    if (DBGBIT(5, 32)) {
      fprintf(gbl.dbgfil,
              "emit_init:DINIT_ZEROES first_data:%d i8cnt:%ld ptrcnt:%d\n",
              first_data, *i8cnt, *ptrcnt);
    }
    if (*ptrcnt) {
      if (!first_data)
        fprintf(ASMFIL, ", ");
      *cptr = put_next_member(*cptr);
      fprintf(ASMFIL, "[" /*]*/);
      *ptrcnt = 0;
    } else if (!(*i8cnt)) {
      if (!first_data)
        fprintf(ASMFIL, ", ");
      *cptr = put_next_member(*cptr);
      fprintf(ASMFIL, "[" /*]*/);
    } else if (*i8cnt) {
      if (!first_data)
        fprintf(ASMFIL, ", ");
    }
    put_zeroes((int)tconval);
    *i8cnt = *i8cnt + ((int)tconval);
    *addr += tconval;
    first_data = 0;
    break;
#ifdef DINIT_PROC
  case DINIT_PROC:
    if (*i8cnt) {
      fprintf(ASMFIL, /*[*/ "] ");
      *i8cnt = 0;
    }
    if (!first_data) {
      fprintf(ASMFIL, ",");
    } else {
      first_data = 0;
    }
    write_proc_pointer((SPTR)tconval);
    (*ptrcnt)++;
    *addr += size_of(DT_CPTR);
    break;
#endif
  case DINIT_LABEL:
    /*  word to be init'ed with address of label 'tconval' */
    al = alignment(DT_CPTR);
    skip_size = ALIGN(*addr, al) - *addr;
    if (DBGBIT(5, 32)) {
      fprintf(gbl.dbgfil,
              "emit_init:DINIT_LABEL first_data:%d i8cnt:%ld ptrcnt:%d\n",
              first_data, *i8cnt, *ptrcnt);
    }

    if (skip_size) { /* if *i8cnt - just add to the end */
      if (!first_data)
        fprintf(ASMFIL, ", ");
      if (*i8cnt) {
        *i8cnt = put_skip(*addr, ALIGN(*addr, al), is_char);
        *i8cnt = 0;
        fprintf(ASMFIL, /*[*/ "], ");
      } else if (*ptrcnt || !(*i8cnt)) {
#ifdef OMP_OFFLOAD_LLVM
        // TODO ompaccel. Hackery for TGT structs. It must be fixed later.
        if (flg.omptarget)
          fprintf(ASMFIL, " ptr ");
        else
#endif
          *cptr = put_next_member(*cptr);
        fprintf(ASMFIL, "[" /*]*/);
        *i8cnt = put_skip(*addr, ALIGN(*addr, al), is_char);
        fprintf(ASMFIL, /*[*/ "], ");
      }
    } else if (*i8cnt) {
      fprintf(ASMFIL, /*[*/ "], ");
      *i8cnt = 0;
    } else if (!first_data)
      fprintf(ASMFIL, ", ");
#ifdef OMP_OFFLOAD_LLVM
    // TODO ompaccel. Hackery for TGT structs. It must be fixed later.
    if (flg.omptarget)
      fprintf(ASMFIL, " ptr ");
    else
#endif
      *cptr = put_next_member(*cptr);
    *addr = ALIGN(*addr, al);
    if (DBGBIT(5, 32)) {
      fprintf(gbl.dbgfil,
              "emit_init:DINIT_LABEL calling put_addr "
              "first_data:%d i8cnt:%ld ptrcnt:%d\n",
              first_data, *i8cnt, *ptrcnt);
    }
    put_addr((SPTR)tconval, 0, DT_NONE); // ???
    (*ptrcnt)++;
    *addr += size_of(DT_CPTR);
    first_data = 0;
    break;
#ifdef DINIT_FUNCCOUNT
  case DINIT_FUNCCOUNT:
    gbl.func_count = tconval;
    break;
#endif
  case DINIT_OFFSET:
    skip_size = (tconval + loc_base) - *addr;
    if (skip_size == 0) {
      *addr = tconval + loc_base;
      break;
    }
    if (DBGBIT(5, 32)) {
      fprintf(gbl.dbgfil,
              "emit_init:DINIT_OFFSET first_data:%d i8cnt:%ld ptrcnt:%d\n",
              first_data, *i8cnt, *ptrcnt);
    }
    if (*ptrcnt) {
      if (!first_data)
        fprintf(ASMFIL, ", ");
      *cptr = put_next_member(*cptr);
      fprintf(ASMFIL, "[" /*]*/);
      *ptrcnt = 0;
    } else if (!(*i8cnt)) {
      if (!first_data)
        fprintf(ASMFIL, ", ");
      *cptr = put_next_member(*cptr);
      fprintf(ASMFIL, "[" /*]*/);
    } else if (*i8cnt) {
      if (!first_data)
        fprintf(ASMFIL, ", ");
    }
    *i8cnt = *i8cnt + put_skip(*addr, tconval + loc_base, is_char);
    *addr = tconval + loc_base;
    first_data = 0;
    break;
  case DINIT_REPEAT:
    *repeat_cnt = tconval;
    break;
  case DINIT_SECT:
    break;
  case DINIT_DATASECT:
    break;
  case DINIT_STRING:
    /* read the string from the dinit file until the length is exhausted */
    *addr += tconval;
    if (tconval == 0) {
      break;
    }
    if (DBGBIT(5, 32)) {
      fprintf(gbl.dbgfil,
              "emit_init:DINIT_STRING first_data:%d i8cnt:%ld ptrcnt:%d\n",
              first_data, *i8cnt, *ptrcnt);
    }
    if (*ptrcnt) {
      if (!first_data)
        fprintf(ASMFIL, ", ");
      *cptr = put_next_member(*cptr);
      fprintf(ASMFIL, "[" /*]*/);
      *ptrcnt = 0;
    } else if (!(*i8cnt)) {
      if (!first_data)
        fprintf(ASMFIL, ", ");
      *cptr = put_next_member(*cptr);
      fprintf(ASMFIL, "[" /*]*/);
    } else if (*i8cnt) {
      if (!first_data)
        fprintf(ASMFIL, ", ");
    }

    /* Output the data */
    *i8cnt += tconval;
    while (tconval > 0) {
      if (tconval != orig_tconval)
        fprintf(ASMFIL, ", ");
      if (tconval > 32) {
        dinit_read_string(32, str);
        put_string_n(str, 32, 0);
        tconval -= 32;
      } else {
        dinit_read_string(tconval, str);
        put_string_n(str, tconval, 0);
        tconval = 0;
      }
    }

    /* We have printed out an entire string, close it */
    first_data = 0;
    break;
  default:
    assert(tdtype > 0, "emit_init:bad dinit rec", tdtype, ERR_Severe);
    size_of_item = size_of(tdtype);

    if (*repeat_cnt > 1) {
      /* TO DO: We may be able to optimize this with zeroinitializer
       * if all i8 before and after this are all zero.
       *
       */
      switch (DTY(tdtype)) {
      case TY_INT8:
      case TY_LOG8:
        if (CONVAL2G(tconval) == 0 &&
            (!XBIT(124, 0x400) || CONVAL1G(tconval) == 0))
          goto do_zeroes;
        break;
      case TY_INT:
      case TY_LOG:
      case TY_SINT:
      case TY_SLOG:
      case TY_BINT:
      case TY_BLOG:
      case TY_FLOAT:
        if (tconval == 0)
          goto do_zeroes;
        break;
      case TY_DBLE:
        if (tconval == stb.dbl0)
          goto do_zeroes;
        break;
#ifdef TARGET_SUPPORTS_QUADFP
      case TY_QUAD:
        if (tconval == stb.quad0)
          goto do_zeroes;
        break;
#endif
      case TY_CMPLX:
        if (CONVAL1G(tconval) == 0 && CONVAL2G(tconval) == 0)
          goto do_zeroes;
        break;
      case TY_DCMPLX:
        if (CONVAL1G(tconval) == stb.dbl0 && CONVAL2G(tconval) == stb.dbl0)
          goto do_zeroes;
        break;
#ifdef TARGET_SUPPORTS_QUADFP
      case TY_QCMPLX:
        if (CONVAL1G(tconval) == stb.quad0 && CONVAL2G(tconval) == stb.quad0)
          goto do_zeroes;
        break;
#endif
#ifdef LONG_DOUBLE_FLOAT128
      case TY_FLOAT128:
        if (tconval == stb.float128_0)
          goto do_zeroes;
        break;
      case TY_CMPLX128:
        if (CONVAL1G(tconval) == stb.float128_0 &&
            CONVAL2G(tconval) == stb.float128_0)
          goto do_zeroes;
        break;
#endif /* LONG_DOUBLE_FLOAT128 */
      default:
        break;
      }
    }
    /* emit data value, loop if repeat count present */
    putval = 1;
    if (size_of_item == 0) {
      putval = 0;
      *repeat_cnt = 1;
      break;
    }
    do {
      if (DTY(tdtype) != TY_PTR && DTY(tdtype) != TY_STRUCT) {
        if (*ptrcnt) {
          if (!first_data)
            fprintf(ASMFIL, ", ");
          *cptr = put_next_member(*cptr);
          fprintf(ASMFIL, " [" /*]*/);
          *ptrcnt = 0;
        } else if (!(*i8cnt)) {
          if (!first_data)
            fprintf(ASMFIL, ", ");
          *cptr = put_next_member(*cptr);
          fprintf(ASMFIL, " [" /*]*/);
        } else if (*i8cnt) {
          if (!first_data)
            fprintf(ASMFIL, ", ");
        }
      }
      switch (DTY(tdtype)) {
      case TY_INT8:
      case TY_LOG8:
      case TY_DWORD:
        if (DBGBIT(5, 32)) {
          fprintf(gbl.dbgfil,
                  "emit_init:put_i32 first_data:%d i8cnt:%ld ptrcnt:%d\n",
                  first_data, *i8cnt, *ptrcnt);
        }
        put_i32(CONVAL2G(tconval));
        fprintf(ASMFIL, ", ");
        if (DBGBIT(5, 32)) {
          fprintf(gbl.dbgfil,
                  "emit_init:put_i32 first_data:%d i8cnt:%ld ptrcnt:%d\n",
                  first_data, *i8cnt, *ptrcnt);
        }
        if (XBIT(124, 0x400)) {
          put_i32(CONVAL1G(tconval));
        } else {
          put_i32(0);
        }
        break;

      case TY_INT:
      case TY_LOG:
      case TY_WORD:
        if (DBGBIT(5, 32)) {
          fprintf(gbl.dbgfil,
                  "emit_init:put_i32 first_data:%d i8cnt:%ld ptrcnt:%d\n",
                  first_data, *i8cnt, *ptrcnt);
        }
        put_i32(tconval);
        break;
      case TY_SINT:
      case TY_SLOG:
        if (DBGBIT(5, 32)) {
          fprintf(gbl.dbgfil,
                  "emit_init:put_i16 first_data:%d i8cnt:%ld ptrcnt:%d\n",
                  first_data, *i8cnt, *ptrcnt);
        }
        put_i16((int)tconval);
        break;

      case TY_BINT:
      case TY_BLOG:
        if (DBGBIT(5, 32)) {
          fprintf(gbl.dbgfil,
                  "emit_init:put_i8 first_data:%d i8cnt:%ld ptrcnt:%d\n",
                  first_data, *i8cnt, *ptrcnt);
        }
        put_i8((int)tconval);
        break;

      case TY_FLOAT:
        if (DBGBIT(5, 32)) {
          fprintf(gbl.dbgfil,
                  "emit_init:put_r4 first_data:%d i8cnt:%ld ptrcnt:%d\n",
                  first_data, *i8cnt, *ptrcnt);
        }
        put_r4(tconval);
        break;

      case TY_DBLE:
        if (DBGBIT(5, 32)) {
          fprintf(gbl.dbgfil,
                  "emit_init:put_r8 first_data:%d i8cnt:%ld ptrcnt:%d\n",
                  first_data, *i8cnt, *ptrcnt);
        }
        put_r8((int)tconval, putval);
        break;

#ifdef TARGET_SUPPORTS_QUADFP
      case TY_QUAD:
        if (DBGBIT(5, 32)) {
          fprintf(gbl.dbgfil,
                  "emit_init:put_r18 first_data:%d i8cnt:%ld ptrcnt:%d\n",
                  first_data, *i8cnt, *ptrcnt);
        }
        put_r16((int)tconval, putval);
        break;
#endif

      case TY_CMPLX:
        if (DBGBIT(5, 32)) {
          fprintf(gbl.dbgfil,
                  "emit_init:put_cmplx_n first_data:%d i8cnt:%ld ptrcnt:%d\n",
                  first_data, *i8cnt, *ptrcnt);
        }
        put_cmplx_n((int)tconval, putval);
        break;

      case TY_DCMPLX:
        if (DBGBIT(5, 32)) {
          fprintf(gbl.dbgfil,
                  "emit_init:put_dcmplx_n first_data:%d i8cnt:%ld ptrcnt:%d\n",
                  first_data, *i8cnt, *ptrcnt);
        }
        put_dcmplx_n((int)tconval, putval);
        break;

#ifdef TARGET_SUPPORTS_QUADFP
      case TY_QCMPLX:
        if (DBGBIT(5, 32)) {
          fprintf(gbl.dbgfil,
                  "emit_init:put_qcmplx_n first_data:%d i8cnt:%ld ptrcnt:%d\n",
                  first_data, *i8cnt, *ptrcnt);
        }
        put_qcmplx_n((int)tconval, putval);
        break;
#endif

      case TY_PTR:
        if (*i8cnt) {
          fprintf(ASMFIL, /*[*/ "], ");
        } else if (!first_data)
          fprintf(ASMFIL, ", ");
        *ptrcnt = *ptrcnt + 1;
        *i8cnt = 0;
        oldptr = *cptr;
        *cptr = put_next_member(*cptr);

        if (DBGBIT(5, 32)) {
          fprintf(gbl.dbgfil,
                  "emit_init:put_addr first_data:%d i8cnt:%ld ptrcnt:%d\n",
                  first_data, *i8cnt, *ptrcnt);
        }
        if (STYPEG(tconval) != ST_CONST) {
          put_addr(SPTR_NULL, tconval, DT_NONE);
        } else {
          SPTR conval1 = SymConval1((SPTR)tconval);
          ISZ_T conval2 = CONVAL2G(tconval);

          if ((!conval1) && (conval2)) {
            fprintf(ASMFIL, "inttoptr (i64 %ld to ", (long)conval2);
            oldptr = put_next_member(oldptr);
            assert(oldptr == *cptr, "emit_init:unexpected behaviour", (int)conval2, ERR_Severe);
            fprintf(ASMFIL, ")");
          } else
            put_addr(conval1, conval2, DT_NONE);
        }
        break;

      case TY_CHAR:
        size_of_item = DTY(DTYPEG(tconval) + 1);
        if (DBGBIT(5, 32)) {
          fprintf(gbl.dbgfil,
                  "emit_init:put_str_n first_data:%d i8cnt:%ld ptrcnt:%d\n",
                  first_data, *i8cnt, *ptrcnt);
        }
        put_string_n(stb.n_base + CONVAL1G((int)tconval),
                     DTY(DTYPEG((int)tconval) + 1), 0);
        break;

      case TY_NCHAR:
        /* need to write nchar in 2 bytes because we make everything as i8 */
        put_ncharstring_n(stb.n_base + CONVAL1G((int)tconval),
                          DTY(DTYPEG((int)tconval) + 1), 16);
        break;

      case TY_STRUCT:
        if (is_empty_typedef(tdtype)) {
          break;
        }
        // TODO: Print the struct?
        break;

#ifdef LONG_DOUBLE_FLOAT128
      case TY_X87:
        put_r8(CONVAL1G(tconval), putval);
        fputc(',', ASMFIL);
        put_r8(CONVAL2G(tconval), putval);
        fputc(',', ASMFIL);
        put_r8(CONVAL3G(tconval), putval);
        fputc(',', ASMFIL);
        put_r8(0, putval);
        put_r8(CONVAL4G(tconval), putval);
        fputc(',', ASMFIL);
        break;
      case TY_X87CMPLX:
        put_r8(CONVAL1G(CONVAL1G(tconval)), putval);
        fputc(',', ASMFIL);
        put_r8(CONVAL2G(CONVAL1G(tconval)), putval);
        fputc(',', ASMFIL);
        put_r8(CONVAL3G(CONVAL1G(tconval)), putval);
        fputc(',', ASMFIL);
        put_r8(CONVAL4G(CONVAL1G(tconval)), putval);
        fputc(',', ASMFIL);
        put_r8(0, putval);
        put_r8(CONVAL1G(CONVAL2G(tconval)), putval);
        fputc(',', ASMFIL);
        put_r8(CONVAL2G(CONVAL2G(tconval)), putval);
        fputc(',', ASMFIL);
        put_r8(CONVAL3G(CONVAL2G(tconval)), putval);
        fputc(',', ASMFIL);
        put_r8(CONVAL4G(CONVAL2G(tconval)), putval);
        fputc(',', ASMFIL);
        put_r8(0, putval);
        break;
#endif /* LONG_DOUBLE_FLOAT128 */

      default:
        interr("emit_init:bad dt", tdtype, ERR_Severe);
      }
      *addr += size_of_item;
      if (DTY(tdtype) != TY_PTR)
        *i8cnt = *i8cnt + size_of_item;
      putval = 0;
      first_data = 0;

    } while (--(*repeat_cnt));
    *repeat_cnt = 1;
    break;
  do_zeroes:
    if (*ptrcnt) {
      if (!first_data)
        fprintf(ASMFIL, ", ");
      *cptr = put_next_member(*cptr);
      fprintf(ASMFIL, " [" /*]*/);
      *ptrcnt = 0;
    } else if (!(*i8cnt)) {
      if (!first_data)
        fprintf(ASMFIL, ", ");
      *cptr = put_next_member(*cptr);
      fprintf(ASMFIL, " [" /*]*/);
    } else if (*i8cnt) {
      if (!first_data)
        fprintf(ASMFIL, ", ");
    }
    if (DBGBIT(5, 32)) {
      fprintf(gbl.dbgfil,
              "emit_init:put_zeroes at end first_data:%d i8cnt:%ld ptrcnt:%d\n",
              first_data, *i8cnt, *ptrcnt);
    }
    put_zeroes((*repeat_cnt) * size_of_item);
    *i8cnt = *i8cnt + (*repeat_cnt) * size_of_item;
    *addr += (*repeat_cnt) * size_of_item;
    *repeat_cnt = 1;
    first_data = 0;
    break;
  }
}

void
put_string_n(const char *p, ISZ_T len, int size)
{
  int n;
  char ch;
  const char *ptrch = "i8";
  char chnm[10];

  /* check for wide string - size is given by caller */
  if (size) {
    snprintf(chnm, sizeof(chnm), "i%d", size);
    ptrch = chnm;
  }

  if (len == 0) {
    fprintf(ASMFIL, "%s 0", ptrch);
    return;
  }
  n = 0;
  while (len--) {
    ch = *p;
    fprintf(ASMFIL, "%s %u", ptrch, ch & 0xff);
    if (len)
      fprintf(ASMFIL, ",");
    ++p;
    ++n;
  }
} /* put_string_n */

static void
put_ncharstring_n(char *p, ISZ_T len, int size_of_char)
{
  int n, bytes;
  const char *ptrch = "i8";
  union {
    char a[2];
    short i;
  } chtmp;

  if (len == 0) {
    fprintf(ASMFIL, "%s 0,", ptrch);
    fprintf(ASMFIL, "%s 0", ptrch);
    return;
  }
  n = 0;

  while (len > 0) {
    int val = kanji_char((unsigned char *)p, len, &bytes);
    p += bytes;
    len -= bytes;
    chtmp.i = val;
    fprintf(ASMFIL, "%s %u, ", ptrch, chtmp.a[0] & 0xff);
    fprintf(ASMFIL, "%s %u", ptrch, chtmp.a[1] & 0xff);
    if (len)
      fprintf(ASMFIL, ",");
  }

} /* put_ncharstring_n */

static void
put_zeroes(ISZ_T len)
{
  ISZ_T i;
  i = len;
  while (i > 32) {
    fprintf(ASMFIL, "i8 0,i8 0,i8 0,i8 0,i8 0,i8 0,i8 0,i8 0,i8 0,i8 0,i8 0,i8 "
                    "0,i8 0,i8 0,i8 0,i8 0,i8 0,i8 0,i8 0,i8 0,i8 0,i8 0,i8 "
                    "0,i8 0,i8 0,i8 0,i8 0,i8 0,i8 0,i8 0,i8 0,i8 0");
    i -= 32;
    if (i)
      fprintf(ASMFIL, ",");
  }
  if (i) {
    while (1) {
      fprintf(ASMFIL, "i8 0");
      i--;
      if (i == 0)
        break;
      fprintf(ASMFIL, ",");
    }
  }
}

static void
put_i8(int val)
{
  i8bit.i8 = (short)val;
  fprintf(ASMFIL, "i8 %u", i8bit.byte[0] & 0xff);
}

/* write:  i8 x1, i8 x2 */
static void
put_i16(int val)
{
  int i;
  i16bit.i16 = val;
  for (i = 0; i < 2; i++) {
    fprintf(ASMFIL, "i8 %u", i16bit.byte[i] & 0xff);
    if (i < 1)
      fprintf(ASMFIL, ",");
  }
}

/* write:  i8 0x?, i8 0x?, i8 0x?, i8 0x? */
void
put_i32(int val)
{
  int i;
  i32bit.i32 = val;
  for (i = 0; i < 4; i++) {
    fprintf(ASMFIL, "i8 %u", i32bit.byte[i] & 0xff);
    if (i < 3)
      fprintf(ASMFIL, ", ");
  }
}

void
put_short(int val)
{
  fprintf(ASMFIL, "i16 %u", val);
}

/* write:  i8 0x?, i8 0x?, i8 0x?, i8 0x? */
static void
put_r4(INT val)
{
  int i;
  i32bit.i32 = val;
  for (i = 0; i < 4; i++) {
    fprintf(ASMFIL, "i8 %u", i32bit.byte[i] & 0xff);
    if (i < 3)
      fprintf(ASMFIL, ",");
  }
}

static void
put_r8(int sptr, int putval)
{
  INT num[2];

  num[0] = CONVAL1G(sptr);
  num[1] = CONVAL2G(sptr);
  if (flg.endian) {
    put_r4(num[0]);
    fprintf(ASMFIL, ",");
    put_r4(num[1]);
  } else {
    put_r4(num[1]);
    fprintf(ASMFIL, ",");
    put_r4(num[0]);
  }
}

#ifdef TARGET_SUPPORTS_QUADFP
static void put_r16(int sptr, int putval)
{
  INT num[4];

  num[0] = CONVAL1G(sptr);
  num[1] = CONVAL2G(sptr);
  num[2] = CONVAL3G(sptr);
  num[3] = CONVAL4G(sptr);
  if (flg.endian) {
    put_r4(num[0]);
    fprintf(ASMFIL, ",");
    put_r4(num[1]);
    fprintf(ASMFIL, ",");
    put_r4(num[2]);
    fprintf(ASMFIL, ",");
    put_r4(num[3]);
  } else {
    put_r4(num[3]);
    fprintf(ASMFIL, ",");
    put_r4(num[2]);
    fprintf(ASMFIL, ",");
    put_r4(num[1]);
    fprintf(ASMFIL, ",");
    put_r4(num[0]);
  }
}
#endif

static void
put_cmplx_n(int sptr, int putval)
{
  put_r4(CONVAL1G(sptr));
  fprintf(ASMFIL, ",");
  put_r4(CONVAL2G(sptr));
}

static void
put_dcmplx_n(int sptr, int putval)
{
  put_r8((int)CONVAL1G(sptr), putval);
  fprintf(ASMFIL, ",");
  put_r8((int)CONVAL2G(sptr), putval);
}

#ifdef TARGET_SUPPORTS_QUADFP
static void put_qcmplx_n(int sptr, int putval)
{
  put_r16((int)CONVAL1G(sptr), putval);
  fprintf(ASMFIL, ",");
  put_r16((int)CONVAL2G(sptr), putval);
}
#endif

/**
   \brief Generate an expression to add an offset to a ptr
   \param offset    the addend
   \param ret_type  the type of the pointer (LL_Type)
   \param ptr_nm    the identifier for the pointer

   For example,
   <pre>
     getelementptr(i8* bitcast(<ret_type> <ptr_nm> to i8*), i32 <offset>)
   </pre>

   The caller expects a string that is an i8*.
 */
LL_Value *
gen_ptr_offset_val(int offset, LL_Type *ret_type, const char *ptr_nm)
{
  /* LL_Type for i8* (used as bitcast target for GEP to get byte offsets) */
  LL_Type *ll_type_i8_ptr =
      ll_get_pointer_type(ll_create_int_type(cpu_llvm_module, 8));

  /* Create an LL_Value from LL_Type ... */
  LL_Value *llv = ll_create_value_from_type(cpu_llvm_module, ret_type, ptr_nm);
  /*... and use it to generate a bitcast instruction to i8* */
  llv = ll_get_const_bitcast(cpu_llvm_module, llv, ll_type_i8_ptr);
  /*... then use it to generate a GEP instruction to get element at a byte
   * offset */
  llv = ll_get_const_gep(cpu_llvm_module, llv, 1, offset);
  return llv;
}

/**
   \brief Produce a getelementptr that would fetch a value out of one of the
   global structs.

   Print something along the lines of (LLVM version dependent):
   \verbatim
     getelementptr (i8* bitcast (%struct$name* @getsname(sptr)
                    to i8*), i32 0) to i8*)
   \endverbatim
 */
void
put_addr(SPTR sptr, ISZ_T off, DTYPE dtype)
{
  const char *name, *elem_type;
  bool is_static_or_common_block_var, in_fortran;

  in_fortran = false;
  in_fortran = true;

  /* Static and common block variables require special handling for now */
  is_static_or_common_block_var = (SCG(sptr) == SC_STATIC);
#ifdef SC_CMBLK
  is_static_or_common_block_var =
      is_static_or_common_block_var || (SCG(sptr) == SC_CMBLK);
#endif

  elem_type = "";
  /* Decide whether we need to provide element type to GEP */
  if (ll_feature_explicit_gep_load_type(&cpu_llvm_module->ir))
    elem_type = "i8, ";

  if (sptr) {
    if ((name = getsname(sptr))) {
      if (is_static_or_common_block_var && in_fortran) {
        /* Statics and common block initializations in Fortran are
         * strings composed using fprintf;
         * FIXME need to generate them using LL_Value, like below */

        /* Statics and common blocks in fortran as stored as structs, we need to
         * add offset inside of struct */
        off += ADDRESSG(sptr);

        /* Text type for contansts is produced via char_type */
        if (STYPEG(sptr) == ST_CONST)
          fprintf(ASMFIL,
                  "getelementptr(%sptr @%s, i32 %ld)",
                  elem_type, name, off);
        /* Structures have type name mirroring variable name */
        else
          fprintf(
              ASMFIL,
              "getelementptr(%sptr @%s, i32 %ld)",
              elem_type, name, off);
      } else {
        /* Convert to LLVM-compatible structures */
        if (!LLTYPE(sptr)) {
          process_sptr(sptr);
        }

        LL_Type *ll_type = LLTYPE(sptr);

        /* Convert to pointer type if needed
         * TODO implications of this are unclear, it works for now
         * because this is only used in data initialization */
        if (ll_type->data_type != LL_PTR && need_ptr(sptr, SCG(sptr), dtype))
          ll_type = ll_get_pointer_type(ll_type);

        /* Produce pointer offset code */
        LL_Value *ll_offset = gen_ptr_offset_val(off, ll_type, SNAME(sptr));
        fprintf(ASMFIL, "%s", ll_offset->data);
      }
    } else
      fprintf(ASMFIL, "null");
  } else if (off == 0)
    fprintf(ASMFIL, "null");
  else
    fprintf(ASMFIL, "%ld", (long)off);
}

DTYPE
mk_struct_for_llvm_init(const char *name, int size)
{
  int tag;
  DTYPE dtype;
  char sname[MXIDLN];

  snprintf(sname, sizeof(sname), "struct%s", name);
  dtype = cg_get_type(6, TY_STRUCT, NOSYM);
  tag = getsymbol(sname);
  DTYPEP(tag, dtype);
  DTY(dtype + 2) = 0; /* size */
  DTY(dtype + 3) = tag;
  DTY(dtype + 4) = 0; /* align */
  DTY(dtype + 5) = 0;
  if (size == 0)
    process_ftn_dtype_struct(dtype, sname, true);
  return dtype;
}

int
add_member_for_llvm(SPTR sym, int prev, DTYPE dtype, ISZ_T size)
{
  SPTR mem = insert_sym_first(sym);
  if (prev > NOSYM)
    SYMLKP(prev, mem);
  DTYPEP(mem, dtype);
  SYMLKP(mem, NOSYM);
  PSMEMP(mem, mem);
  VARIANTP(mem, prev);
  STYPEP(mem, ST_MEMBER);
  CCSYMP(mem, 1);
  ADDRESSP(mem, size);
  SCP(mem, SC_NONE);
  return mem;
}

/**
   \brief Add initilizer routine 'initroutine' to the llvm global ctor array
 */
void
add_init_routine(char *initroutine)
{
  /* Current assumption is that initroutine has a type void with no argument.
     If type of init routine is not void and with argument then bitcast is
     needed.
   */
  llvm_ctor_add(initroutine);
}

/* Add the constructor responsible for initializing the libhugetlb
 * functionality.  The code behind the initialization exists in the following
 * directory: <pgi_root>/pds/libhugetlb/.
 */
void
init_huge_tlb(void)
{
  add_ctor("__pgi_huge_pages_init_zero");
}

/** \brief Add the constructor responsible for -Mflushz */
void
init_flushz(void)
{
  add_ctor("__flushz");
}

/** \brief Add the constructor responsible for -Mdaz */
void
init_daz(void)
{
  add_ctor("__daz");
}

/** \brief Add the constructor responsible for -Ktrap */
void
init_ktrap(void)
{
  add_ctor("__ktrap");
  /* it would be better if there were a llutil process to ultimately emit
   * the global & its init  -- I know can create a symbol & dinit(), but
   * that's too much overhead.
   * A potential probleme is that the name may need an ABI adjustment. e.g,.
   * OSX prepends an underscore to user globals.
   */
  fprintf(ASMFIL, "@__ktrapval = global i32 %d, align 4\n", flg.x[24] & 0x1f9);
}

