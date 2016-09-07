//===-- ELWrapper.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <stdio.h>

// EditLine editor function return codes.
// For user-defined function interface
#define CC_NORM 0
#define CC_NEWLINE 1
#define CC_EOF 2
#define CC_ARGHACK 3
#define CC_REFRESH 4
#define CC_CURSOR 5
#define CC_ERROR 6
#define CC_FATAL 7
#define CC_REDISPLAY 8
#define CC_REFRESH_BEEP 9

// el_set/el_get parameters
#define EL_PROMPT 0      // , el_pfunc_t
#define EL_TERMINAL 1    // , const char *
#define EL_EDITOR 2      // , const char *
#define EL_SIGNAL 3      // , int);
#define EL_BIND 4        // , const char *, ..., NULL
#define EL_TELLTC 5      // , const char *, ..., NULL
#define EL_SETTC 6       // , const char *, ..., NULL
#define EL_ECHOTC 7      // , const char *, ..., NULL
#define EL_SETTY 8       // , const char *, ..., NULL
#define EL_ADDFN 9       // , const char *, const char *, el_func_t
#define EL_HIST 10       // , hist_fun_t, const char *
#define EL_EDITMODE 11   // , int
#define EL_RPROMPT 12    // , el_pfunc_t
#define EL_GETCFN 13     // , el_rfunc_t
#define EL_CLIENTDATA 14 // , void *
#define EL_UNBUFFERED 15 // , int
#define EL_PREP_TERM 16  // , int
#define EL_GETTC 17      // , const char *, ..., NULL
#define EL_GETFP 18      // , int, FILE **
#define EL_SETFP 19      // , int, FILE *
#define EL_REFRESH 20    // , void
#define EL_PROMPT_ESC 21 // , prompt_func, Char);              set/get

#define EL_BUILTIN_GETCFN (NULL)

// history defines
#define H_FUNC 0        // , UTSL
#define H_SETSIZE 1     // , const int
#define H_GETSIZE 2     // , void
#define H_FIRST 3       // , void
#define H_LAST 4        // , void
#define H_PREV 5        // , void
#define H_NEXT 6        // , void
#define H_CURR 8        // , const int
#define H_SET 7         // , int
#define H_ADD 9         // , const char *
#define H_ENTER 10      // , const char *
#define H_APPEND 11     // , const char *
#define H_END 12        // , void
#define H_NEXT_STR 13   // , const char *
#define H_PREV_STR 14   // , const char *
#define H_NEXT_EVENT 15 // , const int
#define H_PREV_EVENT 16 // , const int
#define H_LOAD 17       // , const char *
#define H_SAVE 18       // , const char *
#define H_CLEAR 19      // , void
#define H_SETUNIQUE 20  // , int
#define H_GETUNIQUE 21  // , void
#define H_DEL 22        // , int

struct EditLine {};

struct LineInfo {
  const char *buffer;
  const char *cursor;
  const char *lastchar;
};

struct History {};

struct HistEvent {
  int num;
  const char *str;
};

extern "C" {
// edit line API
EditLine *el_init(const char *, FILE *, FILE *, FILE *);
const char *el_gets(EditLine *, int *);
int el_set(EditLine *, int, ...);

void el_end(EditLine *);
void el_reset(EditLine *);
int el_getc(EditLine *, char *);
void el_push(EditLine *, const char *);
void el_beep(EditLine *);
int el_parse(EditLine *, int, const char **);
int el_get(EditLine *, int, ...);
int el_source(EditLine *, const char *);
void el_resize(EditLine *);
const LineInfo *el_line(EditLine *);
int el_insertstr(EditLine *, const char *);
void el_deletestr(EditLine *, int);

// history API
History *history_init(void);
void history_end(History *);
int history(History *, HistEvent *, int, ...);
};