/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
 * \brief Error handling and reporting module.
 */
#include "error.h"
#include "global.h"
#include "version.h"
#include "main.h"
#include "symtab.h"
#ifdef FE90
#include "ast.h"
#include "scan.h"
#include <string.h>
#endif
#include <stdarg.h>

/* second time -- include error text definitions from errmsg utility */
#define ERRMSG_GET_ERRTXT_TABLE 1
#include "errmsgdf.h"
#undef ERRMSG_GET_ERRTXT_TABLE

static int ndiags[ERR_SEVERITY_SIZE];
static enum error_severity maxfilsev; /* max severity for entire source file */
static int totaldiags = 0; /* total number of messages for this source
                            * file */
static int emit_errmsg = 1;

/** \brief Expand error message template: replace '$' with text of the other
 * two operands
 *
 * \param intxt error message template
 * \param op1 first operand for substitution
 * \param op2 second operand for substitution
 * \return pointer to result
 */
static char *
errfill(const char *intxt, const char *op1, const char *op2)
{
  static char outtxt[200]; /* holds result string */
  char *p;                 /* points into outtxt */
  const char *op;

  /* calculate length of txt and operands to avoid overflow */
  int intxt_len;
  int op_len, op1_len, op2_len;
  int op_adj_len, op2_adj_len;
  int buf_len, tot_len, len_left;

  buf_len = 200;
  op_adj_len = op2_adj_len = 0;
  intxt_len = (intxt != NULL) ? strlen(intxt) : 0;
  op1_len = op_len = (op1 != NULL) ? strlen(op1) : 0;
  op2_len = (op2 != NULL) ? strlen(op2) : 0;
  tot_len = intxt_len + op1_len + op2_len;
  len_left = buf_len;
  if (tot_len > buf_len) {
    len_left = buf_len - intxt_len;
    if (!op2_len)
      op_adj_len = len_left;
    else {
      if (op_len > len_left / 2) {
        if (op2_len > len_left / 2)
          op_adj_len = op2_adj_len = len_left / 2;
        else
          op_adj_len = len_left - op2_len;
      } else
        op2_adj_len = len_left - op1_len;
    }
  }

  p = outtxt;
  op = op1;

  while ((*p = *intxt++) != 0) {
    if (*p++ == '$') {
      p--;
      if (op == 0)
        op = "";
      if (tot_len > buf_len) {
        if (op_adj_len && (op_len != op_adj_len)) {
          strncpy(p, op, op_adj_len - 3);
          p += op_adj_len - 3;
          strcpy(p, "...");
          p += 3;
        } else {
          strncpy(p, op, op_len);
          p += op_len;
        }
        op_len = op2_len;
        op_adj_len = op2_adj_len;
      } else {
        strcpy(p, op);
        p += strlen(op);
      }
      op = op2;
    }
  }
  return outtxt;
}

static void
display_error(error_code_t ecode, enum error_severity sev, int eline,
              const char *op1, const char *op2, int col, const char *srcFile)
{
  static char sevlett[5] = {'X', 'I', 'W', 'S', 'F'};
  const char *formatstr;
  char buff[400];
  int lastmsg;
  const char *msgstr;

  if (sev < ERR_Informational || sev > ERR_Fatal)
    sev = ERR_Fatal;
  /*  check if informationals and warnings are inhibited  */
  if (gbl.nowarn && sev <= ERR_Warning)
    return;
  /* don't count informs if -inform warn */
  if (sev > ERR_Informational || sev >= flg.inform)
    ndiags[sev]++;

  if ((sev > ERR_Informational || sev >= flg.inform) && sev > gbl.maxsev) {
    gbl.maxsev = sev;
    if (sev > maxfilsev)
      maxfilsev = sev;
  }

  if (sev >= flg.inform) {
    if (gbl.curr_file != NULL || srcFile != NULL) {
      if (eline) {
        if (col > 0)
          formatstr = "%s-%c-%04d-%s (%s: %d.%d)";
        else
          formatstr = "%s-%c-%04d-%s (%s: %d)";
      } else
        formatstr = "%s-%c-%04d-%s (%s)";
    } else
      formatstr = "%s-%c-%04d-%s";

    lastmsg = sizeof(errtxt) / sizeof(char *);
    if (ecode < lastmsg) {
      msgstr = errtxt[ecode];
    } else {
      msgstr = "Unknown error code";
    }

    if (!XBIT(0, 0x40000000) && col <= 0 && srcFile == NULL)
      snprintf(&buff[1], sizeof(buff) - 1, formatstr, version.lang,
               sevlett[sev], ecode, errfill(msgstr, op1, op2), gbl.curr_file,
               eline);
    else {
      static const char *sevtext[5] = {"X", "info", "warning", "error", "error"};
      if (col > 0 && (srcFile != NULL || gbl.curr_file != NULL)) {
        snprintf(&buff[1], sizeof(buff) - 1, "\n%s:%d:%d: %s %c%04d: %s",
                 (srcFile != NULL) ? srcFile : gbl.curr_file, eline, col,
                 sevtext[sev], sevlett[sev], ecode, errfill(msgstr, op1, op2));
      } else if (srcFile != NULL) {
        snprintf(&buff[1], sizeof(buff) - 1, "\n%s:%d: %s %c%04d: %s", srcFile,
                 eline, sevtext[sev], sevlett[sev], ecode,
                 errfill(msgstr, op1, op2));
      } else if (gbl.curr_file != NULL) {
        snprintf(&buff[1], sizeof(buff) - 1, "%s(%d) : %s %c%04d : %s",
                 gbl.curr_file, eline, sevtext[sev], sevlett[sev], ecode,
                 errfill(msgstr, op1, op2));
      } else
        snprintf(&buff[1], sizeof(buff) - 1, "%s : %s %c%04d : %s", "",
                 sevtext[sev], sevlett[sev], ecode, errfill(msgstr, op1, op2));
    }
    if (emit_errmsg)
      fprintf(stderr, "%s\n", &buff[1]);
#if DEBUG
    if (DBGBIT(0, 2))
      fprintf(gbl.dbgfil, "%s\n", &buff[1]);
#endif
    if (flg.list || flg.code || flg.xref) {
      if (flg.dbg[14]) {
        buff[0] = '#'; /* make sure listing is assembleable */
        list_line(buff);
      } else {
        list_line(&buff[1]);
      }
    }
  }

  if (sev == ERR_Fatal) {
#ifdef FE90
#if DEBUG
    if (ecode == 7 && DBGBIT(0, 512))
      dump_stg_stat("- subprogram too large");
#endif
#endif
    if (col <= 0 || (srcFile == NULL && gbl.curr_file == NULL)) {
      finish();
    }
  }

  if (sev >= ERR_Severe)
    totaldiags++;

  if (totaldiags >= flg.errorlimit && !DBGBIT(0, 64))
    errfatal(F_0008_Error_limit_exceeded);
}

void
errini(void)
{
  ndiags[1] = ndiags[2] = ndiags[3] = ndiags[4] = gbl.maxsev = totaldiags = 0;
}

void
errversion(void)
{
  fprintf(stderr, "%s/%s %s %s%s%s\n", version.lang, version.target,
          version.host, version.vsn, version.product, version.bld);
  fprintf(stderr, "%s\n", version.copyright);
}

void
error(error_code_t ecode, enum error_severity sev, int eline, const char *op1,
      const char *op2)
{
  display_error(ecode, sev, eline, op1, op2, 0, NULL);
}

void
errlabel(error_code_t ecode, enum error_severity sev, int eline, char *nm,
         const char *op2)
{
  nm += 2; /* skip past .L */
  while (*nm == '0')
    nm++; /* skip over leading 0's */
  if (*nm == 0)
    nm--;
  error(ecode, sev, eline, nm, op2);
}

/* Do printf-style formatting of the message by: compute the size of the
 * required buffer, allocate it, sprintf into it, then free it. */
void
interrf(enum error_severity sev, const char *fmt, ...)
{
  size_t size;
  char *buffer;
  va_list ap;

#if !DEBUG
  if (sev == ERR_Informational)
    return;
#endif
  va_start(ap, fmt);
  size = vsnprintf(NULL, 0, fmt, ap);
  va_end(ap);
  NEW(buffer, char, size + 1);
  va_start(ap, fmt);
  vsprintf(buffer, fmt, ap);
  va_end(ap);
  error(V_0000_Internal_compiler_error_OP1_OP2, sev, gbl.lineno, buffer, 0);
  FREE(buffer);
}

void
interr(const char *txt, int val, enum error_severity sev)
{
  interrf(sev, "%s %7d", txt, val);
}

#if DEBUG
void
dassert_err(const char *filename, int line, const char *expr, const char *txt)
{
  /* Since we reach here only in DEBUG mode, there's no point in
     being clever about creating a single string to pass to interr.
     Just get the information to the compiler developer via stderr. */
  (void)fprintf(stderr, "%s:%d: DEBUG_ASSERT %s failed\n", filename, line,
                expr);
  interr(txt, 0, error_max_severity());
}

void
asrt_failed(const char *filename, int line)
{
  fprintf(stderr, "asrt failed. line %d, file %s\n", line, filename);
  /* Call interr so that we have a common place to set a breakpoint when
     running under a debugger. */
  interr("asrt failed", 0, ERR_Warning);
}
#endif

char *
errnum(int num)
{
  static char n[20];
  sprintf(n, "%d", num);
  return n;
} /* errnum */

void
errinfo(error_code_t ecode)
{
  error(ecode, ERR_Informational, gbl.lineno, CNULL, CNULL);
}

void
errwarn(error_code_t ecode)
{
  error(ecode, ERR_Warning, gbl.lineno, CNULL, CNULL);
}

void
errsev(error_code_t ecode)
{
  error(ecode, ERR_Severe, gbl.lineno, CNULL, CNULL);
}

void
errfatal(error_code_t ecode)
{
  error(ecode, ERR_Fatal, gbl.lineno, CNULL, CNULL);
}

int
summary(bool final, int ipafollows)
{
  static const char *t[5] = {
      "%s/%s %s %s%s%s: compilation successful\n",
      "%s/%s %s %s%s%s: compilation completed with informational messages\n",
      "%s/%s %s %s%s%s: compilation completed with warnings\n",
      "%s/%s %s %s%s%s: compilation completed with severe errors\n",
      "%s/%s %s %s%s%s: compilation aborted\n"};
  static bool empty_file = true;

  if (!final) {
    if (!flg.terse || gbl.maxsev > 1)
      fprintf(stderr,
              "%3d inform, %3d warnings, %3d severes, %1d fatal for %s\n",
              ndiags[1], ndiags[2], ndiags[3], ndiags[4], SYMNAME(gbl.currsub));
    empty_file = false;
  } else if (!empty_file || !ipafollows) {
    if (empty_file && maxfilsev < 3)
      errwarn(S_0006_Input_file_empty);
    if (!flg.terse || gbl.maxsev > 1)
      fprintf(stderr, t[maxfilsev], version.lang, version.target, version.host,
              version.vsn, version.product, version.bld);
  }
  return maxfilsev;
}

void
erremit(int x)
{
  emit_errmsg = x;
}

void
fperror(int errcode)
{
  /* floating point error codes */
  static struct {
    int ovf;
    int unf;
    int invop;
  } lineno = {-1, -1, -1};

  gbl.fperror_status = errcode;
  if (gbl.nofperror)
    return;
  switch (errcode) {
  case FPE_NOERR:
    break;
  case FPE_FPOVF: /* floating point overflow */
    if (lineno.ovf == gbl.lineno)
      break;
    lineno.ovf = gbl.lineno;
    errwarn((enum error_code)129); // FIXME: different enum names per target
    break;
  case FPE_FPUNF: /* floating point underflow */
    if (lineno.unf == gbl.lineno)
      break;
    lineno.unf = gbl.lineno;
    errwarn((enum error_code)130); // FIXME: different enum names per target
    break;
  case FPE_INVOP: /* invalid operand */
    if (lineno.invop == gbl.lineno)
      break;
    lineno.invop = gbl.lineno;
    errwarn((enum error_code)132); // FIXME: different enum names per target
    break;
  default:
    interr("invalid floating point error code", (int)errcode, ERR_Severe);
  }
}

enum error_severity
error_max_severity(void)
{
  return maxfilsev;
}

#ifdef FE90

/** \brief Returns the last substring of a string.
 *
 * This function is used by callers of errWithSrc() below to obtain the
 * last substring of a string. In some cases the operand of an error message
 * has extra words. This is the case with syntax errors where we call the
 * function prettytoken() prior to generating the syntax error. The offending
 * token is typically the last substring. Each substring is separated by one
 * space.
 *
 * We use this substring in column deduction in errWithSrc().
 *
 * \param ptoken is the token string we are processing.
 *
 * \return the last substring, else NULL
 */
char *
getDeduceStr(char *ptoken)
{
  char *lastToken;
  if (ptoken != NULL) {
    lastToken = strrchr(ptoken, ' ');
    if (lastToken != NULL) {
      lastToken++;
    }
  } else {
    lastToken = NULL;
  }
  return lastToken;
}

/** \brief Construct and issue an "enhanced" error message.
 *
 * Construct error message and issue it to user terminal and to listing file
 * if appropriate. This is an "enhanced" error message which means we will
 * also display the source line, column number, and location of the error.
 *
 * Note: First five arguments are the same as function error().
 *
 * \param ecode      Error number
 *
 * \param sev        Error severity (a value in the err_severity enum)
 *
 * \param eline      Source file line number
 *
 * \param op1        String to be expanded into error message * or 0
 *
 * \param op2        String to be expanded into error message * or 0
 *
 * \param col        The column number where the error occurred at if
 *                   available, else 0.
 *
 * \param deduceCol  The operand to use (1 for op1, 2 for op2) to deduce the
 *                   the column number when the col argument is not available.
 *                   Setting this to 0 disables column deduction.
 *
 * \param uniqDeduct If set, this function will only deduce the column if
 *                   the operand specified in deduceCol only occurs once
 *                   in the source line. Otherwise, it will use the first
 *                   occurrence of the operand in the source line.
 *
 * \param deduceVal  If this is a non-NULL character pointer, then use this
 *                   string for column deduction instead of op1 or op2.
 *
 */
void
errWithSrc(error_code_t ecode, enum error_severity sev, int eline,
           const char *op1, const char *op2, int col, int deduceCol,
           bool uniqDeduct, const char *deduceVal)
{
  int i, len;
  char *srcFile = NULL;
  char *srcLine = NULL;
  int srcCol = 0;
  int contNo = 0;

  if (!XBIT(1, 1)) {
    /* Generate old error messages */
    display_error(ecode, sev, eline, op1, op2, 0, NULL);
    return;
  }
  if (eline > 0) {
    srcLine = get_src_line(eline, &srcFile, col, &srcCol, &contNo);
    if (srcFile && (len = strlen(srcFile)) > 0) {
      /* trim trailing whitespace on srcFile */
      char *cp;
      for (cp = (srcFile + (len - 1)); cp != srcFile; --cp) {
        if (!isspace(*cp))
          break;
      }
      if (cp != srcFile) {
        *(cp + 1) = '\0';
      }
    }
    if (deduceCol > 0) {
      /* try to deduce column number */
      char *op;
      char *srcLC = strdup(srcLine);
      char *p;
      if (deduceVal != NULL) {
        op = strdup(deduceVal);
      } else {
        op = strdup((deduceCol == 1) ? op1 : op2);
      }
      len = strlen(srcLC);
      for (i = 0; i < len; ++i) {
        srcLC[i] = tolower(srcLC[i]);
      }
      len = strlen(op);
      for (i = 0; i < len; ++i) {
        op[i] = tolower(op[i]);
      }
      p = srcCol == 0 ? strstr(srcLC, op) : strstr(srcLC + (srcCol-1), op);
      col = 0;
      if (p != NULL) {
        if (uniqDeduct) {
          char *q = strstr(p + 1, op);
          if (q == NULL) {
            /* op only occurs once in srcLine, so we can deduce col */
            col = (int)(p - srcLC) + 1;
          }
        } else {
          /* found op in srcLine, so we can deduce col */
          col = (int)(p - srcLC) + 1;
        }
      }
      FREE(op);
      FREE(srcLC);
    }
  }
  if (!deduceCol || col == 0)
    col = srcCol;
  display_error(ecode, sev, contNo + eline, op1, op2, col, srcFile);
  if (col > 0 && srcLine != NULL) {
    bool isLeadingChars;
    int numLeadingTabs;
    len = strlen(srcLine);
    for (numLeadingTabs = i = 0, isLeadingChars = true; i < len; ++i) {
      if (i == (col - 1)) {
        isLeadingChars = false;
      }
      if (isLeadingChars && srcLine[i] == '\t') {
        /* Keep track of tabs that appear before column number. */
        fputc('\t', stderr);
        ++numLeadingTabs;
      } else if (srcLine[i] == '\n') {
        break;
      } else {
        fputc(srcLine[i], stderr);
      }
    }
    fputc('\n', stderr);

    /* When we first computed col, we counted a tab as one space. So, we need
     * to subtract one from col as we print out the leading tabs.
     */
    for (i = 0; i < numLeadingTabs; ++i) {
      fputc('\t', stderr);
    }
    col -= numLeadingTabs;

    for (i = 0; i < (col - 1); ++i)
      fputc(' ', stderr);
    fputs("^\n", stderr);
  } else {
    fputc('\n', stderr);
  }
  FREE(srcLine);
  FREE(srcFile);
  if (sev == ERR_Fatal) {
    finish();
  }
}
#endif
