/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
 * \file
 * \brief Fortran parser module
 *
 *   contents:
 *
 *     parser()     - parse and semantically analyze one
 *                    user subprogram unit.
 *     parse_init() - initialize parsing of new statement.
 *     next_state(state, tkntyp) - look up next state in parse tables.
 *     prettytoken(tkntyp, tknval) - returns string of token
 */

#include "ccffinfo.h"
#include "gbldefs.h"
#include "gramtk.h"
#include "gramsm.h"
#include "global.h"
#include "error.h"
#include "symtab.h"
#include "semant.h"
#include "semstk.h"
#include "gramdf.h"
#if DEBUG
#include "proddf.h"
#endif
#include "tokdf.h"
#include "scan.h"

#define LOOP while (TRUE)
#define NOSTATE ((int)(-1))

typedef short int PSTACK;
static PSTACK *pstack;   /* parse stack */
static int sst_size = 0; /* current size of parse & semantic stacks */
static int cstate;       /* current parse state */
static int stktop;
static INT ctknval; /* 'value' of current token */
static int next_state(int, int);
static LOGICAL is_declaration(int);

/*  Function pointers indexed by sem.which_pass */
static void (*p_semant1[])(int, SST *) = {semant1, semant1};
static void (*p_semant2[])(int, SST *) = {NULL, semant2};
static void (*p_semant3[])(int, SST *) = {psemant3, semant3};
static void (*p_semantio[])(int, SST *) = {psemantio, semantio};
static void (*p_semsmp[])(int, SST *) = {psemsmp, semsmp};

static void _parser(void); /* core parsing function */
static char *prettytoken(int, INT);

void
parser(void)
{
  static int maxsev;
  LOGICAL skip_first_pass;
  ISZ_T save_bss_addr, save_saddr;
  int save_lineno;

  skip_first_pass = TRUE;
  if (gbl.internal == 0 && sem.mod_cnt <= 1) {
    /*
     * Perform the first pass on a subprogram or all module-contained
     * contained subprograms.  Semantic analysis is not performed on
     * the executable statements.
     */
    skip_first_pass = FALSE;
    sem.which_pass = 0;
    reset_internal_subprograms();
    save_bss_addr = get_bss_addr();
    save_saddr = gbl.saddr;
    save_lineno = gbl.lineno;
    if (!XBIT(120, 0x4000000)) {
      set_allfiles(0); /* Store file indexes */
    }
    fe_init(); /* init the scanner for the first parse */
#if DEBUG
    if (DBGBIT(1, 1) || DBGBIT(2, 1))
      fprintf(gbl.dbgfil, "-----  First Parse  -----\n");
#endif
    _parser();
    maxsev = gbl.maxsev;
    if (sem.mod_cnt) {
      /*
       * Either a module specification or the first module-contained
       * subprogram was just parsed.  If an error occurred, issue the
       * error summary now.
       */
      if (maxsev >= 3)
        summary(FALSE, FALSE);
      if (gbl.rutype == RU_BDATA)
        /*
         * A module specification part was parsed. Either:
         * 1.  a CONTAINS within the module terminated the first
         *     parse (scan sets scn.end_program_unit to TRUE), or
         * 2.  a module without contained subprograms was parsed.
         * Now, just produce the module-created blockdata - the second
         * parse isn't performed.
         */
        return;
      /*
       * At this point, either:
       * 1.  a module specification part with errors was parsed, or
       * 2.  the first module-contained subprogram was parsed (errors
       *     could have been detected).
       * Perform the first parse on the remaining contained subprograms,
       * recording the maximum error level detected for the all of the
       * contained subprograms.
       */
      while (sem.mod_cnt == 1) {
        reinit();
        errini();
        _parser();
        if (gbl.maxsev > maxsev)
          maxsev = gbl.maxsev;
        if (maxsev >= 3 && gbl.currsub)
          summary(FALSE, FALSE);
      }
    }
    resolve_fwd_refs();
    set_bss_addr(save_bss_addr);
    gbl.saddr = save_saddr;
    gbl.lineno = save_lineno;
    if (!XBIT(120, 0x4000000)) {
      set_allfiles(1); /* Retrieve file indexes */
    }
  }
/*
 * At this point, we're ready to perform the second parse (includes
 * executable statements).  The second parse is performed on a host
 * subprogram, an internal subprogram, or module-contained subprogram
 * if errors were not detected during the first parse.
 */
  if (maxsev < 3) {
    fe_restart();
    if (!skip_first_pass)
      reinit();
    sem.which_pass = 1;
    p_semant1[1] = semant1;
    p_semant2[1] = semant2;
#if DEBUG
    if (DBGBIT(1, 1) || DBGBIT(2, 1))
      fprintf(gbl.dbgfil, "-----  Second Parse  -----\n");
#endif
    _parser();
    resolve_fwd_refs();
    if (gbl.internal || sem.mod_cnt > 1)
      /*
       * If we're processing module-contained subprograms, make sure
       * the scanner is initialized for the next module subprogram.
       */
      fe_init();
  } else if (sem.mod_cnt == 2) {
    /*
     * One or more errors were detected in a module subprogram during the
     * the first parse. Don't perform the second parse on any of the
     * module subprograms.  The ensuing call to parser() will start the
     * first parse on the statement after the ENDMODULE statement.
     */
    end_module();
    sem.mod_cnt = 0;
    sem.mod_sym = 0;
    sem.submod_sym = 0;
  }
}

static void
_parser(void)
{
  int tkntyp, newtop, rednum, endflg;
  int t;
  int start, end, nstate;
  int jstart, jend, ptr, i;
  char *ptoken;

  endflg = 0;
  sst_size = SST_SIZE;
  NEW(pstack, PSTACK, sst_size);
  if (pstack == NULL)
    error(7, 4, 0, CNULL, CNULL);
  NEW(sst, SST, sst_size);
  if (sst == NULL)
    error(7, 4, 0, CNULL, CNULL);

  /* set funcline to a best guess value in case profiling info is
     requested for an unnamed program */
  gbl.funcline = gbl.lineno + 1;

  /* loop once for each statement in subprogram unit:  */

  LOOP
  {

    parse_init();

    /* loop once for each token in current Fortran stmt: */

    tkntyp = get_token(&ctknval); /* first token of a statement */

    if (sem.which_pass == 0) {
      /*
       * For the first pass, we want to semantically analyze expressions
       * only if they occur within a declaration statement.  Also, there
       * are semantic actions shared by the semantic analysis routines
       * for declaration and executable statements; for the first
       * pass, we only want these actions to be performed only if they
       * occur within the declarations statements.
       */
      if (is_declaration(tkntyp) ||
          (tkntyp == TK_GENERIC && sem.type_mode == 2) /* generic tbp */
          ) {
        p_semant1[0] = semant1;
        p_semant2[0] = semant2;
      } else {
        p_semant2[0] = psemant2;
      }
      gbl.nowarn = FALSE;
    } else {
      if (is_declaration(tkntyp)) {
        /* warnings issued in first pass for declarations */
        gbl.nowarn = TRUE;
      } else {
        gbl.nowarn = FALSE;
      }
    }

    LOOP
    {
#if DEBUG
      if (tkntyp < 1 || tkntyp >= (sizeof(tokname) / sizeof(char *))) {
        interr("scan error in parser", tkntyp, 3);
        tkntyp = TK_END;
      }
#endif
      if (scn.end_program_unit)
        endflg = 1;

      if (DBGBIT(1, 1)) {
        fprintf(gbl.dbgfil, "tkntyp: %s tknval: %d", tokname[tkntyp], ctknval);
        if (tkntyp == TK_IDENT)
          fprintf(gbl.dbgfil, " (%s)", scn.id.name + ctknval);
        fprintf(gbl.dbgfil, " lineno: %d \n", gbl.lineno);
      }

      /*
       * loop once for each reduction which can be made with tkntyp as
       * look ahead token (note that a production index may be 0):
       */
      LOOP
      {

        /*
         * perform binary search on parse tables to determine if a
         * reduction can be made:
         */
        start = fred[cstate];
        end = fred[cstate + 1] - 1;
        if (start > end)
          break; /* no reduction */
        for (i = start; i <= end; i++) {
          jstart = lset[nset[i]];
          jend = lset[nset[i] + 1] - 1;
          while (jstart <= jend) {
            ptr = (jstart + jend) >> 1;
            t = ls[ptr];
            if (t == tkntyp)
              goto perform_reduction;
            if (t < tkntyp)
              jstart = ptr + 1;
            else
              jend = ptr - 1;
          }
        }
        break; /* no reduction found */

      perform_reduction:
        rednum = prod[i];
        sem.tkntyp = tkntyp;
        if (DBGBIT(2, 1))
#if DEBUG
          fprintf(gbl.dbgfil, "%4d %crod(%4d) %s\n",
                  gbl.lineno, sem.which_pass ? 'P' : 'p',
                  rednum, prodstr[rednum]);
#else
          fprintf(gbl.dbgfil, "     %cednum: %d\n",
                  sem.which_pass ? 'R' : 'r', rednum);
#endif

        /* call appropriate semantic action routine: */

        newtop = stktop - len[rednum] + 1;
        if (rednum < SEM2)
          p_semant1[sem.which_pass](rednum, &sst[newtop]);
        else if (rednum < SEM3)
          p_semant2[sem.which_pass](rednum, &sst[newtop]);
        else if (rednum < SEM4)
          p_semant3[sem.which_pass](rednum, &sst[newtop]);
        else if (rednum < SEM5)
          p_semantio[sem.which_pass](rednum, &sst[newtop]);
        else
          p_semsmp[sem.which_pass](rednum, &sst[newtop]);

        if (sem.ignore_stmt) {
          sem.ignore_stmt = FALSE;
          goto ignore_stmt;
        }

        /* look for reduce transition:    */

        nstate = next_state(pstack[newtop - 1], (int)lhs[rednum]);
        if (nstate == NOSTATE)
          goto issue_error;
        else {
          cstate = nstate;
          pstack[newtop] = nstate;
          stktop = newtop;
        }
      } /* end of reduce loop. */

      /* look for a read transition:  */

      nstate = next_state(cstate, tkntyp);
      if (nstate == NOSTATE) {
        /* tpr 535
         * the grammar cannot be modified to support complex
         * constants of the form '( const-expr , const-expr )' but
         * can modified if a special token is returned for ',' (i.e.,
         * a "complex comma").
         * if a syntax error occurs when the current token is a comma,
         * check if a "complex comma" is legal; if so, continue
         * by parsing as if we have a complex constant, and semant
         * will determine if the real & imag parts are constants.
         */
        if (tkntyp == TK_COMMA) {
          nstate = next_state(cstate, TK_CMPLXCOMMA);
          if (nstate != NOSTATE) {
            if (DBGBIT(1, 1))
              fprintf(gbl.dbgfil, ">>> comma changed to complex comma %d\n",
                      gbl.lineno);
            goto read_trans;
          }
        }
      issue_error:

        ptoken = prettytoken(tkntyp, ctknval);
        errWithSrc(34, 3, gbl.lineno, ptoken, CNULL, getCurrColumn(), 1, false,
                   getDeduceStr(ptoken));
        sem.psfunc = FALSE; /* allow no stmt func defs */
        break;
      }
    read_trans:
      stktop++;
      if (stktop >= sst_size) {
        sst_size += SST_SIZE;
        pstack = (PSTACK *)sccrelal((char *)pstack,
                                    ((BIGUINT64)((sst_size) * sizeof(PSTACK))));
        sst = (SST *)sccrelal((char *)sst, ((BIGUINT64)((sst_size) * sizeof(SST))));
        assert(pstack != NULL, "parser:stack ovflw", stktop, 4);
        assert(sst != NULL, "parser:stack ovflw", stktop, 4);
      }
      pstack[stktop] = nstate;
      SST_SYMP(&sst[stktop], ctknval);
      SST_LINENOP(&sst[stktop], gbl.lineno);
      SST_COLUMNP(&sst[stktop], getCurrColumn());
      cstate = nstate;

      if (tkntyp == TK_EOL) {
        if (endflg == 1)
          goto parse_done;

        if (!scn.multiple_stmts && gbl.eof_flag ) { 
          if (gbl.empty_contains) {
            gbl.internal = 0;
            goto parse_done;
          }

          errsev(22);
          sem.mod_cnt = 0;
          sem.mod_sym = 0;
          sem.submod_sym = 0;
          goto parse_done;
        }
        break;
      }

      tkntyp = get_token(&ctknval); /* next token in the statement */

    } /* end foreach token LOOP */

  ignore_stmt:;

  } /* end foreach statement LOOP */

parse_done:
  FREE(pstack);
  FREE(sst);
  pstack = NULL;
  sst = NULL;
  sst_size = 0;
}

/*  Initialize parser to begin parsing of next Fortran statement */
void
parse_init(void)
{
  pstack[0] = 0;
  pstack[1] = 1;
  cstate = 1;
  stktop = 1;

  scan_reset();
}

/*  Return next parse state, given current state and look ahead
    token.  NOSTATE is returned if there is no next state (syntax
    error).
*/
static int
next_state(int state, int tkntyp)
{
  int start, end, ptr, t;

  start = ftrn[state];
  end = ftrn[state + 1] - 1;

  while (start <= end) {
    ptr = (start + end) >> 1;
    t = ent[tran[ptr]];
    if (t == tkntyp)
      return (tran[ptr]);
    if (t < tkntyp)
      start = ptr + 1;
    else
      end = ptr - 1;
  }
  return (NOSTATE);
}

static char *
prettytoken(int tkntyp, INT tknval)
{
  static char symbuf[132];
  INT v[2];

  switch (tkntyp) {
  case TK_EOL:
    sprintf(symbuf, "end of line");
    break;
  case TK_IDENT:
  case TK_NAMED_CONSTRUCT:
    sprintf(symbuf, "identifier %s", scn.id.name + tknval);
    break;
  case TK_LOGCONST:
    sprintf(symbuf, "logical constant %s",
            tknval == SCFTN_TRUE ? ".TRUE." : ".FALSE.");
    break;
  case TK_K_LOGCONST:
    sprintf(symbuf, "logical literal %s", getprint((int)tknval));
    break;
  case TK_ICON:
    sprintf(symbuf, "integer constant %d", tknval);
    break;
  case TK_K_ICON:
    sprintf(symbuf, "integer literal %s", getprint((int)tknval));
    break;
  case TK_RCON:
    strcpy(symbuf, "real constant ");
    v[0] = tknval;
    v[1] = 0;
    cprintf(symbuf + 14, "%.7e", v);
    break;
  case TK_DCON:
    sprintf(symbuf, "doubleprecision constant %s", getprint((int)tknval));
    break;
  case TK_CCON:
    sprintf(symbuf, "complex constant %s", getprint((int)tknval));
    break;
  case TK_DCCON:
    sprintf(symbuf, "doublecomplex constant %s", getprint((int)tknval));
    break;
  case TK_HOLLERITH:
    sprintf(symbuf, "hollerith constant %10.10s",
            stb.n_base + CONVAL1G(tknval));
    break;
  case TK_NONDDEC:
    sprintf(symbuf, "%s", getprint((int)tknval));
    break;
  case TK_NONDEC:
    sprintf(symbuf, "non-decimal constant %x", tknval);
    break;
  case TK_CMPLXCOMMA:
    sprintf(symbuf, ",");
    break;
  case TK_IOLP:
  case TK_IMPLP:
    sprintf(symbuf, "(");
    break;
  case TK_EQ:
    if (tknval)
      sprintf(symbuf, "==");
    else
      sprintf(symbuf, "%s", tokname[tkntyp]);
    break;
  case TK_GE:
    if (tknval)
      sprintf(symbuf, ">=");
    else
      sprintf(symbuf, "%s", tokname[tkntyp]);
    break;
  case TK_GT:
    if (tknval)
      sprintf(symbuf, ">");
    else
      sprintf(symbuf, "%s", tokname[tkntyp]);
    break;
  case TK_LE:
    if (tknval)
      sprintf(symbuf, "<=");
    else
      sprintf(symbuf, "%s", tokname[tkntyp]);
    break;
  case TK_LT:
    if (tknval)
      sprintf(symbuf, "<");
    else
      sprintf(symbuf, "%s", tokname[tkntyp]);
    break;
  case TK_NE:
    if (tknval == (('/' << 8) | '='))
      sprintf(symbuf, "/=");
    else if (tknval == (('<' << 8) | '>'))
      sprintf(symbuf, "<>");
    else
      sprintf(symbuf, "%s", tokname[tkntyp]);
    break;
  case TK_DIMATTR:
    sprintf(symbuf, "DIMENSION");
    break;
  case TK_MP_ATOMIC:
    sprintf(symbuf, "ATOMIC");
    break;
  case TK_MP_BARRIER:
    sprintf(symbuf, "BARRIER");
    break;
  case TK_MP_CANCEL:
    sprintf(symbuf, "%s", "CANCEL");
    break;
  case TK_MP_CRITICAL:
    sprintf(symbuf, "CRITICAL");
    break;
  case TK_MP_DECLAREREDUCTION:
    sprintf(symbuf, "%s", "DECLAREREDUCTION");
    break;
  case TK_MP_DECLARESIMD:
    sprintf(symbuf, "%s", "DECLARESIMD");
    break;
  case TK_MP_DECLARETARGET:
    sprintf(symbuf, "%s", "DECLARETARGET");
    break;
  case TK_MP_DISTPARDO:
    sprintf(symbuf, "%s", "DISTRIBUTEPARALLELDO");
    break;
  case TK_MP_DISTPARDOSIMD:
    sprintf(symbuf, "%s", "DISTRIBUTEPARALLELDOSIMD");
    break;
  case TK_MP_DISTRIBUTE:
    sprintf(symbuf, "%s", "DISTRIBUTE");
    break;
  case TK_MP_DISTSIMD:
    sprintf(symbuf, "%s", "DISTRIBUTESIMD");
    break;
  case TK_MP_DOSIMD:
    sprintf(symbuf, "%s", "DOSIMD");
    break;
  case TK_MP_DOACROSS:
    sprintf(symbuf, "DOACROSS");
    break;
  case TK_MP_ENDCRITICAL:
    sprintf(symbuf, "ENDCRITICAL");
    break;
  case TK_MP_ENDMASTER:
    sprintf(symbuf, "ENDMASTER");
    break;
  case TK_MP_ENDORDERED:
    sprintf(symbuf, "ENDORDERED");
    break;
  case TK_MP_ENDPARALLEL:
    sprintf(symbuf, "ENDPARALLEL");
    break;
  case TK_MP_ENDPARDO:
    sprintf(symbuf, "ENDPARALLELDO");
    break;
  case TK_MP_ENDPARSECTIONS:
    sprintf(symbuf, "ENDPARALLELSECTIONS");
    break;
  case TK_MP_ENDPDO:
    sprintf(symbuf, "ENDDO");
    break;
  case TK_MP_ENDSECTIONS:
    sprintf(symbuf, "ENDSECTIONS");
    break;
  case TK_MP_ENDSINGLE:
    sprintf(symbuf, "ENDSINGLE");
    break;
  case TK_MP_ENDDOSIMD:
    sprintf(symbuf, "ENDDOSIMD");
    break;
  case TK_MP_ENDDISTPARDO:
    sprintf(symbuf, "%s", "ENDDISTRIBUTEPARALLELDO");
    break;
  case TK_MP_ENDDISTPARDOSIMD:
    sprintf(symbuf, "%s", "ENDDISTRIBUTEPARALELLDOSIMD");
    break;
  case TK_MP_ENDDISTRIBUTE:
    sprintf(symbuf, "%s", "ENDDISTRIBUTE");
    break;
  case TK_MP_ENDDISTSIMD:
    sprintf(symbuf, "%s", "ENDDISTRIBUTESIMD");
    break;
  case TK_MP_ENDPARDOSIMD:
    sprintf(symbuf, "%s", "ENDPARALLELDOSIMD");
    break;
  case TK_MP_ENDSIMD:
    sprintf(symbuf, "%s", "ENDSIMD");
    break;
  case TK_MP_ENDTARGET:
    sprintf(symbuf, "%s", "ENDTARGTARGET");
    break;
  case TK_MP_ENDTASK:
    sprintf(symbuf, "%s", "ENDTASK");
    break;
  case TK_MP_ENDTASKLOOP:
    sprintf(symbuf, "%s", "ENDTASKLOOP");
    break;
  case TK_MP_ENDTASKLOOPSIMD:
    sprintf(symbuf, "%s", "ENDTASKLOOPSIMD");
    break;
  case TK_MP_ENDTEAMS:
    sprintf(symbuf, "%s", "ENDTEAMS");
    break;
  case TK_MP_ENDTEAMSDIST:
    sprintf(symbuf, "%s", "ENDTEAMSDISTRIBUTE");
    break;
  case TK_MP_ENDTEAMSDISTPARDO:
    sprintf(symbuf, "%s", "ENDTEAMSDISTRIBUTEPARALLELDO");
    break;
  case TK_MP_ENDTEAMSDISTPARDOSIMD:
    sprintf(symbuf, "%s", "ENDTEAMSDISTRIBUTEPARALLELDOSIMD");
    break;
  case TK_MP_ENDTEAMSDISTSIMD:
    sprintf(symbuf, "%s", "ENDTEAMSDISTRIBUTESIMD");
    break;
  case TK_MP_FLUSH:
    sprintf(symbuf, "FLUSH");
    break;
  case TK_MP_MASTER:
    sprintf(symbuf, "MASTER");
    break;
  case TK_MP_ORDERED:
    sprintf(symbuf, "ORDERED");
    break;
  case TK_MP_PARALLEL:
    sprintf(symbuf, "PARALLEL");
    break;
  case TK_MP_PARDO:
    sprintf(symbuf, "PARALLELDO");
    break;
  case TK_MP_PARSECTIONS:
    sprintf(symbuf, "PARALLELSECTIONS");
    break;
  case TK_MP_PARDOSIMD:
    sprintf(symbuf, "%s", "PARALLELDOSIMD");
    break;
  case TK_MP_PDO:
    sprintf(symbuf, "DO");
    break;
  case TK_MP_SECTION:
    sprintf(symbuf, "SECTION");
    break;
  case TK_MP_SECTIONS:
    sprintf(symbuf, "SECTIONS");
    break;
  case TK_MP_SINGLE:
    sprintf(symbuf, "SINGLE");
    break;
  case TK_MP_SIMD:
    sprintf(symbuf, "SIMD");
    break;
  case TK_MP_TARGET:
    sprintf(symbuf, "%s", "TARGET");
    break;
  case TK_MP_TARGETDATA:
    sprintf(symbuf, "%s", "TARGETDATA");
    break;
  case TK_MP_TARGETENTERDATA:
    sprintf(symbuf, "%s", "TARGETENTERDATA");
    break;
  case TK_MP_TARGETEXITDATA:
    sprintf(symbuf, "%s", "TARGETEXITDATA");
    break;
  case TK_MP_TARGETUPDATE:
    sprintf(symbuf, "%s", "TARGETUPDATE");
    break;
  case TK_MP_TARGPAR:
    sprintf(symbuf, "%s", "TARGETPAR");
    break;
  case TK_MP_TARGPARDO:
    sprintf(symbuf, "%s", "TARGETPARALLELDO");
    break;
  case TK_MP_TARGPARDOSIMD:
    sprintf(symbuf, "%s", "TARGETPARALLELDOSIMD");
    break;
  case TK_MP_TARGPARSIMD:
    sprintf(symbuf, "%s", "TARGETPARALLELSIMD");
    break;
  case TK_MP_TARGSIMD:
    sprintf(symbuf, "%s", "TARGETSIMD");
    break;
  case TK_MP_TARGTEAMS:
    sprintf(symbuf, "%s", "TARGETTEAMS");
    break;
  case TK_MP_TARGTEAMSDIST:
    sprintf(symbuf, "%s", "TARGETTEAMSDISTRIBUTE");
    break;
  case TK_MP_TARGTEAMSDISTPARDO:
    sprintf(symbuf, "%s", "TARGETTEAMSDISTRIBUTEPARALLELDO");
    break;
  case TK_MP_TARGTEAMSDISTPARDOSIMD:
    sprintf(symbuf, "%s", "TARGETTEAMSDISTRIBUTEPARALLELDOSIMD");
    break;
  case TK_MP_TARGTEAMSDISTSIMD:
    sprintf(symbuf, "%s", "TARGETTEAMSDISTRIBUTESIMD");
    break;
  case TK_MP_TASK:
    sprintf(symbuf, "%s", "TASK");
    break;
  case TK_MP_TASKLOOP:
    sprintf(symbuf, "%s", "TASKLOOP");
    break;
  case TK_MP_TASKLOOPSIMD:
    sprintf(symbuf, "%s", "TASKLOOPSIMD");
    break;
  case TK_MP_TEAMS:
    sprintf(symbuf, "%s", "TEAMS");
    break;
  case TK_MP_TEAMSDIST:
    sprintf(symbuf, "%s", "TEAMSDISTRIBUTE");
    break;
  case TK_MP_TEAMSDISTPARDO:
    sprintf(symbuf, "%s", "TEAMSDISTRIBUTEPARALLELDO");
    break;
  case TK_MP_TEAMSDISTPARDOSIMD:
    sprintf(symbuf, "%s", "TEAMSDISTRIBUTEPARALLELDOSIMD");
    break;
  case TK_MP_TEAMSDISTSIMD:
    sprintf(symbuf, "%s", "TEAMSDISTRIBUTESIMD");
    break;
  case TK_MP_THREADPRIVATE:
    sprintf(symbuf, "THREADPRIVATE");
    break;
  default:
    sprintf(symbuf, "%s", tokname[tkntyp]);
    break;
  }
  return symbuf;
}

static LOGICAL
is_declaration(int tkntyp)
{
  switch (tkntyp) {
  /*
   * It would be better if the tokens which can begin a declaration statement
   * were produced by prstab.
   */
  case TK_ENDSTMT:
  case TK_DIMATTR:
  case TK_DIRECTIVE:
  case TK_EMPTYFILE:
  case TK_ABSTRACT:
#ifdef TK_ACCDECL
  case TK_ACCDECL:
#endif
  case TK_ALIAS:
  case TK_ALIGN:
  case TK_ALLOCATABLE:
  case TK_ASYNCHRONOUS:
  case TK_ATTRIBUTES:
  case TK_AUTOMATIC:
  case TK_BIND:
  case TK_BLOCKDATA:
  case TK_BYTE:
  case TK_CHARACTER:
  case TK_CLASS:
  case TK_COMMON:
  case TK_COMPLEX:
  case TK_CONTIGUOUS:
  case TK_DATA:
  case TK_DIMENSION:
  case TK_DBLECMPLX:
  case TK_DBLEPREC:
#ifdef TK_DECLARE
  case TK_DECLARE:
#endif
  case TK_ELEMENTAL:
  case TK_ENDBLOCKDATA:
  case TK_ENDENUM:
  case TK_ENDFUNCTION:
  case TK_ENDINTERFACE:
  case TK_ENDMAP:
  case TK_ENDMODULE:
  case TK_ENDPROCEDURE:
  case TK_ENDPROGRAM:
  case TK_ENDSTRUCTURE:
  case TK_ENDSUBMODULE:
  case TK_ENDSUBROUTINE:
  case TK_ENDTYPE:
  case TK_ENDUNION:
  case TK_ENTRY:
  case TK_ENUM:
  case TK_ENUMERATOR:
  case TK_EOL: /* just so is_executable() will produce FALSE */
  case TK_EQUIV:
  case TK_EXTERNAL:
  case TK_FINAL:
  case TK_FUNCTION:
  case TK_IGNORE_TKR:
  case TK_IMPORT:
  case TK_IMPLICIT:
  case TK_IMPURE:
  case TK_INCLUDE:
  case TK_INTEGER:
  case TK_INTENT:
  case TK_INTERFACE:
  case TK_INTRINSIC:
  case TK_LOCAL:
  case TK_LOGICAL:
  case TK_MAP:
  case TK_MODULE:
  case TK_MOVEDESC:
  case TK_MP_DECLAREREDUCTION:
  case TK_MP_DECLARESIMD:
  case TK_MP_DECLARETARGET:
  case TK_MP_THREADPRIVATE:
  case TK_NAMELIST:
  case TK_NCHARACTER:
  case TK_NON_INTRINSIC:
  case TK_NOSEQUENCE:
  case TK_OPTIONAL:
  case TK_OPTIONS:
  case TK_PARAMETER:
  case TK_POINTER:
  case TK_PRIVATE:
  case TK_PROCEDURE:
  case TK_PROGRAM:
  case TK_PROTECTED:
  case TK_PUBLIC:
  case TK_PURE:
  case TK_REAL:
  case TK_RECORD:
  case TK_RECURSIVE:
  case TK_SAVE:
  case TK_SEQUENCE:
  case TK_STATIC:
  case TK_STRUCTURE:
  case TK_SUBMODULE:
  case TK_SUBROUTINE:
  case TK_TARGET:
  case TK_TCONTAINS:
  case TK_TPROCEDURE:
  case TK_TYPE:
  case TK_UNION:
  case TK_USE:
  case TK_VALUE:
  case TK_VOLATILE:
    return TRUE;
  default:
    break;
  }
  return FALSE;
}

LOGICAL
is_executable(int tkntyp)
{
  return !is_declaration(tkntyp);
}

#if DEBUG
static FILE *dfile = NULL;

void
dumpsst(SST *stk)
{
  int ast, sptr;
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  switch (SST_IDG(stk)) {
  case S_NULL:
    fprintf(dfile, "null");
    break;
  case S_CONST:
    fprintf(dfile, "const");
    break;
  case S_EXPR:
    fprintf(dfile, "expr");
    break;
  case S_LVALUE:
    fprintf(dfile, "lvalue");
    break;
  case S_LOGEXPR:
    fprintf(dfile, "logexpr");
    break;
  case S_STAR:
    fprintf(dfile, "star");
    break;
  case S_VAL:
    fprintf(dfile, "val");
    break;
  case S_IDENT:
    fprintf(dfile, "ident");
    sptr = SST_SYMG(stk);
    fprintf(dfile, " sptr=%d", sptr);
    if (sptr > 0 && sptr < stb.stg_avail) {
      fprintf(dfile, "=%s", SYMNAME(sptr));
    }
    break;
  case S_LABEL:
    fprintf(dfile, "label");
    break;
  case S_STFUNC:
    fprintf(dfile, "stfunc");
    break;
  case S_REF:
    fprintf(dfile, "ref");
    break;
  case S_TRIPLE:
    fprintf(dfile, "triple");
    break;
  case S_KEYWORD:
    fprintf(dfile, "keyword");
    break;
  case S_ACONST:
    fprintf(dfile, "aconst");
    break;
  case S_SCONST:
    fprintf(dfile, "sconst");
    break;
  case S_DERIVED:
    fprintf(dfile, "derived");
    break;
  default:
    fprintf(dfile, "ID=%d", SST_IDG(stk));
  }
  if (SST_PARENG(stk))
    fprintf(dfile, " paren");
  if (SST_ALIASG(stk))
    fprintf(dfile, " alias");
  ast = SST_ASTG(stk);
  if (ast) {
    fprintf(dfile, " ast=%d", ast);
    if (ast > 0) {
      fprintf(dfile, "\n");
      dump_ast_tree(ast);
    }
  }
  fprintf(dfile, "\n");
} /* dumpsst */

void
dumppstack(void)
{
  int i;

  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  fprintf(dfile, "Semant stack\n");
  for (i = stktop; i >= 0; --i) {
    fprintf(dfile, "[%d] ", i);
    dumpsst(&sst[i]);
  }
} /* dumppstack */
#endif
