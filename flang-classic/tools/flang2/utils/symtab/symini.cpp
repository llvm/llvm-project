/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* symbol initialization for Fortran */
#include "scutil.h"
#include "gbldefs.h"
#include "global.h"
#include "sharedefs.h"
#include "symtab.h"
#include "symnames.h"
#include "utils.h"
#include <algorithm>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <assert.h>

/*---------------------------------------------------------------------
 * Usage:
 *          symini in1 in2 -o out1 out2
 *
 * The following files are opened:
 *
 * in1  -  Intrinsics, generics, and predeclared definitions (symini*.n)
 * in2  -  ILMs definitions (ilmtp.n)
 *
 * out1 -  generated initializations of symbol tables (syminidf.h)
 * out2 -  generated macros which define the predeclared numbers (pd.h)
 *---------------------------------------------------------------------*/

extern STB stb;

/**
 * Formats of lines in the symini*.n input file:
 * .IN name pcnt atyp dtype ILM pname arrayf
 * .GN name siname iname rname dname cname cdname i8name
 * .PD name call type
 */

class SyminiF90 : public UtilityApplication
{

  enum { LT_IN = 1, LT_GN, LT_PD, LT_sh };

  NroffMap elt;
  std::map<std::string, int> argtype;
  std::vector<std::string> ilms;

  SPTR sptr; // index in the symbol table
  int npd;  // the total number of predeclared

  std::string symini_filename;
  std::string ilmtp_filename;
  std::string sphinx_filename;
  FILE *out1;
  FILE *out2;

public:
  SyminiF90(const std::vector<std::string> &args)
  {
    argtype["W"] = DT_WORD;
    argtype["I"] = DT_INT;
    argtype["R"] = DT_REAL;
    argtype["D"] = DT_DBLE;
    argtype["C"] = DT_CMPLX;
    argtype["CD"] = DT_DCMPLX;
    argtype["CQ"] = DT_QCMPLX;
    argtype["SI"] = DT_SINT;
    argtype["H"] = DT_CHAR;
    argtype["N"] = DT_NUMERIC;
    argtype["A"] = DT_ANY;
    argtype["L"] = DT_LOG;
    argtype["SL"] = DT_SLOG;
    argtype["I8"] = DT_INT8;
    argtype["L8"] = DT_LOG8;
    argtype["K"] = DT_NCHAR;
    argtype["Q"] = DT_QUAD;
    argtype["PI"] = __POINT_T;

    elt[".IN"] = LT_IN;
    elt[".GN"] = LT_GN;
    elt[".PD"] = LT_PD;
    elt[".sh"] = LT_sh;

    enum { INPUT, OUTPUT, SPHINX } state = INPUT;
    int filename_argument = 0;
    for (std::vector<std::string>::const_iterator arg = args.begin() + 1,
                                                  E = args.end();
         arg != E; ++arg) {
      if (*arg == "-o") {
        filename_argument = 0;
        state = OUTPUT;
      } else if (*arg == "-s") {
        state = SPHINX;
      } else {
        switch (state) {
        case INPUT:
          switch (filename_argument) {
          case 0:
            symini_filename = *arg;
            break;
          case 1:
            ilmtp_filename = *arg;
            break;
          default:
            usage("too many input files");
          }
          break;
        case OUTPUT: {
          FILE **out = nullptr;
          switch (filename_argument) {
          case 0:
            out = &out1;
            break;
          case 1:
            out = &out2;
            break;
          default:
            usage("too many output files");
          }
          *out = fopen(arg->c_str(), "w");
          if (!*out) {
            usage("can't create output file");
          }
          outputNotModifyComment(*out, arg->c_str(), args[0], false);
          break;
        }
        case SPHINX:
          sphinx_filename = *arg;
          break;
        }
        if (state != SPHINX) {
          ++filename_argument;
        } else {
          state = OUTPUT;
        }
      }
    }
    if (symini_filename == "") {
      usage("no symini.n file is given");
    }
    if (ilmtp_filename == "") {
      usage("no ilmtp.n file is given");
    }
    if (filename_argument != 2) {
      usage("need two output file names");
    }

    sym_init_first();
    init_ilm(); // read ILMs
  }

  int
  run()
  {
    ifs.open(symini_filename.c_str());
    if (!ifs) {
      printError(FATAL, "Can't open ", symini_filename.c_str());
    }
    if (!sphinx_filename.empty()) {
      sphinx.setOutputFile(sphinx_filename);
    }
    process_intrinsics();
    process_generics();
    process_predeclared();
    ifs.close();
    write_symfile();
    return 0;
  }

private:
  void
  usage(const char *error = 0)
  {
    printf("Usage: symini symini.n ilmtp.n -o -n syminidf.h pd.h\n\n");
    printf("symini.n    -- input file with symbol definitions\n");
    printf("ilmtp.n     -- input file with ILM definitions\n");
    printf("syminidf.h  -- transformed symtab.n output\n");
    printf("pd.h        -- generated symtab C header file\n\n");
    if (error) {
      fprintf(stderr, "Invalid command line: %s\n\n", error);
      exit(1);
    }
  }

  // FIXME: same as in fe90/shared/utils/symtab/symini.cpp. REFACTOR!
  /**
     \brief find a symbol in the symbol table if any.

     \param name is a symbol to find

     \return pointer if there is an entry for name in symbol table,
             otherwise return SPTR_NULL.
   */
  SPTR
  find_symbol(const std::string &name)
  {
    auto length = name.length();
    if (length > MAXIDLEN) {
      length = MAXIDLEN;
    }
    INT hashval; /* index into hashtb. */
    HASH_ID(hashval, name.c_str(), length);
    for (SPTR sptr = stb.hashtb[hashval]; sptr; sptr = HASHLKG(sptr)) {
      if (name == SYMNAME(sptr))
        return sptr;
    }
    return SPTR_NULL;
  }

  int
  search_atyp(std::string &atyp)
  {
    auto it = argtype.find(atyp);
    if (it != argtype.end()) {
      return it->second;
    }
    return -1;
  }

  void
  init_ilm()
  {
    NroffMap elt;
    elt[".IL"] = 1; // the value doesn't matter,
    elt[".FL"] = 1; // as long as it is not LT_EOF
    ifs.open(ilmtp_filename.c_str());
    if (!ifs) {
      printError(FATAL, "Can't open ", ilmtp_filename.c_str());
    }
    for (auto lt = getLineAndTokenize(elt); lt != LT_EOF;
         lt = getLineAndTokenize(elt)) {
      ilms.push_back(getToken());
    }
  }

  int
  get_ilm(const std::string &ilmn)
  {
    auto it = std::find(ilms.begin(), ilms.end(), ilmn);
    if (it != ilms.end()) {
      return it - ilms.begin() + 1;
    }
    std::string msg("illegal ilm, ");
    msg += ilmn + ", in template";
    printError(WARN, msg.c_str());
    return 0;
  }

  void
  process_intrinsics()
  {
    auto lt = getLineAndTokenize(elt);
    if (lt != LT_sh) {
      printError(FATAL, "missing .sh before intrinsics");
    }
    /* .IN name pcnt atyp dtype ILM pname arrayf */
    for (lt = getLineAndTokenize(elt); lt == LT_IN;
         lt = getLineAndTokenize(elt)) {
      auto tok = makeLower(getToken()); /* get new intrinsic name */
      const char *tokp = tok.c_str();
      int len = (int)tok.length();
      if (tok[0] == '=') { /* marks I8 specific */
#ifdef TM_I8
        tokp++;
        len--; /* allow I8 specific; eat '=' */
#else
        continue; /* ignore it */
#endif
      }
      sptr = installsym(tokp, len); /* get symbol pointer */
      if (STYPEG(sptr) != ST_UNKNOWN)
        printError(SEVERE, "Redefinition of intrinsic");
      STYPEP(sptr, ST_INTRIN);

      /* pcnt */
      tok = getToken();
      if (!isdigit(tok[0])) {
        printError(SEVERE, "param count missing, assumed to be 1");
        PARAMCTP(sptr, 1);
      } else {
        PARAMCTP(sptr, atoi(tok.c_str()));
      }
      /* atyp */
      tok = getToken();
      auto i = search_atyp(tok);
      if (i == -1) {
        printError(SEVERE, "bad atyp, assumed to be DT_INT");
        i = DT_INT;
      }
      ARGTYPP(sptr, i);

      /* dtype */
      tok = getToken();
      if ((i = search_atyp(tok)) == -1) {
        printError(SEVERE, "bad dtype, assumed to be DT_INT");
        i = DT_INT;
      }
      INTTYPP(sptr, i);

      /* ILM */
      tok = getToken();
      ILMP(sptr, (tok == "tc" || tok[0] == '-') ? 0 : get_ilm(tok));

      /* pname */
      tok = getToken();
      if (tok.empty()) {
        PNMPTRP(sptr, 0);
        ARRAYFP(sptr, 0);
        continue;
      }
      PNMPTRP(sptr, tok[0] == '-' ? 0 : putsname(tok.c_str(), tok.length()));

      /* aflag */
      tok = getToken();
      ARRAYFP(sptr, (tok.length() == 0 || tok[0] == '-') ? 0 : get_ilm(tok));
    }
    if (lt != LT_sh) {
      printError(FATAL, "missing .sh after intrinsics");
    }
  }

  void
  process_generics()
  {
    /* .GN name siname iname rname dname cname cdname i8name */
    auto lt = getLineAndTokenize(elt);
    for (; lt == LT_GN; lt = getLineAndTokenize(elt)) {
      auto tok = makeLower(getToken()); /* get new generic name */
      const char *tokp = tok.c_str();
      int len = (int)tok.length();
      if (tok[0] == '=') { /* marks I8 generic */
#ifdef TM_I8
        tokp++;
        len--; /* allow I8 generic; eat '=' */
#else
        continue; /* ignore it */
#endif
      }
      sptr = installsym(tokp, len); /* get symbol pointer */
      if (STYPEG(sptr) != ST_UNKNOWN)
        printError(SEVERE, "Redefinition of generic");
      STYPEP(sptr, ST_GENERIC);

      /* should make sure types are correct here, but ... */
      /* siname */
      tok = makeLower(getToken());
      if (tok.length() == 0 || tok[0] == '-')
        GSINTP(sptr, 0);
      else {
        auto sptr1 = installsym(tok.c_str(), tok.length());
        if (STYPEG(sptr1) != ST_INTRIN)
          printError(SEVERE, "Non-existent SI intrinsic");
        GSINTP(sptr, sptr1);
      }
      /* iname */
      tok = makeLower(getToken());
      if (tok.length() == 0 || tok[0] == '-')
        GINTP(sptr, 0);
      else {
        auto sptr1 = installsym(tok.c_str(), tok.length());
        if (STYPEG(sptr1) != ST_INTRIN)
          printError(SEVERE, "Non-existent I intrinsic");
        GINTP(sptr, sptr1);
      }
      /* rname */
      tok = makeLower(getToken());
      if (tok.length() == 0 || tok[0] == '-')
        GREALP(sptr, 0);
      else {
        auto sptr1 = installsym(tok.c_str(), tok.length());
        if (STYPEG(sptr1) != ST_INTRIN)
          printError(SEVERE, "Non-existent R intrinsic");
        GREALP(sptr, sptr1);
      }
      /* dname */
      tok = makeLower(getToken());
      if (tok.length() == 0 || tok[0] == '-')
        GDBLEP(sptr, 0);
      else {
        auto sptr1 = installsym(tok.c_str(), tok.length());
        if (STYPEG(sptr1) != ST_INTRIN)
          printError(SEVERE, "Non-existent D intrinsic");
        GDBLEP(sptr, sptr1);
      }
      /* cname */
      tok = makeLower(getToken());
      if (tok.length() == 0 || tok[0] == '-')
        GCMPLXP(sptr, 0);
      else {
        auto sptr1 = installsym(tok.c_str(), tok.length());
        if (STYPEG(sptr1) != ST_INTRIN)
          printError(SEVERE, "Non-existent C intrinsic");
        GCMPLXP(sptr, sptr1);
      }
      /* cdname */
      tok = makeLower(getToken());
      if (tok.length() == 0 || tok[0] == '-')
        GDCMPLXP(sptr, 0);
      else {
        auto sptr1 = installsym(tok.c_str(), tok.length());
        if (STYPEG(sptr1) != ST_INTRIN)
          printError(SEVERE, "Non-existent CD intrinsic");
        GDCMPLXP(sptr, sptr1);
      }
      /* i8name */
      tok = makeLower(getToken());
#ifdef TM_I8
      if (tok.length() == 0 || tok[0] == '-')
        GINT8P(sptr, 0);
      else {
        auto sptr1 = installsym(tok.c_str(), tok.length());
        if (STYPEG(sptr1) != ST_INTRIN)
          printError(SEVERE, "Non-existent I8 intrinsic");
        GINT8P(sptr, sptr1);
      }
#else
      GINT8P(sptr, 0);
#endif
      /* gsame */
      auto sptr1 = find_symbol(std::string(".") + SYMNAME(sptr));
      GSAMEP(sptr, sptr1 != 0 && STYPEG(sptr1) == ST_INTRIN ? sptr1 : 0);
    }
    if (lt != LT_sh) {
      printError(FATAL, "missing .sh after generics");
    }
  }

  void
  process_predeclared()
  {
    /* .PD name call type */
    npd = 0;
    for (auto lt = getLineAndTokenize(elt); lt == LT_PD;
         lt = getLineAndTokenize(elt)) {
      ++npd;
      auto tok = makeLower(getToken()); /* get new predeclared name */
      sptr = installsym(tok.c_str(), tok.length()); /* get symbol pointer */
      if (STYPEG(sptr) != ST_UNKNOWN)
        printError(SEVERE, "Redefinition of predeclared");
      STYPEP(sptr, ST_PD);
      /* output predeclared line */
      auto s = std::string("PD_") + tok;
      fprintf(out2, "#define %-20s%6d\n", s.c_str(), npd);

      /* init PD sym */
      DTYPEP(sptr, DT_NONE);
      PDNUMP(sptr, npd);
    }
  }

  void
  write_symfile()
  {
    /* now write symfile */
    fprintf(out1, "#ifndef SYMINIDF_H_\n#define SYMINIDF_H_\n\n");
    fprintf(out1, "#define INIT_SYMTAB_SIZE %d\n", stb.stg_avail);
    fprintf(out1, "#define INIT_NAMES_SIZE %d\n", stb.namavl);
    fprintf(out1, "static SYM init_sym[INIT_SYMTAB_SIZE] = {\n");
    for (SPTR i = SPTR(0); i < stb.stg_avail; ++i) {
      SYM *xp;

      xp = &stb.stg_base[i];
      assert(xp->stype <= ST_MAX);
      assert(xp->sc <= SC_MAX);
      fprintf(out1, "\t{%s, %s, %d, %d, (DTYPE)%d, (SPTR)%d, (SPTR)%d, %d, "
              "%d,\t/* %s */\n", SYMTYPE_names[xp->stype],
              SC_KIND_names[xp->sc], xp->b3, xp->b4, xp->dtype, xp->hashlk,
              xp->symlk, xp->scope, xp->nmptr, SYMNAME(i));
      fprintf(out1, "\t ");
      for (int i = 1; i != 33; ++i) {
        fprintf(out1, "%d,", 0 /*xp->f*/);
      }
      fprintf(out1, "\n");
      fprintf(out1, "\t %d, %d, %ld, %d, %d, %d, %ld, %d, %d, %d, %d,\n",
              xp->w8, xp->w9, xp->w10, xp->w11, xp->w12, xp->w13, xp->w14,
              xp->w15, xp->w16, xp->w17, xp->w18);
      fprintf(out1, "\t ");
      for (int i = 33; i != 65; ++i) {
        fprintf(out1, "%d,", 0 /*xp->f*/);
      }
      fprintf(out1, "\n");
      fprintf(out1, "\t %d,\n", xp->w20);
      fprintf(out1, "\t ");
      for (int i = 65; i != 97; ++i) {
        fprintf(out1, "%d,", 0 /*xp->f*/);
      }
      fprintf(out1, "\n");
      fprintf(out1, "\t %d, %d, %d,\n", xp->w22, xp->w23, xp->w24);
      fprintf(out1, "\t ");
      for (int i = 97; i != 129; ++i) {
        fprintf(out1, "%d,", 0 /*xp->f*/);
      }
      fprintf(out1, "\n");
      fprintf(out1, "\t %d, %d, %d, %d, %d, %d, %d, %d,},\n", xp->w26,
              xp->w27, xp->w28, xp->w29, xp->w30, xp->w31, xp->w32, xp->palign);
    }
    fprintf(out1, "};\n\n");
    fprintf(out1, "static char init_names[INIT_NAMES_SIZE] = {");
    int j = 16;
    for (int i = 0; i < stb.namavl; ++i) {
      if (j == 16) {
        fprintf(out1, "\n\t");
        j = 0;
      }
      ++j;
      if (stb.n_base[i])
        fprintf(out1, "'%c',", stb.n_base[i]);
      else
        fprintf(out1, "0,  ");
    }
    fprintf(out1, "\n};\n\n");
    fprintf(out1, "static int init_hashtb[HASHSIZE] = {");
    j = 10;
    for (int i = 0; i < HASHSIZE; ++i) {
      if (j == 10) {
        fprintf(out1, "\n\t");
        j = 0;
      }
      ++j;
      fprintf(out1, "%5d, ", stb.hashtb[i]);
    }
    fprintf(out1, "\n};\n");
    fprintf(out1, "#endif // SYMINIDF_H_\n");
  }
};

int
main(int argc, char **argv)
{
  SyminiF90 app(std::vector<std::string>(argv, argv + argc));
  return app.run();
}
