/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
 *  \file
 *  \brief Symbol initialization for Fortran
 */

#include "scutil.h"
#include "gbldefs.h"
#include "global.h"
#include "sharedefs.h"
#include "symtab.h"
#include "symnames.h"
#include "utils.h"
#include <algorithm>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <assert.h>

/*---------------------------------------------------------------------
 * Usage:
 *          symini infile -o out1 out2 out3 out4 out5
 * The following files are opened:
 *
 * ifs  -  input file to the utility (symini.n)
 * out1 -  output file which will contain the initializations of symbol
 *         tables. (syminidf.h)
 * out2 -  output file which will contain the macros which define
 *         the predeclared numbers. (pd.h)
 * out3 -  output file which will contain the macros which define
 *         the 'INTAST' numbers; names are of the form 'I_<name>'. (ast.d)
 * out4 -  output file which will contain the initializations for mapping
 *         the 'INTAST" numbers to their corresponding symtab table
 *         entries. (astdf.d)
 * out5 -  output file which will contain ILMs (ilmtp.h)
 *---------------------------------------------------------------------*/

extern STB stb;

/**
 * .IN name pcnt atyp dtype ILM pname arrayf
 * .GN name siname iname rname dname cname cdname i8name qname cqname
 * .PD name pname dtype
 */
class SyminiFE90 : public UtilityApplication
{

  enum {
    LT_IN = 1,
    LT_GN,
    LT_PD,
    LT_sh,
    LT_AT,
    LT_H1,
    LT_H2,
    LT_H3,
    LT_H4,
    LT_H5,
    LT_H6,
    LT_H7,
    LT_H8, // F2008 iso_c_binding module procedure, mark as ST_ISOC
    LT_H9  // F2008 iso_fortran_env module procedure, mark as ST_FTNENV
  };

  static const char *init_names0[];
  static const char *init_names1[];
  static const char *init_names2[];
  static const char *init_names3[];
  static const size_t init_names0_size;
  static const size_t init_names1_size;
  static const size_t init_names2_size;
  static const size_t init_names3_size;

  std::vector<std::string> intr_kwd;
  std::vector<std::string> ilms;
  std::vector<int> intast_sym;
  std::map<std::string, DTYPE> argtype;
  NroffMap elt;
  std::string sphinx_filename;

  FILE *out1;
  FILE *out2;
  FILE *out3;
  FILE *out4;
  FILE *out5;

  int hpf_lib_first;
  int hpf_lib_last;
  int hpf_local_lib_first;
  int hpf_local_lib_last;
  int craft_first;
  int craft_last;
  int cray_first;
  int cray_last;
  int iso_c_first;
  int iso_c_last;
  int ieeearith_first;
  int ieeearith_last;
  int ieeeexcept_first;
  int ieeeexcept_last;
  SPTR sptr;
  int npd;
  int star_str;

public:
  SyminiFE90(int argc, char **argv)
  {
    hpf_lib_first = 0;
    hpf_lib_last = 0;
    hpf_local_lib_first = 0;
    hpf_local_lib_last = 0;
    craft_first = 0;
    craft_last = 0;
    cray_first = 0;
    cray_last = 0;
    iso_c_first = 0;
    iso_c_last = 0;
    ieeearith_first = 0;
    ieeearith_last = 0;
    ieeeexcept_first = 0;
    ieeeexcept_last = 0;
    npd = 0;
    star_str = 0;
    argtype["W"] = DT_WORD;
    argtype["I"] = DT_INT4;
    argtype["RH"] = DT_REAL2;
    argtype["R"] = DT_REAL4;
    argtype["D"] = DT_REAL8;
    argtype["CH"] = DT_CMPLX4;
    argtype["C"] = DT_CMPLX8;
    argtype["CD"] = DT_CMPLX16;
    argtype["SI"] = DT_SINT;
    argtype["H"] = DT_CHAR;
    argtype["N"] = DT_NUMERIC;
    argtype["A"] = DT_ANY;
    argtype["L"] = DT_LOG4;
    argtype["SL"] = DT_SLOG;
    argtype["I8"] = DT_INT8;
    argtype["L8"] = DT_LOG8;
    argtype["K"] = DT_NCHAR;
    argtype["Q"] = DT_QUAD;
    argtype["CQ"] = DT_QCMPLX;
    argtype["BI"] = DT_BINT;
    argtype["AD"] = DT_ADDR;
    elt[".IN"] = LT_IN;
    elt[".GN"] = LT_GN;
    elt[".PD"] = LT_PD;
    elt[".sh"] = LT_sh;
    elt[".AT"] = LT_AT;
    elt[".H1"] = LT_H1;
    elt[".H2"] = LT_H2;
    elt[".H3"] = LT_H3;
    elt[".H4"] = LT_H4;
    elt[".H5"] = LT_H5;
    elt[".H6"] = LT_H6;
    elt[".H7"] = LT_H7;
    elt[".H8"] = LT_H8;
    elt[".H9"] = LT_H9;

    // FIXME this initializes the global variable stb. In the future
    // STB should become a class with normal C++ class constructors,
    // and this call will not be necessary.
    sym_init_first();

    int output_file_argument = 0;
    for (int arg = 1; arg < argc; ++arg) {
      if (strcmp(argv[arg], "-o") == 0) {
        output_file_argument = 1;
      } else if (strcmp(argv[arg], "-s") == 0) {
        sphinx_filename = argv[++arg];
      } else if (argv[arg][0] == '-') {
        usage("unknown option");
      } else if (0 == output_file_argument) {
        ifs.open(argv[arg]);
      } else {
        switch (output_file_argument) {
        case 1:
          out1 = fopen(argv[arg], "w");
          ++output_file_argument;
          break;
        case 2:
          out2 = fopen(argv[arg], "w");
          ++output_file_argument;
          break;
        case 3:
          out3 = fopen(argv[arg], "w");
          ++output_file_argument;
          break;
        case 4:
          out4 = fopen(argv[arg], "w");
          ++output_file_argument;
          break;
        case 5:
          out5 = fopen(argv[arg], "w");
          ++output_file_argument;
          break;
        default:
          usage("too many output files");
        }
      }
    }
    if (output_file_argument < 5) {
      usage("missing some output file names");
    }
    if (!sphinx_filename.empty()) {
      sphinx.setOutputFile(sphinx_filename);
    }
  }

  int
  run()
  {
    process_intrinsics();
    process_generics();
    process_predeclared();
    consume_macros(LT_H1, hpf_lib_first, hpf_lib_last);
    consume_macros(LT_H2, hpf_local_lib_first, hpf_local_lib_last);
    consume_macros(LT_H3, craft_first, craft_last);
    // FIXME: further refactoring is necessary. The following are
    // variants of the previous three process_*() functions.
    process_cray();          // process_predeclared
    process_iso();           // process_intrinsics
    process_ieeearith();     // process_predeclared
    process_ieeeexcept();    // process_predeclared
    process_miscellaneous(); // process_predeclared

    write_symfile();
    write_out4();
    write_out5();

    return 0;
  }

private:
  void
  usage(const char *error = 0)
  {
    printf("Usage: symini input_file -o output_files\n\n");
    printf("input_file      -- input file with symbol definitions\n");
    printf("-o output_files -- generated C header files.\n\n");
    if (error) {
      fprintf(stderr, "Invalid command line: %s\n\n", error);
      exit(1);
    }
  }

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

  std::string
  nodollar(const std::string &s)
  {
    std::string result;
    for (auto c = s.begin(), E = s.end(); c != E; ++c) {
      result.push_back(*c == '$' ? '_' : *c);
    }
    return result;
  }

  int
  get_ilm(const std::string &ilmn)
  {
    if (ilmn == "" || ilmn[0] == '-') {
      return 0;
    }
    auto it = std::find(ilms.begin(), ilms.end(), ilmn);
    if (it == ilms.end()) {
      ilms.push_back(ilmn);
      return (int)ilms.size();
    }
    return it - ilms.begin() + 1;
  }

  void
  emit_i_intr(SPTR sptr)
  {
    fprintf(out3, "#define I_");
    const char *p = SYMNAME(sptr);
    if (*p == '.') {
      ++p;
    }
    for (; *p != '\0'; ++p) {
      fputc(*p == '$' ? '_' : (islower(*p) ? toupper(*p) : *p), out3);
    }
    int index = (int)intast_sym.size();
    INTASTP(sptr, index);
    fprintf(out3, " %d\n", index);
    intast_sym.push_back(sptr);
  }

  DTYPE
  search_atyp(const std::string &atyp)
  {
    auto it = argtype.find(atyp);
    if (it != argtype.end()) {
      return it->second;
    }
    return DT_NONE;
  }

  void
  consume_macros(int code, int &first, int &last)
  {
    first = 0;
    auto lt = getLineAndTokenize(elt);
    while (lt == code) {
      lt = getLineAndTokenize(elt);
      if (lt == LT_AT) {
        lt = getLineAndTokenize(elt);
      }
    }
    last = sptr;
  }

  void
  process_at(SPTR sptr)
  {
    auto tok = makeLower(getToken());
    switch (tok[0]) {
    case 'e':
      INKINDP(sptr, IK_ELEMENTAL);
      break;
    case 'i':
      INKINDP(sptr, IK_INQUIRY);
      break;
    case 's':
      INKINDP(sptr, IK_SUBROUTINE);
      break;
    case 't':
      INKINDP(sptr, IK_TRANSFORM);
      break;
    default:
      printError(WARN, "Illegal intrinsic kind in .AT");
      return;
    }

    int kindpos = 0;
    std::string buf;
    for (int pos = 1;; ++pos) {
      tok = getToken();
      if (tok.empty()) {
        break;
      }
      buf += std::string(" ") + tok;
      // look for optional KIND argument
      if (tok.substr(0, 5) == "*kind") {
        kindpos = pos;
      }
    }
    if (!buf.empty()) {
      auto it = std::find(intr_kwd.begin(), intr_kwd.end(), buf.substr(1));
      if (it == intr_kwd.end()) {
        intr_kwd.push_back(buf.substr(1));
        it = intr_kwd.end() - 1;
      }
      KWDARGP(sptr, it - intr_kwd.begin());
    } else {
      KWDARGP(sptr, 0);
    }
    int len = 0;
    int i = 1;
    for (std::string::const_iterator c = buf.begin(), E = buf.end(); c != E;
         ++c) {
      if (*c == '.') {
        len--;
        break;
      }
      if (*c == ' ') {
        i = 1;
        continue;
      }
      if (i) {
        len++;
        i = 0;
      }
    }
    KWDCNTP(sptr, len);
    if (STYPEG(sptr) == ST_GENERIC) {
      KINDPOSP(sptr, kindpos);
    }
  }

  void
  oldsyms(const char **init_names, int init_syms_size, int index)
  {
    int i, k;
    SPTR j;
    short *map_init = (short *)malloc(init_syms_size * sizeof(short));
    if (map_init == NULL)
      printError(FATAL, "oldsyms: malloc no space");
    for (i = 0; i < init_syms_size; ++i) {
      for (j = SPTR(0); j < stb.stg_avail; ++j) {
        const char *p, *q;
        p = init_names[i];
        q = SYMNAME(j);
        if (p[0] == '.')
          p++;
        if (q[0] == '.')
          q++;
        if (strcmp(p, q) == 0)
          goto found;
      }
      printError(INFO, "no map for ", init_names[i]);
    found:
      map_init[i] = j;
    }
    fprintf(out1, "\n");
    fprintf(out1, "static int init_syms%d_size = %d;\n", index, init_syms_size);
    fprintf(out1, "\n");
    fprintf(out1, "static short map_init%d[%d] = {\n    ", index,
            init_syms_size);
    k = 0;
    for (i = 0; i < init_syms_size; i++) {
      if (k >= 16) {
        fprintf(out1, "\n    ");
        k = 0;
      }
      fprintf(out1, "%3d,", map_init[i]);
      k++;
    }
    fprintf(out1, "\n};\n");
    free(map_init);
  }

  std::string
  create_symbol(SYMTYPE ST, SPTR *sptr, const char *category,
                bool ignore_dot = false)
  {
    auto t = makeLower(getToken());
    *sptr = installsym(t.c_str(), t.length());
    if (STYPEG(*sptr) != ST_UNKNOWN) {
      printError(SEVERE, "Redefinition of ", category);
    }
    STYPEP(*sptr, ST);
    if (ignore_dot || t[0] != '.') {
      emit_i_intr(*sptr);
    }
    return t;
  }

  DTYPE
  get_type(const std::string &token)
  {
    DTYPE type = search_atyp(token);
    if (type == DT_NONE) {
      printError(SEVERE, "bad type, assumed to be DT_INT");
      type = DT_INT;
    }
    return type;
  }

  /* .IN name pcnt atyp dtype ILM pname arrayf */
  void
  process_intrinsics()
  {
    auto lt = getLineAndTokenize(elt);
    if (lt != LT_sh) {
      printError(FATAL, "missing .sh before intrinsics");
    }
    lt = getLineAndTokenize(elt);
    while (lt == LT_IN) {
      auto tok = create_symbol(ST_INTRIN, &sptr, "intrinsic");
      /* pcnt */
      tok = getToken();
      if (!isdigit(tok[0])) {
        printError(SEVERE, "param count missing, assumed to be 1");
        PARAMCTP(sptr, 1);
      } else {
        PARAMCTP(sptr, atoi(tok.c_str()));
      }
      /* atyp */
      ARGTYPP(sptr, get_type(getToken()));
      /* dtype */
      INTTYPP(sptr, get_type(getToken()));
      /* ILM */
      tok = getToken();
      ILMP(sptr, tok == "tc" ? 0 : get_ilm(tok));
      /* pname */
      tok = getToken();
      if (tok.empty()) {
        PNMPTRP(sptr, 0);
        ARRAYFP(sptr, 0);
      } else {
        if (tok[0] == '-') {
          if (tok.length() == 1)
            PNMPTRP(sptr, 0);
          else
            PNMPTRP(sptr, putsname(tok.c_str(), tok.length()));
        } else if (tok[0] == '*') {
          if (tok.length() == 1) {
            if (star_str == 0)
              star_str = putsname("*", 1);
            PNMPTRP(sptr, star_str);
          } else
            PNMPTRP(sptr, putsname(tok.c_str(), tok.length()));
        } else
          PNMPTRP(sptr, putsname(tok.c_str(), tok.length()));
        /* aflag */
        ARRAYFP(sptr, get_ilm(getToken()));
        tok = getToken();
        NATIVEP(sptr, tok.empty() || tok == "native" ? 0 : 1);
      }
      lt = getLineAndTokenize(elt);
      if (lt == LT_AT) {
        process_at(sptr);
        lt = getLineAndTokenize(elt);
      } else {
        printError(WARN, "missing .AT after .IN");
      }
    }
    if (lt != LT_sh) {
      printError(FATAL, "missing .sh after intrinsics");
    }
  }

  /* .GN name siname iname rname dname cname cdname i8name qname cqname */
  void
  process_generics()
  {
    auto lt = getLineAndTokenize(elt);
    while (lt == LT_GN) {
      auto tok = create_symbol(ST_GENERIC, &sptr, "generic");

      /* should make sure types are correct here, but ... */
      /* siname */
      tok = makeLower(getToken());
      if (tok.empty() || tok[0] == '-')
        GSINTP(sptr, SPTR_NULL);
      else {
        SPTR sptr1 = installsym(tok.c_str(), tok.length());
        if (STYPEG(sptr1) != ST_INTRIN)
          printError(SEVERE, "Non-existent SI intrinsic");
        GSINTP(sptr, sptr1);
        if (tok[0] == '.') {
          INTASTP(sptr1, intast_sym.size() - 1);
        }
      }
      /* iname */
      tok = makeLower(getToken());
      if (tok.empty() || tok[0] == '-')
        GINTP(sptr, SPTR_NULL);
      else {
        SPTR sptr1 = installsym(tok.c_str(), tok.length());
        if (STYPEG(sptr1) != ST_INTRIN)
          printError(SEVERE, "Non-existent I intrinsic");
        GINTP(sptr, sptr1);
        if (tok[0] == '.') {
          INTASTP(sptr1, intast_sym.size() - 1);
        }
      }
      /* rname */
      tok = makeLower(getToken());
      if (tok.length() <= 0 || tok[0] == '-')
        GREALP(sptr, SPTR_NULL);
      else {
        SPTR sptr1 = installsym(tok.c_str(), tok.length());
        if (STYPEG(sptr1) != ST_INTRIN)
          printError(SEVERE, "Non-existent R intrinsic");
        GREALP(sptr, sptr1);
        if (tok[0] == '.') {
          INTASTP(sptr1, intast_sym.size() - 1);
        }
      }
      /* dname */
      tok = makeLower(getToken());
      if (tok.empty() || tok[0] == '-')
        GDBLEP(sptr, SPTR_NULL);
      else {
        SPTR sptr1 = installsym(tok.c_str(), tok.length());
        if (STYPEG(sptr1) != ST_INTRIN)
          printError(SEVERE, "Non-existent D intrinsic");
        GDBLEP(sptr, sptr1);
        if (tok[0] == '.') {
          INTASTP(sptr1, intast_sym.size() - 1);
        }
      }
      /* cname */
      tok = makeLower(getToken());
      if (tok.empty() || tok[0] == '-')
        GCMPLXP(sptr, SPTR_NULL);
      else {
        SPTR sptr1 = installsym(tok.c_str(), tok.length());
        if (STYPEG(sptr1) != ST_INTRIN)
          printError(SEVERE, "Non-existent C intrinsic");
        GCMPLXP(sptr, sptr1);
        if (tok[0] == '.') {
          INTASTP(sptr1, intast_sym.size() - 1);
        }
      }
      /* cdname */
      tok = makeLower(getToken());
      if (tok.empty() || tok[0] == '-')
        GDCMPLXP(sptr, SPTR_NULL);
      else {
        SPTR sptr1 = installsym(tok.c_str(), tok.length());
        if (STYPEG(sptr1) != ST_INTRIN)
          printError(SEVERE, "Non-existent CD intrinsic");
        GDCMPLXP(sptr, sptr1);
        if (tok[0] == '.') {
          INTASTP(sptr1, intast_sym.size() - 1);
        }
      }
      /* i8name */
      tok = makeLower(getToken());
      if (tok.empty() || tok[0] == '-')
        GINT8P(sptr, SPTR_NULL);
      else {
        SPTR sptr1 = installsym(tok.c_str(), tok.length());
        if (STYPEG(sptr1) != ST_INTRIN)
          printError(SEVERE, "Non-existent I8 intrinsic");
        GINT8P(sptr, sptr1);
        if (tok[0] == '.') {
          INTASTP(sptr1, intast_sym.size() - 1);
        }
      }
      /* qname */
      tok = makeLower(getToken());
      if (tok.empty() || tok[0] == '-')
        GQUADP(sptr, SPTR_NULL);
      else {
        SPTR sptr1 = installsym(tok.c_str(), tok.length());
        if (STYPEG(sptr1) != ST_INTRIN)
          printError(SEVERE, "Non-existent Q intrinsic");
        GQUADP(sptr, sptr1);
        if (tok[0] == '.') {
          INTASTP(sptr1, intast_sym.size() - 1);
        }
      }
      /* cqname */
      tok = makeLower(getToken());
      if (tok.empty() || tok[0] == '-')
        GQCMPLXP(sptr, SPTR_NULL);
      else {
        SPTR sptr1 = installsym(tok.c_str(), tok.length());
        if (STYPEG(sptr1) != ST_INTRIN)
          printError(SEVERE, "Non-existent CQ intrinsic");
        GQCMPLXP(sptr, sptr1);
        if (tok[0] == '.') {
          INTASTP(sptr1, intast_sym.size() - 1);
        }
      }
      /* gsame */
      SPTR sptr1 = find_symbol(std::string(".") + SYMNAME(sptr));
      if (sptr1 != 0 && STYPEG(sptr1) == ST_INTRIN) {
        GSAMEP(sptr, sptr1);
        INTASTP(sptr1, intast_sym.size() - 1);
      } else
        GSAMEP(sptr, SPTR_NULL);
      lt = getLineAndTokenize(elt);
      if (lt == LT_AT) {
        process_at(sptr);
        lt = getLineAndTokenize(elt);
      } else
        printError(WARN, "missing .AT after .GN");
    }
    if (lt != LT_sh) {
      printError(FATAL, "missing .sh after generics");
    }
  }

  /* .PD name pname dtype */
  void
  process_predeclared()
  {
    auto lt = getLineAndTokenize(elt);
    while (lt == LT_PD) {
      ++npd;
      auto tok = create_symbol(ST_PD, &sptr, "predeclared", true);
      /* output predeclared line */
      auto s =
          nodollar(std::string("PD_") + (tok[0] == '.' ? tok.substr(1) : tok));
      fprintf(out2, "#define %-20s%6d\n", s.c_str(), npd);

      /* init PD sym */
      DTYPEP(sptr, 0);
      PDNUMP(sptr, npd);

      tok = getToken(); /* get pname */
      if (tok.empty()) {
        printError(WARN, ".PD, no pname for ", SYMNAME(sptr));
      } else {
        if (tok[0] == '-')
          PNMPTRP(sptr, 0);
        else {
          makeLower(tok);
          PNMPTRP(sptr, putsname(tok.c_str(), tok.length()));
        }
        tok = getToken(); /* get dtype */
        if (tok.empty()) {
          printError(WARN, ".PD, no dtype for ", SYMNAME(sptr));
        } else if (tok[0] != '-') {
          INTTYPP(sptr, get_type(tok));
        }
      }

      tok = getToken();
      NATIVEP(sptr, tok.empty() || tok == "native" ? 0 : 1);

      lt = getLineAndTokenize(elt);
      if (lt == LT_AT) {
        process_at(sptr);
        lt = getLineAndTokenize(elt);
      } else
        printError(WARN, "missing .AT after .PD");
    }
  }

  /* .H4 name pname dtype */
  void
  process_cray()
  {
    cray_first = 0;
    auto lt = getLineAndTokenize(elt);
    while (lt == LT_H4) {
      ++npd;
      auto tok = create_symbol(ST_CRAY, &sptr, "predeclared");
      if (cray_first == 0)
        cray_first = sptr;
      if (tok[0] != '.') {
        /* output predeclared line */
        auto s = nodollar(std::string("PD_") + tok);
        fprintf(out2, "#define %-20s%6d\n", s.c_str(), npd);
      }

      /* init PD sym */
      DTYPEP(sptr, 0);
      PDNUMP(sptr, npd);

      tok = makeLower(getToken()); /* get pname */
      if (tok.empty()) {
        printError(WARN, ".H4, no pname for ", SYMNAME(sptr));
      } else {
        PNMPTRP(sptr, tok[0] == '-' ? 0 : putsname(tok.c_str(), tok.length()));
        tok = getToken(); /* get dtype */
        if (tok.empty()) {
          printError(WARN, ".H4, no dtype for ", SYMNAME(sptr));
        } else if (tok[0] != '-') {
          INTTYPP(sptr, get_type(tok));
        }
      }

      lt = getLineAndTokenize(elt);
      if (lt == LT_AT) {
        process_at(sptr);
        lt = getLineAndTokenize(elt);
      } else
        printError(WARN, "missing .AT after .H4");
    }
    cray_last = sptr;
  }

  /* ISO_C_BINDING intrinsics */
  /* .IN name pcnt atyp dtype ILM pname arrayf */
  void
  process_iso()
  {
    iso_c_first = 0;
    auto lt = getLineAndTokenize(elt);

    while (lt == LT_IN) {
      auto tok = create_symbol(ST_ISOC, &sptr, "intrinsic");
      if (iso_c_first == 0)
        iso_c_first = sptr;

      /* pcnt */
      tok = getToken();
      if (!isdigit(tok[0])) {
        printError(SEVERE, "param count missing, assumed to be 1");
        PARAMCTP(sptr, 1);
      } else {
        PARAMCTP(sptr, atoi(tok.c_str()));
      }
      /* atyp */
      ARGTYPP(sptr, get_type(getToken()));
      /* dtype */
      INTTYPP(sptr, get_type(getToken()));
      /* hard code the dtypes for c_loc, c_funloc
         for easier type checking */
      if (INTTYPG(sptr) == DT_ANY)
        DTYPEP(sptr, DT_ADDR);
      /* ILM */
      tok = getToken();
      ILMP(sptr, tok == "tc" ? 0 : get_ilm(tok));
      /* pname */
      tok = getToken();
      if (tok.empty()) {
        PNMPTRP(sptr, 0);
        ARRAYFP(sptr, 0);
      } else {
        if (tok[0] == '-') {
          if (tok.length() == 1)
            PNMPTRP(sptr, 0);
          else
            PNMPTRP(sptr, putsname(tok.c_str(), tok.length()));
        } else if (tok[0] == '*') {
          if (tok.length() == 1) {
            if (star_str == 0)
              star_str = putsname("*", 1);
            PNMPTRP(sptr, star_str);
          } else
            PNMPTRP(sptr, putsname(tok.c_str(), tok.length()));
        } else
          PNMPTRP(sptr, putsname(tok.c_str(), tok.length()));
        /* aflag */
        ARRAYFP(sptr, get_ilm(getToken()));
        tok = getToken();
        NATIVEP(sptr, tok.empty() || tok == "native" ? 0 : 1);
      }

      lt = getLineAndTokenize(elt);
      if (lt == LT_AT) {
        process_at(sptr);
        lt = getLineAndTokenize(elt);
      } else
        printError(WARN, "missing .AT after .IN");
    } /* end while */

    iso_c_last = sptr;
  }

  /* .H5 name pname dtype */
  void
  process_ieeearith()
  {
    ieeearith_first = 0;
    auto lt = getLineAndTokenize(elt);
    while (lt == LT_H5) {
      ++npd;
      auto tok = create_symbol(ST_IEEEARITH, &sptr, "predeclared");
      if (ieeearith_first == 0)
        ieeearith_first = sptr;
      /* output predeclared line */
      auto s = nodollar(std::string("PD_") + tok);
      fprintf(out2, "#define %-20s%6d\n", s.c_str(), npd);

      /* init PD sym */
      DTYPEP(sptr, 0);
      PDNUMP(sptr, npd);

      tok = makeLower(getToken()); /* get pname */
      if (tok.empty()) {
        printError(WARN, ".H5, no pname for ", SYMNAME(sptr));
      } else {
        PNMPTRP(sptr, tok[0] == '-' ? 0 : putsname(tok.c_str(), tok.length()));
        tok = getToken(); /* get dtype */
        if (tok.empty()) {
          printError(WARN, ".H5, no dtype for ", SYMNAME(sptr));
        } else if (tok[0] != '-') {
          INTTYPP(sptr, get_type(tok));
        }
      }

      lt = getLineAndTokenize(elt);
      if (lt == LT_AT) {
        process_at(sptr);
        lt = getLineAndTokenize(elt);
      } else
        printError(WARN, "missing .AT after .H5");
    }
    ieeearith_last = sptr;
  }

  /* .H6 name pname dtype */
  void
  process_ieeeexcept()
  {
    ieeeexcept_first = 0;
    auto lt = getLineAndTokenize(elt);
    while (lt == LT_H6) {
      ++npd;
      auto tok = create_symbol(ST_IEEEEXCEPT, &sptr, "predeclared");
      if (ieeeexcept_first == 0)
        ieeeexcept_first = sptr;
      /* output predeclared line */
      auto s = nodollar(std::string("PD_") + tok);
      fprintf(out2, "#define %-20s%6d\n", s.c_str(), npd);

      /* init PD sym */
      DTYPEP(sptr, 0);
      PDNUMP(sptr, npd);

      tok = makeLower(getToken()); /* get pname */
      if (tok.empty()) {
        printError(WARN, ".H6, no pname for", SYMNAME(sptr));
      } else {
        PNMPTRP(sptr, tok[0] == '-' ? 0 : putsname(tok.c_str(), tok.length()));
        tok = getToken(); /* get dtype */
        if (tok.empty()) {
          printError(WARN, ".H6, no dtype for", SYMNAME(sptr));
        } else if (tok[0] != '-') {
          INTTYPP(sptr, get_type(tok));
        }
      }

      lt = getLineAndTokenize(elt);
      if (lt == LT_AT) {
        process_at(sptr);
        lt = getLineAndTokenize(elt);
      } else
        printError(WARN, "missing .AT after .H6");
    }
    ieeeexcept_last = sptr;
  }

  /* .H7 name pname dtype */
  void
  process_miscellaneous()
  {
    auto lt = getLineAndTokenize(elt);
    while (lt == LT_H7 || lt == LT_H8 || lt == LT_H9) {
      ++npd;
      auto tok = create_symbol(ST_ISOFTNENV, &sptr, "predeclared");
      if (tok[0] == '.') {                // FIXME: not clear why this
        INTASTP(sptr, intast_sym.size()); // is needed, but it's what
        intast_sym.push_back(0);          // the original symini
      }                                   // appears to be doing.
      if (lt == LT_H7) {
        STYPEP(sptr, ST_PD);
      } else if (lt == LT_H8) {
        STYPEP(sptr, ST_ISOC);
      }
      if (tok[0] != '.') {
        /* output predeclared line */
        auto s = nodollar(std::string("PD_") + tok);
        fprintf(out2, "#define %-20s%6d\n", s.c_str(), npd);
      }

      /* init PD sym */
      DTYPEP(sptr, 0);
      PDNUMP(sptr, npd);

      tok = makeLower(getToken()); /* get pname */
      if (tok.empty())
        printError(WARN, ".H7, no pname for ", SYMNAME(sptr));
      else {
        PNMPTRP(sptr, tok[0] == '-' ? 0 : putsname(tok.c_str(), tok.length()));
        tok = getToken(); /* get dtype */
        if (tok.empty()) {
          printError(WARN, ".H7, no dtype for ", SYMNAME(sptr));
        } else if (tok[0] != '-') {
          INTTYPP(sptr, get_type(tok));
        }
      }

      lt = getLineAndTokenize(elt);
      if (lt == LT_AT) {
        process_at(sptr);
        lt = getLineAndTokenize(elt);
      } else
        printError(WARN, "missing .AT after .H7");
    }
  }

  /* now write symfile */
  void
  write_symfile()
  {
    fprintf(out1, "#define HPF_LIB_FIRST %d\n", hpf_lib_first);
    fprintf(out1, "#define HPF_LIB_LAST %d\n", hpf_lib_last);
    fprintf(out1, "#define HPF_LOCAL_LIB_FIRST %d\n", hpf_local_lib_first);
    fprintf(out1, "#define HPF_LOCAL_LIB_LAST %d\n", hpf_local_lib_last);
    fprintf(out1, "#define CRAFT_FIRST %d\n", craft_first);
    fprintf(out1, "#define CRAFT_LAST %d\n", craft_last);
    fprintf(out1, "#define CRAY_FIRST %d\n", cray_first);
    fprintf(out1, "#define CRAY_LAST %d\n", cray_last);
    fprintf(out1, "#define ISO_C_FIRST %d\n", iso_c_first);
    fprintf(out1, "#define ISO_C_LAST %d\n", iso_c_last);
    fprintf(out1, "#define IEEEARITH_FIRST %d\n", ieeearith_first);
    fprintf(out1, "#define IEEEARITH_LAST %d\n", ieeearith_last);
    fprintf(out1, "#define IEEEEXCEPT_FIRST %d\n", ieeeexcept_first);
    fprintf(out1, "#define IEEEEXCEPT_LAST %d\n", ieeeexcept_last);
    fprintf(out1, "#define INIT_SYMTAB_SIZE %d\n", stb.stg_avail);
    fprintf(out1, "#define INIT_NAMES_SIZE %d\n", stb.namavl);
    fprintf(out1, "static SYM init_sym[INIT_SYMTAB_SIZE] = {\n");
    BZERO(stb.stg_base + 0, SYM, 1);
    stb.n_base[0] = '\0';
    for (SPTR i = SPTR(0); i < stb.stg_avail; ++i) {
      SYM *xp = &stb.stg_base[i];
      assert(xp->stype <= ST_MAX);
      assert(xp->sc <= SC_MAX);
      fprintf(out1, "\t{%s, %s, %3d, %3d,\t/* %s */\n",
              SYMTYPE_names[xp->stype], SC_KIND_names[xp->sc], xp->b3, xp->b4,
              SYMNAME(i));
      fprintf(out1, "\t %3d, %3d, %3d, %3d, %5d,\n", xp->dtype, xp->hashlk,
              xp->symlk, xp->scope, xp->nmptr);

      fprintf(out1, "\t ");
      for (int i = 1; i != 33; ++i) {
        fprintf(out1, "%d,", 0 /*xp->f*/);
      }
      fprintf(out1, "\n");
      fprintf(out1, "#ifdef INTIS64\n");
      fprintf(out1, "\t 0,\n");
      fprintf(out1, "#endif\n");
      fprintf(out1, "\t ");
      for (int i = 33; i != 65; ++i) {
        fprintf(out1, "%d,", 0 /*xp->f*/);
      }
      fprintf(out1, "\n");
      fprintf(out1, "#ifdef INTIS64\n");
      fprintf(out1, "\t 0,\n");
      fprintf(out1, "#endif\n");

      fprintf(out1, "\t %5d, %5ld, %5d, %5d, %5d, %5ld, %5d, %5d,\n", xp->w9,
              xp->w10, xp->w11, xp->w12, xp->w13, xp->w14, xp->w15, xp->w16);
      fprintf(out1, "\t %5d, %5d, %5d, %5d, %5d, %5d, %5d, %5d,\n", xp->w17,
              xp->w18, xp->w19, xp->w20, xp->w21, xp->w22, xp->w23, xp->w24);
      fprintf(out1, "\t %5d, %5d, %5d, %5d, %5d, %5d, %5d, %5d,\n", xp->w25,
              xp->w26, xp->w27, xp->w28, xp->uname, xp->w30, xp->w31, xp->w32);

      fprintf(out1, "\t ");
      for (int i = 65; i != 97; ++i) {
        fprintf(out1, "%d,", 0 /*xp->f*/);
      }
      fprintf(out1, "\n");
      fprintf(out1, "#ifdef INTIS64\n");
      fprintf(out1, "\t 0,\n");
      fprintf(out1, "#endif\n");

      fprintf(out1, "\t %5d, %5d, %5d,\n", xp->w34, xp->w35, xp->w36);

      fprintf(out1, "\t ");
      for (int i = 97; i != 129; ++i) {
        fprintf(out1, "%d,", 0 /*xp->f*/);
      }
      fprintf(out1, "\n");
      fprintf(out1, "#ifdef INTIS64\n");
      fprintf(out1, "\t 0,\n");
      fprintf(out1, "#endif\n");

      fprintf(out1, "\t %5d, %5d, %5d, %5d\n", xp->lineno, xp->w39,
              xp->w40, xp->palign);

      fprintf(out1, "\t},\n");
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
    fprintf(out1, "\n};\n\n");

    fprintf(out1, "const char *intrinsic_kwd[%d] = {\n", (int)intr_kwd.size());
    for (int i = 0; i != (int)intr_kwd.size(); ++i)
      fprintf(out1, "    /*%5d */  \"%s\",\n", i, intr_kwd[i].c_str());
    fprintf(out1, "\n};\n");

    oldsyms(init_names0, init_names0_size, 0);
    oldsyms(init_names1, init_names1_size, 1);
    oldsyms(init_names2, init_names2_size, 2);
    oldsyms(init_names3, init_names3_size, 3);
  }

  void
  write_out4()
  {
    fprintf(out4, "int intast_sym[%d] = {", (int)intast_sym.size());
    int j = 10;
    for (int i = 0; i != (int)intast_sym.size(); ++i) {
      if (j == 10) {
        fprintf(out4, "\n\t");
        j = 0;
      }
      ++j;
      fprintf(out4, "%5d, ", intast_sym[i]);
    }
    fprintf(out4, "\n};\n");
  }

  void
  write_out5()
  {
    fprintf(out5, "#define N_ILM %d\n", (int)ilms.size() + 1);
    for (int i = 0; i != (int)ilms.size(); ++i)
      fprintf(out5, "#define IM_%s %d\n", ilms[i].c_str(), i + 1);
  }
};

int
main(int argc, char **argv)
{
  SyminiFE90 app(argc, argv);
  return app.run();
}

/**
 * 6.1 names
 */
const char *SyminiFE90::init_names0[] = {
    "",
    "..sqrt",
    ".sqrt",
    ".dsqrt",
    "dsqrt",
    ".qsqrt",
    ".csqrt",
    "csqrt",
    ".cdsqrt",
    "cdsqrt",
    ".cqsqrt",
    ".alog",
    "alog",
    ".dlog",
    "dlog",
    ".qlog",
    ".clog",
    "clog",
    ".cdlog",
    "cdlog",
    ".cqlog",
    ".alog10",
    "alog10",
    ".dlog10",
    "dlog10",
    ".qlog10",
    "..exp",
    ".exp",
    ".dexp",
    "dexp",
    ".qexp",
    ".cexp",
    "cexp",
    ".cdexp",
    "cdexp",
    ".cqexp",
    "..sin",
    ".sin",
    ".dsin",
    "dsin",
    ".qsin",
    ".csin",
    "csin",
    ".cdsin",
    "cdsin",
    ".cqsin",
    "..sind",
    ".sind",
    ".dsind",
    "dsind",
    ".qsind",
    "..cos",
    ".cos",
    ".dcos",
    "dcos",
    ".qcos",
    ".ccos",
    "ccos",
    ".cdcos",
    "cdcos",
    ".cqcos",
    "..cosd",
    ".cosd",
    ".dcosd",
    "dcosd",
    ".qcosd",
    "..tan",
    ".tan",
    ".dtan",
    "dtan",
    ".qtan",
    "..tand",
    ".tand",
    ".dtand",
    "dtand",
    ".qtand",
    "..asin",
    ".asin",
    ".dasin",
    "dasin",
    ".qasin",
    "..asind",
    ".asind",
    ".dasind",
    "dasind",
    ".qasind",
    "..acos",
    ".acos",
    ".dacos",
    "dacos",
    ".qacos",
    "..acosd",
    ".acosd",
    ".dacosd",
    "dacosd",
    ".qacosd",
    "..atan",
    ".atan",
    ".datan",
    "datan",
    ".qatan",
    "..atand",
    ".atand",
    ".datand",
    "datand",
    ".qatand",
    "..atan2",
    ".atan2",
    ".datan2",
    "datan2",
    ".qatan2",
    "..atan2d",
    ".atan2d",
    ".datan2d",
    "datan2d",
    ".qatan2d",
    "..sinh",
    ".sinh",
    ".dsinh",
    "dsinh",
    ".qsinh",
    "..cosh",
    ".cosh",
    ".dcosh",
    "dcosh",
    ".qcosh",
    "..tanh",
    ".tanh",
    ".dtanh",
    "dtanh",
    ".qtanh",
    "iiabs",
    "jiabs",
    "kiabs",
    ".iabs",
    "iabs",
    "..abs",
    ".abs",
    ".dabs",
    "dabs",
    ".qabs",
    ".cabs",
    "cabs",
    ".cdabs",
    "cdabs",
    ".cqabs",
    "cqabs",
    "..aimag",
    ".aimag",
    ".dimag",
    "dimag",
    ".qimag",
    "..conjg",
    ".conjg",
    ".dconjg",
    "dconjg",
    ".qconjg",
    "dprod",
    "imax0",
    ".max0",
    "max0",
    ".amax1",
    "amax1",
    ".dmax1",
    "dmax1",
    ".kmax",
    "kmax0",
    ".qmax",
    "jmax0",
    "aimax0",
    "amax0",
    "max1",
    "imax1",
    "jmax1",
    "kmax1",
    "ajmax0",
    "imin0",
    ".min0",
    "min0",
    ".amin1",
    "amin1",
    ".dmin1",
    "dmin1",
    ".kmin",
    "kmin0",
    ".qmin",
    "jmin0",
    "amin0",
    "aimin0",
    "min1",
    "imin1",
    "jmin1",
    "kmin1",
    "ajmin0",
    "iidim",
    "jidim",
    ".idim",
    "idim",
    "kidim",
    "..dim",
    ".dim",
    ".ddim",
    "ddim",
    ".qdim",
    "imod",
    "jmod",
    "..mod",
    ".mod",
    "kmod",
    ".amod",
    "amod",
    ".dmod",
    "dmod",
    ".qmod",
    ".imodulo",
    "..modulo",
    ".modulo",
    ".kmodulo",
    "..amodulo",
    ".amodulo",
    "..dmodulo",
    ".dmodulo",
    ".qmodulo",
    "iisign",
    "jisign",
    ".isign",
    "isign",
    "kisign",
    "..sign",
    ".sign",
    ".dsign",
    "dsign",
    ".qsign",
    "iiand",
    ".jiand",
    "jiand",
    ".kiand",
    "iior",
    ".jior",
    "jior",
    ".kior",
    "iieor",
    ".jieor",
    "jieor",
    ".kieor",
    "inot",
    ".jnot",
    "jnot",
    ".knot",
    "iishft",
    ".jishft",
    "jishft",
    "kishft",
    "iibits",
    ".jibits",
    "jibits",
    "kibits",
    "iibset",
    ".jibset",
    "jibset",
    "kibset",
    "bitest",
    ".bjtest",
    "bjtest",
    "bktest",
    "iibclr",
    ".jibclr",
    "jibclr",
    "kibclr",
    "iishftc",
    ".jishftc",
    "jishftc",
    "kishftc",
    ".ilshift",
    ".jlshift",
    ".klshift",
    ".irshift",
    ".jrshift",
    ".krshift",
    ".2sch",
    ".char",
    ".2kch",
    "ichar",
    "lge",
    "lgt",
    "lle",
    "llt",
    "nchar",
    "nlen",
    "nindex",
    "loc",
    "idint",
    "jidint",
    ".2i",
    "ifix",
    "jifix",
    ".jint",
    "iifix",
    ".iint",
    ".2si",
    "int8",
    "iidint",
    "floati",
    "floatj",
    "float",
    "sngl",
    ".2r",
    "dfloti",
    "dfloat",
    "dflotj",
    "dreal",
    ".2d",
    ".2c",
    ".2cd",
    "dint",
    "dnint",
    "..inint",
    ".inint",
    "iidnnt",
    "idnint",
    "..jnint",
    ".jnint",
    "jidnnt",
    "knint",
    "kidnnt",
    "iand",
    "ior",
    "ieor",
    "xor",
    "not",
    "ishft",
    "iint",
    "jint",
    "dble",
    "dcmplx",
    "imag",
    "aimag",
    "conjg",
    "inint",
    "jnint",
    "abs",
    "mod",
    "sign",
    "dim",
    "max",
    "min",
    "sqrt",
    "exp",
    "log",
    "log10",
    "sin",
    "sind",
    "cos",
    "cosd",
    "tan",
    "tand",
    "asin",
    "asind",
    "acos",
    "acosd",
    "atan",
    "atand",
    "atan2",
    "atan2d",
    "sinh",
    "cosh",
    "tanh",
    "ibits",
    "ibset",
    "btest",
    "ibclr",
    "ishftc",
    "lshift",
    "rshift",
    "modulo",
    "date",
    "exit",
    "idate",
    "time",
    "mvbits",
    "real",
    "cmplx",
    "int",
    "aint",
    "anint",
    "nint",
    "char",
    "zext",
    "izext",
    "jzext",
    "ceiling",
    "floor",
    "all",
    "and",
    "any",
    "compl",
    "count",
    "isnan",
    "dot_product",
    "eqv",
    "matmul",
    "matmul_transpose",
    "maxloc",
    "maxval",
    "minloc",
    "minval",
    "merge",
    "neqv",
    "or",
    "pack",
    "product",
    "ran",
    "secnds",
    "shift",
    "sum",
    "spread",
    "transpose",
    "unpack",
    "number_of_processors",
    "lbound",
    "ubound",
    "cshift",
    "eoshift",
    "reshape",
    "shape",
    "size",
    "allocated",
    "date_and_time",
    "cpu_time",
    "random_number",
    "random_seed",
    "system_clock",
    "present",
    "kind",
    "selected_int_kind",
    "selected_real_kind",
    "dlbound",
    "dubound",
    "dshape",
    "dsize",
    "achar",
    "adjustl",
    "adjustr",
    "bit_size",
    "digits",
    "epsilon",
    "exponent",
    "fraction",
    "huge",
    "iachar",
    "index",
    "kindex",
    "logical",
    "maxexponent",
    "minexponent",
    "nearest",
    "precision",
    "radix",
    "range",
    "repeat",
    "rrspacing",
    "scale",
    "set_exponent",
    "spacing",
    "tiny",
    "transfer",
    "trim",
    "verify",
    "scan",
    "len",
    "klen",
    "len_trim",
    "dotproduct",
    "ilen",
    "null",
    "processors_shape",
    ".lastval",
    ".reduce_sum",
    ".reduce_product",
    ".reduce_any",
    ".reduce_all",
    ".reduce_parity",
    ".reduce_iany",
    ".reduce_iall",
    ".reduce_iparity",
    ".reduce_minval",
    ".reduce_maxval",
    ".reduce_firstmax",
    ".reduce_lastmax",
    ".reduce_firstmin",
    ".reduce_lastmin",
    "associated",
    ".ptr2_assign",
    ".nullify",
    ".ptr_copyin",
    ".ptr_copyout",
    ".copyin",
    ".copyout",
    "ranf",
    "ranget",
    "ranset",
    "unit",
    "length",
    "int_mult_upper",
    "cot",
    "dcot",
    "shiftl",
    "shiftr",
    "dshiftl",
    "dshiftr",
    "mask",
};

/**
 * 6.2 names
 */
const char *SyminiFE90::init_names1[] = {
    "",
    "..sqrt",
    ".sqrt",
    ".dsqrt",
    "dsqrt",
    ".qsqrt",
    ".csqrt",
    "csqrt",
    ".cdsqrt",
    "cdsqrt",
    ".cqsqrt",
    ".alog",
    "alog",
    ".dlog",
    "dlog",
    ".qlog",
    ".clog",
    "clog",
    ".cdlog",
    "cdlog",
    ".cqlog",
    ".alog10",
    "alog10",
    ".dlog10",
    "dlog10",
    ".qlog10",
    "..exp",
    ".exp",
    ".dexp",
    "dexp",
    ".qexp",
    ".cexp",
    "cexp",
    ".cdexp",
    "cdexp",
    ".cqexp",
    "..sin",
    ".sin",
    ".dsin",
    "dsin",
    ".qsin",
    ".csin",
    "csin",
    ".cdsin",
    "cdsin",
    ".cqsin",
    "..sind",
    ".sind",
    ".dsind",
    "dsind",
    ".qsind",
    "..cos",
    ".cos",
    ".dcos",
    "dcos",
    ".qcos",
    ".ccos",
    "ccos",
    ".cdcos",
    "cdcos",
    ".cqcos",
    "..cosd",
    ".cosd",
    ".dcosd",
    "dcosd",
    ".qcosd",
    "..tan",
    ".tan",
    ".dtan",
    "dtan",
    ".qtan",
    "..tand",
    ".tand",
    ".dtand",
    "dtand",
    ".qtand",
    "..asin",
    ".asin",
    ".dasin",
    "dasin",
    ".qasin",
    "..asind",
    ".asind",
    ".dasind",
    "dasind",
    ".qasind",
    "..acos",
    ".acos",
    ".dacos",
    "dacos",
    ".qacos",
    "..acosd",
    ".acosd",
    ".dacosd",
    "dacosd",
    ".qacosd",
    "..atan",
    ".atan",
    ".datan",
    "datan",
    ".qatan",
    "..atand",
    ".atand",
    ".datand",
    "datand",
    ".qatand",
    "..atan2",
    ".atan2",
    ".datan2",
    "datan2",
    ".qatan2",
    "..atan2d",
    ".atan2d",
    ".datan2d",
    "datan2d",
    ".qatan2d",
    "..sinh",
    ".sinh",
    ".dsinh",
    "dsinh",
    ".qsinh",
    "..cosh",
    ".cosh",
    ".dcosh",
    "dcosh",
    ".qcosh",
    "..tanh",
    ".tanh",
    ".dtanh",
    "dtanh",
    ".qtanh",
    "iiabs",
    "jiabs",
    "kiabs",
    ".iabs",
    "iabs",
    "..abs",
    ".abs",
    ".dabs",
    "dabs",
    ".qabs",
    ".cabs",
    "cabs",
    ".cdabs",
    "cdabs",
    ".cqabs",
    "cqabs",
    "..aimag",
    ".aimag",
    ".dimag",
    "dimag",
    ".qimag",
    "..conjg",
    ".conjg",
    ".dconjg",
    "dconjg",
    ".qconjg",
    "dprod",
    "imax0",
    ".max0",
    "max0",
    ".amax1",
    "amax1",
    ".dmax1",
    "dmax1",
    ".kmax",
    "kmax0",
    ".qmax",
    "jmax0",
    "aimax0",
    "amax0",
    "max1",
    "imax1",
    "jmax1",
    "kmax1",
    "ajmax0",
    "imin0",
    ".min0",
    "min0",
    ".amin1",
    "amin1",
    ".dmin1",
    "dmin1",
    ".kmin",
    "kmin0",
    ".qmin",
    "jmin0",
    "amin0",
    "aimin0",
    "min1",
    "imin1",
    "jmin1",
    "kmin1",
    "ajmin0",
    "iidim",
    "jidim",
    ".idim",
    "idim",
    "kidim",
    "..dim",
    ".dim",
    ".ddim",
    "ddim",
    ".qdim",
    "imod",
    "jmod",
    "..mod",
    ".mod",
    "kmod",
    ".amod",
    "amod",
    ".dmod",
    "dmod",
    ".qmod",
    ".imodulo",
    "..modulo",
    ".modulo",
    ".kmodulo",
    "..amodulo",
    ".amodulo",
    "..dmodulo",
    ".dmodulo",
    ".qmodulo",
    "iisign",
    "jisign",
    ".isign",
    "isign",
    "kisign",
    "..sign",
    ".sign",
    ".dsign",
    "dsign",
    ".qsign",
    "iiand",
    ".jiand",
    "jiand",
    ".kiand",
    "iior",
    ".jior",
    "jior",
    ".kior",
    "iieor",
    ".jieor",
    "jieor",
    ".kieor",
    "inot",
    ".jnot",
    "jnot",
    ".knot",
    "iishft",
    ".jishft",
    "jishft",
    "kishft",
    "iibits",
    ".jibits",
    "jibits",
    "kibits",
    "iibset",
    ".jibset",
    "jibset",
    "kibset",
    "bitest",
    ".bjtest",
    "bjtest",
    "bktest",
    "iibclr",
    ".jibclr",
    "jibclr",
    "kibclr",
    "iishftc",
    ".jishftc",
    "jishftc",
    "kishftc",
    ".ilshift",
    ".jlshift",
    ".klshift",
    ".irshift",
    ".jrshift",
    ".krshift",
    ".2sch",
    ".char",
    ".2kch",
    "ichar",
    "lge",
    "lgt",
    "lle",
    "llt",
    "nchar",
    "nlen",
    "nindex",
    "loc",
    "idint",
    "jidint",
    ".2i",
    "ifix",
    "jifix",
    ".jint",
    "iifix",
    ".iint",
    ".2si",
    "int1",
    "int2",
    "int4",
    "int8",
    "iidint",
    "floati",
    "floatj",
    "float",
    "sngl",
    ".2r",
    "dfloti",
    "dfloat",
    "dflotj",
    "dreal",
    ".2d",
    ".2c",
    ".2cd",
    "dint",
    "dnint",
    "..inint",
    ".inint",
    "iidnnt",
    "idnint",
    "..jnint",
    ".jnint",
    "jidnnt",
    "knint",
    "kidnnt",
    "iand",
    "ior",
    "ieor",
    "xor",
    "not",
    "ishft",
    "iint",
    "jint",
    "dble",
    "dcmplx",
    "imag",
    "aimag",
    "conjg",
    "inint",
    "jnint",
    "abs",
    "mod",
    "sign",
    "dim",
    "max",
    "min",
    "sqrt",
    "exp",
    "log",
    "log10",
    "sin",
    "sind",
    "cos",
    "cosd",
    "tan",
    "tand",
    "asin",
    "asind",
    "acos",
    "acosd",
    "atan",
    "atand",
    "atan2",
    "atan2d",
    "sinh",
    "cosh",
    "tanh",
    "ibits",
    "ibset",
    "btest",
    "ibclr",
    "ishftc",
    "lshift",
    "rshift",
    "modulo",
    "date",
    "exit",
    "idate",
    "time",
    "mvbits",
    "real",
    "cmplx",
    "int",
    "aint",
    "anint",
    "nint",
    "char",
    "zext",
    "izext",
    "jzext",
    "ceiling",
    "floor",
    "all",
    "and",
    "any",
    "compl",
    "count",
    "isnan",
    "dot_product",
    "eqv",
    "matmul",
    "matmul_transpose",
    "maxloc",
    "maxval",
    "minloc",
    "minval",
    "merge",
    "neqv",
    "or",
    "pack",
    "product",
    "ran",
    "secnds",
    "shift",
    "sum",
    "spread",
    "transpose",
    "unpack",
    "number_of_processors",
    "lbound",
    "ubound",
    "cshift",
    "eoshift",
    "reshape",
    "shape",
    "size",
    "allocated",
    "date_and_time",
    "cpu_time",
    "random_number",
    "random_seed",
    "system_clock",
    "present",
    "kind",
    "selected_int_kind",
    "selected_real_kind",
    "dlbound",
    "dubound",
    "dshape",
    "dsize",
    "achar",
    "adjustl",
    "adjustr",
    "bit_size",
    "digits",
    "epsilon",
    "exponent",
    "fraction",
    "huge",
    "iachar",
    "index",
    "kindex",
    "logical",
    "maxexponent",
    "minexponent",
    "nearest",
    "precision",
    "radix",
    "range",
    "repeat",
    "rrspacing",
    "scale",
    "set_exponent",
    "spacing",
    "tiny",
    "transfer",
    "trim",
    "verify",
    "scan",
    "len",
    "klen",
    "len_trim",
    "dotproduct",
    "ilen",
    "null",
    "int_ptr_kind",
    "processors_shape",
    ".lastval",
    ".reduce_sum",
    ".reduce_product",
    ".reduce_any",
    ".reduce_all",
    ".reduce_parity",
    ".reduce_iany",
    ".reduce_iall",
    ".reduce_iparity",
    ".reduce_minval",
    ".reduce_maxval",
    ".reduce_firstmax",
    ".reduce_lastmax",
    ".reduce_firstmin",
    ".reduce_lastmin",
    "associated",
    ".ptr2_assign",
    ".nullify",
    ".ptr_copyin",
    ".ptr_copyout",
    ".copyin",
    ".copyout",
    "ranf",
    "ranget",
    "ranset",
    "unit",
    "length",
    "int_mult_upper",
    "cot",
    "dcot",
    "shiftl",
    "shiftr",
    "dshiftl",
    "dshiftr",
    "mask",
};

/**
 * 7.0 names
 */
const char *SyminiFE90::init_names2[] = {
    "",
    "..sqrt",
    ".sqrt",
    ".dsqrt",
    "dsqrt",
    ".qsqrt",
    ".csqrt",
    "csqrt",
    ".cdsqrt",
    "cdsqrt",
    ".cqsqrt",
    ".alog",
    "alog",
    ".dlog",
    "dlog",
    ".qlog",
    ".clog",
    "clog",
    ".cdlog",
    "cdlog",
    ".cqlog",
    ".alog10",
    "alog10",
    ".dlog10",
    "dlog10",
    ".qlog10",
    "..exp",
    ".exp",
    ".dexp",
    "dexp",
    ".qexp",
    ".cexp",
    "cexp",
    ".cdexp",
    "cdexp",
    ".cqexp",
    "..sin",
    ".sin",
    ".dsin",
    "dsin",
    ".qsin",
    ".csin",
    "csin",
    ".cdsin",
    "cdsin",
    ".cqsin",
    "..sind",
    ".sind",
    ".dsind",
    "dsind",
    ".qsind",
    "..cos",
    ".cos",
    ".dcos",
    "dcos",
    ".qcos",
    ".ccos",
    "ccos",
    ".cdcos",
    "cdcos",
    ".cqcos",
    "..cosd",
    ".cosd",
    ".dcosd",
    "dcosd",
    ".qcosd",
    "..tan",
    ".tan",
    ".dtan",
    "dtan",
    ".qtan",
    "..tand",
    ".tand",
    ".dtand",
    "dtand",
    ".qtand",
    "..asin",
    ".asin",
    ".dasin",
    "dasin",
    ".qasin",
    "..asind",
    ".asind",
    ".dasind",
    "dasind",
    ".qasind",
    "..acos",
    ".acos",
    ".dacos",
    "dacos",
    ".qacos",
    "..acosd",
    ".acosd",
    ".dacosd",
    "dacosd",
    ".qacosd",
    "..atan",
    ".atan",
    ".datan",
    "datan",
    ".qatan",
    "..atand",
    ".atand",
    ".datand",
    "datand",
    ".qatand",
    "..atan2",
    ".atan2",
    ".datan2",
    "datan2",
    ".qatan2",
    "..atan2d",
    ".atan2d",
    ".datan2d",
    "datan2d",
    ".qatan2d",
    "..sinh",
    ".sinh",
    ".dsinh",
    "dsinh",
    ".qsinh",
    "..cosh",
    ".cosh",
    ".dcosh",
    "dcosh",
    ".qcosh",
    "..tanh",
    ".tanh",
    ".dtanh",
    "dtanh",
    ".qtanh",
    "iiabs",
    "jiabs",
    "kiabs",
    ".iabs",
    "iabs",
    "..abs",
    ".abs",
    ".dabs",
    "dabs",
    ".qabs",
    ".cabs",
    "cabs",
    ".cdabs",
    "cdabs",
    ".cqabs",
    "cqabs",
    "..aimag",
    ".aimag",
    ".dimag",
    "dimag",
    ".qimag",
    "..conjg",
    ".conjg",
    ".dconjg",
    "dconjg",
    ".qconjg",
    "dprod",
    "imax0",
    ".max0",
    "max0",
    ".amax1",
    "amax1",
    ".dmax1",
    "dmax1",
    ".kmax",
    "kmax0",
    ".qmax",
    "jmax0",
    "aimax0",
    "amax0",
    "max1",
    "imax1",
    "jmax1",
    "kmax1",
    "ajmax0",
    "imin0",
    ".min0",
    "min0",
    ".amin1",
    "amin1",
    ".dmin1",
    "dmin1",
    ".kmin",
    "kmin0",
    ".qmin",
    "jmin0",
    "amin0",
    "aimin0",
    "min1",
    "imin1",
    "jmin1",
    "kmin1",
    "ajmin0",
    "iidim",
    "jidim",
    ".idim",
    "idim",
    "kidim",
    "..dim",
    ".dim",
    ".ddim",
    "ddim",
    ".qdim",
    "imod",
    "jmod",
    "..mod",
    ".mod",
    "kmod",
    ".amod",
    "amod",
    ".dmod",
    "dmod",
    ".qmod",
    ".imodulo",
    "..modulo",
    ".modulo",
    ".kmodulo",
    "..amodulo",
    ".amodulo",
    "..dmodulo",
    ".dmodulo",
    ".qmodulo",
    "iisign",
    "jisign",
    ".isign",
    "isign",
    "kisign",
    "..sign",
    ".sign",
    ".dsign",
    "dsign",
    ".qsign",
    "iiand",
    ".jiand",
    "jiand",
    ".kiand",
    "iior",
    ".jior",
    "jior",
    ".kior",
    "iieor",
    ".jieor",
    "jieor",
    ".kieor",
    "inot",
    ".jnot",
    "jnot",
    ".knot",
    "iishft",
    ".jishft",
    "jishft",
    "kishft",
    "iibits",
    ".jibits",
    "jibits",
    "kibits",
    "iibset",
    ".jibset",
    "jibset",
    "kibset",
    "bitest",
    ".bjtest",
    "bjtest",
    "bktest",
    "iibclr",
    ".jibclr",
    "jibclr",
    "kibclr",
    "iishftc",
    ".jishftc",
    "jishftc",
    "kishftc",
    ".ilshift",
    ".jlshift",
    ".klshift",
    ".irshift",
    ".jrshift",
    ".krshift",
    ".2sch",
    ".char",
    ".2kch",
    "ichar",
    "lge",
    "lgt",
    "lle",
    "llt",
    "nchar",
    "nlen",
    "nindex",
    "loc",
    "idint",
    "jidint",
    ".2i",
    "ifix",
    "jifix",
    ".jint",
    "iifix",
    ".iint",
    ".2si",
    "int1",
    "int2",
    "int4",
    "int8",
    "iidint",
    "floati",
    "floatj",
    "float",
    "sngl",
    ".2r",
    "dfloti",
    "dfloat",
    "dflotj",
    "dreal",
    ".2d",
    ".2c",
    ".2cd",
    "dint",
    "dnint",
    "..inint",
    ".inint",
    "iidnnt",
    "idnint",
    "..jnint",
    ".jnint",
    "jidnnt",
    "knint",
    "kidnnt",
    "iand",
    "ior",
    "ieor",
    "xor",
    "not",
    "ishft",
    "iint",
    "jint",
    "dble",
    "dcmplx",
    "imag",
    "aimag",
    "conjg",
    "inint",
    "jnint",
    "abs",
    "mod",
    "sign",
    "dim",
    "max",
    "min",
    "sqrt",
    "exp",
    "log",
    "log10",
    "sin",
    "sind",
    "cos",
    "cosd",
    "tan",
    "tand",
    "asin",
    "asind",
    "acos",
    "acosd",
    "atan",
    "atand",
    "atan2",
    "atan2d",
    "sinh",
    "cosh",
    "tanh",
    "ibits",
    "ibset",
    "btest",
    "ibclr",
    "ishftc",
    "lshift",
    "rshift",
    "modulo",
    "date",
    "exit",
    "idate",
    "time",
    "mvbits",
    "real",
    "cmplx",
    "int",
    "aint",
    "anint",
    "nint",
    "char",
    "zext",
    "izext",
    "jzext",
    "ceiling",
    "floor",
    "all",
    "and",
    "any",
    "compl",
    "count",
    "isnan",
    "dot_product",
    "eqv",
    "matmul",
    "matmul_transpose",
    "maxloc",
    "maxval",
    "minloc",
    "minval",
    "merge",
    "neqv",
    "or",
    "pack",
    "product",
    "ran",
    "secnds",
    "shift",
    "sum",
    "spread",
    "transpose",
    "unpack",
    "number_of_processors",
    "lbound",
    "ubound",
    "cshift",
    "eoshift",
    "reshape",
    "shape",
    "size",
    "allocated",
    "date_and_time",
    "cpu_time",
    "random_number",
    "random_seed",
    "system_clock",
    "present",
    "kind",
    "selected_int_kind",
    "selected_real_kind",
    "dlbound",
    "dubound",
    "dshape",
    "dsize",
    "achar",
    "adjustl",
    "adjustr",
    "bit_size",
    "digits",
    "epsilon",
    "exponent",
    "fraction",
    "huge",
    "iachar",
    "index",
    "kindex",
    "logical",
    "maxexponent",
    "minexponent",
    "nearest",
    "precision",
    "radix",
    "range",
    "repeat",
    "rrspacing",
    "scale",
    "set_exponent",
    "spacing",
    "tiny",
    "transfer",
    "trim",
    "verify",
    "scan",
    "len",
    "klen",
    "len_trim",
    "dotproduct",
    "ilen",
    "null",
    "int_ptr_kind",
    "processors_shape",
    ".lastval",
    ".reduce_sum",
    ".reduce_product",
    ".reduce_any",
    ".reduce_all",
    ".reduce_parity",
    ".reduce_iany",
    ".reduce_iall",
    ".reduce_iparity",
    ".reduce_minval",
    ".reduce_maxval",
    ".reduce_firstmax",
    ".reduce_lastmax",
    ".reduce_firstmin",
    ".reduce_lastmin",
    "associated",
    ".ptr2_assign",
    ".nullify",
    ".ptr_copyin",
    ".ptr_copyout",
    ".copyin",
    ".copyout",
    ".selected_char_kind",
    ".extends_type_of",
    "new_line",
    ".same_type_as",
    ".move_alloc",
    ".getarg",
    ".command_argument_count",
    ".get_command",
    ".get_command_argument",
    ".get_environment_variable",
    "is_iostat_end",
    "is_iostat_eor",
    ".sizeof",
    "ranf",
    "ranget",
    "ranset",
    "unit",
    "length",
    "int_mult_upper",
    "cot",
    "dcot",
    "shiftl",
    "shiftr",
    "dshiftl",
    "dshiftr",
    "mask",
    "c_loc",
    "c_funloc",
    "c_associated",
    "c_f_pointer",
    "c_f_procpointer",
    "ieee_support_datatype",
    "ieee_support_denormal",
    "ieee_support_divide",
    "ieee_support_inf",
    "ieee_support_io",
    "ieee_support_nan",
    "ieee_support_rounding",
    "ieee_support_sqrt",
    "ieee_support_standard",
    "ieee_support_underflow_control",
    "ieee_class",
    "ieee_copy_sign",
    "ieee_is_finite",
    "ieee_is_nan",
    "ieee_is_normal",
    "ieee_is_negative",
    "ieee_logb",
    "ieee_next_after",
    "ieee_rem",
    "ieee_rint",
    "ieee_scalb",
    "ieee_unordered",
    "ieee_value",
    "ieee_selected_real_kind",
    "ieee_get_rounding_mode",
    "ieee_get_underflow_mode",
    "ieee_set_rounding_mode",
    "ieee_set_underflow_mode",
    "ieee_support_flag",
    "ieee_support_halting",
    "ieee_get_flag",
    "ieee_get_halting_mode",
    "ieee_get_status",
    "ieee_set_flag",
    "ieee_set_halting_mode",
    "ieee_set_status",
};

/**
 * Fortran 2008 names
 */
const char *SyminiFE90::init_names3[] = {
    "",
    "..sqrt",
    ".sqrt",
    ".dsqrt",
    "dsqrt",
    ".qsqrt",
    ".csqrt",
    "csqrt",
    ".cdsqrt",
    "cdsqrt",
    ".cqsqrt",
    ".alog",
    "alog",
    ".dlog",
    "dlog",
    ".qlog",
    ".clog",
    "clog",
    ".cdlog",
    "cdlog",
    ".cqlog",
    ".alog10",
    "alog10",
    ".dlog10",
    "dlog10",
    ".qlog10",
    "..exp",
    ".exp",
    ".dexp",
    "dexp",
    ".qexp",
    ".cexp",
    "cexp",
    ".cdexp",
    "cdexp",
    ".cqexp",
    "..sin",
    ".sin",
    ".dsin",
    "dsin",
    ".qsin",
    ".csin",
    "csin",
    ".cdsin",
    "cdsin",
    ".cqsin",
    "..sind",
    ".sind",
    ".dsind",
    "dsind",
    ".qsind",
    "..cos",
    ".cos",
    ".dcos",
    "dcos",
    ".qcos",
    ".ccos",
    "ccos",
    ".cdcos",
    "cdcos",
    ".cqcos",
    "..cosd",
    ".cosd",
    ".dcosd",
    "dcosd",
    ".qcosd",
    "..tan",
    ".tan",
    ".dtan",
    "dtan",
    ".qtan",
    "..tand",
    ".tand",
    ".dtand",
    "dtand",
    ".qtand",
    "..asin",
    ".asin",
    ".dasin",
    "dasin",
    ".qasin",
    "..asind",
    ".asind",
    ".dasind",
    "dasind",
    ".qasind",
    "..acos",
    ".acos",
    ".dacos",
    "dacos",
    ".qacos",
    "..acosd",
    ".acosd",
    ".dacosd",
    "dacosd",
    ".qacosd",
    "..atan",
    ".atan",
    ".datan",
    "datan",
    ".qatan",
    "..atand",
    ".atand",
    ".datand",
    "datand",
    ".qatand",
    "..atan2",
    ".atan2",
    ".datan2",
    "datan2",
    ".qatan2",
    "..atan2d",
    ".atan2d",
    ".datan2d",
    "datan2d",
    ".qatan2d",
    "..sinh",
    ".sinh",
    ".dsinh",
    "dsinh",
    ".qsinh",
    "..cosh",
    ".cosh",
    ".dcosh",
    "dcosh",
    ".qcosh",
    "..tanh",
    ".tanh",
    ".dtanh",
    "dtanh",
    ".qtanh",
    "iiabs",
    "jiabs",
    "kiabs",
    ".iabs",
    "iabs",
    "..abs",
    ".abs",
    ".dabs",
    "dabs",
    ".qabs",
    ".cabs",
    "cabs",
    ".cdabs",
    "cdabs",
    ".cqabs",
    "cqabs",
    "..aimag",
    ".aimag",
    ".dimag",
    "dimag",
    ".qimag",
    "..conjg",
    ".conjg",
    ".dconjg",
    "dconjg",
    ".qconjg",
    "dprod",
    "imax0",
    ".max0",
    "max0",
    ".amax1",
    "amax1",
    ".dmax1",
    "dmax1",
    ".kmax",
    "kmax0",
    ".qmax",
    "jmax0",
    "aimax0",
    "amax0",
    "max1",
    "imax1",
    "jmax1",
    "kmax1",
    "ajmax0",
    "imin0",
    ".min0",
    "min0",
    ".amin1",
    "amin1",
    ".dmin1",
    "dmin1",
    ".kmin",
    "kmin0",
    ".qmin",
    "jmin0",
    "amin0",
    "aimin0",
    "min1",
    "imin1",
    "jmin1",
    "kmin1",
    "ajmin0",
    "iidim",
    "jidim",
    ".idim",
    "idim",
    "kidim",
    "..dim",
    ".dim",
    ".ddim",
    "ddim",
    ".qdim",
    "imod",
    "jmod",
    "..mod",
    ".mod",
    "kmod",
    ".amod",
    "amod",
    ".dmod",
    "dmod",
    ".qmod",
    ".imodulo",
    "..modulo",
    ".modulo",
    ".kmodulo",
    "..amodulo",
    ".amodulo",
    "..dmodulo",
    ".dmodulo",
    ".qmodulo",
    "iisign",
    "jisign",
    ".isign",
    "isign",
    "kisign",
    "..sign",
    ".sign",
    ".dsign",
    "dsign",
    ".qsign",
    "iiand",
    ".jiand",
    "jiand",
    ".kiand",
    "iior",
    ".jior",
    "jior",
    ".kior",
    "iieor",
    ".jieor",
    "jieor",
    ".kieor",
    "inot",
    ".jnot",
    "jnot",
    ".knot",
    "iishft",
    ".jishft",
    "jishft",
    "kishft",
    "iibits",
    ".jibits",
    "jibits",
    "kibits",
    "iibset",
    ".jibset",
    "jibset",
    "kibset",
    "bitest",
    ".bjtest",
    "bjtest",
    "bktest",
    "iibclr",
    ".jibclr",
    "jibclr",
    "kibclr",
    "iishftc",
    ".jishftc",
    "jishftc",
    "kishftc",
    ".ilshift",
    ".jlshift",
    ".klshift",
    ".irshift",
    ".jrshift",
    ".krshift",
    ".2sch",
    ".char",
    ".2kch",
    "ichar",
    "lge",
    "lgt",
    "lle",
    "llt",
    "nchar",
    "nlen",
    "nindex",
    "loc",
    "idint",
    "jidint",
    ".2i",
    "ifix",
    "jifix",
    ".jint",
    "iifix",
    ".iint",
    ".2si",
    "int1",
    "int2",
    "int4",
    "int8",
    "iidint",
    "floati",
    "floatj",
    "float",
    "sngl",
    ".2r",
    "dfloti",
    "dfloat",
    "dflotj",
    "dreal",
    ".2d",
    ".2c",
    ".2cd",
    "dint",
    "dnint",
    "..inint",
    ".inint",
    "iidnnt",
    "idnint",
    "..jnint",
    ".jnint",
    "jidnnt",
    "knint",
    "kidnnt",
    "iand",
    "ior",
    "ieor",
    "xor",
    "not",
    "ishft",
    "iint",
    "jint",
    "dble",
    "dcmplx",
    "imag",
    "aimag",
    "conjg",
    "inint",
    "jnint",
    "abs",
    "mod",
    "sign",
    "dim",
    "max",
    "min",
    "sqrt",
    "exp",
    "log",
    "log10",
    "sin",
    "sind",
    "cos",
    "cosd",
    "tan",
    "tand",
    "asin",
    "asind",
    "acos",
    "acosd",
    "atan",
    "atand",
    "atan2",
    "atan2d",
    "sinh",
    "cosh",
    "tanh",
    "ibits",
    "ibset",
    "btest",
    "ibclr",
    "ishftc",
    "lshift",
    "rshift",
    "modulo",
    "date",
    "exit",
    "idate",
    "time",
    "mvbits",
    "real",
    "cmplx",
    "int",
    "aint",
    "anint",
    "nint",
    "char",
    "zext",
    "izext",
    "jzext",
    "ceiling",
    "floor",
    "all",
    "and",
    "any",
    "compl",
    "count",
    "isnan",
    "dot_product",
    "eqv",
    "matmul",
    "matmul_transpose",
    "maxloc",
    "maxval",
    "minloc",
    "minval",
    "merge",
    "neqv",
    "or",
    "pack",
    "product",
    "ran",
    "secnds",
    "shift",
    "sum",
    "spread",
    "transpose",
    "unpack",
    "number_of_processors",
    "lbound",
    "ubound",
    "cshift",
    "eoshift",
    "reshape",
    "shape",
    "size",
    "allocated",
    "date_and_time",
    "cpu_time",
    "random_number",
    "random_seed",
    "system_clock",
    "present",
    "kind",
    "selected_int_kind",
    "selected_real_kind",
    "dlbound",
    "dubound",
    "dshape",
    "dsize",
    "achar",
    "adjustl",
    "adjustr",
    "bit_size",
    "digits",
    "epsilon",
    "exponent",
    "fraction",
    "huge",
    "iachar",
    "index",
    "kindex",
    "logical",
    "maxexponent",
    "minexponent",
    "nearest",
    "precision",
    "radix",
    "range",
    "repeat",
    "rrspacing",
    "scale",
    "set_exponent",
    "spacing",
    "tiny",
    "transfer",
    "trim",
    "verify",
    "scan",
    "len",
    "klen",
    "len_trim",
    "dotproduct",
    "ilen",
    "null",
    "int_ptr_kind",
    "processors_shape",
    ".lastval",
    ".reduce_sum",
    ".reduce_product",
    ".reduce_any",
    ".reduce_all",
    ".reduce_parity",
    ".reduce_iany",
    ".reduce_iall",
    ".reduce_iparity",
    ".reduce_minval",
    ".reduce_maxval",
    ".reduce_firstmax",
    ".reduce_lastmax",
    ".reduce_firstmin",
    ".reduce_lastmin",
    "associated",
    ".ptr2_assign",
    ".nullify",
    ".ptr_copyin",
    ".ptr_copyout",
    ".copyin",
    ".copyout",
    ".selected_char_kind",
    ".extends_type_of",
    "new_line",
    ".same_type_as",
    ".move_alloc",
    ".getarg",
    ".command_argument_count",
    ".get_command",
    ".get_command_argument",
    ".get_environment_variable",
    "is_iostat_end",
    "is_iostat_eor",
    ".sizeof",
    "ranf",
    "ranget",
    "ranset",
    "unit",
    "length",
    "int_mult_upper",
    "cot",
    "dcot",
    "shiftl",
    "shiftr",
    "dshiftl",
    "dshiftr",
    "mask",
    "c_loc",
    "c_funloc",
    "c_associated",
    "c_f_pointer",
    "c_f_procpointer",
    "ieee_support_datatype",
    "ieee_support_denormal",
    "ieee_support_divide",
    "ieee_support_inf",
    "ieee_support_io",
    "ieee_support_nan",
    "ieee_support_rounding",
    "ieee_support_sqrt",
    "ieee_support_standard",
    "ieee_support_underflow_control",
    "ieee_class",
    "ieee_copy_sign",
    "ieee_is_finite",
    "ieee_is_nan",
    "ieee_is_normal",
    "ieee_is_negative",
    "ieee_logb",
    "ieee_next_after",
    "ieee_rem",
    "ieee_rint",
    "ieee_scalb",
    "ieee_unordered",
    "ieee_value",
    "ieee_selected_real_kind",
    "ieee_get_rounding_mode",
    "ieee_get_underflow_mode",
    "ieee_set_rounding_mode",
    "ieee_set_underflow_mode",
    "ieee_support_flag",
    "ieee_support_halting",
    "ieee_get_flag",
    "ieee_get_halting_mode",
    "ieee_get_status",
    "ieee_set_flag",
    "ieee_set_halting_mode",
    "ieee_set_status",
    "leadz",
    "trailz",
    "popcnt",
    "poppar",
};

const size_t SyminiFE90::init_names0_size =
    sizeof(SyminiFE90::init_names0) / sizeof(char *);
const size_t SyminiFE90::init_names1_size =
    sizeof(SyminiFE90::init_names1) / sizeof(char *);
const size_t SyminiFE90::init_names2_size =
    sizeof(SyminiFE90::init_names2) / sizeof(char *);
const size_t SyminiFE90::init_names3_size =
    sizeof(SyminiFE90::init_names3) / sizeof(char *);
