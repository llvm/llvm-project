/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "utils.h"
typedef void ILM_T; /* FIXME: used in ilm.h, can be anything, doesn't
                      affect code in this file. */
/* FIXME comp/comp.shared/shared/src/ contains sched.h which gets included
   instead of the system sched.h needed by chain of includes starting
   in STL's <string>, which is included in "utils.h".  To prevent the PGI's
   "sched.h" from being included, the more specific path is used to include
   ilm.h".  Moreover, for flang builds this file is in f90exe subdirectory. */
#include "flang2exe/ilm.h"
#include <sstream>

/* The following files are opened:
   IN1  -  input file to the utility (ilmtp.n)
   IN2  -  input file (ilitp.n)
   OUT1 -  output file which will contain the initializations of attributes
           and the templates of the ilms
   OUT2 -  output file which will contain the macros which define the ilm
           numbers and various associated information
   OUT3 -  output file which will contain the nroff input file for the
           specific language we are processing (i.e. all other languages
           have their macros stripped from the input ilmtp.n file).
*/

class ILMTPApp : UtilityApplication {

  enum {
    LT_IL = 1, // line type .IL ilm type ...
    LT_AT,     // line type .AT ilm attributes
    LT_OP,     // line type .OP ili result opr ...
    LT_4P,     // 32-bit specific .OP
    LT_8P,     // 64-bit specific .OP
    LT_CL,     // C language specific ILM
    LT_FL,     // FORTRAN language specific ILM
    LT_AL,     // Ada language specific ILM
    LT_OL,     // OpenCL language specific ILM
    LT_CP,     // C for power ILM
    LT_CH      // C and C++ for hammer (X86) ILM
  };

  struct ILI_data {
    ILI_data() : oprs(0)
    {}
    ILI_data(const std::string& name, int oprs) : name(name), oprs(oprs)
    {}
    std::string name;
    int oprs;
  };

  std::vector<ILI_data> ilis;
  std::vector<std::string> ilmaux;
  std::vector<int> ilmopnd;
  std::vector<int> ilmtp;
  size_t n_tmp;
  size_t temps;
  size_t ilmnum;

  struct ILMO {
    ILMO(const char* name, int size, int type, int cnvflg)
      : name(name), size(size), type(type), cnvflg(cnvflg)
    {}
    std::string name;
    int size;
    int type;
    int cnvflg;
  };
  std::vector<ILMO> ilmo;
  std::vector<ILMO>::const_iterator found_ilmo;

  NroffMap elt;
  NroffMap ili_elt;
  std::map<std::string, int> languages; // Languages selected with the -l option
  std::map<std::string, int> bitwidths; // Bit width selected with the -b option
  std::map<std::string, int> oprtypes;
  std::map<std::string, unsigned int> att_types;
  std::vector<std::string> ilmtypes;
  std::vector<std::string> names;
  std::vector<ILMINFO> ilms;

  std::string lang_macro;
  int lt_lang;
  std::string bits_macro;
  int lt_bits;
  bool skip_ilm;

  int xoprs;
  int ilicnt;
  int tpladr;
  int nflag;
  char type;
  int lastc;
  unsigned int oprflag;

  std::string IN1;
  std::string IN2;
  FILE* OUT1;
  FILE* OUT2;
  std::ofstream OUT3;
  std::string sphinx_filename;

public:

  ILMTPApp(const std::vector<std::string>& args)
  {
    elt["xxx"] = LT_UNK;
    elt[".IL"] = LT_IL;
    elt[".AT"] = LT_AT;
    elt[".OP"] = LT_OP;
    elt[".4P"] = LT_4P;
    elt[".8P"] = LT_8P;
    elt[".CL"] = LT_CL;
    elt[".FL"] = LT_FL;
    elt[".AL"] = LT_AL;
    elt[".OL"] = LT_OL;
    elt[".CP"] = LT_CP;
    elt[".CH"] = LT_CH;

    ili_elt[".IL"] = LT_IL;

    // Languages selected with the -l option.
    languages["scc"]   = LT_CL;
    languages["scftn"] = LT_FL;
    languages["scada"] = LT_AL;
    languages["pgocl"] = LT_OL;

    // Bit widths selected with the -b option.
    bitwidths["32"] = LT_4P;
    bitwidths["64"] = LT_8P;

    oprtypes["lnk"] = 0;
    oprtypes["sym"] = 1;
    oprtypes["stc"] = 2;
    oprtypes["n"]   = 3;

    // NOTE: "var" is set if "n" is an operand
    att_types["spec"]              = 0x80000000;
    att_types["trm"]               = 0x40000000;
    att_types["var"]               = 0x20000000;
    att_types["vec"]               = 0x10000000;
    att_types["dcmplx"]            = 0x08000000;
    att_types["i8"]                = 0x04000000;
    att_types["x87cmplx"]          = 0x02000000;
    att_types["noinlc"]            = 0x01000000;
    att_types["doubledoublecmplx"] = 0x00800000;
    att_types["float128cmplx"]     = 0x00400000;
    att_types["qcmplx"]            = 0x00200000;

    // first letter must be unique */
    ilmtypes.push_back("arth");
    ilmtypes.push_back("branch");
    ilmtypes.push_back("cons");
    ilmtypes.push_back("fstr");
    ilmtypes.push_back("intr");
    ilmtypes.push_back("load");
    ilmtypes.push_back("misc");
    ilmtypes.push_back("proc");
    ilmtypes.push_back("ref");
    ilmtypes.push_back("store");
    ilmtypes.push_back("trans");
    ilmtypes.push_back("SMP");

    ilmo.push_back(ILMO("drret", 5, ILMO_DRRET, 0));
    ilmo.push_back(ILMO("arret", 5, ILMO_ARRET, 0));
    ilmo.push_back(ILMO("spret", 5, ILMO_SPRET, 0));
    ilmo.push_back(ILMO("dpret", 5, ILMO_DPRET, 0));
    ilmo.push_back(ILMO("qpret", 5, ILMO_QPRET, 0));
    ilmo.push_back(ILMO("krret", 5, ILMO_KRRET, 0));
    ilmo.push_back(ILMO("drpos", 5, ILMO_DRPOS, 0));
    ilmo.push_back(ILMO("arpos", 5, ILMO_ARPOS, 0));
    ilmo.push_back(ILMO("sppos", 5, ILMO_SPPOS, 0));
    ilmo.push_back(ILMO("dppos", 5, ILMO_DPPOS, 0));
    ilmo.push_back(ILMO("null",  4, ILMO_NULL,  0));
    ilmo.push_back(ILMO("isp(",  4, ILMO_ISP,   0));
    ilmo.push_back(ILMO("idp(",  4, ILMO_IDP,   0));
    ilmo.push_back(ILMO("=xr'",  4, ILMO_XRSYM, 0));
    ilmo.push_back(ILMO("=xd'",  4, ILMO_XDSYM, 0));
    ilmo.push_back(ILMO("=e_'",  4, ILMO__ESYM, 0));
    ilmo.push_back(ILMO("=i'",   3, ILMO_ISYM,  0));
    ilmo.push_back(ILMO("=r'",   3, ILMO_RSYM,  0));
    ilmo.push_back(ILMO("=d'",   3, ILMO_DSYM,  0));
    ilmo.push_back(ILMO("=e'",   3, ILMO_ESYM,  0));
    ilmo.push_back(ILMO("=l'",   3, ILMO_LSYM,  0));
    ilmo.push_back(ILMO("=ll'",  4, ILMO_LLSYM, 0));
    ilmo.push_back(ILMO("dr(",   3, ILMO_DR,    0));
    ilmo.push_back(ILMO("ar(",   3, ILMO_AR,    0));
    ilmo.push_back(ILMO("sp(",   3, ILMO_SP,    0));
    ilmo.push_back(ILMO("dp(",   3, ILMO_DP,    0));
    ilmo.push_back(ILMO("scz",   3, ILMO_SCZ,   0));
    ilmo.push_back(ILMO("scf",   3, ILMO_SCF,   0));
    ilmo.push_back(ILMO("sz",    2, ILMO_SZ,    0));
    ilmo.push_back(ILMO("iv",    2, ILMO_IV,   -1));
    ilmo.push_back(ILMO("rp",    2, ILMO_RP,    0));
    ilmo.push_back(ILMO("ip",    2, ILMO_IP,    0));
    ilmo.push_back(ILMO("rr",    2, ILMO_RR,    0));
    ilmo.push_back(ILMO("ir",    2, ILMO_IR,    0));
    ilmo.push_back(ILMO("kr",    2, ILMO_KR,    0));
    ilmo.push_back(ILMO("eq",    2, ILMO_IV,    1));
    ilmo.push_back(ILMO("ne",    2, ILMO_IV,    2));
    ilmo.push_back(ILMO("lt",    2, ILMO_IV,    3));
    ilmo.push_back(ILMO("ge",    2, ILMO_IV,    4));
    ilmo.push_back(ILMO("le",    2, ILMO_IV,    5));
    ilmo.push_back(ILMO("gt",    2, ILMO_IV,    6));
    ilmo.push_back(ILMO("r",     1, ILMO_R,     0));
    ilmo.push_back(ILMO("t",     1, ILMO_T,     0));
    ilmo.push_back(ILMO("v",     1, ILMO_V,     0));
    ilmo.push_back(ILMO("p",     1, ILMO_P,     0));

    ilmopnd.push_back(0); // zero is reserved for the null operand

    n_tmp = 0;
    ilmnum = 0;
    ilms.push_back(ILMINFO());
    lt_lang = LT_EOF;
    lt_bits = LT_EOF;
    skip_ilm = false;

    /* First parse ilmtp specific command line arguments. */
    enum { LANG, BITS, INPUT, OUTPUT, NROFF, SPHINX } state = INPUT;
    int filename_argument = 0;
    for (std::vector<std::string>::const_iterator
           arg = args.begin() + 1, E = args.end(); arg != E; ++arg) {
      if (arg->substr(0,2) == "-I") {
        continue;
      } else if (*arg == "-l") {
        state = LANG;
      } else if (*arg == "-b") {
        state = BITS;
      } else if (*arg == "-o") {
        filename_argument = 0;
        state = OUTPUT;
      } else if (*arg == "-n") {
        state = NROFF;
      } else if (*arg == "-s") {
        state = SPHINX;
      } else {
        switch (state) {
        case LANG:
          {
            // Determine language specific macro LT_CL, LT_FL, or LT_AL
            auto it = languages.find(*arg);
            if (it == languages.end()) {
              printError(FATAL, "illegal language in command line argument -l");
            }
            lt_lang = it->second;
            if (const char *s = elt.stringOf(lt_lang)) {
              lang_macro = s + 1;
              ili_elt[s] = lt_lang;
            }
            if (lt_lang == LT_OL) {
              if (const char *s = elt.stringOf(LT_CL)) {
                ili_elt[s] = LT_CL;
              }
            }
            state = INPUT;
            break;
          }
        case BITS:
          {
            auto it = bitwidths.find(*arg);
            if (it == bitwidths.end()) {
              printError(FATAL, "illegal bit width in command line argument -b");
            }
            lt_bits = it->second;
            if (const char *s = elt.stringOf(lt_bits)) {
              bits_macro = s + 1;
              ili_elt[s] = lt_bits;
            }
            state = INPUT;
            break;
          }
        case INPUT:
          if (filename_argument == 0) {
            IN1 = *arg;
          } else if (filename_argument == 1) {
            IN2 = *arg;
          } else {
            printError(FATAL, "too many input files");
            // FIXME extra input file argument.
          }
          ++filename_argument;
          break;
        case OUTPUT:
          if (filename_argument == 0) {
            OUT1 = fopen(arg->c_str(), "w");
          } else if (filename_argument == 1) {
            OUT2 = fopen(arg->c_str(), "w");
          }
          ++filename_argument;
          break;
        case NROFF:
          OUT3.open(arg->c_str());
          state = OUTPUT;
          break;
        case SPHINX:
          sphinx_filename = *arg;
          state = OUTPUT;
          break;
        }
      }
    }
    if (lt_lang == LT_EOF) {
      printError(FATAL, "missing language in command line argument -l");
    }
    init_ili();
  }

  int run()
  {
    if (!sphinx_filename.empty()) {
      sphinx.setOutputFile(sphinx_filename);
    }
    bool output_enabled = true;
    ifs.open(IN1.c_str());
    if (!ifs) {
      printError(FATAL, "Can't open ", IN1.c_str());
    }
    /* initialize file OUT1 - this begins the initializations of the main ILM
       structure.  The first element of this structure array is wasted, i.e.
       ILM whose value is 0 is not defined
    */
    fputs("ILMINFO ilms[] = {\n", OUT1);
    fputs("   { \" \", 'z', 0, 0, 0, 0, 0 }", OUT1);

    /* begin main loop -- this loop will read the input file searching for
       lines beginning with ".IL".  When one is found, its .OP lines are processed
    */
    auto lt = getLineAndTokenize(elt, &OUT3);
    /* Put out language specific .xL macro definition */
    OUT3 << ".de " << lang_macro << "\n"
         << ".IL \\\\$1 \\\\$2 \\\\$3 \\\\$4 \\\\$5 \\\\$6 "
         << "\\\\$7 \\\\$8 \\\\$9\n"
         << "..\n";

    /* Redefine .4P or .8P as .OP if enabled with -b. */
    if (lt_bits != LT_EOF) {
      OUT3 << ".de " << bits_macro << "\n"
           << ".OP \\\\$1 \\\\$2 \\\\$3 \\\\$4 \\\\$5 \\\\$6 "
           << "\\\\$7 \\\\$8 \\\\$9\n"
           << "..\n";
    }

    // Print out a name for ILM opcode type
    fprintf(OUT2, "#ifndef ILMTP_H_\n#define ILMTP_H_\n"
                  "typedef enum ILM_OP {\n");

    while (lt != LT_EOF) {
      if (lt == LT_AT || lt == LT_OP || lt == LT_4P || lt == LT_8P)
        printError(FATAL, "input sequence error");
      if (do_lang_ilm(lt, lt_lang)) {
        if (skip_ilm)
          output_enabled = true;
        skip_ilm = false;
        // Make nroff output file generic rather than lang specific
        line = std::string(".IL") + line.substr(3);
      } else {
        if (!skip_ilm)
          output_enabled = false;
        skip_ilm = true;
      }
      if (output_enabled) {
        OUT3 << line << '\n';
        fputs(",\n", OUT1); /* terminiate previous line */
      }
      auto tok = getToken();
      if (!skip_ilm) {
        ilmnum++; /* get new ilm name */
        /* Remember ILM name */
        names.push_back(tok);
      }
      if (output_enabled) {
      fprintf(OUT1, "   {\"%s\", ", tok.c_str()); /* name of ilm */

      /* output ilm number */

      fprintf(OUT2, "  IM_%s", tok.c_str());
      fprintf(OUT2, " = %ld,\n", ilmnum);
      }
      /*    get the type of the ilm   */

      tok = getToken();
      bool is_valid_type = false;
      for (std::vector<std::string>::const_iterator
             it = ilmtypes.begin(), E = ilmtypes.end(); it != E; ++it) {
        if (tok.substr(0, 3) == it->substr(0, 3)) {
          is_valid_type = true;
          break;
        }
      }
      if (!is_valid_type) {
        printError(FATAL, "illegal ILM type - ilmtp");
      }
      type = tok[0];
      oprflag = 0; /* ilm expansion is table driven */
      xoprs = 0;
      nflag = 0;
      lastc = '.';
      while (1) {
        tok = getToken();
        if (tok.empty())
          break;
        bool is_valid_oprtype = false;
        int oprtype = 0;
        for (std::map<std::string, int>::const_iterator
               it = oprtypes.begin(), E = oprtypes.end(); it != E; ++it) {
          auto s = it->first;
          if (tok.substr(0, s.length()) == s) {
            is_valid_oprtype = true;
            oprtype = it->second;
            break;
          }
        }
        if (!is_valid_oprtype) {
          printError(FATAL, "illegal ilm operand - ilmtp");
        }
        lastc = tok[tok.length() - 1];
        if (oprtype == OPR_N && xoprs == 0)
          nflag = 1;
        oprflag = oprflag | (oprtype << xoprs * 2);
        if (lastc != '*' && lastc != '+')
          xoprs++;
      }
      if (((lastc == '*' || lastc == '+') ^ nflag) == 1)
        printError(INFO, "inconsistent use of n and <opr>* (<opr>+)");

      //    scan .AT (attribute) line	if one exists

      oprflag |= ((unsigned int)nflag) << 29;
      if ((lt = getLineAndTokenize(elt, output_enabled ? &OUT3 : nullptr)) == LT_AT) {
        if (output_enabled)
          OUT3 << line << '\n';
        while (1) {
          tok = getToken();
          if (tok.empty())
            break;
          bool is_valid_att_type = false;
          int att_type = 0;
          for (std::map<std::string, unsigned int>::const_iterator
                 it = att_types.begin(), E = att_types.end(); it != E; ++it) {
            auto s = it->first;
            if (tok.substr(0, s.length()) == s) {
              is_valid_att_type = true;
              att_type = it->second;
              break;
            }
          }
          if (!is_valid_att_type) {
            printError(FATAL, "illegal ilm attribute - ilmtp");
          }
          if (oprflag & att_type) {
            printError(SEVERE, "attribute bit already set");
          }
          oprflag |= att_type;
        }
        lt = getLineAndTokenize(elt, output_enabled ? &OUT3 : nullptr);
      }
      if (output_enabled)
        fprintf(OUT1, "'%c', %d, 0x%lx, ", type, xoprs, (unsigned long)oprflag);

      // scan lines defining the ILI expansion (beginning with .OP, .4P, .8P)

      temps = 0;
      ilicnt = 0;
      tpladr = ilmtp.size();
      while (lt == LT_OP || lt == LT_4P || lt == LT_8P) {
        if (output_enabled)
          OUT3 << line << '\n';
        /* Skip .4P or .8P lines if they're not for the right bitwidth. */
        if (!skip_ilm && (lt == LT_OP || lt == lt_bits)) {
          ilicnt++;
          ili_op(); /* go process the line */
        }
        lt = getLineAndTokenize(elt, output_enabled ? &OUT3 : nullptr);
      };

      // output template address and the number of ili in the expansion
      if (!skip_ilm) {
        if (ilicnt == 0)
          tpladr = 0;
        if (output_enabled)
          fprintf(OUT1, "%ld, %d, %d }", temps, tpladr, ilicnt);
        ilms.push_back(ILMINFO());
        ilms[ilmnum].pattern = tpladr;
        ilms[ilmnum].ilict = ilicnt;
        if (temps > n_tmp)
          n_tmp = temps;
      }
    }
    fputs("\n} ;\n", OUT1);

    /* the main structure is complete. now, output ilmtp (templates) */

    fputs("short ilmtp[] = {\n", OUT1);
    size_t i = 1;
    nflag = 0;
    while (1) {
      if ((ilicnt = ilms[i].ilict) != 0) {
        if (nflag)
          fputs(",\n", OUT1);
        tpladr = ilms[i].pattern;
        fprintf(OUT1, "   /* %4d */ ", tpladr);
        while (1) {
          int ilix = ilmtp[tpladr++];
          fprintf(OUT1, "%3d  /* %-7s*/, %d", ilix, ilis[ilix].name.c_str(),
                  ilmtp[tpladr++]);
          for (int n = ilis[ilix].oprs; n > 0; n--)
            fprintf(OUT1, ", %d", ilmtp[tpladr++]);
          if (--ilicnt <= 0)
            break;
          fputs(",\n              ", OUT1);
        }
        nflag = 1;
      }
      if (++i > ilmnum)
        break;
    }
    fputs("\n} ;\n", OUT1);

    /* output ilmopnd -- the operand table */

    fprintf(OUT1, "short ilmopnd[] = {\n%11s 0,\n", " ");
    i = 1; /* remember, the first operand(entry) is
            * wasted */
    while (1) {
      if (i % 5 == 0)
        fprintf(OUT1, "   /*%3ld */", i);
      else
        fprintf(OUT1, "%11s", " ");
      fprintf(OUT1, " %d, %d", ilmopnd[i], ilmopnd[i + 1]);
      i = i + 2;
      if (i >= ilmopnd.size())
        break;
      fputs(",\n", OUT1);
    };
    fputs("\n} ;\n", OUT1);

    /* output ilmaux, the strings of constants and external names" */

    fputs("const char *ilmaux[] = {\n", OUT1);
    i = 0;
    while (1) {
      if (i % 5 == 0)
        fprintf(OUT1, "   /*%3ld */", i);
      else
        fprintf(OUT1, "%11s", " ");
      fprintf(OUT1, " \"%s\"", ilmaux[i].c_str());
      if (++i >= ilmaux.size())
        break;
      fputs(",\n", OUT1);
    };
    fputs("\n} ;\n", OUT1);

    /* output the sizes of the tables (macros) */

    fprintf(OUT2, "  N_ILM\n");
    /* close off enum */
    fprintf(OUT2, "} ILM_OP;\n");

    fprintf(OUT2, "\n");

    /* Print out all the defines for ILM opcodes
     * Note: this is tricky, thanks to skip_ilm stuff above */
    for (i = 0; i < ilmnum; ++i) {
      if (!names[i].empty())
        fprintf(OUT2, "#define IM_%s IM_%s\n", names[i].c_str(), names[i].c_str());
    }
    fprintf(OUT2, "#define N_ILM N_ILM\n#endif // ILMTP_H_\n");

    fprintf(stderr, "\n#define N_ILM %ld\n#define N_ILMTP %ld\n", ilmnum + 1, ilmtp.size());
    fprintf(stderr, "#define N_ILMOPND %ld\n#define N_ILMAUX %ld\n", ilmopnd.size(), ilmaux.size());
    fprintf(stderr, "#define N_ILMTEMPS %ld\n", n_tmp + 1);

    return 0;
  }

private:

  bool do_lang_ilm(int lt, int lt_lang)
  {
    if (lt == LT_IL || lt == lt_lang || (lt_lang == LT_OL && lt == LT_CL))
      return true;
#ifdef TARGET_POWER
    if (lt_lang == LT_CL && lt == LT_CP)
      return true;
#endif
#ifdef TARGET_X86
    if (lt_lang == LT_CL && lt == LT_CH)
      return true;
#endif
    return false;
  }

  void init_ili()
  {
    ifs.open(IN2.c_str());
    if (!ifs) {
      printError(FATAL, "Can't open ", IN2.c_str());
    }
    ilis.push_back(ILI_data());
    for (auto lt = getLineAndTokenize(ili_elt); lt != LT_EOF;
         lt = getLineAndTokenize(ili_elt)) {
      auto name = getToken();
      int oprs = 0;
      for (; !getToken().empty(); ++oprs) {}
      ilis.push_back(ILI_data(name, oprs));
    }
  }

  int get_op(std::string& op)
  {
    for (std::vector<ILMO>::const_iterator it = ilmo.begin(), E = ilmo.end();
         it != E; ++it) {
      if (it->name == op.substr(0, it->size)) {
        found_ilmo = it;
        return it->type;
      }
    }
    printError(FATAL, "illegal ili operand - get_op");
    found_ilmo = ilmo.end();
    return 0;
  }

  int find_op(int opr[])
  {
    if (opr[0] == 0)
      return 0;
    int i;
    // remember, first word is wasted
    for (i = 1; i != (int) ilmopnd.size(); i = i + 2)
      if (opr[0] == ilmopnd[i] && opr[1] == ilmopnd[i + 1])
        return i;
    // not found in the table, add to the end
    i = ilmopnd.size();
    ilmopnd.push_back(opr[0]);
    ilmopnd.push_back(opr[1]);
    return i;
  }

  int ssym(int opr, const std::string& name)
  {
    int i, iliopr[2];
    for (i = 0; i < (int) ilmaux.size(); i++) {
      if (name == ilmaux[i]) {
        iliopr[0] = opr;
        iliopr[1] = i;
        return (find_op(iliopr));
      };
    };
    i = ilmopnd.size();
    ilmopnd.push_back(opr);
    ilmopnd.push_back(ilmaux.size());
    ilmaux.push_back(name);
    return i;
  }

  int get_ili(const std::string& ilin)
  {
    for (int i = ilis.size() - 1; i > 0; --i)
      if (ilin == ilis[i].name)
        return i;
    std::string msg("illegal ili, ");
    msg += ilin + ", in template";
    printError(INFO, msg.c_str());
    return 0;
  }

  void ili_op()
  {
    int iliopr[2];
    // get the ili from the input line
    auto ilix = get_ili(getToken());
    ilmtp.push_back(ilix);
    // get the result operand
    auto tok = getToken();
    if ((iliopr[0] = get_op(tok)) == ILMO_T) {
      iliopr[1] = atoi(tok.substr(1).c_str());
      if (iliopr[1] > (int) temps)
        temps = iliopr[1];
    } else {
      iliopr[1] = 1;
    }
    ilmtp.push_back(find_op(iliopr));

    // scan for the operands

    int noprs = 0;
    while (1)
    {
      tok = getToken();
      if (tok.empty())
        break;
      noprs++;
      switch (iliopr[0] = get_op(tok)) {
      case ILMO_DRRET:
      case ILMO_ARRET:
      case ILMO_SPRET:
      case ILMO_DPRET:
      case ILMO_QPRET:
      case ILMO_KRRET:
        iliopr[1] = 0;
        ilmtp.push_back(find_op(iliopr));
        break;
      case ILMO_NULL:
        ilmtp.push_back(0);
        break;
      case ILMO_ISYM:
      case ILMO_RSYM:
      case ILMO_DSYM:
      case ILMO_ESYM:
      case ILMO_LSYM:
        ilmtp.push_back(ssym(iliopr[0], tok.substr(3)));
        break;
      case ILMO_LLSYM:
        ilmtp.push_back(ssym(iliopr[0], tok.substr(4)));
        break;
      case ILMO_XRSYM:
      case ILMO_XDSYM:
      case ILMO__ESYM:
        ilmtp.push_back(ssym(iliopr[0], tok.substr(4)));
        break;
      case ILMO_P:
      case ILMO_V:
        iliopr[1] = atoi(tok.substr(1).c_str());
        ilmtp.push_back(find_op(iliopr));
        break;
      case ILMO_SZ:
        iliopr[1] = atoi(tok.substr(2).c_str());
        ilmtp.push_back(find_op(iliopr));
        break;
      case ILMO_T:
        iliopr[1] = atoi(tok.substr(1).c_str());
        ilmtp.push_back(find_op(iliopr));
        if (iliopr[1] > (int) temps)
          temps = iliopr[1];
        break;
      case ILMO_IV:
        if (found_ilmo->cnvflg == -1)
          iliopr[1] = atoi(tok.substr(2).c_str());
        else
          iliopr[1] = found_ilmo->cnvflg;
        ilmtp.push_back(find_op(iliopr));
        break;
      case ILMO_RP:
      case ILMO_IP:
        iliopr[1] = atoi(tok.substr(2).c_str());
        ilmtp.push_back(find_op(iliopr));
        break;
      case ILMO_DR:
      case ILMO_AR:
      case ILMO_SP:
      case ILMO_DP:
      case ILMO_ISP:
      case ILMO_IDP:
      case ILMO_SCZ:
      case ILMO_SCF:
        iliopr[1] = atoi(tok.substr(3).c_str());
        ilmtp.push_back(find_op(iliopr));
        break;
      case ILMO_DRPOS:
      case ILMO_ARPOS:
      case ILMO_SPPOS:
      case ILMO_DPPOS:
        {
          auto s = tok.substr(6);
          iliopr[1] = atoi(s.c_str());
          auto pos = s.find_first_of(',');
          if (pos == std::string::npos) {
            printError(FATAL, "illegal ili operand - ilm_op");
            break;
          }
          iliopr[1] |= (atoi(s.substr(pos + 1).c_str()) << 8);
          ilmtp.push_back(find_op(iliopr));
          break;
        }
      };
    };
    if (ilis[ilix].oprs != noprs && ilix != 0) {
      int ii;
      ii = ilis[ilix].oprs - noprs;
      std::ostringstream oss;
      oss << noprs << " oprs given instead of " << ilis[ilix].oprs
          << " for ili " << ilis[ilix].name;
      if (ii < 0) {
        while (++ii < 1) {
          ilmtp.pop_back();
        }
        oss << " - ignoring extras\n";
      } else {
        while (ii-- > 0) {
          ilmtp.push_back(0);
        }
        oss << " - filling with zeros\n";
      }
      printError(INFO, oss.str().c_str());
    }
  }
};

int
main(int argc, char** argv)
{
  collectIncludePaths(argc, argv);
  ILMTPApp app(std::vector<std::string>(argv, argv + argc));
  return app.run();
}
