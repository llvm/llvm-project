/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
 *  \file
 *  \brief - machine characters utils definitions
 */

#include "utils.h"

/* the following files are opened:
 *    in1  -  input file to the utility (machar.n)
 *    in2  -  input file to the utility (symtab.h)
 *    out1 -  output file to contain initialization of dtype info
 *    out2 -  output file to contain C macro definitions for target machine
 *            characteristics.
 */

class Machar : public UtilityApplication
{

  enum {
    LT_DN = 1, //  line type .DN define action name
    LT_DA,     //  line type .DA action definition
    LT_TY,     //  line type %TY data type size/align def
    LT_TM,     //  line type %TM characteristic
    LT_NM,     //  line type .NM number of machines
    LT_MA      //  line type .MA machine number
  };

  NroffMap elt;
  std::map<int, std::string> dtypestr; /* Name of TY_xxx */

  struct DTypeInfo {
    DTypeInfo()
        : size(0), align(0), bits(0), scale(0), fval(-1)
#ifdef FE90
          ,
          target_kind(0)
#endif
    {
    }
    char size;   /* Size in bytes */
    char align;  /* Alignment in bytes */
    short bits;  /* Bits per data type */
    short scale; /* Scaling factor (power of 2) per data type */
    short fval;  /* How a function value is returned; possibly neg. */
#ifdef FE90
    short target_kind; /* target kind value for this type */
    std::string target_type;
/* target data type name for this type */
#endif
  };

  std::vector<std::vector<DTypeInfo>> dtypeinfo;
  int cm; /* current machine */
  int nm; /* number of machines */

  int ty_max;

  std::string inbuf; /* Will contain .DM or .DF lines */

  std::string machar_n_filename;
  std::string symtab_h_filename;
  std::string sphinx_filename;
  FILE *out1;
  FILE *out2;

public:
  Machar(std::vector<std::string> args)
  {
    elt["xxx"] = LT_UNK;
    elt[".DN"] = LT_DN;
    elt[".DA"] = LT_DA;
    elt["%TY"] = LT_TY;
    elt["%TM"] = LT_TM;
    elt[".NM"] = LT_NM;
    elt[".MA"] = LT_MA;

    nm = 1; // number of machines
    cm = 0; // current machine
    enum { INPUT, OUTPUT, SPHINX } state = INPUT;
    int filename_argument = 0;
    for (std::vector<std::string>::const_iterator arg = args.begin() + 1,
                                                  E = args.end();
         arg != E; ++arg) {
      if (*arg == "-o") {
        state = OUTPUT;
        filename_argument = 0;
      } else if (*arg == "-s") {
        state = SPHINX;
      } else {
        switch (state) {
        case INPUT: {
          if (filename_argument == 0) {
            machar_n_filename = *arg;
          } else {
            symtab_h_filename = *arg;
          }
          ++filename_argument;
          break;
        }
        case OUTPUT: {
          if (filename_argument == 0) {
            out1 = fopen(arg->c_str(), "w");
          } else {
            out2 = fopen(arg->c_str(), "w");
          }
          ++filename_argument;
          break;
        }
        case SPHINX: {
          sphinx_filename = *arg;
          state = OUTPUT;
          break;
        }
        }
      }
    }
  }

  int
  run()
  {
    get_types(); // load dtypeinfo with TY_xxx values
    ifs.open(machar_n_filename.c_str());
    if (!ifs) {
      printError(FATAL, "Can't open ", machar_n_filename.c_str());
    }
    if (!sphinx_filename.empty()) {
      sphinx.setOutputFile(sphinx_filename);
    }
    dtypeinfo.push_back(std::vector<DTypeInfo>(ty_max + 1));
    fputs("\n", out1);

    /* Main loop reads the input file searching for lines beginning with ".DM".
       When one is found, subsequent ".DF" lines are processed.  */

    int lt = LT_UNK;
    while (lt != LT_EOF) {
      switch (lt) {
      case LT_DN:
        lt = proc_dn();
        break;

      case LT_TM:
        proc_tm();
        lt = read_line();
        break;

      case LT_TY:
        proc_ty();
        lt = read_line();
        break;

      case LT_NM:
        proc_nm();
        lt = read_line();
        break;

      case LT_MA:
        proc_ma();
        lt = read_line();
        break;

      default:
        lt = read_line();
        break;
      }
    }

    write_output();

    return 0;
  }

private:
  //! True if text is whitespace over index range [i..j).
  bool
  is_space(const std::string &text, size_t i, size_t j)
  {
    return line.find_first_not_of(" ", i, j - i) >= j;
  }

  void
  get_types()
  {
    // File to be opened is a C header, not an Nroff file, but to minimize 
    // changes to legacy code, we use ifs anyway.
    ifs.open(symtab_h_filename.c_str());
    if (!ifs) {
      printError(FATAL, "Can't open ", symtab_h_filename.c_str());
    }

    const size_t npos = std::string::npos;
    this->ty_max = -1;
    while (getline(ifs, line)) {
      // Following code would be *much* simpler if we could use C++11
      // regular expression matching, or write this entire program in Python.

      // Check if line contains TY_... token.  The check that there
      // are no non-space *before* TY_ comes later.
      size_t i = line.find("TY_");
      if (i == npos)
        continue;
      size_t j =
          line.find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789", i);
      // [i,j) is the range of the first TY_... token

      // Check if line contains a decimal value after the TY_... token.
      size_t k = line.find_first_of("0123456789", j);
      if (k == npos)
        continue;
      size_t m = line.find_first_not_of("0123456789", k);
      // [k,m) is the range of the first decimal value

      // Now check whether line is "#define token value" OR "token = value[,]"
      if (line.substr(0, 7) == "#define") {
        if (!is_space(line, 7, i) || !is_space(line, j, k) ||
            !is_space(line, m, line.size()))
          continue;
      } else if (is_space(line, 0, i)) {
        size_t q = line.find("=");
        if (q == npos || !is_space(line, j, q) || !is_space(line, q + 1, k))
          continue;
        // Skip over optional comma
        size_t n = m;
        if (n < line.size() && line[n] == ',')
          ++n;
        // At this point, a mismatch is a sign that something is wrong.
        if (!is_space(line, n, line.size()))
          printError(SEVERE, "comma or end of line expected");
      } else {
        continue;
      }
      int val = atoi(line.substr(k, m - k).c_str());
      std::string s = line.substr(i, j - i);
      if (s == "TY_MAX") {
        ty_max = val;
        break;
      }
      // Each value can be named twice, once via a proper enumeration
      // and once as a "#define" for backwards compatibility.
      // That's okay, but if we have the same value with two names,
      // something is very wrong.
      if (dtypestr.count(val) != 0 && dtypestr[val] != s)
        printError(SEVERE, "duplicate TY_... value with different name!");

      dtypestr[val] = s;
    }
    ifs.close();
    if (ty_max < 0)
      printError(SEVERE, "TY_MAX not found!");
    for (int i = 0; i <= ty_max; ++i)
      if (!dtypestr.count(i))
        printError(SEVERE, "missing TY_... value");
  }

  // FIXME     printError(SEVERE, "Not enough arguments on %Tx line");

  /**
     Process .TY directive
  */
  void
  proc_ty()
  {
    tokenize_line("%");
    auto type = getToken();
    auto &dti = dtypeinfo[cm];
    for (int i = 0; i <= ty_max; ++i) {
      if (type == dtypestr[i]) {
        auto tok = getToken();
        dti[i].size = atoi(tok.c_str());
        tok = getToken();
        dti[i].align = atoi(tok.c_str());
        tok = getToken();
        dti[i].bits = atoi(tok.c_str());
        tok = getToken();
        dti[i].scale = atoi(tok.c_str());
        /* check for reg or mem -- this indicates how
           the value of a function of this type is returned. */
        auto reg = getToken();
        if (reg.substr(0, 3) == "reg") {
          dti[i].fval = 0;
          auto num = reg.substr(3);
          if (!num.empty())
            dti[i].fval |= (atoi(num.c_str())) << 2;
        } else if (reg.substr(0, 3) == "mem") {
          dti[i].fval = 1;
          auto num = reg.substr(3);
          if (!num.empty())
            dti[i].fval |= (atoi(num.c_str())) << 2;
        } else
          dti[i].fval = -1;
#ifdef FE90
        (void)getToken();
        dti[i].target_type = getToken();
        dti[i].target_kind = atoi(getToken().c_str());
#endif
        break;
      }
    }
  }

  void
  proc_tm()
  {
    tokenize_line("%");
    auto machine = getToken();
    (void)getToken();
    auto yes_or_no = getToken();
    if (yes_or_no == "yes") {
      fputs("#define ", out1);
      fputs(machine.c_str(), out1);
      fputs("\n", out1);
    } else if (yes_or_no != "no")
      printError(SEVERE, "Illegal yes/no response on %TM line");
  }

  int
  proc_dn()
  {
    tokenize_line(" ");
    (void)getToken();
    /* Process .DN directive */
    fputs("#define ", out1);
    fputs(getToken().c_str(), out1);

    /* Process .DA directive(s) */
    int lt;
    while ((lt = getLineAndTokenize(elt)) != LT_EOF && (lt == LT_DA)) {
      /* Continue previous line */
      fputs("  \\\n", out1);
      auto tok = getToken();
      if (tok.empty())
        printError(SEVERE, "Illegal .DA line");
      fputs("        ", out1);
      fputs(tok.c_str(), out1);
    }
    /* Terminate the last line output */
    fputs("\n", out1);

    return lt;
  }

  /**
     for .NM number, get the number
  */
  void
  proc_nm()
  {
    tokenize_line(" \t");
    (void)getToken();
    nm = atoi(getToken().c_str());
    for (int i = 1; i < nm; ++i) {
      dtypeinfo.push_back(std::vector<DTypeInfo>(ty_max + 1));
    }
  }

  void
  proc_ma()
  {
    tokenize_line(" \t");
    (void)getToken();
    cm = atoi(getToken().c_str());
  }

  int
  read_line()
  {
    int result = LT_EOF;
    while(getline(ifs, line)) {
      sphinx.process(line);
      if (auto it = elt.match(line)) {
        result = it;
        break;
      }
    }
    return result;
  }

  void
  tokenize_line(const std::string &separators)
  {
    tokens.clear();
    for (std::string::const_iterator s = line.begin(), B = line.begin(),
                                     E = line.end();
         s != E;) {
      for (; s != E && separators.find(*s) != std::string::npos; ++s) {
      }
      auto e = s != E ? s + 1 : E;
      for (; e != E && separators.find(*e) == std::string::npos; ++e) {
      }
      tokens.push_back(s == e ? std::string() : line.substr(s - B, e - s));
      s = e == E ? e : e + 1;
    }
    token = tokens.begin();
  }

  void
  write_output()
  {

    /* Output structure definition of dtype info (to machardf.h since
     * dtypeinfo is declared as static.
     */
    fputs("\n\ntypedef struct {\n", out2);
    fputs("    char          size;\n", out2);
    fputs("    char          align;\n", out2);
    fputs("    short         bits;\n", out2);
    fputs("    short         scale;\n", out2);
    fputs("    short         fval;\n", out2);
#ifdef FE90
    fputs("    short         target_kind;\n", out2);
    fputs("    const char   *target_type;\n", out2);
#endif
    fputs("} DTYPEINFO;\n", out2);

    if (nm == 1) {
      /* Output data type sizes, alignments, and bits to machardf.h */
      fputs("\n\nstatic DTYPEINFO dtypeinfo[] = {\n", out2);
      fputs("    /* size, alignment, bits, scale, fval for... */\n", out2);
      for (int i = 0; i <= ty_max; i++) {
#ifdef FE90
        fprintf(out2, "      { %2d, %2d, %3d, %2d, %2d, %2d, \"%s\" }",
                dtypeinfo[cm][i].size, dtypeinfo[cm][i].align,
                dtypeinfo[cm][i].bits, dtypeinfo[cm][i].scale,
                dtypeinfo[cm][i].fval, dtypeinfo[cm][i].target_kind,
                dtypeinfo[cm][i].target_type.c_str());
#else
        fprintf(out2, "      { %2d, %2d, %3d, %2d, %2d }",
                dtypeinfo[cm][i].size, dtypeinfo[cm][i].align,
                dtypeinfo[cm][i].bits, dtypeinfo[cm][i].scale,
                dtypeinfo[cm][i].fval);
#endif
        if (i < ty_max)
          fputs(",", out2);
        fprintf(out2, "    /* %s */\n", dtypestr[i].c_str());
      }
      fputs("};\n", out2);
    } else {
      for (cm = 0; cm < nm; ++cm) {
        fprintf(out2, "\n\nstatic DTYPEINFO dtypeinfo%d[] = {\n", cm);
        fputs("    /* size, alignment, bits, scale, fval for... */\n", out2);
        for (int i = 0; i <= ty_max; i++) {
#ifdef FE90
          fprintf(out2, "      { %2d, %2d, %3d, %2d, %2d, %2d, \"%s\" }",
                  dtypeinfo[cm][i].size, dtypeinfo[cm][i].align,
                  dtypeinfo[cm][i].bits, dtypeinfo[cm][i].scale,
                  dtypeinfo[cm][i].fval, dtypeinfo[cm][i].target_kind,
                  dtypeinfo[cm][i].target_type.c_str());
#else
          fprintf(out2, "      { %2d, %2d, %3d, %2d, %2d }",
                  dtypeinfo[cm][i].size, dtypeinfo[cm][i].align,
                  dtypeinfo[cm][i].bits, dtypeinfo[cm][i].scale,
                  dtypeinfo[cm][i].fval);
#endif
          if (i < ty_max)
            fputs(",", out2);
          fprintf(out2, "    /* %s */\n", dtypestr[i].c_str());
        }
        fputs("};\n", out2);
      }
      fputs("\n\nstatic DTYPEINFO* dtypeinfo = dtypeinfo0;\n", out2);
    }

#ifdef FE90
    fputs("\nvoid\n"
          "rehost_machar( int host )\n"
          "{\n",
          out2);
    if (nm > 1) {
      fputs("    switch( host ){\n", out2);
      for (cm = 0; cm < nm; ++cm) {
        fprintf(out2, "	case %d : dtypeinfo = dtypeinfo%d; break;\n", cm, cm);
      }
      fputs("    }\n", out2);
    }
    fputs("}/* rehost_machar */\n", out2);
#else
    if (nm > 1) {
      fputs("\nvoid\n"
            "rehost_machar( int host )\n"
            "{\n",
            out2);
      fputs("    switch( host ){\n", out2);
      for (cm = 0; cm < nm; ++cm) {
        fprintf(out2, "	case %d : dtypeinfo = dtypeinfo%d; break;\n", cm, cm);
      }
      fputs("    }\n", out2);
      fputs("}/* rehost_machar */\n", out2);
    }
#endif
  }
};

int
main(int argc, char **argv)
{
  Machar app(std::vector<std::string>(argv, argv + argc));
  return app.run();
}
