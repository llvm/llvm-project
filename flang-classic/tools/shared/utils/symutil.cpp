/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/*  SYMTAB utility program. */

/**
 * \file
 * \brief symutil.c - SYMTAB utility program
 */

#include "gbldefs.h"
#include "utils.h"
#include <algorithm>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
//#include <set>

// FIXME: merge symutil and symini into a single application.

class Symutil : public UtilityApplication
{
  // enumerate input line types
  enum {
    LT_OC = 1,
    LT_SF,
    LT_ST,
    LT_Sc,
    LT_SM,
    LT_SE,
    LT_SI,
    LT_FL,
    LT_TY,
    LT_DT,
    LT_DE,
    LT_PD,
    LT_TA,
    LT_Ik
  };

  NroffMap elt;
  std::vector<std::string> ocnames;
  std::vector<std::string> scnames;
  std::vector<std::string> iknames;

  struct Field {
    std::string name; // field name
    std::string type; // type name (optional)
    int size;         // size in bytes
    int offs;         // offset in bytes starting from 0 ltor, ttob
    bool shared;      // true if shared
    bool flag;        // true if a flag
    bool shareduse;   // true if used in all symtypes

    bool operator<(const Field &f) const
    {
      bool result = false;
      bool equal = false;
      if (flag && f.flag) {
        result = offs < f.offs;
        equal = offs == f.offs;
      } else if (flag) {
        result = true;
      } else if (f.flag) {
        result = false;
      } else {
        result = offs < f.offs;
        equal = offs == f.offs;
      }
      if (equal) {
        result = name < f.name;
      }
      return result;
    }
  };
  std::vector<Field> fields;

  struct Symbol {
    std::string stype;       // symbol type
    int oclass;              // overloading class
    std::string sname;       // name for this sym type
    std::vector<int> fields; // fields for this sym
  };
  std::vector<Symbol> symbols;

  struct Type {
    std::string name;
    std::string sname;
    std::vector<int> attribs;
  };
  std::vector<Type> types;

  struct PDType {
    PDType(int pdnum) : pdnum(pdnum)
    {
    }
    PDType(std::string &name, int pdnum) : name(name), pdnum(pdnum)
    {
    }
    std::string name;
    int pdnum;
  };
  std::vector<PDType> pd_dtypes;

  std::vector<std::string> attrnames;

#if defined(PGHPF)
  static const int SYMLEN = 41;
#else
  static const int SYMLEN = 33;
#endif

  // Generate run time checking code for symbol table field access macros.
  enum {
    OFF,
    KNOWN, // check access to symbols with known types only
    ALL    // check access to all symbols, including ST_UNKNOWN
  } checkmode;

  std::ostringstream tmpss;
  std::string symtab_n_filename;
  std::string symtab_in_h_filename;
  std::string sphinx_filename;

  std::ofstream out1; // symtab.out.n
  std::ofstream out2; // symtab.h
  std::ofstream out3; // symtabdf.h
  std::ofstream out4; // symtabdf.cpp
  std::ofstream out5; // symnames.h (optional)

public:
  Symutil(const std::vector<std::string> &args)
  {
    elt[".OC"] = LT_OC;
    elt[".SF"] = LT_SF;
    elt[".ST"] = LT_ST;
    elt[".Sc"] = LT_Sc;
    elt[".SM"] = LT_SM;
    elt[".SE"] = LT_SE;
    elt[".SI"] = LT_SI;
    elt[".FL"] = LT_FL;
    elt[".TY"] = LT_TY;
    elt[".DT"] = LT_DT;
    elt[".DE"] = LT_DE;
    elt[".PD"] = LT_PD;
    elt[".TA"] = LT_TA;
    elt[".Ik"] = LT_Ik;

    checkmode = OFF;

    enum { INPUT, OUTPUT, NROFF, SPHINX } state = INPUT;
    int filename_argument = 0;
    // FIXME: MSVC fails to infer type correctly when auto used in
    // this case.  Replace with 'auto' when moved to a newer MSVC.
    for (std::vector<std::string>::const_iterator arg = args.begin() + 1,
                                                  E = args.end();
         arg != E; ++arg) {
      if (*arg == "-check-known") {
        checkmode = KNOWN;
      } else if (*arg == "-check") {
        checkmode = ALL;
      } else if (*arg == "-o") {
        filename_argument = 0;
        state = OUTPUT;
      } else if (*arg == "-n") {
        if (state != OUTPUT) {
          usage("-n option must be used only after -o option");
        }
        state = NROFF;
      } else if (*arg == "-s") {
        state = SPHINX;
      } else {
        std::ofstream *os = nullptr;
        bool is_nroff = false;
        switch (state) {
        case INPUT:
          switch (filename_argument) {
          case 0:
            symtab_n_filename = *arg;
            break;
          case 1:
            symtab_in_h_filename = *arg;
            break;
          default:
            usage("too many input files");
          }
          break;
        case OUTPUT:
          switch (filename_argument) {
          case 0:
            os = &out1;
            break;
          case 1:
            os = &out2;
            break;
          case 2:
            os = &out3;
            break;
          case 3:
            os = &out4;
            break;
          case 4:
            os = &out5;
            break;
          default:
            usage("too many output files");
          }
          break;
        case NROFF:
          switch (filename_argument) {
          case 0:
            os = &out1;
            break;
          case 1:
            os = &out2;
            break;
          case 2:
            os = &out3;
            break;
          default:
            usage("too many output files");
          }
          is_nroff = true;
          state = OUTPUT;
          break;
        case SPHINX:
          sphinx_filename = *arg;
          break;
        }
        if (state == OUTPUT) {
          os->open(arg->c_str());
          if (!*os) {
            usage("can't create output file");
          }
          outputNotModifyComment(os, *arg, args[0], is_nroff);
        }
        if (state == SPHINX) {
          state = OUTPUT;
        } else {
          ++filename_argument;
        }
      }
    }
    if (symtab_n_filename == "") {
      usage("no symtab.n file is given");
    }
    if (symtab_in_h_filename == "") {
      usage("no symtab.in.h file is given");
    }
    if (!out1 || !out2 || !out3 || !out4) {
      usage("output file is missing");
    }
  }

  int
  run()
  {
    read_symtab_n();
    write_symtab();
    return 0;
  }

private:
  void
  usage(const char *error = 0)
  {
    printf("Usage: symutil [-check-known | -check] symtab.n symtab.in.h -o -n "
           "symtab.out.n symtab.h symtabdf.h symtabdf.cpp\n\n");
    printf("symtab.n    -- input file with symbol definitions\n");
    printf("symtab.in.h -- input header file with place holders for data read "
           "from symtab.n\n");
    printf("symtab.out  -- transformed symtab.n output\n");
    printf("symtab.h    -- generated symtab C header file\n");
    printf("symtabdf.h  -- generated C header file with supplemental "
           "declarations.\n");
    printf("symtabdf.cpp  -- generated CXX source file with supplemental "
           "definitions.\n\n");
    if (error) {
      fprintf(stderr, "Invalid command line: %s\n\n", error);
      exit(1);
    }
  }
 
  void flushline(std::ostream* outf)
  {
    *outf<<line<<"\n";
  }

  void
  read_symtab_n()
  {
    // FIXME: get rid of c_str() when C++11 STL is available
    ifs.open(symtab_n_filename.c_str());
    if (!ifs) {
      printError(FATAL, "Can't open ", symtab_n_filename.c_str());
    }
    if (!sphinx_filename.empty()) {
      sphinx.setOutputFile(sphinx_filename);
    }
    std::vector<std::string> dtfields;
    std::vector<int> cursyms;
    std::ostream *os = &out1;
    int pdoffs = 0;

    auto lt = getLineAndTokenize(elt, os);
    while (1) { // once for each line
      auto tok = getToken();
      switch (lt) {
      case LT_OC:
        if (tok.length() > 1) {
          ocnames.push_back(tok);
        }
        break;
      case LT_Sc:
        if (tok.length() > 1) {
          scnames.push_back(tok);
        }
        break;
      case LT_Ik:
        if (tok.length() > 1) {
          iknames.push_back(tok);
        }
        break;
      case LT_SF:
        if (tok.length() > 1) {
          (void)add_field(tok, true, false, true);
        }
        break;
      case LT_ST: // symbol type
        if (tok.length() > 1) {
          (void)findsym(tok);
        }
        break;
      case LT_SM:
        flushsym(cursyms); // flush info for current sym
        cursyms.clear();
        if (tok.length() == 1) {
          /* end of SM definitions */
          os = &out1;
          goto again;
        }
        while (!tok.empty()) {
          cursyms.push_back(findsym(tok));
          tok = getToken();
        }
        // redirect output up to the next SM
        tmpss.str("");
        tmpss.clear();
        os = &tmpss;
        flushline(&out1);
        goto again;

      case LT_SI:
        addoclass(cursyms, tok);
        tok = getToken();
        for (std::vector<int>::size_type idx = 0; !tok.empty(); ++idx) {
          addsname(cursyms, idx, tok);
          tok = getToken();
        }
        break;

      case LT_SE:
        add_field_to_symbol(cursyms, add_field(tok, false, false,
                                               checkmode && cursyms[0] == 0));
        break;

      case LT_FL:
        add_field_to_symbol(
            cursyms, add_field(tok, false, true, checkmode && cursyms[0] == 0));
        break;

      case LT_TY:
        if (tok.length() > 1) {
          types.push_back(Type());
          auto it = types.end() - 1;
          it->name = tok;
          it->sname = getToken();
          tok = getToken();
          while (!tok.empty()) {
            it->attribs.push_back(addattr(tok));
            tok = getToken();
          }
        }
        break;

      case LT_DT:
        // flush info for current dt
        flushdt(cursyms, dtfields);
        cursyms.clear();
        if (tok.length() == 1) {
          // end of DT definitions
          os = &out1;
          goto again;
        }
        while (!tok.empty()) {
          cursyms.push_back(finddt(tok));
          tok = getToken();
        }
        dtfields.clear();
        // redirect output up to next DT
        tmpss.str("");
        tmpss.clear();
        os = &tmpss;
        flushline(&out1);
        goto again;

      case LT_DE:
        dtfields.push_back(tok);
        break;

      case LT_PD:
        if (tok.length() == 1) {
          if (tok[0] == 'B') {
            pdoffs = 0;
            out3 << "int pd_dtype[DT_MAX+1] = {\n";
          } else if (tok[0] == 'E') {
            out3 << "};\n";
            pd_dtypes.push_back(PDType(pdoffs));
          }
          break;
        }
        pd_dtypes.push_back(PDType(tok, pdoffs));
        out3 << "    /* " << std::left << std::setw(13) << tok << "*/";
        getToken(); // skip sname
        tok = getToken();
        while (!tok.empty()) {
          out3 << ' ' << tok << ',';
          tok = getToken();
          ++pdoffs;
        }
        out3 << '\n';
        break;

      case LT_EOF:
        return;
      default:
        printError(FATAL, "Unknown LT: can't happen\n");
      }
      flushline(os);
    again:
      lt = getLineAndTokenize(elt, os);
    }
  }

  void
  flushdt(std::vector<int> &cursyms, std::vector<std::string> &dtfields)
  {
    if (cursyms.empty())
      return;
    for (std::vector<int>::const_iterator it = cursyms.begin(),
                                          E = cursyms.end();
         it != E; ++it) {
      out1 << ".TS\n"
           << "tab(%) allbox;\n";
      for (std::vector<std::string>::size_type k = 0; k <= dtfields.size();
           ++k) {
        out1 << "cw(0.8i) ";
      }
      out1 << ".\n" << types[*it].name;
      for (std::vector<std::string>::size_type k = 0; k != dtfields.size();
           ++k) {
        out1 << "%" << dtfields[k];
      }
      out1 << "\n.TE\n";
    }
    // append temp file contents to troff output
    out1 << tmpss.str();
  }

  int
  finddt(std::string &tok)
  {
    for (int i = 0; i != (int)types.size(); ++i)
      if (tok == types[i].name)
        return i;
    printError(WARN, "Undefined TY word");
    return 0;
  }

  int
  addattr(std::string &tok)
  {
    auto it = std::find(attrnames.begin(), attrnames.end(), tok);
    if (it != attrnames.end()) {
      return it - attrnames.begin();
    }
    attrnames.push_back(tok);
    return attrnames.size() - 1;
  }

  /**
    * Print the declaration of an enumeration, both as an enum
    * and as reflexive "#define"s.  The enum permits debuggers
    * to show values nicely; the latter permits the preprocessor
    * to query whether a value exists.
    *
    * \param enum_name: name of the enumeration and typedef.
    * \param name_of_max: "...MAX" name of maximum value, or empty string
    *                     to suppress printing it.
    * \param n: number of defined values.
    * \param name_of: functor mapping i to name of ith enumeration value.
    * \param value_of: functor mapping i to value of ith enumeration value.
    * \param allow_declare_as_int: if true, allow enumeration to be defined
    *     as int if DECLARE_enum_name_AS_INT is #defined.
    * \param strut: optional "id = value" to be injected into enumeration.
    *               Used to force enum size.
    * \param lwst   optional "id = value" to be injected into the enumeration.
    *               Used to force enum to be signed (for comparison, etc.)
    *
    * FIXME - it's a waste of space/compilation time for this method
    * to be a template.  When we have C++11 std::function, declare
    * name_of and value_of as std::function<std::string,size_t> would
    * be more frugal.
    */
  template <typename Fname, typename Fvalue>
  void
  write_enum_with_defines(const std::string &enum_name,
                          const std::string &name_of_max, size_t n,
                          Fname name_of, Fvalue value_of,
                          bool allow_declare_as_int = false,
                          const char *strut = nullptr,
                          const char *lwst = nullptr)
  {
    // Print the enumeration values
    if (allow_declare_as_int)
      out2 << "#ifndef DECLARE_" << enum_name << "_AS_INT\n\n";
    out2 << "typedef enum " << enum_name << " {\n";
    if (lwst)
      out2 << "    " << lwst << ",\n";
    for (size_t i = 0; i < n; ++i)
      out2 << "    " << name_of(i) << " = " << value_of(i)
           << (i < n - 1 || strut ? "," : "") << "\n";
    // Print the strut declaration if one was supplied
    if (strut)
      out2 << "    " << strut << "\n";
    out2 << "} " << enum_name << ";\n\n";
    // Print reflexive #define for each value
    for (size_t i = 0; i < n; ++i)
      out2 << "#define " << name_of(i) << " " << name_of(i) << "\n";
    out2 << "\n";
    if (allow_declare_as_int) {
      // Print deprecated alternative - enumeration type becomes an int,
      // and values are #define'd so that preprocessor can evaluate them.
      out2 << "#else\n\n"
              "/* DEPRECATED TEMPORARY WORKAROUND FOR BACKWARDS COMPATIBILITY "
              "*/\n"
              "typedef int "
           << enum_name << ";\n";
      for (size_t i = 0; i < n; ++i)
        out2 << "#define " << name_of(i) << " " << value_of(i) << "\n";
      out2 << "\n"
              "#endif\n\n";
    }
    if (!name_of_max.empty()) {
      // Print the maximum value
      out2 << "#define " << name_of_max << " " << value_of(n - 1) << "\n\n";
    }

    if (out5.is_open()) {
      // Print array mapping enum values to names.
      out5 << "const char *" << enum_name << "_names[] = {\n";
      size_t index = 0; // index in array
      for (size_t i = 0; i < n; ++i, ++index) {
        size_t value = value_of(i);
        for (; index < value; ++index) {
          out5 << "    /* " << index << " */  \"" << enum_name << ' ' << index
               << "\",\n";
        }
        out5 << "    /* " << value << " */  \"" << name_of(i) << "\",\n";
      }
      out5 << "};\n\n";
    }
  }

  void
  write_symtab()
  {
    // FIXME: get rid of c_str() when C++11 STL is available
    std::ifstream ifs(symtab_in_h_filename.c_str());
    if (!ifs) {
      printError(FATAL, "Can't open ", symtab_in_h_filename.c_str());
    }
    std::sort(fields.begin(), fields.end());
    // read the symtab.h boilerplate file and replace NROFF macros
    for (std::string line; std::getline(ifs, line); ) {
      auto lt = elt.match(line);
      if (!lt) {
        out2 << line << '\n'; // because getline strips EOL char.
        continue;
      }
      switch (lt) {
      case LT_OC: // print overloading classes
        write_enum_with_defines("OVCLASS", "OC_MAX", ocnames.size(),
                                [&](size_t i) { return ocnames[i]; },
                                [](size_t i) { return i + 1; });
        break;
      case LT_Sc: // print storage classes
        write_enum_with_defines("SC_KIND", "SC_MAX", scnames.size(),
                                [&](size_t i) { return scnames[i]; },
                                [](size_t i) { return i; });
        break;
      case LT_Ik: // print intrinsic kinds
        for (std::vector<std::string>::size_type i = 0; i != iknames.size();
             ++i)
          out2 << "#define " << iknames[i] << " " << i << "\n";
        out2 << "#define IK_MAX " << iknames.size() - 1 << "\n\n";
        break;
      case LT_ST: // print symbol types
        write_enum_with_defines("SYMTYPE", "ST_MAX", symbols.size(),
                                [&](size_t i) { return symbols[i].stype; },
                                [](size_t i) { return i; });
        break;
      case LT_TY: // print type names
        write_enum_with_defines("TY_KIND", "TY_MAX", types.size(),
                                [&](size_t i) { return types[i].name; },
                                [](size_t i) { return i; }, true);

        // pd_dtypes contains one extra element, which holds the DT_MAX+1 value.
        // We print that one specially after writing the regular enum values.
        write_enum_with_defines("DTYPE", "", pd_dtypes.size() - 1,
                                [&](size_t i) { return pd_dtypes[i].name; },
                                [&](size_t i) { return pd_dtypes[i].pdnum; },
                                true, "DT_MAXIMUM_POSSIBLE_INDEX = 0x7FFFFFFF",
                                "DT_MINUMUM_PSEUDO_VALUE = -32768");
        out2 << "#define DT_MAX " << pd_dtypes.back().pdnum - 1 << "\n";
        break;
      case LT_TA: // print type attributes
        for (std::vector<std::string>::size_type i = 0; i < attrnames.size();
             ++i)
          out2 << "#define _TY_" << attrnames[i] << " " << (1 << i) << "\n";
        out2 << "\n";
        out3 << "\nshort dttypes[TY_MAX+1] = {\n";
        for (std::vector<Type>::size_type i = 0; i != types.size(); ++i) {
          out3 << "    /* " << std::setw(13) << types[i].name << "*/ ";
          auto k = types[i].attribs.size();
          if (k == 0) {
            out3 << "0,\n";
          } else {
            out3 << "_TY_" << attrnames[types[i].attribs[0]];
            for (std::vector<int>::size_type j = 1; j < k; ++j)
              out3 << "|_TY_" << attrnames[types[i].attribs[j]];
            out3 << ",\n";
          }
        }
        out3 << "};\n";
        break;
      case LT_SE: // print fields access macros
        write_symtab_check_funcs();
        for (std::vector<Field>::size_type i = 0; i != fields.size(); ++i) {
          auto field = fields[i];
          auto name = field.name;
          if (field.flag) {
            write_symtab_accessors(field, i, "f", field.offs);
          } else if (field.shared) {
            if ("flags" == name || "flags2" == name ||
                "flags3" == name || "flags4" == name)
              continue;
            write_symtab_accessors(field, i, makeLower(name), 0);
          } else {
            std::string name;
            int offset;
            switch (field.size) {
            case 1:
              name = "b";
              offset = field.offs + 1;
              break;
            case 2:
              name = "hw";
              offset = field.offs / 2 + 1;
              break;
            case 4:
              name = "w";
              offset = field.offs / 4 + 1;
              break;
            default:
              // FIXME check this error when fields is filled.
              printError(WARN, "Field not b,h,w in macro");
              offset = 0;
              break;
            }
            write_symtab_accessors(field, i, name, offset);
          }
        }
        if (checkmode) {
          out2 << "extern char sym_type_check[ST_MAX+1][" << fields.size()
               << "];\n";
        }
        break;
      default:
        printError(WARN, "Unknown line type");
        break;
      }
    }

    // write dinit for STB
    out3 << "\nextern STB stb;\n";  // declaration into .h file
    out4 << "\n#include \"symacc.h\"\n"; // forward declaration of the STB type
    out4 << "\n#pragma clang diagnostic ignored "; // work around Clang warning
    out4 << "\"-Wmissing-field-initializers\"\n";  // (cont'd)
    out4 << "\nSTB stb = {\n    {"; // definition  into .c file
    int j = 6, k;
    for (std::vector<Symbol>::size_type i = 0; i != symbols.size(); ++i) {
      if ((j += (k = symbols[i].sname.length() + 3)) > 80) {
        out4 << "\n     ";
        j = 6 + k;
      }
      out4 << "\"" << symbols[i].sname << "\",";
    }
    out4 << "},\n";

    j = 6;
    out4 << "    {";
    for (std::vector<Symbol>::size_type i = 0; i != symbols.size(); ++i) {
      if ((j += (k = ocnames[symbols[i].oclass - 1].length() + 1)) > 80) {
        out4 << "\n     ";
        j = 6 + k;
      }
      out4 << ocnames[symbols[i].oclass - 1] << ",";
    }
    out4 << "},\n";

    out4 << "    {\"???\",";
    for (std::vector<std::string>::size_type i = 0; i != ocnames.size(); ++i)
      out4 << "\"" << ocnames[i] << "\",";
    out4 << "},\n";

    out4 << "    {";
    for (std::vector<std::string>::size_type i = 0; i != scnames.size(); ++i)
      out4 << "\"" << scnames[i] << "\",";
    out4 << "},\n";

    out4 << "    {";
    for (std::vector<Type>::size_type i = 0; i != types.size(); ++i) {
      if ((j += (k = types[i].sname.length() + 3)) > 80) {
        out4 << "\n     ";
        j = 6 + k;
      }
      out4 << "\"" << types[i].sname << "\",";
    }
    out4 << "},\n};\n";

    if (checkmode) {
      int x;
      out3 << "\n\nchar sym_type_check[ST_MAX+1][" << fields.size()
           << "] = {\n";
      for (std::vector<Symbol>::size_type s = 0; s != symbols.size(); ++s) {
        auto ch = ' ';
        for (std::vector<Field>::size_type i = 0; i != fields.size(); ++i) {
          x = 1;
          /* In KNOWN, allow all accesses to s==0 (ST_UNKNOWN). */
          if (fields[i].shareduse || (checkmode == KNOWN && s == 0)) {
            x = 0;
          } else {
            for (std::vector<int>::size_type sf = 0;
                 sf != symbols[s].fields.size(); ++sf) {
              if (symbols[s].fields[sf] == (int)i) {
                x = 0;
                break;
              }
            }
          }
          out3 << ch << x;
          ch = ',';
        }
        if (s < symbols.size() - 1) {
          out3 << ", /* " << symbols[s].stype << " */\n";
        } else {
          out3 << "}; /* " << symbols[s].stype << " */\n";
        }
      }
      for (std::vector<Field>::size_type i = 0; i != fields.size(); ++i) {
        out3 << "/* field " << i << " = " << fields[i].name << "G */\n";
      }
    }
  }

  /*
    Generate get and put macros for field like these:
      #define GINTG(s)   ((SPTR)stb.stg_base[s].w9)
      #define GINTP(s,v) (stb.stg_base[s].w9 = check_SPTR(v))
    Omit '(SPTR)' and 'check_SPTR' if the field doesn't have a type mentioned.
    Replace stb.stg_base[s] by a call to check_sym_type() in checkmode.
  */
  void write_symtab_accessors(Field field, int index, std::string name,
                              int offset)
  {
    std::stringstream symref;
    if (!checkmode || field.shareduse) {
      symref << "stb.stg_base[s].";
    } else {
      symref << "check_sym_type(s, " << index << ", \"" << field.name
             << "\")->";
    }
    symref << name;
    if (offset > 0) {
      symref << offset;
    }

    out2 << "#define " << std::left << field.name << "G(s)   (";
    if (!field.type.empty()) {
      out2 << "(" << field.type << ")";
    }
    out2 << symref.str() << ")\n";
    out2 << "#define " << field.name << "P(s,v) (" << symref.str() << " = ";
    if (!field.type.empty()) {
      out2 << "check_" << field.type;
    }
    out2 << "(v))\n";
  }

  // Generate functions like "SPTR check_SPTR(SPTR)" for each field type.
  // Also declaration for check_sym_type().
  void
  write_symtab_check_funcs()
  {
    if (checkmode) {
      out2 << "BEGIN_DECL_WITH_C_LINKAGE\n";
      out2 << "SYM *check_sym_type(SPTR s, int index, const char *name);\n";
      out2 << "END_DECL_WITH_C_LINKAGE\n";
    }
  }

  void
  flushsym(std::vector<int> &cursyms)
  {
    int j, k;
    int offs;
    int output;
    int last;

    if (cursyms.empty())
      return;
    /* add shared fields not already present */
    for (std::vector<int>::const_iterator it = cursyms.begin(),
                                          E = cursyms.end();
         it != E; ++it) {
      for (std::vector<Field>::size_type j = 0; j != fields.size(); ++j) {
        if (!fields[j].shared && !fields[j].flag)
          continue;
        auto addit = true;
        for (std::vector<int>::size_type k = 0; k != symbols[*it].fields.size();
             ++k)
          if (chk_overlap(j, symbols[*it].fields[k], false)) {
            addit = false;
            break;
          }
        if (addit) {
          symbols[*it].fields.push_back(j);
        }
      }
      std::sort(symbols[*it].fields.begin(), symbols[*it].fields.end(),
                [this](int f1, int f2) { return fields[f1] < fields[f2]; });
    }

    /* write symbol picture */
    auto indir = cursyms[0];
    out1 << ".sz -4\n"
         << ".TS\n"
         << "tab(%);\n";
    j = 0;
    auto p = symbols[indir].fields.begin();
    k = (int)symbols[indir].fields.size();
    while (j < k && fields[*p].flag) {
      ++j;
      ++p;
    }
    offs = 0;
    last = 1;
    /* write the table format lines */
    for (int i = 0; i < SYMLEN; ++i) {
      out1 << "n cw(0.75i) sw(0.75i) sw(0.75i) sw(0.75i)\n";
      out1 << "n ";
      output = 0;
      last = 1;
      while (output < 4) {
        if (j < k && fields[*p].offs == offs) {
          switch (fields[*p].size) {
          case 1:
            out1 << "| cw(0.75i) ";
            break;
          case 2:
            out1 << "| cw(0.75i) sw(0.75i) ";
            break;
          case 3:
            out1 << "| cw(0.75i) sw(0.75i) sw(0.75i) ";
            break;
          case 4:
            out1 << "| cw(0.75i) sw(0.75i) sw(0.75i) sw(0.75i) ";
            break;
          default:
            printError(FATAL, "Bad size in field");
            return;
          }
          last = 1;
          offs += fields[*p].size;
          output += fields[*p].size;
          ++j;
          ++p;
        } else {
          if (last)
            out1 << "| ";
          out1 << "cw(0.75i) ";
          ++output;
          ++offs;
          last = 0;
        }
      }
      out1 << "|\n";
    }
    out1 << "n cw(0.75i) sw(0.75i) sw(0.75i) sw(0.75i) .\n";

    j = 0;
    p = symbols[indir].fields.begin();
    k = (int)symbols[indir].fields.size();
    while (j < k && fields[*p].flag) {
      ++j;
      ++p;
    }
    offs = 0;
    /* write the data lines */
    for (int i = 0; i < SYMLEN; ++i) {
      out1 << "%_\n" << i + 1;
      output = 0;
      while (output < 4) {
        if (j < k && fields[*p].offs == offs) {
          out1 << "%" << fields[*p].name;
          offs += fields[*p].size;
          output += fields[*p].size;
          ++j;
          ++p;
        } else {
          out1 << '%';
          ++output;
          ++offs;
        }
      }
      out1 << '\n';
    }
    out1 << "%_\n"
         << ".TE\n"
         << ".sz +4\n"
         << tmpss.str(); // append temp file contents to troff output
  }

  void
  addoclass(std::vector<int> &cursyms, std::string &oc)
  {
    int ocnum1;

    auto it = std::find(ocnames.begin(), ocnames.end(), oc);
    if (it == ocnames.end()) {
      printError(WARN, "Unknown oclass name");
      ocnum1 = 0;
    } else {
      ocnum1 = it - ocnames.begin() + 1;
    }
    for (std::vector<int>::const_iterator it = cursyms.begin(),
                                          E = cursyms.end();
         it != E; ++it) {
      symbols[*it].oclass = ocnum1;
    }
  }

  void
  add_field_to_symbol(std::vector<int> &cursyms, int field)
  {
    for (std::vector<int>::const_iterator it = cursyms.begin(),
                                          E = cursyms.end();
         it != E; ++it) {
      auto found = false;
      for (std::vector<int>::const_iterator f = symbols[*it].fields.begin(),
                                            E = symbols[*it].fields.end();
           f != E; ++f) {
        if (field == *f) {
          printError(WARN, "Field already specified for this sym ",
                     fields[field].name.c_str());
          found = true;
          break;
        }
        if (chk_overlap(field, *f, true)) {
          found = true;
          break;
        }
      }
      if (!found) {
        symbols[*it].fields.push_back(field);
      }
    }
  }

  bool
  chk_overlap(int f1, int f2, bool flag)
  {
    int t1, t2, t3, t4;

    if (fields[f1].flag && fields[f2].flag) {
      if (fields[f1].offs == fields[f2].offs) {
        if (flag) {
          std::ostringstream oss;
          oss << "name " << fields[f1].name << " overlaps name "
              << fields[f2].name << " at offset " << fields[f1].offs << "\n";
          printError(WARN, "Flag overlaps one already defined",
                     oss.str().c_str());
        }
        return true;
      }
    } else if (fields[f1].flag || fields[f2].flag)
      return false;
    t1 = fields[f1].offs; // check for field overlap
    t2 = t1 + fields[f1].size - 1;
    t3 = fields[f2].offs;
    t4 = t3 + fields[f2].size - 1;
    if ((t1 >= t3 && t1 <= t4) || (t2 >= t3 && t2 <= t4) ||
        (t3 >= t1 && t3 <= t2) || (t4 >= t1 && t4 <= t2)) {
      if (flag) {
        std::ostringstream oss;
        oss << "name " << fields[f1].name << " overlaps name "
            << fields[f2].name << " at offset " << fields[f1].offs << "\n";
        printError(WARN, "Field overlaps one already defined",
                   oss.str().c_str());
      }
      return true;
    }
    return false;
  }

  void
  addsname(std::vector<int> &cursyms, std::vector<int>::size_type symidx,
           std::string &name)
  {
    if (symidx >= cursyms.size()) {
      printError(WARN, ".SI sname count doesn't match .SM line");
      return;
    }
    symbols[cursyms[symidx]].sname = name;
  }

  int
  findsym(std::string &tok)
  {
    for (int i = 0; i != (int)symbols.size(); ++i) {
      if (tok == symbols[i].stype) {
        return i;
      }
    }
    symbols.push_back(Symbol());
    auto it = symbols.end() - 1;
    it->stype = tok;
    return it - symbols.begin();
  }

  int
  add_field(std::string &ptok, bool sharedflag, bool flagflag, bool shareduse)
  {
    int size = 0, offs = 0;
    const char *aftp; // position after w<d>

    for (auto it = fields.begin(), E = fields.end(); it != E; ++it)
      if (ptok == it->name)
        return it - fields.begin();

    fields.push_back(Field());
    auto it = fields.end() - 1;
    it->name = ptok;
    auto tok = getToken();
    if (tok.empty()) {
      printError(WARN, "Field location not specified");
      goto fixup;
    }
    /* parse location */
    if (flagflag) {
      if (tok[0] != 'f')
        goto badloc;
      sscanf(tok.substr(1).c_str(), "%d", &offs);
      if (offs == 0)
        goto badloc;
      goto done;
    }
    if (tok[0] != 'w' || tok[1] < '1')
      goto badloc;
    offs = 0;
    sscanf(tok.substr(1).c_str(), "%d", &offs);
    if (offs == 0 || offs > SYMLEN)
      goto badloc;
    aftp = tok.c_str() + 2;
    if (offs > 9)
      aftp++;
    offs = (offs - 1) * 4;
    if (aftp[0] == 0)
      size = 4;
    else if (aftp[0] != ':') {
      printError(WARN, ": must follow word spec");
      size = 4;
    } else if (aftp[1] != 'h' && aftp[1] != 'b') {
      printError(WARN, "Bad subfield spec");
      size = 4;
    } else if (aftp[1] == 'h') {
      size = 2;
      if (aftp[2] < '1' || aftp[2] > '2')
        printError(WARN, "Bad halfword spec");
      else
        offs += (aftp[2] - '1') * 2;
    } else if (aftp[1] == 'b') {
      size = 1;
      if (aftp[2] < '1' || aftp[2] > '4')
        printError(WARN, "Bad byte spec");
      else
        offs += aftp[2] - '1';
      if (aftp[3] == '-') {
        if (fields[fields.size() - 1].name != "flags" || aftp[4] < '1' ||
            aftp[4] > '4')
          printError(WARN, "Bad flags spec");
        else
          size = (aftp[4] - aftp[2]) + 1;
      }
    }

    it->type = getToken(); // optional type: may be empty
  done:
    it->size = size;
    it->offs = offs;
    it->shared = sharedflag;
    it->shareduse = shareduse;
    it->flag = flagflag;
    return it - fields.begin();

  badloc:
    printError(WARN, "Bad field location");
  // intentional fall through
  fixup:
    it->size = fields[fields.size() - 1].offs = 0;
    it->shared = false;
    it->flag = false;
    it->shareduse = shareduse;
    return it - fields.begin();
  }
};

/**
   \brief main function of the application.

   Create an object of the application class Symutil passing the
   vector of command line arguments and run the application.
 */
int
main(int argc, char **argv)
{
  Symutil app(std::vector<std::string>(argv, argv + argc));
  return app.run();
}
