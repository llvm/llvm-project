/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
   \file
   \brief Various definitions for the utility programs, ilmpt, machar,
   symutil, and symini.

   If compiled as C++ code this file provides a definition of the base
   class for various utility programs that generate documentation and
   header files for compilers in the build process.

   Define the preprocessor symbol USE_OLD_C_UTILS *before* including
   this file if you need the old C interfaces.

   The NroffInStream, NroffMap, and NroffTokenStream interfaces are always
   provided.
 */

#ifndef _UTILS_UTILS_H
#define _UTILS_UTILS_H

#include "universal.h"

#if defined(__cplusplus)

#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <list>
#include <map>

void collectIncludePaths(int argc, char *argv[]);
extern std::list<std::string> includePaths;

/// types of errors printError can recognize.
enum ErrorSeverity { INFO = 1, WARN, SEVERE, FATAL };

/** \brief Similar to an std::ifstream, but intended for reading nroff files.
   Self-closing when end of input is reached.  Nroff include directives
   (.so "foo.n") are handled automatically. */
class NroffInStream
{
  /* Context used for processing an input file. */
  struct context {
    std::ifstream *file;  /**< file descriptor */
    std::string filename; /**< file name */
    int lineno;           /**< file line number */
    context(const char *filename_)
        : file(new std::ifstream(filename_)), filename(filename_), lineno(0)
    {
    }
  };

  //! Stack of open files.
  std::vector<context> stack;

  //! Top of stack
  context &
  tos()
  {
    return stack.back();
  }

  void
  push_file(const char *filename)
  {
    stack.push_back(context(filename));
  }

  void pop_file();

public:
  /** Open file.  Use operator!() to determine if file was successfully opened.
   */
  void
  open(const char *filename)
  {
    assert(stack.empty());
    push_file(filename);
  }

  /** Close all files.  No-op if file is not open. */
  void close();

  /**
     \brief print an error message with a line number if non-zero and
     the error severity level. Exit if the error is FATAL.

     \param sev error severity;
     \param txt error text to be written;
     \param additional text to be appended to the error message.
     \return this function doesn't return if the error severity is
     FATAL, instead it calls exit() to terminate the utility program.
  */
  void printError(ErrorSeverity sev, const char *txt1, const char *txt2 = "");

  /** True if error occurred for innermost file.  This method exists to simplify
      replacing uses of std::ifstream with NroffInStream. */
  bool operator!()
  {
    return tos().file->operator!();
  }

  friend bool getline(NroffInStream &f, std::string &s);

  ~NroffInStream()
  {
    close();
  }
};

//! Get line of input from the given stream.
bool getline(NroffInStream &f, std::string &s);

/** \brief Map from nroff commands to integers.
    This class is used in conjunction with class NroffTokenStream. */
class NroffMap
{
  /** Note: with C++11, a std::unordered_map could be used instead, but
      the time savings are probably neglible. */
  std::map<std::string, int> myMap;

public:
  /** Helper class that enables using subscript syntax to initialize an
     NroffMap.  Clients should not mention the name 'proxy'. */
  class proxy
  {
    NroffMap *map;
    const char *string;

    proxy(NroffMap *map_, const char *string_) : map(map_), string(string_)
    {
    }

    friend class NroffMap;

  public:
    //! See comments for NroffMap::operator[]
    void operator=(int code);
  };

  /** Return proxy that can be used to assign a code to an nroffPrimitive.
      an NroffMap. E.g.:

      NroffMap m;
      m[".xy"] = 42;

      The string must have three characters, and the code must be non-zero.
      These restrictions are checked by proxy::operator=.
      */
  proxy operator[](const char *nroffPrimitive)
  {
    return proxy(this, nroffPrimitive);
  }

  /** If current line begins with one of the strings in this map,
      return the corresponding code.  Otherwise return 0. */
  int match(const std::string &line) const;

  /** Given a code, find the corresponding string.
      Return nullptr code not found. */
  const char *stringOf(int code);
};

/** Class for reading nroff tokens from a file.
    Like NroffInStream, it is self-closing. */
class NroffTokenStream
{
  NroffInStream ifs;
  std::string buf;
  const char *pos; //!< pointer into buf

public:
  //! Create NroffTokenStream for a file.
  NroffTokenStream(const char *filename) : pos(nullptr)
  {
    ifs.open(filename);
  }
  /** Advance to next line that has a map code.  Return 0 if none found. */
  int get_line(const NroffMap &map);

  /** Get next token on curent line. */
  bool get_token(std::string &tok);

  void
  printError(ErrorSeverity sev, const char *txt1)
  {
    ifs.printError(sev, txt1);
  }
};

#endif /* defined(__cplusplus) */

#if defined(__cplusplus) && !defined(USE_OLD_C_UTILS)
#include <cstdio>
#include <algorithm>
#include <iomanip>
#include <iterator>
#include <sstream>

#define LT_UNK -1
#define LT_EOF 0

/**
   \class Manages conversion of NROFF input to Sphinx format.
 */
class SphinxConverter
{

  typedef std::vector<std::string> cell_type;
  typedef std::vector<cell_type> table_row_type;

  static const bool DEBUG = false;
  /// Sphinx convention specifies which character is used for underlining
  /// headers of various levels.  This array maps header levels to the specific
  /// character.
  char heading[6];
  std::stringstream source; ///< entire input file read into the stream buffer.
  std::stringstream target; ///< a buffer to hold the generated RST.
  std::ofstream ofs;        ///< Sphinx output file stream.
  std::vector<std::string> tokens;
  std::vector<std::string> listStack;
  bool op_started;
  int lineno;
  bool genRST; ///< If set, generate RST. Otherwise, just eat lines.
public:
  /**
     \brief Initializes the internal data structures.
   */
  SphinxConverter()
  {
    heading[0] = '#';
    heading[1] = '*';
    heading[2] = '=';
    heading[3] = '-';
    heading[4] = '^';
    heading[5] = '"';
    genRST = true;
  }

  /**
     \brief Write the last RST file.
   */
  ~SphinxConverter()
  {
    if (ofs.is_open()) {
      generateRST();
      ofs.close();
    }
  }

  /**
     \brief Create the Sphinx output file.
   */
  void
  setOutputFile(const std::string &s)
  {
    if (ofs.is_open()) {
      generateRST();
      ofs.close();
    }
    ofs.open(s.c_str());
    if (!ofs) {
      std::cerr << "Can't create file " << s << '\n';
      exit(1);
    }
    source.str("");
    source.clear();
    target.str("");
    target.clear();
    listStack.clear();
    op_started = false;
    lineno = 0;
  }

  /**
     \brief Collect a single line of an input file and store in a string stream
     for running the conversion process later on the entire source file.
   */
  void
  process(const std::string &s)
  {
    int startPos = 0;
    int len, dirLen;

    dirLen = chkCommentString(s);
    if (dirLen || !genRST) {
      len = s.length();
      if (dirLen == 2 && len >= 12 &&
          ((s.substr(2, 9)).compare("RST_ONLY ")) == 0) {
        /// found \#RST_ONLY directive
        startPos = 11;
      } else if (dirLen == 4 && len >= 13 &&
                 ((s.substr(4, 9)).compare("RST_ONLY ")) == 0) {
        /// found .\"#RST_ONLY directive
        startPos = 13;
      } else {
        return;
      }
    }
    auto r = (startPos == 0)
                 ? escapeInlineMarkup(cutTrailingSpace(s))
                 : escapeInlineMarkup(cutTrailingSpace(s.substr(startPos)));
    source << r // join lines if new line is escaped with backslash
           << (r.empty() || r[r.length() - 1] != '\\' ? '\n' : ' ');
  }

private:
  /**
    \brief Check to see if an input line is a special directive that begins
           with either a \# NROFF comment or an NROFF comment that begins with
           a .\" followed a # character.

     The \#NO_RST directive is used to turn off RST generation and the
     \#END_NO_RST directive is used to reenable RST generation.
     The .\"#NO_RST and .\"#END_NO_RST directives are also recognized.

     \param s is the string to process

     \returns 2 for \# style directive, 4 for .\"# style directive, 0 for no
      directive.
  */
  int
  chkCommentString(const std::string &s)
  {
    int len, pos, dirLen;
    bool toggleGenRST;

    len = s.length();

    if (len >= 2 && s[0] == '\\' && s[1] == '#') {
      dirLen = 2;
    } else if (len >= 3 && s[0] == '.' && s[1] == '\\' && s[2] == '\"') {
      dirLen = 4;
    } else {
      dirLen = 0;
    }

    if (dirLen > 0) {
      /// We have an NROFF \# or .\" comment
      if (dirLen == 4 && len >= 4 && s[3] == '#') {
        pos = 2;
      } else {
        pos = 0;
      }

      if (len >= 8) {
        /// Check for special \#NO_RST or \#END_NO_RST directive in comment
        /// or .\"#NO_RST or .\"#END_NO_RST directive in comment`
        if (len >= 12 && s[dirLen] == 'E' && s[dirLen + 1] == 'N' &&
            s[dirLen + 2] == 'D' && s[dirLen + 3] == '_') {
          pos += 6;
          toggleGenRST = true;
        } else {
          pos += 2;
          toggleGenRST = false;
        }
        std::string str2 = s.substr(pos, 6);
        if (str2.compare("NO_RST") == 0) {
          genRST = toggleGenRST;
        }
      }
    }
    return dirLen;
  }

  /**
     \brief Generate an output file in RST format.
   */
  void
  generateRST()
  {
    if (!ofs.is_open()) {
      return;
    }
    expandDefinedStrings();
    for (std::string s; std::getline(source, s); ++lineno) {
      parseOneLineRequest(s, source, target, true);
    }
    ofs << target.str();
  }

  /**
     \brief Convert a line in NROFF input to a Sphinx formatted line.
   */
  void
  parseOneLineRequest(const std::string &s, std::stringstream &src,
                      std::stringstream &tgt, bool newline = false)
  {
    if (DEBUG)
      tgt << "\n..\n  Line " << lineno << " <" << s << ">\n";
    if (s.empty()) {
      if (DEBUG)
        tgt << "\n..\n  EMPTY\n";
      return;
    }
    // comments
    if (s.substr(0, 2) == "\\\"" || s.substr(0, 3) == ".\\\"") {
      if (DEBUG)
        tgt << "\n..\n  COMMENT\n";
      return;
    }
    if (s.substr(0, 2) == "\\&") {
      tgt << s.substr(2) << '\n';
      return;
    }
    if (s[0] != '.') {
      if (op_started) {
        tgt << '\n';
        op_started = false;
      }
      tgt << s << '\n';
      return;
    }
    tokenize(s);
    if (tokens.empty()) {
      tgt << s;
      return;
    }
    auto tag = tokens[0];
    if (DEBUG)
      tgt << "\n..\n  Tag <" << tag << ">\n";
    if (tag == ".de") {
      for (std::string s; std::getline(src, s) && s.substr(0, 2) != "..";
           ++lineno) {
      }
      ++lineno;
      return;
    }
    if (tag != ".OP") {
      op_started = false;
    }
    if (tag == ".NS") {
      auto title = tokens.size() > 3 ? tokens[3] + tokens[2] : tokens[2];
      std::string lining(title.length(), '*');
      tgt << "\n\n" << lining << '\n' << title << '\n' << lining << '\n';
      return;
    }
    if (tag == ".sh") {
      auto level = atoi(tokens[1].c_str());
      std::string lining(tokens[2].length(), heading[level]);
      tgt << "\n\n" << lining << '\n' << tokens[2] << '\n' << lining << '\n';
      return;
    }
    if (tag == ".SM") {
      if (tokens[1] == "E") {
        return;
      }
      std::ostringstream ss;
      ss << tokens[1];
      for (std::vector<std::string>::const_iterator it = tokens.begin() + 2,
                                                    E = tokens.end();
           it != E; ++it) {
        ss << ", " << *it;
      }
      std::string lining(ss.str().length(), heading[4]);
      tgt << "\n\n" << lining << '\n' << ss.str() << '\n' << lining << '\n';
      return;
    }
    if (tag == ".OP") {
      if (!op_started) {
        op_started = true;
        tgt << "\n.. code-block:: none\n\n";
      }
      tgt << "  ";
      for (std::vector<std::string>::const_iterator it = tokens.begin() + 1,
                                                    E = tokens.end();
           it != E; ++it) {
        tgt << " " << *it;
      }
      tgt << "\n";
      return;
    }
    // a list begins
    if (tag == ".ba" || // augments the base indent by n.
        tag == ".ip" || tag == ".np" || tag == ".BL" || tag == ".CL" ||
        tag == ".CP" || tag == ".DE" || tag == ".FL" || tag == ".GN" ||
        tag == ".H1" || tag == ".H2" || tag == ".H3" || tag == ".H4" ||
        tag == ".H5" || tag == ".H6" || tag == ".H7" || tag == ".H8" ||
        tag == ".H9" || tag == ".Ik" || tag == ".IL" || tag == ".IN" ||
        tag == ".IP" || tag == ".OL" || tag == ".OC" || tag == ".OV" ||
        tag == ".PD" || tag == ".Sc" || tag == ".SE" || tag == ".SF" ||
        tag == ".ST" || tag == ".TY" || tag == ".XF") {
      src.seekg(-s.length() - 1, std::ios_base::cur);
      parseList(src, tgt);
      return;
    }
    // a table begins
    if (tag == ".TS") {
      parseTable(src, tgt);
      return;
    }
    // a code block begins
    if (tag == ".CS") {
      parseCodeBlock(src, tgt);
      return;
    }
    // a verbatim block begins
    if (tag == ".nf") {
      parseLineBlock(src, tgt);
      return;
    }
    if (tag == ".us" || tag == ".US") {
      if (tokens.size() > 1) {
        tgt << "\n*" << tokens[1];
        for (std::vector<std::string>::const_iterator it = tokens.begin() + 2,
                                                      E = tokens.end();
             it != E; ++it) {
          tgt << ' ' << *it;
        }
        tgt << "* --- ";
      }
      return;
    }
    if (tag == ".XB") {
      if (tokens.size() > 1) {
        tgt << "\n**" << tokens[1];
        for (std::vector<std::string>::const_iterator it = tokens.begin() + 2,
                                                      E = tokens.end();
             it != E; ++it) {
          tgt << ' ' << *it;
        }
        tgt << "**\n";
      }
      return;
    }
    if (tag == ".b") {
      generateHighlightedItem("**", tgt);
      return;
    }
    if (tag == ".cw" || tag == ".MA" || tag == ".NM") {
      generateHighlightedItem("``", tgt);
      return;
    }
    if (tag == ".i") {
      generateHighlightedItem("*", tgt);
      return;
    }
    if (tag == ".q") {
      generateHighlightedItem("\"", tgt);
      return;
    }
    if (tag == ".DN") {
      if (tokens.size() > 1) {
        tgt << "\n``" << tokens[1] << "``\n";
      }
      return;
    }
    if (tag == ".DA") {
      if (tokens.size() > 1) {
        tgt << "   ``" << tokens[1] << "``\n";
      }
      return;
    }
    if (tag == ".AT") {
      if (tokens.size() > 1) {
        tgt << "\n*Attributes*:";
        for (std::vector<std::string>::const_iterator it = tokens.begin() + 1,
                                                      E = tokens.end();
             it != E; ++it) {
          tgt << " " << *it;
        }
        tgt << "\n";
      }
      return;
    }
    if (tag == ".SI") {
      if (tokens.size() > 1) {
        tgt << "*" << tokens[1] << "*";
        for (std::vector<std::string>::const_iterator it = tokens.begin() + 2,
                                                      E = tokens.end();
             it != E; ++it) {
          tgt << " " << *it;
        }
        tgt << "\n\n";
      }
      return;
    }
    if (tag == ".ul") {
      std::string t;
      if (std::getline(src, t)) {
        tgt << '*' << t << "*\n";
      }
      return;
    }
    if (tag == ".lp" || tag == ".br") {
      tgt << '\n';
      return;
    }
    if (tag == ".(b" || tag == ".)b") {
      // FIXME: Couldn't find what .(b .)b pair stands for.
      //        It's not defined in groff_me macro package.
      // tgt << '\n';
      return;
    }
    if (tag == ".(z" || tag == ".)z") {
      // FIXME: .(z .)z means floating in groff_me.
      //        How to represent this in RST?
      return;
    }
    if (tag == ".bp" || // begin new page
        tag == ".ce" || // FIXME: center next n lines. Not needed.
        tag == ".EP" || //
        tag == ".ft" || // font
        tag == ".hl" || // FIXME: horizontal line. Not needed.
        tag == ".ne" || // FIXME: can't find what this request does
        tag == ".nr" || //
        tag == ".re" || // Reset tabs to default values.
        tag == ".sp" || //
        tag == ".sz" || // Augment the point size by n points.
        tag == ".ta") { //
      return;
    }
    tgt << s << (newline ? "\n" : "");
  }

  void
  parseList(std::stringstream &src, std::stringstream &tgt)
  {
    std::string s;
    if (!std::getline(src, s)) {
      return;
    }
    tokenize(s);
    auto tag = tokens[0];
    if (tag == ".FL" || tag == ".CL" || tag == ".OL") {
      listStack.push_back(".IL");
    } else {
      listStack.push_back(tag);
    }
    // format the list item header
    std::stringstream ss;
    if (DEBUG)
      ss << "\n..\n  Start list <" << s << ">\n";
    if (tag == ".BL") {
      ss << "*  ";
    } else if (tag == ".OV") {
      if (tokens.size() > 2) {
        ss << "``" << tokens[1] << " (" << tokens[2] << ")``\n";
      }
    } else if (tag == ".CL" || tag == ".FL" || tag == ".IL" || tag == ".OL" ||
               tag == ".CP") {
      if (tokens.size() < 4) {
        ss << "``" << tokens[1] << "``\n";
      } else if (tokens.size() > 3) {
        ss << "#. **" << tokens[1] << "**";
        for (std::vector<std::string>::size_type it = 3; it < tokens.size();
             ++it) {
          ss << " " << tokens[it];
        }
        ss << "    *Type*: *" << tokens[2] << "*\n\n";
      }
    } else if (tag == ".TY" || tag == ".PD") {
      if (tokens.size() > 1 && tokens[1] != "B" && tokens[1] != "E") {
        std::vector<std::string>::const_iterator it = tokens.begin() + 1,
                                                 E = tokens.end();
        ss << "``" << *it++ << "``";
        if (it != E) {
          ss << " ``" << *it++ << "``";
          for (; it != E; ++it) {
            ss << " " << *it;
          }
        }
      }
      ss << '\n';
    } else if (tag == ".IN" || tag == ".GN") {
      if (tokens.size() > 1) {
        for (std::vector<std::string>::const_iterator it = tokens.begin() + 1,
                                                      E = tokens.end();
             it != E; ++it) {
          ss << "``" << *it << "`` ";
        }
      }
    } else if (tag == ".ip") {
      if (tokens.size() > 1) {
        ss << tokens[1] << '\n';
      } else {
        ss << "*  ";
      }
    } else if (tag == ".np") {
      ss << "#. ";
    } else if (tag == ".ba") {
      ss << '\n';
    } else if (!(tokens.size() > 1 &&
                 (tag == ".Ik" || tag == ".OC" || tag == ".Ik" ||
                  tag == ".ST" || tag == ".SF" || tag == ".Sc") &&
                 (tokens[1] == "B" || tokens[1] == "E"))) {
      if (tokens.size() > 1) {
        ss << "``" << tokens[1] << "``\n";
      }
    }
    // consume the lines up to the next list item, possibly nested
    while (std::getline(src, s)) {
      if (s.empty()) {
        ss << '\n';
        continue;
      }
      tokenize(s);
      auto tag = tokens[0];
      if (tag == ".sh" || tag == ".lp" || tag == ".SM") {
        src.seekg(-s.length() - 1, std::ios_base::cur);
        generateListItem(ss, tgt);
        listStack.clear();
        return;
      } else if (tag == ".ip" || tag == ".np" || tag == ".BL" || tag == ".CL" ||
                 tag == ".CP" || tag == ".DE" || tag == ".FL" || tag == ".GN" ||
                 tag == ".H1" || tag == ".H2" || tag == ".H3" || tag == ".H4" ||
                 tag == ".H5" || tag == ".H6" || tag == ".H7" || tag == ".H8" ||
                 tag == ".H9" || tag == ".Ik" || tag == ".IL" || tag == ".IN" ||
                 tag == ".IP" || tag == ".OL" || tag == ".OC" || tag == ".OV" ||
                 tag == ".PD" || tag == ".Sc" || tag == ".SE" || tag == ".SF" ||
                 tag == ".ST" || tag == ".TY" || tag == ".XF") {
        if (tag == ".FL" || tag == ".CL" || tag == ".OL") {
          tag = ".IL";
        }
        auto it = listStack.begin();
        for (auto E = listStack.end(); it != E; ++it) {
          if (tag == *it) {
            listStack.erase(it, E);
            src.seekg(-s.length() - 1, std::ios_base::cur);
            generateListItem(ss, tgt);
            return;
          }
        }
        listStack.push_back(tag);
        ss << '\n';
        src.seekg(-s.length() - 1, std::ios_base::cur);
        parseList(src, ss);
        ss << '\n';
      } else if (tag == ".ba") {
        if (tokens.size() > 1) {
          if (tokens[1][0] == '+') {
            src.seekg(-s.length() - 1, std::ios_base::cur);
            parseList(src, ss);
          } else {
            generateListItem(ss, tgt);
            return;
          }
        }
        ss << '\n';
      } else {
        parseOneLineRequest(s, src, ss);
      }
    }
    generateListItem(ss, tgt);
  }

  void
  parseTable(std::stringstream &src, std::stringstream &tgt)
  {
    std::ostringstream oss;
    for (std::string s; std::getline(src, s);) {
      auto pos = s.find_first_of(" \t");
      auto tag = s.substr(0, pos);
      if (tag == ".TE") {
        if (DEBUG)
          tgt << "\n..\n  END OF TABLE\n";
        generateTable(oss.str(), tgt);
        return;
      } else {
        if (DEBUG)
          tgt << "\n..\n  STORE THE LINE FOR FUTURE\n";
        if (!s.empty()) {
          oss << s << '\n';
        }
      }
    }
  }

  void
  parseCodeBlock(std::stringstream &src, std::stringstream &tgt)
  {
    tgt << "\n.. code-block:: none\n\n";
    for (std::string s; getline(src, s);) {
      if (s == ".CE") {
        tgt << '\n';
        return;
      }
      if (s.empty()) {
        tgt << '\n';
      } else if (s.substr(0, 2) == "\\&") {
        tgt << "   " << s.substr(2) << '\n';
      } else {
        tgt << "   " << s << '\n';
      }
    }
  }

  void
  parseLineBlock(std::stringstream &src, std::stringstream &tgt)
  {
    tgt << "\n.. line-block::\n";
    for (std::string s; std::getline(src, s);) {
      auto pos = s.find_first_of(" \t");
      auto tag = s.substr(0, pos);
      if (tag == ".fi") {
        tgt << '\n';
        return;
      } else {
        if (!s.empty()) {
          tgt << "    " << s;
        }
        tgt << '\n';
      }
    }
  }

  /**
     \brief Create a new string without any trailing white space from
     the given input string.

     \param s the input string.
     \return a copy of \a s without any trailing space.
   */
  std::string
  cutTrailingSpace(const std::string &s) const
  {
    if (s.empty()) {
      return s;
    }
    auto pos = s.find_last_not_of(" \t");
    // s is not empty therefore it must contain only white space
    if (pos == std::string::npos) {
      return std::string();
    }
    // anything after pos, if exists, must be white space, so cut it.
    return s.substr(0, pos + 1);
  }

  std::string
  escapeInlineMarkup(const std::string &s) const
  {
    if (s.empty()) {
      return s;
    }
    std::string r;
    for (std::string::const_iterator c = s.begin(), E = s.end(); c != E; ++c) {
      if (*c == '*' || *c == '`' ||
          (*c == '_' &&
           (c + 1 == E || *(c + 1) == '\'' || *(c + 1) == '\t' ||
            *(c + 1) == ' ' || *(c + 1) == '"' || *(c + 1) == ',' ||
            *(c + 1) == '.' || *(c + 1) == ';' || *(c + 1) == ':' ||
            *(c + 1) == ')'))) {
        r.append(1, '\\');
      } else if (*c == '\\' && c + 1 != E && *(c + 1) == ' ') {
        continue;
      }
      r.append(1, *c);
    }
    return r;
  }

  /**
     \brief Find and convert strings defined with .ds requests.
   */
  void
  expandDefinedStrings()
  {
    std::ostringstream os;
    lineno = 1;
    for (std::string s; getline(source, s);) {
      s = expandStringInline(s, "\\(bu", "*");
      s = expandStringInline(s, "\\(em", "---");
      s = expandStringInline(s, "\\\\*(SC", "**Flang**");
      s = expandPairedString(s, "\\f(CW", "``", "\\fP", "``");
      s = expandPairedString(s, "\\\\*(cf", "``", "\\fP", "``");
      s = expandPairedString(s, "\\\\*(cf", "``", "\\\\*(rf", "``");
      s = expandPairedString(s, "\\\\*(cr", "``", "\\\\*(rf", "``");
      s = expandPairedString(s, "\\\\*(mf", "``", "\\\\*(rf", "``");
      s = expandPairedString(s, "\\\\*(ff", "*", "\\\\*(rf", "*");
      s = expandPairedString(s, "\\\\*(tf", "*", "\\\\*(rf", "*");
      s = expandPairedString(s, "\\fI", "*", "\\fP", "*");
      os << s << '\n';
    }
    if (DEBUG)
      target << "\n..\n  TOTAL LINES: " << lineno << '\n';
    source.str(os.str());
    source.clear();
    lineno = 1;
  }

  /**
     \brief Do simple one to one .ds string expansions for a single line.
   */
  std::string
  expandStringInline(const std::string &line, const std::string &s,
                     const std::string &r) const
  {
    if (line.empty()) {
      return line;
    }
    auto result = line;
    auto len = s.length();
    std::string::size_type pos = 0;
    while (1) {
      pos = result.find(s, pos);
      if (pos == std::string::npos) {
        break;
      }
      result.replace(pos, len, r);
    }
    return result;
  }

  /**
     \brief Do paired replacement of markup encoded with .ds strings.

     For example, fixed font or code is expressed as ``code`` constructs in RST,
     but Nroff explicitly sets and resets font for a substring of interest,
     e.g. \*(cfcode\*(rf, where cf and rf are names defined previous with .ds
     requests.
   */
  std::string
  expandPairedString(const std::string &line, const std::string &sl,
                     const std::string &rl, const std::string &sr,
                     const std::string &rr) const
  {
    if (line.empty()) {
      return line;
    }
    auto result = line;
    auto lenl = sl.length();
    auto lenr = sr.length();
    std::string::size_type posl = 0;
    while (1) {
      posl = result.find(sl, posl);
      if (posl == std::string::npos) {
        break;
      }
      auto posr = result.find(sr, posl);
      if (posr == std::string::npos) {
        break;
      }
      // order of the following two statements is important, DO NOT change.
      result.replace(posr, lenr, rr);
      result.replace(posl, lenl, rl); // posr is invalid after this replacement
    }
    return result;
  }

  void
  tokenize(const std::string &s,
           const std::string &separator = std::string(" \t"),
           const bool allow_empty = false)
  {
    tokens.clear();
    for (std::string::const_iterator head = s.begin(), B = s.begin(),
                                     E = s.end();
         head != E;) {
      for (; head != E && separator.find_first_of(*head) != std::string::npos;
           ++head) {
        if (allow_empty) {
          tokens.push_back(std::string());
        }
      }
      auto tail = head != E ? head + 1 : E;
      for (; tail != E; ++tail) {
        if (*head == '"') {
          if (*tail == '"')
            break;
        } else if (separator.find_first_of(*tail) != std::string::npos) {
          break;
        }
      }
      if (head != E && *head == '"')
        ++head;
      tokens.push_back(head == tail ? std::string()
                                    : s.substr(head - B, tail - head));
      head = tail == E ? tail : tail + 1;
      if (allow_empty && head == E && tail != E) {
        tokens.push_back(std::string());
      }
    }
  }

  void
  generateTable(const std::string &s, std::stringstream &tgt)
  {
    std::string token_separator = "\t";
    std::istringstream iss(s);
    std::vector<cell_type> formatting;
    std::vector<std::string::size_type> widths;
    std::vector<table_row_type> table;
    table_row_type::size_type columns;
    table_row_type table_row;
    cell_type cell;
    bool is_multiline = false;
    bool end_of_options = false;
    // read table formatting options
    for (std::string s; std::getline(iss, s);) {
      if (DEBUG)
        tgt << "\n..\n  OPTIONS Line <" << s << ">\n";
      tokenize(s);
      if (tokens.empty()) {
        continue;
      }
      auto &last_token = tokens.back();
      if (last_token.end()[-1] == ';') {
        for (std::vector<std::string>::const_iterator it = tokens.begin(),
                                                      E = tokens.end();
             it != E; ++it) {
          if (it->find_first_of('%') != std::string::npos) {
            token_separator = '%';
          }
        }
        continue;
      }
      // remove '|' tokens
      std::vector<std::string> refined_tokens;
      for (std::vector<std::string>::iterator it = tokens.begin();
           it != tokens.end(); ++it) {
        if (DEBUG)
          tgt << "\n..\n  TOKEN " << it - tokens.begin() << " <" << *it
              << ">\n";
        if (*it != "|" && *it != "." && *it != "|.") {
          refined_tokens.push_back(*it);
        } else if (*it == "." || *it == "|.") {
          end_of_options = true;
        }
      }
      formatting.push_back(refined_tokens);
      auto &last_refined_token = refined_tokens.back();
      if (end_of_options || last_refined_token.end()[-1] == '.') {
        if (DEBUG) {
          tgt << "\n..\n  FORMATTING SPEC:\n";
          for (std::vector<cell_type>::size_type it = 0; it < formatting.size();
               ++it) {
            tgt << "  " << it;
            for (cell_type::const_iterator ci = formatting[it].begin(),
                                           E = formatting[it].end();
                 ci != E; ++ci) {
              tgt << " " << *ci;
            }
            tgt << '\n';
          }
        }
        break;
      }
    }
    // read table contents
    for (std::string s; std::getline(iss, s);) {
      if (DEBUG)
        tgt << "\n..\n  TBLDATA Line <" << s << ">\n";
      if (s == ".TH") {
        continue; // skip the running header request
      }
      if (s == ".T&") {
        std::getline(iss, s);
        continue; // skip table continue command and the following format opts
      }
      tokenize(s, token_separator, true);
      auto it = tokens.begin();
      if (is_multiline) {
        if (tokens.empty() || tokens[0] != "T}") {
          cell.push_back(s);
          continue;
        }
        table_row.push_back(cell);
        is_multiline = false;
        cell.clear();
        ++it;
      }
      for (auto E = tokens.end(); it != E; ++it) {
        if (*it == "T{") {
          is_multiline = true;
        } else {
          cell.push_back(*it);
          table_row.push_back(cell);
          cell.clear();
        }
      }
      if (is_multiline) {
        continue;
      }
      table.push_back(table_row);
      table_row.clear();
    }
    // check options consistency
    columns = 0;
    if (!formatting.empty()) {
      for (std::vector<cell_type>::const_iterator it = formatting.begin(),
                                                  E = formatting.end();
           it != E; ++it) {
        if (it->size() > columns) {
          columns = it->size();
        }
      }
    }
    if (DEBUG)
      tgt << "\n..\n  #COLUMNS " << columns << '\n';
    // calculate each column width
    int row_counter = 0;
    bool do_corrections = false;
    for (std::vector<table_row_type>::const_iterator r = table.begin(),
                                                     rE = table.end();
         r != rE; ++r, ++row_counter) {
      if (DEBUG)
        tgt << "\n..\n  ROW " << std::setw(3) << row_counter;
      if (r->size() < columns) {
        do_corrections = true;
        if (DEBUG)
          tgt << '\n';
        continue;
      }
      std::string::size_type column = 0;
      for (table_row_type::const_iterator c = r->begin(), cE = r->end();
           c != cE; ++c, ++column) {
        std::string::size_type width = 0;
        for (cell_type::const_iterator s = c->begin(), sE = c->end(); s != sE;
             ++s) {
          if (s->length() > width) {
            width = s->length();
          }
        }
        if (DEBUG) {
          tgt << " COL " << column << " (" << std::setw(2) << width << ")";
        }
        if (column == widths.size()) {
          widths.push_back(width);
        } else if (widths[column] < width) {
          widths[column] = width;
        }
      }
      if (DEBUG)
        tgt << '\n';
    }
    if (DEBUG)
      tgt << "\n..\n  WIDTHS SIZE " << widths.size() << '\n';
    std::string::size_type table_width = 0;
    for (std::vector<std::string::size_type>::iterator it = widths.begin(),
                                                       E = widths.end();
         it != E; ++it) {
      if (*it > 0) {
        table_width += *it + 1;
      }
    }
    --table_width;
    if (do_corrections) {
      for (std::vector<table_row_type>::const_iterator r = table.begin(),
                                                       rE = table.end();
           r != rE; ++r, ++row_counter) {
        if (r->size() == columns) {
          continue;
        }
        std::string::size_type column = 0;
        std::string::size_type cumulative_width = 0;
        for (table_row_type::const_iterator c = r->begin(), cE = r->end();
             c != cE; ++c, ++column) {
          std::string::size_type width = 0;
          for (cell_type::const_iterator s = c->begin(), sE = c->end(); s != sE;
               ++s) {
            if (s->length() > width) {
              width = s->length();
            }
          }
          if (c + 1 != cE) {
            if (widths[column] < width) {
              table_width += width - widths[column];
              widths[column] = width;
            }
            cumulative_width += widths[column] + 1;
          } else if (cumulative_width + width > table_width) {
            table_width += width - widths.back();
            widths.back() = width;
          }
        }
      }
    }
    if (DEBUG) {
      tgt << "\n..\n  WIDTHS:";
      for (std::vector<std::string::size_type>::const_iterator
               it = widths.begin(),
               E = widths.end();
           it != E; ++it) {
        tgt << " " << *it;
      }
      tgt << '\n';
    }
    assert(!widths.empty());
    // fix up standalone cells with width 0
    for (std::vector<table_row_type>::iterator r = table.begin(),
                                               rE = table.end();
         r != rE; ++r) {
      std::vector<std::string::size_type>::const_iterator w = widths.begin();
      for (table_row_type::iterator c = r->begin(), cE = r->end(); c != cE;
           ++c, ++w) {
        if (*w == 0 && w + 1 != widths.end() && c + 1 == cE && !c->empty() &&
            c->at(0).length() > 1) {
          r->push_back(*c);
          break;
        }
      }
    }
    // output the formatted table
    char row_separator = '-';
    if (DEBUG)
      tgt << "\n..\n  TABLE WIDTH " << table_width << '\n';
    tgt << '\n';
    for (std::vector<table_row_type>::const_iterator r = table.begin(),
                                                     rE = table.end();
         r != rE; ++r) {
      if (r->size() == 1 && (r->at(0)[0] == "=" || r->at(0)[0] == "_")) {
        if (r->at(0)[0] == "=") {
          row_separator = '=';
        }
        continue;
      }
      for (std::vector<std::string::size_type>::const_iterator
               it = widths.begin(),
               E = widths.end();
           it != E; ++it) {
        if (*it > 0) {
          tgt << "+" << std::string(*it, row_separator);
        }
      }
      row_separator = '-'; // reset separator regardless of its value.
      tgt << "+\n";
      cell_type::size_type lines = 1;
      for (table_row_type::const_iterator c = r->begin(), cE = r->end();
           c != cE; ++c) {
        if (c->size() > lines) {
          lines = c->size();
        }
      }
      for (cell_type::size_type line = 0; line < lines; ++line) {
        std::vector<std::string::size_type>::const_iterator w = widths.begin();
        std::string::size_type width_so_far = 0;
        for (table_row_type::const_iterator c = r->begin(), cE = r->end();
             c != cE; ++c, ++w) {
          if (*w == 0) {
            continue;
          }
          tgt << '|';
          if (line < c->size()) {
            tgt << c->at(line);
            width_so_far += c->at(line).length();
            if (c + 1 != cE) {
              tgt << std::string(*w - c->at(line).length(), ' ');
              width_so_far += *w - c->at(line).length() + 1;
            }
          } else {
            tgt << std::string(*w, ' ');
            width_so_far += *w + 1;
          }
        }
        if (width_so_far < table_width) {
          if (width_so_far == 0) {
            --width_so_far;
            tgt << '|';
          }
          tgt << std::string(table_width - width_so_far, ' ');
        }
        tgt << "|\n";
      }
    }
    for (std::vector<std::string::size_type>::const_iterator
             it = widths.begin(),
             E = widths.end();
         it != E; ++it) {
      if (*it > 0) {
        tgt << "+" << std::string(*it, '-');
      }
    }
    tgt << "+\n\n";
  }

  void
  generateListItem(std::stringstream &src, std::stringstream &tgt)
  {
    src.seekg(0);
    std::string s;
    std::getline(src, s);
    tgt << '\n';
    if (!s.empty()) {
      tgt << s;
    }
    tgt << '\n';
    while (std::getline(src, s)) {
      if (!s.empty()) {
        tgt << "   " << s;
      }
      tgt << '\n';
    }
  }

  void
  generateHighlightedItem(const std::string &markup, std::stringstream &tgt)
  {
    if (tokens.size() > 1) {
      tgt << markup << tokens[1] << markup;
      for (std::vector<std::string>::const_iterator it = tokens.begin() + 2,
                                                    E = tokens.end();
           it != E; ++it) {
        tgt << *it;
        if (it + 1 != E) {
          tgt << ' ';
        }
      }
      tgt << '\n';
    }
  }
};

/**
   \class is the base class for the implementations of utility programs ilmtp,
   machar, symutil, and symini.  It provides the methods common to all utility
   derived classes, such as reading lines from an input file, splitting a line
   to tokens, identifying the type of an NROFF directive represented by a line,
   and reporting errors with corresponding line numbers in the error messages.
 */
class UtilityApplication
{

protected:
  SphinxConverter sphinx;          ///< sphinx output generator
  std::vector<std::string> tokens; ///< array of tokens in the last line
  std::vector<std::string>::const_iterator token; ///< tokens array iterator
  NroffInStream ifs;                              ///< input file stream
  std::string line; ///< the current line read from ifs

  void
  printError(ErrorSeverity sev, const char *txt1, const char *txt2 = "")
  {
    ifs.printError(sev, txt1, txt2);
  }

  /**
     \brief Create a string that is a result of converting every
     character of the given string to the lower case.
     \param s is the input string to be converted.
     \return a converted copy of the input string \a s.
   */
  std::string
  makeLower(const std::string &s)
  {
    std::string result;
    for (auto c = s.begin(), E = s.end(); c != E; ++c) {
      result.push_back(isupper(*c) ? tolower(*c) : *c);
    }
    return result;
  }

  /**
     \brief Read a line from the input stream and tokenize the string.
     \param elt map of line type strings to numeric type codes.
     \param os a pointer to an output stream.
     \return the type of line if found one of the lines with type in
             elt, otherwise LT_EOF when the entire input stream is
             read.
     If os is not nullptr, output every line that is discarded to the
     stream pointed by os.
  */
  int
  getLineAndTokenize(NroffMap &elt, std::ostream *os = nullptr)
  {
    tokens.clear();
    int result = LT_EOF;
    while (getline(ifs, line)) {
      sphinx.process(line);
      if (int item = elt.match(line)) {
        // collect tokens skipping the nroff macro at the line's head
        // which is the 3 first characters
        for (std::string::const_iterator s = line.begin() + 3, B = line.begin(),
                                         E = line.end();
             s != E;) {
          for (; *s == ' '; ++s) {
          } // skip over initial spaces
          // 'e' will point to one character after the end of a token
          auto e = s != E ? s + 1 : E; // next after s or end of line
          // move to next pairing \" or space if didn't start with \"
          for (; e != E && *e != (*s == '"' ? '"' : ' '); ++e) {
          }
          if (s != E && *s == '"') {
            ++s;          // exclude quotes
            if (e == E) { // validate quotes are paired
              printError(SEVERE, "double quote missing from end of string");
            }
          }
          tokens.push_back(s == e ? std::string() : line.substr(s - B, e - s));
          s = e == E ? e : e + 1;
        }
        result = item;
        break;
      }
      if (os) {
        *os << line << '\n';
      }
    }
    token = tokens.begin();
    return result;
  }

  /**
     \brief return the token currently pointed to by the token
     iterator and advance the iterator to the next token.  If no more
     tokens in the sequence, return an empty string.
   */
  std::string
  getToken()
  {
    return token != tokens.end() ? *token++ : std::string();
  }

  /**
     \brief output the comment asking not to modify the file produced
     by an utility application.
   */
  void
  outputNotModifyComment(FILE *out, const std::string &filename,
                         const std::string &progname, bool is_nroff)
  {
    auto pos = progname.find_last_of('/');
    std::string s =
        (pos == std::string::npos) ? progname : progname.substr(pos + 1);
    fputs(is_nroff ? ".\\\" " : "/* ", out);
    fprintf(out,
            "%s - This file written by utility program %s.  Do not modify.",
            filename.c_str(), s.c_str());
    fputs(is_nroff ? "\n" : " */\n", out);
  }

  /**
     \brief output the comment asking not to modify the file produced
     by an utility application.
   */
  void
  outputNotModifyComment(std::ostream *out, const std::string &filename,
                         const std::string &progname, bool is_nroff)
  {
    auto pos = progname.find_last_of('/');
    std::string s =
        (pos == std::string::npos) ? progname : progname.substr(pos + 1);
    if (is_nroff)
      *out << ".\\\" ";
    else
      *out << "/* ";
    *out << filename << " - This file written by utility program " << s
         << ".  Do not modify.";
    if (is_nroff)
      *out << "\n";
    else
      *out << " */\n";
  }
};

#else

BEGIN_DECL_WITH_C_LINKAGE

#define MAXLINE 300
#define FUNCTION
#define SUBROUTINE void
#define LOOP while (1)
#define INT int
#define UINT unsigned int
#define UCHAR char

#define LT_EOF 0
#define LT_UNK -1
#ifndef TRUE
#define TRUE 1
#define FALSE 0
#endif

typedef struct {
  const char *str;
  char ltnum;
} LT;

extern FILE *infile[10], *outfile[10];

#define IN1 infile[0]
#define IN2 infile[1]
#define IN3 infile[2]
#define IN4 infile[3]

#define OUT1 outfile[0]
#define OUT2 outfile[1]
#define OUT3 outfile[2]
#define OUT4 outfile[3]
#define OUT5 outfile[4]

#define put_error put_err

void put_error(int sev, const char *txt);
void open_files(int argc, char **argv);
int get_line(FILE *funit, LT ltypes[]);
void get_token(char out[], int *len);
int get_line1(FILE *funit, LT ltypes[], FILE *outf);
void flush_line(FILE *outf);
void output_off(void);
void output_on(void);
void ili_op(void);
void init_ili(LT *elt);
int get_ili(char *ilin);

END_DECL_WITH_C_LINKAGE

#endif

#endif // _UTILS_UTILS_H

/*
 Local Variables:
 mode: c++
 End:
*/
