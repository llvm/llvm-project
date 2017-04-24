//===- COFF/ModuleDef.cpp -------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Windows-specific.
// A parser for the module-definition file (.def file).
// Parsed results are directly written to Config global variable.
//
// The format of module-definition files are described in this document:
// https://msdn.microsoft.com/en-us/library/28d6s79h.aspx
//
//===----------------------------------------------------------------------===//

#include "Config.h"
#include "Error.h"
#include "Memory.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/raw_ostream.h"
#include <system_error>

using namespace llvm;

namespace lld {
namespace coff {
namespace {

enum Kind {
  Unknown,
  Eof,
  Identifier,
  Comma,
  Equal,
  KwBase,
  KwConstant,
  KwData,
  KwExports,
  KwHeapsize,
  KwLibrary,
  KwName,
  KwNoname,
  KwPrivate,
  KwStacksize,
  KwVersion,
};

struct Token {
  explicit Token(Kind T = Unknown, StringRef S = "") : K(T), Value(S) {}
  Kind K;
  StringRef Value;
};

static bool isDecorated(StringRef Sym) {
  return Sym.startswith("_") || Sym.startswith("@") || Sym.startswith("?");
}

class Lexer {
public:
  explicit Lexer(StringRef S) : Buf(S) {}

  Token lex() {
    Buf = Buf.trim();
    if (Buf.empty())
      return Token(Eof);

    switch (Buf[0]) {
    case '\0':
      return Token(Eof);
    case ';': {
      size_t End = Buf.find('\n');
      Buf = (End == Buf.npos) ? "" : Buf.drop_front(End);
      return lex();
    }
    case '=':
      Buf = Buf.drop_front();
      return Token(Equal, "=");
    case ',':
      Buf = Buf.drop_front();
      return Token(Comma, ",");
    case '"': {
      StringRef S;
      std::tie(S, Buf) = Buf.substr(1).split('"');
      return Token(Identifier, S);
    }
    default: {
      size_t End = Buf.find_first_of("=,\r\n \t\v");
      StringRef Word = Buf.substr(0, End);
      Kind K = llvm::StringSwitch<Kind>(Word)
                   .Case("BASE", KwBase)
                   .Case("CONSTANT", KwConstant)
                   .Case("DATA", KwData)
                   .Case("EXPORTS", KwExports)
                   .Case("HEAPSIZE", KwHeapsize)
                   .Case("LIBRARY", KwLibrary)
                   .Case("NAME", KwName)
                   .Case("NONAME", KwNoname)
                   .Case("PRIVATE", KwPrivate)
                   .Case("STACKSIZE", KwStacksize)
                   .Case("VERSION", KwVersion)
                   .Default(Identifier);
      Buf = (End == Buf.npos) ? "" : Buf.drop_front(End);
      return Token(K, Word);
    }
    }
  }

private:
  StringRef Buf;
};

class Parser {
public:
  explicit Parser(StringRef S) : Lex(S) {}

  void parse() {
    do {
      parseOne();
    } while (Tok.K != Eof);
  }

private:
  void read() {
    if (Stack.empty()) {
      Tok = Lex.lex();
      return;
    }
    Tok = Stack.back();
    Stack.pop_back();
  }

  void readAsInt(uint64_t *I) {
    read();
    if (Tok.K != Identifier || Tok.Value.getAsInteger(10, *I))
      fatal("integer expected");
  }

  void expect(Kind Expected, StringRef Msg) {
    read();
    if (Tok.K != Expected)
      fatal(Msg);
  }

  void unget() { Stack.push_back(Tok); }

  void parseOne() {
    read();
    switch (Tok.K) {
    case Eof:
      return;
    case KwExports:
      for (;;) {
        read();
        if (Tok.K != Identifier) {
          unget();
          return;
        }
        parseExport();
      }
    case KwHeapsize:
      parseNumbers(&Config->HeapReserve, &Config->HeapCommit);
      return;
    case KwStacksize:
      parseNumbers(&Config->StackReserve, &Config->StackCommit);
      return;
    case KwLibrary:
    case KwName: {
      bool IsDll = Tok.K == KwLibrary; // Check before parseName.
      std::string Name;
      parseName(&Name, &Config->ImageBase);

      // Append the appropriate file extension if not already present.
      StringRef Ext = IsDll ? ".dll" : ".exe";
      if (!StringRef(Name).endswith_lower(Ext))
        Name += Ext;

      // Set the output file, but don't override /out if it was already passed.
      if (Config->OutputFile.empty())
        Config->OutputFile = Name;
      return;
    }
    case KwVersion:
      parseVersion(&Config->MajorImageVersion, &Config->MinorImageVersion);
      return;
    default:
      fatal("unknown directive: " + Tok.Value);
    }
  }

  void parseExport() {
    Export E;
    E.Name = Tok.Value;
    read();
    if (Tok.K == Equal) {
      read();
      if (Tok.K != Identifier)
        fatal("identifier expected, but got " + Tok.Value);
      E.ExtName = E.Name;
      E.Name = Tok.Value;
    } else {
      unget();
    }

    if (Config->Machine == I386) {
      if (!isDecorated(E.Name))
        E.Name = Saver.save("_" + E.Name);
      if (!E.ExtName.empty() && !isDecorated(E.ExtName))
        E.ExtName = Saver.save("_" + E.ExtName);
    }

    for (;;) {
      read();
      if (Tok.K == Identifier && Tok.Value[0] == '@') {
        Tok.Value.drop_front().getAsInteger(10, E.Ordinal);
        read();
        if (Tok.K == KwNoname) {
          E.Noname = true;
        } else {
          unget();
        }
        continue;
      }
      if (Tok.K == KwData) {
        E.Data = true;
        continue;
      }
      if (Tok.K == KwConstant) {
        warn("CONSTANT keyword is obsolete; use DATA");
        E.Constant = true;
        continue;
      }
      if (Tok.K == KwPrivate) {
        E.Private = true;
        continue;
      }
      unget();
      Config->Exports.push_back(E);
      return;
    }
  }

  // HEAPSIZE/STACKSIZE reserve[,commit]
  void parseNumbers(uint64_t *Reserve, uint64_t *Commit) {
    readAsInt(Reserve);
    read();
    if (Tok.K != Comma) {
      unget();
      Commit = nullptr;
      return;
    }
    readAsInt(Commit);
  }

  // NAME outputPath [BASE=address]
  void parseName(std::string *Out, uint64_t *Baseaddr) {
    read();
    if (Tok.K == Identifier) {
      *Out = Tok.Value;
    } else {
      *Out = "";
      unget();
      return;
    }
    read();
    if (Tok.K == KwBase) {
      expect(Equal, "'=' expected");
      readAsInt(Baseaddr);
    } else {
      unget();
      *Baseaddr = 0;
    }
  }

  // VERSION major[.minor]
  void parseVersion(uint32_t *Major, uint32_t *Minor) {
    read();
    if (Tok.K != Identifier)
      fatal("identifier expected, but got " + Tok.Value);
    StringRef V1, V2;
    std::tie(V1, V2) = Tok.Value.split('.');
    if (V1.getAsInteger(10, *Major))
      fatal("integer expected, but got " + Tok.Value);
    if (V2.empty())
      *Minor = 0;
    else if (V2.getAsInteger(10, *Minor))
      fatal("integer expected, but got " + Tok.Value);
  }

  Lexer Lex;
  Token Tok;
  std::vector<Token> Stack;
};

} // anonymous namespace

void parseModuleDefs(MemoryBufferRef MB) { Parser(MB.getBuffer()).parse(); }

} // namespace coff
} // namespace lld
