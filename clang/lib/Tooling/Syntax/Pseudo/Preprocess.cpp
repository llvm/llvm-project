//===--- Preprocess.cpp - Preprocess token streams ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Syntax/Pseudo/Preprocess.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/TokenKinds.h"
#include "llvm/Support/FormatVariadic.h"

namespace clang {
namespace syntax {
namespace pseudo {
namespace {

class PPParser {
public:
  explicit PPParser(const TokenStream &Code) : Code(Code), Tok(&Code.front()) {}
  void parse(PPStructure *Result) { parse(Result, /*TopLevel=*/true); }

private:
  // Roles that a directive might take within a conditional block.
  enum class Cond { None, If, Else, End };
  static Cond classifyDirective(tok::PPKeywordKind K) {
    switch (K) {
    case clang::tok::pp_if:
    case clang::tok::pp_ifdef:
    case clang::tok::pp_ifndef:
      return Cond::If;
    case clang::tok::pp_elif:
    case clang::tok::pp_elifdef:
    case clang::tok::pp_elifndef:
    case clang::tok::pp_else:
      return Cond::Else;
    case clang::tok::pp_endif:
      return Cond::End;
    default:
      return Cond::None;
    }
  }

  // Parses tokens starting at Tok into PP.
  // If we reach an End or Else directive that ends PP, returns it.
  // If TopLevel is true, then we do not expect End and always return None.
  llvm::Optional<PPStructure::Directive> parse(PPStructure *PP, bool TopLevel) {
    auto StartsDirective =
        [&, AllowDirectiveAt((const Token *)nullptr)]() mutable {
          if (Tok->flag(LexFlags::StartsPPLine)) {
            // If we considered a comment at the start of a PP-line, it doesn't
            // start a directive but the directive can still start after it.
            if (Tok->Kind == tok::comment)
              AllowDirectiveAt = Tok + 1;
            return Tok->Kind == tok::hash;
          }
          return Tok->Kind == tok::hash && AllowDirectiveAt == Tok;
        };
    // Each iteration adds one chunk (or returns, if we see #endif).
    while (Tok->Kind != tok::eof) {
      // If there's no directive here, we have a code chunk.
      if (!StartsDirective()) {
        const Token *Start = Tok;
        do
          ++Tok;
        while (Tok->Kind != tok::eof && !StartsDirective());
        PP->Chunks.push_back(PPStructure::Code{
            Token::Range{Code.index(*Start), Code.index(*Tok)}});
        continue;
      }

      // We have some kind of directive.
      PPStructure::Directive Directive;
      parseDirective(&Directive);
      Cond Kind = classifyDirective(Directive.Kind);
      if (Kind == Cond::If) {
        // #if or similar, starting a nested conditional block.
        PPStructure::Conditional Conditional;
        Conditional.Branches.emplace_back();
        Conditional.Branches.back().first = std::move(Directive);
        parseConditional(&Conditional);
        PP->Chunks.push_back(std::move(Conditional));
      } else if ((Kind == Cond::Else || Kind == Cond::End) && !TopLevel) {
        // #endif or similar, ending this PPStructure scope.
        // (#endif is unexpected at the top level, treat as simple directive).
        return std::move(Directive);
      } else {
        // #define or similar, a simple directive at the current scope.
        PP->Chunks.push_back(std::move(Directive));
      }
    }
    return None;
  }

  // Parse the rest of a conditional section, after seeing the If directive.
  // Returns after consuming the End directive.
  void parseConditional(PPStructure::Conditional *C) {
    assert(C->Branches.size() == 1 &&
           C->Branches.front().second.Chunks.empty() &&
           "Should be ready to parse first branch body");
    while (Tok->Kind != tok::eof) {
      auto Terminator = parse(&C->Branches.back().second, /*TopLevel=*/false);
      if (!Terminator) {
        assert(Tok->Kind == tok::eof && "gave up parsing before eof?");
        C->End.Tokens = Token::Range::emptyAt(Code.index(*Tok));
        return;
      }
      if (classifyDirective(Terminator->Kind) == Cond::End) {
        C->End = std::move(*Terminator);
        return;
      }
      assert(classifyDirective(Terminator->Kind) == Cond::Else &&
             "ended branch unexpectedly");
      C->Branches.emplace_back();
      C->Branches.back().first = std::move(*Terminator);
    }
  }

  // Parse a directive. Tok is the hash.
  void parseDirective(PPStructure::Directive *D) {
    assert(Tok->Kind == tok::hash);

    // Directive spans from the hash until the end of line or file.
    const Token *Begin = Tok++;
    while (Tok->Kind != tok::eof && !Tok->flag(LexFlags::StartsPPLine))
      ++Tok;
    ArrayRef<Token> Tokens{Begin, Tok};
    D->Tokens = {Code.index(*Tokens.begin()), Code.index(*Tokens.end())};

    // Directive name is the first non-comment token after the hash.
    Tokens = Tokens.drop_front().drop_while(
        [](const Token &T) { return T.Kind == tok::comment; });
    if (!Tokens.empty())
      D->Kind = PPKeywords.get(Tokens.front().text()).getPPKeywordID();
  }

  const TokenStream &Code;
  const Token *Tok;
  clang::IdentifierTable PPKeywords;
};

} // namespace

PPStructure PPStructure::parse(const TokenStream &Code) {
  PPStructure Result;
  PPParser(Code).parse(&Result);
  return Result;
}

static void dump(llvm::raw_ostream &OS, const PPStructure &, unsigned Indent);
static void dump(llvm::raw_ostream &OS, const PPStructure::Directive &Directive,
                 unsigned Indent) {
  OS.indent(Indent) << llvm::formatv("#{0} ({1} tokens)\n",
                                     tok::getPPKeywordSpelling(Directive.Kind),
                                     Directive.Tokens.size());
}
static void dump(llvm::raw_ostream &OS, const PPStructure::Code &Code,
                 unsigned Indent) {
  OS.indent(Indent) << llvm::formatv("code ({0} tokens)\n", Code.Tokens.size());
}
static void dump(llvm::raw_ostream &OS,
                 const PPStructure::Conditional &Conditional, unsigned Indent) {
  for (const auto &Branch : Conditional.Branches) {
    dump(OS, Branch.first, Indent);
    dump(OS, Branch.second, Indent + 2);
  }
  dump(OS, Conditional.End, Indent);
}

static void dump(llvm::raw_ostream &OS, const PPStructure::Chunk &Chunk,
                 unsigned Indent) {
  switch (Chunk.kind()) {
  case PPStructure::Chunk::K_Empty:
    llvm_unreachable("invalid chunk");
  case PPStructure::Chunk::K_Code:
    return dump(OS, (const PPStructure::Code &)Chunk, Indent);
  case PPStructure::Chunk::K_Directive:
    return dump(OS, (const PPStructure::Directive &)Chunk, Indent);
  case PPStructure::Chunk::K_Conditional:
    return dump(OS, (const PPStructure::Conditional &)Chunk, Indent);
  }
}

static void dump(llvm::raw_ostream &OS, const PPStructure &PP,
                 unsigned Indent) {
  for (const auto &Chunk : PP.Chunks)
    dump(OS, Chunk, Indent);
}

// Define operator<< in terms of dump() functions above.
#define OSTREAM_DUMP(Type)                                                     \
  llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Type &T) {        \
    dump(OS, T, 0);                                                            \
    return OS;                                                                 \
  }
OSTREAM_DUMP(PPStructure)
OSTREAM_DUMP(PPStructure::Chunk)
OSTREAM_DUMP(PPStructure::Directive)
OSTREAM_DUMP(PPStructure::Conditional)
OSTREAM_DUMP(PPStructure::Code)
#undef OSTREAM_DUMP

} // namespace pseudo
} // namespace syntax
} // namespace clang
