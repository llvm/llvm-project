//===--- DirectiveMap.cpp - Find and strip preprocessor directives --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-pseudo/DirectiveMap.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/TokenKinds.h"
#include "llvm/Support/FormatVariadic.h"

namespace clang {
namespace pseudo {
namespace {

class DirectiveParser {
public:
  explicit DirectiveParser(const TokenStream &Code)
      : Code(Code), Tok(&Code.front()) {}
  void parse(DirectiveMap *Result) { parse(Result, /*TopLevel=*/true); }

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

  // Parses tokens starting at Tok into Map.
  // If we reach an End or Else directive that ends Map, returns it.
  // If TopLevel is true, then we do not expect End and always return None.
  llvm::Optional<DirectiveMap::Directive> parse(DirectiveMap *Map,
                                                bool TopLevel) {
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
        Map->Chunks.push_back(DirectiveMap::Code{
            Token::Range{Code.index(*Start), Code.index(*Tok)}});
        continue;
      }

      // We have some kind of directive.
      DirectiveMap::Directive Directive;
      parseDirective(&Directive);
      Cond Kind = classifyDirective(Directive.Kind);
      if (Kind == Cond::If) {
        // #if or similar, starting a nested conditional block.
        DirectiveMap::Conditional Conditional;
        Conditional.Branches.emplace_back();
        Conditional.Branches.back().first = std::move(Directive);
        parseConditional(&Conditional);
        Map->Chunks.push_back(std::move(Conditional));
      } else if ((Kind == Cond::Else || Kind == Cond::End) && !TopLevel) {
        // #endif or similar, ending this PStructure scope.
        // (#endif is unexpected at the top level, treat as simple directive).
        return std::move(Directive);
      } else {
        // #define or similar, a simple directive at the current scope.
        Map->Chunks.push_back(std::move(Directive));
      }
    }
    return None;
  }

  // Parse the rest of a conditional section, after seeing the If directive.
  // Returns after consuming the End directive.
  void parseConditional(DirectiveMap::Conditional *C) {
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
  void parseDirective(DirectiveMap::Directive *D) {
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

DirectiveMap DirectiveMap::parse(const TokenStream &Code) {
  DirectiveMap Result;
  DirectiveParser(Code).parse(&Result);
  return Result;
}

static void dump(llvm::raw_ostream &OS, const DirectiveMap &, unsigned Indent);
static void dump(llvm::raw_ostream &OS,
                 const DirectiveMap::Directive &Directive, unsigned Indent,
                 bool Taken = false) {
  OS.indent(Indent) << llvm::formatv(
      "#{0} ({1} tokens){2}\n", tok::getPPKeywordSpelling(Directive.Kind),
      Directive.Tokens.size(), Taken ? " TAKEN" : "");
}
static void dump(llvm::raw_ostream &OS, const DirectiveMap::Code &Code,
                 unsigned Indent) {
  OS.indent(Indent) << llvm::formatv("code ({0} tokens)\n", Code.Tokens.size());
}
static void dump(llvm::raw_ostream &OS,
                 const DirectiveMap::Conditional &Conditional,
                 unsigned Indent) {
  for (unsigned I = 0; I < Conditional.Branches.size(); ++I) {
    const auto &Branch = Conditional.Branches[I];
    dump(OS, Branch.first, Indent, Conditional.Taken == I);
    dump(OS, Branch.second, Indent + 2);
  }
  dump(OS, Conditional.End, Indent);
}

static void dump(llvm::raw_ostream &OS, const DirectiveMap::Chunk &Chunk,
                 unsigned Indent) {
  switch (Chunk.kind()) {
  case DirectiveMap::Chunk::K_Empty:
    llvm_unreachable("invalid chunk");
  case DirectiveMap::Chunk::K_Code:
    return dump(OS, (const DirectiveMap::Code &)Chunk, Indent);
  case DirectiveMap::Chunk::K_Directive:
    return dump(OS, (const DirectiveMap::Directive &)Chunk, Indent);
  case DirectiveMap::Chunk::K_Conditional:
    return dump(OS, (const DirectiveMap::Conditional &)Chunk, Indent);
  }
}

static void dump(llvm::raw_ostream &OS, const DirectiveMap &Map,
                 unsigned Indent) {
  for (const auto &Chunk : Map.Chunks)
    dump(OS, Chunk, Indent);
}

// Define operator<< in terms of dump() functions above.
#define OSTREAM_DUMP(Type)                                                     \
  llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Type &T) {        \
    dump(OS, T, 0);                                                            \
    return OS;                                                                 \
  }
OSTREAM_DUMP(DirectiveMap)
OSTREAM_DUMP(DirectiveMap::Chunk)
OSTREAM_DUMP(DirectiveMap::Directive)
OSTREAM_DUMP(DirectiveMap::Conditional)
OSTREAM_DUMP(DirectiveMap::Code)
#undef OSTREAM_DUMP

namespace {
// Makes choices about conditional branches.
//
// Generally it tries to maximize the amount of useful code we see.
//
// Caveat: each conditional is evaluated independently. Consider this code:
//   #ifdef WINDOWS
//     bool isWindows = true;
//   #endif
//   #ifndef WINDOWS
//     bool isWindows = false;
//   #endif
// We take both branches and define isWindows twice. We could track more state
// in order to produce a consistent view, but this is complex.
class BranchChooser {
public:
  BranchChooser(const TokenStream &Code) : Code(Code) {}

  void choose(DirectiveMap &M) { walk(M); }

private:
  // Describes code seen by making particular branch choices. Higher is better.
  struct Score {
    int Tokens = 0; // excluding comments and directives
    int Directives = 0;
    int Errors = 0; // #error directives

    bool operator>(const Score &Other) const {
      // Seeing errors is bad, other things are good.
      return std::make_tuple(-Errors, Tokens, Directives) >
             std::make_tuple(-Other.Errors, Other.Tokens, Other.Directives);
    }

    Score &operator+=(const Score &Other) {
      Tokens += Other.Tokens;
      Directives += Other.Directives;
      Errors += Other.Errors;
      return *this;
    }
  };

  Score walk(DirectiveMap::Code &C) {
    Score S;
    for (const Token &T : Code.tokens(C.Tokens))
      if (T.Kind != tok::comment)
        ++S.Tokens;
    return S;
  }

  Score walk(DirectiveMap::Directive &D) {
    Score S;
    S.Directives = 1;
    S.Errors = D.Kind == tok::pp_error;
    return S;
  }

  Score walk(DirectiveMap::Chunk &C) {
    switch (C.kind()) {
    case DirectiveMap::Chunk::K_Code:
      return walk((DirectiveMap::Code &)C);
    case DirectiveMap::Chunk::K_Directive:
      return walk((DirectiveMap::Directive &)C);
    case DirectiveMap::Chunk::K_Conditional:
      return walk((DirectiveMap::Conditional &)C);
    case DirectiveMap::Chunk::K_Empty:
      break;
    }
    llvm_unreachable("bad chunk kind");
  }

  Score walk(DirectiveMap &M) {
    Score S;
    for (DirectiveMap::Chunk &C : M.Chunks)
      S += walk(C);
    return S;
  }

  Score walk(DirectiveMap::Conditional &C) {
    Score Best;
    bool MayTakeTrivial = true;
    bool TookTrivial = false;

    for (unsigned I = 0; I < C.Branches.size(); ++I) {
      // Walk the branch to make its nested choices in any case.
      Score BranchScore = walk(C.Branches[I].second);
      // If we already took an #if 1, don't consider any other branches.
      if (TookTrivial)
        continue;
      // Is this a trivial #if 0 or #if 1?
      if (auto TriviallyTaken = isTakenWhenReached(C.Branches[I].first)) {
        if (!*TriviallyTaken)
          continue; // Don't consider #if 0 even if it scores well.
        if (MayTakeTrivial)
          TookTrivial = true;
      } else {
        // After a nontrivial condition, #elif 1 isn't guaranteed taken.
        MayTakeTrivial = false;
      }
      // Is this the best branch so far? (Including if it's #if 1).
      if (TookTrivial || !C.Taken.hasValue() || BranchScore > Best) {
        Best = BranchScore;
        C.Taken = I;
      }
    }
    return Best;
  }

  // Return true if the directive starts an always-taken conditional branch,
  // false if the branch is never taken, and None otherwise.
  llvm::Optional<bool> isTakenWhenReached(const DirectiveMap::Directive &Dir) {
    switch (Dir.Kind) {
    case clang::tok::pp_if:
    case clang::tok::pp_elif:
      break; // handled below
    case clang::tok::pp_else:
      return true;
    default: // #ifdef etc
      return llvm::None;
    }

    const auto &Tokens = Code.tokens(Dir.Tokens);
    assert(!Tokens.empty() && Tokens.front().Kind == tok::hash);
    const Token &Name = Tokens.front().nextNC();
    const Token &Value = Name.nextNC();
    // Does the condition consist of exactly one token?
    if (&Value >= Tokens.end() || &Value.nextNC() < Tokens.end())
      return llvm::None;
    return llvm::StringSwitch<llvm::Optional<bool>>(Value.text())
        .Cases("true", "1", true)
        .Cases("false", "0", false)
        .Default(llvm::None);
  }

  const TokenStream &Code;
};

} // namespace

void chooseConditionalBranches(DirectiveMap &Map, const TokenStream &Code) {
  BranchChooser{Code}.choose(Map);
}

} // namespace pseudo
} // namespace clang
