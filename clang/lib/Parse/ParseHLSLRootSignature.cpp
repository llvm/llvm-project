#include "clang/AST/Attr.h"
#include "clang/Parse/Parser.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>
#include <memory>

using namespace clang;

struct RSTknInfo {
  enum RSTok {
    RootFlags,
    RootConstants,
    RootCBV,
    RootSRV,
    RootUAV,
    DescriptorTable,
    StaticSampler,
    Number,
    Character,
    StrConst,
    EoF
  };

  RSTknInfo() {}

  RSTok Kind = RSTok::EoF;
  StringRef Text;
};

class RootSignaturParser {
  StringRef Signature;

public:
  RootSignaturParser(StringRef Signature) : Signature(Signature) {}

  void ParseRootDefinition() {
    do {
      getNextToken();

      switch (CurTok.Kind) {

      case RSTknInfo::RootFlags:
        getNextToken();
        assert(CurTok.Kind == RSTknInfo::Character && CurTok.Text == "(" &&
               "Missing tkn in root signature");

        ParseRootFlag();

        break;
      default:
        llvm_unreachable("Root Element still not suported");
      }
    } while (CurTok.Kind != RSTknInfo::EoF);
  }

private:
  RSTknInfo CurTok;
  std::string IdentifierStr;

  void consumeWhitespace() { Signature = Signature.ltrim(" \t\v\f\r"); }

  RSTknInfo gettok() {
    char LastChar = ' ';
    RSTknInfo Response;

    while (isspace(LastChar))
      LastChar = Signature.front();

    if (isalpha(LastChar)) {
      IdentifierStr = LastChar;
      while (isalnum((LastChar = Signature.front())))
        IdentifierStr += LastChar;

      RSTknInfo::RSTok Tok =
          llvm::StringSwitch<RSTknInfo::RSTok>(IdentifierStr)
              .Case("RootFlags", RSTknInfo::RootFlags)
              .Case("RootConstants", RSTknInfo::RootConstants)
              .Case("RootCBV", RSTknInfo::RootCBV)
              .Case("RootSRV", RSTknInfo::RootSRV)
              .Case("RootUAV", RSTknInfo::RootUAV)
              .Case("DescriptorTable", RSTknInfo::DescriptorTable)
              .Case("StaticSampler", RSTknInfo::StaticSampler)
              .Default(RSTknInfo::StrConst);

      Response.Kind = Tok;
      Response.Text = StringRef(IdentifierStr);
      return Response;
    }

    if (isdigit(LastChar)) {
      std::string NumStr;

      do {
        NumStr += LastChar;
        LastChar = Signature.front();
      } while (isdigit(LastChar));

      Response.Kind = RSTknInfo::Number;
      Response.Text = StringRef(IdentifierStr);
      return Response;
    }

    if (LastChar == EOF) {
      Response.Kind = RSTknInfo::EoF;
      return Response;
    }

    Response.Kind = RSTknInfo::Character;
    Response.Text = StringRef(std::string(1, LastChar));
    return Response;
  }

  RSTknInfo getNextToken() { return CurTok = gettok(); }
};

Attr *Parser::ParseHLSLRootSignature(StringRef Signature,
                                     ParsedAttributes &Attrs,
                                     SourceLocation *EndLoc) {
  RootSignaturParser RSParser(Signature);
  RSParser.ParseRootDefinition();
  return nullptr;
}
