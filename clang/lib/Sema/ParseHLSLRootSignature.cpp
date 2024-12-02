#include "clang/AST/Attr.h"
#include "clang/Sema/HLSLRootSignature.h"
#include "clang/Sema/ParsedAttr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"

using namespace clang;
using namespace llvm::hlsl;

void RootSignaturParser::ParseRootDefinition() {
  do {
    getNextToken();

    switch (CurTok.Kind) {

    case RSTknInfo::RootFlags:
      getNextToken();
      assert(CurTok.Kind == RSTknInfo::Character && CurTok.Text == "(" &&
             "Missing tkn in root signature");
      break;
      ParseRootFlag();
    case RSTknInfo::RootCBV:
    default:
      llvm_unreachable("Root Element still not suported");
    }
  } while (CurTok.Kind != RSTknInfo::EoF);
}

RSTknInfo RootSignaturParser::gettok() {
  char LastChar = ' ';
  RSTknInfo Response;

  while (isspace(LastChar)) {
    LastChar = nextChar();
  }

  if (isalpha(LastChar)) {
    IdentifierStr = LastChar;
    while (isalnum(curChar()) || curChar() == '_') {
      LastChar = nextChar();
      IdentifierStr += LastChar;
    }

    RSTknInfo::RSTok Tok =
        llvm::StringSwitch<RSTknInfo::RSTok>(IdentifierStr)
            .Case("RootFlags", RSTknInfo::RootFlags)
            .Case("RootConstants", RSTknInfo::RootConstants)
            .Case("RootCBV", RSTknInfo::RootCBV)
            .Case("RootSRV", RSTknInfo::RootSRV)
            .Case("RootUAV", RSTknInfo::RootUAV)
            .Case("DescriptorTable", RSTknInfo::DescriptorTable)
            .Case("StaticSampler", RSTknInfo::StaticSampler)
            .Case("DENY_VERTEX_SHADER_ROOT_ACCESS", RSTknInfo::RootFlag)
            .Case("ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT", RSTknInfo::RootFlag)
            .Case("DENY_HULL_SHADER_ROOT_ACCESS", RSTknInfo::RootFlag)
            .Case("DENY_DOMAIN_SHADER_ROOT_ACCESS", RSTknInfo::RootFlag)
            .Case("DENY_GEOMETRY_SHADER_ROOT_ACCESS", RSTknInfo::RootFlag)
            .Case("DENY_PIXEL_SHADER_ROOT_ACCESS", RSTknInfo::RootFlag)
            .Case("DENY_AMPLIFICATION_SHADER_ROOT_ACCESS", RSTknInfo::RootFlag)
            .Case("DENY_MESH_SHADER_ROOT_ACCESS", RSTknInfo::RootFlag)
            .Case("ALLOW_STREAM_OUTPUT", RSTknInfo::RootFlag)
            .Case("LOCAL_ROOT_SIGNATURE", RSTknInfo::RootFlag)
            .Case("CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED", RSTknInfo::RootFlag)
            .Case("SAMPLER_HEAP_DIRECTLY_INDEXED", RSTknInfo::RootFlag)
            .Case("AllowLowTierReservedHwCbLimit", RSTknInfo::RootFlag)
            .Default(RSTknInfo::EoF);

    assert(Tok != RSTknInfo::EoF && "invalid string in ROOT SIGNATURE");

    Response.Kind = Tok;
    Response.Text = StringRef(IdentifierStr);
    return Response;
  }

  if (isdigit(LastChar)) {
    std::string NumStr;

    do {
      NumStr += LastChar;
      LastChar = nextChar();
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

void RootSignaturParser::ParseRootFlag() {

  do {
    getNextToken();

    if (CurTok.Kind == RSTknInfo::RootFlag) {
      if (CurTok.Text == "DENY_VERTEX_SHADER_ROOT_ACCESS") {
      } else if (CurTok.Text == "ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT") {
      } else if (CurTok.Text == "DENY_HULL_SHADER_ROOT_ACCESS") {
      } else if (CurTok.Text == "DENY_DOMAIN_SHADER_ROOT_ACCESS") {
      } else if (CurTok.Text == "DENY_GEOMETRY_SHADER_ROOT_ACCESS") {
      } else if (CurTok.Text == "DENY_PIXEL_SHADER_ROOT_ACCESS") {
      } else if (CurTok.Text == "DENY_AMPLIFICATION_SHADER_ROOT_ACCESS") {
      } else if (CurTok.Text == "DENY_MESH_SHADER_ROOT_ACCESS") {
      } else if (CurTok.Text == "ALLOW_STREAM_OUTPUT") {
      } else if (CurTok.Text == "LOCAL_ROOT_SIGNATURE") {
      } else if (CurTok.Text == "CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED") {
      } else if (CurTok.Text == "SAMPLER_HEAP_DIRECTLY_INDEXED") {
      } else if (CurTok.Text == "AllowLowTierReservedHwCbLimit") {
      }
    }

  } while (curChar() == ',');
}
