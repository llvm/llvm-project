#include "clang/Sema/ParseHLSLRootSignature.h"

namespace llvm {
namespace hlsl {
namespace root_signature {

// TODO: Hook up with Sema to properly report semantic/validation errors
bool Parser::ReportError() { return true; }

bool Parser::ParseRootFlags() {
  // Set to RootFlags::None and skip whitespace to catch when we have RootFlags(
  // )
  RootFlags Flags = RootFlags::None;
  Buffer = Buffer.drop_while(isspace);
  bool First = true;

  // Loop until we reach the end of the rootflags
  while (!Buffer.starts_with(")")) {
    // Trim expected | when more than 1 flag
    if (!First && !Buffer.consume_front("|"))
      return ReportError();
    First = false;

    // Remove any whitespace
    Buffer = Buffer.drop_while(isspace);

    RootFlags CurFlag;
    if (ParseRootFlag(CurFlag))
      return ReportError();
    Flags |= CurFlag;

    // Remove any whitespace
    Buffer = Buffer.drop_while(isspace);
  }

  // Create and push the root element on the parsed elements
  Elements->push_back(RootElement(Flags));
  return false;
}

template <typename EnumType>
bool Parser::ParseEnum(SmallVector<std::pair<StringLiteral, EnumType>> Mapping,
                       EnumType &Enum) {
  // Retrieve enum
  Token = Buffer.take_while([](char C) { return isalnum(C) || C == '_'; });
  Buffer = Buffer.drop_front(Token.size());

  // Try to get the case-insensitive enum
  auto Switch = llvm::StringSwitch<std::optional<EnumType>>(Token);
  for (auto Pair : Mapping)
    Switch.CaseLower(Pair.first, Pair.second);
  auto MaybeEnum = Switch.Default(std::nullopt);
  if (!MaybeEnum)
    return true;
  Enum = *MaybeEnum;

  return false;
}

bool Parser::ParseRootFlag(RootFlags &Flag) {
  SmallVector<std::pair<StringLiteral, RootFlags>> Mapping = {
      {"0", RootFlags::None},
      {"ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT",
       RootFlags::AllowInputAssemblerInputLayout},
      {"DENY_VERTEX_SHADER_ROOT_ACCESS", RootFlags::DenyVertexShaderRootAccess},
      {"DENY_HULL_SHADER_ROOT_ACCESS", RootFlags::DenyHullShaderRootAccess},
      {"DENY_DOMAIN_SHADER_ROOT_ACCESS", RootFlags::DenyDomainShaderRootAccess},
      {"DENY_GEOMETRY_SHADER_ROOT_ACCESS",
       RootFlags::DenyGeometryShaderRootAccess},
      {"DENY_PIXEL_SHADER_ROOT_ACCESS", RootFlags::DenyPixelShaderRootAccess},
      {"ALLOW_STREAM_OUTPUT", RootFlags::AllowStreamOutput},
      {"LOCAL_ROOT_SIGNATURE", RootFlags::LocalRootSignature},
      {"DENY_AMPLIFICATION_SHADER_ROOT_ACCESS",
       RootFlags::DenyAmplificationShaderRootAccess},
      {"DENY_MESH_SHADER_ROOT_ACCESS", RootFlags::DenyMeshShaderRootAccess},
      {"CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED",
       RootFlags::CBVSRVUAVHeapDirectlyIndexed},
      {"SAMPLER_HEAP_DIRECTLY_INDEXED", RootFlags::SamplerHeapDirectlyIndexed},
      {"AllowLowTierReservedHwCbLimit",
       RootFlags::AllowLowTierReservedHwCbLimit},
  };

  return ParseEnum<RootFlags>(Mapping, Flag);
}

bool Parser::ParseRootElement() {
  // Define different ParserMethods to use StringSwitch for dispatch
  enum class ParserMethod {
    ReportError,
    ParseRootFlags,
  };

  // Retreive which method should be used
  auto Method = llvm::StringSwitch<ParserMethod>(Token)
                    .Case("RootFlags", ParserMethod::ParseRootFlags)
                    .Default(ParserMethod::ReportError);

  // Dispatch on the correct method
  bool Error = false;
  switch (Method) {
  case ParserMethod::ReportError:
    Error = true;
    break;
  case ParserMethod::ParseRootFlags:
    Error = ParseRootFlags();
    break;
  case ParserMethod::ParseRootParameter:
    Error = ParseRootParameter();
    break;
  }

  if (Error)
    return ReportError();

  return false;
}

bool Parser::Parse() {
  bool First = true;
  while (!Buffer.empty()) {
    // Trim expected comma when more than 1 root element
    if (!First && !Buffer.consume_front(","))
      return ReportError();
    First = false;

    // Remove any whitespace
    Buffer = Buffer.drop_while(isspace);

    // Retrieve the root element identifier
    auto Split = Buffer.split('(');
    Token = Split.first;
    Buffer = Split.second;

    // Dispatch to the applicable root element parser
    if (ParseRootElement())
      return ReportError();

    // Then we can clean up the remaining ")"
    if (!Buffer.consume_front(")"))
      return ReportError();
  }

  // All input has been correctly parsed
  return false;
}

} // namespace root_signature
} // namespace hlsl
} // namespace llvm
