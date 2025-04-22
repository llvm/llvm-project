/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Code to translate different attribute spellings into internal identifiers  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: Attr.td                                                              *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

  switch (getParsedKind()) {
    case IgnoredAttribute:
    case UnknownAttribute:
    case NoSemaHandlerAttribute:
      llvm_unreachable("Ignored/unknown shouldn't get here");
  case AT_AArch64SVEPcs: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_AArch64VectorPcs: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_AMDGPUFlatWorkGroupSize: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    break;
  }
  case AT_AMDGPUKernelCall: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_AMDGPUMaxNumWorkGroups: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    break;
  }
  case AT_AMDGPUNumSGPR: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    break;
  }
  case AT_AMDGPUNumVGPR: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    break;
  }
  case AT_AMDGPUWavesPerEU: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    break;
  }
  case AT_Interrupt: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_ARMInterruptSaveFP: {
    return 0;
    break;
  }
  case AT_ARMSaveFP: {
    return 0;
    break;
  }
  case AT_AVRSignal: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_AbiTag: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    break;
  }
  case AT_AcquireCapability: {
    if (Name.size() == 18 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (Name.size() == 18 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (Name.size() == 25 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 2;
    if (Name.size() == 25 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 3;
    if (Name.size() == 23 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 4;
    if (Name.size() == 20 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 5;
    break;
  }
  case AT_AcquireHandle: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_AcquiredAfter: {
    return 0;
    break;
  }
  case AT_AcquiredBefore: {
    return 0;
    break;
  }
  case AT_AddressSpace: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_Alias: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_AlignValue: {
    return 0;
    break;
  }
  case AT_Aligned: {
    if (Name == "aligned" && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (Name == "aligned" && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (Name == "aligned" && getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    if (Name.size() == 5 && getSyntax() == AttributeCommonInfo::AS_Declspec && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 3;
    if (Name == "alignas" && getSyntax() == AttributeCommonInfo::AS_Keyword && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 4;
    if (Name.size() == 8 && getSyntax() == AttributeCommonInfo::AS_Keyword && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 5;
    break;
  }
  case AT_AllocAlign: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_AllocSize: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_Allocating: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_AlwaysDestroy: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    break;
  }
  case AT_AlwaysInline: {
    if (Name == "always_inline" && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (Name == "always_inline" && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (Name == "always_inline" && getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    if (Name == "always_inline" && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 3;
    if (Name == "always_inline" && getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 4;
    if (Name == "__forceinline" && getSyntax() == AttributeCommonInfo::AS_Keyword && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 5;
    break;
  }
  case AT_AnalyzerNoReturn: {
    return 0;
    break;
  }
  case AT_Annotate: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_AnnotateType: {
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    break;
  }
  case AT_AnyX86NoCallerSavedRegisters: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_AnyX86NoCfCheck: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_ArcWeakrefUnavailable: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_ArgumentWithTypeTag: {
    if (Name.size() == 22 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (Name.size() == 22 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (Name.size() == 22 && getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    if (Name.size() == 21 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 3;
    if (Name.size() == 21 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 4;
    if (Name.size() == 21 && getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 5;
    break;
  }
  case AT_ArmAgnostic: {
    return 0;
    break;
  }
  case AT_ArmBuiltinAlias: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_ArmIn: {
    return 0;
    break;
  }
  case AT_ArmInOut: {
    return 0;
    break;
  }
  case AT_ArmLocallyStreaming: {
    return 0;
    break;
  }
  case AT_ArmMveStrictPolymorphism: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_ArmNew: {
    return 0;
    break;
  }
  case AT_ArmOut: {
    return 0;
    break;
  }
  case AT_ArmPreserves: {
    return 0;
    break;
  }
  case AT_ArmStreaming: {
    return 0;
    break;
  }
  case AT_ArmStreamingCompatible: {
    return 0;
    break;
  }
  case AT_ArmSveVectorBits: {
    return 0;
    break;
  }
  case AT_Artificial: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_AssertCapability: {
    if (Name.size() == 17 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (Name.size() == 17 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (Name.size() == 24 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 2;
    if (Name.size() == 24 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 3;
    if (Name.size() == 21 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 4;
    if (Name.size() == 18 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 5;
    break;
  }
  case AT_AssumeAligned: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_Atomic: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_Availability: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_AvailableOnlyInDefaultEvalMethod: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_BPFFastCall: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_BPFPreserveAccessIndex: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_BPFPreserveStaticOffset: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_BTFDeclTag: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_BTFTypeTag: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_Blocking: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_Blocks: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_BuiltinAlias: {
    if (Name.size() == 13 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 0;
    if (Name.size() == 13 && getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (Name.size() == 19 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 2;
    break;
  }
  case AT_CDecl: {
    if (Name.size() == 5 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (Name.size() == 5 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (Name.size() == 5 && getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    if (Name.size() == 7 && getSyntax() == AttributeCommonInfo::AS_Keyword && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 3;
    if (Name.size() == 6 && getSyntax() == AttributeCommonInfo::AS_Keyword && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 4;
    break;
  }
  case AT_CFAuditedTransfer: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_CFConsumed: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_CFGuard: {
    if (getSyntax() == AttributeCommonInfo::AS_Declspec && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 3;
    break;
  }
  case AT_CFICanonicalJumpTable: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_CFReturnsNotRetained: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_CFReturnsRetained: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_CFUnknownTransfer: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_CPUDispatch: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    if (getSyntax() == AttributeCommonInfo::AS_Declspec && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 3;
    break;
  }
  case AT_CPUSpecific: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    if (getSyntax() == AttributeCommonInfo::AS_Declspec && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 3;
    break;
  }
  case AT_CUDAConstant: {
    if (Name.size() == 8 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (Name.size() == 12 && getSyntax() == AttributeCommonInfo::AS_Declspec && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 1;
    break;
  }
  case AT_CUDADevice: {
    if (Name.size() == 6 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (Name.size() == 10 && getSyntax() == AttributeCommonInfo::AS_Declspec && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 1;
    break;
  }
  case AT_CUDADeviceBuiltinSurfaceType: {
    if (Name.size() == 27 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (Name.size() == 31 && getSyntax() == AttributeCommonInfo::AS_Declspec && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 1;
    break;
  }
  case AT_CUDADeviceBuiltinTextureType: {
    if (Name.size() == 27 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (Name.size() == 31 && getSyntax() == AttributeCommonInfo::AS_Declspec && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 1;
    break;
  }
  case AT_CUDAGlobal: {
    if (Name.size() == 6 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (Name.size() == 10 && getSyntax() == AttributeCommonInfo::AS_Declspec && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 1;
    break;
  }
  case AT_CUDAGridConstant: {
    if (Name.size() == 13 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (Name.size() == 17 && getSyntax() == AttributeCommonInfo::AS_Declspec && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 1;
    break;
  }
  case AT_CUDAHost: {
    if (Name.size() == 4 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (Name.size() == 8 && getSyntax() == AttributeCommonInfo::AS_Declspec && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 1;
    break;
  }
  case AT_CUDAInvalidTarget: {
    return 0;
    break;
  }
  case AT_CUDALaunchBounds: {
    if (Name.size() == 13 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (Name.size() == 17 && getSyntax() == AttributeCommonInfo::AS_Declspec && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 1;
    break;
  }
  case AT_CUDAShared: {
    if (Name.size() == 6 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (Name.size() == 10 && getSyntax() == AttributeCommonInfo::AS_Declspec && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 1;
    break;
  }
  case AT_CXX11NoReturn: {
    if (Name.size() == 8 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (Name.size() == 8 && getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 1;
    if (Name.size() == 9 && getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 2;
    break;
  }
  case AT_CXXAssume: {
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 3;
    break;
  }
  case AT_CallableWhen: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    break;
  }
  case AT_Callback: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_CalledOnce: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_Capability: {
    if (Name.size() == 10 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (Name.size() == 10 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (Name.size() == 17 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 2;
    if (Name.size() == 17 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 3;
    break;
  }
  case AT_CarriesDependency: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 1;
    break;
  }
  case AT_Cleanup: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_ClspvLibclcBuiltin: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_CmseNSCall: {
    return 0;
    break;
  }
  case AT_CmseNSEntry: {
    return 0;
    break;
  }
  case AT_CodeAlign: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_CodeModel: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_CodeSeg: {
    return 0;
    break;
  }
  case AT_Cold: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_Common: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_Const: {
    if (Name.size() == 5 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (Name.size() == 5 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (Name.size() == 5 && getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    if (Name.size() == 7 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 3;
    if (Name.size() == 7 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 4;
    if (Name.size() == 7 && getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 5;
    break;
  }
  case AT_ConstInit: {
    if (Name.size() == 9 && getSyntax() == AttributeCommonInfo::AS_Keyword && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (Name.size() == 31 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 1;
    if (Name.size() == 31 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_Constructor: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_Consumable: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    break;
  }
  case AT_ConsumableAutoCast: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    break;
  }
  case AT_ConsumableSetOnRead: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    break;
  }
  case AT_Convergent: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_CoroAwaitElidable: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_CoroAwaitElidableArgument: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_CoroDisableLifetimeBound: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_CoroLifetimeBound: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_CoroOnlyDestroyWhenComplete: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_CoroReturnType: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_CoroWrapper: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_CountedBy: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_CountedByOrNull: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_DLLExport: {
    if (getSyntax() == AttributeCommonInfo::AS_Declspec && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 3;
    break;
  }
  case AT_DLLExportStaticLocal: {
    return 0;
    break;
  }
  case AT_DLLImport: {
    if (getSyntax() == AttributeCommonInfo::AS_Declspec && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 3;
    break;
  }
  case AT_DLLImportStaticLocal: {
    return 0;
    break;
  }
  case AT_Deprecated: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    if (getSyntax() == AttributeCommonInfo::AS_Declspec && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 3;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 4;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 5;
    break;
  }
  case AT_Destructor: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_DiagnoseAsBuiltin: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_DiagnoseIf: {
    return 0;
    break;
  }
  case AT_DisableSanitizerInstrumentation: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_DisableTailCalls: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_EmptyBases: {
    return 0;
    break;
  }
  case AT_EnableIf: {
    return 0;
    break;
  }
  case AT_EnforceTCB: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_EnforceTCBLeaf: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_EnumExtensibility: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_Error: {
    if (Name.size() == 5 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (Name.size() == 5 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (Name.size() == 5 && getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    if (Name.size() == 7 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 3;
    if (Name.size() == 7 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 4;
    if (Name.size() == 7 && getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 5;
    break;
  }
  case AT_ExcludeFromExplicitInstantiation: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_ExplicitInit: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_ExtVectorType: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_ExternalSourceSymbol: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_FallThrough: {
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 3;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 4;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 5;
    break;
  }
  case AT_FastCall: {
    if (Name.size() == 8 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (Name.size() == 8 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (Name.size() == 8 && getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    if (Name.size() == 10 && getSyntax() == AttributeCommonInfo::AS_Keyword && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 3;
    if (Name.size() == 9 && getSyntax() == AttributeCommonInfo::AS_Keyword && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 4;
    break;
  }
  case AT_FlagEnum: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_Flatten: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_Format: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_FormatArg: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_FormatMatches: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_FunctionReturnThunks: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_GNUInline: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_GuardedBy: {
    return 0;
    break;
  }
  case AT_GuardedVar: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    break;
  }
  case AT_HIPManaged: {
    if (Name.size() == 7 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (Name.size() == 11 && getSyntax() == AttributeCommonInfo::AS_Declspec && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 1;
    break;
  }
  case AT_HLSLContainedType: {
    return 0;
    break;
  }
  case AT_HLSLControlFlowHint: {
    if (Name.size() == 6 && getSyntax() == AttributeCommonInfo::AS_Microsoft && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (Name.size() == 7 && getSyntax() == AttributeCommonInfo::AS_Microsoft && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 1;
    break;
  }
  case AT_HLSLGroupSharedAddressSpace: {
    return 0;
    break;
  }
  case AT_HLSLLoopHint: {
    if (Name.size() == 6 && getSyntax() == AttributeCommonInfo::AS_Microsoft && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (Name.size() == 4 && getSyntax() == AttributeCommonInfo::AS_Microsoft && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 1;
    break;
  }
  case AT_HLSLNumThreads: {
    return 0;
    break;
  }
  case AT_HLSLPackOffset: {
    return 0;
    break;
  }
  case AT_HLSLParamModifier: {
    if (Name.size() == 2 && getSyntax() == AttributeCommonInfo::AS_Keyword && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (Name.size() == 5 && getSyntax() == AttributeCommonInfo::AS_Keyword && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 1;
    if (Name.size() == 3 && getSyntax() == AttributeCommonInfo::AS_Keyword && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 2;
    break;
  }
  case AT_HLSLROV: {
    return 0;
    break;
  }
  case AT_HLSLRawBuffer: {
    return 0;
    break;
  }
  case AT_HLSLResourceBinding: {
    return 0;
    break;
  }
  case AT_HLSLResourceClass: {
    return 0;
    break;
  }
  case AT_HLSLSV_DispatchThreadID: {
    return 0;
    break;
  }
  case AT_HLSLSV_GroupID: {
    return 0;
    break;
  }
  case AT_HLSLSV_GroupIndex: {
    return 0;
    break;
  }
  case AT_HLSLSV_GroupThreadID: {
    return 0;
    break;
  }
  case AT_HLSLShader: {
    return 0;
    break;
  }
  case AT_HLSLWaveSize: {
    return 0;
    break;
  }
  case AT_Hot: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_HybridPatchable: {
    if (getSyntax() == AttributeCommonInfo::AS_Declspec && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 3;
    break;
  }
  case AT_IBAction: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_IBOutlet: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_IBOutletCollection: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_IFunc: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_InitPriority: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    break;
  }
  case AT_IntelOclBicc: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    break;
  }
  case AT_InternalLinkage: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_LTOVisibilityPublic: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_LayoutVersion: {
    return 0;
    break;
  }
  case AT_Leaf: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_LifetimeBound: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    break;
  }
  case AT_LifetimeCaptureBy: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    break;
  }
  case AT_Likely: {
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    break;
  }
  case AT_LoaderUninitialized: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_LockReturned: {
    return 0;
    break;
  }
  case AT_Lockable: {
    return 0;
    break;
  }
  case AT_LocksExcluded: {
    return 0;
    break;
  }
  case AT_LoopHint: {
    if (Name.size() == 4 && getSyntax() == AttributeCommonInfo::AS_Pragma && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 0;
    if (Name.size() == 6 && getSyntax() == AttributeCommonInfo::AS_Pragma && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 1;
    if (Name.size() == 8 && getSyntax() == AttributeCommonInfo::AS_Pragma && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 2;
    if (Name.size() == 14 && getSyntax() == AttributeCommonInfo::AS_Pragma && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 3;
    if (Name.size() == 16 && getSyntax() == AttributeCommonInfo::AS_Pragma && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 4;
    break;
  }
  case AT_M68kRTD: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_MIGServerRoutine: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_MSABI: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_MSAllocator: {
    return 0;
    break;
  }
  case AT_MSConstexpr: {
    return 0;
    break;
  }
  case AT_MSInheritance: {
    if (Name.size() == 20 && getSyntax() == AttributeCommonInfo::AS_Keyword && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (Name.size() == 22 && getSyntax() == AttributeCommonInfo::AS_Keyword && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 1;
    if (Name.size() == 21 && getSyntax() == AttributeCommonInfo::AS_Keyword && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 2;
    if (Name.size() == 25 && getSyntax() == AttributeCommonInfo::AS_Keyword && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 3;
    break;
  }
  case AT_MSNoVTable: {
    return 0;
    break;
  }
  case AT_MSStruct: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_MatrixType: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_MayAlias: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_MaybeUndef: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_MicroMips: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_MinSize: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_MinVectorWidth: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_Mips16: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_MipsLongCall: {
    if (Name.size() == 9 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (Name.size() == 9 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (Name.size() == 9 && getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    if (Name.size() == 3 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 3;
    if (Name.size() == 3 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 4;
    if (Name.size() == 3 && getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 5;
    break;
  }
  case AT_MipsShortCall: {
    if (Name.size() == 10 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (Name.size() == 10 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (Name.size() == 10 && getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    if (Name.size() == 4 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 3;
    if (Name.size() == 4 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 4;
    if (Name.size() == 4 && getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 5;
    break;
  }
  case AT_Mode: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_MustTail: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_NSConsumed: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_NSConsumesSelf: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_NSErrorDomain: {
    return 0;
    break;
  }
  case AT_NSReturnsAutoreleased: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_NSReturnsNotRetained: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_NSReturnsRetained: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_NVPTXKernel: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_Naked: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    if (getSyntax() == AttributeCommonInfo::AS_Declspec && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 3;
    break;
  }
  case AT_NeonPolyVectorType: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_NeonVectorType: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_NoAlias: {
    return 0;
    break;
  }
  case AT_NoBuiltin: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_NoCommon: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_NoConvergent: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    if (getSyntax() == AttributeCommonInfo::AS_Declspec && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 3;
    break;
  }
  case AT_NoDebug: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_NoDeref: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_NoDestroy: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    break;
  }
  case AT_NoDuplicate: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_NoEscape: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_NoInline: {
    if (Name.size() == 12 && getSyntax() == AttributeCommonInfo::AS_Keyword && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (Name.size() == 8 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 1;
    if (Name.size() == 8 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    if (Name.size() == 8 && getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 3;
    if (Name.size() == 8 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 4;
    if (Name.size() == 8 && getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 5;
    if (Name.size() == 8 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::MSVC)
      return 6;
    if (Name.size() == 8 && getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::MSVC)
      return 7;
    if (Name.size() == 8 && getSyntax() == AttributeCommonInfo::AS_Declspec && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 8;
    break;
  }
  case AT_NoInstrumentFunction: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_NoMerge: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_NoMicroMips: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_NoMips16: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_NoProfileFunction: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_NoRandomizeLayout: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_NoReturn: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    if (getSyntax() == AttributeCommonInfo::AS_Declspec && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 3;
    break;
  }
  case AT_NoSanitize: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 3;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 4;
    break;
  }
  case AT_NoSanitizeSpecific: {
    if (Name.size() == 26 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (Name.size() == 26 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (Name.size() == 26 && getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    if (Name.size() == 19 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 3;
    if (Name.size() == 19 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 4;
    if (Name.size() == 19 && getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 5;
    if (Name == "no_sanitize_thread" && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 6;
    if (Name == "no_sanitize_thread" && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 7;
    if (Name == "no_sanitize_thread" && getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 8;
    if (Name == "no_sanitize_memory" && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 9;
    if (Name == "no_sanitize_memory" && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 10;
    if (Name == "no_sanitize_memory" && getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 11;
    break;
  }
  case AT_NoSpecializations: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    break;
  }
  case AT_NoSpeculativeLoadHardening: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_NoSplitStack: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_NoStackProtector: {
    if (Name.size() == 18 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (Name.size() == 18 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (Name.size() == 18 && getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    if (Name.size() == 18 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 3;
    if (Name.size() == 18 && getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 4;
    if (Name.size() == 11 && getSyntax() == AttributeCommonInfo::AS_Declspec && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 5;
    break;
  }
  case AT_NoThreadSafetyAnalysis: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_NoThrow: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    if (getSyntax() == AttributeCommonInfo::AS_Declspec && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 3;
    break;
  }
  case AT_NoTrivialAutoVarInit: {
    return 0;
    break;
  }
  case AT_NoUniqueAddress: {
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::MSVC)
      return 1;
    break;
  }
  case AT_NoUwtable: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_NonAllocating: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_NonBlocking: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_NonNull: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_NotTailCalled: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_OMPAssume: {
    return 0;
    break;
  }
  case AT_OSConsumed: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_OSConsumesThis: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_OSReturnsNotRetained: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_OSReturnsRetained: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_OSReturnsRetainedOnNonZero: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_OSReturnsRetainedOnZero: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_ObjCBoxable: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_ObjCBridge: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_ObjCBridgeMutable: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_ObjCBridgeRelated: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_ObjCClassStub: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_ObjCDesignatedInitializer: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_ObjCDirect: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_ObjCDirectMembers: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_ObjCException: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_ObjCExplicitProtocolImpl: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_ObjCExternallyRetained: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_ObjCGC: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_ObjCIndependentClass: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_ObjCInertUnsafeUnretained: {
    return 0;
    break;
  }
  case AT_ObjCKindOf: {
    return 0;
    break;
  }
  case AT_ObjCMethodFamily: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_ObjCNSObject: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_ObjCNonLazyClass: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_ObjCNonRuntimeProtocol: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_ObjCOwnership: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_ObjCPreciseLifetime: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_ObjCRequiresPropertyDefs: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_ObjCRequiresSuper: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_ObjCReturnsInnerPointer: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_ObjCRootClass: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_ObjCRuntimeName: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_ObjCRuntimeVisible: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_ObjCSubclassingRestricted: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_OpenACCRoutineAnnot: {
    return 0;
    break;
  }
  case AT_OpenCLAccess: {
    if (Name.size() == 11 && getSyntax() == AttributeCommonInfo::AS_Keyword && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (Name.size() == 9 && getSyntax() == AttributeCommonInfo::AS_Keyword && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 1;
    if (Name == "__write_only" && getSyntax() == AttributeCommonInfo::AS_Keyword && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 2;
    if (Name == "write_only" && getSyntax() == AttributeCommonInfo::AS_Keyword && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 3;
    if (Name == "__read_write" && getSyntax() == AttributeCommonInfo::AS_Keyword && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 4;
    if (Name == "read_write" && getSyntax() == AttributeCommonInfo::AS_Keyword && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 5;
    break;
  }
  case AT_OpenCLConstantAddressSpace: {
    if (Name.size() == 10 && getSyntax() == AttributeCommonInfo::AS_Keyword && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (Name.size() == 8 && getSyntax() == AttributeCommonInfo::AS_Keyword && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 1;
    if (Name.size() == 15 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 2;
    if (Name.size() == 15 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 3;
    if (Name.size() == 15 && getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 4;
    break;
  }
  case AT_OpenCLGenericAddressSpace: {
    if (Name.size() == 9 && getSyntax() == AttributeCommonInfo::AS_Keyword && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (Name.size() == 7 && getSyntax() == AttributeCommonInfo::AS_Keyword && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 1;
    if (Name.size() == 14 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 2;
    if (Name.size() == 14 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 3;
    if (Name.size() == 14 && getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 4;
    break;
  }
  case AT_OpenCLGlobalAddressSpace: {
    if (Name.size() == 8 && getSyntax() == AttributeCommonInfo::AS_Keyword && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (Name.size() == 6 && getSyntax() == AttributeCommonInfo::AS_Keyword && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 1;
    if (Name.size() == 13 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 2;
    if (Name.size() == 13 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 3;
    if (Name.size() == 13 && getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 4;
    break;
  }
  case AT_OpenCLGlobalDeviceAddressSpace: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_OpenCLGlobalHostAddressSpace: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_OpenCLIntelReqdSubGroupSize: {
    return 0;
    break;
  }
  case AT_OpenCLKernel: {
    if (Name.size() == 8 && getSyntax() == AttributeCommonInfo::AS_Keyword && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (Name.size() == 6 && getSyntax() == AttributeCommonInfo::AS_Keyword && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 1;
    break;
  }
  case AT_OpenCLLocalAddressSpace: {
    if (Name.size() == 7 && getSyntax() == AttributeCommonInfo::AS_Keyword && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (Name.size() == 5 && getSyntax() == AttributeCommonInfo::AS_Keyword && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 1;
    if (Name.size() == 12 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 2;
    if (Name.size() == 12 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 3;
    if (Name.size() == 12 && getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 4;
    break;
  }
  case AT_OpenCLNoSVM: {
    return 0;
    break;
  }
  case AT_OpenCLPrivateAddressSpace: {
    if (Name.size() == 9 && getSyntax() == AttributeCommonInfo::AS_Keyword && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (Name.size() == 7 && getSyntax() == AttributeCommonInfo::AS_Keyword && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 1;
    if (Name.size() == 14 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 2;
    if (Name.size() == 14 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 3;
    if (Name.size() == 14 && getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 4;
    break;
  }
  case AT_OpenCLUnrollHint: {
    return 0;
    break;
  }
  case AT_OptimizeNone: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_Overloadable: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_Owner: {
    return 0;
    break;
  }
  case AT_Ownership: {
    if (Name == "ownership_holds" && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (Name == "ownership_holds" && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (Name == "ownership_holds" && getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    if (Name.size() == 17 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 3;
    if (Name.size() == 17 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 4;
    if (Name.size() == 17 && getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 5;
    if (Name == "ownership_takes" && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 6;
    if (Name == "ownership_takes" && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 7;
    if (Name == "ownership_takes" && getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 8;
    break;
  }
  case AT_Packed: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_ParamTypestate: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    break;
  }
  case AT_Pascal: {
    if (Name.size() == 6 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (Name.size() == 6 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (Name.size() == 6 && getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    if (Name.size() == 8 && getSyntax() == AttributeCommonInfo::AS_Keyword && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 3;
    if (Name.size() == 7 && getSyntax() == AttributeCommonInfo::AS_Keyword && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 4;
    break;
  }
  case AT_PassObjectSize: {
    if (Name.size() == 16 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (Name.size() == 16 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (Name.size() == 16 && getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    if (Name.size() == 24 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 3;
    if (Name.size() == 24 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 4;
    if (Name.size() == 24 && getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 5;
    break;
  }
  case AT_PatchableFunctionEntry: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_Pcs: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_Pointer: {
    return 0;
    break;
  }
  case AT_PointerAuth: {
    return 0;
    break;
  }
  case AT_PragmaClangBSSSection: {
    return 0;
    break;
  }
  case AT_PragmaClangDataSection: {
    return 0;
    break;
  }
  case AT_PragmaClangRelroSection: {
    return 0;
    break;
  }
  case AT_PragmaClangRodataSection: {
    return 0;
    break;
  }
  case AT_PragmaClangTextSection: {
    return 0;
    break;
  }
  case AT_PreferredName: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    break;
  }
  case AT_PreferredType: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_PreserveAll: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_PreserveMost: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_PreserveNone: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_PtGuardedBy: {
    return 0;
    break;
  }
  case AT_PtGuardedVar: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    break;
  }
  case AT_Ptr32: {
    return 0;
    break;
  }
  case AT_Ptr64: {
    return 0;
    break;
  }
  case AT_Pure: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_RISCVRVVVectorBits: {
    return 0;
    break;
  }
  case AT_RISCVVLSCC: {
    if (Name.size() == 6 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::RISCV)
      return 0;
    if (Name.size() == 6 && getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::RISCV)
      return 1;
    if (Name.size() == 12 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 2;
    if (Name.size() == 12 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 3;
    if (Name.size() == 12 && getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 4;
    break;
  }
  case AT_RISCVVectorCC: {
    if (Name.size() == 9 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::RISCV)
      return 0;
    if (Name.size() == 9 && getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::RISCV)
      return 1;
    if (Name.size() == 15 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 2;
    if (Name.size() == 15 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 3;
    if (Name.size() == 15 && getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 4;
    break;
  }
  case AT_RandomizeLayout: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_ReadOnlyPlacement: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_RegCall: {
    if (Name.size() == 7 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (Name.size() == 7 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (Name.size() == 7 && getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    if (Name.size() == 9 && getSyntax() == AttributeCommonInfo::AS_Keyword && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 3;
    break;
  }
  case AT_Regparm: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_Reinitializes: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    break;
  }
  case AT_ReleaseCapability: {
    if (Name.size() == 18 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (Name.size() == 18 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (Name.size() == 25 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 2;
    if (Name.size() == 25 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 3;
    if (Name.size() == 26 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 4;
    if (Name.size() == 26 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 5;
    if (Name.size() == 15 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 6;
    if (Name.size() == 15 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 7;
    break;
  }
  case AT_ReleaseHandle: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_ReqdWorkGroupSize: {
    return 0;
    break;
  }
  case AT_RequiresCapability: {
    if (Name.size() == 19 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (Name.size() == 19 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (Name.size() == 24 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 2;
    if (Name.size() == 24 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 3;
    if (Name.size() == 26 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 4;
    if (Name.size() == 26 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 5;
    if (Name.size() == 21 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 6;
    if (Name.size() == 21 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 7;
    break;
  }
  case AT_Restrict: {
    if (Name.size() == 8 && getSyntax() == AttributeCommonInfo::AS_Declspec && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (Name.size() == 6 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 1;
    if (Name.size() == 6 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    if (Name.size() == 6 && getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 3;
    break;
  }
  case AT_Retain: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_ReturnTypestate: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    break;
  }
  case AT_ReturnsNonNull: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_ReturnsTwice: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_SPtr: {
    return 0;
    break;
  }
  case AT_SYCLKernel: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_SYCLKernelEntryPoint: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_SYCLSpecialClass: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_ScopedLockable: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    break;
  }
  case AT_Section: {
    if (Name.size() == 7 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (Name.size() == 7 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (Name.size() == 7 && getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    if (Name.size() == 8 && getSyntax() == AttributeCommonInfo::AS_Declspec && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 3;
    break;
  }
  case AT_SelectAny: {
    if (getSyntax() == AttributeCommonInfo::AS_Declspec && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 3;
    break;
  }
  case AT_Sentinel: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_SetTypestate: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    break;
  }
  case AT_SizedBy: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_SizedByOrNull: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_SpeculativeLoadHardening: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_StandaloneDebug: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    break;
  }
  case AT_StdCall: {
    if (Name.size() == 7 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (Name.size() == 7 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (Name.size() == 7 && getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    if (Name.size() == 9 && getSyntax() == AttributeCommonInfo::AS_Keyword && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 3;
    if (Name.size() == 8 && getSyntax() == AttributeCommonInfo::AS_Keyword && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 4;
    break;
  }
  case AT_StrictFP: {
    return 0;
    break;
  }
  case AT_StrictGuardStackCheck: {
    return 0;
    break;
  }
  case AT_Suppress: {
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GSL)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 3;
    break;
  }
  case AT_SwiftAsync: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_SwiftAsyncCall: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_SwiftAsyncContext: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_SwiftAsyncError: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_SwiftAsyncName: {
    return 0;
    break;
  }
  case AT_SwiftAttr: {
    return 0;
    break;
  }
  case AT_SwiftBridge: {
    return 0;
    break;
  }
  case AT_SwiftBridgedTypedef: {
    return 0;
    break;
  }
  case AT_SwiftCall: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_SwiftContext: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_SwiftError: {
    return 0;
    break;
  }
  case AT_SwiftErrorResult: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_SwiftIndirectResult: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_SwiftName: {
    return 0;
    break;
  }
  case AT_SwiftNewType: {
    if (Name == "swift_newtype" && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (Name == "swift_wrapper" && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 1;
    break;
  }
  case AT_SwiftObjCMembers: {
    return 0;
    break;
  }
  case AT_SwiftPrivate: {
    return 0;
    break;
  }
  case AT_SysVABI: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_TLSModel: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_Target: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_TargetClones: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_TargetVersion: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_TestTypestate: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    break;
  }
  case AT_ThisCall: {
    if (Name.size() == 8 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (Name.size() == 8 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (Name.size() == 8 && getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    if (Name.size() == 10 && getSyntax() == AttributeCommonInfo::AS_Keyword && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 3;
    if (Name.size() == 9 && getSyntax() == AttributeCommonInfo::AS_Keyword && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 4;
    break;
  }
  case AT_Thread: {
    return 0;
    break;
  }
  case AT_TransparentUnion: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_TrivialABI: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    break;
  }
  case AT_TryAcquireCapability: {
    if (Name.size() == 22 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (Name.size() == 22 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (Name.size() == 29 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 2;
    if (Name.size() == 29 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 3;
    if (Name.size() == 26 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 4;
    if (Name.size() == 23 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 5;
    break;
  }
  case AT_TypeNonNull: {
    return 0;
    break;
  }
  case AT_TypeNullUnspecified: {
    return 0;
    break;
  }
  case AT_TypeNullable: {
    return 0;
    break;
  }
  case AT_TypeNullableResult: {
    return 0;
    break;
  }
  case AT_TypeTagForDatatype: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_TypeVisibility: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_UPtr: {
    return 0;
    break;
  }
  case AT_Unavailable: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_Uninitialized: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    break;
  }
  case AT_Unlikely: {
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    break;
  }
  case AT_UnsafeBufferUsage: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_Unused: {
    if (Name.size() == 12 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (Name.size() == 6 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 1;
    if (Name.size() == 6 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    if (Name.size() == 6 && getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 3;
    if (Name.size() == 12 && getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 4;
    break;
  }
  case AT_UseHandle: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_Used: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_UsingIfExists: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    break;
  }
  case AT_Uuid: {
    if (getSyntax() == AttributeCommonInfo::AS_Declspec && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_Microsoft && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 1;
    break;
  }
  case AT_VTablePointerAuthentication: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_VecReturn: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    break;
  }
  case AT_VecTypeHint: {
    return 0;
    break;
  }
  case AT_VectorCall: {
    if (Name.size() == 10 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (Name.size() == 10 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (Name.size() == 10 && getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    if (Name.size() == 12 && getSyntax() == AttributeCommonInfo::AS_Keyword && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 3;
    if (Name.size() == 11 && getSyntax() == AttributeCommonInfo::AS_Keyword && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 4;
    break;
  }
  case AT_VectorSize: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_Visibility: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_WarnUnused: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_WarnUnusedResult: {
    if (Name.size() == 9 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (Name.size() == 9 && getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 1;
    if (Name.size() == 18 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    if (Name.size() == 18 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 3;
    if (Name.size() == 18 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 4;
    if (Name.size() == 18 && getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 5;
    break;
  }
  case AT_Weak: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_WeakImport: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_WeakRef: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_WebAssemblyExportName: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_WebAssemblyFuncref: {
    return 0;
    break;
  }
  case AT_WebAssemblyImportModule: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_WebAssemblyImportName: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_WorkGroupSizeHint: {
    return 0;
    break;
  }
  case AT_X86ForceAlignArgPointer: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  case AT_XRayInstrument: {
    if (Name.size() == 22 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (Name.size() == 22 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (Name.size() == 22 && getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    if (Name.size() == 21 && getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 3;
    if (Name.size() == 21 && getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 4;
    if (Name.size() == 21 && getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 5;
    break;
  }
  case AT_XRayLogArgs: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::CLANG)
      return 2;
    break;
  }
  case AT_ZeroCallUsedRegs: {
    if (getSyntax() == AttributeCommonInfo::AS_GNU && ComputedScope == AttributeCommonInfo::Scope::NONE)
      return 0;
    if (getSyntax() == AttributeCommonInfo::AS_CXX11 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 1;
    if (getSyntax() == AttributeCommonInfo::AS_C23 && ComputedScope == AttributeCommonInfo::Scope::GNU)
      return 2;
    break;
  }
  }
  return 0;
