/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Attribute deserialization code                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: Attr.td                                                              *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

  switch (Kind) {
  case attr::AArch64SVEPcs: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) AArch64SVEPcsAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::AArch64VectorPcs: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) AArch64VectorPcsAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::AMDGPUFlatWorkGroupSize: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    Expr * min = Record.readExpr();
    Expr * max = Record.readExpr();
    New = new (Context) AMDGPUFlatWorkGroupSizeAttr(Context, Info, min, max);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::AMDGPUKernelCall: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) AMDGPUKernelCallAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::AMDGPUMaxNumWorkGroups: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    Expr * maxNumWorkGroupsX = Record.readExpr();
    Expr * maxNumWorkGroupsY = Record.readExpr();
    Expr * maxNumWorkGroupsZ = Record.readExpr();
    New = new (Context) AMDGPUMaxNumWorkGroupsAttr(Context, Info, maxNumWorkGroupsX, maxNumWorkGroupsY, maxNumWorkGroupsZ);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::AMDGPUNumSGPR: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    unsigned numSGPR = Record.readInt();
    New = new (Context) AMDGPUNumSGPRAttr(Context, Info, numSGPR);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::AMDGPUNumVGPR: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    unsigned numVGPR = Record.readInt();
    New = new (Context) AMDGPUNumVGPRAttr(Context, Info, numVGPR);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::AMDGPUWavesPerEU: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    Expr * min = Record.readExpr();
    Expr * max = Record.readExpr();
    New = new (Context) AMDGPUWavesPerEUAttr(Context, Info, min, max);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ARMInterrupt: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    ARMInterruptAttr::InterruptType interrupt(static_cast<ARMInterruptAttr::InterruptType>(Record.readInt()));
    New = new (Context) ARMInterruptAttr(Context, Info, interrupt);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ARMInterruptSaveFP: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    ARMInterruptSaveFPAttr::InterruptType interrupt(static_cast<ARMInterruptSaveFPAttr::InterruptType>(Record.readInt()));
    New = new (Context) ARMInterruptSaveFPAttr(Context, Info, interrupt);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ARMSaveFP: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) ARMSaveFPAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::AVRInterrupt: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) AVRInterruptAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::AVRSignal: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) AVRSignalAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::AbiTag: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    unsigned tagsSize = Record.readInt();
    SmallVector<StringRef, 4> tags;
    tags.reserve(tagsSize);
    SmallVector<std::string, 4> tagsStorage;
    tagsStorage.reserve(tagsSize);
    for (unsigned i = 0; i != tagsSize; ++i)
      tagsStorage.push_back(Record.readString());
    for (unsigned i = 0; i != tagsSize; ++i)
      tags.push_back(tagsStorage[i]);
    New = new (Context) AbiTagAttr(Context, Info, tags.data(), tagsSize);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::AcquireCapability: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    unsigned argsSize = Record.readInt();
    SmallVector<Expr *, 4> args;
    args.reserve(argsSize);
    for (unsigned i = 0; i != argsSize; ++i)
      args.push_back(Record.readExpr());
    New = new (Context) AcquireCapabilityAttr(Context, Info, args.data(), argsSize);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::AcquireHandle: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    std::string handleType= Record.readString();
    New = new (Context) AcquireHandleAttr(Context, Info, handleType);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::AcquiredAfter: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    unsigned argsSize = Record.readInt();
    SmallVector<Expr *, 4> args;
    args.reserve(argsSize);
    for (unsigned i = 0; i != argsSize; ++i)
      args.push_back(Record.readExpr());
    New = new (Context) AcquiredAfterAttr(Context, Info, args.data(), argsSize);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::AcquiredBefore: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    unsigned argsSize = Record.readInt();
    SmallVector<Expr *, 4> args;
    args.reserve(argsSize);
    for (unsigned i = 0; i != argsSize; ++i)
      args.push_back(Record.readExpr());
    New = new (Context) AcquiredBeforeAttr(Context, Info, args.data(), argsSize);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::AddressSpace: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    int addressSpace = Record.readInt();
    New = new (Context) AddressSpaceAttr(Context, Info, addressSpace);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Alias: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    std::string aliasee= Record.readString();
    New = new (Context) AliasAttr(Context, Info, aliasee);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::AlignMac68k: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) AlignMac68kAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::AlignNatural: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) AlignNaturalAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::AlignValue: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    Expr * alignment = Record.readExpr();
    New = new (Context) AlignValueAttr(Context, Info, alignment);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Aligned: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    bool isalignmentExpr = Record.readInt();
    void *alignmentPtr;
    if (isalignmentExpr)
      alignmentPtr = Record.readExpr();
    else
      alignmentPtr = Record.readTypeSourceInfo();
    New = new (Context) AlignedAttr(Context, Info, isalignmentExpr, alignmentPtr);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::AllocAlign: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    ParamIdx paramIndex = ParamIdx::deserialize(Record.readInt());
    New = new (Context) AllocAlignAttr(Context, Info, paramIndex);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::AllocSize: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    ParamIdx elemSizeParam = ParamIdx::deserialize(Record.readInt());
    ParamIdx numElemsParam = ParamIdx::deserialize(Record.readInt());
    New = new (Context) AllocSizeAttr(Context, Info, elemSizeParam, numElemsParam);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Allocating: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) AllocatingAttr(Context, Info);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::AlwaysDestroy: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) AlwaysDestroyAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::AlwaysInline: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) AlwaysInlineAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::AnalyzerNoReturn: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) AnalyzerNoReturnAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Annotate: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    unsigned delayedArgsSize = Record.readInt();
    SmallVector<Expr *, 4> delayedArgs;
    delayedArgs.reserve(delayedArgsSize);
    for (unsigned i = 0; i != delayedArgsSize; ++i)
      delayedArgs.push_back(Record.readExpr());
    std::string annotation= Record.readString();
    unsigned argsSize = Record.readInt();
    SmallVector<Expr *, 4> args;
    args.reserve(argsSize);
    for (unsigned i = 0; i != argsSize; ++i)
      args.push_back(Record.readExpr());
    New = new (Context) AnnotateAttr(Context, Info, annotation, args.data(), argsSize);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    cast<AnnotateAttr>(New)->setDelayedArgs(Context, delayedArgs.data(), delayedArgsSize);
    break;
  }
  case attr::AnnotateType: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    unsigned delayedArgsSize = Record.readInt();
    SmallVector<Expr *, 4> delayedArgs;
    delayedArgs.reserve(delayedArgsSize);
    for (unsigned i = 0; i != delayedArgsSize; ++i)
      delayedArgs.push_back(Record.readExpr());
    std::string annotation= Record.readString();
    unsigned argsSize = Record.readInt();
    SmallVector<Expr *, 4> args;
    args.reserve(argsSize);
    for (unsigned i = 0; i != argsSize; ++i)
      args.push_back(Record.readExpr());
    New = new (Context) AnnotateTypeAttr(Context, Info, annotation, args.data(), argsSize);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    cast<AnnotateTypeAttr>(New)->setDelayedArgs(Context, delayedArgs.data(), delayedArgsSize);
    break;
  }
  case attr::AnyX86Interrupt: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) AnyX86InterruptAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::AnyX86NoCallerSavedRegisters: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) AnyX86NoCallerSavedRegistersAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::AnyX86NoCfCheck: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) AnyX86NoCfCheckAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ArcWeakrefUnavailable: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) ArcWeakrefUnavailableAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ArgumentWithTypeTag: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    IdentifierInfo * argumentKind = Record.readIdentifier();
    ParamIdx argumentIdx = ParamIdx::deserialize(Record.readInt());
    ParamIdx typeTagIdx = ParamIdx::deserialize(Record.readInt());
    bool isPointer = Record.readInt();
    New = new (Context) ArgumentWithTypeTagAttr(Context, Info, argumentKind, argumentIdx, typeTagIdx, isPointer);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ArmAgnostic: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    unsigned agnosticArgsSize = Record.readInt();
    SmallVector<StringRef, 4> agnosticArgs;
    agnosticArgs.reserve(agnosticArgsSize);
    SmallVector<std::string, 4> agnosticArgsStorage;
    agnosticArgsStorage.reserve(agnosticArgsSize);
    for (unsigned i = 0; i != agnosticArgsSize; ++i)
      agnosticArgsStorage.push_back(Record.readString());
    for (unsigned i = 0; i != agnosticArgsSize; ++i)
      agnosticArgs.push_back(agnosticArgsStorage[i]);
    New = new (Context) ArmAgnosticAttr(Context, Info, agnosticArgs.data(), agnosticArgsSize);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ArmBuiltinAlias: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    IdentifierInfo * builtinName = Record.readIdentifier();
    New = new (Context) ArmBuiltinAliasAttr(Context, Info, builtinName);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ArmIn: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    unsigned inArgsSize = Record.readInt();
    SmallVector<StringRef, 4> inArgs;
    inArgs.reserve(inArgsSize);
    SmallVector<std::string, 4> inArgsStorage;
    inArgsStorage.reserve(inArgsSize);
    for (unsigned i = 0; i != inArgsSize; ++i)
      inArgsStorage.push_back(Record.readString());
    for (unsigned i = 0; i != inArgsSize; ++i)
      inArgs.push_back(inArgsStorage[i]);
    New = new (Context) ArmInAttr(Context, Info, inArgs.data(), inArgsSize);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ArmInOut: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    unsigned inOutArgsSize = Record.readInt();
    SmallVector<StringRef, 4> inOutArgs;
    inOutArgs.reserve(inOutArgsSize);
    SmallVector<std::string, 4> inOutArgsStorage;
    inOutArgsStorage.reserve(inOutArgsSize);
    for (unsigned i = 0; i != inOutArgsSize; ++i)
      inOutArgsStorage.push_back(Record.readString());
    for (unsigned i = 0; i != inOutArgsSize; ++i)
      inOutArgs.push_back(inOutArgsStorage[i]);
    New = new (Context) ArmInOutAttr(Context, Info, inOutArgs.data(), inOutArgsSize);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ArmLocallyStreaming: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) ArmLocallyStreamingAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ArmMveStrictPolymorphism: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) ArmMveStrictPolymorphismAttr(Context, Info);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ArmNew: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    unsigned newArgsSize = Record.readInt();
    SmallVector<StringRef, 4> newArgs;
    newArgs.reserve(newArgsSize);
    SmallVector<std::string, 4> newArgsStorage;
    newArgsStorage.reserve(newArgsSize);
    for (unsigned i = 0; i != newArgsSize; ++i)
      newArgsStorage.push_back(Record.readString());
    for (unsigned i = 0; i != newArgsSize; ++i)
      newArgs.push_back(newArgsStorage[i]);
    New = new (Context) ArmNewAttr(Context, Info, newArgs.data(), newArgsSize);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ArmOut: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    unsigned outArgsSize = Record.readInt();
    SmallVector<StringRef, 4> outArgs;
    outArgs.reserve(outArgsSize);
    SmallVector<std::string, 4> outArgsStorage;
    outArgsStorage.reserve(outArgsSize);
    for (unsigned i = 0; i != outArgsSize; ++i)
      outArgsStorage.push_back(Record.readString());
    for (unsigned i = 0; i != outArgsSize; ++i)
      outArgs.push_back(outArgsStorage[i]);
    New = new (Context) ArmOutAttr(Context, Info, outArgs.data(), outArgsSize);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ArmPreserves: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    unsigned preserveArgsSize = Record.readInt();
    SmallVector<StringRef, 4> preserveArgs;
    preserveArgs.reserve(preserveArgsSize);
    SmallVector<std::string, 4> preserveArgsStorage;
    preserveArgsStorage.reserve(preserveArgsSize);
    for (unsigned i = 0; i != preserveArgsSize; ++i)
      preserveArgsStorage.push_back(Record.readString());
    for (unsigned i = 0; i != preserveArgsSize; ++i)
      preserveArgs.push_back(preserveArgsStorage[i]);
    New = new (Context) ArmPreservesAttr(Context, Info, preserveArgs.data(), preserveArgsSize);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ArmStreaming: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) ArmStreamingAttr(Context, Info);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ArmStreamingCompatible: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) ArmStreamingCompatibleAttr(Context, Info);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Artificial: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) ArtificialAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::AsmLabel: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    std::string label= Record.readString();
    bool isLiteralLabel = Record.readInt();
    New = new (Context) AsmLabelAttr(Context, Info, label, isLiteralLabel);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::AssertCapability: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    unsigned argsSize = Record.readInt();
    SmallVector<Expr *, 4> args;
    args.reserve(argsSize);
    for (unsigned i = 0; i != argsSize; ++i)
      args.push_back(Record.readExpr());
    New = new (Context) AssertCapabilityAttr(Context, Info, args.data(), argsSize);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::AssumeAligned: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    Expr * alignment = Record.readExpr();
    Expr * offset = Record.readExpr();
    New = new (Context) AssumeAlignedAttr(Context, Info, alignment, offset);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Atomic: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    unsigned atomicOptionsSize = Record.readInt();
    SmallVector<AtomicAttr::ConsumedOption, 4> atomicOptions;
    atomicOptions.reserve(atomicOptionsSize);
    for (unsigned i = atomicOptionsSize; i; --i)
      atomicOptions.push_back(static_cast<AtomicAttr::ConsumedOption>(Record.readInt()));
    New = new (Context) AtomicAttr(Context, Info, atomicOptions.data(), atomicOptionsSize);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Availability: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    IdentifierInfo * platform = Record.readIdentifier();
    VersionTuple introduced= Record.readVersionTuple();
    VersionTuple deprecated= Record.readVersionTuple();
    VersionTuple obsoleted= Record.readVersionTuple();
    bool unavailable = Record.readInt();
    std::string message= Record.readString();
    bool strict = Record.readInt();
    std::string replacement= Record.readString();
    int priority = Record.readInt();
    IdentifierInfo * environment = Record.readIdentifier();
    New = new (Context) AvailabilityAttr(Context, Info, platform, introduced, deprecated, obsoleted, unavailable, message, strict, replacement, priority, environment);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::AvailableOnlyInDefaultEvalMethod: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) AvailableOnlyInDefaultEvalMethodAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::BPFFastCall: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) BPFFastCallAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::BPFPreserveAccessIndex: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) BPFPreserveAccessIndexAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::BPFPreserveStaticOffset: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) BPFPreserveStaticOffsetAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::BTFDeclTag: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    std::string bTFDeclTag= Record.readString();
    New = new (Context) BTFDeclTagAttr(Context, Info, bTFDeclTag);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::BTFTypeTag: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    std::string bTFTypeTag= Record.readString();
    New = new (Context) BTFTypeTagAttr(Context, Info, bTFTypeTag);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Blocking: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) BlockingAttr(Context, Info);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Blocks: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    BlocksAttr::BlockType type(static_cast<BlocksAttr::BlockType>(Record.readInt()));
    New = new (Context) BlocksAttr(Context, Info, type);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Builtin: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    unsigned iD = Record.readInt();
    New = new (Context) BuiltinAttr(Context, Info, iD);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::BuiltinAlias: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    IdentifierInfo * builtinName = Record.readIdentifier();
    New = new (Context) BuiltinAliasAttr(Context, Info, builtinName);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::C11NoReturn: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) C11NoReturnAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::CDecl: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) CDeclAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::CFAuditedTransfer: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) CFAuditedTransferAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::CFConsumed: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) CFConsumedAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::CFGuard: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    CFGuardAttr::GuardArg guard(static_cast<CFGuardAttr::GuardArg>(Record.readInt()));
    New = new (Context) CFGuardAttr(Context, Info, guard);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::CFICanonicalJumpTable: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) CFICanonicalJumpTableAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::CFReturnsNotRetained: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) CFReturnsNotRetainedAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::CFReturnsRetained: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) CFReturnsRetainedAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::CFUnknownTransfer: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) CFUnknownTransferAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::CPUDispatch: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    unsigned cpusSize = Record.readInt();
    SmallVector<IdentifierInfo *, 4> cpus;
    cpus.reserve(cpusSize);
    for (unsigned i = 0; i != cpusSize; ++i)
      cpus.push_back(Record.readIdentifier());
    New = new (Context) CPUDispatchAttr(Context, Info, cpus.data(), cpusSize);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::CPUSpecific: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    unsigned cpusSize = Record.readInt();
    SmallVector<IdentifierInfo *, 4> cpus;
    cpus.reserve(cpusSize);
    for (unsigned i = 0; i != cpusSize; ++i)
      cpus.push_back(Record.readIdentifier());
    New = new (Context) CPUSpecificAttr(Context, Info, cpus.data(), cpusSize);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::CUDAConstant: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) CUDAConstantAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::CUDADevice: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) CUDADeviceAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::CUDADeviceBuiltinSurfaceType: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) CUDADeviceBuiltinSurfaceTypeAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::CUDADeviceBuiltinTextureType: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) CUDADeviceBuiltinTextureTypeAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::CUDAGlobal: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) CUDAGlobalAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::CUDAGridConstant: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) CUDAGridConstantAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::CUDAHost: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) CUDAHostAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::CUDAInvalidTarget: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) CUDAInvalidTargetAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::CUDALaunchBounds: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    Expr * maxThreads = Record.readExpr();
    Expr * minBlocks = Record.readExpr();
    Expr * maxBlocks = Record.readExpr();
    New = new (Context) CUDALaunchBoundsAttr(Context, Info, maxThreads, minBlocks, maxBlocks);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::CUDAShared: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) CUDASharedAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::CXX11NoReturn: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) CXX11NoReturnAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::CXXAssume: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    Expr * assumption = Record.readExpr();
    New = new (Context) CXXAssumeAttr(Context, Info, assumption);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::CallableWhen: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    unsigned callableStatesSize = Record.readInt();
    SmallVector<CallableWhenAttr::ConsumedState, 4> callableStates;
    callableStates.reserve(callableStatesSize);
    for (unsigned i = callableStatesSize; i; --i)
      callableStates.push_back(static_cast<CallableWhenAttr::ConsumedState>(Record.readInt()));
    New = new (Context) CallableWhenAttr(Context, Info, callableStates.data(), callableStatesSize);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Callback: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    unsigned encodingSize = Record.readInt();
    SmallVector<int, 4> encoding;
    encoding.reserve(encodingSize);
    for (unsigned i = 0; i != encodingSize; ++i)
      encoding.push_back(Record.readInt());
    New = new (Context) CallbackAttr(Context, Info, encoding.data(), encodingSize);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::CalledOnce: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) CalledOnceAttr(Context, Info);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Capability: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    std::string name= Record.readString();
    New = new (Context) CapabilityAttr(Context, Info, name);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::CapturedRecord: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) CapturedRecordAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::CarriesDependency: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) CarriesDependencyAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Cleanup: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    FunctionDecl * functionDecl = Record.readDeclAs<FunctionDecl >();
    New = new (Context) CleanupAttr(Context, Info, functionDecl);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ClspvLibclcBuiltin: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) ClspvLibclcBuiltinAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::CmseNSCall: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) CmseNSCallAttr(Context, Info);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::CmseNSEntry: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) CmseNSEntryAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::CodeAlign: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    Expr * alignment = Record.readExpr();
    New = new (Context) CodeAlignAttr(Context, Info, alignment);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::CodeModel: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    llvm::CodeModel::Model model(static_cast<llvm::CodeModel::Model>(Record.readInt()));
    New = new (Context) CodeModelAttr(Context, Info, model);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::CodeSeg: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    std::string name= Record.readString();
    New = new (Context) CodeSegAttr(Context, Info, name);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Cold: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) ColdAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Common: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) CommonAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Const: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) ConstAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ConstInit: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) ConstInitAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Constructor: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    int priority = Record.readInt();
    New = new (Context) ConstructorAttr(Context, Info, priority);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Consumable: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    ConsumableAttr::ConsumedState defaultState(static_cast<ConsumableAttr::ConsumedState>(Record.readInt()));
    New = new (Context) ConsumableAttr(Context, Info, defaultState);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ConsumableAutoCast: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) ConsumableAutoCastAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ConsumableSetOnRead: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) ConsumableSetOnReadAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Convergent: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) ConvergentAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::CoroAwaitElidable: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) CoroAwaitElidableAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::CoroAwaitElidableArgument: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) CoroAwaitElidableArgumentAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::CoroDisableLifetimeBound: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) CoroDisableLifetimeBoundAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::CoroLifetimeBound: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) CoroLifetimeBoundAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::CoroOnlyDestroyWhenComplete: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) CoroOnlyDestroyWhenCompleteAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::CoroReturnType: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) CoroReturnTypeAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::CoroWrapper: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) CoroWrapperAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::CountedBy: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    Expr * count = Record.readExpr();
    int nestedLevel = Record.readInt();
    New = new (Context) CountedByAttr(Context, Info, count, nestedLevel);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::CountedByOrNull: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    Expr * count = Record.readExpr();
    int nestedLevel = Record.readInt();
    New = new (Context) CountedByOrNullAttr(Context, Info, count, nestedLevel);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::DLLExport: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) DLLExportAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::DLLExportStaticLocal: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) DLLExportStaticLocalAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::DLLImport: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) DLLImportAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::DLLImportStaticLocal: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) DLLImportStaticLocalAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Deprecated: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    std::string message= Record.readString();
    std::string replacement= Record.readString();
    New = new (Context) DeprecatedAttr(Context, Info, message, replacement);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Destructor: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    int priority = Record.readInt();
    New = new (Context) DestructorAttr(Context, Info, priority);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::DiagnoseAsBuiltin: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    FunctionDecl * function = Record.readDeclAs<FunctionDecl >();
    unsigned argIndicesSize = Record.readInt();
    SmallVector<unsigned, 4> argIndices;
    argIndices.reserve(argIndicesSize);
    for (unsigned i = 0; i != argIndicesSize; ++i)
      argIndices.push_back(Record.readInt());
    New = new (Context) DiagnoseAsBuiltinAttr(Context, Info, function, argIndices.data(), argIndicesSize);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::DiagnoseIf: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    Expr * cond = Record.readExpr();
    std::string message= Record.readString();
    DiagnoseIfAttr::DefaultSeverity defaultSeverity(static_cast<DiagnoseIfAttr::DefaultSeverity>(Record.readInt()));
    std::string warningGroup= Record.readString();
    bool argDependent = Record.readInt();
    NamedDecl * parent = Record.readDeclAs<NamedDecl >();
    New = new (Context) DiagnoseIfAttr(Context, Info, cond, message, defaultSeverity, warningGroup, argDependent, parent);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::DisableSanitizerInstrumentation: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) DisableSanitizerInstrumentationAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::DisableTailCalls: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) DisableTailCallsAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::EmptyBases: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) EmptyBasesAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::EnableIf: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    Expr * cond = Record.readExpr();
    std::string message= Record.readString();
    New = new (Context) EnableIfAttr(Context, Info, cond, message);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::EnforceTCB: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    std::string tCBName= Record.readString();
    New = new (Context) EnforceTCBAttr(Context, Info, tCBName);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::EnforceTCBLeaf: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    std::string tCBName= Record.readString();
    New = new (Context) EnforceTCBLeafAttr(Context, Info, tCBName);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::EnumExtensibility: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    EnumExtensibilityAttr::Kind extensibility(static_cast<EnumExtensibilityAttr::Kind>(Record.readInt()));
    New = new (Context) EnumExtensibilityAttr(Context, Info, extensibility);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Error: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    std::string userDiagnostic= Record.readString();
    New = new (Context) ErrorAttr(Context, Info, userDiagnostic);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ExcludeFromExplicitInstantiation: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) ExcludeFromExplicitInstantiationAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ExplicitInit: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) ExplicitInitAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ExtVectorType: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    Expr * numElements = Record.readExpr();
    New = new (Context) ExtVectorTypeAttr(Context, Info, numElements);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ExternalSourceSymbol: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    std::string language= Record.readString();
    std::string definedIn= Record.readString();
    bool generatedDeclaration = Record.readInt();
    std::string uSR= Record.readString();
    New = new (Context) ExternalSourceSymbolAttr(Context, Info, language, definedIn, generatedDeclaration, uSR);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::FallThrough: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) FallThroughAttr(Context, Info);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::FastCall: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) FastCallAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Final: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) FinalAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::FlagEnum: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) FlagEnumAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Flatten: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) FlattenAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Format: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    IdentifierInfo * type = Record.readIdentifier();
    int formatIdx = Record.readInt();
    int firstArg = Record.readInt();
    New = new (Context) FormatAttr(Context, Info, type, formatIdx, firstArg);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::FormatArg: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    ParamIdx formatIdx = ParamIdx::deserialize(Record.readInt());
    New = new (Context) FormatArgAttr(Context, Info, formatIdx);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::FormatMatches: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    IdentifierInfo * type = Record.readIdentifier();
    int formatIdx = Record.readInt();
    Expr * expectedFormat = Record.readExpr();
    New = new (Context) FormatMatchesAttr(Context, Info, type, formatIdx, expectedFormat);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::FunctionReturnThunks: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    FunctionReturnThunksAttr::Kind thunkType(static_cast<FunctionReturnThunksAttr::Kind>(Record.readInt()));
    New = new (Context) FunctionReturnThunksAttr(Context, Info, thunkType);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::GNUInline: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) GNUInlineAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::GuardedBy: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    Expr * arg = Record.readExpr();
    New = new (Context) GuardedByAttr(Context, Info, arg);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::GuardedVar: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) GuardedVarAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::HIPManaged: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) HIPManagedAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::HLSLContainedType: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    TypeSourceInfo * type = Record.readTypeSourceInfo();
    New = new (Context) HLSLContainedTypeAttr(Context, Info, type);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::HLSLControlFlowHint: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) HLSLControlFlowHintAttr(Context, Info);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::HLSLGroupSharedAddressSpace: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) HLSLGroupSharedAddressSpaceAttr(Context, Info);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::HLSLLoopHint: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    unsigned directive = Record.readInt();
    New = new (Context) HLSLLoopHintAttr(Context, Info, directive);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::HLSLNumThreads: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    int x = Record.readInt();
    int y = Record.readInt();
    int z = Record.readInt();
    New = new (Context) HLSLNumThreadsAttr(Context, Info, x, y, z);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::HLSLPackOffset: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    int subcomponent = Record.readInt();
    int component = Record.readInt();
    New = new (Context) HLSLPackOffsetAttr(Context, Info, subcomponent, component);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::HLSLParamModifier: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    bool mergedSpelling = Record.readInt();
    New = new (Context) HLSLParamModifierAttr(Context, Info, mergedSpelling);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::HLSLROV: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) HLSLROVAttr(Context, Info);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::HLSLRawBuffer: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) HLSLRawBufferAttr(Context, Info);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::HLSLResourceBinding: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    std::string slot= Record.readString();
    std::string space= Record.readString();
    New = new (Context) HLSLResourceBindingAttr(Context, Info, slot, space);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::HLSLResourceClass: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    llvm::hlsl::ResourceClass resourceClass(static_cast<llvm::hlsl::ResourceClass>(Record.readInt()));
    New = new (Context) HLSLResourceClassAttr(Context, Info, resourceClass);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::HLSLSV_DispatchThreadID: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) HLSLSV_DispatchThreadIDAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::HLSLSV_GroupID: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) HLSLSV_GroupIDAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::HLSLSV_GroupIndex: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) HLSLSV_GroupIndexAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::HLSLSV_GroupThreadID: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) HLSLSV_GroupThreadIDAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::HLSLShader: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    llvm::Triple::EnvironmentType type(static_cast<llvm::Triple::EnvironmentType>(Record.readInt()));
    New = new (Context) HLSLShaderAttr(Context, Info, type);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::HLSLWaveSize: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    int min = Record.readInt();
    int max = Record.readInt();
    int preferred = Record.readInt();
    New = new (Context) HLSLWaveSizeAttr(Context, Info, min, max, preferred);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Hot: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) HotAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::HybridPatchable: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) HybridPatchableAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::IBAction: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) IBActionAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::IBOutlet: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) IBOutletAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::IBOutletCollection: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    TypeSourceInfo * interface_ = Record.readTypeSourceInfo();
    New = new (Context) IBOutletCollectionAttr(Context, Info, interface_);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::IFunc: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    std::string resolver= Record.readString();
    New = new (Context) IFuncAttr(Context, Info, resolver);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::InitPriority: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    unsigned priority = Record.readInt();
    New = new (Context) InitPriorityAttr(Context, Info, priority);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::InitSeg: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    std::string section= Record.readString();
    New = new (Context) InitSegAttr(Context, Info, section);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::IntelOclBicc: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) IntelOclBiccAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::InternalLinkage: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) InternalLinkageAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::LTOVisibilityPublic: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) LTOVisibilityPublicAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::LayoutVersion: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    unsigned version = Record.readInt();
    New = new (Context) LayoutVersionAttr(Context, Info, version);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Leaf: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) LeafAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::LifetimeBound: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) LifetimeBoundAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::LifetimeCaptureBy: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    unsigned paramsSize = Record.readInt();
    SmallVector<int, 4> params;
    params.reserve(paramsSize);
    for (unsigned i = 0; i != paramsSize; ++i)
      params.push_back(Record.readInt());
    New = new (Context) LifetimeCaptureByAttr(Context, Info, params.data(), paramsSize);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Likely: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) LikelyAttr(Context, Info);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::LoaderUninitialized: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) LoaderUninitializedAttr(Context, Info);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::LockReturned: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    Expr * arg = Record.readExpr();
    New = new (Context) LockReturnedAttr(Context, Info, arg);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::LocksExcluded: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    unsigned argsSize = Record.readInt();
    SmallVector<Expr *, 4> args;
    args.reserve(argsSize);
    for (unsigned i = 0; i != argsSize; ++i)
      args.push_back(Record.readExpr());
    New = new (Context) LocksExcludedAttr(Context, Info, args.data(), argsSize);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::LoopHint: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    LoopHintAttr::OptionType option(static_cast<LoopHintAttr::OptionType>(Record.readInt()));
    LoopHintAttr::LoopHintState state(static_cast<LoopHintAttr::LoopHintState>(Record.readInt()));
    Expr * value = Record.readExpr();
    New = new (Context) LoopHintAttr(Context, Info, option, state, value);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::M68kInterrupt: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    unsigned number = Record.readInt();
    New = new (Context) M68kInterruptAttr(Context, Info, number);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::M68kRTD: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) M68kRTDAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::MIGServerRoutine: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) MIGServerRoutineAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::MSABI: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) MSABIAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::MSAllocator: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) MSAllocatorAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::MSConstexpr: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) MSConstexprAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::MSInheritance: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    bool bestCase = Record.readInt();
    New = new (Context) MSInheritanceAttr(Context, Info, bestCase);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::MSNoVTable: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) MSNoVTableAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::MSP430Interrupt: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    unsigned number = Record.readInt();
    New = new (Context) MSP430InterruptAttr(Context, Info, number);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::MSStruct: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) MSStructAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::MSVtorDisp: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    unsigned vdm = Record.readInt();
    New = new (Context) MSVtorDispAttr(Context, Info, vdm);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::MaxFieldAlignment: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    unsigned alignment = Record.readInt();
    New = new (Context) MaxFieldAlignmentAttr(Context, Info, alignment);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::MayAlias: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) MayAliasAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::MaybeUndef: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) MaybeUndefAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::MicroMips: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) MicroMipsAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::MinSize: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) MinSizeAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::MinVectorWidth: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    unsigned vectorWidth = Record.readInt();
    New = new (Context) MinVectorWidthAttr(Context, Info, vectorWidth);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Mips16: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) Mips16Attr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::MipsInterrupt: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    MipsInterruptAttr::InterruptType interrupt(static_cast<MipsInterruptAttr::InterruptType>(Record.readInt()));
    New = new (Context) MipsInterruptAttr(Context, Info, interrupt);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::MipsLongCall: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) MipsLongCallAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::MipsShortCall: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) MipsShortCallAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Mode: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    IdentifierInfo * mode = Record.readIdentifier();
    New = new (Context) ModeAttr(Context, Info, mode);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::MustTail: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) MustTailAttr(Context, Info);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::NSConsumed: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) NSConsumedAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::NSConsumesSelf: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) NSConsumesSelfAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::NSErrorDomain: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    IdentifierInfo * errorDomain = Record.readIdentifier();
    New = new (Context) NSErrorDomainAttr(Context, Info, errorDomain);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::NSReturnsAutoreleased: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) NSReturnsAutoreleasedAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::NSReturnsNotRetained: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) NSReturnsNotRetainedAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::NSReturnsRetained: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) NSReturnsRetainedAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::NVPTXKernel: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) NVPTXKernelAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Naked: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) NakedAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::NoAlias: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) NoAliasAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::NoBuiltin: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    unsigned builtinNamesSize = Record.readInt();
    SmallVector<StringRef, 4> builtinNames;
    builtinNames.reserve(builtinNamesSize);
    SmallVector<std::string, 4> builtinNamesStorage;
    builtinNamesStorage.reserve(builtinNamesSize);
    for (unsigned i = 0; i != builtinNamesSize; ++i)
      builtinNamesStorage.push_back(Record.readString());
    for (unsigned i = 0; i != builtinNamesSize; ++i)
      builtinNames.push_back(builtinNamesStorage[i]);
    New = new (Context) NoBuiltinAttr(Context, Info, builtinNames.data(), builtinNamesSize);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::NoCommon: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) NoCommonAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::NoConvergent: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) NoConvergentAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::NoDebug: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) NoDebugAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::NoDeref: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) NoDerefAttr(Context, Info);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::NoDestroy: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) NoDestroyAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::NoDuplicate: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) NoDuplicateAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::NoEscape: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) NoEscapeAttr(Context, Info);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::NoInline: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) NoInlineAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::NoInstrumentFunction: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) NoInstrumentFunctionAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::NoMerge: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) NoMergeAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::NoMicroMips: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) NoMicroMipsAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::NoMips16: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) NoMips16Attr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::NoProfileFunction: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) NoProfileFunctionAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::NoRandomizeLayout: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) NoRandomizeLayoutAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::NoReturn: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) NoReturnAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::NoSanitize: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    unsigned sanitizersSize = Record.readInt();
    SmallVector<StringRef, 4> sanitizers;
    sanitizers.reserve(sanitizersSize);
    SmallVector<std::string, 4> sanitizersStorage;
    sanitizersStorage.reserve(sanitizersSize);
    for (unsigned i = 0; i != sanitizersSize; ++i)
      sanitizersStorage.push_back(Record.readString());
    for (unsigned i = 0; i != sanitizersSize; ++i)
      sanitizers.push_back(sanitizersStorage[i]);
    New = new (Context) NoSanitizeAttr(Context, Info, sanitizers.data(), sanitizersSize);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::NoSpecializations: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    std::string message= Record.readString();
    New = new (Context) NoSpecializationsAttr(Context, Info, message);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::NoSpeculativeLoadHardening: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) NoSpeculativeLoadHardeningAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::NoSplitStack: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) NoSplitStackAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::NoStackProtector: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) NoStackProtectorAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::NoThreadSafetyAnalysis: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) NoThreadSafetyAnalysisAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::NoThrow: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) NoThrowAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::NoTrivialAutoVarInit: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) NoTrivialAutoVarInitAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::NoUniqueAddress: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) NoUniqueAddressAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::NoUwtable: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) NoUwtableAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::NonAllocating: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    Expr * cond = Record.readExpr();
    New = new (Context) NonAllocatingAttr(Context, Info, cond);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::NonBlocking: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    Expr * cond = Record.readExpr();
    New = new (Context) NonBlockingAttr(Context, Info, cond);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::NonNull: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    unsigned argsSize = Record.readInt();
    SmallVector<ParamIdx, 4> args;
    args.reserve(argsSize);
    for (unsigned i = 0; i != argsSize; ++i)
      args.push_back(ParamIdx::deserialize(Record.readInt()));
    New = new (Context) NonNullAttr(Context, Info, args.data(), argsSize);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::NotTailCalled: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) NotTailCalledAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::OMPAllocateDecl: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    OMPAllocateDeclAttr::AllocatorTypeTy allocatorType(static_cast<OMPAllocateDeclAttr::AllocatorTypeTy>(Record.readInt()));
    Expr * allocator = Record.readExpr();
    Expr * alignment = Record.readExpr();
    New = new (Context) OMPAllocateDeclAttr(Context, Info, allocatorType, allocator, alignment);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::OMPAssume: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    std::string assumption= Record.readString();
    New = new (Context) OMPAssumeAttr(Context, Info, assumption);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::OMPCaptureKind: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    unsigned captureKindVal = Record.readInt();
    New = new (Context) OMPCaptureKindAttr(Context, Info, captureKindVal);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::OMPCaptureNoInit: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) OMPCaptureNoInitAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::OMPDeclareSimdDecl: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    OMPDeclareSimdDeclAttr::BranchStateTy branchState(static_cast<OMPDeclareSimdDeclAttr::BranchStateTy>(Record.readInt()));
    Expr * simdlen = Record.readExpr();
    unsigned uniformsSize = Record.readInt();
    SmallVector<Expr *, 4> uniforms;
    uniforms.reserve(uniformsSize);
    for (unsigned i = 0; i != uniformsSize; ++i)
      uniforms.push_back(Record.readExpr());
    unsigned alignedsSize = Record.readInt();
    SmallVector<Expr *, 4> aligneds;
    aligneds.reserve(alignedsSize);
    for (unsigned i = 0; i != alignedsSize; ++i)
      aligneds.push_back(Record.readExpr());
    unsigned alignmentsSize = Record.readInt();
    SmallVector<Expr *, 4> alignments;
    alignments.reserve(alignmentsSize);
    for (unsigned i = 0; i != alignmentsSize; ++i)
      alignments.push_back(Record.readExpr());
    unsigned linearsSize = Record.readInt();
    SmallVector<Expr *, 4> linears;
    linears.reserve(linearsSize);
    for (unsigned i = 0; i != linearsSize; ++i)
      linears.push_back(Record.readExpr());
    unsigned modifiersSize = Record.readInt();
    SmallVector<unsigned, 4> modifiers;
    modifiers.reserve(modifiersSize);
    for (unsigned i = 0; i != modifiersSize; ++i)
      modifiers.push_back(Record.readInt());
    unsigned stepsSize = Record.readInt();
    SmallVector<Expr *, 4> steps;
    steps.reserve(stepsSize);
    for (unsigned i = 0; i != stepsSize; ++i)
      steps.push_back(Record.readExpr());
    New = new (Context) OMPDeclareSimdDeclAttr(Context, Info, branchState, simdlen, uniforms.data(), uniformsSize, aligneds.data(), alignedsSize, alignments.data(), alignmentsSize, linears.data(), linearsSize, modifiers.data(), modifiersSize, steps.data(), stepsSize);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::OMPDeclareTargetDecl: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    OMPDeclareTargetDeclAttr::MapTypeTy mapType(static_cast<OMPDeclareTargetDeclAttr::MapTypeTy>(Record.readInt()));
    OMPDeclareTargetDeclAttr::DevTypeTy devType(static_cast<OMPDeclareTargetDeclAttr::DevTypeTy>(Record.readInt()));
    Expr * indirectExpr = Record.readExpr();
    bool indirect = Record.readInt();
    unsigned level = Record.readInt();
    New = new (Context) OMPDeclareTargetDeclAttr(Context, Info, mapType, devType, indirectExpr, indirect, level);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::OMPDeclareVariant: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    Expr * variantFuncRef = Record.readExpr();
    OMPTraitInfo * traitInfos = Record.readOMPTraitInfo();
    unsigned adjustArgsNothingSize = Record.readInt();
    SmallVector<Expr *, 4> adjustArgsNothing;
    adjustArgsNothing.reserve(adjustArgsNothingSize);
    for (unsigned i = 0; i != adjustArgsNothingSize; ++i)
      adjustArgsNothing.push_back(Record.readExpr());
    unsigned adjustArgsNeedDevicePtrSize = Record.readInt();
    SmallVector<Expr *, 4> adjustArgsNeedDevicePtr;
    adjustArgsNeedDevicePtr.reserve(adjustArgsNeedDevicePtrSize);
    for (unsigned i = 0; i != adjustArgsNeedDevicePtrSize; ++i)
      adjustArgsNeedDevicePtr.push_back(Record.readExpr());
    unsigned appendArgsSize = Record.readInt();
    SmallVector<OMPInteropInfo, 4> appendArgs;
    appendArgs.reserve(appendArgsSize);
    for (unsigned I = 0, E = appendArgsSize; I != E; ++I) {
      bool IsTarget = Record.readBool();
      bool IsTargetSync = Record.readBool();
      appendArgs.emplace_back(IsTarget, IsTargetSync);
    }
    New = new (Context) OMPDeclareVariantAttr(Context, Info, variantFuncRef, traitInfos, adjustArgsNothing.data(), adjustArgsNothingSize, adjustArgsNeedDevicePtr.data(), adjustArgsNeedDevicePtrSize, appendArgs.data(), appendArgsSize);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::OMPReferencedVar: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    Expr * ref = Record.readExpr();
    New = new (Context) OMPReferencedVarAttr(Context, Info, ref);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::OMPThreadPrivateDecl: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) OMPThreadPrivateDeclAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::OSConsumed: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) OSConsumedAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::OSConsumesThis: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) OSConsumesThisAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::OSReturnsNotRetained: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) OSReturnsNotRetainedAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::OSReturnsRetained: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) OSReturnsRetainedAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::OSReturnsRetainedOnNonZero: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) OSReturnsRetainedOnNonZeroAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::OSReturnsRetainedOnZero: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) OSReturnsRetainedOnZeroAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ObjCBoxable: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) ObjCBoxableAttr(Context, Info);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ObjCBridge: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    IdentifierInfo * bridgedType = Record.readIdentifier();
    New = new (Context) ObjCBridgeAttr(Context, Info, bridgedType);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ObjCBridgeMutable: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    IdentifierInfo * bridgedType = Record.readIdentifier();
    New = new (Context) ObjCBridgeMutableAttr(Context, Info, bridgedType);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ObjCBridgeRelated: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    IdentifierInfo * relatedClass = Record.readIdentifier();
    IdentifierInfo * classMethod = Record.readIdentifier();
    IdentifierInfo * instanceMethod = Record.readIdentifier();
    New = new (Context) ObjCBridgeRelatedAttr(Context, Info, relatedClass, classMethod, instanceMethod);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ObjCClassStub: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) ObjCClassStubAttr(Context, Info);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ObjCDesignatedInitializer: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) ObjCDesignatedInitializerAttr(Context, Info);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ObjCDirect: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) ObjCDirectAttr(Context, Info);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ObjCDirectMembers: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) ObjCDirectMembersAttr(Context, Info);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ObjCException: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) ObjCExceptionAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ObjCExplicitProtocolImpl: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) ObjCExplicitProtocolImplAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ObjCExternallyRetained: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) ObjCExternallyRetainedAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ObjCGC: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    IdentifierInfo * kind = Record.readIdentifier();
    New = new (Context) ObjCGCAttr(Context, Info, kind);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ObjCIndependentClass: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) ObjCIndependentClassAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ObjCInertUnsafeUnretained: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) ObjCInertUnsafeUnretainedAttr(Context, Info);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ObjCKindOf: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) ObjCKindOfAttr(Context, Info);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ObjCMethodFamily: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    ObjCMethodFamilyAttr::FamilyKind family(static_cast<ObjCMethodFamilyAttr::FamilyKind>(Record.readInt()));
    New = new (Context) ObjCMethodFamilyAttr(Context, Info, family);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ObjCNSObject: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) ObjCNSObjectAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ObjCNonLazyClass: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) ObjCNonLazyClassAttr(Context, Info);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ObjCNonRuntimeProtocol: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) ObjCNonRuntimeProtocolAttr(Context, Info);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ObjCOwnership: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    IdentifierInfo * kind = Record.readIdentifier();
    New = new (Context) ObjCOwnershipAttr(Context, Info, kind);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ObjCPreciseLifetime: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) ObjCPreciseLifetimeAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ObjCRequiresPropertyDefs: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) ObjCRequiresPropertyDefsAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ObjCRequiresSuper: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) ObjCRequiresSuperAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ObjCReturnsInnerPointer: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) ObjCReturnsInnerPointerAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ObjCRootClass: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) ObjCRootClassAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ObjCRuntimeName: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    std::string metadataName= Record.readString();
    New = new (Context) ObjCRuntimeNameAttr(Context, Info, metadataName);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ObjCRuntimeVisible: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) ObjCRuntimeVisibleAttr(Context, Info);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ObjCSubclassingRestricted: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) ObjCSubclassingRestrictedAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::OpenACCRoutineAnnot: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) OpenACCRoutineAnnotAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::OpenACCRoutineDecl: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) OpenACCRoutineDeclAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    readOpenACCRoutineDeclAttr(cast<OpenACCRoutineDeclAttr>(New));
    break;
  }
  case attr::OpenCLAccess: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) OpenCLAccessAttr(Context, Info);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::OpenCLConstantAddressSpace: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) OpenCLConstantAddressSpaceAttr(Context, Info);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::OpenCLGenericAddressSpace: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) OpenCLGenericAddressSpaceAttr(Context, Info);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::OpenCLGlobalAddressSpace: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) OpenCLGlobalAddressSpaceAttr(Context, Info);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::OpenCLGlobalDeviceAddressSpace: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) OpenCLGlobalDeviceAddressSpaceAttr(Context, Info);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::OpenCLGlobalHostAddressSpace: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) OpenCLGlobalHostAddressSpaceAttr(Context, Info);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::OpenCLIntelReqdSubGroupSize: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    unsigned subGroupSize = Record.readInt();
    New = new (Context) OpenCLIntelReqdSubGroupSizeAttr(Context, Info, subGroupSize);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::OpenCLKernel: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) OpenCLKernelAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::OpenCLLocalAddressSpace: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) OpenCLLocalAddressSpaceAttr(Context, Info);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::OpenCLPrivateAddressSpace: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) OpenCLPrivateAddressSpaceAttr(Context, Info);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::OpenCLUnrollHint: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    unsigned unrollHint = Record.readInt();
    New = new (Context) OpenCLUnrollHintAttr(Context, Info, unrollHint);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::OptimizeNone: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) OptimizeNoneAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Overloadable: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) OverloadableAttr(Context, Info);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Override: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) OverrideAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Owner: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    TypeSourceInfo * derefType = Record.readTypeSourceInfo();
    New = new (Context) OwnerAttr(Context, Info, derefType);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Ownership: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    IdentifierInfo * module = Record.readIdentifier();
    unsigned argsSize = Record.readInt();
    SmallVector<ParamIdx, 4> args;
    args.reserve(argsSize);
    for (unsigned i = 0; i != argsSize; ++i)
      args.push_back(ParamIdx::deserialize(Record.readInt()));
    New = new (Context) OwnershipAttr(Context, Info, module, args.data(), argsSize);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Packed: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) PackedAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ParamTypestate: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    ParamTypestateAttr::ConsumedState paramState(static_cast<ParamTypestateAttr::ConsumedState>(Record.readInt()));
    New = new (Context) ParamTypestateAttr(Context, Info, paramState);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Pascal: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) PascalAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::PassObjectSize: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    int type = Record.readInt();
    New = new (Context) PassObjectSizeAttr(Context, Info, type);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::PatchableFunctionEntry: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    unsigned count = Record.readInt();
    int offset = Record.readInt();
    std::string section= Record.readString();
    New = new (Context) PatchableFunctionEntryAttr(Context, Info, count, offset, section);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Pcs: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    PcsAttr::PCSType pCS(static_cast<PcsAttr::PCSType>(Record.readInt()));
    New = new (Context) PcsAttr(Context, Info, pCS);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Pointer: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    TypeSourceInfo * derefType = Record.readTypeSourceInfo();
    New = new (Context) PointerAttr(Context, Info, derefType);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::PointerAuth: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    int key = Record.readInt();
    bool addressDiscriminated = Record.readInt();
    int extraDiscriminator = Record.readInt();
    New = new (Context) PointerAuthAttr(Context, Info, key, addressDiscriminated, extraDiscriminator);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::PragmaClangBSSSection: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    std::string name= Record.readString();
    New = new (Context) PragmaClangBSSSectionAttr(Context, Info, name);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::PragmaClangDataSection: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    std::string name= Record.readString();
    New = new (Context) PragmaClangDataSectionAttr(Context, Info, name);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::PragmaClangRelroSection: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    std::string name= Record.readString();
    New = new (Context) PragmaClangRelroSectionAttr(Context, Info, name);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::PragmaClangRodataSection: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    std::string name= Record.readString();
    New = new (Context) PragmaClangRodataSectionAttr(Context, Info, name);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::PragmaClangTextSection: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    std::string name= Record.readString();
    New = new (Context) PragmaClangTextSectionAttr(Context, Info, name);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::PreferredName: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    TypeSourceInfo * typedefType = Record.readTypeSourceInfo();
    New = new (Context) PreferredNameAttr(Context, Info, typedefType);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::PreferredType: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    TypeSourceInfo * type = Record.readTypeSourceInfo();
    New = new (Context) PreferredTypeAttr(Context, Info, type);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::PreserveAll: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) PreserveAllAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::PreserveMost: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) PreserveMostAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::PreserveNone: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) PreserveNoneAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::PtGuardedBy: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    Expr * arg = Record.readExpr();
    New = new (Context) PtGuardedByAttr(Context, Info, arg);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::PtGuardedVar: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) PtGuardedVarAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Ptr32: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) Ptr32Attr(Context, Info);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Ptr64: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) Ptr64Attr(Context, Info);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Pure: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) PureAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::RISCVInterrupt: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    RISCVInterruptAttr::InterruptType interrupt(static_cast<RISCVInterruptAttr::InterruptType>(Record.readInt()));
    New = new (Context) RISCVInterruptAttr(Context, Info, interrupt);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::RISCVVLSCC: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    unsigned vectorWidth = Record.readInt();
    New = new (Context) RISCVVLSCCAttr(Context, Info, vectorWidth);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::RISCVVectorCC: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) RISCVVectorCCAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::RandomizeLayout: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) RandomizeLayoutAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ReadOnlyPlacement: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) ReadOnlyPlacementAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::RegCall: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) RegCallAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Reinitializes: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) ReinitializesAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ReleaseCapability: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    unsigned argsSize = Record.readInt();
    SmallVector<Expr *, 4> args;
    args.reserve(argsSize);
    for (unsigned i = 0; i != argsSize; ++i)
      args.push_back(Record.readExpr());
    New = new (Context) ReleaseCapabilityAttr(Context, Info, args.data(), argsSize);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ReleaseHandle: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    std::string handleType= Record.readString();
    New = new (Context) ReleaseHandleAttr(Context, Info, handleType);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ReqdWorkGroupSize: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    Expr * xDim = Record.readExpr();
    Expr * yDim = Record.readExpr();
    Expr * zDim = Record.readExpr();
    New = new (Context) ReqdWorkGroupSizeAttr(Context, Info, xDim, yDim, zDim);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::RequiresCapability: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    unsigned argsSize = Record.readInt();
    SmallVector<Expr *, 4> args;
    args.reserve(argsSize);
    for (unsigned i = 0; i != argsSize; ++i)
      args.push_back(Record.readExpr());
    New = new (Context) RequiresCapabilityAttr(Context, Info, args.data(), argsSize);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Restrict: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    Expr * deallocator = Record.readExpr();
    ParamIdx deallocatorPtrArgIndex = ParamIdx::deserialize(Record.readInt());
    New = new (Context) RestrictAttr(Context, Info, deallocator, deallocatorPtrArgIndex);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Retain: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) RetainAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ReturnTypestate: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    ReturnTypestateAttr::ConsumedState state(static_cast<ReturnTypestateAttr::ConsumedState>(Record.readInt()));
    New = new (Context) ReturnTypestateAttr(Context, Info, state);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ReturnsNonNull: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) ReturnsNonNullAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ReturnsTwice: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) ReturnsTwiceAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::SPtr: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) SPtrAttr(Context, Info);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::SYCLKernel: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) SYCLKernelAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::SYCLKernelEntryPoint: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    TypeSourceInfo * kernelName = Record.readTypeSourceInfo();
    bool invalid = Record.readInt();
    New = new (Context) SYCLKernelEntryPointAttr(Context, Info, kernelName, invalid);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::SYCLSpecialClass: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) SYCLSpecialClassAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ScopedLockable: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) ScopedLockableAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Section: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    std::string name= Record.readString();
    New = new (Context) SectionAttr(Context, Info, name);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::SelectAny: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) SelectAnyAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Sentinel: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    int sentinel = Record.readInt();
    int nullPos = Record.readInt();
    New = new (Context) SentinelAttr(Context, Info, sentinel, nullPos);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::SetTypestate: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    SetTypestateAttr::ConsumedState newState(static_cast<SetTypestateAttr::ConsumedState>(Record.readInt()));
    New = new (Context) SetTypestateAttr(Context, Info, newState);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::SizedBy: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    Expr * size = Record.readExpr();
    int nestedLevel = Record.readInt();
    New = new (Context) SizedByAttr(Context, Info, size, nestedLevel);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::SizedByOrNull: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    Expr * size = Record.readExpr();
    int nestedLevel = Record.readInt();
    New = new (Context) SizedByOrNullAttr(Context, Info, size, nestedLevel);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::SpeculativeLoadHardening: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) SpeculativeLoadHardeningAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::StandaloneDebug: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) StandaloneDebugAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::StdCall: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) StdCallAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::StrictFP: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) StrictFPAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::StrictGuardStackCheck: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) StrictGuardStackCheckAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Suppress: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    unsigned diagnosticIdentifiersSize = Record.readInt();
    SmallVector<StringRef, 4> diagnosticIdentifiers;
    diagnosticIdentifiers.reserve(diagnosticIdentifiersSize);
    SmallVector<std::string, 4> diagnosticIdentifiersStorage;
    diagnosticIdentifiersStorage.reserve(diagnosticIdentifiersSize);
    for (unsigned i = 0; i != diagnosticIdentifiersSize; ++i)
      diagnosticIdentifiersStorage.push_back(Record.readString());
    for (unsigned i = 0; i != diagnosticIdentifiersSize; ++i)
      diagnosticIdentifiers.push_back(diagnosticIdentifiersStorage[i]);
    New = new (Context) SuppressAttr(Context, Info, diagnosticIdentifiers.data(), diagnosticIdentifiersSize);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::SwiftAsync: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    SwiftAsyncAttr::Kind kind(static_cast<SwiftAsyncAttr::Kind>(Record.readInt()));
    ParamIdx completionHandlerIndex = ParamIdx::deserialize(Record.readInt());
    New = new (Context) SwiftAsyncAttr(Context, Info, kind, completionHandlerIndex);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::SwiftAsyncCall: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) SwiftAsyncCallAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::SwiftAsyncContext: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) SwiftAsyncContextAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::SwiftAsyncError: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    SwiftAsyncErrorAttr::ConventionKind convention(static_cast<SwiftAsyncErrorAttr::ConventionKind>(Record.readInt()));
    unsigned handlerParamIdx = Record.readInt();
    New = new (Context) SwiftAsyncErrorAttr(Context, Info, convention, handlerParamIdx);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::SwiftAsyncName: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    std::string name= Record.readString();
    New = new (Context) SwiftAsyncNameAttr(Context, Info, name);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::SwiftAttr: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    std::string attribute= Record.readString();
    New = new (Context) SwiftAttrAttr(Context, Info, attribute);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::SwiftBridge: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    std::string swiftType= Record.readString();
    New = new (Context) SwiftBridgeAttr(Context, Info, swiftType);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::SwiftBridgedTypedef: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) SwiftBridgedTypedefAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::SwiftCall: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) SwiftCallAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::SwiftContext: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) SwiftContextAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::SwiftError: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    SwiftErrorAttr::ConventionKind convention(static_cast<SwiftErrorAttr::ConventionKind>(Record.readInt()));
    New = new (Context) SwiftErrorAttr(Context, Info, convention);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::SwiftErrorResult: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) SwiftErrorResultAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::SwiftImportAsNonGeneric: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) SwiftImportAsNonGenericAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::SwiftImportPropertyAsAccessors: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) SwiftImportPropertyAsAccessorsAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::SwiftIndirectResult: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) SwiftIndirectResultAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::SwiftName: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    std::string name= Record.readString();
    New = new (Context) SwiftNameAttr(Context, Info, name);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::SwiftNewType: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    SwiftNewTypeAttr::NewtypeKind newtypeKind(static_cast<SwiftNewTypeAttr::NewtypeKind>(Record.readInt()));
    New = new (Context) SwiftNewTypeAttr(Context, Info, newtypeKind);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::SwiftObjCMembers: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) SwiftObjCMembersAttr(Context, Info);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::SwiftPrivate: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) SwiftPrivateAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::SwiftVersionedAddition: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    VersionTuple version= Record.readVersionTuple();
    Attr *additionalAttr = Record.readAttr();    bool isReplacedByActive = Record.readInt();
    New = new (Context) SwiftVersionedAdditionAttr(Context, Info, version, additionalAttr, isReplacedByActive);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::SwiftVersionedRemoval: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    VersionTuple version= Record.readVersionTuple();
    unsigned rawKind = Record.readInt();
    bool isReplacedByActive = Record.readInt();
    New = new (Context) SwiftVersionedRemovalAttr(Context, Info, version, rawKind, isReplacedByActive);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::SysVABI: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) SysVABIAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::TLSModel: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    std::string model= Record.readString();
    New = new (Context) TLSModelAttr(Context, Info, model);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Target: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    std::string featuresStr= Record.readString();
    New = new (Context) TargetAttr(Context, Info, featuresStr);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::TargetClones: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    unsigned featuresStrsSize = Record.readInt();
    SmallVector<StringRef, 4> featuresStrs;
    featuresStrs.reserve(featuresStrsSize);
    SmallVector<std::string, 4> featuresStrsStorage;
    featuresStrsStorage.reserve(featuresStrsSize);
    for (unsigned i = 0; i != featuresStrsSize; ++i)
      featuresStrsStorage.push_back(Record.readString());
    for (unsigned i = 0; i != featuresStrsSize; ++i)
      featuresStrs.push_back(featuresStrsStorage[i]);
    New = new (Context) TargetClonesAttr(Context, Info, featuresStrs.data(), featuresStrsSize);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::TargetVersion: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    std::string namesStr= Record.readString();
    New = new (Context) TargetVersionAttr(Context, Info, namesStr);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::TestTypestate: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    TestTypestateAttr::ConsumedState testState(static_cast<TestTypestateAttr::ConsumedState>(Record.readInt()));
    New = new (Context) TestTypestateAttr(Context, Info, testState);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ThisCall: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) ThisCallAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Thread: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) ThreadAttr(Context, Info);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::TransparentUnion: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) TransparentUnionAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::TrivialABI: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) TrivialABIAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::TryAcquireCapability: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    Expr * successValue = Record.readExpr();
    unsigned argsSize = Record.readInt();
    SmallVector<Expr *, 4> args;
    args.reserve(argsSize);
    for (unsigned i = 0; i != argsSize; ++i)
      args.push_back(Record.readExpr());
    New = new (Context) TryAcquireCapabilityAttr(Context, Info, successValue, args.data(), argsSize);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::TypeNonNull: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) TypeNonNullAttr(Context, Info);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::TypeNullUnspecified: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) TypeNullUnspecifiedAttr(Context, Info);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::TypeNullable: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) TypeNullableAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::TypeNullableResult: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) TypeNullableResultAttr(Context, Info);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::TypeTagForDatatype: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    IdentifierInfo * argumentKind = Record.readIdentifier();
    TypeSourceInfo * matchingCType = Record.readTypeSourceInfo();
    bool layoutCompatible = Record.readInt();
    bool mustBeNull = Record.readInt();
    New = new (Context) TypeTagForDatatypeAttr(Context, Info, argumentKind, matchingCType, layoutCompatible, mustBeNull);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::TypeVisibility: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    TypeVisibilityAttr::VisibilityType visibility(static_cast<TypeVisibilityAttr::VisibilityType>(Record.readInt()));
    New = new (Context) TypeVisibilityAttr(Context, Info, visibility);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::UPtr: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) UPtrAttr(Context, Info);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Unavailable: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    std::string message= Record.readString();
    UnavailableAttr::ImplicitReason implicitReason(static_cast<UnavailableAttr::ImplicitReason>(Record.readInt()));
    New = new (Context) UnavailableAttr(Context, Info, message, implicitReason);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Uninitialized: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) UninitializedAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Unlikely: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) UnlikelyAttr(Context, Info);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::UnsafeBufferUsage: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) UnsafeBufferUsageAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Unused: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) UnusedAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::UseHandle: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    std::string handleType= Record.readString();
    New = new (Context) UseHandleAttr(Context, Info, handleType);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Used: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) UsedAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::UsingIfExists: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) UsingIfExistsAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Uuid: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    std::string guid= Record.readString();
    MSGuidDecl * guidDecl = Record.readDeclAs<MSGuidDecl >();
    New = new (Context) UuidAttr(Context, Info, guid, guidDecl);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::VTablePointerAuthentication: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    VTablePointerAuthenticationAttr::VPtrAuthKeyType key(static_cast<VTablePointerAuthenticationAttr::VPtrAuthKeyType>(Record.readInt()));
    VTablePointerAuthenticationAttr::AddressDiscriminationMode addressDiscrimination(static_cast<VTablePointerAuthenticationAttr::AddressDiscriminationMode>(Record.readInt()));
    VTablePointerAuthenticationAttr::ExtraDiscrimination extraDiscrimination(static_cast<VTablePointerAuthenticationAttr::ExtraDiscrimination>(Record.readInt()));
    int customDiscriminationValue = Record.readInt();
    New = new (Context) VTablePointerAuthenticationAttr(Context, Info, key, addressDiscrimination, extraDiscrimination, customDiscriminationValue);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::VecReturn: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) VecReturnAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::VecTypeHint: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    TypeSourceInfo * typeHint = Record.readTypeSourceInfo();
    New = new (Context) VecTypeHintAttr(Context, Info, typeHint);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::VectorCall: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) VectorCallAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Visibility: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    VisibilityAttr::VisibilityType visibility(static_cast<VisibilityAttr::VisibilityType>(Record.readInt()));
    New = new (Context) VisibilityAttr(Context, Info, visibility);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::WarnUnused: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) WarnUnusedAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::WarnUnusedResult: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    std::string message= Record.readString();
    New = new (Context) WarnUnusedResultAttr(Context, Info, message);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::Weak: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) WeakAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::WeakImport: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) WeakImportAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::WeakRef: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    std::string aliasee= Record.readString();
    New = new (Context) WeakRefAttr(Context, Info, aliasee);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::WebAssemblyExportName: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    std::string exportName= Record.readString();
    New = new (Context) WebAssemblyExportNameAttr(Context, Info, exportName);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::WebAssemblyFuncref: {
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) WebAssemblyFuncrefAttr(Context, Info);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::WebAssemblyImportModule: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    std::string importModule= Record.readString();
    New = new (Context) WebAssemblyImportModuleAttr(Context, Info, importModule);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::WebAssemblyImportName: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    std::string importName= Record.readString();
    New = new (Context) WebAssemblyImportNameAttr(Context, Info, importName);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::WorkGroupSizeHint: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    Expr * xDim = Record.readExpr();
    Expr * yDim = Record.readExpr();
    Expr * zDim = Record.readExpr();
    New = new (Context) WorkGroupSizeHintAttr(Context, Info, xDim, yDim, zDim);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::X86ForceAlignArgPointer: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) X86ForceAlignArgPointerAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::XRayInstrument: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    New = new (Context) XRayInstrumentAttr(Context, Info);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::XRayLogArgs: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    unsigned argumentCount = Record.readInt();
    New = new (Context) XRayLogArgsAttr(Context, Info, argumentCount);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  case attr::ZeroCallUsedRegs: {
    bool isInherited = Record.readInt();
    bool isImplicit = Record.readInt();
    bool isPackExpansion = Record.readInt();
    ZeroCallUsedRegsAttr::ZeroCallUsedRegsKind zeroCallUsedRegs(static_cast<ZeroCallUsedRegsAttr::ZeroCallUsedRegsKind>(Record.readInt()));
    New = new (Context) ZeroCallUsedRegsAttr(Context, Info, zeroCallUsedRegs);
    cast<InheritableAttr>(New)->setInherited(isInherited);
    New->setImplicit(isImplicit);
    New->setPackExpansion(isPackExpansion);
    break;
  }
  }
