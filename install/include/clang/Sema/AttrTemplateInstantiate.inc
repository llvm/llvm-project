/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Template instantiation code for attributes                                 *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: Attr.td                                                              *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace clang {
namespace sema {

Attr *instantiateTemplateAttribute(const Attr *At, ASTContext &C, Sema &S,
        const MultiLevelTemplateArgumentList &TemplateArgs) {
  switch (At->getKind()) {
    case attr::AArch64SVEPcs: {
      const auto *A = cast<AArch64SVEPcsAttr>(At);
      return A->clone(C);
    }
    case attr::AArch64VectorPcs: {
      const auto *A = cast<AArch64VectorPcsAttr>(At);
      return A->clone(C);
    }
    case attr::AMDGPUFlatWorkGroupSize: {
      const auto *A = cast<AMDGPUFlatWorkGroupSizeAttr>(At);
      return A->clone(C);
    }
    case attr::AMDGPUKernelCall: {
      const auto *A = cast<AMDGPUKernelCallAttr>(At);
      return A->clone(C);
    }
    case attr::AMDGPUMaxNumWorkGroups: {
      const auto *A = cast<AMDGPUMaxNumWorkGroupsAttr>(At);
      return A->clone(C);
    }
    case attr::AMDGPUNumSGPR: {
      const auto *A = cast<AMDGPUNumSGPRAttr>(At);
      return A->clone(C);
    }
    case attr::AMDGPUNumVGPR: {
      const auto *A = cast<AMDGPUNumVGPRAttr>(At);
      return A->clone(C);
    }
    case attr::AMDGPUWavesPerEU: {
      const auto *A = cast<AMDGPUWavesPerEUAttr>(At);
      return A->clone(C);
    }
    case attr::ARMInterrupt: {
      const auto *A = cast<ARMInterruptAttr>(At);
      return A->clone(C);
    }
    case attr::ARMInterruptSaveFP: {
      const auto *A = cast<ARMInterruptSaveFPAttr>(At);
      return A->clone(C);
    }
    case attr::ARMSaveFP: {
      const auto *A = cast<ARMSaveFPAttr>(At);
      return A->clone(C);
    }
    case attr::AVRInterrupt: {
      const auto *A = cast<AVRInterruptAttr>(At);
      return A->clone(C);
    }
    case attr::AVRSignal: {
      const auto *A = cast<AVRSignalAttr>(At);
      return A->clone(C);
    }
    case attr::AbiTag: {
      const auto *A = cast<AbiTagAttr>(At);
      return A->clone(C);
    }
    case attr::AcquireCapability: {
      const auto *A = cast<AcquireCapabilityAttr>(At);
      auto *tempInstArgs = new (C, 16) Expr *[A->args_size()];
      {
        EnterExpressionEvaluationContext Unevaluated(S, Sema::ExpressionEvaluationContext::Unevaluated);
        Expr * *TI = tempInstArgs;
        Expr * *I = A->args_begin();
        Expr * *E = A->args_end();
        for (; I != E; ++I, ++TI) {
          ExprResult Result = S.SubstExpr(*I, TemplateArgs);
          if (Result.isInvalid())
            return nullptr;
          *TI = Result.get();
        }
      }
      return new (C) AcquireCapabilityAttr(C, *A, tempInstArgs, A->args_size());
    }
    case attr::AcquireHandle: {
      const auto *A = cast<AcquireHandleAttr>(At);
      return A->clone(C);
    }
    case attr::AcquiredAfter: {
      const auto *A = cast<AcquiredAfterAttr>(At);
      auto *tempInstArgs = new (C, 16) Expr *[A->args_size()];
      {
        EnterExpressionEvaluationContext Unevaluated(S, Sema::ExpressionEvaluationContext::Unevaluated);
        Expr * *TI = tempInstArgs;
        Expr * *I = A->args_begin();
        Expr * *E = A->args_end();
        for (; I != E; ++I, ++TI) {
          ExprResult Result = S.SubstExpr(*I, TemplateArgs);
          if (Result.isInvalid())
            return nullptr;
          *TI = Result.get();
        }
      }
      return new (C) AcquiredAfterAttr(C, *A, tempInstArgs, A->args_size());
    }
    case attr::AcquiredBefore: {
      const auto *A = cast<AcquiredBeforeAttr>(At);
      auto *tempInstArgs = new (C, 16) Expr *[A->args_size()];
      {
        EnterExpressionEvaluationContext Unevaluated(S, Sema::ExpressionEvaluationContext::Unevaluated);
        Expr * *TI = tempInstArgs;
        Expr * *I = A->args_begin();
        Expr * *E = A->args_end();
        for (; I != E; ++I, ++TI) {
          ExprResult Result = S.SubstExpr(*I, TemplateArgs);
          if (Result.isInvalid())
            return nullptr;
          *TI = Result.get();
        }
      }
      return new (C) AcquiredBeforeAttr(C, *A, tempInstArgs, A->args_size());
    }
    case attr::AddressSpace: {
      const auto *A = cast<AddressSpaceAttr>(At);
      return A->clone(C);
    }
    case attr::Alias: {
      const auto *A = cast<AliasAttr>(At);
      return A->clone(C);
    }
    case attr::AlignMac68k: {
      const auto *A = cast<AlignMac68kAttr>(At);
      return A->clone(C);
    }
    case attr::AlignNatural: {
      const auto *A = cast<AlignNaturalAttr>(At);
      return A->clone(C);
    }
    case attr::AlignValue: {
      const auto *A = cast<AlignValueAttr>(At);
      return A->clone(C);
    }
    case attr::Aligned: {
      const auto *A = cast<AlignedAttr>(At);
      return A->clone(C);
    }
    case attr::AllocAlign: {
      const auto *A = cast<AllocAlignAttr>(At);
      return A->clone(C);
    }
    case attr::AllocSize: {
      const auto *A = cast<AllocSizeAttr>(At);
      return new (C) AllocSizeAttr(C, *A, A->getElemSizeParam(), A->getNumElemsParam());
    }
    case attr::Allocating: {
      const auto *A = cast<AllocatingAttr>(At);
      return A->clone(C);
    }
    case attr::AlwaysDestroy: {
      const auto *A = cast<AlwaysDestroyAttr>(At);
      return A->clone(C);
    }
    case attr::AlwaysInline: {
      const auto *A = cast<AlwaysInlineAttr>(At);
      return A->clone(C);
    }
    case attr::AnalyzerNoReturn: {
      const auto *A = cast<AnalyzerNoReturnAttr>(At);
      return A->clone(C);
    }
    case attr::Annotate: {
      const auto *A = cast<AnnotateAttr>(At);
      return A->clone(C);
    }
    case attr::AnnotateType: {
      const auto *A = cast<AnnotateTypeAttr>(At);
      return A->clone(C);
    }
    case attr::AnyX86Interrupt: {
      const auto *A = cast<AnyX86InterruptAttr>(At);
      return A->clone(C);
    }
    case attr::AnyX86NoCallerSavedRegisters: {
      const auto *A = cast<AnyX86NoCallerSavedRegistersAttr>(At);
      return A->clone(C);
    }
    case attr::AnyX86NoCfCheck: {
      const auto *A = cast<AnyX86NoCfCheckAttr>(At);
      return A->clone(C);
    }
    case attr::ArcWeakrefUnavailable: {
      const auto *A = cast<ArcWeakrefUnavailableAttr>(At);
      return A->clone(C);
    }
    case attr::ArgumentWithTypeTag: {
      const auto *A = cast<ArgumentWithTypeTagAttr>(At);
      return A->clone(C);
    }
    case attr::ArmAgnostic: {
      const auto *A = cast<ArmAgnosticAttr>(At);
      return A->clone(C);
    }
    case attr::ArmBuiltinAlias: {
      const auto *A = cast<ArmBuiltinAliasAttr>(At);
      return A->clone(C);
    }
    case attr::ArmIn: {
      const auto *A = cast<ArmInAttr>(At);
      return A->clone(C);
    }
    case attr::ArmInOut: {
      const auto *A = cast<ArmInOutAttr>(At);
      return A->clone(C);
    }
    case attr::ArmLocallyStreaming: {
      const auto *A = cast<ArmLocallyStreamingAttr>(At);
      return A->clone(C);
    }
    case attr::ArmMveStrictPolymorphism: {
      const auto *A = cast<ArmMveStrictPolymorphismAttr>(At);
      return A->clone(C);
    }
    case attr::ArmNew: {
      const auto *A = cast<ArmNewAttr>(At);
      return A->clone(C);
    }
    case attr::ArmOut: {
      const auto *A = cast<ArmOutAttr>(At);
      return A->clone(C);
    }
    case attr::ArmPreserves: {
      const auto *A = cast<ArmPreservesAttr>(At);
      return A->clone(C);
    }
    case attr::ArmStreaming: {
      const auto *A = cast<ArmStreamingAttr>(At);
      return A->clone(C);
    }
    case attr::ArmStreamingCompatible: {
      const auto *A = cast<ArmStreamingCompatibleAttr>(At);
      return A->clone(C);
    }
    case attr::Artificial: {
      const auto *A = cast<ArtificialAttr>(At);
      return A->clone(C);
    }
    case attr::AsmLabel: {
      const auto *A = cast<AsmLabelAttr>(At);
      return A->clone(C);
    }
    case attr::AssertCapability: {
      const auto *A = cast<AssertCapabilityAttr>(At);
      auto *tempInstArgs = new (C, 16) Expr *[A->args_size()];
      {
        EnterExpressionEvaluationContext Unevaluated(S, Sema::ExpressionEvaluationContext::Unevaluated);
        Expr * *TI = tempInstArgs;
        Expr * *I = A->args_begin();
        Expr * *E = A->args_end();
        for (; I != E; ++I, ++TI) {
          ExprResult Result = S.SubstExpr(*I, TemplateArgs);
          if (Result.isInvalid())
            return nullptr;
          *TI = Result.get();
        }
      }
      return new (C) AssertCapabilityAttr(C, *A, tempInstArgs, A->args_size());
    }
    case attr::AssumeAligned: {
      const auto *A = cast<AssumeAlignedAttr>(At);
      return A->clone(C);
    }
    case attr::Atomic: {
      const auto *A = cast<AtomicAttr>(At);
      return A->clone(C);
    }
    case attr::Availability: {
      const auto *A = cast<AvailabilityAttr>(At);
      return A->clone(C);
    }
    case attr::AvailableOnlyInDefaultEvalMethod: {
      const auto *A = cast<AvailableOnlyInDefaultEvalMethodAttr>(At);
      return A->clone(C);
    }
    case attr::BPFFastCall: {
      const auto *A = cast<BPFFastCallAttr>(At);
      return A->clone(C);
    }
    case attr::BPFPreserveAccessIndex: {
      const auto *A = cast<BPFPreserveAccessIndexAttr>(At);
      return A->clone(C);
    }
    case attr::BPFPreserveStaticOffset: {
      const auto *A = cast<BPFPreserveStaticOffsetAttr>(At);
      return A->clone(C);
    }
    case attr::BTFDeclTag: {
      const auto *A = cast<BTFDeclTagAttr>(At);
      return A->clone(C);
    }
    case attr::BTFTypeTag: {
      const auto *A = cast<BTFTypeTagAttr>(At);
      return A->clone(C);
    }
    case attr::Blocking: {
      const auto *A = cast<BlockingAttr>(At);
      return A->clone(C);
    }
    case attr::Blocks: {
      const auto *A = cast<BlocksAttr>(At);
      return A->clone(C);
    }
    case attr::Builtin: {
      const auto *A = cast<BuiltinAttr>(At);
      return A->clone(C);
    }
    case attr::BuiltinAlias: {
      const auto *A = cast<BuiltinAliasAttr>(At);
      return A->clone(C);
    }
    case attr::C11NoReturn: {
      const auto *A = cast<C11NoReturnAttr>(At);
      return A->clone(C);
    }
    case attr::CDecl: {
      const auto *A = cast<CDeclAttr>(At);
      return A->clone(C);
    }
    case attr::CFAuditedTransfer: {
      const auto *A = cast<CFAuditedTransferAttr>(At);
      return A->clone(C);
    }
    case attr::CFConsumed: {
      const auto *A = cast<CFConsumedAttr>(At);
      return A->clone(C);
    }
    case attr::CFGuard: {
      const auto *A = cast<CFGuardAttr>(At);
      return A->clone(C);
    }
    case attr::CFICanonicalJumpTable: {
      const auto *A = cast<CFICanonicalJumpTableAttr>(At);
      return A->clone(C);
    }
    case attr::CFReturnsNotRetained: {
      const auto *A = cast<CFReturnsNotRetainedAttr>(At);
      return A->clone(C);
    }
    case attr::CFReturnsRetained: {
      const auto *A = cast<CFReturnsRetainedAttr>(At);
      return A->clone(C);
    }
    case attr::CFUnknownTransfer: {
      const auto *A = cast<CFUnknownTransferAttr>(At);
      return A->clone(C);
    }
    case attr::CPUDispatch: {
      const auto *A = cast<CPUDispatchAttr>(At);
      return A->clone(C);
    }
    case attr::CPUSpecific: {
      const auto *A = cast<CPUSpecificAttr>(At);
      return A->clone(C);
    }
    case attr::CUDAConstant: {
      const auto *A = cast<CUDAConstantAttr>(At);
      return A->clone(C);
    }
    case attr::CUDADevice: {
      const auto *A = cast<CUDADeviceAttr>(At);
      return A->clone(C);
    }
    case attr::CUDADeviceBuiltinSurfaceType: {
      const auto *A = cast<CUDADeviceBuiltinSurfaceTypeAttr>(At);
      return A->clone(C);
    }
    case attr::CUDADeviceBuiltinTextureType: {
      const auto *A = cast<CUDADeviceBuiltinTextureTypeAttr>(At);
      return A->clone(C);
    }
    case attr::CUDAGlobal: {
      const auto *A = cast<CUDAGlobalAttr>(At);
      return A->clone(C);
    }
    case attr::CUDAGridConstant: {
      const auto *A = cast<CUDAGridConstantAttr>(At);
      return A->clone(C);
    }
    case attr::CUDAHost: {
      const auto *A = cast<CUDAHostAttr>(At);
      return A->clone(C);
    }
    case attr::CUDAInvalidTarget: {
      const auto *A = cast<CUDAInvalidTargetAttr>(At);
      return A->clone(C);
    }
    case attr::CUDALaunchBounds: {
      const auto *A = cast<CUDALaunchBoundsAttr>(At);
      return A->clone(C);
    }
    case attr::CUDAShared: {
      const auto *A = cast<CUDASharedAttr>(At);
      return A->clone(C);
    }
    case attr::CXX11NoReturn: {
      const auto *A = cast<CXX11NoReturnAttr>(At);
      return A->clone(C);
    }
    case attr::CXXAssume: {
      const auto *A = cast<CXXAssumeAttr>(At);
      return A->clone(C);
    }
    case attr::CallableWhen: {
      const auto *A = cast<CallableWhenAttr>(At);
      return A->clone(C);
    }
    case attr::Callback: {
      const auto *A = cast<CallbackAttr>(At);
      return A->clone(C);
    }
    case attr::CalledOnce: {
      const auto *A = cast<CalledOnceAttr>(At);
      return A->clone(C);
    }
    case attr::Capability: {
      const auto *A = cast<CapabilityAttr>(At);
      return A->clone(C);
    }
    case attr::CapturedRecord: {
      const auto *A = cast<CapturedRecordAttr>(At);
      return A->clone(C);
    }
    case attr::CarriesDependency: {
      const auto *A = cast<CarriesDependencyAttr>(At);
      return A->clone(C);
    }
    case attr::Cleanup: {
      const auto *A = cast<CleanupAttr>(At);
      return A->clone(C);
    }
    case attr::ClspvLibclcBuiltin: {
      const auto *A = cast<ClspvLibclcBuiltinAttr>(At);
      return A->clone(C);
    }
    case attr::CmseNSCall: {
      const auto *A = cast<CmseNSCallAttr>(At);
      return A->clone(C);
    }
    case attr::CmseNSEntry: {
      const auto *A = cast<CmseNSEntryAttr>(At);
      return A->clone(C);
    }
    case attr::CodeAlign: {
      const auto *A = cast<CodeAlignAttr>(At);
      return A->clone(C);
    }
    case attr::CodeModel: {
      const auto *A = cast<CodeModelAttr>(At);
      return A->clone(C);
    }
    case attr::CodeSeg: {
      const auto *A = cast<CodeSegAttr>(At);
      return A->clone(C);
    }
    case attr::Cold: {
      const auto *A = cast<ColdAttr>(At);
      return A->clone(C);
    }
    case attr::Common: {
      const auto *A = cast<CommonAttr>(At);
      return A->clone(C);
    }
    case attr::Const: {
      const auto *A = cast<ConstAttr>(At);
      return A->clone(C);
    }
    case attr::ConstInit: {
      const auto *A = cast<ConstInitAttr>(At);
      return A->clone(C);
    }
    case attr::Constructor: {
      const auto *A = cast<ConstructorAttr>(At);
      return A->clone(C);
    }
    case attr::Consumable: {
      const auto *A = cast<ConsumableAttr>(At);
      return A->clone(C);
    }
    case attr::ConsumableAutoCast: {
      const auto *A = cast<ConsumableAutoCastAttr>(At);
      return A->clone(C);
    }
    case attr::ConsumableSetOnRead: {
      const auto *A = cast<ConsumableSetOnReadAttr>(At);
      return A->clone(C);
    }
    case attr::Convergent: {
      const auto *A = cast<ConvergentAttr>(At);
      return A->clone(C);
    }
    case attr::CoroAwaitElidable: {
      const auto *A = cast<CoroAwaitElidableAttr>(At);
      return A->clone(C);
    }
    case attr::CoroAwaitElidableArgument: {
      const auto *A = cast<CoroAwaitElidableArgumentAttr>(At);
      return A->clone(C);
    }
    case attr::CoroDisableLifetimeBound: {
      const auto *A = cast<CoroDisableLifetimeBoundAttr>(At);
      return A->clone(C);
    }
    case attr::CoroLifetimeBound: {
      const auto *A = cast<CoroLifetimeBoundAttr>(At);
      return A->clone(C);
    }
    case attr::CoroOnlyDestroyWhenComplete: {
      const auto *A = cast<CoroOnlyDestroyWhenCompleteAttr>(At);
      return A->clone(C);
    }
    case attr::CoroReturnType: {
      const auto *A = cast<CoroReturnTypeAttr>(At);
      return A->clone(C);
    }
    case attr::CoroWrapper: {
      const auto *A = cast<CoroWrapperAttr>(At);
      return A->clone(C);
    }
    case attr::CountedBy: {
      const auto *A = cast<CountedByAttr>(At);
      return A->clone(C);
    }
    case attr::CountedByOrNull: {
      const auto *A = cast<CountedByOrNullAttr>(At);
      return A->clone(C);
    }
    case attr::DLLExport: {
      const auto *A = cast<DLLExportAttr>(At);
      return A->clone(C);
    }
    case attr::DLLExportStaticLocal: {
      const auto *A = cast<DLLExportStaticLocalAttr>(At);
      return A->clone(C);
    }
    case attr::DLLImport: {
      const auto *A = cast<DLLImportAttr>(At);
      return A->clone(C);
    }
    case attr::DLLImportStaticLocal: {
      const auto *A = cast<DLLImportStaticLocalAttr>(At);
      return A->clone(C);
    }
    case attr::Deprecated: {
      const auto *A = cast<DeprecatedAttr>(At);
      return A->clone(C);
    }
    case attr::Destructor: {
      const auto *A = cast<DestructorAttr>(At);
      return A->clone(C);
    }
    case attr::DiagnoseAsBuiltin: {
      const auto *A = cast<DiagnoseAsBuiltinAttr>(At);
      return A->clone(C);
    }
    case attr::DiagnoseIf: {
      const auto *A = cast<DiagnoseIfAttr>(At);
      Expr * tempInstCond;
      {
        EnterExpressionEvaluationContext Unevaluated(S, Sema::ExpressionEvaluationContext::Unevaluated);
        ExprResult Result = S.SubstExpr(A->getCond(), TemplateArgs);
        if (Result.isInvalid())
          return nullptr;
        tempInstCond = Result.get();
      }
      return new (C) DiagnoseIfAttr(C, *A, tempInstCond, A->getMessage(), A->getDefaultSeverity(), A->getWarningGroup(), A->getArgDependent(), A->getParent());
    }
    case attr::DisableSanitizerInstrumentation: {
      const auto *A = cast<DisableSanitizerInstrumentationAttr>(At);
      return A->clone(C);
    }
    case attr::DisableTailCalls: {
      const auto *A = cast<DisableTailCallsAttr>(At);
      return A->clone(C);
    }
    case attr::EmptyBases: {
      const auto *A = cast<EmptyBasesAttr>(At);
      return A->clone(C);
    }
    case attr::EnableIf: {
      const auto *A = cast<EnableIfAttr>(At);
      Expr * tempInstCond;
      {
        EnterExpressionEvaluationContext Unevaluated(S, Sema::ExpressionEvaluationContext::Unevaluated);
        ExprResult Result = S.SubstExpr(A->getCond(), TemplateArgs);
        if (Result.isInvalid())
          return nullptr;
        tempInstCond = Result.get();
      }
      return new (C) EnableIfAttr(C, *A, tempInstCond, A->getMessage());
    }
    case attr::EnforceTCB: {
      const auto *A = cast<EnforceTCBAttr>(At);
      return A->clone(C);
    }
    case attr::EnforceTCBLeaf: {
      const auto *A = cast<EnforceTCBLeafAttr>(At);
      return A->clone(C);
    }
    case attr::EnumExtensibility: {
      const auto *A = cast<EnumExtensibilityAttr>(At);
      return A->clone(C);
    }
    case attr::Error: {
      const auto *A = cast<ErrorAttr>(At);
      return A->clone(C);
    }
    case attr::ExcludeFromExplicitInstantiation: {
      const auto *A = cast<ExcludeFromExplicitInstantiationAttr>(At);
      return A->clone(C);
    }
    case attr::ExplicitInit: {
      const auto *A = cast<ExplicitInitAttr>(At);
      return A->clone(C);
    }
    case attr::ExtVectorType: {
      const auto *A = cast<ExtVectorTypeAttr>(At);
      return A->clone(C);
    }
    case attr::ExternalSourceSymbol: {
      const auto *A = cast<ExternalSourceSymbolAttr>(At);
      return A->clone(C);
    }
    case attr::FallThrough: {
      const auto *A = cast<FallThroughAttr>(At);
      return A->clone(C);
    }
    case attr::FastCall: {
      const auto *A = cast<FastCallAttr>(At);
      return A->clone(C);
    }
    case attr::Final: {
      const auto *A = cast<FinalAttr>(At);
      return A->clone(C);
    }
    case attr::FlagEnum: {
      const auto *A = cast<FlagEnumAttr>(At);
      return A->clone(C);
    }
    case attr::Flatten: {
      const auto *A = cast<FlattenAttr>(At);
      return A->clone(C);
    }
    case attr::Format: {
      const auto *A = cast<FormatAttr>(At);
      return A->clone(C);
    }
    case attr::FormatArg: {
      const auto *A = cast<FormatArgAttr>(At);
      return A->clone(C);
    }
    case attr::FormatMatches: {
      const auto *A = cast<FormatMatchesAttr>(At);
      return A->clone(C);
    }
    case attr::FunctionReturnThunks: {
      const auto *A = cast<FunctionReturnThunksAttr>(At);
      return A->clone(C);
    }
    case attr::GNUInline: {
      const auto *A = cast<GNUInlineAttr>(At);
      return A->clone(C);
    }
    case attr::GuardedBy: {
      const auto *A = cast<GuardedByAttr>(At);
      Expr * tempInstArg;
      {
        EnterExpressionEvaluationContext Unevaluated(S, Sema::ExpressionEvaluationContext::Unevaluated);
        ExprResult Result = S.SubstExpr(A->getArg(), TemplateArgs);
        if (Result.isInvalid())
          return nullptr;
        tempInstArg = Result.get();
      }
      return new (C) GuardedByAttr(C, *A, tempInstArg);
    }
    case attr::GuardedVar: {
      const auto *A = cast<GuardedVarAttr>(At);
      return A->clone(C);
    }
    case attr::HIPManaged: {
      const auto *A = cast<HIPManagedAttr>(At);
      return A->clone(C);
    }
    case attr::HLSLContainedType: {
      const auto *A = cast<HLSLContainedTypeAttr>(At);
      return A->clone(C);
    }
    case attr::HLSLControlFlowHint: {
      const auto *A = cast<HLSLControlFlowHintAttr>(At);
      return A->clone(C);
    }
    case attr::HLSLGroupSharedAddressSpace: {
      const auto *A = cast<HLSLGroupSharedAddressSpaceAttr>(At);
      return A->clone(C);
    }
    case attr::HLSLLoopHint: {
      const auto *A = cast<HLSLLoopHintAttr>(At);
      return A->clone(C);
    }
    case attr::HLSLNumThreads: {
      const auto *A = cast<HLSLNumThreadsAttr>(At);
      return A->clone(C);
    }
    case attr::HLSLPackOffset: {
      const auto *A = cast<HLSLPackOffsetAttr>(At);
      return A->clone(C);
    }
    case attr::HLSLParamModifier: {
      const auto *A = cast<HLSLParamModifierAttr>(At);
      return A->clone(C);
    }
    case attr::HLSLROV: {
      const auto *A = cast<HLSLROVAttr>(At);
      return A->clone(C);
    }
    case attr::HLSLRawBuffer: {
      const auto *A = cast<HLSLRawBufferAttr>(At);
      return A->clone(C);
    }
    case attr::HLSLResourceBinding: {
      const auto *A = cast<HLSLResourceBindingAttr>(At);
      return A->clone(C);
    }
    case attr::HLSLResourceClass: {
      const auto *A = cast<HLSLResourceClassAttr>(At);
      return A->clone(C);
    }
    case attr::HLSLSV_DispatchThreadID: {
      const auto *A = cast<HLSLSV_DispatchThreadIDAttr>(At);
      return A->clone(C);
    }
    case attr::HLSLSV_GroupID: {
      const auto *A = cast<HLSLSV_GroupIDAttr>(At);
      return A->clone(C);
    }
    case attr::HLSLSV_GroupIndex: {
      const auto *A = cast<HLSLSV_GroupIndexAttr>(At);
      return A->clone(C);
    }
    case attr::HLSLSV_GroupThreadID: {
      const auto *A = cast<HLSLSV_GroupThreadIDAttr>(At);
      return A->clone(C);
    }
    case attr::HLSLShader: {
      const auto *A = cast<HLSLShaderAttr>(At);
      return A->clone(C);
    }
    case attr::HLSLWaveSize: {
      const auto *A = cast<HLSLWaveSizeAttr>(At);
      return A->clone(C);
    }
    case attr::Hot: {
      const auto *A = cast<HotAttr>(At);
      return A->clone(C);
    }
    case attr::HybridPatchable: {
      const auto *A = cast<HybridPatchableAttr>(At);
      return A->clone(C);
    }
    case attr::IBAction: {
      const auto *A = cast<IBActionAttr>(At);
      return A->clone(C);
    }
    case attr::IBOutlet: {
      const auto *A = cast<IBOutletAttr>(At);
      return A->clone(C);
    }
    case attr::IBOutletCollection: {
      const auto *A = cast<IBOutletCollectionAttr>(At);
      return A->clone(C);
    }
    case attr::IFunc: {
      const auto *A = cast<IFuncAttr>(At);
      return A->clone(C);
    }
    case attr::InitPriority: {
      const auto *A = cast<InitPriorityAttr>(At);
      return A->clone(C);
    }
    case attr::InitSeg: {
      const auto *A = cast<InitSegAttr>(At);
      return A->clone(C);
    }
    case attr::IntelOclBicc: {
      const auto *A = cast<IntelOclBiccAttr>(At);
      return A->clone(C);
    }
    case attr::InternalLinkage: {
      const auto *A = cast<InternalLinkageAttr>(At);
      return A->clone(C);
    }
    case attr::LTOVisibilityPublic: {
      const auto *A = cast<LTOVisibilityPublicAttr>(At);
      return A->clone(C);
    }
    case attr::LayoutVersion: {
      const auto *A = cast<LayoutVersionAttr>(At);
      return A->clone(C);
    }
    case attr::Leaf: {
      const auto *A = cast<LeafAttr>(At);
      return A->clone(C);
    }
    case attr::LifetimeBound: {
      const auto *A = cast<LifetimeBoundAttr>(At);
      return A->clone(C);
    }
    case attr::LifetimeCaptureBy: {
      const auto *A = cast<LifetimeCaptureByAttr>(At);
      return A->clone(C);
    }
    case attr::Likely: {
      const auto *A = cast<LikelyAttr>(At);
      return A->clone(C);
    }
    case attr::LoaderUninitialized: {
      const auto *A = cast<LoaderUninitializedAttr>(At);
      return A->clone(C);
    }
    case attr::LockReturned: {
      const auto *A = cast<LockReturnedAttr>(At);
      Expr * tempInstArg;
      {
        EnterExpressionEvaluationContext Unevaluated(S, Sema::ExpressionEvaluationContext::Unevaluated);
        ExprResult Result = S.SubstExpr(A->getArg(), TemplateArgs);
        if (Result.isInvalid())
          return nullptr;
        tempInstArg = Result.get();
      }
      return new (C) LockReturnedAttr(C, *A, tempInstArg);
    }
    case attr::LocksExcluded: {
      const auto *A = cast<LocksExcludedAttr>(At);
      auto *tempInstArgs = new (C, 16) Expr *[A->args_size()];
      {
        EnterExpressionEvaluationContext Unevaluated(S, Sema::ExpressionEvaluationContext::Unevaluated);
        Expr * *TI = tempInstArgs;
        Expr * *I = A->args_begin();
        Expr * *E = A->args_end();
        for (; I != E; ++I, ++TI) {
          ExprResult Result = S.SubstExpr(*I, TemplateArgs);
          if (Result.isInvalid())
            return nullptr;
          *TI = Result.get();
        }
      }
      return new (C) LocksExcludedAttr(C, *A, tempInstArgs, A->args_size());
    }
    case attr::LoopHint: {
      const auto *A = cast<LoopHintAttr>(At);
      return A->clone(C);
    }
    case attr::M68kInterrupt: {
      const auto *A = cast<M68kInterruptAttr>(At);
      return A->clone(C);
    }
    case attr::M68kRTD: {
      const auto *A = cast<M68kRTDAttr>(At);
      return A->clone(C);
    }
    case attr::MIGServerRoutine: {
      const auto *A = cast<MIGServerRoutineAttr>(At);
      return A->clone(C);
    }
    case attr::MSABI: {
      const auto *A = cast<MSABIAttr>(At);
      return A->clone(C);
    }
    case attr::MSAllocator: {
      const auto *A = cast<MSAllocatorAttr>(At);
      return A->clone(C);
    }
    case attr::MSConstexpr: {
      const auto *A = cast<MSConstexprAttr>(At);
      return A->clone(C);
    }
    case attr::MSInheritance: {
      const auto *A = cast<MSInheritanceAttr>(At);
      return A->clone(C);
    }
    case attr::MSNoVTable: {
      const auto *A = cast<MSNoVTableAttr>(At);
      return A->clone(C);
    }
    case attr::MSP430Interrupt: {
      const auto *A = cast<MSP430InterruptAttr>(At);
      return A->clone(C);
    }
    case attr::MSStruct: {
      const auto *A = cast<MSStructAttr>(At);
      return A->clone(C);
    }
    case attr::MSVtorDisp: {
      const auto *A = cast<MSVtorDispAttr>(At);
      return A->clone(C);
    }
    case attr::MaxFieldAlignment: {
      const auto *A = cast<MaxFieldAlignmentAttr>(At);
      return A->clone(C);
    }
    case attr::MayAlias: {
      const auto *A = cast<MayAliasAttr>(At);
      return A->clone(C);
    }
    case attr::MaybeUndef: {
      const auto *A = cast<MaybeUndefAttr>(At);
      return A->clone(C);
    }
    case attr::MicroMips: {
      const auto *A = cast<MicroMipsAttr>(At);
      return A->clone(C);
    }
    case attr::MinSize: {
      const auto *A = cast<MinSizeAttr>(At);
      return A->clone(C);
    }
    case attr::MinVectorWidth: {
      const auto *A = cast<MinVectorWidthAttr>(At);
      return A->clone(C);
    }
    case attr::Mips16: {
      const auto *A = cast<Mips16Attr>(At);
      return A->clone(C);
    }
    case attr::MipsInterrupt: {
      const auto *A = cast<MipsInterruptAttr>(At);
      return A->clone(C);
    }
    case attr::MipsLongCall: {
      const auto *A = cast<MipsLongCallAttr>(At);
      return A->clone(C);
    }
    case attr::MipsShortCall: {
      const auto *A = cast<MipsShortCallAttr>(At);
      return A->clone(C);
    }
    case attr::Mode: {
      const auto *A = cast<ModeAttr>(At);
      return A->clone(C);
    }
    case attr::MustTail: {
      const auto *A = cast<MustTailAttr>(At);
      return A->clone(C);
    }
    case attr::NSConsumed: {
      const auto *A = cast<NSConsumedAttr>(At);
      return A->clone(C);
    }
    case attr::NSConsumesSelf: {
      const auto *A = cast<NSConsumesSelfAttr>(At);
      return A->clone(C);
    }
    case attr::NSErrorDomain: {
      const auto *A = cast<NSErrorDomainAttr>(At);
      return A->clone(C);
    }
    case attr::NSReturnsAutoreleased: {
      const auto *A = cast<NSReturnsAutoreleasedAttr>(At);
      return A->clone(C);
    }
    case attr::NSReturnsNotRetained: {
      const auto *A = cast<NSReturnsNotRetainedAttr>(At);
      return A->clone(C);
    }
    case attr::NSReturnsRetained: {
      const auto *A = cast<NSReturnsRetainedAttr>(At);
      return A->clone(C);
    }
    case attr::NVPTXKernel: {
      const auto *A = cast<NVPTXKernelAttr>(At);
      return A->clone(C);
    }
    case attr::Naked: {
      const auto *A = cast<NakedAttr>(At);
      return A->clone(C);
    }
    case attr::NoAlias: {
      const auto *A = cast<NoAliasAttr>(At);
      return A->clone(C);
    }
    case attr::NoBuiltin: {
      const auto *A = cast<NoBuiltinAttr>(At);
      return A->clone(C);
    }
    case attr::NoCommon: {
      const auto *A = cast<NoCommonAttr>(At);
      return A->clone(C);
    }
    case attr::NoConvergent: {
      const auto *A = cast<NoConvergentAttr>(At);
      return A->clone(C);
    }
    case attr::NoDebug: {
      const auto *A = cast<NoDebugAttr>(At);
      return A->clone(C);
    }
    case attr::NoDeref: {
      const auto *A = cast<NoDerefAttr>(At);
      return A->clone(C);
    }
    case attr::NoDestroy: {
      const auto *A = cast<NoDestroyAttr>(At);
      return A->clone(C);
    }
    case attr::NoDuplicate: {
      const auto *A = cast<NoDuplicateAttr>(At);
      return A->clone(C);
    }
    case attr::NoEscape: {
      const auto *A = cast<NoEscapeAttr>(At);
      return A->clone(C);
    }
    case attr::NoInline: {
      const auto *A = cast<NoInlineAttr>(At);
      return A->clone(C);
    }
    case attr::NoInstrumentFunction: {
      const auto *A = cast<NoInstrumentFunctionAttr>(At);
      return A->clone(C);
    }
    case attr::NoMerge: {
      const auto *A = cast<NoMergeAttr>(At);
      return A->clone(C);
    }
    case attr::NoMicroMips: {
      const auto *A = cast<NoMicroMipsAttr>(At);
      return A->clone(C);
    }
    case attr::NoMips16: {
      const auto *A = cast<NoMips16Attr>(At);
      return A->clone(C);
    }
    case attr::NoProfileFunction: {
      const auto *A = cast<NoProfileFunctionAttr>(At);
      return A->clone(C);
    }
    case attr::NoRandomizeLayout: {
      const auto *A = cast<NoRandomizeLayoutAttr>(At);
      return A->clone(C);
    }
    case attr::NoReturn: {
      const auto *A = cast<NoReturnAttr>(At);
      return A->clone(C);
    }
    case attr::NoSanitize: {
      const auto *A = cast<NoSanitizeAttr>(At);
      return A->clone(C);
    }
    case attr::NoSpecializations: {
      const auto *A = cast<NoSpecializationsAttr>(At);
      return new (C) NoSpecializationsAttr(C, *A, A->getMessage());
    }
    case attr::NoSpeculativeLoadHardening: {
      const auto *A = cast<NoSpeculativeLoadHardeningAttr>(At);
      return A->clone(C);
    }
    case attr::NoSplitStack: {
      const auto *A = cast<NoSplitStackAttr>(At);
      return A->clone(C);
    }
    case attr::NoStackProtector: {
      const auto *A = cast<NoStackProtectorAttr>(At);
      return A->clone(C);
    }
    case attr::NoThreadSafetyAnalysis: {
      const auto *A = cast<NoThreadSafetyAnalysisAttr>(At);
      return A->clone(C);
    }
    case attr::NoThrow: {
      const auto *A = cast<NoThrowAttr>(At);
      return A->clone(C);
    }
    case attr::NoTrivialAutoVarInit: {
      const auto *A = cast<NoTrivialAutoVarInitAttr>(At);
      return A->clone(C);
    }
    case attr::NoUniqueAddress: {
      const auto *A = cast<NoUniqueAddressAttr>(At);
      return A->clone(C);
    }
    case attr::NoUwtable: {
      const auto *A = cast<NoUwtableAttr>(At);
      return A->clone(C);
    }
    case attr::NonAllocating: {
      const auto *A = cast<NonAllocatingAttr>(At);
      return A->clone(C);
    }
    case attr::NonBlocking: {
      const auto *A = cast<NonBlockingAttr>(At);
      return A->clone(C);
    }
    case attr::NonNull: {
      const auto *A = cast<NonNullAttr>(At);
      return A->clone(C);
    }
    case attr::NotTailCalled: {
      const auto *A = cast<NotTailCalledAttr>(At);
      return A->clone(C);
    }
    case attr::OMPAllocateDecl: {
      const auto *A = cast<OMPAllocateDeclAttr>(At);
      return A->clone(C);
    }
    case attr::OMPAssume: {
      const auto *A = cast<OMPAssumeAttr>(At);
      return A->clone(C);
    }
    case attr::OMPCaptureKind: {
      const auto *A = cast<OMPCaptureKindAttr>(At);
      return A->clone(C);
    }
    case attr::OMPCaptureNoInit: {
      const auto *A = cast<OMPCaptureNoInitAttr>(At);
      return A->clone(C);
    }
    case attr::OMPDeclareSimdDecl: {
      const auto *A = cast<OMPDeclareSimdDeclAttr>(At);
      return A->clone(C);
    }
    case attr::OMPDeclareTargetDecl: {
      const auto *A = cast<OMPDeclareTargetDeclAttr>(At);
      return A->clone(C);
    }
    case attr::OMPDeclareVariant: {
      const auto *A = cast<OMPDeclareVariantAttr>(At);
      return A->clone(C);
    }
    case attr::OMPReferencedVar: {
      const auto *A = cast<OMPReferencedVarAttr>(At);
      return A->clone(C);
    }
    case attr::OMPThreadPrivateDecl: {
      const auto *A = cast<OMPThreadPrivateDeclAttr>(At);
      return A->clone(C);
    }
    case attr::OSConsumed: {
      const auto *A = cast<OSConsumedAttr>(At);
      return A->clone(C);
    }
    case attr::OSConsumesThis: {
      const auto *A = cast<OSConsumesThisAttr>(At);
      return A->clone(C);
    }
    case attr::OSReturnsNotRetained: {
      const auto *A = cast<OSReturnsNotRetainedAttr>(At);
      return A->clone(C);
    }
    case attr::OSReturnsRetained: {
      const auto *A = cast<OSReturnsRetainedAttr>(At);
      return A->clone(C);
    }
    case attr::OSReturnsRetainedOnNonZero: {
      const auto *A = cast<OSReturnsRetainedOnNonZeroAttr>(At);
      return A->clone(C);
    }
    case attr::OSReturnsRetainedOnZero: {
      const auto *A = cast<OSReturnsRetainedOnZeroAttr>(At);
      return A->clone(C);
    }
    case attr::ObjCBoxable: {
      const auto *A = cast<ObjCBoxableAttr>(At);
      return A->clone(C);
    }
    case attr::ObjCBridge: {
      const auto *A = cast<ObjCBridgeAttr>(At);
      return A->clone(C);
    }
    case attr::ObjCBridgeMutable: {
      const auto *A = cast<ObjCBridgeMutableAttr>(At);
      return A->clone(C);
    }
    case attr::ObjCBridgeRelated: {
      const auto *A = cast<ObjCBridgeRelatedAttr>(At);
      return A->clone(C);
    }
    case attr::ObjCClassStub: {
      const auto *A = cast<ObjCClassStubAttr>(At);
      return A->clone(C);
    }
    case attr::ObjCDesignatedInitializer: {
      const auto *A = cast<ObjCDesignatedInitializerAttr>(At);
      return A->clone(C);
    }
    case attr::ObjCDirect: {
      const auto *A = cast<ObjCDirectAttr>(At);
      return A->clone(C);
    }
    case attr::ObjCDirectMembers: {
      const auto *A = cast<ObjCDirectMembersAttr>(At);
      return A->clone(C);
    }
    case attr::ObjCException: {
      const auto *A = cast<ObjCExceptionAttr>(At);
      return A->clone(C);
    }
    case attr::ObjCExplicitProtocolImpl: {
      const auto *A = cast<ObjCExplicitProtocolImplAttr>(At);
      return A->clone(C);
    }
    case attr::ObjCExternallyRetained: {
      const auto *A = cast<ObjCExternallyRetainedAttr>(At);
      return A->clone(C);
    }
    case attr::ObjCGC: {
      const auto *A = cast<ObjCGCAttr>(At);
      return A->clone(C);
    }
    case attr::ObjCIndependentClass: {
      const auto *A = cast<ObjCIndependentClassAttr>(At);
      return A->clone(C);
    }
    case attr::ObjCInertUnsafeUnretained: {
      const auto *A = cast<ObjCInertUnsafeUnretainedAttr>(At);
      return A->clone(C);
    }
    case attr::ObjCKindOf: {
      const auto *A = cast<ObjCKindOfAttr>(At);
      return A->clone(C);
    }
    case attr::ObjCMethodFamily: {
      const auto *A = cast<ObjCMethodFamilyAttr>(At);
      return A->clone(C);
    }
    case attr::ObjCNSObject: {
      const auto *A = cast<ObjCNSObjectAttr>(At);
      return A->clone(C);
    }
    case attr::ObjCNonLazyClass: {
      const auto *A = cast<ObjCNonLazyClassAttr>(At);
      return A->clone(C);
    }
    case attr::ObjCNonRuntimeProtocol: {
      const auto *A = cast<ObjCNonRuntimeProtocolAttr>(At);
      return A->clone(C);
    }
    case attr::ObjCOwnership: {
      const auto *A = cast<ObjCOwnershipAttr>(At);
      return A->clone(C);
    }
    case attr::ObjCPreciseLifetime: {
      const auto *A = cast<ObjCPreciseLifetimeAttr>(At);
      return A->clone(C);
    }
    case attr::ObjCRequiresPropertyDefs: {
      const auto *A = cast<ObjCRequiresPropertyDefsAttr>(At);
      return A->clone(C);
    }
    case attr::ObjCRequiresSuper: {
      const auto *A = cast<ObjCRequiresSuperAttr>(At);
      return A->clone(C);
    }
    case attr::ObjCReturnsInnerPointer: {
      const auto *A = cast<ObjCReturnsInnerPointerAttr>(At);
      return A->clone(C);
    }
    case attr::ObjCRootClass: {
      const auto *A = cast<ObjCRootClassAttr>(At);
      return A->clone(C);
    }
    case attr::ObjCRuntimeName: {
      const auto *A = cast<ObjCRuntimeNameAttr>(At);
      return A->clone(C);
    }
    case attr::ObjCRuntimeVisible: {
      const auto *A = cast<ObjCRuntimeVisibleAttr>(At);
      return A->clone(C);
    }
    case attr::ObjCSubclassingRestricted: {
      const auto *A = cast<ObjCSubclassingRestrictedAttr>(At);
      return A->clone(C);
    }
    case attr::OpenACCRoutineAnnot: {
      const auto *A = cast<OpenACCRoutineAnnotAttr>(At);
      return A->clone(C);
    }
    case attr::OpenACCRoutineDecl: {
      const auto *A = cast<OpenACCRoutineDeclAttr>(At);
      return A->clone(C);
    }
    case attr::OpenCLAccess: {
      const auto *A = cast<OpenCLAccessAttr>(At);
      return A->clone(C);
    }
    case attr::OpenCLConstantAddressSpace: {
      const auto *A = cast<OpenCLConstantAddressSpaceAttr>(At);
      return A->clone(C);
    }
    case attr::OpenCLGenericAddressSpace: {
      const auto *A = cast<OpenCLGenericAddressSpaceAttr>(At);
      return A->clone(C);
    }
    case attr::OpenCLGlobalAddressSpace: {
      const auto *A = cast<OpenCLGlobalAddressSpaceAttr>(At);
      return A->clone(C);
    }
    case attr::OpenCLGlobalDeviceAddressSpace: {
      const auto *A = cast<OpenCLGlobalDeviceAddressSpaceAttr>(At);
      return A->clone(C);
    }
    case attr::OpenCLGlobalHostAddressSpace: {
      const auto *A = cast<OpenCLGlobalHostAddressSpaceAttr>(At);
      return A->clone(C);
    }
    case attr::OpenCLIntelReqdSubGroupSize: {
      const auto *A = cast<OpenCLIntelReqdSubGroupSizeAttr>(At);
      return A->clone(C);
    }
    case attr::OpenCLKernel: {
      const auto *A = cast<OpenCLKernelAttr>(At);
      return A->clone(C);
    }
    case attr::OpenCLLocalAddressSpace: {
      const auto *A = cast<OpenCLLocalAddressSpaceAttr>(At);
      return A->clone(C);
    }
    case attr::OpenCLPrivateAddressSpace: {
      const auto *A = cast<OpenCLPrivateAddressSpaceAttr>(At);
      return A->clone(C);
    }
    case attr::OpenCLUnrollHint: {
      const auto *A = cast<OpenCLUnrollHintAttr>(At);
      return A->clone(C);
    }
    case attr::OptimizeNone: {
      const auto *A = cast<OptimizeNoneAttr>(At);
      return A->clone(C);
    }
    case attr::Overloadable: {
      const auto *A = cast<OverloadableAttr>(At);
      return A->clone(C);
    }
    case attr::Override: {
      const auto *A = cast<OverrideAttr>(At);
      return A->clone(C);
    }
    case attr::Owner: {
      const auto *A = cast<OwnerAttr>(At);
      return A->clone(C);
    }
    case attr::Ownership: {
      const auto *A = cast<OwnershipAttr>(At);
      return A->clone(C);
    }
    case attr::Packed: {
      const auto *A = cast<PackedAttr>(At);
      return A->clone(C);
    }
    case attr::ParamTypestate: {
      const auto *A = cast<ParamTypestateAttr>(At);
      return A->clone(C);
    }
    case attr::Pascal: {
      const auto *A = cast<PascalAttr>(At);
      return A->clone(C);
    }
    case attr::PassObjectSize: {
      const auto *A = cast<PassObjectSizeAttr>(At);
      return A->clone(C);
    }
    case attr::PatchableFunctionEntry: {
      const auto *A = cast<PatchableFunctionEntryAttr>(At);
      return A->clone(C);
    }
    case attr::Pcs: {
      const auto *A = cast<PcsAttr>(At);
      return A->clone(C);
    }
    case attr::Pointer: {
      const auto *A = cast<PointerAttr>(At);
      return A->clone(C);
    }
    case attr::PointerAuth: {
      const auto *A = cast<PointerAuthAttr>(At);
      return A->clone(C);
    }
    case attr::PragmaClangBSSSection: {
      const auto *A = cast<PragmaClangBSSSectionAttr>(At);
      return A->clone(C);
    }
    case attr::PragmaClangDataSection: {
      const auto *A = cast<PragmaClangDataSectionAttr>(At);
      return A->clone(C);
    }
    case attr::PragmaClangRelroSection: {
      const auto *A = cast<PragmaClangRelroSectionAttr>(At);
      return A->clone(C);
    }
    case attr::PragmaClangRodataSection: {
      const auto *A = cast<PragmaClangRodataSectionAttr>(At);
      return A->clone(C);
    }
    case attr::PragmaClangTextSection: {
      const auto *A = cast<PragmaClangTextSectionAttr>(At);
      return A->clone(C);
    }
    case attr::PreferredName: {
      const auto *A = cast<PreferredNameAttr>(At);
      TypeSourceInfo * tempInstTypedefType =
        S.SubstType(A->getTypedefTypeLoc(), TemplateArgs, A->getLoc(), A->getAttrName());
      if (!tempInstTypedefType)
        return nullptr;
      return new (C) PreferredNameAttr(C, *A, tempInstTypedefType);
    }
    case attr::PreferredType: {
      const auto *A = cast<PreferredTypeAttr>(At);
      return A->clone(C);
    }
    case attr::PreserveAll: {
      const auto *A = cast<PreserveAllAttr>(At);
      return A->clone(C);
    }
    case attr::PreserveMost: {
      const auto *A = cast<PreserveMostAttr>(At);
      return A->clone(C);
    }
    case attr::PreserveNone: {
      const auto *A = cast<PreserveNoneAttr>(At);
      return A->clone(C);
    }
    case attr::PtGuardedBy: {
      const auto *A = cast<PtGuardedByAttr>(At);
      Expr * tempInstArg;
      {
        EnterExpressionEvaluationContext Unevaluated(S, Sema::ExpressionEvaluationContext::Unevaluated);
        ExprResult Result = S.SubstExpr(A->getArg(), TemplateArgs);
        if (Result.isInvalid())
          return nullptr;
        tempInstArg = Result.get();
      }
      return new (C) PtGuardedByAttr(C, *A, tempInstArg);
    }
    case attr::PtGuardedVar: {
      const auto *A = cast<PtGuardedVarAttr>(At);
      return A->clone(C);
    }
    case attr::Ptr32: {
      const auto *A = cast<Ptr32Attr>(At);
      return A->clone(C);
    }
    case attr::Ptr64: {
      const auto *A = cast<Ptr64Attr>(At);
      return A->clone(C);
    }
    case attr::Pure: {
      const auto *A = cast<PureAttr>(At);
      return A->clone(C);
    }
    case attr::RISCVInterrupt: {
      const auto *A = cast<RISCVInterruptAttr>(At);
      return A->clone(C);
    }
    case attr::RISCVVLSCC: {
      const auto *A = cast<RISCVVLSCCAttr>(At);
      return A->clone(C);
    }
    case attr::RISCVVectorCC: {
      const auto *A = cast<RISCVVectorCCAttr>(At);
      return A->clone(C);
    }
    case attr::RandomizeLayout: {
      const auto *A = cast<RandomizeLayoutAttr>(At);
      return A->clone(C);
    }
    case attr::ReadOnlyPlacement: {
      const auto *A = cast<ReadOnlyPlacementAttr>(At);
      return A->clone(C);
    }
    case attr::RegCall: {
      const auto *A = cast<RegCallAttr>(At);
      return A->clone(C);
    }
    case attr::Reinitializes: {
      const auto *A = cast<ReinitializesAttr>(At);
      return A->clone(C);
    }
    case attr::ReleaseCapability: {
      const auto *A = cast<ReleaseCapabilityAttr>(At);
      auto *tempInstArgs = new (C, 16) Expr *[A->args_size()];
      {
        EnterExpressionEvaluationContext Unevaluated(S, Sema::ExpressionEvaluationContext::Unevaluated);
        Expr * *TI = tempInstArgs;
        Expr * *I = A->args_begin();
        Expr * *E = A->args_end();
        for (; I != E; ++I, ++TI) {
          ExprResult Result = S.SubstExpr(*I, TemplateArgs);
          if (Result.isInvalid())
            return nullptr;
          *TI = Result.get();
        }
      }
      return new (C) ReleaseCapabilityAttr(C, *A, tempInstArgs, A->args_size());
    }
    case attr::ReleaseHandle: {
      const auto *A = cast<ReleaseHandleAttr>(At);
      return A->clone(C);
    }
    case attr::ReqdWorkGroupSize: {
      const auto *A = cast<ReqdWorkGroupSizeAttr>(At);
      return A->clone(C);
    }
    case attr::RequiresCapability: {
      const auto *A = cast<RequiresCapabilityAttr>(At);
      auto *tempInstArgs = new (C, 16) Expr *[A->args_size()];
      {
        EnterExpressionEvaluationContext Unevaluated(S, Sema::ExpressionEvaluationContext::Unevaluated);
        Expr * *TI = tempInstArgs;
        Expr * *I = A->args_begin();
        Expr * *E = A->args_end();
        for (; I != E; ++I, ++TI) {
          ExprResult Result = S.SubstExpr(*I, TemplateArgs);
          if (Result.isInvalid())
            return nullptr;
          *TI = Result.get();
        }
      }
      return new (C) RequiresCapabilityAttr(C, *A, tempInstArgs, A->args_size());
    }
    case attr::Restrict: {
      const auto *A = cast<RestrictAttr>(At);
      return A->clone(C);
    }
    case attr::Retain: {
      const auto *A = cast<RetainAttr>(At);
      return A->clone(C);
    }
    case attr::ReturnTypestate: {
      const auto *A = cast<ReturnTypestateAttr>(At);
      return A->clone(C);
    }
    case attr::ReturnsNonNull: {
      const auto *A = cast<ReturnsNonNullAttr>(At);
      return A->clone(C);
    }
    case attr::ReturnsTwice: {
      const auto *A = cast<ReturnsTwiceAttr>(At);
      return A->clone(C);
    }
    case attr::SPtr: {
      const auto *A = cast<SPtrAttr>(At);
      return A->clone(C);
    }
    case attr::SYCLKernel: {
      const auto *A = cast<SYCLKernelAttr>(At);
      return A->clone(C);
    }
    case attr::SYCLKernelEntryPoint: {
      const auto *A = cast<SYCLKernelEntryPointAttr>(At);
      TypeSourceInfo * tempInstKernelName =
        S.SubstType(A->getKernelNameLoc(), TemplateArgs, A->getLoc(), A->getAttrName());
      if (!tempInstKernelName)
        return nullptr;
      return new (C) SYCLKernelEntryPointAttr(C, *A, tempInstKernelName, A->getInvalid());
    }
    case attr::SYCLSpecialClass: {
      const auto *A = cast<SYCLSpecialClassAttr>(At);
      return A->clone(C);
    }
    case attr::ScopedLockable: {
      const auto *A = cast<ScopedLockableAttr>(At);
      return A->clone(C);
    }
    case attr::Section: {
      const auto *A = cast<SectionAttr>(At);
      return A->clone(C);
    }
    case attr::SelectAny: {
      const auto *A = cast<SelectAnyAttr>(At);
      return A->clone(C);
    }
    case attr::Sentinel: {
      const auto *A = cast<SentinelAttr>(At);
      return A->clone(C);
    }
    case attr::SetTypestate: {
      const auto *A = cast<SetTypestateAttr>(At);
      return A->clone(C);
    }
    case attr::SizedBy: {
      const auto *A = cast<SizedByAttr>(At);
      return A->clone(C);
    }
    case attr::SizedByOrNull: {
      const auto *A = cast<SizedByOrNullAttr>(At);
      return A->clone(C);
    }
    case attr::SpeculativeLoadHardening: {
      const auto *A = cast<SpeculativeLoadHardeningAttr>(At);
      return A->clone(C);
    }
    case attr::StandaloneDebug: {
      const auto *A = cast<StandaloneDebugAttr>(At);
      return A->clone(C);
    }
    case attr::StdCall: {
      const auto *A = cast<StdCallAttr>(At);
      return A->clone(C);
    }
    case attr::StrictFP: {
      const auto *A = cast<StrictFPAttr>(At);
      return A->clone(C);
    }
    case attr::StrictGuardStackCheck: {
      const auto *A = cast<StrictGuardStackCheckAttr>(At);
      return A->clone(C);
    }
    case attr::Suppress: {
      const auto *A = cast<SuppressAttr>(At);
      return A->clone(C);
    }
    case attr::SwiftAsync: {
      const auto *A = cast<SwiftAsyncAttr>(At);
      return A->clone(C);
    }
    case attr::SwiftAsyncCall: {
      const auto *A = cast<SwiftAsyncCallAttr>(At);
      return A->clone(C);
    }
    case attr::SwiftAsyncContext: {
      const auto *A = cast<SwiftAsyncContextAttr>(At);
      return A->clone(C);
    }
    case attr::SwiftAsyncError: {
      const auto *A = cast<SwiftAsyncErrorAttr>(At);
      return A->clone(C);
    }
    case attr::SwiftAsyncName: {
      const auto *A = cast<SwiftAsyncNameAttr>(At);
      return A->clone(C);
    }
    case attr::SwiftAttr: {
      const auto *A = cast<SwiftAttrAttr>(At);
      return A->clone(C);
    }
    case attr::SwiftBridge: {
      const auto *A = cast<SwiftBridgeAttr>(At);
      return A->clone(C);
    }
    case attr::SwiftBridgedTypedef: {
      const auto *A = cast<SwiftBridgedTypedefAttr>(At);
      return A->clone(C);
    }
    case attr::SwiftCall: {
      const auto *A = cast<SwiftCallAttr>(At);
      return A->clone(C);
    }
    case attr::SwiftContext: {
      const auto *A = cast<SwiftContextAttr>(At);
      return A->clone(C);
    }
    case attr::SwiftError: {
      const auto *A = cast<SwiftErrorAttr>(At);
      return A->clone(C);
    }
    case attr::SwiftErrorResult: {
      const auto *A = cast<SwiftErrorResultAttr>(At);
      return A->clone(C);
    }
    case attr::SwiftImportAsNonGeneric: {
      const auto *A = cast<SwiftImportAsNonGenericAttr>(At);
      return A->clone(C);
    }
    case attr::SwiftImportPropertyAsAccessors: {
      const auto *A = cast<SwiftImportPropertyAsAccessorsAttr>(At);
      return A->clone(C);
    }
    case attr::SwiftIndirectResult: {
      const auto *A = cast<SwiftIndirectResultAttr>(At);
      return A->clone(C);
    }
    case attr::SwiftName: {
      const auto *A = cast<SwiftNameAttr>(At);
      return A->clone(C);
    }
    case attr::SwiftNewType: {
      const auto *A = cast<SwiftNewTypeAttr>(At);
      return A->clone(C);
    }
    case attr::SwiftObjCMembers: {
      const auto *A = cast<SwiftObjCMembersAttr>(At);
      return A->clone(C);
    }
    case attr::SwiftPrivate: {
      const auto *A = cast<SwiftPrivateAttr>(At);
      return A->clone(C);
    }
    case attr::SwiftVersionedAddition: {
      const auto *A = cast<SwiftVersionedAdditionAttr>(At);
      return A->clone(C);
    }
    case attr::SwiftVersionedRemoval: {
      const auto *A = cast<SwiftVersionedRemovalAttr>(At);
      return A->clone(C);
    }
    case attr::SysVABI: {
      const auto *A = cast<SysVABIAttr>(At);
      return A->clone(C);
    }
    case attr::TLSModel: {
      const auto *A = cast<TLSModelAttr>(At);
      return A->clone(C);
    }
    case attr::Target: {
      const auto *A = cast<TargetAttr>(At);
      return A->clone(C);
    }
    case attr::TargetClones: {
      const auto *A = cast<TargetClonesAttr>(At);
      return A->clone(C);
    }
    case attr::TargetVersion: {
      const auto *A = cast<TargetVersionAttr>(At);
      return A->clone(C);
    }
    case attr::TestTypestate: {
      const auto *A = cast<TestTypestateAttr>(At);
      return A->clone(C);
    }
    case attr::ThisCall: {
      const auto *A = cast<ThisCallAttr>(At);
      return A->clone(C);
    }
    case attr::Thread: {
      const auto *A = cast<ThreadAttr>(At);
      return A->clone(C);
    }
    case attr::TransparentUnion: {
      const auto *A = cast<TransparentUnionAttr>(At);
      return A->clone(C);
    }
    case attr::TrivialABI: {
      const auto *A = cast<TrivialABIAttr>(At);
      return A->clone(C);
    }
    case attr::TryAcquireCapability: {
      const auto *A = cast<TryAcquireCapabilityAttr>(At);
      Expr * tempInstSuccessValue;
      {
        EnterExpressionEvaluationContext Unevaluated(S, Sema::ExpressionEvaluationContext::Unevaluated);
        ExprResult Result = S.SubstExpr(A->getSuccessValue(), TemplateArgs);
        if (Result.isInvalid())
          return nullptr;
        tempInstSuccessValue = Result.get();
      }
      auto *tempInstArgs = new (C, 16) Expr *[A->args_size()];
      {
        EnterExpressionEvaluationContext Unevaluated(S, Sema::ExpressionEvaluationContext::Unevaluated);
        Expr * *TI = tempInstArgs;
        Expr * *I = A->args_begin();
        Expr * *E = A->args_end();
        for (; I != E; ++I, ++TI) {
          ExprResult Result = S.SubstExpr(*I, TemplateArgs);
          if (Result.isInvalid())
            return nullptr;
          *TI = Result.get();
        }
      }
      return new (C) TryAcquireCapabilityAttr(C, *A, tempInstSuccessValue, tempInstArgs, A->args_size());
    }
    case attr::TypeNonNull: {
      const auto *A = cast<TypeNonNullAttr>(At);
      return A->clone(C);
    }
    case attr::TypeNullUnspecified: {
      const auto *A = cast<TypeNullUnspecifiedAttr>(At);
      return A->clone(C);
    }
    case attr::TypeNullable: {
      const auto *A = cast<TypeNullableAttr>(At);
      return A->clone(C);
    }
    case attr::TypeNullableResult: {
      const auto *A = cast<TypeNullableResultAttr>(At);
      return A->clone(C);
    }
    case attr::TypeTagForDatatype: {
      const auto *A = cast<TypeTagForDatatypeAttr>(At);
      return A->clone(C);
    }
    case attr::TypeVisibility: {
      return nullptr;
    }
    case attr::UPtr: {
      const auto *A = cast<UPtrAttr>(At);
      return A->clone(C);
    }
    case attr::Unavailable: {
      const auto *A = cast<UnavailableAttr>(At);
      return A->clone(C);
    }
    case attr::Uninitialized: {
      const auto *A = cast<UninitializedAttr>(At);
      return A->clone(C);
    }
    case attr::Unlikely: {
      const auto *A = cast<UnlikelyAttr>(At);
      return A->clone(C);
    }
    case attr::UnsafeBufferUsage: {
      const auto *A = cast<UnsafeBufferUsageAttr>(At);
      return A->clone(C);
    }
    case attr::Unused: {
      const auto *A = cast<UnusedAttr>(At);
      return A->clone(C);
    }
    case attr::UseHandle: {
      const auto *A = cast<UseHandleAttr>(At);
      return A->clone(C);
    }
    case attr::Used: {
      const auto *A = cast<UsedAttr>(At);
      return A->clone(C);
    }
    case attr::UsingIfExists: {
      const auto *A = cast<UsingIfExistsAttr>(At);
      return A->clone(C);
    }
    case attr::Uuid: {
      const auto *A = cast<UuidAttr>(At);
      return A->clone(C);
    }
    case attr::VTablePointerAuthentication: {
      const auto *A = cast<VTablePointerAuthenticationAttr>(At);
      return A->clone(C);
    }
    case attr::VecReturn: {
      const auto *A = cast<VecReturnAttr>(At);
      return A->clone(C);
    }
    case attr::VecTypeHint: {
      const auto *A = cast<VecTypeHintAttr>(At);
      return A->clone(C);
    }
    case attr::VectorCall: {
      const auto *A = cast<VectorCallAttr>(At);
      return A->clone(C);
    }
    case attr::Visibility: {
      return nullptr;
    }
    case attr::WarnUnused: {
      const auto *A = cast<WarnUnusedAttr>(At);
      return A->clone(C);
    }
    case attr::WarnUnusedResult: {
      const auto *A = cast<WarnUnusedResultAttr>(At);
      return A->clone(C);
    }
    case attr::Weak: {
      const auto *A = cast<WeakAttr>(At);
      return A->clone(C);
    }
    case attr::WeakImport: {
      const auto *A = cast<WeakImportAttr>(At);
      return A->clone(C);
    }
    case attr::WeakRef: {
      const auto *A = cast<WeakRefAttr>(At);
      return A->clone(C);
    }
    case attr::WebAssemblyExportName: {
      const auto *A = cast<WebAssemblyExportNameAttr>(At);
      return A->clone(C);
    }
    case attr::WebAssemblyFuncref: {
      const auto *A = cast<WebAssemblyFuncrefAttr>(At);
      return A->clone(C);
    }
    case attr::WebAssemblyImportModule: {
      const auto *A = cast<WebAssemblyImportModuleAttr>(At);
      return A->clone(C);
    }
    case attr::WebAssemblyImportName: {
      const auto *A = cast<WebAssemblyImportNameAttr>(At);
      return A->clone(C);
    }
    case attr::WorkGroupSizeHint: {
      const auto *A = cast<WorkGroupSizeHintAttr>(At);
      return A->clone(C);
    }
    case attr::X86ForceAlignArgPointer: {
      const auto *A = cast<X86ForceAlignArgPointerAttr>(At);
      return A->clone(C);
    }
    case attr::XRayInstrument: {
      const auto *A = cast<XRayInstrumentAttr>(At);
      return A->clone(C);
    }
    case attr::XRayLogArgs: {
      const auto *A = cast<XRayLogArgsAttr>(At);
      return A->clone(C);
    }
    case attr::ZeroCallUsedRegs: {
      const auto *A = cast<ZeroCallUsedRegsAttr>(At);
      return A->clone(C);
    }
  } // end switch
  llvm_unreachable("Unknown attribute!");
  return nullptr;
}

Attr *instantiateTemplateAttributeForDecl(const Attr *At,
 ASTContext &C, Sema &S,
        const MultiLevelTemplateArgumentList &TemplateArgs) {
  switch (At->getKind()) {
    case attr::AArch64SVEPcs: {
      return nullptr;
    }
    case attr::AArch64VectorPcs: {
      return nullptr;
    }
    case attr::AMDGPUFlatWorkGroupSize: {
      return nullptr;
    }
    case attr::AMDGPUKernelCall: {
      return nullptr;
    }
    case attr::AMDGPUMaxNumWorkGroups: {
      return nullptr;
    }
    case attr::AMDGPUNumSGPR: {
      return nullptr;
    }
    case attr::AMDGPUNumVGPR: {
      return nullptr;
    }
    case attr::AMDGPUWavesPerEU: {
      return nullptr;
    }
    case attr::ARMInterrupt: {
      return nullptr;
    }
    case attr::ARMInterruptSaveFP: {
      return nullptr;
    }
    case attr::ARMSaveFP: {
      return nullptr;
    }
    case attr::AVRInterrupt: {
      return nullptr;
    }
    case attr::AVRSignal: {
      return nullptr;
    }
    case attr::AbiTag: {
      const auto *A = cast<AbiTagAttr>(At);
      return A->clone(C);
    }
    case attr::AcquireCapability: {
      return nullptr;
    }
    case attr::AcquireHandle: {
      return nullptr;
    }
    case attr::AcquiredAfter: {
      return nullptr;
    }
    case attr::AcquiredBefore: {
      return nullptr;
    }
    case attr::AddressSpace: {
      return nullptr;
    }
    case attr::Alias: {
      return nullptr;
    }
    case attr::AlignMac68k: {
      return nullptr;
    }
    case attr::AlignNatural: {
      return nullptr;
    }
    case attr::AlignValue: {
      return nullptr;
    }
    case attr::Aligned: {
      return nullptr;
    }
    case attr::AllocAlign: {
      return nullptr;
    }
    case attr::AllocSize: {
      return nullptr;
    }
    case attr::Allocating: {
      return nullptr;
    }
    case attr::AlwaysDestroy: {
      return nullptr;
    }
    case attr::AlwaysInline: {
      return nullptr;
    }
    case attr::AnalyzerNoReturn: {
      return nullptr;
    }
    case attr::Annotate: {
      return nullptr;
    }
    case attr::AnnotateType: {
      return nullptr;
    }
    case attr::AnyX86Interrupt: {
      return nullptr;
    }
    case attr::AnyX86NoCallerSavedRegisters: {
      return nullptr;
    }
    case attr::AnyX86NoCfCheck: {
      return nullptr;
    }
    case attr::ArcWeakrefUnavailable: {
      return nullptr;
    }
    case attr::ArgumentWithTypeTag: {
      return nullptr;
    }
    case attr::ArmAgnostic: {
      return nullptr;
    }
    case attr::ArmBuiltinAlias: {
      return nullptr;
    }
    case attr::ArmIn: {
      return nullptr;
    }
    case attr::ArmInOut: {
      return nullptr;
    }
    case attr::ArmLocallyStreaming: {
      return nullptr;
    }
    case attr::ArmMveStrictPolymorphism: {
      return nullptr;
    }
    case attr::ArmNew: {
      return nullptr;
    }
    case attr::ArmOut: {
      return nullptr;
    }
    case attr::ArmPreserves: {
      return nullptr;
    }
    case attr::ArmStreaming: {
      return nullptr;
    }
    case attr::ArmStreamingCompatible: {
      return nullptr;
    }
    case attr::Artificial: {
      return nullptr;
    }
    case attr::AsmLabel: {
      return nullptr;
    }
    case attr::AssertCapability: {
      return nullptr;
    }
    case attr::AssumeAligned: {
      return nullptr;
    }
    case attr::Atomic: {
      return nullptr;
    }
    case attr::Availability: {
      return nullptr;
    }
    case attr::AvailableOnlyInDefaultEvalMethod: {
      return nullptr;
    }
    case attr::BPFFastCall: {
      return nullptr;
    }
    case attr::BPFPreserveAccessIndex: {
      return nullptr;
    }
    case attr::BPFPreserveStaticOffset: {
      return nullptr;
    }
    case attr::BTFDeclTag: {
      return nullptr;
    }
    case attr::BTFTypeTag: {
      return nullptr;
    }
    case attr::Blocking: {
      return nullptr;
    }
    case attr::Blocks: {
      return nullptr;
    }
    case attr::Builtin: {
      return nullptr;
    }
    case attr::BuiltinAlias: {
      return nullptr;
    }
    case attr::C11NoReturn: {
      return nullptr;
    }
    case attr::CDecl: {
      return nullptr;
    }
    case attr::CFAuditedTransfer: {
      return nullptr;
    }
    case attr::CFConsumed: {
      return nullptr;
    }
    case attr::CFGuard: {
      return nullptr;
    }
    case attr::CFICanonicalJumpTable: {
      return nullptr;
    }
    case attr::CFReturnsNotRetained: {
      return nullptr;
    }
    case attr::CFReturnsRetained: {
      return nullptr;
    }
    case attr::CFUnknownTransfer: {
      return nullptr;
    }
    case attr::CPUDispatch: {
      return nullptr;
    }
    case attr::CPUSpecific: {
      return nullptr;
    }
    case attr::CUDAConstant: {
      return nullptr;
    }
    case attr::CUDADevice: {
      return nullptr;
    }
    case attr::CUDADeviceBuiltinSurfaceType: {
      const auto *A = cast<CUDADeviceBuiltinSurfaceTypeAttr>(At);
      return A->clone(C);
    }
    case attr::CUDADeviceBuiltinTextureType: {
      const auto *A = cast<CUDADeviceBuiltinTextureTypeAttr>(At);
      return A->clone(C);
    }
    case attr::CUDAGlobal: {
      return nullptr;
    }
    case attr::CUDAGridConstant: {
      return nullptr;
    }
    case attr::CUDAHost: {
      return nullptr;
    }
    case attr::CUDAInvalidTarget: {
      return nullptr;
    }
    case attr::CUDALaunchBounds: {
      return nullptr;
    }
    case attr::CUDAShared: {
      return nullptr;
    }
    case attr::CXX11NoReturn: {
      return nullptr;
    }
    case attr::CXXAssume: {
      return nullptr;
    }
    case attr::CallableWhen: {
      return nullptr;
    }
    case attr::Callback: {
      return nullptr;
    }
    case attr::CalledOnce: {
      return nullptr;
    }
    case attr::Capability: {
      return nullptr;
    }
    case attr::CapturedRecord: {
      return nullptr;
    }
    case attr::CarriesDependency: {
      return nullptr;
    }
    case attr::Cleanup: {
      return nullptr;
    }
    case attr::ClspvLibclcBuiltin: {
      return nullptr;
    }
    case attr::CmseNSCall: {
      return nullptr;
    }
    case attr::CmseNSEntry: {
      return nullptr;
    }
    case attr::CodeAlign: {
      return nullptr;
    }
    case attr::CodeModel: {
      return nullptr;
    }
    case attr::CodeSeg: {
      return nullptr;
    }
    case attr::Cold: {
      return nullptr;
    }
    case attr::Common: {
      return nullptr;
    }
    case attr::Const: {
      return nullptr;
    }
    case attr::ConstInit: {
      return nullptr;
    }
    case attr::Constructor: {
      return nullptr;
    }
    case attr::Consumable: {
      return nullptr;
    }
    case attr::ConsumableAutoCast: {
      return nullptr;
    }
    case attr::ConsumableSetOnRead: {
      return nullptr;
    }
    case attr::Convergent: {
      return nullptr;
    }
    case attr::CoroAwaitElidable: {
      return nullptr;
    }
    case attr::CoroAwaitElidableArgument: {
      return nullptr;
    }
    case attr::CoroDisableLifetimeBound: {
      return nullptr;
    }
    case attr::CoroLifetimeBound: {
      return nullptr;
    }
    case attr::CoroOnlyDestroyWhenComplete: {
      return nullptr;
    }
    case attr::CoroReturnType: {
      return nullptr;
    }
    case attr::CoroWrapper: {
      return nullptr;
    }
    case attr::CountedBy: {
      return nullptr;
    }
    case attr::CountedByOrNull: {
      return nullptr;
    }
    case attr::DLLExport: {
      return nullptr;
    }
    case attr::DLLExportStaticLocal: {
      return nullptr;
    }
    case attr::DLLImport: {
      return nullptr;
    }
    case attr::DLLImportStaticLocal: {
      return nullptr;
    }
    case attr::Deprecated: {
      const auto *A = cast<DeprecatedAttr>(At);
      return A->clone(C);
    }
    case attr::Destructor: {
      return nullptr;
    }
    case attr::DiagnoseAsBuiltin: {
      return nullptr;
    }
    case attr::DiagnoseIf: {
      return nullptr;
    }
    case attr::DisableSanitizerInstrumentation: {
      return nullptr;
    }
    case attr::DisableTailCalls: {
      return nullptr;
    }
    case attr::EmptyBases: {
      return nullptr;
    }
    case attr::EnableIf: {
      return nullptr;
    }
    case attr::EnforceTCB: {
      return nullptr;
    }
    case attr::EnforceTCBLeaf: {
      return nullptr;
    }
    case attr::EnumExtensibility: {
      return nullptr;
    }
    case attr::Error: {
      return nullptr;
    }
    case attr::ExcludeFromExplicitInstantiation: {
      const auto *A = cast<ExcludeFromExplicitInstantiationAttr>(At);
      return A->clone(C);
    }
    case attr::ExplicitInit: {
      return nullptr;
    }
    case attr::ExtVectorType: {
      return nullptr;
    }
    case attr::ExternalSourceSymbol: {
      return nullptr;
    }
    case attr::FallThrough: {
      return nullptr;
    }
    case attr::FastCall: {
      return nullptr;
    }
    case attr::Final: {
      return nullptr;
    }
    case attr::FlagEnum: {
      return nullptr;
    }
    case attr::Flatten: {
      return nullptr;
    }
    case attr::Format: {
      return nullptr;
    }
    case attr::FormatArg: {
      return nullptr;
    }
    case attr::FormatMatches: {
      return nullptr;
    }
    case attr::FunctionReturnThunks: {
      return nullptr;
    }
    case attr::GNUInline: {
      return nullptr;
    }
    case attr::GuardedBy: {
      return nullptr;
    }
    case attr::GuardedVar: {
      return nullptr;
    }
    case attr::HIPManaged: {
      return nullptr;
    }
    case attr::HLSLContainedType: {
      return nullptr;
    }
    case attr::HLSLControlFlowHint: {
      return nullptr;
    }
    case attr::HLSLGroupSharedAddressSpace: {
      return nullptr;
    }
    case attr::HLSLLoopHint: {
      return nullptr;
    }
    case attr::HLSLNumThreads: {
      return nullptr;
    }
    case attr::HLSLPackOffset: {
      return nullptr;
    }
    case attr::HLSLParamModifier: {
      return nullptr;
    }
    case attr::HLSLROV: {
      return nullptr;
    }
    case attr::HLSLRawBuffer: {
      return nullptr;
    }
    case attr::HLSLResourceBinding: {
      return nullptr;
    }
    case attr::HLSLResourceClass: {
      return nullptr;
    }
    case attr::HLSLSV_DispatchThreadID: {
      return nullptr;
    }
    case attr::HLSLSV_GroupID: {
      return nullptr;
    }
    case attr::HLSLSV_GroupIndex: {
      return nullptr;
    }
    case attr::HLSLSV_GroupThreadID: {
      return nullptr;
    }
    case attr::HLSLShader: {
      return nullptr;
    }
    case attr::HLSLWaveSize: {
      return nullptr;
    }
    case attr::Hot: {
      return nullptr;
    }
    case attr::HybridPatchable: {
      return nullptr;
    }
    case attr::IBAction: {
      return nullptr;
    }
    case attr::IBOutlet: {
      return nullptr;
    }
    case attr::IBOutletCollection: {
      return nullptr;
    }
    case attr::IFunc: {
      return nullptr;
    }
    case attr::InitPriority: {
      return nullptr;
    }
    case attr::InitSeg: {
      return nullptr;
    }
    case attr::IntelOclBicc: {
      return nullptr;
    }
    case attr::InternalLinkage: {
      return nullptr;
    }
    case attr::LTOVisibilityPublic: {
      return nullptr;
    }
    case attr::LayoutVersion: {
      return nullptr;
    }
    case attr::Leaf: {
      return nullptr;
    }
    case attr::LifetimeBound: {
      return nullptr;
    }
    case attr::LifetimeCaptureBy: {
      return nullptr;
    }
    case attr::Likely: {
      return nullptr;
    }
    case attr::LoaderUninitialized: {
      return nullptr;
    }
    case attr::LockReturned: {
      return nullptr;
    }
    case attr::LocksExcluded: {
      return nullptr;
    }
    case attr::LoopHint: {
      return nullptr;
    }
    case attr::M68kInterrupt: {
      return nullptr;
    }
    case attr::M68kRTD: {
      return nullptr;
    }
    case attr::MIGServerRoutine: {
      return nullptr;
    }
    case attr::MSABI: {
      return nullptr;
    }
    case attr::MSAllocator: {
      return nullptr;
    }
    case attr::MSConstexpr: {
      return nullptr;
    }
    case attr::MSInheritance: {
      return nullptr;
    }
    case attr::MSNoVTable: {
      return nullptr;
    }
    case attr::MSP430Interrupt: {
      return nullptr;
    }
    case attr::MSStruct: {
      return nullptr;
    }
    case attr::MSVtorDisp: {
      return nullptr;
    }
    case attr::MaxFieldAlignment: {
      return nullptr;
    }
    case attr::MayAlias: {
      return nullptr;
    }
    case attr::MaybeUndef: {
      return nullptr;
    }
    case attr::MicroMips: {
      return nullptr;
    }
    case attr::MinSize: {
      return nullptr;
    }
    case attr::MinVectorWidth: {
      return nullptr;
    }
    case attr::Mips16: {
      return nullptr;
    }
    case attr::MipsInterrupt: {
      return nullptr;
    }
    case attr::MipsLongCall: {
      return nullptr;
    }
    case attr::MipsShortCall: {
      return nullptr;
    }
    case attr::Mode: {
      return nullptr;
    }
    case attr::MustTail: {
      return nullptr;
    }
    case attr::NSConsumed: {
      return nullptr;
    }
    case attr::NSConsumesSelf: {
      return nullptr;
    }
    case attr::NSErrorDomain: {
      return nullptr;
    }
    case attr::NSReturnsAutoreleased: {
      return nullptr;
    }
    case attr::NSReturnsNotRetained: {
      return nullptr;
    }
    case attr::NSReturnsRetained: {
      return nullptr;
    }
    case attr::NVPTXKernel: {
      return nullptr;
    }
    case attr::Naked: {
      return nullptr;
    }
    case attr::NoAlias: {
      return nullptr;
    }
    case attr::NoBuiltin: {
      return nullptr;
    }
    case attr::NoCommon: {
      return nullptr;
    }
    case attr::NoConvergent: {
      return nullptr;
    }
    case attr::NoDebug: {
      return nullptr;
    }
    case attr::NoDeref: {
      return nullptr;
    }
    case attr::NoDestroy: {
      return nullptr;
    }
    case attr::NoDuplicate: {
      return nullptr;
    }
    case attr::NoEscape: {
      return nullptr;
    }
    case attr::NoInline: {
      return nullptr;
    }
    case attr::NoInstrumentFunction: {
      return nullptr;
    }
    case attr::NoMerge: {
      return nullptr;
    }
    case attr::NoMicroMips: {
      return nullptr;
    }
    case attr::NoMips16: {
      return nullptr;
    }
    case attr::NoProfileFunction: {
      return nullptr;
    }
    case attr::NoRandomizeLayout: {
      return nullptr;
    }
    case attr::NoReturn: {
      return nullptr;
    }
    case attr::NoSanitize: {
      return nullptr;
    }
    case attr::NoSpecializations: {
      const auto *A = cast<NoSpecializationsAttr>(At);
      return new (C) NoSpecializationsAttr(C, *A, A->getMessage());
    }
    case attr::NoSpeculativeLoadHardening: {
      return nullptr;
    }
    case attr::NoSplitStack: {
      return nullptr;
    }
    case attr::NoStackProtector: {
      return nullptr;
    }
    case attr::NoThreadSafetyAnalysis: {
      return nullptr;
    }
    case attr::NoThrow: {
      return nullptr;
    }
    case attr::NoTrivialAutoVarInit: {
      return nullptr;
    }
    case attr::NoUniqueAddress: {
      return nullptr;
    }
    case attr::NoUwtable: {
      return nullptr;
    }
    case attr::NonAllocating: {
      return nullptr;
    }
    case attr::NonBlocking: {
      return nullptr;
    }
    case attr::NonNull: {
      return nullptr;
    }
    case attr::NotTailCalled: {
      return nullptr;
    }
    case attr::OMPAllocateDecl: {
      return nullptr;
    }
    case attr::OMPAssume: {
      return nullptr;
    }
    case attr::OMPCaptureKind: {
      return nullptr;
    }
    case attr::OMPCaptureNoInit: {
      return nullptr;
    }
    case attr::OMPDeclareSimdDecl: {
      return nullptr;
    }
    case attr::OMPDeclareTargetDecl: {
      return nullptr;
    }
    case attr::OMPDeclareVariant: {
      return nullptr;
    }
    case attr::OMPReferencedVar: {
      return nullptr;
    }
    case attr::OMPThreadPrivateDecl: {
      return nullptr;
    }
    case attr::OSConsumed: {
      return nullptr;
    }
    case attr::OSConsumesThis: {
      return nullptr;
    }
    case attr::OSReturnsNotRetained: {
      return nullptr;
    }
    case attr::OSReturnsRetained: {
      return nullptr;
    }
    case attr::OSReturnsRetainedOnNonZero: {
      return nullptr;
    }
    case attr::OSReturnsRetainedOnZero: {
      return nullptr;
    }
    case attr::ObjCBoxable: {
      return nullptr;
    }
    case attr::ObjCBridge: {
      return nullptr;
    }
    case attr::ObjCBridgeMutable: {
      return nullptr;
    }
    case attr::ObjCBridgeRelated: {
      return nullptr;
    }
    case attr::ObjCClassStub: {
      return nullptr;
    }
    case attr::ObjCDesignatedInitializer: {
      return nullptr;
    }
    case attr::ObjCDirect: {
      return nullptr;
    }
    case attr::ObjCDirectMembers: {
      return nullptr;
    }
    case attr::ObjCException: {
      return nullptr;
    }
    case attr::ObjCExplicitProtocolImpl: {
      return nullptr;
    }
    case attr::ObjCExternallyRetained: {
      return nullptr;
    }
    case attr::ObjCGC: {
      return nullptr;
    }
    case attr::ObjCIndependentClass: {
      return nullptr;
    }
    case attr::ObjCInertUnsafeUnretained: {
      return nullptr;
    }
    case attr::ObjCKindOf: {
      return nullptr;
    }
    case attr::ObjCMethodFamily: {
      return nullptr;
    }
    case attr::ObjCNSObject: {
      return nullptr;
    }
    case attr::ObjCNonLazyClass: {
      return nullptr;
    }
    case attr::ObjCNonRuntimeProtocol: {
      return nullptr;
    }
    case attr::ObjCOwnership: {
      return nullptr;
    }
    case attr::ObjCPreciseLifetime: {
      return nullptr;
    }
    case attr::ObjCRequiresPropertyDefs: {
      return nullptr;
    }
    case attr::ObjCRequiresSuper: {
      return nullptr;
    }
    case attr::ObjCReturnsInnerPointer: {
      return nullptr;
    }
    case attr::ObjCRootClass: {
      return nullptr;
    }
    case attr::ObjCRuntimeName: {
      return nullptr;
    }
    case attr::ObjCRuntimeVisible: {
      return nullptr;
    }
    case attr::ObjCSubclassingRestricted: {
      return nullptr;
    }
    case attr::OpenACCRoutineAnnot: {
      return nullptr;
    }
    case attr::OpenACCRoutineDecl: {
      return nullptr;
    }
    case attr::OpenCLAccess: {
      return nullptr;
    }
    case attr::OpenCLConstantAddressSpace: {
      return nullptr;
    }
    case attr::OpenCLGenericAddressSpace: {
      return nullptr;
    }
    case attr::OpenCLGlobalAddressSpace: {
      return nullptr;
    }
    case attr::OpenCLGlobalDeviceAddressSpace: {
      return nullptr;
    }
    case attr::OpenCLGlobalHostAddressSpace: {
      return nullptr;
    }
    case attr::OpenCLIntelReqdSubGroupSize: {
      return nullptr;
    }
    case attr::OpenCLKernel: {
      return nullptr;
    }
    case attr::OpenCLLocalAddressSpace: {
      return nullptr;
    }
    case attr::OpenCLPrivateAddressSpace: {
      return nullptr;
    }
    case attr::OpenCLUnrollHint: {
      return nullptr;
    }
    case attr::OptimizeNone: {
      return nullptr;
    }
    case attr::Overloadable: {
      return nullptr;
    }
    case attr::Override: {
      return nullptr;
    }
    case attr::Owner: {
      return nullptr;
    }
    case attr::Ownership: {
      return nullptr;
    }
    case attr::Packed: {
      return nullptr;
    }
    case attr::ParamTypestate: {
      return nullptr;
    }
    case attr::Pascal: {
      return nullptr;
    }
    case attr::PassObjectSize: {
      return nullptr;
    }
    case attr::PatchableFunctionEntry: {
      return nullptr;
    }
    case attr::Pcs: {
      return nullptr;
    }
    case attr::Pointer: {
      return nullptr;
    }
    case attr::PointerAuth: {
      return nullptr;
    }
    case attr::PragmaClangBSSSection: {
      return nullptr;
    }
    case attr::PragmaClangDataSection: {
      return nullptr;
    }
    case attr::PragmaClangRelroSection: {
      return nullptr;
    }
    case attr::PragmaClangRodataSection: {
      return nullptr;
    }
    case attr::PragmaClangTextSection: {
      return nullptr;
    }
    case attr::PreferredName: {
      const auto *A = cast<PreferredNameAttr>(At);
      TypeSourceInfo * tempInstTypedefType =
        S.SubstType(A->getTypedefTypeLoc(), TemplateArgs, A->getLoc(), A->getAttrName());
      if (!tempInstTypedefType)
        return nullptr;
      return new (C) PreferredNameAttr(C, *A, tempInstTypedefType);
    }
    case attr::PreferredType: {
      return nullptr;
    }
    case attr::PreserveAll: {
      return nullptr;
    }
    case attr::PreserveMost: {
      return nullptr;
    }
    case attr::PreserveNone: {
      return nullptr;
    }
    case attr::PtGuardedBy: {
      return nullptr;
    }
    case attr::PtGuardedVar: {
      return nullptr;
    }
    case attr::Ptr32: {
      return nullptr;
    }
    case attr::Ptr64: {
      return nullptr;
    }
    case attr::Pure: {
      return nullptr;
    }
    case attr::RISCVInterrupt: {
      return nullptr;
    }
    case attr::RISCVVLSCC: {
      return nullptr;
    }
    case attr::RISCVVectorCC: {
      return nullptr;
    }
    case attr::RandomizeLayout: {
      return nullptr;
    }
    case attr::ReadOnlyPlacement: {
      return nullptr;
    }
    case attr::RegCall: {
      return nullptr;
    }
    case attr::Reinitializes: {
      return nullptr;
    }
    case attr::ReleaseCapability: {
      return nullptr;
    }
    case attr::ReleaseHandle: {
      return nullptr;
    }
    case attr::ReqdWorkGroupSize: {
      return nullptr;
    }
    case attr::RequiresCapability: {
      return nullptr;
    }
    case attr::Restrict: {
      return nullptr;
    }
    case attr::Retain: {
      return nullptr;
    }
    case attr::ReturnTypestate: {
      return nullptr;
    }
    case attr::ReturnsNonNull: {
      return nullptr;
    }
    case attr::ReturnsTwice: {
      return nullptr;
    }
    case attr::SPtr: {
      return nullptr;
    }
    case attr::SYCLKernel: {
      return nullptr;
    }
    case attr::SYCLKernelEntryPoint: {
      return nullptr;
    }
    case attr::SYCLSpecialClass: {
      return nullptr;
    }
    case attr::ScopedLockable: {
      return nullptr;
    }
    case attr::Section: {
      return nullptr;
    }
    case attr::SelectAny: {
      return nullptr;
    }
    case attr::Sentinel: {
      return nullptr;
    }
    case attr::SetTypestate: {
      return nullptr;
    }
    case attr::SizedBy: {
      return nullptr;
    }
    case attr::SizedByOrNull: {
      return nullptr;
    }
    case attr::SpeculativeLoadHardening: {
      return nullptr;
    }
    case attr::StandaloneDebug: {
      return nullptr;
    }
    case attr::StdCall: {
      return nullptr;
    }
    case attr::StrictFP: {
      return nullptr;
    }
    case attr::StrictGuardStackCheck: {
      return nullptr;
    }
    case attr::Suppress: {
      return nullptr;
    }
    case attr::SwiftAsync: {
      return nullptr;
    }
    case attr::SwiftAsyncCall: {
      return nullptr;
    }
    case attr::SwiftAsyncContext: {
      return nullptr;
    }
    case attr::SwiftAsyncError: {
      return nullptr;
    }
    case attr::SwiftAsyncName: {
      return nullptr;
    }
    case attr::SwiftAttr: {
      return nullptr;
    }
    case attr::SwiftBridge: {
      return nullptr;
    }
    case attr::SwiftBridgedTypedef: {
      return nullptr;
    }
    case attr::SwiftCall: {
      return nullptr;
    }
    case attr::SwiftContext: {
      return nullptr;
    }
    case attr::SwiftError: {
      return nullptr;
    }
    case attr::SwiftErrorResult: {
      return nullptr;
    }
    case attr::SwiftImportAsNonGeneric: {
      return nullptr;
    }
    case attr::SwiftImportPropertyAsAccessors: {
      return nullptr;
    }
    case attr::SwiftIndirectResult: {
      return nullptr;
    }
    case attr::SwiftName: {
      return nullptr;
    }
    case attr::SwiftNewType: {
      return nullptr;
    }
    case attr::SwiftObjCMembers: {
      return nullptr;
    }
    case attr::SwiftPrivate: {
      return nullptr;
    }
    case attr::SwiftVersionedAddition: {
      return nullptr;
    }
    case attr::SwiftVersionedRemoval: {
      return nullptr;
    }
    case attr::SysVABI: {
      return nullptr;
    }
    case attr::TLSModel: {
      return nullptr;
    }
    case attr::Target: {
      return nullptr;
    }
    case attr::TargetClones: {
      return nullptr;
    }
    case attr::TargetVersion: {
      return nullptr;
    }
    case attr::TestTypestate: {
      return nullptr;
    }
    case attr::ThisCall: {
      return nullptr;
    }
    case attr::Thread: {
      return nullptr;
    }
    case attr::TransparentUnion: {
      return nullptr;
    }
    case attr::TrivialABI: {
      return nullptr;
    }
    case attr::TryAcquireCapability: {
      return nullptr;
    }
    case attr::TypeNonNull: {
      return nullptr;
    }
    case attr::TypeNullUnspecified: {
      return nullptr;
    }
    case attr::TypeNullable: {
      return nullptr;
    }
    case attr::TypeNullableResult: {
      return nullptr;
    }
    case attr::TypeTagForDatatype: {
      return nullptr;
    }
    case attr::TypeVisibility: {
      return nullptr;
    }
    case attr::UPtr: {
      return nullptr;
    }
    case attr::Unavailable: {
      const auto *A = cast<UnavailableAttr>(At);
      return A->clone(C);
    }
    case attr::Uninitialized: {
      return nullptr;
    }
    case attr::Unlikely: {
      return nullptr;
    }
    case attr::UnsafeBufferUsage: {
      return nullptr;
    }
    case attr::Unused: {
      return nullptr;
    }
    case attr::UseHandle: {
      return nullptr;
    }
    case attr::Used: {
      return nullptr;
    }
    case attr::UsingIfExists: {
      return nullptr;
    }
    case attr::Uuid: {
      return nullptr;
    }
    case attr::VTablePointerAuthentication: {
      return nullptr;
    }
    case attr::VecReturn: {
      return nullptr;
    }
    case attr::VecTypeHint: {
      return nullptr;
    }
    case attr::VectorCall: {
      return nullptr;
    }
    case attr::Visibility: {
      return nullptr;
    }
    case attr::WarnUnused: {
      return nullptr;
    }
    case attr::WarnUnusedResult: {
      return nullptr;
    }
    case attr::Weak: {
      return nullptr;
    }
    case attr::WeakImport: {
      return nullptr;
    }
    case attr::WeakRef: {
      return nullptr;
    }
    case attr::WebAssemblyExportName: {
      return nullptr;
    }
    case attr::WebAssemblyFuncref: {
      return nullptr;
    }
    case attr::WebAssemblyImportModule: {
      return nullptr;
    }
    case attr::WebAssemblyImportName: {
      return nullptr;
    }
    case attr::WorkGroupSizeHint: {
      return nullptr;
    }
    case attr::X86ForceAlignArgPointer: {
      return nullptr;
    }
    case attr::XRayInstrument: {
      return nullptr;
    }
    case attr::XRayLogArgs: {
      return nullptr;
    }
    case attr::ZeroCallUsedRegs: {
      return nullptr;
    }
  } // end switch
  llvm_unreachable("Unknown attribute!");
  return nullptr;
}

} // end namespace sema
} // end namespace clang
