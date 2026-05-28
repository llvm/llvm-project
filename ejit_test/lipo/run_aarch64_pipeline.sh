#!/bin/bash
# EJIT aarch64 ejit.o pipeline — trimmed for bare-metal.
# Run from the llvm-project root, after building:
#   ninja -C build_release_aarch64 LLVMEJIT
#
# Uses the x86_64 ld.lld as cross-linker (lld handles any ELF arch).

set -euo pipefail

BUILD_DIR="${1:-build_release_aarch64}"
LD="${2:-build_release_x86/bin/ld.lld}"
CXX="${3:-aarch64-linux-gnu-g++}"
OUTPUT="${4:-ejit_test/lipo/ejit_aarch64.o}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR="$SCRIPT_DIR/.lipo_work_aarch64"

rm -rf "$WORK_DIR"

# ── Exclude list ──────────────────────────────────────────────────────────────
# These .o files are not needed for bare-metal EJIT runtime:
#
# OS/format-specific:
#   WinEH/WasmEH/Windows*/CFGuard/MSVCPaths — Windows-only
#   RuntimeDyldCOFF/MachO — non-ELF runtimes
#   COFFPlatform/MachOPlatform/COFFVCRuntimeSupport/MachOObjectFormat — non-ELF
#
# Optimization passes not used by EJIT pipeline:
#   ScalarOpts: LoopDataPrefetch, ScalarizeMaskedMemIntrin,
#     PartiallyInlineLibCalls, MergeICmps, LowerConstantIntrinsics,
#     ConstantHoisting, EarlyCSE, LICM, LoopStrengthReduce, LoopTermFold,
#     SeparateConstOffsetFromGEP
#   TransformUtils: BypassSlowDivision, Debugify, EntryExitInstrumenter,
#     LowerIFunc, LowerGlobalDtors, SampleProfileInference,
#     SampleProfileLoaderBaseUtil, PredicateInfo, CanonicalizeFreezeInLoops,
#     EscapeEnumerator, IntegerDivision, LowerAtomic, LowerInvoke,
#     LowerMemIntrinsics, LowerVectorIntrinsics, MemoryTaggingSupport,
#     SizeOpts, AssumeBundleBuilder, DemoteRegToStack
#
# CodeGen machine passes not used by EJIT (uses its own minimal pipeline):
#   Stack protector/ShadowStack/SafeStack, GlobalMerge,
#   IndirectBr/ImplicitNullChecks, MachineFunctionSplitter, ResetMachine,
#   PseudoProbe*, ExpandLargeDivRem/ExpandMemCmp/ExpandReductions,
#   InterleavedAccess/InterleavedLoadCombine, MachinePipeliner,
#   MachineTraceMetrics, LiveDebugValues/Variables, LiveIntervalCalc,
#   LiveRangeShrink, LiveRegMatrix/RegUnits/Stacks, MachineLateInstrsCleanup,
#   MachineVerifier, DetectDeadLanes, DeadMachineInstructionElim,
#   AggressiveAntiDepBreaker/CriticalAntiDepBreaker, BreakFalseDeps,
#   CalcSpillWeights, RenameIndependentSubregs, RegisterCoalescer,
#   TailDuplication/Duplicator, UnreachableBlockElim, BasicBlockPathCloning,
#   LocalStackSlotAllocation, ShrinkWrap, LazyMachineBlockFrequencyInfo,
#   MachineBlockFrequencyInfo, MachineCSE/Sink/CopyPropagation,
#   MachineCombiner, RegAllocBasic/PBQP, MLRegAlloc*, MemProf*,
#   IndexedMemProfData, DataAccessProf, SymbolRemappingReader,
#   ItaniumManglingCanonicalizer, RegAllocEvictionAdvisor,
#   RegAllocPriorityAdvisor, LiveInterval/Intervals/IntervalUnion,
#   LiveRangeCalc/Edit, SlotIndexes, VirtRegMap,
#   TwoAddressInstructionPass, PHIEliminationUtils,
#   MachineBlockPlacement, BranchFolding/Relaxation,
#   EarlyIfConversion/IfConversion, PrologEpilogInserter
#
# MIR debug/metadata not needed at runtime:
#   MIRCanonicalizerPass, MIRFSDiscriminator, MIRNamerPass,
#   MIRPrintingPass, MIRPrinter, MIRSampleProfile, MIRVRegNamerUtils
#
# GC/Instrumentation/Sanitizer/misc:
#   GCMetadata/Printer, GCRootLowering, GCEmptyBasicBlocks,
#   HardwareLoops, TypePromotion, ReplaceWithVeclib,
#   ComplexDeinterleavingPass, SelectOptimize, JMCInstrumenter,
#   XRayInstrumentation, SanitizerBinaryMetadata, SwiftErrorValueTracking,
#   SjLjEHPrepare, EHContGuardTargets, DwarfEHPrepare, KCFI, FaultMaps,
#   StackMapLivenessAnalysis/StackMaps, PatchableFunction, FEntryInserter,
#   FuncletLayout, BasicBlockSections*, RemoveRedundantDebugValues,
#   DroppedVariableStatsMIR, RegUsageInfo*, RegisterUsageInfo,
#   ScoreboardHazardRecognizer, PostRAHazardRecognizer

EXCLUDES=(
  # OS/format
  InstrProfCorrelator  WinEHPrepare  WindowScheduler
  WindowsSecureHotPatching  WasmEHPrepare
  CFGuard  MSVCPaths
  RuntimeDyldCOFF  RuntimeDyldMachO
  MachOObjectFormat
  COFFPlatform  MachOPlatform  COFFVCRuntimeSupport

  # ScalarOpts unused
  LoopDataPrefetch  ScalarizeMaskedMemIntrin
  PartiallyInlineLibCalls  MergeICmps  LowerConstantIntrinsics
  ConstantHoisting  EarlyCSE  LICM
  LoopStrengthReduce  LoopTermFold  SeparateConstOffsetFromGEP

  # TransformUtils unused
  BypassSlowDivision  Debugify  EntryExitInstrumenter
  LowerIFunc  LowerGlobalDtors
  SampleProfileInference  SampleProfileLoaderBaseUtil
  PredicateInfo  CanonicalizeFreezeInLoops
  EscapeEnumerator  IntegerDivision  LowerAtomic
  LowerInvoke  LowerMemIntrinsics  LowerVectorIntrinsics
  MemoryTaggingSupport  SizeOpts  AssumeBundleBuilder  DemoteRegToStack

  # CodeGen machine passes unused
  ShadowStackGCLowering  SafeStack  SafeStackLayout  StackProtector
  GlobalMerge  GlobalMergeFunctions
  IndirectBrExpandPass  ImplicitNullChecks
  MachineFunctionSplitter  ResetMachineFunctionPass
  PseudoProbePrinter  PseudoProbeInserter
  ExpandLargeDivRem  ExpandMemCmp  ExpandReductions
  InterleavedAccessPass  InterleavedLoadCombinePass
  MachinePipeliner  MachineTraceMetrics
  LiveDebugValues  LiveDebugVariables
  LiveIntervalCalc  LiveRangeShrink
  LiveRegMatrix  LiveRegUnits  LiveStacks
  MachineLateInstrsCleanup  MachineVerifier
  DetectDeadLanes  DeadMachineInstructionElim
  AggressiveAntiDepBreaker  CriticalAntiDepBreaker
  BreakFalseDeps  CalcSpillWeights
  RenameIndependentSubregs  RegisterCoalescer
  TailDuplication  TailDuplicator
  UnreachableBlockElim  BasicBlockPathCloning
  LocalStackSlotAllocation  ShrinkWrap
  LazyMachineBlockFrequencyInfo  MachineBlockFrequencyInfo
  MachineCSE  MachineSink  MachineCopyPropagation
  MachineCombiner  RegAllocBasic  RegAllocPBQP
  MLRegAllocEvictAdvisor  MLRegAllocPriorityAdvisor
  MemProf  MemProfRadixTree  MemProfSummary
  IndexedMemProfData  DataAccessProf
  SymbolRemappingReader  ItaniumManglingCanonicalizer
  RegAllocEvictionAdvisor  RegAllocPriorityAdvisor
  LiveInterval  LiveIntervals  LiveIntervalUnion
  LiveRangeCalc  LiveRangeEdit
  SlotIndexes  VirtRegMap  TwoAddressInstructionPass
  PHIEliminationUtils
  MachineBlockPlacement  BranchFolding  BranchRelaxation
  EarlyIfConversion  IfConversion  PrologEpilogInserter

  # MIR / debug metadata
  MIRCanonicalizerPass  MIRFSDiscriminator
  MIRNamerPass  MIRPrintingPass  MIRPrinter
  MIRSampleProfile  MIRVRegNamerUtils

  # GC / instrumentation / misc
  GCMetadata  GCMetadataPrinter  GCRootLowering  GCEmptyBasicBlocks
  HardwareLoops  TypePromotion  ReplaceWithVeclib
  ComplexDeinterleavingPass  SelectOptimize  JMCInstrumenter
  XRayInstrumentation  SanitizerBinaryMetadata
  SwiftErrorValueTracking  SjLjEHPrepare  EHContGuardTargets
  DwarfEHPrepare  KCFI  FaultMaps
  StackMapLivenessAnalysis  StackMaps  PatchableFunction
  FEntryInserter  FuncletLayout  BasicBlockSections
  BasicBlockSectionsProfileReader
  RemoveRedundantDebugValues  DroppedVariableStatsMIR
  RegUsageInfoCollector  RegUsageInfoPropagate
  RegisterUsageInfo  ScoreboardHazardRecognizer
  PostRAHazardRecognizer
)

EXCLUDE_FLAGS=""
for x in "${EXCLUDES[@]}"; do
  EXCLUDE_FLAGS="$EXCLUDE_FLAGS --exclude=$x"
done

# ── Step 1: extract ───────────────────────────────────────────────────────────
python3 "$SCRIPT_DIR/lipo.py" extract \
  --arch=aarch64 --build-dir="$BUILD_DIR" \
  --cxx="$CXX" --ld="$LD" \
  $EXCLUDE_FLAGS

# ── Step 2: gc-merge ─────────────────────────────────────────────────────────
python3 "$SCRIPT_DIR/lipo.py" gc-merge \
  --input="$SCRIPT_DIR/libejit_lipo_aarch64.a" \
  --build-dir="$BUILD_DIR" --ld="$LD"

# ── Step 3: merge ─────────────────────────────────────────────────────────────
python3 "$SCRIPT_DIR/lipo.py" merge \
  --input="$SCRIPT_DIR/libejit_lipo_aarch64_gc.a" \
  --build-dir="$BUILD_DIR" --ld="$LD" \
  --output="$OUTPUT"

echo ""
echo "Done: $(ls -lh "$OUTPUT" | awk '{print $5, $NF}')"
size -A "$OUTPUT" 2>/dev/null | head -12
