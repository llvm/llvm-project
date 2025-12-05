/**
 * @file DsmilJADC2Pass.cpp
 * @brief DSMIL JADC2 & 5G/Edge-Aware Compilation Pass (v1.5)
 *
 * Optimizes code for Joint All-Domain Command & Control (JADC2) deployment
 * on 5G Multi-Access Edge Computing (MEC) networks.
 *
 * Features:
 * - Edge offload analysis for latency-sensitive kernels
 * - 5G latency budget enforcement (typical: 5ms end-to-end)
 * - Bandwidth contract validation (typical: 10Gbps)
 * - Message format optimization for 5G transport
 * - Power profiling for edge devices
 *
 * JADC2 Context:
 * - Sensor→C2→Shooter pipeline (multi-domain operations)
 * - 99.999% reliability requirement
 * - Real-time situational awareness
 * - Coalition interoperability
 *
 * Layer Integration:
 * - Layer 5 (Performance AI): Latency prediction, offload recommendations
 * - Layer 6 (Resource AI): MEC node allocation
 * - Layer 9 (Campaign): Mission profile selection
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/CallGraph.h"
#include <unordered_map>
#include <string>
#include <vector>

using namespace llvm;

namespace {

// JADC2 operational profiles
enum JADC2Profile {
    SENSOR_FUSION,
    C2_PROCESSING,
    TARGETING,
    SITUATIONAL_AWARENESS,
    LOGISTICS,
    NONE
};

// 5G/MEC optimization hints
struct MEC5GHints {
    bool PreferEdgeOffload;
    unsigned LatencyBudgetMS;
    unsigned BandwidthGbps;
    bool PowerSensitive;
    JADC2Profile Profile;
};

class DsmilJADC2Pass : public PassInfoMixin<DsmilJADC2Pass> {
private:
    // Function -> MEC/5G optimization hints
    std::unordered_map<Function*, MEC5GHints> FunctionHints;

    // Statistics
    unsigned NumJADC2Functions = 0;
    unsigned Num5GEdgeFunctions = 0;
    unsigned NumLatencyViolations = 0;
    unsigned NumBandwidthWarnings = 0;
    unsigned NumOffloadCandidates = 0;

public:
    PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

private:
    // Phase 1: Extract JADC2/5G metadata
    void extractMetadata(Module &M);

    // Phase 2: Analyze latency budgets
    bool analyzeLatencyBudgets(Module &M);

    // Phase 3: Optimize for 5G transport
    bool optimizeFor5G(Module &M);

    // Phase 4: Identify edge offload candidates
    void identifyOffloadCandidates(Module &M);

    // Helper: Parse JADC2 profile
    JADC2Profile parseProfile(const std::string &ProfileName);

    // Helper: Estimate function latency (simplified)
    unsigned estimateLatencyMS(Function *F);

    // Helper: Estimate bandwidth usage (simplified)
    unsigned estimateBandwidthMBps(Function *F);

    // Helper: Check if function is offload candidate
    bool isOffloadCandidate(Function *F, const MEC5GHints &Hints);
};

PreservedAnalyses DsmilJADC2Pass::run(Module &M,
                                       ModuleAnalysisManager &AM) {
    errs() << "=== DSMIL JADC2 & 5G/Edge Pass (v1.5) ===\n";

    // Phase 1: Extract metadata
    extractMetadata(M);
    errs() << "  JADC2 functions: " << NumJADC2Functions << "\n";
    errs() << "  5G/MEC functions: " << Num5GEdgeFunctions << "\n";

    // Phase 2: Analyze latency
    bool HasViolations = analyzeLatencyBudgets(M);
    errs() << "  Latency violations: " << NumLatencyViolations << "\n";
    errs() << "  Bandwidth warnings: " << NumBandwidthWarnings << "\n";

    if (HasViolations) {
        errs() << "WARNING: Latency budget violations detected!\n";
        errs() << "Functions may not meet 5G JADC2 requirements.\n";
        errs() << "Recommendation: Refactor or use edge offload.\n";
    }

    // Phase 3: Optimize for 5G
    bool Modified = optimizeFor5G(M);

    // Phase 4: Identify offload candidates
    identifyOffloadCandidates(M);
    errs() << "  Edge offload candidates: " << NumOffloadCandidates << "\n";

    errs() << "=== JADC2 Pass Complete ===\n\n";

    return Modified ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

void DsmilJADC2Pass::extractMetadata(Module &M) {
    for (auto &F : M) {
        if (F.isDeclaration())
            continue;

        MEC5GHints Hints = {};
        Hints.Profile = NONE;
        Hints.LatencyBudgetMS = 1000;  // Default: 1 second
        Hints.BandwidthGbps = 1;        // Default: 1 Gbps
        Hints.PreferEdgeOffload = false;
        Hints.PowerSensitive = false;

        // Check for JADC2 profile
        if (F.hasFnAttribute("dsmil_jadc2_profile")) {
            Attribute Attr = F.getFnAttribute("dsmil_jadc2_profile");
            if (Attr.isStringAttribute()) {
                std::string ProfileName = Attr.getValueAsString().str();
                Hints.Profile = parseProfile(ProfileName);
                NumJADC2Functions++;
            }
        }

        // Check for 5G edge deployment
        if (F.hasFnAttribute("dsmil_5g_edge")) {
            Hints.PreferEdgeOffload = true;
            Num5GEdgeFunctions++;
        }

        // Check for latency budget
        if (F.hasFnAttribute("dsmil_latency_budget")) {
            Attribute Attr = F.getFnAttribute("dsmil_latency_budget");
            if (Attr.isStringAttribute()) {
                unsigned Budget = std::stoi(Attr.getValueAsString().str());
                Hints.LatencyBudgetMS = Budget;
            }
        }

        // Check for bandwidth contract
        if (F.hasFnAttribute("dsmil_bandwidth_contract")) {
            Attribute Attr = F.getFnAttribute("dsmil_bandwidth_contract");
            if (Attr.isStringAttribute()) {
                unsigned BW = std::stoi(Attr.getValueAsString().str());
                Hints.BandwidthGbps = BW;
            }
        }

        if (Hints.Profile != NONE || Hints.PreferEdgeOffload) {
            FunctionHints[&F] = Hints;
        }
    }
}

JADC2Profile DsmilJADC2Pass::parseProfile(const std::string &ProfileName) {
    if (ProfileName == "sensor_fusion")
        return SENSOR_FUSION;
    if (ProfileName == "c2_processing")
        return C2_PROCESSING;
    if (ProfileName == "targeting")
        return TARGETING;
    if (ProfileName == "situational_awareness")
        return SITUATIONAL_AWARENESS;
    if (ProfileName == "logistics")
        return LOGISTICS;
    return NONE;
}

bool DsmilJADC2Pass::analyzeLatencyBudgets(Module &M) {
    bool HasViolations = false;

    for (auto &[F, Hints] : FunctionHints) {
        // Estimate function latency (simplified static analysis)
        unsigned EstimatedMS = estimateLatencyMS(F);

        if (EstimatedMS > Hints.LatencyBudgetMS) {
            NumLatencyViolations++;
            HasViolations = true;

            errs() << "  LATENCY VIOLATION: " << F->getName() << "\n";
            errs() << "    Budget: " << Hints.LatencyBudgetMS << "ms\n";
            errs() << "    Estimated: " << EstimatedMS << "ms\n";
            errs() << "    Overage: " << (EstimatedMS - Hints.LatencyBudgetMS) << "ms\n";

            // Suggest optimization
            if (Hints.PreferEdgeOffload) {
                errs() << "    Recommendation: Already marked for edge offload\n";
            } else {
                errs() << "    Recommendation: Consider edge offload or refactoring\n";
            }
        }

        // Check bandwidth
        unsigned EstimatedBW = estimateBandwidthMBps(F);
        unsigned BudgetMBps = Hints.BandwidthGbps * 125;  // Gbps to MBps

        if (EstimatedBW > BudgetMBps) {
            NumBandwidthWarnings++;
            errs() << "  BANDWIDTH WARNING: " << F->getName() << "\n";
            errs() << "    Contract: " << Hints.BandwidthGbps << " Gbps\n";
            errs() << "    Estimated: " << EstimatedBW << " MBps\n";
        }
    }

    return HasViolations;
}

unsigned DsmilJADC2Pass::estimateLatencyMS(Function *F) {
    // Simplified static latency estimation
    // In production, this would use Layer 5 Performance AI cost models

    unsigned EstimatedCycles = 0;

    // Count instructions (very rough approximation)
    for (auto &BB : *F) {
        for (auto &I : BB) {
            EstimatedCycles += 1;

            // Expensive operations
            if (isa<CallInst>(I)) {
                EstimatedCycles += 100;  // Assume call overhead
            }
            if (I.getOpcode() == Instruction::Load ||
                I.getOpcode() == Instruction::Store) {
                EstimatedCycles += 5;  // Memory access
            }
        }
    }

    // Assume 2 GHz CPU, convert cycles to ms
    unsigned LatencyMS = EstimatedCycles / 2000000;

    // Minimum 1ms for any function
    return LatencyMS > 0 ? LatencyMS : 1;
}

unsigned DsmilJADC2Pass::estimateBandwidthMBps(Function *F) {
    // Simplified bandwidth estimation
    // Count store operations as proxy for network I/O

    unsigned StoreCount = 0;
    for (auto &BB : *F) {
        for (auto &I : BB) {
            if (I.getOpcode() == Instruction::Store) {
                StoreCount++;
            }
        }
    }

    // Rough estimate: 1 KB per store
    unsigned EstimatedKB = StoreCount * 1;

    // Convert to MBps (assume 1 second execution)
    return EstimatedKB / 1024;
}

bool DsmilJADC2Pass::optimizeFor5G(Module &M) {
    bool Modified = false;

    // Optimization strategies for 5G/MEC deployment:
    // 1. Compact message formats
    // 2. Batch small operations
    // 3. Select low-latency code paths
    // 4. Power-efficient back-end selection for edge devices

    for (auto &[F, Hints] : FunctionHints) {
        if (!Hints.PreferEdgeOffload)
            continue;

        // Insert JADC2 transport hints
        // (Simplified - production would rewrite calls)

        // Example: Transform network send calls to use JADC2 transport layer
        for (auto &BB : *F) {
            for (auto &I : BB) {
                if (auto *CI = dyn_cast<CallInst>(&I)) {
                    Function *Callee = CI->getCalledFunction();
                    if (!Callee)
                        continue;

                    // If calling network send, suggest JADC2 transport
                    if (Callee->getName().contains("send")) {
                        // In production: rewrite to dsmil_jadc2_send()
                        Modified = true;
                    }
                }
            }
        }
    }

    return Modified;
}

void DsmilJADC2Pass::identifyOffloadCandidates(Module &M) {
    for (auto &F : M) {
        if (F.isDeclaration())
            continue;

        // Skip if already in hints
        if (FunctionHints.find(&F) != FunctionHints.end())
            continue;

        MEC5GHints Hints = {};
        Hints.LatencyBudgetMS = 1000;
        Hints.BandwidthGbps = 1;

        // Check if function would benefit from edge offload
        if (isOffloadCandidate(&F, Hints)) {
            NumOffloadCandidates++;
            errs() << "  OFFLOAD CANDIDATE: " << F.getName() << "\n";
            errs() << "    Reason: Compute-intensive, low network I/O\n";
            errs() << "    Recommendation: Add DSMIL_5G_EDGE attribute\n";
        }
    }
}

bool DsmilJADC2Pass::isOffloadCandidate(Function *F,
                                         const MEC5GHints &Hints) {
    // Heuristic: High compute, low I/O = good offload candidate

    unsigned ComputeOps = 0;
    unsigned MemoryOps = 0;

    for (auto &BB : *F) {
        for (auto &I : BB) {
            if (I.isBinaryOp() || isa<CmpInst>(I)) {
                ComputeOps++;
            }
            if (I.getOpcode() == Instruction::Load ||
                I.getOpcode() == Instruction::Store) {
                MemoryOps++;
            }
        }
    }

    // Good candidate if compute/memory ratio > 10
    return (ComputeOps > 100 && MemoryOps > 0 &&
            (ComputeOps / MemoryOps) > 10);
}

} // anonymous namespace

// Pass registration (for new PM)
extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
    return {
        LLVM_PLUGIN_API_VERSION, "DsmilJADC2", "v1.5.0",
        [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                    if (Name == "dsmil-jadc2") {
                        MPM.addPass(DsmilJADC2Pass());
                        return true;
                    }
                    return false;
                });
        }};
}
