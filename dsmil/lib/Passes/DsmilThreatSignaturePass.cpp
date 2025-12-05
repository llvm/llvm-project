/**
 * @file DsmilThreatSignaturePass.cpp
 * @brief DSLLVM Threat Signature Embedding Pass (v1.4 - Feature 2.2)
 *
 * Embeds non-identifying threat signatures in binaries for future forensics.
 * Layer 62 (Forensics/SIEM) uses signatures to correlate observed malware
 * with known-good templates.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SHA256.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/CFG.h"
#include "dsmil_threat_signature.h"
#include <map>
#include <set>
#include <string>
#include <vector>

#define DEBUG_TYPE "dsmil-threat-signature"

using namespace llvm;

// Command-line options
static cl::opt<bool> EnableThreatSig(
    "dsmil-threat-signature",
    cl::desc("Enable threat signature embedding"),
    cl::init(false));

static cl::opt<std::string> ThreatSigOutput(
    "dsmil-threat-signature-output",
    cl::desc("Output path for threat signature JSON"),
    cl::init("threat-signature.json"));

namespace {

/**
 * Threat Signature Embedding Pass
 */
class DsmilThreatSignaturePass : public PassInfoMixin<DsmilThreatSignaturePass> {
private:
  bool Enabled;
  std::string OutputPath;

  // Collected data
  std::vector<std::string> FunctionNames;
  std::set<std::string> CryptoAlgorithms;
  std::set<std::string> ProtocolSchemas;
  std::vector<uint8_t> CFGHash;

  /**
   * Compute CFG hash for module
   */
  void computeCFGHash(Module &M) {
    // Simplified CFG hashing: concatenate function names and basic block counts
    std::string CFGData;

    for (auto &F : M) {
      if (F.isDeclaration())
        continue;

      CFGData += F.getName().str();
      CFGData += std::to_string(F.size());  // Number of basic blocks

      // Add simplified CFG structure
      for (auto &BB : F) {
        CFGData += std::to_string(BB.size());  // Instructions per block
      }
    }

    // Compute SHA-256 hash
    auto Hash = SHA256::hash(arrayRefFromStringRef(StringRef(CFGData)));
    CFGHash.assign(Hash.begin(), Hash.end());
  }

  /**
   * Extract crypto patterns from function
   */
  void extractCryptoPatterns(Function &F) {
    // Check for crypto-related attributes
    if (F.hasFnAttribute("dsmil_secret")) {
      // This function uses constant-time crypto
      CryptoAlgorithms.insert("constant_time_enforced");
    }

    // Look for known crypto function names
    StringRef Name = F.getName();
    if (Name.contains("aes")) CryptoAlgorithms.insert("AES");
    if (Name.contains("kem") || Name.contains("kyber")) CryptoAlgorithms.insert("ML-KEM");
    if (Name.contains("dsa") || Name.contains("dilithium")) CryptoAlgorithms.insert("ML-DSA");
    if (Name.contains("sha") || Name.contains("hash")) CryptoAlgorithms.insert("SHA");
    if (Name.contains("gcm")) CryptoAlgorithms.insert("GCM");
  }

  /**
   * Extract protocol schemas from function
   */
  void extractProtocolSchemas(Function &F) {
    StringRef Name = F.getName();

    // Detect protocol usage from function names
    if (Name.contains("tls")) ProtocolSchemas.insert("TLS");
    if (Name.contains("http")) ProtocolSchemas.insert("HTTP");
    if (Name.contains("quic")) ProtocolSchemas.insert("QUIC");
  }

  /**
   * Generate threat signature JSON
   */
  void generateSignatureJSON(Module &M) {
    using namespace llvm::json;

    Object Signature;
    Signature["version"] = DSMIL_THREAT_SIGNATURE_VERSION;
    Signature["schema"] = "dsmil-threat-signature-v1";
    Signature["module"] = M.getName().str();

    // CFG fingerprint
    Object CFG;
    CFG["algorithm"] = "CFG-SHA256";

    // Convert hash to hex string
    std::string HashHex;
    for (uint8_t Byte : CFGHash) {
      char Buf[3];
      snprintf(Buf, sizeof(Buf), "%02x", Byte);
      HashHex += Buf;
    }
    CFG["hash"] = HashHex;
    CFG["num_functions"] = (int64_t)FunctionNames.size();

    Array FuncArray;
    for (const auto &FName : FunctionNames) {
      FuncArray.push_back(FName);
    }
    CFG["functions_included"] = std::move(FuncArray);

    Signature["control_flow_fingerprint"] = std::move(CFG);

    // Crypto patterns
    Array CryptoArray;
    for (const auto &Algo : CryptoAlgorithms) {
      Object CryptoObj;
      CryptoObj["algorithm"] = Algo;
      CryptoArray.push_back(std::move(CryptoObj));
    }
    Signature["crypto_patterns"] = std::move(CryptoArray);

    // Protocol schemas
    Array ProtocolArray;
    for (const auto &Proto : ProtocolSchemas) {
      Object ProtoObj;
      ProtoObj["protocol"] = Proto;
      ProtocolArray.push_back(std::move(ProtoObj));
    }
    Signature["protocol_schemas"] = std::move(ProtocolArray);

    // Write to file
    std::error_code EC;
    raw_fd_ostream OS(OutputPath, EC);
    if (!EC) {
      OS << formatv("{0:2}", json::Value(std::move(Signature)));
      OS.close();
      errs() << "[DSMIL Threat Signature] Generated: " << OutputPath << "\n";
    }
  }

public:
  DsmilThreatSignaturePass()
    : Enabled(EnableThreatSig.getValue()),
      OutputPath(ThreatSigOutput.getValue()) {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM) {
    if (!Enabled)
      return PreservedAnalyses::all();

    LLVM_DEBUG(dbgs() << "[DSMIL Threat Signature] Processing module: "
                      << M.getName() << "\n");

    // Collect function names and patterns
    for (auto &F : M) {
      if (F.isDeclaration())
        continue;

      FunctionNames.push_back(F.getName().str());
      extractCryptoPatterns(F);
      extractProtocolSchemas(F);
    }

    // Compute CFG hash
    computeCFGHash(M);

    // Generate signature JSON
    generateSignatureJSON(M);

    errs() << "[DSMIL Threat Signature] Summary:\n";
    errs() << "  Functions: " << FunctionNames.size() << "\n";
    errs() << "  Crypto patterns: " << CryptoAlgorithms.size() << "\n";
    errs() << "  Protocol schemas: " << ProtocolSchemas.size() << "\n";

    // No IR modifications
    return PreservedAnalyses::all();
  }

  static bool isRequired() { return true; }
};

} // end anonymous namespace

// Register the pass
extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {
    LLVM_PLUGIN_API_VERSION, "DsmilThreatSignaturePass", LLVM_VERSION_STRING,
    [](PassBuilder &PB) {
      PB.registerPipelineParsingCallback(
        [](StringRef Name, ModulePassManager &MPM,
           ArrayRef<PassBuilder::PipelineElement>) {
          if (Name == "dsmil-threat-signature") {
            MPM.addPass(DsmilThreatSignaturePass());
            return true;
          }
          return false;
        });
    }
  };
}
