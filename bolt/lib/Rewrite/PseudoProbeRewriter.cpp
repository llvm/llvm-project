//===- bolt/Rewrite/PseudoProbeRewriter.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implement support for pseudo probes.
//
//===----------------------------------------------------------------------===//

#include "bolt/Core/BinaryFunction.h"
#include "bolt/Rewrite/MetadataRewriter.h"
#include "bolt/Rewrite/MetadataRewriters.h"
#include "bolt/Utils/CommandLineOpts.h"
#include "llvm/IR/Function.h"
#include "llvm/MC/MCPseudoProbe.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LEB128.h"

#undef DEBUG_TYPE
#define DEBUG_TYPE "pseudo-probe-rewriter"

using namespace llvm;
using namespace bolt;

namespace opts {

enum PrintPseudoProbesOptions {
  PPP_None = 0,
  PPP_Probes_Section_Decode = 0x1,
  PPP_Probes_Address_Conversion = 0x2,
  PPP_Encoded_Probes = 0x3,
  PPP_All = 0xf
};

static cl::opt<PrintPseudoProbesOptions> PrintPseudoProbes(
    "print-pseudo-probes", cl::desc("print pseudo probe info"),
    cl::init(PPP_None),
    cl::values(clEnumValN(PPP_Probes_Section_Decode, "decode",
                          "decode probes section from binary"),
               clEnumValN(PPP_Probes_Address_Conversion, "address_conversion",
                          "update address2ProbesMap with output block address"),
               clEnumValN(PPP_Encoded_Probes, "encoded_probes",
                          "display the encoded probes in binary section"),
               clEnumValN(PPP_All, "all", "enable all debugging printout")),
    cl::Hidden, cl::cat(BoltCategory));

} // namespace opts

namespace {
class PseudoProbeRewriter final : public MetadataRewriter {
  /// .pseudo_probe_desc section.
  /// Contains information about pseudo probe description, like its related
  /// function
  ErrorOr<BinarySection &> PseudoProbeDescSection{std::errc::bad_address};

  /// .pseudo_probe section.
  /// Contains information about pseudo probe details, like its address
  ErrorOr<BinarySection &> PseudoProbeSection{std::errc::bad_address};

  /// Update address of MCDecodedPseudoProbe.
  void updatePseudoProbes();

  /// Encode MCDecodedPseudoProbe.
  void encodePseudoProbes();

  /// Parse .pseudo_probe_desc section and .pseudo_probe section
  /// Setup Pseudo probe decoder
  void parsePseudoProbe();

  /// PseudoProbe decoder
  MCPseudoProbeDecoder ProbeDecoder;

public:
  PseudoProbeRewriter(BinaryContext &BC)
      : MetadataRewriter("pseudo-probe-rewriter", BC) {}

  Error postEmitFinalizer() override;
};

Error PseudoProbeRewriter::postEmitFinalizer() {
  parsePseudoProbe();
  updatePseudoProbes();

  return Error::success();
}

void PseudoProbeRewriter::parsePseudoProbe() {
  PseudoProbeDescSection = BC.getUniqueSectionByName(".pseudo_probe_desc");
  PseudoProbeSection = BC.getUniqueSectionByName(".pseudo_probe");

  if (!PseudoProbeDescSection && !PseudoProbeSection) {
    // pesudo probe is not added to binary. It is normal and no warning needed.
    return;
  }

  // If only one section is found, it might mean the ELF is corrupted.
  if (!PseudoProbeDescSection) {
    errs() << "BOLT-WARNING: fail in reading .pseudo_probe_desc binary\n";
    return;
  } else if (!PseudoProbeSection) {
    errs() << "BOLT-WARNING: fail in reading .pseudo_probe binary\n";
    return;
  }

  StringRef Contents = PseudoProbeDescSection->getContents();
  if (!ProbeDecoder.buildGUID2FuncDescMap(
          reinterpret_cast<const uint8_t *>(Contents.data()),
          Contents.size())) {
    errs() << "BOLT-WARNING: fail in building GUID2FuncDescMap\n";
    return;
  }

  MCPseudoProbeDecoder::Uint64Set GuidFilter;
  MCPseudoProbeDecoder::Uint64Map FuncStartAddrs;
  for (const BinaryFunction *F : BC.getAllBinaryFunctions()) {
    for (const MCSymbol *Sym : F->getSymbols()) {
      FuncStartAddrs[Function::getGUID(NameResolver::restore(Sym->getName()))] =
          F->getAddress();
    }
  }
  Contents = PseudoProbeSection->getContents();
  if (!ProbeDecoder.buildAddress2ProbeMap(
          reinterpret_cast<const uint8_t *>(Contents.data()), Contents.size(),
          GuidFilter, FuncStartAddrs)) {
    ProbeDecoder.getAddress2ProbesMap().clear();
    errs() << "BOLT-WARNING: fail in building Address2ProbeMap\n";
    return;
  }

  if (opts::PrintPseudoProbes == opts::PrintPseudoProbesOptions::PPP_All ||
      opts::PrintPseudoProbes ==
          opts::PrintPseudoProbesOptions::PPP_Probes_Section_Decode) {
    outs() << "Report of decoding input pseudo probe binaries \n";
    ProbeDecoder.printGUID2FuncDescMap(outs());
    ProbeDecoder.printProbesForAllAddresses(outs());
  }
}

void PseudoProbeRewriter::updatePseudoProbes() {
  // check if there is pseudo probe section decoded
  if (ProbeDecoder.getAddress2ProbesMap().empty())
    return;
  // input address converted to output
  AddressProbesMap &Address2ProbesMap = ProbeDecoder.getAddress2ProbesMap();
  const GUIDProbeFunctionMap &GUID2Func = ProbeDecoder.getGUID2FuncDescMap();

  for (auto &AP : Address2ProbesMap) {
    BinaryFunction *F = BC.getBinaryFunctionContainingAddress(AP.first);
    // If F is removed, eliminate all probes inside it from inline tree
    // Setting probes' addresses as INT64_MAX means elimination
    if (!F) {
      for (MCDecodedPseudoProbe &Probe : AP.second)
        Probe.setAddress(INT64_MAX);
      continue;
    }
    // If F is not emitted, the function will remain in the same address as its
    // input
    if (!F->isEmitted())
      continue;

    uint64_t Offset = AP.first - F->getAddress();
    const BinaryBasicBlock *BB = F->getBasicBlockContainingOffset(Offset);
    uint64_t BlkOutputAddress = BB->getOutputAddressRange().first;
    // Check if block output address is defined.
    // If not, such block is removed from binary. Then remove the probes from
    // inline tree
    if (BlkOutputAddress == 0) {
      for (MCDecodedPseudoProbe &Probe : AP.second)
        Probe.setAddress(INT64_MAX);
      continue;
    }

    unsigned ProbeTrack = AP.second.size();
    std::list<MCDecodedPseudoProbe>::iterator Probe = AP.second.begin();
    while (ProbeTrack != 0) {
      if (Probe->isBlock()) {
        Probe->setAddress(BlkOutputAddress);
      } else if (Probe->isCall()) {
        // A call probe may be duplicated due to ICP
        // Go through output of InputOffsetToAddressMap to collect all related
        // probes
        const InputOffsetToAddressMapTy &Offset2Addr =
            F->getInputOffsetToAddressMap();
        auto CallOutputAddresses = Offset2Addr.equal_range(Offset);
        auto CallOutputAddress = CallOutputAddresses.first;
        if (CallOutputAddress == CallOutputAddresses.second) {
          Probe->setAddress(INT64_MAX);
        } else {
          Probe->setAddress(CallOutputAddress->second);
          CallOutputAddress = std::next(CallOutputAddress);
        }

        while (CallOutputAddress != CallOutputAddresses.second) {
          AP.second.push_back(*Probe);
          AP.second.back().setAddress(CallOutputAddress->second);
          Probe->getInlineTreeNode()->addProbes(&(AP.second.back()));
          CallOutputAddress = std::next(CallOutputAddress);
        }
      }
      Probe = std::next(Probe);
      ProbeTrack--;
    }
  }

  if (opts::PrintPseudoProbes == opts::PrintPseudoProbesOptions::PPP_All ||
      opts::PrintPseudoProbes ==
          opts::PrintPseudoProbesOptions::PPP_Probes_Address_Conversion) {
    outs() << "Pseudo Probe Address Conversion results:\n";
    // table that correlates address to block
    std::unordered_map<uint64_t, StringRef> Addr2BlockNames;
    for (auto &F : BC.getBinaryFunctions())
      for (BinaryBasicBlock &BinaryBlock : F.second)
        Addr2BlockNames[BinaryBlock.getOutputAddressRange().first] =
            BinaryBlock.getName();

    // scan all addresses -> correlate probe to block when print out
    std::vector<uint64_t> Addresses;
    for (auto &Entry : Address2ProbesMap)
      Addresses.push_back(Entry.first);
    llvm::sort(Addresses);
    for (uint64_t Key : Addresses) {
      for (MCDecodedPseudoProbe &Probe : Address2ProbesMap[Key]) {
        if (Probe.getAddress() == INT64_MAX)
          outs() << "Deleted Probe: ";
        else
          outs() << "Address: " << format_hex(Probe.getAddress(), 8) << " ";
        Probe.print(outs(), GUID2Func, true);
        // print block name only if the probe is block type and undeleted.
        if (Probe.isBlock() && Probe.getAddress() != INT64_MAX)
          outs() << format_hex(Probe.getAddress(), 8) << " Probe is in "
                 << Addr2BlockNames[Probe.getAddress()] << "\n";
      }
    }
    outs() << "=======================================\n";
  }

  // encode pseudo probes with updated addresses
  encodePseudoProbes();
}

void PseudoProbeRewriter::encodePseudoProbes() {
  // Buffer for new pseudo probes section
  SmallString<8> Contents;
  MCDecodedPseudoProbe *LastProbe = nullptr;

  auto EmitInt = [&](uint64_t Value, uint32_t Size) {
    const bool IsLittleEndian = BC.AsmInfo->isLittleEndian();
    uint64_t Swapped = support::endian::byte_swap(
        Value, IsLittleEndian ? support::little : support::big);
    unsigned Index = IsLittleEndian ? 0 : 8 - Size;
    auto Entry = StringRef(reinterpret_cast<char *>(&Swapped) + Index, Size);
    Contents.append(Entry.begin(), Entry.end());
  };

  auto EmitULEB128IntValue = [&](uint64_t Value) {
    SmallString<128> Tmp;
    raw_svector_ostream OSE(Tmp);
    encodeULEB128(Value, OSE, 0);
    Contents.append(OSE.str().begin(), OSE.str().end());
  };

  auto EmitSLEB128IntValue = [&](int64_t Value) {
    SmallString<128> Tmp;
    raw_svector_ostream OSE(Tmp);
    encodeSLEB128(Value, OSE);
    Contents.append(OSE.str().begin(), OSE.str().end());
  };

  // Emit indiviual pseudo probes in a inline tree node
  // Probe index, type, attribute, address type and address are encoded
  // Address of the first probe is absolute.
  // Other probes' address are represented by delta
  auto EmitDecodedPseudoProbe = [&](MCDecodedPseudoProbe *&CurProbe) {
    assert(!isSentinelProbe(CurProbe->getAttributes()) &&
           "Sentinel probes should not be emitted");
    EmitULEB128IntValue(CurProbe->getIndex());
    uint8_t PackedType = CurProbe->getType() | (CurProbe->getAttributes() << 4);
    uint8_t Flag =
        LastProbe ? ((int8_t)MCPseudoProbeFlag::AddressDelta << 7) : 0;
    EmitInt(Flag | PackedType, 1);
    if (LastProbe) {
      // Emit the delta between the address label and LastProbe.
      int64_t Delta = CurProbe->getAddress() - LastProbe->getAddress();
      EmitSLEB128IntValue(Delta);
    } else {
      // Emit absolute address for encoding the first pseudo probe.
      uint32_t AddrSize = BC.AsmInfo->getCodePointerSize();
      EmitInt(CurProbe->getAddress(), AddrSize);
    }
  };

  std::map<InlineSite, MCDecodedPseudoProbeInlineTree *,
           std::greater<InlineSite>>
      Inlinees;

  // DFS of inline tree to emit pseudo probes in all tree node
  // Inline site index of a probe is emitted first.
  // Then tree node Guid, size of pseudo probes and children nodes, and detail
  // of contained probes are emitted Deleted probes are skipped Root node is not
  // encoded to binaries. It's a "wrapper" of inline trees of each function.
  std::list<std::pair<uint64_t, MCDecodedPseudoProbeInlineTree *>> NextNodes;
  const MCDecodedPseudoProbeInlineTree &Root =
      ProbeDecoder.getDummyInlineRoot();
  for (auto Child = Root.getChildren().begin();
       Child != Root.getChildren().end(); ++Child)
    Inlinees[Child->first] = Child->second.get();

  for (auto Inlinee : Inlinees)
    // INT64_MAX is "placeholder" of unused callsite index field in the pair
    NextNodes.push_back({INT64_MAX, Inlinee.second});

  Inlinees.clear();

  while (!NextNodes.empty()) {
    uint64_t ProbeIndex = NextNodes.back().first;
    MCDecodedPseudoProbeInlineTree *Cur = NextNodes.back().second;
    NextNodes.pop_back();

    if (Cur->Parent && !Cur->Parent->isRoot())
      // Emit probe inline site
      EmitULEB128IntValue(ProbeIndex);

    // Emit probes grouped by GUID.
    LLVM_DEBUG({
      dbgs().indent(MCPseudoProbeTable::DdgPrintIndent);
      dbgs() << "GUID: " << Cur->Guid << "\n";
    });
    // Emit Guid
    EmitInt(Cur->Guid, 8);
    // Emit number of probes in this node
    uint64_t Deleted = 0;
    for (MCDecodedPseudoProbe *&Probe : Cur->getProbes())
      if (Probe->getAddress() == INT64_MAX)
        Deleted++;
    LLVM_DEBUG(dbgs() << "Deleted Probes:" << Deleted << "\n");
    uint64_t ProbesSize = Cur->getProbes().size() - Deleted;
    EmitULEB128IntValue(ProbesSize);
    // Emit number of direct inlinees
    EmitULEB128IntValue(Cur->getChildren().size());
    // Emit probes in this group
    for (MCDecodedPseudoProbe *&Probe : Cur->getProbes()) {
      if (Probe->getAddress() == INT64_MAX)
        continue;
      EmitDecodedPseudoProbe(Probe);
      LastProbe = Probe;
    }

    for (auto Child = Cur->getChildren().begin();
         Child != Cur->getChildren().end(); ++Child)
      Inlinees[Child->first] = Child->second.get();
    for (const auto &Inlinee : Inlinees) {
      assert(Cur->Guid != 0 && "non root tree node must have nonzero Guid");
      NextNodes.push_back({std::get<1>(Inlinee.first), Inlinee.second});
      LLVM_DEBUG({
        dbgs().indent(MCPseudoProbeTable::DdgPrintIndent);
        dbgs() << "InlineSite: " << std::get<1>(Inlinee.first) << "\n";
      });
    }
    Inlinees.clear();
  }

  // Create buffer for new contents for the section
  // Freed when parent section is destroyed
  uint8_t *Output = new uint8_t[Contents.str().size()];
  memcpy(Output, Contents.str().data(), Contents.str().size());
  BC.registerOrUpdateSection(".pseudo_probe", PseudoProbeSection->getELFType(),
                             PseudoProbeSection->getELFFlags(), Output,
                             Contents.str().size(), 1);
  if (opts::PrintPseudoProbes == opts::PrintPseudoProbesOptions::PPP_All ||
      opts::PrintPseudoProbes ==
          opts::PrintPseudoProbesOptions::PPP_Encoded_Probes) {
    // create a dummy decoder;
    MCPseudoProbeDecoder DummyDecoder;
    StringRef DescContents = PseudoProbeDescSection->getContents();
    DummyDecoder.buildGUID2FuncDescMap(
        reinterpret_cast<const uint8_t *>(DescContents.data()),
        DescContents.size());
    StringRef ProbeContents = PseudoProbeSection->getOutputContents();
    MCPseudoProbeDecoder::Uint64Set GuidFilter;
    MCPseudoProbeDecoder::Uint64Map FuncStartAddrs;
    for (const BinaryFunction *F : BC.getAllBinaryFunctions()) {
      const uint64_t Addr =
          F->isEmitted() ? F->getOutputAddress() : F->getAddress();
      FuncStartAddrs[Function::getGUID(
          NameResolver::restore(F->getOneName()))] = Addr;
    }
    DummyDecoder.buildAddress2ProbeMap(
        reinterpret_cast<const uint8_t *>(ProbeContents.data()),
        ProbeContents.size(), GuidFilter, FuncStartAddrs);
    DummyDecoder.printProbesForAllAddresses(outs());
  }
}
} // namespace

std::unique_ptr<MetadataRewriter>
llvm::bolt::createPseudoProbeRewriter(BinaryContext &BC) {
  return std::make_unique<PseudoProbeRewriter>(BC);
}
