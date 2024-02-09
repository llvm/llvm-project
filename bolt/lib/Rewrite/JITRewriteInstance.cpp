//===- bolt/Rewrite/JITRewriteInstance.cpp - JIT rewriter -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Rewrite/JITRewriteInstance.h"
#include "bolt/Core/BinaryContext.h"
#include "bolt/Core/BinaryEmitter.h"
#include "bolt/Core/BinaryFunction.h"
#include "bolt/Core/JumpTable.h"
#include "bolt/Core/MCPlusBuilder.h"
#include "bolt/Profile/DataAggregator.h"
#include "bolt/Rewrite/BinaryPassManager.h"
#include "bolt/Rewrite/RewriteInstance.h"
#include "bolt/Utils/Utils.h"
#include "llvm/MC/MCAsmLayout.h"
#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/Object/SymbolSize.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include <memory>

namespace opts {

using namespace llvm;
extern cl::opt<unsigned> AlignText;
extern cl::opt<bool> PrintSections;
extern cl::opt<bool> PrintDisasm;
extern cl::opt<bool> PrintCFG;
extern cl::opt<unsigned> Verbosity;
} // namespace opts

namespace llvm {
namespace bolt {

#define DEBUG_TYPE "bolt"

Expected<std::unique_ptr<JITRewriteInstance>>
JITRewriteInstance::createJITRewriteInstance(JournalingStreams Logger,
                                             bool IsPIC) {
  Error Err = Error::success();
  std::unique_ptr<JITRewriteInstance> JITRI(
      new JITRewriteInstance(Logger, IsPIC, Err));
  if (Err)
    return std::move(Err);
  return std::move(JITRI);
}

JITRewriteInstance::JITRewriteInstance(JournalingStreams Logger, bool IsPIC,
                                       Error &Err)
    : StrPool(StrAllocator) {
  ErrorAsOutParameter EAO(&Err);
  Triple TheTriple(sys::getDefaultTargetTriple().c_str());

  auto BCOrErr = BinaryContext::createBinaryContext(
      TheTriple, StringRef("JIT input file"), nullptr, IsPIC, nullptr, Logger);
  if (Error E = BCOrErr.takeError()) {
    Err = std::move(E);
    return;
  }
  BC = std::move(BCOrErr.get());
  BC->initializeTarget(std::unique_ptr<MCPlusBuilder>(
      createMCPlusBuilder(BC->TheTriple->getArch(), BC->MIA.get(),
                          BC->MII.get(), BC->MRI.get(), BC->STI.get())));
  BC->FirstAllocAddress = 0;
  BC->LayoutStartAddress = 0xffffffffffffffff;
}

JITRewriteInstance::~JITRewriteInstance() {}

void JITRewriteInstance::adjustCommandLineOptions() {
  if (!opts::AlignText.getNumOccurrences())
    opts::AlignText = BC->PageAlign;
}

Error JITRewriteInstance::preprocessProfileData() {
  if (!ProfileReader)
    return Error::success();
  if (Error E = ProfileReader->preprocessProfile(*BC.get()))
    return Error(std::move(E));
  return Error::success();
}

Error JITRewriteInstance::processProfileDataPreCFG() {
  if (!ProfileReader)
    return Error::success();
  if (Error E = ProfileReader->readProfilePreCFG(*BC.get()))
    return Error(std::move(E));
  return Error::success();
}

Error JITRewriteInstance::processProfileData() {
  if (!ProfileReader)
    return Error::success();
  if (Error E = ProfileReader->readProfile(*BC.get()))
    return Error(std::move(E));
  return Error::success();
}

Error JITRewriteInstance::disassembleFunctions() {
  for (auto &BFI : BC->getBinaryFunctions()) {
    BinaryFunction &Function = BFI.second;
    if (!Function.isSimple())
      continue;
    if (Error E = Function.disassemble())
      return Error(std::move(E));
    if (opts::PrintDisasm)
      Function.print(BC->outs(), "after disassembly");
  }
  return Error::success();
}

Error JITRewriteInstance::buildFunctionsCFG() {
  for (auto &BFI : BC->getBinaryFunctions()) {
    BinaryFunction &Function = BFI.second;
    if (!Function.isSimple())
      continue;
    if (auto NewE = handleErrors(
            Function.buildCFG(/*AllocId*/ 0), [&](const BOLTError &E) -> Error {
              if (E.isFatal())
                return Error(std::make_unique<BOLTError>(std::move(E)));
              if (!E.getMessage().empty())
                E.log(BC->errs());
              return Error::success();
            })) {
      return Error(std::move(NewE));
    }
  }
  return Error::success();
}

void JITRewriteInstance::postProcessFunctions() {
  for (auto &BFI : BC->getBinaryFunctions()) {
    BinaryFunction &Function = BFI.second;
    if (Function.empty() || !Function.isSimple())
      continue;
    Function.postProcessCFG();
    if (opts::PrintCFG)
      Function.print(outs(), "after building cfg");
  }
}

void JITRewriteInstance::registerJITSection(StringRef Name, uint64_t Address,
                                            StringRef Data, unsigned Alignment,
                                            unsigned ELFType,
                                            unsigned ELFFlags) {
  auto *Sec =
      new BinarySection(*BC, Name, const_cast<uint8_t *>(Data.bytes_begin()),
                        Data.size(), Alignment, ELFType, ELFFlags);
  Sec->setAddress(Address);
  BC->registerSection(Sec);
}

void JITRewriteInstance::registerJITFunction(StringRef Name, uintptr_t Addr,
                                             size_t Size) {
  if (ErrorOr<BinarySection &> Sec = BC->getSectionForAddress(Addr))
    BC->createBinaryFunction(Name.str(), *Sec, Addr, Size);
}

Error JITRewriteInstance::notifyObjectLoaded(const object::ObjectFile &Obj) {
  for (const object::SectionRef &Section : Obj.sections()) {
    Expected<StringRef> SectionName = Section.getName();
    if (Error E = SectionName.takeError())
      return Error(std::move(E));
    // Only register sections with names.
    if (SectionName->empty())
      continue;

    StringRef UniqueSectionName = StrPool.save(NR.uniquify(*SectionName));
    unsigned ELFType = ELFSectionRef(Section).getType();
    unsigned ELFFlags = ELFSectionRef(Section).getFlags();
    if (ELFType == ELF::SHT_NOBITS)
      continue;

    const uint64_t Address = Section.getAddress();
    const uint64_t Size = Section.getSize();
    StringRef Contents =
        StringRef(reinterpret_cast<const char *>(Address), Size);
    if (Contents.empty())
      continue;

    this->registerJITSection(UniqueSectionName, Section.getAddress(), Contents,
                             Section.getAlignment().value(), ELFType, ELFFlags);
    LLVM_DEBUG(
        dbgs() << "BOLT-DEBUG: registering section " << *SectionName << " @ 0x"
               << Twine::utohexstr(Section.getAddress()) << ":0x"
               << Twine::utohexstr(Section.getAddress() + Section.getSize())
               << "\n");
  }

  if (opts::PrintSections) {
    BC->outs() << "BOLT-INFO: Sections from original binary:\n";
    BC->printSections(BC->outs());
  }

  std::vector<SymbolRef> FunctionSymbols;
  for (const SymbolRef &S : Obj.symbols()) {
    auto TypeOrErr = S.getType();
    if (Error E = TypeOrErr.takeError())
      return Error(std::move(E));
    SymbolRef::Type Type = *TypeOrErr;
    if (Type == SymbolRef::ST_Function)
      FunctionSymbols.push_back(S);
  }

  if (FunctionSymbols.empty())
    return Error::success();

  Error SortErrors = Error::success();
  llvm::stable_sort(FunctionSymbols, [&](const SymbolRef &LHS,
                                         const SymbolRef &RHS) {
    auto LHSAddrOrErr = LHS.getAddress();
    auto RHSAddrOrErr = RHS.getAddress();
    if (auto E =
            joinErrors(LHSAddrOrErr.takeError(), RHSAddrOrErr.takeError())) {
      SortErrors = joinErrors(std::move(SortErrors), std::move(E));
      return false;
    }
    return *LHSAddrOrErr < *RHSAddrOrErr;
  });
  if (SortErrors)
    return Error(std::move(SortErrors));

  for (size_t Index = 0; Index < FunctionSymbols.size(); ++Index) {
    auto AddrOrErr = FunctionSymbols[Index].getAddress();
    if (auto E = AddrOrErr.takeError())
      return Error(std::move(E));

    const uint64_t Address = *AddrOrErr;
    ErrorOr<BinarySection &> Section = BC->getSectionForAddress(Address);
    if (!Section)
      continue;

    auto NameOrErr = FunctionSymbols[Index].getName();
    auto FlagsOrErr = FunctionSymbols[Index].getFlags();
    auto SecOrErr = FunctionSymbols[Index].getSection();
    if (auto E = joinErrors(
            joinErrors(NameOrErr.takeError(), FlagsOrErr.takeError()),
            SecOrErr.takeError()))
      return Error(std::move(E));
    std::string SymbolName = NameOrErr->str();
    // Uniquify names of local symbols.
    if (!(*FlagsOrErr & SymbolRef::SF_Global))
      SymbolName = NR.uniquify(SymbolName);

    section_iterator S = *SecOrErr;
    uint64_t EndAddress = S->getAddress() + S->getSize();

    size_t NFIndex = Index + 1;
    // Skip aliases.
    auto NextAddrOrErr = FunctionSymbols[NFIndex].getAddress();
    if (auto E = NextAddrOrErr.takeError())
      return Error(std::move(E));
    uint64_t NextAddr = *NextAddrOrErr;
    while (NFIndex < FunctionSymbols.size() && NextAddr == Address) {
      ++NFIndex;
      auto NFAddrOrErr = FunctionSymbols[NFIndex].getAddress();
      if (auto E = NFAddrOrErr.takeError())
        return Error(std::move(E));
      NextAddr = *NFAddrOrErr;
    }

    auto NFSecOrErr = FunctionSymbols[NFIndex].getSection();
    if (auto E = NFSecOrErr.takeError())
      return Error(std::move(E));
    if (NFIndex < FunctionSymbols.size() && S == *NFSecOrErr) {
      auto EndAddressOrErr = FunctionSymbols[NFIndex].getAddress();
      if (auto E = EndAddressOrErr.takeError())
        return Error(std::move(E));
      EndAddress = *EndAddressOrErr;
    }

    const uint64_t SymbolSize = EndAddress - Address;
    const auto It = BC->getBinaryFunctions().find(Address);
    if (It == BC->getBinaryFunctions().end()) {
      LLVM_DEBUG(dbgs() << "BOLT-DEBUG: creating binary function for "
                        << SymbolName << "\n");
      BC->createBinaryFunction(std::move(SymbolName), *Section, Address,
                               SymbolSize);
    } else {
      It->second.addAlternativeName(std::move(SymbolName));
    }
  }

  for (auto &BFI : BC->getBinaryFunctions()) {
    BinaryFunction &Function = BFI.second;
    Function.setMaxSize(Function.getSize());

    ErrorOr<ArrayRef<uint8_t>> FunctionData = Function.getData();
    if (!FunctionData) {
      BC->errs() << "BOLT-ERROR: corresponding section is non-executable or "
                 << "empty for function " << Function << '\n';
      continue;
    }

    if (Function.getSize() == 0)
      Function.setSimple(false);
  }

  return Error::success();
}

void JITRewriteInstance::disableAllFunctions() {
  for (auto &BFI : BC->getBinaryFunctions()) {
    BinaryFunction &Function = BFI.second;
    Function.setSimple(false);
  }
}

void JITRewriteInstance::processFunctionContaining(uint64_t Address) {
  if (BinaryFunction *Func = BC->getBinaryFunctionContainingAddress(Address))
    Func->setSimple(true);
}

Error JITRewriteInstance::setProfile(StringRef Filename) {
  if (!sys::fs::exists(Filename))
    return errorCodeToError(make_error_code(errc::no_such_file_or_directory));

  ProfileReader = std::make_unique<DataAggregator>(Filename);
  return Error::success();
}

Error JITRewriteInstance::run() {
  adjustCommandLineOptions();

  if (Error E = preprocessProfileData())
    return Error(std::move(E));

  if (Error E = disassembleFunctions())
    return Error(std::move(E));

  if (Error E = processProfileDataPreCFG())
    return Error(std::move(E));

  if (Error E = buildFunctionsCFG())
    return Error(std::move(E));

  if (Error E = processProfileData())
    return Error(std::move(E));

  postProcessFunctions();

  return Error::success();
}

void JITRewriteInstance::printAll(raw_ostream &OS) {
  for (auto &BFI : BC->getBinaryFunctions()) {
    BinaryFunction &Function = BFI.second;
    if (Function.empty())
      continue;
    Function.print(OS, "after building cfg");
  }
}

void JITRewriteInstance::printFunctionContaining(raw_ostream &OS,
                                                 uint64_t Address) {
  if (BinaryFunction *Func = BC->getBinaryFunctionContainingAddress(Address)) {
    OS << formatv("Printing function containg address {0:x}\n", Address);
    Func->print(OS, "JIT on-demand inspection");
  }
}

} // namespace bolt
} // namespace llvm
