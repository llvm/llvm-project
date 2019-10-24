/*******************************************************************************
 *
 * University of Illinois/NCSA
 * Open Source License
 *
 * Copyright (c) 2003-2017 University of Illinois at Urbana-Champaign.
 * Modifications (c) 2018 Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *     * Redistributions of source code must retain the above copyright notice,
 *       this list of conditions and the following disclaimers.
 *
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimers in the
 *       documentation and/or other materials provided with the distribution.
 *
 *     * Neither the names of the LLVM Team, University of Illinois at
 *       Urbana-Champaign, nor the names of its contributors may be used to
 *       endorse or promote products derived from this Software without specific
 *       prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 ******************************************************************************/

#include "comgr-objdump.h"
#include "comgr.h"
#include "lld/Common/TargetOptionsCommandFlags.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Triple.h"
#include "llvm/CodeGen/FaultMaps.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/Symbolize/Symbolize.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCDisassembler/MCRelocationInfo.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/COFFImportFile.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/MachO.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/Wasm.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cctype>
#include <cstring>
#include <system_error>
#include <unordered_map>
#include <utility>

using namespace llvm;
using namespace object;

cl::opt<bool> Disassemble(
    "disassemble",
    cl::desc("Display assembler mnemonics for the machine instructions"));
static cl::alias Disassembled("d", cl::desc("Alias for --disassemble"),
                              cl::aliasopt(Disassemble));

cl::opt<bool> DisassembleAll(
    "disassemble-all",
    cl::desc("Display assembler mnemonics for the machine instructions"));
static cl::alias DisassembleAlld("D", cl::desc("Alias for --disassemble-all"),
                                 cl::aliasopt(DisassembleAll));

cl::opt<bool> Demangle("demangle", cl::desc("Demangle symbols names"),
                       cl::init(false));

static cl::alias DemangleShort("C", cl::desc("Alias for --demangle"),
                               cl::aliasopt(Demangle));

static cl::list<std::string>
    DisassembleFunctions("df", cl::CommaSeparated,
                         cl::desc("List of functions to disassemble"));

cl::opt<bool>
    Relocations("reloc",
                cl::desc("Display the relocation entries in the file"));
static cl::alias RelocationsShort("r", cl::desc("Alias for --reloc"),
                                  cl::NotHidden, cl::aliasopt(Relocations));

cl::opt<bool> DynamicRelocations(
    "dynamic-reloc",
    cl::desc("Display the dynamic relocation entries in the file"));
static cl::alias DynamicRelocationsd("R", cl::desc("Alias for --dynamic-reloc"),
                                     cl::aliasopt(DynamicRelocations));

cl::opt<bool> SectionContents("full-contents",
                              cl::desc("Display the content of each section"));
static cl::alias SectionContentsShort("s",
                                      cl::desc("Alias for --full-contents"),
                                      cl::aliasopt(SectionContents));

cl::opt<bool> SymbolTable("syms", cl::desc("Display the symbol table"));
static cl::alias SymbolTableShort("t", cl::desc("Alias for --syms"),
                                  cl::NotHidden, cl::aliasopt(SymbolTable));

cl::opt<bool> ExportsTrie("exports-trie",
                          cl::desc("Display mach-o exported symbols"));

cl::opt<bool> Rebase("rebase", cl::desc("Display mach-o rebasing info"));

cl::opt<bool> Bind("bind", cl::desc("Display mach-o binding info"));

cl::opt<bool> LazyBind("lazy-bind",
                       cl::desc("Display mach-o lazy binding info"));

cl::opt<bool> WeakBind("weak-bind",
                       cl::desc("Display mach-o weak binding info"));

cl::opt<bool> RawClangAST(
    "raw-clang-ast",
    cl::desc("Dump the raw binary contents of the clang AST section"));

static cl::opt<bool>
    MachOOpt("macho", cl::desc("Use MachO specific object file parser"));
static cl::alias MachOm("m", cl::desc("Alias for --macho"),
                        cl::aliasopt(MachOOpt));

cl::opt<std::string> TripleName("triple",
                                cl::desc("Target triple to disassemble for, "
                                         "see -version for available targets"));

std::string MCPU;

cl::opt<std::string> ArchName("arch-name",
                              cl::desc("Target arch to disassemble for, "
                                       "see -version for available targets"));

cl::opt<bool> SectionHeaders("section-headers",
                             cl::desc("Display summaries of the "
                                      "headers for each section."));
static cl::alias SectionHeadersShort("headers",
                                     cl::desc("Alias for --section-headers"),
                                     cl::aliasopt(SectionHeaders));
static cl::alias SectionHeadersShorter("h",
                                       cl::desc("Alias for --section-headers"),
                                       cl::aliasopt(SectionHeaders));

cl::list<std::string>
    FilterSections("section",
                   cl::desc("Operate on the specified sections only. "
                            "With -macho dump segment,section"));
cl::alias static FilterSectionsj("j", cl::desc("Alias for --section"),
                                 cl::aliasopt(FilterSections));

cl::opt<bool> NoShowRawInsn("no-show-raw-insn",
                            cl::desc("When disassembling "
                                     "instructions, do not print "
                                     "the instruction bytes."));
cl::opt<bool> NoLeadingAddr("no-leading-addr",
                            cl::desc("Print no leading address"));

cl::opt<bool> UnwindInfo("unwind-info", cl::desc("Display unwind information"));

static cl::alias UnwindInfoShort("u", cl::desc("Alias for --unwind-info"),
                                 cl::aliasopt(UnwindInfo));

cl::opt<bool> PrivateHeaders("private-headers",
                             cl::desc("Display format specific file headers"));

cl::opt<bool>
    FirstPrivateHeader("private-header",
                       cl::desc("Display only the first format specific file "
                                "header"));

static cl::alias PrivateHeadersShort("p",
                                     cl::desc("Alias for --private-headers"),
                                     cl::aliasopt(PrivateHeaders));

cl::opt<bool>
    FileHeaders("file-headers",
                cl::desc("Display the contents of the overall file header"));

static cl::alias FileHeadersShort("f", cl::desc("Alias for --file-headers"),
                                  cl::aliasopt(FileHeaders));

cl::opt<bool> ArchiveHeaders("archive-headers",
                             cl::desc("Display archive header information"));

cl::alias ArchiveHeadersShort("a", cl::desc("Alias for --archive-headers"),
                              cl::aliasopt(ArchiveHeaders));

cl::opt<bool> PrintImmHex("print-imm-hex",
                          cl::desc("Use hex format for immediate values"));

cl::opt<bool> PrintFaultMaps("fault-map-section",
                             cl::desc("Display contents of faultmap section"));

cl::opt<DIDumpType> DwarfDumpType(
    "dwarf", cl::init(DIDT_Null), cl::desc("Dump of dwarf debug sections:"),
    cl::values(clEnumValN(DIDT_DebugFrame, "frames", ".debug_frame")));

cl::opt<bool> PrintSource(
    "source",
    cl::desc(
        "Display source inlined with disassembly. Implies disassemble object"));

cl::alias PrintSourceShort("S", cl::desc("Alias for -source"),
                           cl::aliasopt(PrintSource));

cl::opt<bool> PrintLines("line-numbers",
                         cl::desc("Display source line numbers with "
                                  "disassembly. Implies disassemble object"));

cl::alias PrintLinesShort("l", cl::desc("Alias for -line-numbers"),
                          cl::aliasopt(PrintLines));

cl::opt<unsigned long long>
    StartAddress("start-address", cl::desc("Disassemble beginning at address"),
                 cl::value_desc("address"), cl::init(0));
cl::opt<unsigned long long> StopAddress("stop-address",
                                        cl::desc("Stop disassembly at address"),
                                        cl::value_desc("address"),
                                        cl::init(UINT64_MAX));
static StringRef ToolName = "DisassemblerAction";

typedef std::vector<std::tuple<uint64_t, StringRef, uint8_t>> SectionSymbolsTy;

namespace {
typedef std::function<bool(llvm::object::SectionRef const &)> FilterPredicate;

class SectionFilterIterator {
public:
  SectionFilterIterator(FilterPredicate P,
                        llvm::object::section_iterator const &I,
                        llvm::object::section_iterator const &E)
      : Predicate(std::move(P)), Iterator(I), End(E) {
    ScanPredicate();
  }
  const llvm::object::SectionRef &operator*() const { return *Iterator; }
  SectionFilterIterator &operator++() {
    ++Iterator;
    ScanPredicate();
    return *this;
  }
  bool operator!=(SectionFilterIterator const &Other) const {
    return Iterator != Other.Iterator;
  }

private:
  void ScanPredicate() {
    while (Iterator != End && !Predicate(*Iterator)) {
      ++Iterator;
    }
  }
  FilterPredicate Predicate;
  llvm::object::section_iterator Iterator;
  llvm::object::section_iterator End;
};

class SectionFilter {
public:
  SectionFilter(FilterPredicate P, llvm::object::ObjectFile const &O)
      : Predicate(std::move(P)), Object(O) {}
  SectionFilterIterator begin() {
    return SectionFilterIterator(Predicate, Object.section_begin(),
                                 Object.section_end());
  }
  SectionFilterIterator end() {
    return SectionFilterIterator(Predicate, Object.section_end(),
                                 Object.section_end());
  }

private:
  FilterPredicate Predicate;
  llvm::object::ObjectFile const &Object;
};

SectionFilter ToolSectionFilter(llvm::object::ObjectFile const &O) {
  return SectionFilter(
      [](llvm::object::SectionRef const &S) {
        if (FilterSections.empty())
          return true;
        Expected<StringRef> SecNameOrErr = S.getName();
        if (!SecNameOrErr) {
          consumeError(SecNameOrErr.takeError());
          return false;
        }
        StringRef String = *SecNameOrErr;
        return is_contained(FilterSections, String);
      },
      O);
}
} // namespace

static void error(std::error_code EC) {
  if (!EC)
    return;

  errs() << ToolName << ": error reading file: " << EC.message() << ".\n";
  errs().flush();
  exit(1);
}

static LLVM_ATTRIBUTE_NORETURN void error(Twine Message) {
  errs() << ToolName << ": " << Message << ".\n";
  errs().flush();
  exit(1);
}

static LLVM_ATTRIBUTE_NORETURN void report_error(StringRef File,
                                                 Twine Message) {
  errs() << ToolName << ": '" << File << "': " << Message << ".\n";
  exit(1);
}

static LLVM_ATTRIBUTE_NORETURN void report_error(StringRef File,
                                                 std::error_code EC) {
  assert(EC);
  errs() << ToolName << ": '" << File << "': " << EC.message() << ".\n";
  exit(1);
}

static LLVM_ATTRIBUTE_NORETURN void report_error(StringRef File,
                                                 llvm::Error E) {
  assert(E);
  std::string Buf;
  raw_string_ostream OS(Buf);
  logAllUnhandledErrors(std::move(E), OS, "");
  OS.flush();
  errs() << ToolName << ": '" << File << "': " << Buf;
  exit(1);
}

static LLVM_ATTRIBUTE_NORETURN void
report_error(StringRef ArchiveName, StringRef FileName, llvm::Error E,
             StringRef ArchitectureName = StringRef()) {
  assert(E);
  errs() << ToolName << ": ";
  if (ArchiveName != "")
    errs() << ArchiveName << "(" << FileName << ")";
  else
    errs() << "'" << FileName << "'";
  if (!ArchitectureName.empty())
    errs() << " (for architecture " << ArchitectureName << ")";
  std::string Buf;
  raw_string_ostream OS(Buf);
  logAllUnhandledErrors(std::move(E), OS, "");
  OS.flush();
  errs() << ": " << Buf;
  exit(1);
}

static LLVM_ATTRIBUTE_NORETURN void
report_error(StringRef ArchiveName, const object::Archive::Child &C,
             llvm::Error E, StringRef ArchitectureName = StringRef()) {
  Expected<StringRef> NameOrErr = C.getName();
  // TODO: if we have a error getting the name then it would be nice to print
  // the index of which archive member this is and or its offset in the
  // archive instead of "???" as the name.
  if (!NameOrErr) {
    consumeError(NameOrErr.takeError());
    report_error(ArchiveName, "???", std::move(E), ArchitectureName);
  } else
    report_error(ArchiveName, NameOrErr.get(), std::move(E), ArchitectureName);
}

static LLVM_ATTRIBUTE_NORETURN void report_error(llvm::Error E,
                                                 StringRef File) {
  report_error(File, std::move(E));
}

template <typename T, typename... Ts>
T unwrapOrError(Expected<T> EO, Ts &&... Args) {
  if (EO)
    return std::move(*EO);
  report_error(EO.takeError(), std::forward<Ts>(Args)...);
}

static const Target *getTarget(const ObjectFile *Obj = nullptr) {
  // Figure out the target triple.
  llvm::Triple TheTriple("unknown-unknown-unknown");
  if (TripleName.empty()) {
    if (Obj) {
      TheTriple = Obj->makeTriple();
    }
  } else {
    TheTriple.setTriple(Triple::normalize(TripleName));
    // Use the triple, but also try to combine with ARM build attributes.
    if (Obj) {
      auto Arch = Obj->getArch();
      if (Arch == Triple::arm || Arch == Triple::armeb) {
        Obj->setARMSubArch(TheTriple);
      }
    }
  }

  // Get the target specific parser.
  std::string Error;
  const Target *TheTarget =
      TargetRegistry::lookupTarget(ArchName, TheTriple, Error);
  if (!TheTarget) {
    if (Obj)
      report_error(Obj->getFileName(), "can't find target: " + Error);
    else
      error("can't find target: " + Error);
  }

  // Update the triple name and return the found target.
  TripleName = TheTriple.getTriple();
  return TheTarget;
}

static bool RelocAddressLess(RelocationRef a, RelocationRef b) {
  return a.getOffset() < b.getOffset();
}

namespace {
class SourcePrinter {
protected:
  DILineInfo OldLineInfo;
  const ObjectFile *Obj = nullptr;
  std::unique_ptr<symbolize::LLVMSymbolizer> Symbolizer;
  // File name to file contents of source
  std::unordered_map<std::string, std::unique_ptr<MemoryBuffer>> SourceCache;
  // Mark the line endings of the cached source
  std::unordered_map<std::string, std::vector<StringRef>> LineCache;

private:
  bool cacheSource(const std::string File);

public:
  SourcePrinter() = default;
  SourcePrinter(const ObjectFile *Obj, StringRef DefaultArch) : Obj(Obj) {
    symbolize::LLVMSymbolizer::Options SymbolizerOpts;
    SymbolizerOpts.PrintFunctions = DILineInfoSpecifier::FunctionNameKind::None;
    SymbolizerOpts.Demangle = false;
    SymbolizerOpts.DefaultArch = DefaultArch;
    Symbolizer.reset(new symbolize::LLVMSymbolizer(SymbolizerOpts));
  }
  virtual ~SourcePrinter() = default;
  virtual void printSourceLine(raw_ostream &OS,
                               object::SectionedAddress Address,
                               StringRef Delimiter = "; ");
};

bool SourcePrinter::cacheSource(const std::string File) {
  auto BufferOrError = MemoryBuffer::getFile(File);
  if (!BufferOrError)
    return false;
  // Chomp the file to get lines
  size_t BufferSize = (*BufferOrError)->getBufferSize();
  const char *BufferStart = (*BufferOrError)->getBufferStart();
  for (const char *Start = BufferStart, *End = BufferStart;
       End < BufferStart + BufferSize; End++)
    if (*End == '\n' || End == BufferStart + BufferSize - 1 ||
        (*End == '\r' && *(End + 1) == '\n')) {
      LineCache[File].push_back(StringRef(Start, End - Start));
      if (*End == '\r')
        End++;
      Start = End + 1;
    }
  SourceCache[File] = std::move(*BufferOrError);
  return true;
}

void SourcePrinter::printSourceLine(raw_ostream &OS,
                                    object::SectionedAddress Address,
                                    StringRef Delimiter) {
  if (!Symbolizer)
    return;
  DILineInfo LineInfo = DILineInfo();
  auto ExpectecLineInfo =
      Symbolizer->symbolizeCode(Obj->getFileName(), Address);
  if (!ExpectecLineInfo)
    consumeError(ExpectecLineInfo.takeError());
  else
    LineInfo = *ExpectecLineInfo;

  if ((LineInfo.FileName == "<invalid>") || OldLineInfo.Line == LineInfo.Line ||
      LineInfo.Line == 0)
    return;

  if (PrintLines)
    OS << Delimiter << LineInfo.FileName << ":" << LineInfo.Line << "\n";
  if (PrintSource) {
    if (SourceCache.find(LineInfo.FileName) == SourceCache.end())
      if (!cacheSource(LineInfo.FileName))
        return;
    auto FileBuffer = SourceCache.find(LineInfo.FileName);
    if (FileBuffer != SourceCache.end()) {
      auto LineBuffer = LineCache.find(LineInfo.FileName);
      if (LineBuffer != LineCache.end()) {
        if (LineInfo.Line > LineBuffer->second.size())
          return;
        // Vector begins at 0, line numbers are non-zero
        OS << Delimiter << LineBuffer->second[LineInfo.Line - 1].ltrim()
           << "\n";
      }
    }
  }
  OldLineInfo = LineInfo;
}

static bool isArmElf(const ObjectFile *Obj) {
  return (Obj->isELF() &&
          (Obj->getArch() == Triple::aarch64 ||
           Obj->getArch() == Triple::aarch64_be ||
           Obj->getArch() == Triple::arm || Obj->getArch() == Triple::armeb ||
           Obj->getArch() == Triple::thumb ||
           Obj->getArch() == Triple::thumbeb));
}

class PrettyPrinter {
public:
  virtual ~PrettyPrinter() = default;
  virtual void printInst(MCInstPrinter &IP, const MCInst *MI,
                         ArrayRef<uint8_t> Bytes,
                         object::SectionedAddress Address, raw_ostream &OS,
                         StringRef Annot, MCSubtargetInfo const &STI,
                         SourcePrinter *SP) {
    if (SP && (PrintSource || PrintLines))
      SP->printSourceLine(OS, Address);
    if (!NoLeadingAddr)
      OS << format("%8" PRIx64 ":", Address.Address);
    if (!NoShowRawInsn) {
      OS << "\t";
      dumpBytes(Bytes, OS);
    }
    if (MI)
      IP.printInst(MI, OS, "", STI);
    else
      OS << " <unknown>";
  }
};
PrettyPrinter PrettyPrinterInst;
class HexagonPrettyPrinter : public PrettyPrinter {
public:
  void printLead(ArrayRef<uint8_t> Bytes, uint64_t Address, raw_ostream &OS) {
    uint32_t opcode =
        (Bytes[3] << 24) | (Bytes[2] << 16) | (Bytes[1] << 8) | Bytes[0];
    if (!NoLeadingAddr)
      OS << format("%8" PRIx64 ":", Address);
    if (!NoShowRawInsn) {
      OS << "\t";
      dumpBytes(Bytes.slice(0, 4), OS);
      OS << format("%08" PRIx32, opcode);
    }
  }
  void printInst(MCInstPrinter &IP, const MCInst *MI, ArrayRef<uint8_t> Bytes,
                 object::SectionedAddress Address, raw_ostream &OS,
                 StringRef Annot, MCSubtargetInfo const &STI,
                 SourcePrinter *SP) override {
    if (SP && (PrintSource || PrintLines))
      SP->printSourceLine(OS, Address, "");
    if (!MI) {
      printLead(Bytes, Address.Address, OS);
      OS << " <unknown>";
      return;
    }
    std::string Buffer;
    {
      raw_string_ostream TempStream(Buffer);
      IP.printInst(MI, TempStream, "", STI);
    }
    StringRef Contents(Buffer);
    // Split off bundle attributes
    auto PacketBundle = Contents.rsplit('\n');
    // Split off first instruction from the rest
    auto HeadTail = PacketBundle.first.split('\n');
    auto Preamble = " { ";
    auto Separator = "";
    while (!HeadTail.first.empty()) {
      OS << Separator;
      Separator = "\n";
      if (SP && (PrintSource || PrintLines))
        SP->printSourceLine(OS, Address, "");
      printLead(Bytes, Address.Address, OS);
      OS << Preamble;
      Preamble = "   ";
      StringRef Inst;
      auto Duplex = HeadTail.first.split('\v');
      if (!Duplex.second.empty()) {
        OS << Duplex.first;
        OS << "; ";
        Inst = Duplex.second;
      } else
        Inst = HeadTail.first;
      OS << Inst;
      Bytes = Bytes.slice(4);
      Address.Address += 4;
      HeadTail = HeadTail.second.split('\n');
    }
    OS << " } " << PacketBundle.second;
  }
};
HexagonPrettyPrinter HexagonPrettyPrinterInst;

class AMDGCNPrettyPrinter : public PrettyPrinter {
public:
  void printInst(MCInstPrinter &IP, const MCInst *MI, ArrayRef<uint8_t> Bytes,
                 object::SectionedAddress Address, raw_ostream &OS,
                 StringRef Annot, MCSubtargetInfo const &STI,
                 SourcePrinter *SP) override {
    if (SP && (PrintSource || PrintLines))
      SP->printSourceLine(OS, Address);

    if (!MI) {
      OS << " <unknown>";
      return;
    }

    SmallString<40> InstStr;
    raw_svector_ostream IS(InstStr);

    IP.printInst(MI, IS, "", STI);

    OS << left_justify(IS.str(), 60)
       << format("// %012" PRIX64 ": ", Address.Address);
    typedef support::ulittle32_t U32;
    for (auto D : makeArrayRef(reinterpret_cast<const U32 *>(Bytes.data()),
                               Bytes.size() / sizeof(U32)))
      // D should be explicitly casted to uint32_t here as it is passed
      // by format to snprintf as vararg.
      OS << format("%08" PRIX32 " ", static_cast<uint32_t>(D));

    if (!Annot.empty())
      OS << "// " << Annot;
  }
};
AMDGCNPrettyPrinter AMDGCNPrettyPrinterInst;

class BPFPrettyPrinter : public PrettyPrinter {
public:
  void printInst(MCInstPrinter &IP, const MCInst *MI, ArrayRef<uint8_t> Bytes,
                 object::SectionedAddress Address, raw_ostream &OS,
                 StringRef Annot, MCSubtargetInfo const &STI,
                 SourcePrinter *SP) override {
    if (SP && (PrintSource || PrintLines))
      SP->printSourceLine(OS, Address);
    if (!NoLeadingAddr)
      OS << format("%8" PRId64 ":", Address.Address / 8);
    if (!NoShowRawInsn) {
      OS << "\t";
      dumpBytes(Bytes, OS);
    }
    if (MI)
      IP.printInst(MI, OS, "", STI);
    else
      OS << " <unknown>";
  }
};
BPFPrettyPrinter BPFPrettyPrinterInst;

PrettyPrinter &selectPrettyPrinter(Triple const &Triple) {
  switch (Triple.getArch()) {
  default:
    return PrettyPrinterInst;
  case Triple::hexagon:
    return HexagonPrettyPrinterInst;
  case Triple::amdgcn:
    return AMDGCNPrettyPrinterInst;
  case Triple::bpfel:
  case Triple::bpfeb:
    return BPFPrettyPrinterInst;
  }
}
} // namespace

template <class ELFT>
static std::error_code getRelocationValueString(const ELFObjectFile<ELFT> *Obj,
                                                const RelocationRef &RelRef,
                                                SmallVectorImpl<char> &Result) {
  DataRefImpl Rel = RelRef.getRawDataRefImpl();

  typedef typename ELFObjectFile<ELFT>::Elf_Sym Elf_Sym;
  typedef typename ELFObjectFile<ELFT>::Elf_Shdr Elf_Shdr;
  typedef typename ELFObjectFile<ELFT>::Elf_Rela Elf_Rela;

  const ELFFile<ELFT> &EF = *Obj->getELFFile();

  auto SecOrErr = EF.getSection(Rel.d.a);
  if (!SecOrErr)
    return errorToErrorCode(SecOrErr.takeError());
  const Elf_Shdr *Sec = *SecOrErr;
  auto SymTabOrErr = EF.getSection(Sec->sh_link);
  if (!SymTabOrErr)
    return errorToErrorCode(SymTabOrErr.takeError());
  const Elf_Shdr *SymTab = *SymTabOrErr;
  assert(SymTab->sh_type == ELF::SHT_SYMTAB ||
         SymTab->sh_type == ELF::SHT_DYNSYM);
  auto StrTabSec = EF.getSection(SymTab->sh_link);
  if (!StrTabSec)
    return errorToErrorCode(StrTabSec.takeError());
  auto StrTabOrErr = EF.getStringTable(*StrTabSec);
  if (!StrTabOrErr)
    return errorToErrorCode(StrTabOrErr.takeError());
  StringRef StrTab = *StrTabOrErr;
  int64_t addend = 0;
  // If there is no Symbol associated with the relocation, we set the undef
  // boolean value to 'true'. This will prevent us from calling functions that
  // requires the relocation to be associated with a symbol.
  bool undef = false;
  switch (Sec->sh_type) {
  default:
    return object_error::parse_failed;
  case ELF::SHT_REL: {
    // TODO: Read implicit addend from section data.
    break;
  }
  case ELF::SHT_RELA: {
    const Elf_Rela *ERela = Obj->getRela(Rel);
    addend = ERela->r_addend;
    undef = ERela->getSymbol(false) == 0;
    break;
  }
  }
  StringRef Target;
  if (!undef) {
    symbol_iterator SI = RelRef.getSymbol();
    const Elf_Sym *symb = Obj->getSymbol(SI->getRawDataRefImpl());
    if (symb->getType() == ELF::STT_SECTION) {
      Expected<section_iterator> SymSI = SI->getSection();
      if (!SymSI)
        return errorToErrorCode(SymSI.takeError());
      const Elf_Shdr *SymSec = Obj->getSection((*SymSI)->getRawDataRefImpl());
      auto SecName = EF.getSectionName(SymSec);
      if (!SecName)
        return errorToErrorCode(SecName.takeError());
      Target = *SecName;
    } else {
      Expected<StringRef> SymName = symb->getName(StrTab);
      if (!SymName)
        return errorToErrorCode(SymName.takeError());
      Target = *SymName;
    }
  } else
    Target = "*ABS*";

  // Default scheme is to print Target, as well as "+ <addend>" for nonzero
  // addend. Should be acceptable for all normal purposes.
  std::string fmtbuf;
  raw_string_ostream fmt(fmtbuf);
  fmt << Target;
  if (addend != 0)
    fmt << (addend < 0 ? "" : "+") << addend;
  fmt.flush();
  Result.append(fmtbuf.begin(), fmtbuf.end());
  return std::error_code();
}

static std::error_code getRelocationValueString(const ELFObjectFileBase *Obj,
                                                const RelocationRef &Rel,
                                                SmallVectorImpl<char> &Result) {
  if (auto *ELF32LE = dyn_cast<ELF32LEObjectFile>(Obj))
    return getRelocationValueString(ELF32LE, Rel, Result);
  if (auto *ELF64LE = dyn_cast<ELF64LEObjectFile>(Obj))
    return getRelocationValueString(ELF64LE, Rel, Result);
  if (auto *ELF32BE = dyn_cast<ELF32BEObjectFile>(Obj))
    return getRelocationValueString(ELF32BE, Rel, Result);
  auto *ELF64BE = cast<ELF64BEObjectFile>(Obj);
  return getRelocationValueString(ELF64BE, Rel, Result);
}

static std::error_code getRelocationValueString(const COFFObjectFile *Obj,
                                                const RelocationRef &Rel,
                                                SmallVectorImpl<char> &Result) {
  symbol_iterator SymI = Rel.getSymbol();
  Expected<StringRef> SymNameOrErr = SymI->getName();
  if (!SymNameOrErr)
    return errorToErrorCode(SymNameOrErr.takeError());
  StringRef SymName = *SymNameOrErr;
  Result.append(SymName.begin(), SymName.end());
  return std::error_code();
}

static void printRelocationTargetName(const MachOObjectFile *O,
                                      const MachO::any_relocation_info &RE,
                                      raw_string_ostream &fmt) {
  bool IsScattered = O->isRelocationScattered(RE);

  // Target of a scattered relocation is an address.  In the interest of
  // generating pretty output, scan through the symbol table looking for a
  // symbol that aligns with that address.  If we find one, print it.
  // Otherwise, we just print the hex address of the target.
  if (IsScattered) {
    uint32_t Val = O->getPlainRelocationSymbolNum(RE);

    for (const SymbolRef &Symbol : O->symbols()) {
      std::error_code ec;
      Expected<uint64_t> Addr = Symbol.getAddress();
      if (!Addr)
        report_error(O->getFileName(), Addr.takeError());
      if (*Addr != Val)
        continue;
      Expected<StringRef> Name = Symbol.getName();
      if (!Name)
        report_error(O->getFileName(), Name.takeError());
      fmt << *Name;
      return;
    }

    // If we couldn't find a symbol that this relocation refers to, try
    // to find a section beginning instead.
    for (const SectionRef &Section : ToolSectionFilter(*O)) {
      std::error_code ec;

      uint64_t Addr = Section.getAddress();
      if (Addr != Val)
        continue;
      Expected<StringRef> NameOrErr = Section.getName();
      if (!NameOrErr)
        report_error(O->getFileName(), NameOrErr.takeError());
      fmt << *NameOrErr;
      return;
    }

    fmt << format("0x%x", Val);
    return;
  }

  StringRef S;
  bool isExtern = O->getPlainRelocationExternal(RE);
  uint64_t Val = O->getPlainRelocationSymbolNum(RE);

  if (O->getAnyRelocationType(RE) == MachO::ARM64_RELOC_ADDEND) {
    fmt << format("0x%0" PRIx64, Val);
    return;
  } else if (isExtern) {
    symbol_iterator SI = O->symbol_begin();
    advance(SI, Val);
    Expected<StringRef> SOrErr = SI->getName();
    if (!SOrErr)
      report_error(O->getFileName(), SOrErr.takeError());
    S = *SOrErr;
  } else {
    section_iterator SI = O->section_begin();
    // Adjust for the fact that sections are 1-indexed.
    advance(SI, Val - 1);
    Expected<StringRef> SOrErr = SI->getName();
    if (!SOrErr)
      consumeError(SOrErr.takeError());
    else
      S = *SOrErr;
  }

  fmt << S;
}

static std::error_code getRelocationValueString(const WasmObjectFile *Obj,
                                                const RelocationRef &RelRef,
                                                SmallVectorImpl<char> &Result) {
  const wasm::WasmRelocation &Rel = Obj->getWasmRelocation(RelRef);
  std::string fmtbuf;
  raw_string_ostream fmt(fmtbuf);
  fmt << Rel.Index << (Rel.Addend < 0 ? "" : "+") << Rel.Addend;
  fmt.flush();
  Result.append(fmtbuf.begin(), fmtbuf.end());
  return std::error_code();
}

static std::error_code getRelocationValueString(const MachOObjectFile *Obj,
                                                const RelocationRef &RelRef,
                                                SmallVectorImpl<char> &Result) {
  DataRefImpl Rel = RelRef.getRawDataRefImpl();
  MachO::any_relocation_info RE = Obj->getRelocation(Rel);

  unsigned Arch = Obj->getArch();

  std::string fmtbuf;
  raw_string_ostream fmt(fmtbuf);
  unsigned Type = Obj->getAnyRelocationType(RE);
  bool IsPCRel = Obj->getAnyRelocationPCRel(RE);

  // Determine any addends that should be displayed with the relocation.
  // These require decoding the relocation type, which is triple-specific.

  // X86_64 has entirely custom relocation types.
  if (Arch == Triple::x86_64) {
    bool isPCRel = Obj->getAnyRelocationPCRel(RE);

    switch (Type) {
    case MachO::X86_64_RELOC_GOT_LOAD:
    case MachO::X86_64_RELOC_GOT: {
      printRelocationTargetName(Obj, RE, fmt);
      fmt << "@GOT";
      if (isPCRel)
        fmt << "PCREL";
      break;
    }
    case MachO::X86_64_RELOC_SUBTRACTOR: {
      DataRefImpl RelNext = Rel;
      Obj->moveRelocationNext(RelNext);
      MachO::any_relocation_info RENext = Obj->getRelocation(RelNext);

      // X86_64_RELOC_SUBTRACTOR must be followed by a relocation of type
      // X86_64_RELOC_UNSIGNED.
      // NOTE: Scattered relocations don't exist on x86_64.
      unsigned RType = Obj->getAnyRelocationType(RENext);
      if (RType != MachO::X86_64_RELOC_UNSIGNED)
        report_error(Obj->getFileName(), "Expected X86_64_RELOC_UNSIGNED after "
                                         "X86_64_RELOC_SUBTRACTOR.");

      // The X86_64_RELOC_UNSIGNED contains the minuend symbol;
      // X86_64_RELOC_SUBTRACTOR contains the subtrahend.
      printRelocationTargetName(Obj, RENext, fmt);
      fmt << "-";
      printRelocationTargetName(Obj, RE, fmt);
      break;
    }
    case MachO::X86_64_RELOC_TLV:
      printRelocationTargetName(Obj, RE, fmt);
      fmt << "@TLV";
      if (isPCRel)
        fmt << "P";
      break;
    case MachO::X86_64_RELOC_SIGNED_1:
      printRelocationTargetName(Obj, RE, fmt);
      fmt << "-1";
      break;
    case MachO::X86_64_RELOC_SIGNED_2:
      printRelocationTargetName(Obj, RE, fmt);
      fmt << "-2";
      break;
    case MachO::X86_64_RELOC_SIGNED_4:
      printRelocationTargetName(Obj, RE, fmt);
      fmt << "-4";
      break;
    default:
      printRelocationTargetName(Obj, RE, fmt);
      break;
    }
    // X86 and ARM share some relocation types in common.
  } else if (Arch == Triple::x86 || Arch == Triple::arm ||
             Arch == Triple::ppc) {
    // Generic relocation types...
    switch (Type) {
    case MachO::GENERIC_RELOC_PAIR: // prints no info
      return std::error_code();
    case MachO::GENERIC_RELOC_SECTDIFF: {
      DataRefImpl RelNext = Rel;
      Obj->moveRelocationNext(RelNext);
      MachO::any_relocation_info RENext = Obj->getRelocation(RelNext);

      // X86 sect diff's must be followed by a relocation of type
      // GENERIC_RELOC_PAIR.
      unsigned RType = Obj->getAnyRelocationType(RENext);

      if (RType != MachO::GENERIC_RELOC_PAIR)
        report_error(Obj->getFileName(), "Expected GENERIC_RELOC_PAIR after "
                                         "GENERIC_RELOC_SECTDIFF.");

      printRelocationTargetName(Obj, RE, fmt);
      fmt << "-";
      printRelocationTargetName(Obj, RENext, fmt);
      break;
    }
    }

    if (Arch == Triple::x86 || Arch == Triple::ppc) {
      switch (Type) {
      case MachO::GENERIC_RELOC_LOCAL_SECTDIFF: {
        DataRefImpl RelNext = Rel;
        Obj->moveRelocationNext(RelNext);
        MachO::any_relocation_info RENext = Obj->getRelocation(RelNext);

        // X86 sect diff's must be followed by a relocation of type
        // GENERIC_RELOC_PAIR.
        unsigned RType = Obj->getAnyRelocationType(RENext);
        if (RType != MachO::GENERIC_RELOC_PAIR)
          report_error(Obj->getFileName(), "Expected GENERIC_RELOC_PAIR after "
                                           "GENERIC_RELOC_LOCAL_SECTDIFF.");

        printRelocationTargetName(Obj, RE, fmt);
        fmt << "-";
        printRelocationTargetName(Obj, RENext, fmt);
        break;
      }
      case MachO::GENERIC_RELOC_TLV: {
        printRelocationTargetName(Obj, RE, fmt);
        fmt << "@TLV";
        if (IsPCRel)
          fmt << "P";
        break;
      }
      default:
        printRelocationTargetName(Obj, RE, fmt);
      }
    } else { // ARM-specific relocations
      switch (Type) {
      case MachO::ARM_RELOC_HALF:
      case MachO::ARM_RELOC_HALF_SECTDIFF: {
        // Half relocations steal a bit from the length field to encode
        // whether this is an upper16 or a lower16 relocation.
        bool isUpper = (Obj->getAnyRelocationLength(RE) & 0x1) == 1;

        if (isUpper)
          fmt << ":upper16:(";
        else
          fmt << ":lower16:(";
        printRelocationTargetName(Obj, RE, fmt);

        DataRefImpl RelNext = Rel;
        Obj->moveRelocationNext(RelNext);
        MachO::any_relocation_info RENext = Obj->getRelocation(RelNext);

        // ARM half relocs must be followed by a relocation of type
        // ARM_RELOC_PAIR.
        unsigned RType = Obj->getAnyRelocationType(RENext);
        if (RType != MachO::ARM_RELOC_PAIR)
          report_error(Obj->getFileName(), "Expected ARM_RELOC_PAIR after "
                                           "ARM_RELOC_HALF");

        // NOTE: The half of the target virtual address is stashed in the
        // address field of the secondary relocation, but we can't reverse
        // engineer the constant offset from it without decoding the movw/movt
        // instruction to find the other half in its immediate field.

        // ARM_RELOC_HALF_SECTDIFF encodes the second section in the
        // symbol/section pointer of the follow-on relocation.
        if (Type == MachO::ARM_RELOC_HALF_SECTDIFF) {
          fmt << "-";
          printRelocationTargetName(Obj, RENext, fmt);
        }

        fmt << ")";
        break;
      }
      default: {
        printRelocationTargetName(Obj, RE, fmt);
      }
      }
    }
  } else
    printRelocationTargetName(Obj, RE, fmt);

  fmt.flush();
  Result.append(fmtbuf.begin(), fmtbuf.end());
  return std::error_code();
}

static std::error_code getRelocationValueString(const RelocationRef &Rel,
                                                SmallVectorImpl<char> &Result) {
  const ObjectFile *Obj = Rel.getObject();
  if (auto *ELF = dyn_cast<ELFObjectFileBase>(Obj))
    return getRelocationValueString(ELF, Rel, Result);
  if (auto *COFF = dyn_cast<COFFObjectFile>(Obj))
    return getRelocationValueString(COFF, Rel, Result);
  if (auto *Wasm = dyn_cast<WasmObjectFile>(Obj))
    return getRelocationValueString(Wasm, Rel, Result);
  if (auto *MachO = dyn_cast<MachOObjectFile>(Obj))
    return getRelocationValueString(MachO, Rel, Result);
  llvm_unreachable("unknown object file format");
}

/// @brief Indicates whether this relocation should hidden when listing
/// relocations, usually because it is the trailing part of a multipart
/// relocation that will be printed as part of the leading relocation.
static bool getHidden(RelocationRef RelRef) {
  const ObjectFile *Obj = RelRef.getObject();
  auto *MachO = dyn_cast<MachOObjectFile>(Obj);
  if (!MachO)
    return false;

  unsigned Arch = MachO->getArch();
  DataRefImpl Rel = RelRef.getRawDataRefImpl();
  uint64_t Type = MachO->getRelocationType(Rel);

  // On arches that use the generic relocations, GENERIC_RELOC_PAIR
  // is always hidden.
  if (Arch == Triple::x86 || Arch == Triple::arm || Arch == Triple::ppc) {
    if (Type == MachO::GENERIC_RELOC_PAIR)
      return true;
  } else if (Arch == Triple::x86_64) {
    // On x86_64, X86_64_RELOC_UNSIGNED is hidden only when it follows
    // an X86_64_RELOC_SUBTRACTOR.
    if (Type == MachO::X86_64_RELOC_UNSIGNED && Rel.d.a > 0) {
      DataRefImpl RelPrev = Rel;
      RelPrev.d.a--;
      uint64_t PrevType = MachO->getRelocationType(RelPrev);
      if (PrevType == MachO::X86_64_RELOC_SUBTRACTOR)
        return true;
    }
  }

  return false;
}

static uint8_t getElfSymbolType(const ObjectFile *Obj, const SymbolRef &Sym) {
  assert(Obj->isELF());
  if (auto *Elf32LEObj = dyn_cast<ELF32LEObjectFile>(Obj))
    return Elf32LEObj->getSymbol(Sym.getRawDataRefImpl())->getType();
  if (auto *Elf64LEObj = dyn_cast<ELF64LEObjectFile>(Obj))
    return Elf64LEObj->getSymbol(Sym.getRawDataRefImpl())->getType();
  if (auto *Elf32BEObj = dyn_cast<ELF32BEObjectFile>(Obj))
    return Elf32BEObj->getSymbol(Sym.getRawDataRefImpl())->getType();
  if (auto *Elf64BEObj = cast<ELF64BEObjectFile>(Obj))
    return Elf64BEObj->getSymbol(Sym.getRawDataRefImpl())->getType();
  llvm_unreachable("Unsupported binary format");
}

template <class ELFT>
static void
addDynamicElfSymbols(const ELFObjectFile<ELFT> *Obj,
                     std::map<SectionRef, SectionSymbolsTy> &AllSymbols) {
  for (auto Symbol : Obj->getDynamicSymbolIterators()) {
    uint8_t SymbolType = Symbol.getELFType();
    if (SymbolType != ELF::STT_FUNC || Symbol.getSize() == 0)
      continue;

    Expected<uint64_t> AddressOrErr = Symbol.getAddress();
    if (!AddressOrErr)
      report_error(Obj->getFileName(), AddressOrErr.takeError());
    uint64_t Address = *AddressOrErr;

    Expected<StringRef> Name = Symbol.getName();
    if (!Name)
      report_error(Obj->getFileName(), Name.takeError());
    if (Name->empty())
      continue;

    Expected<section_iterator> SectionOrErr = Symbol.getSection();
    if (!SectionOrErr)
      report_error(Obj->getFileName(), SectionOrErr.takeError());
    section_iterator SecI = *SectionOrErr;
    if (SecI == Obj->section_end())
      continue;

    AllSymbols[*SecI].emplace_back(Address, *Name, SymbolType);
  }
}

static void
addDynamicElfSymbols(const ObjectFile *Obj,
                     std::map<SectionRef, SectionSymbolsTy> &AllSymbols) {
  assert(Obj->isELF());
  if (auto *Elf32LEObj = dyn_cast<ELF32LEObjectFile>(Obj))
    addDynamicElfSymbols(Elf32LEObj, AllSymbols);
  else if (auto *Elf64LEObj = dyn_cast<ELF64LEObjectFile>(Obj))
    addDynamicElfSymbols(Elf64LEObj, AllSymbols);
  else if (auto *Elf32BEObj = dyn_cast<ELF32BEObjectFile>(Obj))
    addDynamicElfSymbols(Elf32BEObj, AllSymbols);
  else if (auto *Elf64BEObj = cast<ELF64BEObjectFile>(Obj))
    addDynamicElfSymbols(Elf64BEObj, AllSymbols);
  else
    llvm_unreachable("Unsupported binary format");
}

void llvm::DisassemHelper::DisassembleObject(const ObjectFile *Obj,
                                             bool InlineRelocs) {
  if (StartAddress > StopAddress)
    error("Start address should be less than stop address");

  const Target *TheTarget = getTarget(Obj);

  // Package up features to be passed to target/subtarget
  SubtargetFeatures Features = Obj->getFeatures();
  std::vector<std::string> MAttrs = lld::getMAttrs();
  if (MAttrs.size()) {
    for (unsigned i = 0; i != MAttrs.size(); ++i)
      Features.AddFeature(MAttrs[i]);
  }

  std::unique_ptr<const MCRegisterInfo> MRI(
      TheTarget->createMCRegInfo(TripleName));
  if (!MRI)
    report_error(Obj->getFileName(),
                 "no register info for target " + TripleName);

  // Set up disassembler.
  llvm::MCTargetOptions MCOptions;
  std::unique_ptr<const MCAsmInfo> AsmInfo(
      TheTarget->createMCAsmInfo(*MRI, TripleName, MCOptions));
  if (!AsmInfo)
    report_error(Obj->getFileName(),
                 "no assembly info for target " + TripleName);
  std::unique_ptr<const MCSubtargetInfo> STI(
      TheTarget->createMCSubtargetInfo(TripleName, MCPU, Features.getString()));
  if (!STI)
    report_error(Obj->getFileName(),
                 "no subtarget info for target " + TripleName);
  std::unique_ptr<const MCInstrInfo> MII(TheTarget->createMCInstrInfo());
  if (!MII)
    report_error(Obj->getFileName(),
                 "no instruction info for target " + TripleName);
  MCObjectFileInfo MOFI;
  MCContext Ctx(AsmInfo.get(), MRI.get(), &MOFI);
  // FIXME: for now initialize MCObjectFileInfo with default values
  MOFI.InitMCObjectFileInfo(Triple(TripleName), false, Ctx);

  std::unique_ptr<MCDisassembler> DisAsm(
      TheTarget->createMCDisassembler(*STI, Ctx));
  if (!DisAsm)
    report_error(Obj->getFileName(),
                 "no disassembler for target " + TripleName);

  std::unique_ptr<const MCInstrAnalysis> MIA(
      TheTarget->createMCInstrAnalysis(MII.get()));

  int AsmPrinterVariant = AsmInfo->getAssemblerDialect();
  std::unique_ptr<MCInstPrinter> IP(TheTarget->createMCInstPrinter(
      Triple(TripleName), AsmPrinterVariant, *AsmInfo, *MII, *MRI));
  if (!IP)
    report_error(Obj->getFileName(),
                 "no instruction printer for target " + TripleName);
  IP->setPrintImmHex(PrintImmHex);
  PrettyPrinter &PIP = selectPrettyPrinter(Triple(TripleName));

  StringRef Fmt = Obj->getBytesInAddress() > 4 ? "\t\t%016" PRIx64 ":  "
                                               : "\t\t\t%08" PRIx64 ":  ";

  SourcePrinter SP(Obj, TheTarget->getName());

  // Create a mapping, RelocSecs = SectionRelocMap[S], where sections
  // in RelocSecs contain the relocations for section S.
  std::error_code EC;
  std::map<SectionRef, SmallVector<SectionRef, 1>> SectionRelocMap;
  uint64_t I = (uint64_t)-1;
  for (const SectionRef &Section : ToolSectionFilter(*Obj)) {
    ++I;
    Expected<section_iterator> Sec2OrErr = Section.getRelocatedSection();
    if (!Sec2OrErr)
      report_error(Obj->getFileName(),
                   "section (" + Twine(I) +
                       "): failed to get a relocated section: " +
                       toString(Sec2OrErr.takeError()));
    section_iterator Sec2 = *Sec2OrErr;
    if (Sec2 != Obj->section_end())
      SectionRelocMap[*Sec2].push_back(Section);
  }

  // Create a mapping from virtual address to symbol name.  This is used to
  // pretty print the symbols while disassembling.
  std::map<SectionRef, SectionSymbolsTy> AllSymbols;
  for (const SymbolRef &Symbol : Obj->symbols()) {
    Expected<uint64_t> AddressOrErr = Symbol.getAddress();
    if (!AddressOrErr)
      report_error(Obj->getFileName(), AddressOrErr.takeError());
    uint64_t Address = *AddressOrErr;

    Expected<StringRef> Name = Symbol.getName();
    if (!Name)
      report_error(Obj->getFileName(), Name.takeError());
    if (Name->empty())
      continue;

    Expected<section_iterator> SectionOrErr = Symbol.getSection();
    if (!SectionOrErr)
      report_error(Obj->getFileName(), SectionOrErr.takeError());
    section_iterator SecI = *SectionOrErr;
    if (SecI == Obj->section_end())
      continue;

    uint8_t SymbolType = ELF::STT_NOTYPE;
    if (Obj->isELF())
      SymbolType = getElfSymbolType(Obj, Symbol);

    AllSymbols[*SecI].emplace_back(Address, *Name, SymbolType);
  }
  if (AllSymbols.empty() && Obj->isELF())
    addDynamicElfSymbols(Obj, AllSymbols);

  // Create a mapping from virtual address to section.
  std::vector<std::pair<uint64_t, SectionRef>> SectionAddresses;
  for (SectionRef Sec : Obj->sections())
    SectionAddresses.emplace_back(Sec.getAddress(), Sec);
  array_pod_sort(SectionAddresses.begin(), SectionAddresses.end());

  // Linked executables (.exe and .dll files) typically don't include a real
  // symbol table but they might contain an export table.
  if (const auto *COFFObj = dyn_cast<COFFObjectFile>(Obj)) {
    for (const auto &ExportEntry : COFFObj->export_directories()) {
      StringRef Name;
      error(ExportEntry.getSymbolName(Name));
      if (Name.empty())
        continue;
      uint32_t RVA;
      error(ExportEntry.getExportRVA(RVA));

      uint64_t VA = COFFObj->getImageBase() + RVA;
      auto Sec = std::upper_bound(
          SectionAddresses.begin(), SectionAddresses.end(), VA,
          [](uint64_t LHS, const std::pair<uint64_t, SectionRef> &RHS) {
            return LHS < RHS.first;
          });
      if (Sec != SectionAddresses.begin())
        --Sec;
      else
        Sec = SectionAddresses.end();

      if (Sec != SectionAddresses.end())
        AllSymbols[Sec->second].emplace_back(VA, Name, ELF::STT_NOTYPE);
    }
  }

  // Sort all the symbols, this allows us to use a simple binary search to find
  // a symbol near an address.
  for (std::pair<const SectionRef, SectionSymbolsTy> &SecSyms : AllSymbols)
    array_pod_sort(SecSyms.second.begin(), SecSyms.second.end());

  for (const SectionRef &Section : ToolSectionFilter(*Obj)) {
    if (!DisassembleAll && (!Section.isText() || Section.isVirtual()))
      continue;

    uint64_t SectionAddr = Section.getAddress();
    uint64_t SectSize = Section.getSize();
    if (!SectSize)
      continue;

    // Get the list of all the symbols in this section.
    SectionSymbolsTy &Symbols = AllSymbols[Section];
    std::vector<uint64_t> DataMappingSymsAddr;
    std::vector<uint64_t> TextMappingSymsAddr;
    if (isArmElf(Obj)) {
      for (const auto &Symb : Symbols) {
        uint64_t Address = std::get<0>(Symb);
        StringRef Name = std::get<1>(Symb);
        if (Name.startswith("$d"))
          DataMappingSymsAddr.push_back(Address - SectionAddr);
        if (Name.startswith("$x"))
          TextMappingSymsAddr.push_back(Address - SectionAddr);
        if (Name.startswith("$a"))
          TextMappingSymsAddr.push_back(Address - SectionAddr);
        if (Name.startswith("$t"))
          TextMappingSymsAddr.push_back(Address - SectionAddr);
      }
    }

    std::sort(DataMappingSymsAddr.begin(), DataMappingSymsAddr.end());
    std::sort(TextMappingSymsAddr.begin(), TextMappingSymsAddr.end());

    if (Obj->isELF() && Obj->getArch() == Triple::amdgcn) {
      // AMDGPU disassembler uses symbolizer for printing labels
      std::unique_ptr<MCRelocationInfo> RelInfo(
          TheTarget->createMCRelocationInfo(TripleName, Ctx));
      if (RelInfo) {
        std::unique_ptr<MCSymbolizer> Symbolizer(TheTarget->createMCSymbolizer(
            TripleName, nullptr, nullptr, &Symbols, &Ctx, std::move(RelInfo)));
        DisAsm->setSymbolizer(std::move(Symbolizer));
      }
    }

    // Make a list of all the relocations for this section.
    std::vector<RelocationRef> Rels;
    if (InlineRelocs) {
      for (const SectionRef &RelocSec : SectionRelocMap[Section]) {
        for (const RelocationRef &Reloc : RelocSec.relocations()) {
          Rels.push_back(Reloc);
        }
      }
    }

    // Sort relocations by address.
    std::sort(Rels.begin(), Rels.end(), RelocAddressLess);

    StringRef SegmentName = "";
    if (const MachOObjectFile *MachO = dyn_cast<const MachOObjectFile>(Obj)) {
      DataRefImpl DR = Section.getRawDataRefImpl();
      SegmentName = MachO->getSectionFinalSegmentName(DR);
    }
    StringRef name = unwrapOrError(Section.getName(), Obj->getFileName());

    if ((SectionAddr <= StopAddress) &&
        (SectionAddr + SectSize) >= StartAddress) {
      OutS << "Disassembly of section ";
      if (!SegmentName.empty())
        OutS << SegmentName << ",";
      OutS << name << ':';
    }

    // If the section has no symbol at the start, just insert a dummy one.
    if (Symbols.empty() || std::get<0>(Symbols[0]) != 0) {
      Symbols.insert(
          Symbols.begin(),
          std::make_tuple(SectionAddr, name,
                          Section.isText() ? ELF::STT_FUNC : ELF::STT_OBJECT));
    }

    SmallString<40> Comments;
    raw_svector_ostream CommentStream(Comments);

    StringRef BytesStr;
    Expected<StringRef> ExpBytesStr = Section.getContents();
    if (ExpBytesStr)
      BytesStr = *ExpBytesStr;
    else
      consumeError(ExpBytesStr.takeError());

    ArrayRef<uint8_t> Bytes(reinterpret_cast<const uint8_t *>(BytesStr.data()),
                            BytesStr.size());

    uint64_t Size;
    uint64_t Index;

    std::vector<RelocationRef>::const_iterator rel_cur = Rels.begin();
    std::vector<RelocationRef>::const_iterator rel_end = Rels.end();
    // Disassemble symbol by symbol.
    for (unsigned si = 0, se = Symbols.size(); si != se; ++si) {
      uint64_t Start = std::get<0>(Symbols[si]) - SectionAddr;
      // The end is either the section end or the beginning of the next
      // symbol.
      uint64_t End = (si == se - 1)
                         ? SectSize
                         : std::get<0>(Symbols[si + 1]) - SectionAddr;
      // Don't try to disassemble beyond the end of section contents.
      if (End > SectSize)
        End = SectSize;
      // If this symbol has the same address as the next symbol, then skip it.
      if (Start >= End)
        continue;

      // Check if we need to skip symbol
      // Skip if the symbol's data is not between StartAddress and StopAddress
      if (End + SectionAddr < StartAddress ||
          Start + SectionAddr > StopAddress) {
        continue;
      }

      // Stop disassembly at the stop address specified
      if (End + SectionAddr > StopAddress)
        End = StopAddress - SectionAddr;

      if (Obj->isELF() && Obj->getArch() == Triple::amdgcn) {
        // make size 4 bytes folded
        End = Start + ((End - Start) & ~0x3ull);
        if (std::get<2>(Symbols[si]) == ELF::STT_AMDGPU_HSA_KERNEL) {
          // skip amd_kernel_code_t at the begining of kernel symbol (256 bytes)
          Start += 256;
        }
        if (si == se - 1 ||
            std::get<2>(Symbols[si + 1]) == ELF::STT_AMDGPU_HSA_KERNEL) {
          // cut trailing zeroes at the end of kernel
          // cut up to 256 bytes
          const uint64_t EndAlign = 256;
          const auto Limit = End - (std::min)(EndAlign, End - Start);
          while (End > Limit && *reinterpret_cast<const support::ulittle32_t *>(
                                    &Bytes[End - 4]) == 0)
            End -= 4;
        }
      }

      // COMGR TBD: Get rid of ".text:"??
      OutS << '\n' << std::get<1>(Symbols[si]) << ":\n";

#ifndef NDEBUG
      raw_ostream &DebugOut = DebugFlag ? dbgs() : nulls();
#else
      raw_ostream &DebugOut = nulls();
#endif

      for (Index = Start; Index < End; Index += Size) {
        MCInst Inst;

        if (Index + SectionAddr < StartAddress ||
            Index + SectionAddr > StopAddress) {
          // skip byte by byte till StartAddress is reached
          Size = 1;
          continue;
        }
        // AArch64 ELF binaries can interleave data and text in the
        // same section. We rely on the markers introduced to
        // understand what we need to dump. If the data marker is within a
        // function, it is denoted as a word/short etc
        if (isArmElf(Obj) && std::get<2>(Symbols[si]) != ELF::STT_OBJECT &&
            !DisassembleAll) {
          uint64_t Stride = 0;

          auto DAI = std::lower_bound(DataMappingSymsAddr.begin(),
                                      DataMappingSymsAddr.end(), Index);
          if (DAI != DataMappingSymsAddr.end() && *DAI == Index) {
            // Switch to data.
            while (Index < End) {
              OutS << format("%8" PRIx64 ":", SectionAddr + Index);
              OutS << "\t";
              if (Index + 4 <= End) {
                Stride = 4;
                dumpBytes(Bytes.slice(Index, 4), OutS);
                OutS << "\t.word\t";
                uint32_t Data = 0;
                if (Obj->isLittleEndian()) {
                  const auto Word =
                      reinterpret_cast<const support::ulittle32_t *>(
                          Bytes.data() + Index);
                  Data = *Word;
                } else {
                  const auto Word = reinterpret_cast<const support::ubig32_t *>(
                      Bytes.data() + Index);
                  Data = *Word;
                }
                OutS << "0x" << format("%08" PRIx32, Data);
              } else if (Index + 2 <= End) {
                Stride = 2;
                dumpBytes(Bytes.slice(Index, 2), OutS);
                OutS << "\t\t.short\t";
                uint16_t Data = 0;
                if (Obj->isLittleEndian()) {
                  const auto Short =
                      reinterpret_cast<const support::ulittle16_t *>(
                          Bytes.data() + Index);
                  Data = *Short;
                } else {
                  const auto Short =
                      reinterpret_cast<const support::ubig16_t *>(Bytes.data() +
                                                                  Index);
                  Data = *Short;
                }
                OutS << "0x" << format("%04" PRIx16, Data);
              } else {
                Stride = 1;
                dumpBytes(Bytes.slice(Index, 1), OutS);
                OutS << "\t\t.byte\t";
                OutS << "0x" << format("%02" PRIx8, Bytes.slice(Index, 1)[0]);
              }
              Index += Stride;
              OutS << "\n";
              auto TAI = std::lower_bound(TextMappingSymsAddr.begin(),
                                          TextMappingSymsAddr.end(), Index);
              if (TAI != TextMappingSymsAddr.end() && *TAI == Index)
                break;
            }
          }
        }

        // If there is a data symbol inside an ELF text section and we are only
        // disassembling text (applicable all architectures),
        // we are in a situation where we must print the data and not
        // disassemble it.
        if (Obj->isELF() && std::get<2>(Symbols[si]) == ELF::STT_OBJECT &&
            !DisassembleAll && Section.isText()) {
          // print out data up to 8 bytes at a time in hex and ascii
          uint8_t AsciiData[9] = {'\0'};
          uint8_t Byte;
          int NumBytes = 0;

          for (Index = Start; Index < End; Index += 1) {
            if (((SectionAddr + Index) < StartAddress) ||
                ((SectionAddr + Index) > StopAddress))
              continue;
            if (NumBytes == 0) {
              OutS << format("%8" PRIx64 ":", SectionAddr + Index);
              OutS << "\t";
            }
            Byte = Bytes.slice(Index)[0];
            OutS << format(" %02x", Byte);
            AsciiData[NumBytes] = isprint(Byte) ? Byte : '.';

            uint8_t IndentOffset = 0;
            NumBytes++;
            if (Index == End - 1 || NumBytes > 8) {
              // Indent the space for less than 8 bytes data.
              // 2 spaces for byte and one for space between bytes
              IndentOffset = 3 * (8 - NumBytes);
              for (int Excess = 8 - NumBytes; Excess < 8; Excess++)
                AsciiData[Excess] = '\0';
              NumBytes = 8;
            }
            if (NumBytes == 8) {
              AsciiData[8] = '\0';
              OutS << std::string(IndentOffset, ' ') << "         ";
              OutS << reinterpret_cast<char *>(AsciiData);
              OutS << '\n';
              NumBytes = 0;
            }
          }
        }
        if (Index >= End)
          break;

        // Disassemble a real instruction or a data when disassemble all is
        // provided
        bool Disassembled = DisAsm->getInstruction(
            Inst, Size, Bytes.slice(Index), SectionAddr + Index, DebugOut,
            CommentStream);
        if (Size == 0)
          Size = 1;

        PIP.printInst(
            *IP, Disassembled ? &Inst : nullptr, Bytes.slice(Index, Size),
            {SectionAddr + Index, Section.getIndex()}, OutS, "", *STI, &SP);
        OutS << CommentStream.str();
        Comments.clear();

        // Try to resolve the target of a call, tail call, etc. to a specific
        // symbol.
        if (MIA && (MIA->isCall(Inst) || MIA->isUnconditionalBranch(Inst) ||
                    MIA->isConditionalBranch(Inst))) {
          uint64_t Target;
          if (MIA->evaluateBranch(Inst, SectionAddr + Index, Size, Target)) {
            // In a relocatable object, the target's section must reside in
            // the same section as the call instruction or it is accessed
            // through a relocation.
            //
            // In a non-relocatable object, the target may be in any section.
            //
            // N.B. We don't walk the relocations in the relocatable case yet.
            auto *TargetSectionSymbols = &Symbols;
            if (!Obj->isRelocatableObject()) {
              auto SectionAddress = std::upper_bound(
                  SectionAddresses.begin(), SectionAddresses.end(), Target,
                  [](uint64_t LHS, const std::pair<uint64_t, SectionRef> &RHS) {
                    return LHS < RHS.first;
                  });
              if (SectionAddress != SectionAddresses.begin()) {
                --SectionAddress;
                TargetSectionSymbols = &AllSymbols[SectionAddress->second];
              } else {
                TargetSectionSymbols = nullptr;
              }
            }

            // Find the first symbol in the section whose offset is less than
            // or equal to the target.
            if (TargetSectionSymbols) {
              auto TargetSym = std::upper_bound(
                  TargetSectionSymbols->begin(), TargetSectionSymbols->end(),
                  Target,
                  [](uint64_t LHS,
                     const std::tuple<uint64_t, StringRef, uint8_t> &RHS) {
                    return LHS < std::get<0>(RHS);
                  });
              if (TargetSym != TargetSectionSymbols->begin()) {
                --TargetSym;
                uint64_t TargetAddress = std::get<0>(*TargetSym);
                StringRef TargetName = std::get<1>(*TargetSym);
                OutS << " <" << TargetName;
                uint64_t Disp = Target - TargetAddress;
                if (Disp)
                  OutS << "+0x" << utohexstr(Disp);
                OutS << '>';
              }
            }
          }
        }
        OutS << "\n";

        // Print relocation for instruction.
        while (rel_cur != rel_end) {
          bool hidden = getHidden(*rel_cur);
          uint64_t addr = rel_cur->getOffset();
          SmallString<16> name;
          SmallString<32> val;

          // If this relocation is hidden, skip it.
          if (hidden || ((SectionAddr + addr) < StartAddress)) {
            ++rel_cur;
            continue;
          }

          // Stop when rel_cur's address is past the current instruction.
          if (addr >= Index + Size)
            break;
          rel_cur->getTypeName(name);
          error(getRelocationValueString(*rel_cur, val));
          OutS << format(Fmt.data(), SectionAddr + addr) << name << "\t" << val
               << "\n";
          ++rel_cur;
        }
      }
    }
  }
}

void llvm::DisassemHelper::PrintRelocations(const ObjectFile *Obj) {
  StringRef Fmt = Obj->getBytesInAddress() > 4 ? "%016" PRIx64 : "%08" PRIx64;
  // Regular objdump doesn't print relocations in non-relocatable object
  // files.
  if (!Obj->isRelocatableObject())
    return;

  for (const SectionRef &Section : ToolSectionFilter(*Obj)) {
    if (Section.relocation_begin() == Section.relocation_end())
      continue;
    StringRef secname = unwrapOrError(Section.getName(), Obj->getFileName());
    OutS << "RELOCATION RECORDS FOR [" << secname << "]:\n";
    for (const RelocationRef &Reloc : Section.relocations()) {
      bool hidden = getHidden(Reloc);
      uint64_t address = Reloc.getOffset();
      SmallString<32> relocname;
      SmallString<32> valuestr;
      if (address < StartAddress || address > StopAddress || hidden)
        continue;
      Reloc.getTypeName(relocname);
      error(getRelocationValueString(Reloc, valuestr));
      OutS << format(Fmt.data(), address) << " " << relocname << " " << valuestr
           << "\n";
    }
    OutS << "\n";
  }
}

void llvm::DisassemHelper::PrintSectionHeaders(const ObjectFile *Obj) {
  OutS << "Sections:\n"
          "Idx Name          Size      Address          Type\n";
  unsigned i = 0;
  for (const SectionRef &Section : ToolSectionFilter(*Obj)) {
    StringRef Name = unwrapOrError(Section.getName(), Obj->getFileName());
    uint64_t Address = Section.getAddress();
    uint64_t Size = Section.getSize();
    bool Text = Section.isText();
    bool Data = Section.isData();
    bool BSS = Section.isBSS();
    std::string Type = (std::string(Text ? "TEXT " : "") +
                        (Data ? "DATA " : "") + (BSS ? "BSS" : ""));
    OutS << format("%3d %-13s %08" PRIx64 " %016" PRIx64 " %s\n", i,
                   Name.str().c_str(), Size, Address, Type.c_str());
    ++i;
  }
}

void llvm::DisassemHelper::PrintSectionContents(const ObjectFile *Obj) {
  std::error_code EC;
  for (const SectionRef &Section : ToolSectionFilter(*Obj)) {
    StringRef Contents;
    StringRef Name = unwrapOrError(Section.getName(), Obj->getFileName());
    uint64_t BaseAddr = Section.getAddress();
    uint64_t Size = Section.getSize();
    if (!Size)
      continue;

    OutS << "Contents of section " << Name << ":\n";
    if (Section.isBSS()) {
      OutS << format("<skipping contents of bss section at [%04" PRIx64
                     ", %04" PRIx64 ")>\n",
                     BaseAddr, BaseAddr + Size);
      continue;
    }

    Expected<StringRef> ExpContents = Section.getContents();
    if (ExpContents)
      Contents = *ExpContents;
    else
      consumeError(ExpContents.takeError());

    // Dump out the content as hex and printable ascii characters.
    for (std::size_t addr = 0, end = Contents.size(); addr < end; addr += 16) {
      OutS << format(" %04" PRIx64 " ", BaseAddr + addr);
      // Dump line of hex.
      for (std::size_t i = 0; i < 16; ++i) {
        if (i != 0 && i % 4 == 0)
          OutS << ' ';
        if (addr + i < end)
          OutS << hexdigit((Contents[addr + i] >> 4) & 0xF, true)
               << hexdigit(Contents[addr + i] & 0xF, true);
        else
          OutS << "  ";
      }
      // Print ascii.
      OutS << "  ";
      for (std::size_t i = 0; i < 16 && addr + i < end; ++i) {
        if (std::isprint(static_cast<unsigned char>(Contents[addr + i]) & 0xFF))
          OutS << Contents[addr + i];
        else
          OutS << ".";
      }
      OutS << "\n";
    }
  }
}

void llvm::DisassemHelper::PrintSymbolTable(const ObjectFile *o,
                                            StringRef ArchiveName,
                                            StringRef ArchitectureName) {
  OutS << "SYMBOL TABLE:\n";

#ifdef NOT_LIBCOMGR
  if (const COFFObjectFile *coff = dyn_cast<const COFFObjectFile>(o)) {
    printCOFFSymbolTable(coff);
    return;
  }
#endif
  for (const SymbolRef &Symbol : o->symbols()) {
    Expected<uint64_t> AddressOrError = Symbol.getAddress();
    if (!AddressOrError)
      report_error(ArchiveName, o->getFileName(), AddressOrError.takeError(),
                   ArchitectureName);
    uint64_t Address = *AddressOrError;
    if ((Address < StartAddress) || (Address > StopAddress))
      continue;
    Expected<SymbolRef::Type> TypeOrError = Symbol.getType();
    if (!TypeOrError)
      report_error(ArchiveName, o->getFileName(), TypeOrError.takeError(),
                   ArchitectureName);
    SymbolRef::Type Type = *TypeOrError;
    uint32_t Flags = Symbol.getFlags();
    Expected<section_iterator> SectionOrErr = Symbol.getSection();
    if (!SectionOrErr)
      report_error(ArchiveName, o->getFileName(), SectionOrErr.takeError(),
                   ArchitectureName);
    section_iterator Section = *SectionOrErr;
    StringRef Name;
    if (Type == SymbolRef::ST_Debug && Section != o->section_end()) {
      Expected<StringRef> NameOrErr = Section->getName();
      if (!NameOrErr)
        consumeError(NameOrErr.takeError());
      else
        Name = *NameOrErr;
    } else {
      Expected<StringRef> NameOrErr = Symbol.getName();
      if (!NameOrErr)
        report_error(ArchiveName, o->getFileName(), NameOrErr.takeError(),
                     ArchitectureName);
      Name = *NameOrErr;
    }

    bool Global = Flags & SymbolRef::SF_Global;
    bool Weak = Flags & SymbolRef::SF_Weak;
    bool Absolute = Flags & SymbolRef::SF_Absolute;
    bool Common = Flags & SymbolRef::SF_Common;
    bool Hidden = Flags & SymbolRef::SF_Hidden;

    char GlobLoc = ' ';
    if (Type != SymbolRef::ST_Unknown)
      GlobLoc = Global ? 'g' : 'l';
    char Debug =
        (Type == SymbolRef::ST_Debug || Type == SymbolRef::ST_File) ? 'd' : ' ';
    char FileFunc = ' ';
    if (Type == SymbolRef::ST_File)
      FileFunc = 'f';
    else if (Type == SymbolRef::ST_Function)
      FileFunc = 'F';

    const char *Fmt = o->getBytesInAddress() > 4 ? "%016" PRIx64 : "%08" PRIx64;

    OutS << format(Fmt, Address) << " "
         << GlobLoc            // Local -> 'l', Global -> 'g', Neither -> ' '
         << (Weak ? 'w' : ' ') // Weak?
         << ' '                // Constructor. Not supported yet.
         << ' '                // Warning. Not supported yet.
         << ' '                // Indirect reference to another symbol.
         << Debug              // Debugging (d) or dynamic (D) symbol.
         << FileFunc           // Name of function (F), file (f) or object (O).
         << ' ';
    if (Absolute) {
      OutS << "*ABS*";
    } else if (Common) {
      OutS << "*COM*";
    } else if (Section == o->section_end()) {
      OutS << "*UND*";
    } else {
      if (const MachOObjectFile *MachO = dyn_cast<const MachOObjectFile>(o)) {
        DataRefImpl DR = Section->getRawDataRefImpl();
        StringRef SegmentName = MachO->getSectionFinalSegmentName(DR);
        OutS << SegmentName << ",";
      }
      StringRef SectionName = unwrapOrError(Section->getName(), o->getFileName());
      OutS << SectionName;
    }

    OutS << '\t';
    if (Common || isa<ELFObjectFileBase>(o)) {
      uint64_t Val =
          Common ? Symbol.getAlignment() : ELFSymbolRef(Symbol).getSize();
      OutS << format("\t %08" PRIx64 " ", Val);
    }

    if (Hidden) {
      OutS << ".hidden ";
    }
    OutS << Name << '\n';
  }
}

void llvm::DisassemHelper::PrintUnwindInfo(const ObjectFile *o) {
  OutS << "Unwind info:\n\n";

#ifdef NOT_LIBCOMGR
  if (const COFFObjectFile *coff = dyn_cast<COFFObjectFile>(o)) {
    printCOFFUnwindInfo(coff);
  } else if (const MachOObjectFile *MachO = dyn_cast<MachOObjectFile>(o))
    printMachOUnwindInfo(MachO);
  else {
#endif
    // TODO: Extract DWARF dump tool to objdump.
    ErrS << "This operation is only currently supported "
            "for COFF and MachO object files.\n";
    return;
#ifdef NOT_LIBCOMGR
  }
#endif
}

void llvm::DisassemHelper::printExportsTrie(const ObjectFile *o) {
  OutS << "Exports trie:\n";
#ifdef NOT_LIBCOMGR
  if (const MachOObjectFile *MachO = dyn_cast<MachOObjectFile>(o))
    printMachOExportsTrie(MachO);
  else {
#endif
    ErrS << "This operation is only currently supported "
            "for Mach-O executable files.\n";
    return;
#ifdef NOT_LIBCOMGR
  }
#endif
}

void llvm::DisassemHelper::printRebaseTable(ObjectFile *o) {
  OutS << "Rebase table:\n";
#ifdef NOT_LIBCOMGR
  if (MachOObjectFile *MachO = dyn_cast<MachOObjectFile>(o))
    printMachORebaseTable(MachO);
  else {
#endif
    ErrS << "This operation is only currently supported "
            "for Mach-O executable files.\n";
    return;
#ifdef NOT_LIBCOMGR
  }
#endif
}

void llvm::DisassemHelper::printBindTable(ObjectFile *o) {
  OutS << "Bind table:\n";
#ifdef NOT_LIBCOMGR
  if (MachOObjectFile *MachO = dyn_cast<MachOObjectFile>(o))
    printMachOBindTable(MachO);
  else {
#endif
    ErrS << "This operation is only currently supported "
            "for Mach-O executable files.\n";
    return;
#ifdef NOT_LIBCOMGR
  }
#endif
}

void llvm::DisassemHelper::printLazyBindTable(ObjectFile *o) {
  OutS << "Lazy bind table:\n";
#ifdef NOT_LIBCOMGR
  if (MachOObjectFile *MachO = dyn_cast<MachOObjectFile>(o))
    printMachOLazyBindTable(MachO);
  else {
#endif
    ErrS << "This operation is only currently supported "
            "for Mach-O executable files.\n";
    return;
#ifdef NOT_LIBCOMGR
  }
#endif
}

void llvm::DisassemHelper::printWeakBindTable(ObjectFile *o) {
  OutS << "Weak bind table:\n";
#ifdef NOT_LIBCOMGR
  if (MachOObjectFile *MachO = dyn_cast<MachOObjectFile>(o))
    printMachOWeakBindTable(MachO);
  else {
#endif
    ErrS << "This operation is only currently supported "
            "for Mach-O executable files.\n";
    return;
#ifdef NOT_LIBCOMGR
  }
#endif
}

/// Dump the raw contents of the __clangast section so the output can be piped
/// into llvm-bcanalyzer.
void llvm::DisassemHelper::printRawClangAST(const ObjectFile *Obj) {
  if (OutS.is_displayed()) {
    ErrS << "The -raw-clang-ast option will dump the raw binary contents of "
            "the clang ast section.\n"
            "Please redirect the output to a file or another program such as "
            "llvm-bcanalyzer.\n";
    return;
  }

  StringRef ClangASTSectionName("__clangast");
  if (isa<COFFObjectFile>(Obj)) {
    ClangASTSectionName = "clangast";
  }

  Optional<object::SectionRef> ClangASTSection;
  for (auto Sec : ToolSectionFilter(*Obj)) {
    StringRef Name;
    auto NameOrErr = Sec.getName();
    if (!NameOrErr) // FIXME: Need better error handling.
      consumeError(NameOrErr.takeError());
    else
      Name = *NameOrErr;
    if (Name == ClangASTSectionName) {
      ClangASTSection = Sec;
      break;
    }
  }
  if (!ClangASTSection)
    return;

  StringRef ClangASTContents;
  Expected<StringRef> ExpClangASTContents =
      ClangASTSection.getValue().getContents();
  if (ExpClangASTContents)
    ClangASTContents = *ExpClangASTContents;
  else
    consumeError(ExpClangASTContents.takeError());

  OutS.write(ClangASTContents.data(), ClangASTContents.size());
}

void llvm::DisassemHelper::printFaultMaps(const ObjectFile *Obj) {
  const char *FaultMapSectionName = nullptr;

  if (isa<ELFObjectFileBase>(Obj)) {
    FaultMapSectionName = ".llvm_faultmaps";
  } else if (isa<MachOObjectFile>(Obj)) {
    FaultMapSectionName = "__llvm_faultmaps";
  } else {
    ErrS << "This operation is only currently supported "
            "for ELF and Mach-O executable files.\n";
    return;
  }

  Optional<object::SectionRef> FaultMapSection;

  for (auto Sec : ToolSectionFilter(*Obj)) {
    StringRef Name;
    auto NameOrErr = Sec.getName();
    if (!NameOrErr) // FIXME: Need better error handling.
      consumeError(NameOrErr.takeError());
    else
      Name = *NameOrErr;
    if (Name == FaultMapSectionName) {
      FaultMapSection = Sec;
      break;
    }
  }

  OutS << "FaultMap table:\n";

  if (!FaultMapSection.hasValue()) {
    OutS << "<not found>\n";
    return;
  }

  StringRef FaultMapContents;
  Expected<StringRef> ExpFaultMapContents =
      FaultMapSection.getValue().getContents();
  if (ExpFaultMapContents)
    FaultMapContents = *ExpFaultMapContents;
  else
    consumeError(ExpFaultMapContents.takeError());

  FaultMapParser FMP(FaultMapContents.bytes_begin(),
                     FaultMapContents.bytes_end());

  OutS << FMP;
}

void llvm::DisassemHelper::printPrivateFileHeaders(const ObjectFile *o,
                                                   bool onlyFirst) {
  if (o->isELF())
    return printELFFileHeader(o);
#ifdef NOT_LIBCOMGR
  if (o->isCOFF())
    return printCOFFFileHeader(o);
  if (o->isWasm())
    return printWasmFileHeader(o);
  if (o->isMachO()) {
    printMachOFileHeader(o);
    if (!onlyFirst)
      printMachOLoadCommands(o);
    return;
  }
#endif
  report_error(o->getFileName(), "Invalid/Unsupported object file format");
}

void llvm::DisassemHelper::DumpObject(ObjectFile *o,
                                      const Archive *a = nullptr) {
  StringRef ArchiveName = a != nullptr ? a->getFileName() : "";
  // Avoid other output when using a raw option.
  if (!RawClangAST) {
    OutS << '\n';
    if (a)
      OutS << a->getFileName() << "(" << o->getFileName() << ")";
    else
      OutS << o->getFileName();
    OutS << ":\tfile format " << o->getFileFormatName() << "\n\n";
  }

  if (Disassemble)
    DisassembleObject(o, Relocations);
  if (Relocations && !Disassemble)
    PrintRelocations(o);
  if (SectionHeaders)
    PrintSectionHeaders(o);
  if (SectionContents)
    PrintSectionContents(o);
  if (SymbolTable)
    PrintSymbolTable(o, ArchiveName);
  if (UnwindInfo)
    PrintUnwindInfo(o);
  if (PrivateHeaders || FirstPrivateHeader)
    printPrivateFileHeaders(o, FirstPrivateHeader);
  if (ExportsTrie)
    printExportsTrie(o);
  if (Rebase)
    printRebaseTable(o);
  if (Bind)
    printBindTable(o);
  if (LazyBind)
    printLazyBindTable(o);
  if (WeakBind)
    printWeakBindTable(o);
  if (RawClangAST)
    printRawClangAST(o);
  if (PrintFaultMaps)
    printFaultMaps(o);
  if (DwarfDumpType != DIDT_Null) {
    std::unique_ptr<DIContext> DICtx = DWARFContext::create(*o);
    // Dump the complete DWARF structure.
    DIDumpOptions DumpOpts;
    DumpOpts.DumpType = DwarfDumpType;
    DICtx->dump(OutS, DumpOpts);
  }
}

#ifdef NOT_LIBCOMGR
void llvm::DisassemHelper::DumpObject(const COFFImportFile *I,
                                      const Archive *A) {
  StringRef ArchiveName = A ? A->getFileName() : "";

  // Avoid other output when using a raw option.
  if (!RawClangAST)
    OutS << '\n'
         << ArchiveName << "(" << I->getFileName() << ")"
         << ":\tfile format COFF-import-file"
         << "\n\n";

  if (SymbolTable)
    printCOFFSymbolTable(I);
}
#endif

/// @brief Dump each object file in \a a;
void llvm::DisassemHelper::DumpArchive(const Archive *a) {
  Error Err = Error::success();
  for (auto &C : a->children(Err)) {
    Expected<std::unique_ptr<Binary>> ChildOrErr = C.getAsBinary();
    if (!ChildOrErr) {
      if (auto E = isNotObjectErrorInvalidFileType(ChildOrErr.takeError()))
        report_error(a->getFileName(), C, std::move(E));
      continue;
    }
    if (ObjectFile *o = dyn_cast<ObjectFile>(&*ChildOrErr.get()))
      DumpObject(o, a);
#ifdef NOT_LIBCOMGR
    else if (COFFImportFile *I = dyn_cast<COFFImportFile>(&*ChildOrErr.get()))
      DumpObject(I, a);
#endif
    else
      report_error(a->getFileName(), object_error::invalid_file_type);
  }
  if (Err)
    report_error(a->getFileName(), std::move(Err));
}

/// @brief Open file and figure out how to dump it.
void llvm::DisassemHelper::DumpInput(StringRef file) {

  // Attempt to open the binary.
  Expected<OwningBinary<Binary>> BinaryOrErr = createBinary(file);
  if (!BinaryOrErr)
    report_error(file, BinaryOrErr.takeError());
  Binary &Binary = *BinaryOrErr.get().getBinary();

  if (Archive *a = dyn_cast<Archive>(&Binary))
    DumpArchive(a);
  else if (ObjectFile *o = dyn_cast<ObjectFile>(&Binary))
    DumpObject(o);
  else
    report_error(file, object_error::invalid_file_type);
}

// -----------------------------------------------------------------------------------
// For libcomgr.so:
//
// Modified to exclude main function, but with DisassemHelper class
// with member functions DisassembleAction (llvm-objdump)
//
// The disassembled output is stored in the "result_buffer". Each disassemble
// action uses
// a new DisassemHelper object. The result_buffer will be released in the
// destructor of
// DisassemHelper.
// ------------------------------------------------------------------------------------
amd_comgr_status_t
llvm::DisassemHelper::disassembleAction(StringRef Input,
                                        ArrayRef<std::string> Options) {
  // Register the target printer for --version.
  cl::AddExtraVersionPrinter(TargetRegistry::printRegisteredTargetsForVersion);

  SmallVector<const char *, 20> ArgV;
  ArgV.push_back(nullptr);
  for (auto &Option : Options)
    ArgV.push_back(Option.c_str());
  size_t ArgC = ArgV.size();
  ArgV.push_back(nullptr);
  COMGR::clearLLVMOptions();
  cl::ParseCommandLineOptions(ArgC, ArgV.data(), "llvm object file dumper\n",
                              &ErrS);
  MCPU = lld::getCPUStr();

  ErrorOr<std::unique_ptr<MemoryBuffer>> BufOrErr =
      MemoryBuffer::getMemBuffer(Input);
  if (std::error_code EC = BufOrErr.getError()) {
    ErrS << "DisassembleAction : forming Buffer.\n";
    ErrS << "DisassembleAction : error reading file: " << EC.message() << ".\n";
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }
  std::unique_ptr<MemoryBuffer> &Buffer = BufOrErr.get();

  Expected<std::unique_ptr<Binary>> BinOrErr =
      createBinary(Buffer->getMemBufferRef());
  if (!BinOrErr) {
    ErrS << "DisassembleAction : forming Bin.\n";
    return AMD_COMGR_STATUS_ERROR;
  }
  std::unique_ptr<Binary> &Bin = BinOrErr.get();

  Expected<OwningBinary<Binary>> BinaryOrErr =
      OwningBinary<Binary>(std::move(Bin), std::move(Buffer));
  if (!BinaryOrErr) {
    ErrS << "DisassembleAction : forming Binary.\n";
    return AMD_COMGR_STATUS_ERROR;
  }
  Binary &Binary = *BinaryOrErr.get().getBinary();

  if (Archive *a = dyn_cast<Archive>(&Binary))
    DumpArchive(a);
  else if (ObjectFile *o = dyn_cast<ObjectFile>(&Binary))
    DumpObject(o);
  else
    report_error("comgr-objdump.cpp", object_error::invalid_file_type);

  OutS.flush();

  return AMD_COMGR_STATUS_SUCCESS;
}
