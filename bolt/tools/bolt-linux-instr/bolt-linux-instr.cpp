
//===------- bolt/tools/bolt-linux-instr/bolt-linux-instr.cpp -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/Binary.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

#include <stack>

using namespace llvm;
using namespace object;

namespace {

cl::OptionCategory
    LinuxInstrDataCat("Linux kernel instrumentation data options");

cl::SubCommand DumpSubCommand("dump", "Dump Linux kernel instrumentation data");

cl::SubCommand DiffSubCommand("diff", "Diff two dumps");

cl::opt<std::string> VmlinuxFilename("v", cl::desc("The vmlinux filename"),
                                     cl::value_desc("filename"), cl::Required,
                                     cl::sub(DumpSubCommand),
                                     cl::sub(DiffSubCommand),
                                     cl::cat(LinuxInstrDataCat));

cl::opt<std::string> OutputFilename("o",
                                    cl::desc("The output .fdata/.dat filename"),
                                    cl::value_desc("filename"), cl::Required,
                                    cl::sub(DumpSubCommand),
                                    cl::sub(DiffSubCommand),
                                    cl::cat(LinuxInstrDataCat));

cl::opt<std::string> Dat1Filename(cl::Positional,
                                  cl::desc("<1st .dat filename>"), cl::Required,
                                  cl::sub(DiffSubCommand),
                                  cl::cat(LinuxInstrDataCat));

cl::opt<std::string> Dat2Filename(cl::Positional,
                                  cl::desc("<2nd .dat filename>"), cl::Optional,
                                  cl::sub(DiffSubCommand),
                                  cl::cat(LinuxInstrDataCat));

class ELFCore {
public:
  ELFCore(const std::string Filename) : Filename(Filename) {}

  Error init() {
    ErrorOr<std::unique_ptr<MemoryBuffer>> MBOrErr =
        MemoryBuffer::getFileSlice(Filename, 1024 * 1024, 0, true);
    if (std::error_code EC = MBOrErr.getError())
      return createStringError(EC.message());
    HeaderMB = std::move(*MBOrErr);

    Expected<ELF64LEFile> EFOrErr =
        ELFFile<ELF64LE>::create(HeaderMB->getBuffer());
    if (Error E = EFOrErr.takeError())
      return E;
    EF = std::make_unique<ELF64LEFile>(std::move(*EFOrErr));
    return Error::success();
  }

  template <typename T> Expected<T> read(uint64_t Addr) const {
    Expected<std::unique_ptr<MemoryBuffer>> MBOrErr = read(Addr, sizeof(T));
    if (Error E = MBOrErr.takeError())
      return E;
    return *reinterpret_cast<const T *>((*MBOrErr)->getBuffer().data());
  }

  Expected<std::unique_ptr<MemoryBuffer>> read(uint64_t Addr,
                                               uint64_t Size) const {
    auto ProgramHeaders = EF->program_headers();
    if (Error E = ProgramHeaders.takeError())
      return E;

    for (auto PH : *ProgramHeaders) {
      if (PH.p_memsz != PH.p_filesz)
        continue;

      if (PH.p_vaddr <= Addr && Addr + Size <= PH.p_vaddr + PH.p_memsz) {
        const uint64_t Offset = PH.p_offset + (Addr - PH.p_vaddr);

        ErrorOr<std::unique_ptr<MemoryBuffer>> MBOrErr =
            MemoryBuffer::getFileSlice(Filename, Size, Offset, true);
        if (std::error_code EC = MBOrErr.getError())
          return createStringError(EC.message());
        return std::move(*MBOrErr);
      }
    }
    return createStringError("invalid range");
  }

  StringRef getFilename() const { return Filename; }

private:
  std::unique_ptr<MemoryBuffer> HeaderMB;
  std::unique_ptr<ELF64LEFile> EF;

  const std::string Filename;
};

class ELFObj {
public:
  ELFObj(const std::string &Filename) : Filename(Filename) {}

  Error init() {
    Expected<OwningBinary<Binary>> OwnBinOrErr = createBinary(Filename);
    if (Error E = OwnBinOrErr.takeError())
      return E;
    OwnBin = std::make_unique<OwningBinary<Binary>>(std::move(*OwnBinOrErr));

    EF = dyn_cast<ELFObjectFile<ELF64LE>>(OwnBin->getBinary());
    if (!EF)
      return createStringError("not an ELF64LE object file");

    for (const ELFSymbolRef &Sym : EF->symbols()) {
      Expected<StringRef> NameOrErr = Sym.getName();
      if (!NameOrErr)
        continue;
      StringRef Name = NameOrErr.get();

      Expected<uint64_t> ValueOrErr = Sym.getValue();
      if (!ValueOrErr)
        continue;
      uint64_t Value = ValueOrErr.get();

      SymbolValues[Name] = Value;
    }

    return Error::success();
  }

  Expected<uint64_t> getSymbolValue(StringRef Name) const {
    if (!SymbolValues.contains(Name))
      return createStringError("unknown symbol");
    return SymbolValues.at(Name);
  }

  Expected<SectionRef> getSection(StringRef Name) const {
    for (auto Section : EF->sections()) {
      Expected<StringRef> NameOrErr = Section.getName();
      if (NameOrErr && *NameOrErr == Name)
        return Section;
    }
    return createStringError("unknown section");
  }

  Expected<StringRef> getSectionContents(StringRef Name) const {
    Expected<SectionRef> SectionOrErr = getSection(Name);
    if (Error E = SectionOrErr.takeError())
      return E;
    return SectionOrErr->getContents();
  }

  StringRef getFilename() const { return Filename; }

private:
  StringMap<uint64_t> SymbolValues;

  ELFObjectFile<ELF64LE> *EF;
  std::unique_ptr<OwningBinary<Binary>> OwnBin;

  std::string Filename;
};

raw_fd_ostream &operator<<(raw_fd_ostream &OS, std::error_code EC) {
  OS << EC.message();
  return OS;
}

template <typename T> void report_error(const T &Msg) {
  errs() << Msg << "\n";
  exit(EXIT_FAILURE);
}

template <typename T, typename... Args>
void report_error(const T &Msg, const Args &...Others) {
  errs() << Msg << " : ";
  report_error(Others...);
}

std::unique_ptr<MemoryBuffer> readELFCore(const ELFCore &EC, uint64_t Addr,
                                          uint64_t Size) {
  Expected<std::unique_ptr<MemoryBuffer>> MBOrErr = EC.read(Addr, Size);
  if (Error E = MBOrErr.takeError())
    report_error(formatv("{0}:{1:x}:{2:x}", EC.getFilename(), Addr, Size),
                 std::move(E));
  return std::move(*MBOrErr);
}

template <typename T> T readELFCore(const ELFCore &EC, uint64_t Addr) {
  std::unique_ptr<MemoryBuffer> MB = readELFCore(EC, Addr, sizeof(T));
  return *reinterpret_cast<const T *>(MB->getBuffer().data());
}

uint64_t getSymbolValue(const ELFObj &EO, StringRef Name) {
  Expected<uint64_t> ValueOrErr = EO.getSymbolValue(Name);
  if (Error E = ValueOrErr.takeError())
    report_error(Name, std::move(E));
  return *ValueOrErr;
}

int dumpMode() {
  ELFObj Vmlinux(VmlinuxFilename);
  if (Error E = Vmlinux.init())
    report_error(VmlinuxFilename, std::move(E));

  ELFCore PK("/proc/kcore");
  if (Error E = PK.init())
    report_error(PK.getFilename(), std::move(E));

  // sanity check
  {
    StringRef ToCheck = "Linux version ";
    uint64_t LinuxBannerAddr = getSymbolValue(Vmlinux, "linux_banner");
    std::unique_ptr<MemoryBuffer> MB =
        readELFCore(PK, LinuxBannerAddr, ToCheck.size());
    if (MB->getBuffer() != ToCheck)
      report_error(formatv("'{0}' is not found at {1}:{2:x}", ToCheck,
                           PK.getFilename(), LinuxBannerAddr));
  }

  uint64_t BoltInstrLocationsAddr =
      getSymbolValue(Vmlinux, "__bolt_instr_locations");
  uint64_t BoltNumCounters =
      readELFCore<uint64_t>(PK, getSymbolValue(Vmlinux, "__bolt_num_counters"));

  outs() << formatv(
      "INFO: __bolt_instr_locations={0:x}, __bolt_num_counters={1:x}\n",
      BoltInstrLocationsAddr, BoltNumCounters);

  std::unique_ptr<MemoryBuffer> MB =
      readELFCore(PK, BoltInstrLocationsAddr, BoltNumCounters * 8);

  std::error_code EC;
  raw_fd_ostream OutoutFile(OutputFilename, EC, sys::fs::OpenFlags::OF_None);
  if (EC)
    report_error(OutputFilename, EC);

  OutoutFile.write(MB->getBufferStart(), MB->getBufferSize());
  return EXIT_SUCCESS;
}

std::unique_ptr<MemoryBuffer> readFile(StringRef Filename) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> MBOrErr =
      MemoryBuffer::getFile(Filename,
                            /* IsText */ false,
                            /* RequiresNullTerminator */ false,
                            /* IsVolatile */ false, Align(8));
  if (std::error_code EC = MBOrErr.getError())
    report_error(Filename, EC);
  return std::move(*MBOrErr);
}

template <typename T>
std::unique_ptr<std::vector<T>> readFileAsVector(StringRef Filename) {
  std::unique_ptr<MemoryBuffer> MB = readFile(Filename);
  uint64_t Size = MB->getBufferSize();
  if (!Size || Size % sizeof(T))
    report_error(formatv("{0} : unexpected size", Filename));

  return std::move(std::make_unique<std::vector<T>>(
      reinterpret_cast<const T *>(MB->getBufferStart()),
      reinterpret_cast<const T *>(MB->getBufferEnd())));
}

StringRef readSectionContents(const ELFObj &EO, StringRef Name) {
  Expected<StringRef> Contents = EO.getSectionContents(Name);
  if (Error E = Contents.takeError())
    report_error(EO.getFilename(), Name, std::move(E));
  return *Contents;
}

struct Location {
  uint32_t FunctionName;
  uint32_t Offset;
};

struct CallDescription {
  Location From;
  uint32_t FromNode;
  Location To;
  uint32_t Counter;
  uint64_t TargetAddress;
};

using IndCallDescription = Location;

struct IndCallTargetDescription {
  Location Loc;
  uint64_t Address;
};

struct EdgeDescription {
  Location From;
  uint32_t FromNode;
  Location To;
  uint32_t ToNode;
  uint32_t Counter;
};

struct InstrumentedNode {
  uint32_t Node;
  uint32_t Counter;
};

struct EntryNode {
  uint64_t Node;
  uint64_t Address;
};

struct FunctionDescription {
  uint32_t NumLeafNodes;
  const InstrumentedNode *LeafNodes;
  uint32_t NumEdges;
  const EdgeDescription *Edges;
  uint32_t NumCalls;
  const CallDescription *Calls;
  uint32_t NumEntryNodes;
  const EntryNode *EntryNodes;

  /// Constructor will parse the serialized function metadata written by BOLT
  FunctionDescription(const uint8_t *FuncDescData);

  uint64_t getSize() const {
    return 16 + NumLeafNodes * sizeof(InstrumentedNode) +
           NumEdges * sizeof(EdgeDescription) +
           NumCalls * sizeof(CallDescription) +
           NumEntryNodes * sizeof(EntryNode);
  }
};

FunctionDescription::FunctionDescription(const uint8_t *FuncDescData) {
  const uint8_t *Ptr = FuncDescData;
  NumLeafNodes = *reinterpret_cast<const uint32_t *>(Ptr);
  LeafNodes = reinterpret_cast<const InstrumentedNode *>(Ptr + 4);
  Ptr += 4 + NumLeafNodes * sizeof(InstrumentedNode);

  NumEdges = *reinterpret_cast<const uint32_t *>(Ptr);
  Edges = reinterpret_cast<const EdgeDescription *>(Ptr + 4);
  Ptr += 4 + NumEdges * sizeof(EdgeDescription);

  NumCalls = *reinterpret_cast<const uint32_t *>(Ptr);
  Calls = reinterpret_cast<const CallDescription *>(Ptr + 4);
  Ptr += 4 + NumCalls * sizeof(CallDescription);

  NumEntryNodes = *reinterpret_cast<const uint32_t *>(Ptr);
  EntryNodes = reinterpret_cast<const EntryNode *>(Ptr + 4);
}

struct CallFlowEntry {
  uint64_t Val{0};
  uint64_t Calls{0};
};

struct ProfileWriterContext {
  std::unique_ptr<std::vector<uint64_t>> Dat;

  const uint8_t *FuncDescData{nullptr};
  const char *Strings{nullptr};
};

struct Edge {
  uint32_t Node; // Index in nodes array regarding the destination of this edge
  uint32_t ID;   // Edge index in an array comprising all edges of the graph
};

struct Node {
  uint32_t NumInEdges{0};     // Input edge count used to size InEdge
  uint32_t NumOutEdges{0};    // Output edge count used to size OutEdges
  std::vector<Edge> InEdges;  // Created and managed by \p Graph
  std::vector<Edge> OutEdges; // ditto
};

struct Graph {
  uint32_t NumNodes;
  std::vector<Node> CFGNodes;
  std::vector<Node> SpanningTreeNodes;
  std::vector<uint64_t> EdgeFreqs;
  std::vector<uint64_t> CallFreqs;
  const FunctionDescription &FD;

  Graph(const FunctionDescription &FD, const uint64_t *Counters,
        ProfileWriterContext &Ctx);

private:
  void computeEdgeFrequencies(const uint64_t *Counters,
                              ProfileWriterContext &Ctx);
};

Graph::Graph(const FunctionDescription &FD, const uint64_t *Counters,
             ProfileWriterContext &Ctx)
    : FD(FD) {

  // First pass to determine number of nodes
  int32_t MaxNodes = -1;
  for (uint32_t I = 0; I < FD.NumEdges; ++I)
    MaxNodes = std::max({static_cast<int32_t>(FD.Edges[I].FromNode),
                         static_cast<int32_t>(FD.Edges[I].ToNode), MaxNodes});

  for (uint32_t I = 0; I < FD.NumLeafNodes; ++I)
    MaxNodes = std::max({static_cast<int32_t>(FD.LeafNodes[I].Node), MaxNodes});

  for (uint32_t I = 0; I < FD.NumCalls; ++I)
    MaxNodes = std::max({static_cast<int32_t>(FD.Calls[I].FromNode), MaxNodes});

  // No nodes? Nothing to do
  if (MaxNodes < 0) {
    NumNodes = 0;
    return;
  }
  ++MaxNodes;
  NumNodes = static_cast<uint32_t>(MaxNodes);

  // Initial allocations
  CFGNodes = std::vector<Node>(MaxNodes);
  SpanningTreeNodes = std::vector<Node>(MaxNodes);

  // Figure out how much to allocate to each vector (in/out edge sets)
  for (uint32_t I = 0; I < FD.NumEdges; ++I) {
    const uint32_t Src = FD.Edges[I].FromNode;
    const uint32_t Dst = FD.Edges[I].ToNode;

    CFGNodes[Src].NumOutEdges++;
    CFGNodes[Dst].NumInEdges++;

    if (FD.Edges[I].Counter == 0xffffffff) {
      SpanningTreeNodes[Src].NumOutEdges++;
      SpanningTreeNodes[Dst].NumInEdges++;
    }
  }

  // Allocate in/out edge sets
  for (int I = 0; I < MaxNodes; ++I) {
    CFGNodes[I].InEdges = std::vector<Edge>(CFGNodes[I].NumInEdges);
    CFGNodes[I].OutEdges = std::vector<Edge>(CFGNodes[I].NumOutEdges);
    SpanningTreeNodes[I].InEdges =
        std::vector<Edge>(SpanningTreeNodes[I].NumInEdges);
    SpanningTreeNodes[I].OutEdges =
        std::vector<Edge>(SpanningTreeNodes[I].NumOutEdges);
    CFGNodes[I].NumInEdges = 0;
    CFGNodes[I].NumOutEdges = 0;
    SpanningTreeNodes[I].NumInEdges = 0;
    SpanningTreeNodes[I].NumOutEdges = 0;
  }

  // Fill in/out edge sets
  for (uint32_t I = 0; I < FD.NumEdges; ++I) {
    const uint32_t Src = FD.Edges[I].FromNode;
    const uint32_t Dst = FD.Edges[I].ToNode;
    Edge *E = &CFGNodes[Src].OutEdges[CFGNodes[Src].NumOutEdges++];
    E->Node = Dst;
    E->ID = I;

    E = &CFGNodes[Dst].InEdges[CFGNodes[Dst].NumInEdges++];
    E->Node = Src;
    E->ID = I;

    if (FD.Edges[I].Counter == 0xffffffff) {
      E = &SpanningTreeNodes[Src]
               .OutEdges[SpanningTreeNodes[Src].NumOutEdges++];
      E->Node = Dst;
      E->ID = I;

      E = &SpanningTreeNodes[Dst].InEdges[SpanningTreeNodes[Dst].NumInEdges++];
      E->Node = Src;
      E->ID = I;
    }
  }

  computeEdgeFrequencies(Counters, Ctx);
}

/// Auxiliary map structure for fast lookups of which calls map to each node of
/// the function CFG
struct NodeToCallsMap {
  NodeToCallsMap(const FunctionDescription &FD, uint32_t NumNodes)
      : Entries(NumNodes) {
    for (uint32_t I = 0; I < FD.NumCalls; ++I)
      ++Entries[FD.Calls[I].FromNode].NumCalls;

    for (uint32_t I = 0; I < Entries.size(); ++I) {
      Entries[I].Calls = std::vector<uint32_t>(Entries[I].NumCalls);
      Entries[I].NumCalls = 0;
    }

    for (uint32_t I = 0; I < FD.NumCalls; ++I) {
      MapEntry &Entry = Entries[FD.Calls[I].FromNode];
      Entry.Calls[Entry.NumCalls++] = I;
    }
  }

  /// Set the frequency of all calls in node \p NodeID to Freq. However, if
  /// the calls have their own counters and do not depend on the basic block
  /// counter, this means they have landing pads and throw exceptions. In this
  /// case, set their frequency with their counters and return the maximum
  /// value observed in such counters. This will be used as the new frequency
  /// at basic block entry. This is used to fix the CFG edge frequencies in the
  /// presence of exceptions.
  uint64_t visitAllCallsIn(uint32_t NodeID, uint64_t Freq,
                           std::vector<uint64_t> &CallFreqs,
                           const FunctionDescription &FD,
                           const uint64_t *Counters,
                           ProfileWriterContext &Ctx) const {
    const MapEntry &Entry = Entries[NodeID];
    uint64_t MaxValue = 0;
    for (int I = 0, E = Entry.NumCalls; I != E; ++I) {
      const uint32_t CallID = Entry.Calls[I];
      const CallDescription &CallDesc = FD.Calls[CallID];
      if (CallDesc.Counter == 0xffffffff) {
        CallFreqs[CallID] = Freq;
      } else {
        const uint64_t CounterVal = Counters[CallDesc.Counter];
        CallFreqs[CallID] = CounterVal;
        if (CounterVal > MaxValue)
          MaxValue = CounterVal;
      }
    }
    return MaxValue;
  }

  struct MapEntry {
    uint32_t NumCalls{0};
    std::vector<uint32_t> Calls;
  };
  std::vector<MapEntry> Entries;
};

void Graph::computeEdgeFrequencies(const uint64_t *Counters,
                                   ProfileWriterContext &Ctx) {
  if (NumNodes == 0)
    return;

  EdgeFreqs = std::vector<uint64_t>(FD.NumEdges);
  CallFreqs = std::vector<uint64_t>(FD.NumCalls);

  // Setup a lookup for calls present in each node (BB)
  NodeToCallsMap CallMap(FD, NumNodes);

  // Perform a bottom-up, BFS traversal of the spanning tree in G. Edges in the
  // spanning tree don't have explicit counters. We must infer their value using
  // a linear combination of other counters (sum of counters of the outgoing
  // edges minus sum of counters of the incoming edges).
  std::stack<uint32_t> Stack;
  enum Status : uint8_t { S_NEW = 0, S_VISITING, S_VISITED };
  std::vector<Status> Visited(NumNodes);
  std::vector<uint64_t> LeafFrequency(NumNodes);
  std::vector<uint64_t> EntryAddress(NumNodes);

  // Setup a fast lookup for frequency of leaf nodes, which have special
  // basic block frequency instrumentation (they are not edge profiled).
  for (uint32_t I = 0; I < FD.NumLeafNodes; ++I)
    LeafFrequency[FD.LeafNodes[I].Node] = Counters[FD.LeafNodes[I].Counter];

  for (uint32_t I = 0; I < FD.NumEntryNodes; ++I)
    EntryAddress[FD.EntryNodes[I].Node] = FD.EntryNodes[I].Address;

  // Add all root nodes to the stack
  for (uint32_t I = 0; I < NumNodes; ++I)
    if (SpanningTreeNodes[I].NumInEdges == 0)
      Stack.push(I);

  if (Stack.empty())
    return;

  // Add all known edge counts, will infer the rest
  for (uint32_t I = 0; I < FD.NumEdges; ++I) {
    const uint32_t C = FD.Edges[I].Counter;
    if (C == 0xffffffff) // inferred counter - we will compute its value
      continue;
    EdgeFreqs[I] = Counters[C];
  }

  while (!Stack.empty()) {
    const uint32_t Cur = Stack.top();
    Stack.pop();

    // This shouldn't happen in a tree
    assert(Visited[Cur] != S_VISITED &&
           "should not have visited nodes in stack");

    if (Visited[Cur] == S_NEW) {
      Visited[Cur] = S_VISITING;
      Stack.push(Cur);
      for (int I = 0, E = SpanningTreeNodes[Cur].NumOutEdges; I < E; ++I) {
        const uint32_t Succ = SpanningTreeNodes[Cur].OutEdges[I].Node;
        Stack.push(Succ);
      }
      continue;
    }

    Visited[Cur] = S_VISITED;

    // Establish our node frequency based on outgoing edges, which should all be
    // resolved by now.
    uint64_t CurNodeFreq = LeafFrequency[Cur];
    // Not a leaf?
    if (!CurNodeFreq) {
      for (int I = 0, E = CFGNodes[Cur].NumOutEdges; I != E; ++I) {
        const uint32_t SuccEdge = CFGNodes[Cur].OutEdges[I].ID;
        CurNodeFreq += EdgeFreqs[SuccEdge];
      }
    }

    const uint64_t CallFreq =
        CallMap.visitAllCallsIn(Cur, CurNodeFreq, CallFreqs, FD, Counters, Ctx);
    if (CallFreq > CurNodeFreq)
      CurNodeFreq = CallFreq;

    // No parent? Reached a tree root, limit to call frequency updating.
    if (SpanningTreeNodes[Cur].NumInEdges == 0)
      continue;

    assert(SpanningTreeNodes[Cur].NumInEdges == 1 && "must have 1 parent");
    const uint32_t ParentEdge = SpanningTreeNodes[Cur].InEdges[0].ID;

    // Calculate parent edge freq.
    int64_t ParentEdgeFreq = CurNodeFreq;
    for (int I = 0, E = CFGNodes[Cur].NumInEdges; I != E; ++I) {
      const uint32_t PredEdge = CFGNodes[Cur].InEdges[I].ID;
      ParentEdgeFreq -= EdgeFreqs[PredEdge];
    }

    // Sometimes the conservative CFG that BOLT builds will lead to incorrect
    // flow computation. For example, in a BB that transitively calls the exit
    // syscall, BOLT will add a fall-through successor even though it should not
    // have any successors. So this block execution will likely be wrong. We
    // tolerate this imperfection since this case should be quite infrequent.
    if (ParentEdgeFreq < 0)
      ParentEdgeFreq = 0;

    EdgeFreqs[ParentEdge] = ParentEdgeFreq;
  }
}

void readDescriptions(const ELFObj &Vmlinux, ProfileWriterContext &Ctx) {
  StringRef BoltNote = readSectionContents(Vmlinux, ".bolt.instr.tables");

  const uint8_t *Ptr = BoltNote.bytes_begin() + 20;
  uint32_t IndCallDescSize = *reinterpret_cast<const uint32_t *>(Ptr);
  Ptr += 4 + IndCallDescSize;
  uint32_t IndCallTargetDescSize = *reinterpret_cast<const uint32_t *>(Ptr);
  Ptr += 4 + IndCallTargetDescSize;
  uint32_t FuncDescSize = *reinterpret_cast<const uint32_t *>(Ptr);
  Ctx.FuncDescData = Ptr + 4;
  Ctx.Strings = reinterpret_cast<const char *>(Ptr + 4 + FuncDescSize);
}

/// Output Location to the fdata file
void serializeLoc(raw_fd_ostream &OS, const ProfileWriterContext &Ctx,
                  const Location Loc) {
  // fdata location format: Type Name Offset
  // Type 1 - regular symbol
  OS << "1 " << Ctx.Strings + Loc.FunctionName << " "
     << Twine::utohexstr(Loc.Offset) << " ";
}

const uint8_t *writeFunctionProfile(raw_fd_ostream &OS,
                                    ProfileWriterContext &Ctx,
                                    const uint8_t *FuncDescData) {
  const FunctionDescription FD(FuncDescData);
  const uint8_t *Next = FuncDescData + FD.getSize();

  Graph G(FD, Ctx.Dat->data(), Ctx);
  if (G.EdgeFreqs.empty() && G.CallFreqs.empty())
    return Next;

  for (uint32_t I = 0; I < FD.NumEdges; ++I) {
    const uint64_t Freq = G.EdgeFreqs[I];
    if (Freq == 0)
      continue;
    const EdgeDescription *Desc = &FD.Edges[I];
    serializeLoc(OS, Ctx, Desc->From);
    serializeLoc(OS, Ctx, Desc->To);
    OS << "0 " << Freq << "\n";
  }

  for (uint32_t I = 0; I < FD.NumCalls; ++I) {
    const uint64_t Freq = G.CallFreqs[I];
    if (Freq == 0)
      continue;
    const CallDescription *Desc = &FD.Calls[I];
    serializeLoc(OS, Ctx, Desc->From);
    serializeLoc(OS, Ctx, Desc->To);
    OS << "0 " << Freq << "\n";
  }

  return Next;
}

int diffMode() {
  ProfileWriterContext Ctx;

  std::unique_ptr<std::vector<uint64_t>> Dat1 =
      readFileAsVector<uint64_t>(Dat1Filename);

  if (!Dat2Filename.empty()) {
    std::unique_ptr<std::vector<uint64_t>> Dat2 =
        readFileAsVector<uint64_t>(Dat2Filename);
    if (Dat1->size() != Dat2->size())
      report_error(".dat files are not of the same size");

    for (uint64_t i = 0; i < Dat1->size(); ++i)
      (*Dat2)[i] -= (*Dat1)[i];
    Dat1 = std::move(Dat2);
  }

  Ctx.Dat = std::move(Dat1);

  std::error_code EC;
  raw_fd_ostream OutoutFile(OutputFilename, EC, sys::fs::OpenFlags::OF_None);
  if (EC)
    report_error(OutputFilename, EC);

  if (StringRef(OutputFilename).ends_with(".dat")) {
    OutoutFile.write(reinterpret_cast<char *>(Ctx.Dat->data()),
                     Ctx.Dat->size() * sizeof(uint64_t));
    return EXIT_SUCCESS;
  }

  ELFObj Vmlinux(VmlinuxFilename);
  if (Error E = Vmlinux.init())
    report_error(VmlinuxFilename, std::move(E));

  readDescriptions(Vmlinux, Ctx);

  const uint8_t *FuncDescData = Ctx.FuncDescData;
  while (reinterpret_cast<uint64_t>(FuncDescData) <
         reinterpret_cast<uint64_t>(Ctx.Strings))
    FuncDescData = writeFunctionProfile(OutoutFile, Ctx, FuncDescData);
  assert(reinterpret_cast<uint64_t>(FuncDescData) ==
         reinterpret_cast<uint64_t>(Ctx.Strings));
  return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char **argv) {
  cl::HideUnrelatedOptions({LinuxInstrDataCat});
  cl::ParseCommandLineOptions(argc, argv);

  if (DumpSubCommand)
    return dumpMode();

  if (DiffSubCommand)
    return diffMode();

  cl::PrintHelpMessage();
  return EXIT_FAILURE;
}
