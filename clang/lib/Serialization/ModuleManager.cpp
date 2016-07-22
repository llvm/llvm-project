//===--- ModuleManager.cpp - Module Manager ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the ModuleManager class, which manages a set of loaded
//  modules for the ASTReader.
//
//===----------------------------------------------------------------------===//
#include "clang/Serialization/ModuleManager.h"
#include "clang/Frontend/PCHContainerOperations.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/ModuleMap.h"
#include "clang/Serialization/GlobalModuleIndex.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include <system_error>

#ifndef NDEBUG
#include "llvm/Support/GraphWriter.h"
#endif

using namespace clang;
using namespace serialization;

ModuleFile *ModuleManager::lookup(StringRef Name) {
  const FileEntry *Entry = FileMgr.getFile(Name, /*openFile=*/false,
                                           /*cacheFailure=*/false);
  if (Entry)
    return lookup(Entry);

  return nullptr;
}

ModuleFile *ModuleManager::lookup(const FileEntry *File) {
  llvm::DenseMap<const FileEntry *, ModuleFile *>::iterator Known
    = Modules.find(File);
  if (Known == Modules.end())
    return nullptr;

  return Known->second;
}

std::unique_ptr<llvm::MemoryBuffer>
ModuleManager::lookupBuffer(StringRef Name) {
  const FileEntry *Entry = FileMgr.getFile(Name, /*openFile=*/false,
                                           /*cacheFailure=*/false);
  return std::move(InMemoryBuffers[Entry]);
}

ModuleManager::AddModuleResult
ModuleManager::addModule(StringRef FileName, ModuleKind Type,
                         SourceLocation ImportLoc, ModuleFile *ImportedBy,
                         unsigned Generation,
                         off_t ExpectedSize, time_t ExpectedModTime,
                         ASTFileSignature ExpectedSignature,
                         ASTFileSignatureReader ReadSignature,
                         ModuleFile *&Module,
                         std::string &ErrorStr) {
  Module = nullptr;

  // Look for the file entry. This only fails if the expected size or
  // modification time differ.
  const FileEntry *Entry;
  if (Type == MK_ExplicitModule) {
    // If we're not expecting to pull this file out of the module cache, it
    // might have a different mtime due to being moved across filesystems in
    // a distributed build. The size must still match, though. (As must the
    // contents, but we can't check that.)
    ExpectedModTime = 0;
  }
  if (lookupModuleFile(FileName, ExpectedSize, ExpectedModTime, Entry)) {
    ErrorStr = "module file out of date";
    return OutOfDate;
  }

  if (!Entry && FileName != "-") {
    ErrorStr = "module file not found";
    return Missing;
  }

  // Check whether we already loaded this module, before
  ModuleFile *&ModuleEntry = Modules[Entry];
  bool NewModule = false;
  if (!ModuleEntry) {
    // Allocate a new module.
    ModuleFile *New = new ModuleFile(Type, Generation);
    New->Index = Chain.size();
    New->FileName = FileName.str();
    New->File = Entry;
    New->ImportLoc = ImportLoc;
    Chain.push_back(New);
    if (!New->isModule())
      PCHChain.push_back(New);
    if (!ImportedBy)
      Roots.push_back(New);
    NewModule = true;
    ModuleEntry = New;

    New->InputFilesValidationTimestamp = 0;
    if (New->Kind == MK_ImplicitModule) {
      std::string TimestampFilename = New->getTimestampFilename();
      vfs::Status Status;
      // A cached stat value would be fine as well.
      if (!FileMgr.getNoncachedStatValue(TimestampFilename, Status))
        New->InputFilesValidationTimestamp =
            Status.getLastModificationTime().toEpochTime();
    }

    // Load the contents of the module
    if (std::unique_ptr<llvm::MemoryBuffer> Buffer = lookupBuffer(FileName)) {
      // The buffer was already provided for us.
      New->Buffer = std::move(Buffer);
    } else {
      // Open the AST file.
      llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> Buf(
          (std::error_code()));
      if (FileName == "-") {
        Buf = llvm::MemoryBuffer::getSTDIN();
      } else {
        // Leave the FileEntry open so if it gets read again by another
        // ModuleManager it must be the same underlying file.
        // FIXME: Because FileManager::getFile() doesn't guarantee that it will
        // give us an open file, this may not be 100% reliable.
        Buf = FileMgr.getBufferForFile(New->File,
                                       /*IsVolatile=*/false,
                                       /*ShouldClose=*/false);
      }

      if (!Buf) {
        ErrorStr = Buf.getError().message();
        return Missing;
      }

      New->Buffer = std::move(*Buf);
    }

    // Initialize the stream.
    PCHContainerRdr.ExtractPCH(New->Buffer->getMemBufferRef(), New->StreamFile);
  }

  if (ExpectedSignature) {
    if (NewModule)
      ModuleEntry->Signature = ReadSignature(ModuleEntry->StreamFile);
    else
      assert(ModuleEntry->Signature == ReadSignature(ModuleEntry->StreamFile));

    if (ModuleEntry->Signature != ExpectedSignature) {
      ErrorStr = ModuleEntry->Signature ? "signature mismatch"
                                        : "could not read module signature";

      if (NewModule) {
        // Remove the module file immediately, since removeModules might try to
        // invalidate the file cache for Entry, and that is not safe if this
        // module is *itself* up to date, but has an out-of-date importer.
        Modules.erase(Entry);
        assert(Chain.back() == ModuleEntry);
        Chain.pop_back();
        if (!ModuleEntry->isModule())
          PCHChain.pop_back();
        if (Roots.back() == ModuleEntry)
          Roots.pop_back();
        else
          assert(ImportedBy);
        delete ModuleEntry;
      }
      return OutOfDate;
    }
  }
  
  if (ImportedBy) {
    ModuleEntry->ImportedBy.insert(ImportedBy);
    ImportedBy->Imports.insert(ModuleEntry);
  } else {
    if (!ModuleEntry->DirectlyImported)
      ModuleEntry->ImportLoc = ImportLoc;
    
    ModuleEntry->DirectlyImported = true;
  }

  Module = ModuleEntry;
  return NewModule? NewlyLoaded : AlreadyLoaded;
}

void ModuleManager::removeModules(
    ModuleIterator first, ModuleIterator last,
    llvm::SmallPtrSetImpl<ModuleFile *> &LoadedSuccessfully,
    ModuleMap *modMap) {
  if (first == last)
    return;

  // Explicitly clear VisitOrder since we might not notice it is stale.
  VisitOrder.clear();

  // Collect the set of module file pointers that we'll be removing.
  llvm::SmallPtrSet<ModuleFile *, 4> victimSet(first, last);

  auto IsVictim = [&](ModuleFile *MF) {
    return victimSet.count(MF);
  };
  // Remove any references to the now-destroyed modules.
  for (unsigned i = 0, n = Chain.size(); i != n; ++i) {
    Chain[i]->ImportedBy.remove_if(IsVictim);
  }
  Roots.erase(std::remove_if(Roots.begin(), Roots.end(), IsVictim),
              Roots.end());

  // Remove the modules from the PCH chain.
  for (auto I = first; I != last; ++I) {
    if (!(*I)->isModule()) {
      PCHChain.erase(std::find(PCHChain.begin(), PCHChain.end(), *I),
                     PCHChain.end());
      break;
    }
  }

  // Delete the modules and erase them from the various structures.
  for (ModuleIterator victim = first; victim != last; ++victim) {
    Modules.erase((*victim)->File);

    if (modMap) {
      StringRef ModuleName = (*victim)->ModuleName;
      if (Module *mod = modMap->findModule(ModuleName)) {
        mod->setASTFile(nullptr);
      }
    }

    // Files that didn't make it through ReadASTCore successfully will be
    // rebuilt (or there was an error). Invalidate them so that we can load the
    // new files that will be renamed over the old ones.
    if (LoadedSuccessfully.count(*victim) == 0)
      FileMgr.invalidateCache((*victim)->File);

    delete *victim;
  }

  // Remove the modules from the chain.
  Chain.erase(first, last);
}

void
ModuleManager::addInMemoryBuffer(StringRef FileName,
                                 std::unique_ptr<llvm::MemoryBuffer> Buffer) {

  const FileEntry *Entry =
      FileMgr.getVirtualFile(FileName, Buffer->getBufferSize(), 0);
  InMemoryBuffers[Entry] = std::move(Buffer);
}

ModuleManager::VisitState *ModuleManager::allocateVisitState() {
  // Fast path: if we have a cached state, use it.
  if (FirstVisitState) {
    VisitState *Result = FirstVisitState;
    FirstVisitState = FirstVisitState->NextState;
    Result->NextState = nullptr;
    return Result;
  }

  // Allocate and return a new state.
  return new VisitState(size());
}

void ModuleManager::returnVisitState(VisitState *State) {
  assert(State->NextState == nullptr && "Visited state is in list?");
  State->NextState = FirstVisitState;
  FirstVisitState = State;
}

void ModuleManager::setGlobalIndex(GlobalModuleIndex *Index) {
  GlobalIndex = Index;
  if (!GlobalIndex) {
    ModulesInCommonWithGlobalIndex.clear();
    return;
  }

  // Notify the global module index about all of the modules we've already
  // loaded.
  for (unsigned I = 0, N = Chain.size(); I != N; ++I) {
    if (!GlobalIndex->loadedModuleFile(Chain[I])) {
      ModulesInCommonWithGlobalIndex.push_back(Chain[I]);
    }
  }
}

void ModuleManager::moduleFileAccepted(ModuleFile *MF) {
  if (!GlobalIndex || GlobalIndex->loadedModuleFile(MF))
    return;

  ModulesInCommonWithGlobalIndex.push_back(MF);
}

ModuleManager::ModuleManager(FileManager &FileMgr,
                             const PCHContainerReader &PCHContainerRdr)
    : FileMgr(FileMgr), PCHContainerRdr(PCHContainerRdr), GlobalIndex(),
      FirstVisitState(nullptr) {}

ModuleManager::~ModuleManager() {
  for (unsigned i = 0, e = Chain.size(); i != e; ++i)
    delete Chain[e - i - 1];
  delete FirstVisitState;
}

void ModuleManager::visit(llvm::function_ref<bool(ModuleFile &M)> Visitor,
                          llvm::SmallPtrSetImpl<ModuleFile *> *ModuleFilesHit) {
  // If the visitation order vector is the wrong size, recompute the order.
  if (VisitOrder.size() != Chain.size()) {
    unsigned N = size();
    VisitOrder.clear();
    VisitOrder.reserve(N);
    
    // Record the number of incoming edges for each module. When we
    // encounter a module with no incoming edges, push it into the queue
    // to seed the queue.
    SmallVector<ModuleFile *, 4> Queue;
    Queue.reserve(N);
    llvm::SmallVector<unsigned, 4> UnusedIncomingEdges;
    UnusedIncomingEdges.resize(size());
    for (ModuleFile *M : llvm::reverse(*this)) {
      unsigned Size = M->ImportedBy.size();
      UnusedIncomingEdges[M->Index] = Size;
      if (!Size)
        Queue.push_back(M);
    }

    // Traverse the graph, making sure to visit a module before visiting any
    // of its dependencies.
    while (!Queue.empty()) {
      ModuleFile *CurrentModule = Queue.pop_back_val();
      VisitOrder.push_back(CurrentModule);

      // For any module that this module depends on, push it on the
      // stack (if it hasn't already been marked as visited).
      for (auto M = CurrentModule->Imports.rbegin(),
                MEnd = CurrentModule->Imports.rend();
           M != MEnd; ++M) {
        // Remove our current module as an impediment to visiting the
        // module we depend on. If we were the last unvisited module
        // that depends on this particular module, push it into the
        // queue to be visited.
        unsigned &NumUnusedEdges = UnusedIncomingEdges[(*M)->Index];
        if (NumUnusedEdges && (--NumUnusedEdges == 0))
          Queue.push_back(*M);
      }
    }

    assert(VisitOrder.size() == N && "Visitation order is wrong?");

    delete FirstVisitState;
    FirstVisitState = nullptr;
  }

  VisitState *State = allocateVisitState();
  unsigned VisitNumber = State->NextVisitNumber++;

  // If the caller has provided us with a hit-set that came from the global
  // module index, mark every module file in common with the global module
  // index that is *not* in that set as 'visited'.
  if (ModuleFilesHit && !ModulesInCommonWithGlobalIndex.empty()) {
    for (unsigned I = 0, N = ModulesInCommonWithGlobalIndex.size(); I != N; ++I)
    {
      ModuleFile *M = ModulesInCommonWithGlobalIndex[I];
      if (!ModuleFilesHit->count(M))
        State->VisitNumber[M->Index] = VisitNumber;
    }
  }

  for (unsigned I = 0, N = VisitOrder.size(); I != N; ++I) {
    ModuleFile *CurrentModule = VisitOrder[I];
    // Should we skip this module file?
    if (State->VisitNumber[CurrentModule->Index] == VisitNumber)
      continue;

    // Visit the module.
    assert(State->VisitNumber[CurrentModule->Index] == VisitNumber - 1);
    State->VisitNumber[CurrentModule->Index] = VisitNumber;
    if (!Visitor(*CurrentModule))
      continue;

    // The visitor has requested that cut off visitation of any
    // module that the current module depends on. To indicate this
    // behavior, we mark all of the reachable modules as having been visited.
    ModuleFile *NextModule = CurrentModule;
    do {
      // For any module that this module depends on, push it on the
      // stack (if it hasn't already been marked as visited).
      for (llvm::SetVector<ModuleFile *>::iterator
             M = NextModule->Imports.begin(),
             MEnd = NextModule->Imports.end();
           M != MEnd; ++M) {
        if (State->VisitNumber[(*M)->Index] != VisitNumber) {
          State->Stack.push_back(*M);
          State->VisitNumber[(*M)->Index] = VisitNumber;
        }
      }

      if (State->Stack.empty())
        break;

      // Pop the next module off the stack.
      NextModule = State->Stack.pop_back_val();
    } while (true);
  }

  returnVisitState(State);
}

bool ModuleManager::lookupModuleFile(StringRef FileName,
                                     off_t ExpectedSize,
                                     time_t ExpectedModTime,
                                     const FileEntry *&File) {
  // Open the file immediately to ensure there is no race between stat'ing and
  // opening the file.
  File = FileMgr.getFile(FileName, /*openFile=*/true, /*cacheFailure=*/false);

  if (!File && FileName != "-") {
    return false;
  }

  if ((ExpectedSize && ExpectedSize != File->getSize()) ||
      (ExpectedModTime && ExpectedModTime != File->getModificationTime()))
    // Do not destroy File, as it may be referenced. If we need to rebuild it,
    // it will be destroyed by removeModules.
    return true;

  return false;
}

#ifndef NDEBUG
namespace llvm {
  template<>
  struct GraphTraits<ModuleManager> {
    typedef ModuleFile NodeType;
    typedef llvm::SetVector<ModuleFile *>::const_iterator ChildIteratorType;
    typedef ModuleManager::ModuleConstIterator nodes_iterator;
    
    static ChildIteratorType child_begin(NodeType *Node) {
      return Node->Imports.begin();
    }

    static ChildIteratorType child_end(NodeType *Node) {
      return Node->Imports.end();
    }
    
    static nodes_iterator nodes_begin(const ModuleManager &Manager) {
      return Manager.begin();
    }
    
    static nodes_iterator nodes_end(const ModuleManager &Manager) {
      return Manager.end();
    }
  };
  
  template<>
  struct DOTGraphTraits<ModuleManager> : public DefaultDOTGraphTraits {
    explicit DOTGraphTraits(bool IsSimple = false)
      : DefaultDOTGraphTraits(IsSimple) { }
    
    static bool renderGraphFromBottomUp() {
      return true;
    }

    std::string getNodeLabel(ModuleFile *M, const ModuleManager&) {
      return M->ModuleName;
    }
  };
}

void ModuleManager::viewGraph() {
  llvm::ViewGraph(*this, "Modules");
}
#endif
