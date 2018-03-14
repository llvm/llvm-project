//==-- examples/clang-interpreter/Manager.cpp - Clang C Interpreter Example -=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Manager.h"

#ifdef CLANG_INTERPRETER_WIN_EXCEPTIONS
#include "llvm/Support/DynamicLibrary.h"

#define WIN32_LEAN_AND_MEAN
#define NOGDI
#define NOMINMAX
#include <windows.h>
#endif

namespace interpreter {

using namespace llvm;

void SingleSectionMemoryManager::Block::Reset(uint8_t *Ptr, uintptr_t Size) {
  assert(Ptr != nullptr && "Bad allocation");
  Addr = Ptr;
  End = Ptr ? Ptr + Size : nullptr;
}

uint8_t *SingleSectionMemoryManager::Block::Next(uintptr_t Size,
                                                 unsigned Alignment) {
  uintptr_t Out = (uintptr_t)Addr;

  // Align the out pointer properly
  if (!Alignment)
    Alignment = 16;
  Out = (Out + Alignment - 1) & ~(uintptr_t)(Alignment - 1);

  // RuntimeDyld should have called reserveAllocationSpace with an amount that
  // will fit all required alignemnts...but assert on this to make sure.
  assert((Out + Size) <= (uintptr_t)End && "Out of bounds");

  // Set the next Addr to deliver at the end of this one.
  Addr = (uint8_t *)(Out + Size);
  return (uint8_t *)Out;
}

uint8_t *SingleSectionMemoryManager::allocateCodeSection(uintptr_t Size,
                                                         unsigned Align,
                                                         unsigned ID,
                                                         StringRef Name) {
  return Code.Next(Size, Align);
}

uint8_t *SingleSectionMemoryManager::allocateDataSection(
    uintptr_t Size, unsigned Align, unsigned ID, StringRef Name, bool RO) {
  return RO ? ROData.Next(Size, Align) : RWData.Next(Size, Align);
}

void SingleSectionMemoryManager::reserveAllocationSpace(
    uintptr_t CodeSize, uint32_t CodeAlign, uintptr_t ROSize, uint32_t ROAlign,
    uintptr_t RWSize, uint32_t RWAlign) {
  // FIXME: Ideally this should be one contiguous block, with Code, ROData,
  // and RWData pointing to sub-blocks within, but setting the correct
  // permissions for that wouldn't work unless we over-allocated to have each
  // Block.Base aligned on a page boundary.
  const unsigned SecID = 0;
  Code.Reset(SectionMemoryManager::allocateCodeSection(CodeSize, CodeAlign,
                                                       SecID, "code"),
             CodeSize);

  ROData.Reset(SectionMemoryManager::allocateDataSection(ROSize, ROAlign, SecID,
                                                         "rodata", true/*RO*/),
               ROSize);

  RWData.Reset(SectionMemoryManager::allocateDataSection(RWSize, RWAlign, SecID,
                                                         "rwdata", false/*RO*/),
               RWSize);

#ifdef CLANG_INTERPRETER_WIN_EXCEPTIONS
  ImageBase =
      (uintptr_t)std::min(std::min(Code.Addr, ROData.Addr), RWData.Addr);
#endif
}

#ifdef CLANG_INTERPRETER_WIN_EXCEPTIONS

// Map an "ImageBase" to a range of adresses that can throw.
//
class SEHFrameHandler {
  typedef SingleSectionMemoryManager::EHFrameInfos EHFrameInfos;
  typedef std::vector<std::pair<DWORD, DWORD>> ImageRanges;
  typedef std::map<uintptr_t, ImageRanges> ImageBaseMap;
  ImageBaseMap m_Map;

  static void MergeRanges(ImageRanges &Ranges);
  uintptr_t FindEHFrame(uintptr_t Caller);

public:
  static __declspec(noreturn) void __stdcall RaiseSEHException(void *, void *);
  void RegisterEHFrames(uintptr_t ImageBase, const EHFrameInfos &Frames,
                        bool Block = true);
  void DeRegisterEHFrames(uintptr_t ImageBase, const EHFrameInfos &Frames);
};

// Merge overlaping ranges for faster searching with throwing PC
void SEHFrameHandler::MergeRanges(ImageRanges &Ranges) {
  std::sort(Ranges.begin(), Ranges.end());

  ImageRanges Merged;
  ImageRanges::iterator It = Ranges.begin();
  auto Current = *(It)++;
  while (It != Ranges.end()) {
    if (Current.second + 1 < It->first) {
      Merged.push_back(Current);
      Current = *(It);
    } else
      Current.second = std::max(Current.second, It->second);
    ++It;
  }
  Merged.emplace_back(Current);
  Ranges.swap(Merged);
}

// Find the "ImageBase" for Caller/PC who is throwing an exception
uintptr_t SEHFrameHandler::FindEHFrame(uintptr_t Caller) {
  for (auto &&Itr : m_Map) {
    const uintptr_t ImgBase = Itr.first;
    for (auto &&Rng : Itr.second) {
      if (Caller >= (ImgBase + Rng.first) && Caller <= (ImgBase + Rng.second))
        return ImgBase;
    }
  }
  return 0;
}

// Register a range of adresses for a single section that
void SEHFrameHandler::RegisterEHFrames(uintptr_t ImageBase,
                                       const EHFrameInfos &Frames, bool Block) {
  if (Frames.empty())
    return;
  assert(m_Map.find(ImageBase) == m_Map.end());

  ImageBaseMap::mapped_type &Ranges = m_Map[ImageBase];
  ImageRanges::value_type *BlockRange = nullptr;
  if (Block) {
    // Merge all unwind adresses into a single contiguous block for faster
    // searching later.
    Ranges.emplace_back(std::numeric_limits<DWORD>::max(),
                        std::numeric_limits<DWORD>::min());
    BlockRange = &Ranges.back();
  }

  for (auto &&Frame : Frames) {
    assert(m_Map.find(DWORD64(Frame.Addr)) == m_Map.end() &&
           "Runtime function should not be a key!");

    PRUNTIME_FUNCTION RFunc = reinterpret_cast<PRUNTIME_FUNCTION>(Frame.Addr);
    const size_t N = Frame.Size / sizeof(RUNTIME_FUNCTION);
    if (BlockRange) {
      for (PRUNTIME_FUNCTION It = RFunc, End = RFunc + N; It < End; ++It) {
        BlockRange->first = std::min(BlockRange->first, It->BeginAddress);
        BlockRange->second = std::max(BlockRange->second, It->EndAddress);
      }
    } else {
      for (PRUNTIME_FUNCTION It = RFunc, End = RFunc + N; It < End; ++It)
        Ranges.emplace_back(It->BeginAddress, It->EndAddress);
    }

    ::RtlAddFunctionTable(RFunc, N, ImageBase);
  }

  if (!Block)
    MergeRanges(Ranges); // Initial sort and merge
}

void SEHFrameHandler::DeRegisterEHFrames(uintptr_t ImageBase,
                                         const EHFrameInfos &Frames) {
  if (Frames.empty())
    return;

  auto Itr = m_Map.find(ImageBase);
  if (Itr != m_Map.end()) {
    // Remove the ImageBase from lookup
    m_Map.erase(Itr);

    // Unregister all the PRUNTIME_FUNCTIONs
    for (auto &&Frame : Frames)
      ::RtlDeleteFunctionTable(reinterpret_cast<PRUNTIME_FUNCTION>(Frame.Addr));
  }
}

// FIXME: Rather than this static and overriding _CxxThrowException via
// DynamicLibrary::AddSymbol, a better route would be to transform the call
// to _CxxThrowException(Arg0, Arg1) -> RaiseSEHException(Arg0, Arg1, this)
// where 'this' is the SingleSectionMemoryManager instance.  This could probably
// be done with clang, and definitely possible by injecting an llvm-IR function
// into the module with the name '_CxxThrowException'
//
static SEHFrameHandler sFrameHandler;

void SingleSectionMemoryManager::deregisterEHFrames() {
  sFrameHandler.DeRegisterEHFrames(ImageBase, EHFrames);
  EHFrameInfos().swap(EHFrames);
}

bool SingleSectionMemoryManager::finalizeMemory(std::string *ErrMsg) {
  sFrameHandler.RegisterEHFrames(ImageBase, EHFrames);
  ImageBase = 0;
  return SectionMemoryManager::finalizeMemory(ErrMsg);
}

SingleSectionMemoryManager::SingleSectionMemoryManager() {
  // Override Windows _CxxThrowException to call into our local version that
  // can throw to and from the JIT.
  sys::DynamicLibrary::AddSymbol(
      "_CxxThrowException",
      (void *)(uintptr_t)&SEHFrameHandler::RaiseSEHException);
}

// Adapted from VisualStudio/VC/crt/src/vcruntime/throw.cpp
#ifdef _WIN64
#define _EH_RELATIVE_OFFSETS 1
#endif
// The NT Exception # that we use
#define EH_EXCEPTION_NUMBER ('msc' | 0xE0000000)
// The magic # identifying this version
#define EH_MAGIC_NUMBER1 0x19930520
#define EH_PURE_MAGIC_NUMBER1 0x01994000
// Number of parameters in exception record
#define EH_EXCEPTION_PARAMETERS 4

// A generic exception record
struct EHExceptionRecord {
  DWORD ExceptionCode;
  DWORD ExceptionFlags;               // Flags determined by NT
  _EXCEPTION_RECORD *ExceptionRecord; // Extra exception record (unused)
  void *ExceptionAddress;             // Address at which exception occurred
  DWORD NumberParameters; // No. of parameters = EH_EXCEPTION_PARAMETERS
  struct EHParameters {
    DWORD magicNumber;            // = EH_MAGIC_NUMBER1
    void *pExceptionObject;       // Pointer to the actual object thrown
    struct ThrowInfo *pThrowInfo; // Description of thrown object
#if _EH_RELATIVE_OFFSETS
    DWORD64 pThrowImageBase; // Image base of thrown object
#endif
  } params;
};

__declspec(noreturn) void __stdcall
SEHFrameHandler::RaiseSEHException(void *CxxExcept, void *Info) {
  uintptr_t Caller;
  static_assert(sizeof(Caller) == sizeof(PVOID), "Size mismatch");

  USHORT Frames = CaptureStackBackTrace(1, 1, (PVOID *)&Caller, NULL);
  assert(Frames && "No frames captured");
  (void)Frames;

  const DWORD64 BaseAddr = sFrameHandler.FindEHFrame(Caller);
  if (BaseAddr == 0)
    _CxxThrowException(CxxExcept, (_ThrowInfo *)Info);

  // A generic exception record
  EHExceptionRecord Exception = {
      EH_EXCEPTION_NUMBER,      // Exception number
      EXCEPTION_NONCONTINUABLE, // Exception flags (we don't do resume)
      nullptr,                  // Additional record (none)
      nullptr,                  // Address of exception (OS fills in)
      EH_EXCEPTION_PARAMETERS,  // Number of parameters
      {EH_MAGIC_NUMBER1, CxxExcept, (struct ThrowInfo *)Info,
#if _EH_RELATIVE_OFFSETS
       BaseAddr
#endif
      }};

// const ThrowInfo* pTI = (const ThrowInfo*)Info;

#ifdef THROW_ISWINRT
  if (pTI && (THROW_ISWINRT((*pTI)))) {
    // The pointer to the ExceptionInfo structure is stored sizeof(void*)
    // infront of each WinRT Exception Info.
    ULONG_PTR *EPtr = *reinterpret_cast<ULONG_PTR **>(CxxExcept);
    EPtr--;

    WINRTEXCEPTIONINFO **ppWei = reinterpret_cast<WINRTEXCEPTIONINFO **>(EPtr);
    pTI = (*ppWei)->throwInfo;
    (*ppWei)->PrepareThrow(ppWei);
  }
#endif

  // If the throw info indicates this throw is from a pure region,
  // set the magic number to the Pure one, so only a pure-region
  // catch will see it.
  //
  // Also use the Pure magic number on Win64 if we were unable to
  // determine an image base, since that was the old way to determine
  // a pure throw, before the TI_IsPure bit was added to the FuncInfo
  // attributes field.
  if (Info != nullptr) {
#ifdef THROW_ISPURE
    if (THROW_ISPURE(*pTI))
      Exception.params.magicNumber = EH_PURE_MAGIC_NUMBER1;
#if _EH_RELATIVE_OFFSETS
    else
#endif // _EH_RELATIVE_OFFSETS
#endif // THROW_ISPURE

    // Not quite sure what this is about, but pThrowImageBase can never be 0
    // here, as that is used to mark when an "ImageBase" was not found.
#if 0 && _EH_RELATIVE_OFFSETS
    if (Exception.params.pThrowImageBase == 0)
      Exception.params.magicNumber = EH_PURE_MAGIC_NUMBER1;
#endif // _EH_RELATIVE_OFFSETS
  }

// Hand it off to the OS:
#if defined(_M_X64) && defined(_NTSUBSET_)
  RtlRaiseException((PEXCEPTION_RECORD)&Exception);
#else
  RaiseException(Exception.ExceptionCode, Exception.ExceptionFlags,
                 Exception.NumberParameters, (PULONG_PTR)&Exception.params);
#endif
}

#endif // CLANG_INTERPRETER_WIN_EXCEPTIONS

} // namespace interpreter
