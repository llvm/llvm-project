//===-- examples/clang-interpreter/Manager.h - Clang C Interpreter Example -==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_EXAMPLE_INTERPRETER_MANAGER_H
#define CLANG_EXAMPLE_INTERPRETER_MANAGER_H

#include "llvm/ExecutionEngine/SectionMemoryManager.h"

#if defined(_WIN32) && defined(_WIN64)
#define CLANG_INTERPRETER_COFF_FORMAT
#define CLANG_INTERPRETER_WIN_EXCEPTIONS
#endif

namespace interpreter {

class SingleSectionMemoryManager : public llvm::SectionMemoryManager {
  struct Block {
    uint8_t *Addr = nullptr, *End = nullptr;
    void Reset(uint8_t *Ptr, uintptr_t Size);
    uint8_t *Next(uintptr_t Size, unsigned Alignment);
  };
  Block Code, ROData, RWData;

public:
  uint8_t *allocateCodeSection(uintptr_t Size, unsigned Align, unsigned ID,
                               llvm::StringRef Name) final;

  uint8_t *allocateDataSection(uintptr_t Size, unsigned Align, unsigned ID,
                               llvm::StringRef Name, bool RO) final;

  void reserveAllocationSpace(uintptr_t CodeSize, uint32_t CodeAlign,
                              uintptr_t ROSize, uint32_t ROAlign,
                              uintptr_t RWSize, uint32_t RWAlign) final;

  bool needsToReserveAllocationSpace() override { return true; }

#ifdef CLANG_INTERPRETER_WIN_EXCEPTIONS
  using llvm::SectionMemoryManager::EHFrameInfos;

  SingleSectionMemoryManager();

  void deregisterEHFrames() override;

  bool finalizeMemory(std::string *ErrMsg) override;

private:
  uintptr_t ImageBase = 0;
#endif
};

}

#endif // CLANG_EXAMPLE_INTERPRETER_MANAGER_H
