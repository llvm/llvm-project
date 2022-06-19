if ("X86" IN_LIST LLVM_TARGETS_TO_BUILD)
  message(STATUS "Targeting X86 in llvm-bolt")
  add_subdirectory(X86)
endif()

if ("AArch64" IN_LIST LLVM_TARGETS_TO_BUILD)
  message(STATUS "Targeting AArch64 in llvm-bolt")
  add_subdirectory(AArch64)
endif()
