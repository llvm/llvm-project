set(LLVM_LINK_COMPONENTS 
  ${LLVM_TARGETS_TO_BUILD}
  BitWriter
  Core
  BinaryFormat
  MC
  Target
  TransformUtils
  Analysis
  Passes
  IRReader
  Object
  Support
  CodeGen
  LTO)

if(NOT CLANG_BUILT_STANDALONE)
  set(tablegen_deps intrinsics_gen)
endif()

add_clang_executable(clang-linker-wrapper
  ClangLinkerWrapper.cpp
  OffloadWrapper.cpp

  DEPENDS
  ${tablegen_deps}
  )

set(CLANG_LINKER_WRAPPER_LIB_DEPS
  clangBasic
  )

add_dependencies(clang clang-linker-wrapper)

target_link_libraries(clang-linker-wrapper
  PRIVATE
  ${CLANG_LINKER_WRAPPER_LIB_DEPS}
  )

install(TARGETS clang-linker-wrapper RUNTIME DESTINATION bin)
