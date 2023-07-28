include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/.. )

set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsDescs
  AllTargetsInfos
  FrontendOpenMP
  support
  )

# Needed by LLVM's CMake checks because this file defines multiple targets.
set(LLVM_OPTIONAL_SOURCES CIRTidyMain.cpp CIRTidyToolMain.cpp)

add_clang_library(CIRTidyMain
  CIRTidyMain.cpp

  LINK_LIBS
  CIRTidy
  clangTidy
  MLIRIR
  ${ALL_CLANG_TIDY_CHECKS}
  MLIRIR

  DEPENDS
  omp_gen
  )

clang_target_link_libraries(CIRTidyMain
  PRIVATE
  clangBasic
  clangTooling
  clangToolingCore
  )

add_clang_tool(cir-tidy
  CIRTidyToolMain.cpp
  )
add_dependencies(cir-tidy
  clang-resource-headers
  )

target_link_libraries(cir-tidy
  PRIVATE
  CIRTidyMain
  CIRTidy
  )

install(TARGETS cir-tidy
  DESTINATION bin
  )
