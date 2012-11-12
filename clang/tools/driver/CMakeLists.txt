set( LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  asmparser
  bitreader
  bitwriter
  codegen
  instrumentation
  ipo
  linker
  selectiondag
  )

add_clang_executable(clang
  driver.cpp
  cc1_main.cpp
  cc1as_main.cpp
  )

target_link_libraries(clang
  clangFrontendTool
  clangAST
  clangAnalysis
  clangBasic
  clangCodeGen
  clangDriver
  clangEdit
  clangFrontend
  clangLex
  clangParse
  clangEdit
  clangARCMigrate
  clangRewriteCore
  clangRewriteFrontend
  clangSema
  clangSerialization
  clangStaticAnalyzerFrontend
  clangStaticAnalyzerCheckers
  clangStaticAnalyzerCore
  )

set_target_properties(clang PROPERTIES VERSION ${CLANG_EXECUTABLE_VERSION})

add_dependencies(clang clang-headers)

if(UNIX)
  set(CLANGXX_LINK_OR_COPY create_symlink)
# Create a relative symlink
  set(clang_binary "clang${CMAKE_EXECUTABLE_SUFFIX}")
else()
  set(CLANGXX_LINK_OR_COPY copy)
  set(clang_binary "${LLVM_BINARY_DIR}/bin/${CMAKE_CFG_INTDIR}/clang${CMAKE_EXECUTABLE_SUFFIX}")
endif()

# Create the clang++ symlink in the build directory.
set(clang_pp "${LLVM_BINARY_DIR}/bin/${CMAKE_CFG_INTDIR}/clang++${CMAKE_EXECUTABLE_SUFFIX}")
add_custom_command(TARGET clang POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E ${CLANGXX_LINK_OR_COPY} "${clang_binary}" "${clang_pp}")

set_property(DIRECTORY APPEND
  PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${clang_pp})

install(TARGETS clang
  RUNTIME DESTINATION bin)

# Create the clang++ symlink at installation time.
install(SCRIPT clang_symlink.cmake -DCMAKE_INSTALL_PREFIX=\"${CMAKE_INSTALL_PREFIX}\")
