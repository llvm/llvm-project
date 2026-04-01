# See notes in lib/TableGen/CMakeLists.txt regarding why this library is
# declared as it is. The same dependency rules apply.
llvm_add_library(AIIRTblgenLib STATIC
  AiirTblgenMain.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Tools/aiir-tblgen

  DISABLE_LLVM_LINK_LLVM_DYLIB

  LINK_COMPONENTS
  TableGen

  LINK_LIBS PUBLIC
  AIIRTableGen
  )

aiir_check_all_link_libraries(AIIRTblgenLib)

add_aiir_library_install(AIIRTblgenLib)
