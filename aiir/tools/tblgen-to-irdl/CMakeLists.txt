set(LLVM_LINK_COMPONENTS
  TableGen
)

add_tablegen(tblgen-to-irdl AIIR
  DESTINATION "${AIIR_TOOLS_INSTALL_DIR}"
  EXPORT AIIR
  tblgen-to-irdl.cpp
  OpDefinitionsGen.cpp
  )

target_link_libraries(tblgen-to-irdl
  PRIVATE
  AIIRIR
  AIIRIRDL
  AIIRTblgenLib
  AIIRSupport
)

aiir_check_all_link_libraries(tblgen-to-irdl)
