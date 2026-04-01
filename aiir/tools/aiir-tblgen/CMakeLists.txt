set(LLVM_LINK_COMPONENTS
  CodeGenTypes
  Demangle
  Support
  TableGen
)

add_tablegen(aiir-tblgen AIIR
  DESTINATION "${AIIR_TOOLS_INSTALL_DIR}"
  EXPORT AIIR
  AttrOrTypeDefGen.cpp
  AttrOrTypeFormatGen.cpp
  BytecodeDialectGen.cpp
  DialectGen.cpp
  DialectInterfacesGen.cpp
  DirectiveCommonGen.cpp
  EnumsGen.cpp
  EnumPythonBindingGen.cpp
  FormatGen.cpp
  LLVMIRConversionGen.cpp
  LLVMIRIntrinsicGen.cpp
  aiir-tblgen.cpp
  OmpOpGen.cpp
  OpClass.cpp
  OpDefinitionsGen.cpp
  OpDocGen.cpp
  OpFormatGen.cpp
  OpGenHelpers.cpp
  OpInterfacesGen.cpp
  OpPythonBindingGen.cpp
  PassCAPIGen.cpp
  PassDocGen.cpp
  PassGen.cpp
  RewriterGen.cpp
  SPIRVUtilsGen.cpp
  TosaUtilsGen.cpp
  CppGenUtilities.cpp
  )

target_link_libraries(aiir-tblgen
  PRIVATE
  AIIRTblgenLib)

aiir_check_all_link_libraries(aiir-tblgen)
