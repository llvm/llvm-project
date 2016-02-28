set(LLVM_TARGET_DEFINITIONS UniversalDriverOptions.td)
tablegen(LLVM UniversalDriverOptions.inc -gen-opt-parser-defs)
set(LLVM_TARGET_DEFINITIONS CoreOptions.td)
tablegen(LLVM CoreOptions.inc -gen-opt-parser-defs)
set(LLVM_TARGET_DEFINITIONS DarwinLdOptions.td)
tablegen(LLVM DarwinLdOptions.inc -gen-opt-parser-defs)
add_public_tablegen_target(DriverOptionsTableGen)

add_lld_library(lldDriver
  CoreDriver.cpp
  DarwinLdDriver.cpp
  Driver.cpp
  UniversalDriver.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLD_INCLUDE_DIR}/lld/Driver

  LINK_LIBS
    lldConfig
    lldMachO
    lldCOFF
    lldELF
    lldCore
    lldReaderWriter
    lldYAML
    LLVMObject
    LLVMOption
    LLVMSupport
  )

add_dependencies(lldDriver DriverOptionsTableGen)

