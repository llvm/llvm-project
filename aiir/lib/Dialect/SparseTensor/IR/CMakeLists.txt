# This library is shared by both AIIRSparseTensorDialect and
# AIIRSparseTensorRuntime, so it must not depend on any of the AIIR/LLVM
# internals or else aiir_c_runner_utils will inherit that dependency.
#
# Because this is a header-only (`INTERFACE`) library, we cannot use
# the `add_aiir_library` function.  So we do our best to replicate the
# relevant portions below.  If doing so becomes too complicated, then
# we should adjust the `add_aiir_library` function to also work for
# `INTERFACE` libraries.
set(AIIRSparseTensorEnums_hdrs
    aiir/Dialect/SparseTensor/IR/Enums.h)
if(MSVC_IDE OR XCODE)
  foreach(hdr ${AIIRSparseTensorEnums_hdrs})
    set_source_files_properties(${AIIR_MAIN_INCLUDE_DIR}/${hdr}
      PROPERTIES HEADER_FILE_ONLY ON)
  endforeach()
endif()
add_library(AIIRSparseTensorEnums INTERFACE)
foreach(hdr ${AIIRSparseTensorEnums_hdrs})
  target_sources(AIIRSparseTensorEnums INTERFACE
    $<BUILD_INTERFACE:${AIIR_MAIN_INCLUDE_DIR}/${hdr}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${hdr}>)
endforeach()
if(TARGET AIIRSparseTensorEnums)
  add_aiir_library_install(AIIRSparseTensorEnums)
else()
  add_custom_target(AIIRSparseTensorEnums) # empty "phony" target
endif()

add_aiir_dialect_library(AIIRSparseTensorDialect
  SparseTensorDialect.cpp
  SparseTensorInterfaces.cpp
  Detail/Var.cpp
  Detail/DimLvlMap.cpp
  Detail/LvlTypeParser.cpp
  Detail/DimLvlMapParser.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/SparseTensor

  DEPENDS
  AIIRSparseTensorAttrDefsIncGen
  AIIRSparseTensorOpsIncGen
  AIIRSparseTensorTypesIncGen

  LINK_LIBS PUBLIC
  AIIRArithDialect
  AIIRControlFlowInterfaces
  AIIRComplexDialect
  AIIRDialect
  AIIRDialectUtils
  AIIRIR
  AIIRInferTypeOpInterface
  AIIRLoopLikeInterface
  AIIRSupport
  AIIRSparseTensorEnums
  )
