include_directories(${LLVM_MAIN_SRC_DIR}/../mlir/include)
include_directories(${LLVM_BINARY_DIR}/tools/mlir/include)

if ((MINGW OR CYGWIN) AND CLANG_BUILD_CLANG_DYLIB AND NOT BUILD_SHARED_LIBS)
  get_property(save_clang_libs GLOBAL PROPERTY CLANG_STATIC_LIBS)
  get_property(cir_clang_libs GLOBAL PROPERTY CLANG_CIR_STATIC_LIBS)
  set_property(GLOBAL PROPERTY CLANG_STATIC_LIBS ${cir_clang_libs})
endif()

add_subdirectory(Dialect)
add_subdirectory(CodeGen)
add_subdirectory(FrontendAction)
add_subdirectory(Interfaces)
add_subdirectory(Lowering)

add_clang_library(CIRRegisterAllDialects
  RegisterAllDialects.cpp

  DEPENDS
  mlir-headers

  LINK_LIBS PUBLIC
  MLIRCIR
  CIROpenACCSupport
  CIROpenMPSupport
  MLIRArithDialect
  MLIRDLTIDialect
  MLIRLLVMDialect
)

if ((MINGW OR CYGWIN) AND CLANG_BUILD_CLANG_DYLIB AND NOT BUILD_SHARED_LIBS)
  get_property(cir_clang_libs GLOBAL PROPERTY CLANG_STATIC_LIBS)
  set_property(GLOBAL PROPERTY CLANG_STATIC_LIBS ${save_clang_libs})
  set_property(GLOBAL PROPERTY CLANG_CIR_STATIC_LIBS ${cir_clang_libs})

  set(libs ${cir_clang_libs} MLIROptLib MLIRTranslateLib)
  set(seen)
  while (libs)
    list(POP_BACK libs lib)
    if (lib IN_LIST seen)
      continue()
    endif()
    list(APPEND seen ${lib})
    if (TARGET ${lib})
      get_target_property(m ${lib} LINK_LIBRARIES)
      if (m)
        list(APPEND libs ${m})
      endif()
    endif()
    if (lib IN_LIST cir_clang_libs OR lib IN_LIST save_clang_libs)
      continue()
    endif()
    if (lib MATCHES "^LLVM")
      # no-op
    elseif (lib MATCHES "^MLIR")
      list(APPEND cir_mlir_libs ${lib})
      if (TARGET obj.${lib})
        list(APPEND cir_mlir_objs $<TARGET_OBJECTS:obj.${lib}>)
      endif()
    else()
      list(APPEND cir_mlir_lib_deps ${lib})
    endif()
  endwhile()

  foreach (lib ${cir_clang_libs})
    list(APPEND _OBJECTS $<TARGET_OBJECTS:obj.${lib}>)
    get_target_property(interface ${lib} LINK_LIBRARIES)
    list(FILTER interface EXCLUDE REGEX "^clang")
    foreach (i ${interface})
      if (NOT i IN_LIST cir_clang_libs AND
          NOT i IN_LIST save_clang_libs AND
          NOT i IN_LIST cir_mlir_libs)
        list(APPEND _DEPS ${i})
      endif()
    endforeach()
  endforeach()
  list(REMOVE_DUPLICATES _DEPS)

  if (CLANG_LINK_CLANG_DYLIB)
    set(INSTALL_WITH_TOOLCHAIN INSTALL_WITH_TOOLCHAIN)
  endif()

  foreach (lib ${save_clang_libs})
    list(APPEND libname $<TARGET_FILE:${lib}>)
  endforeach()

  add_clang_library(clang-cpp-cir-mlir
    SHARED
    ${INSTALL_WITH_TOOLCHAIN}
    ${cir_mlir_objs}
    LINK_LIBS
    ${cir_mlir_lib_deps}
  )
  target_link_options(clang-cpp-cir-mlir PRIVATE LINKER:--export-all-symbols)

  get_host_tool_path(llvm-nm LLVM_NM llvm_nm_exe llvm_nm_target)
  get_host_tool_path(llvm-dlltool LLVM_DLLTOOL llvm_dlltool_exe llvm_dlltool_target)
  set(GEN_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/generate-explib.py)
  add_custom_command(
    OUTPUT clang-cpp.exp.a
    COMMAND "${Python3_EXECUTABLE}" ${GEN_SCRIPT} ${libname} --dlltool "${llvm_dlltool_exe}" --nm "${llvm_nm_exe}" -o ${CMAKE_CURRENT_BINARY_DIR}/clang-cpp.exp.a -d $<TARGET_FILE_NAME:clang-cpp>
    DEPENDS ${GEN_SCRIPT} ${llvm_nm_target} ${llvm_dlltool_target} ${save_clang_libs}
    COMMAND_EXPAND_LISTS
    VERBATIM
  )
  add_custom_target(clang-cpp.exp.a-target DEPENDS clang-cpp.exp.a)

  add_clang_library(clang-cpp-cir
    SHARED
    ${INSTALL_WITH_TOOLCHAIN}
    ${_OBJECTS}

    LINK_LIBS
    ${CMAKE_CURRENT_BINARY_DIR}/clang-cpp.exp.a
    ${_DEPS}

    DEPENDS
    clang-cpp.exp.a-target
  )
  target_link_libraries(clang-cpp-cir PUBLIC clang-cpp-cir-mlir)
  target_link_options(clang-cpp-cir PRIVATE LINKER:--export-all-symbols)
endif()
