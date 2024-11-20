function(cannonicalize_cmake_boolean var)
    if(${var})
        set(${var} 1 PARENT_SCOPE)
    else()
        set(${var} 0 PARENT_SCOPE)
    endif()
endfunction()

cannonicalize_cmake_boolean(COMGR_DISABLE_SPIRV)

configure_file(lit.site.cfg.py.in lit.site.cfg.py @ONLY)

if (NOT DEFINED LLVM_LIT_PATH)
  # Comgr source build
  if (EXISTS "${LLVM_TOOLS_BINARY_DIR}/../../bin/llvm-lit")
    set(LLVM_LIT_PATH "${LLVM_TOOLS_BINARY_DIR}/../../bin/llvm-lit")
  # LLVM external projects build
  else()
    set(LLVM_LIT_PATH "${LLVM_TOOLS_BINARY_DIR}/llvm-lit")
  endif()
endif()

add_custom_target(test-lit COMMAND "${LLVM_LIT_PATH}"
                  "${CMAKE_CURRENT_BINARY_DIR}" -v)

macro(add_comgr_lit_binary name)
  add_executable("${name}" "comgr-sources/${name}.c")
  set_target_properties("${name}" PROPERTIES
    C_STANDARD 99
    C_STANDARD_REQUIRED Yes
    C_EXTENSIONS No)
  target_link_libraries("${name}" amd_comgr)
  add_dependencies(check-comgr "${name}")
endmacro()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_comgr_lit_binary(source-to-bc-with-dev-libs)
add_comgr_lit_binary(spirv-translator)
add_comgr_lit_binary(compile-minimal-test)

add_dependencies(check-comgr test-lit)
