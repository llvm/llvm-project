set(TEST_INPUT_BINARIES)
# Create target ${name} which depends on a clang command to compile ${input} to
# ${output}, with any additional arguments from ${ARGN}, and add it to the
# TEST_INPUT_BINARIES target list.
macro(add_test_input_binary name input output)
  add_custom_command(
    OUTPUT "${output}"
    COMMAND "$<TARGET_FILE:clang>" --target=amdgcn-amd-amdhsa -mcpu=gfx803
    ${ARGN} "${CMAKE_CURRENT_SOURCE_DIR}/${input}" -o "${output}"
    VERBATIM
    DEPENDS clang "${input}")
  add_custom_target("${name}"
    DEPENDS "${output}"
    SOURCES "${input}")
  list(APPEND TEST_INPUT_BINARIES "${name}")
endmacro()

add_test_input_binary(reloc1 reloc1.cl reloc1.o -c -mno-code-object-v3)
add_test_input_binary(reloc2 reloc2.cl reloc2.o -c -mno-code-object-v3)
add_test_input_binary(shared shared.cl shared.so -mno-code-object-v3)
add_test_input_binary(shared-v3 shared.cl shared-v3.so -mcode-object-v3)
configure_file("source1.cl" "source1.cl" COPYONLY)
configure_file("source1-device-libs.cl" "source1-device-libs.cl" COPYONLY)
configure_file("source2.cl" "source2.cl" COPYONLY)
configure_file("include-a.h" "include-a.h" COPYONLY)
configure_file("source1.s" "source1.s" COPYONLY)
configure_file("shared.cl" "shared.cl" COPYONLY)

# Create executable ${name} and accompanying test ${name} built from
# test/${name}.cl
macro(add_comgr_test name)
  add_executable("${name}" "${name}.c")
  set_target_properties("${name}" PROPERTIES
    C_STANDARD 99
    C_STANDARD_REQUIRED Yes
    C_EXTENSIONS No)
  target_compile_definitions("${name}"
    PRIVATE -DTEST_OBJ_DIR=\"${CMAKE_CURRENT_BINARY_DIR}\")
  target_link_libraries("${name}"
    amd_comgr)
  add_dependencies("${name}"
    ${TEST_INPUT_BINARIES})
  add_test(NAME "${name}"
    COMMAND "${name}")
  # Windows binaries have no equivalent to RPATH, so we must set their PATH to
  # include the .lib/.dll directory.
  if (NOT(UNIX))
    set_test_properties("${name}"
      PROPERTIES ENVIRONMENT "PATH=$<TARGET_LINKER_FILE_DIR:amd_comgr>")
  endif()
endmacro()

add_comgr_test(data_test)
add_comgr_test(disasm_llvm_reloc_test)
add_comgr_test(disasm_llvm_so_test)
add_comgr_test(disasm_instr_test)
add_comgr_test(disasm_options_test)
add_comgr_test(metadata_tp_test)
add_comgr_test(metadata_yaml_test)
add_comgr_test(metadata_msgpack_test)
# FIXME: Add this test again when upstream LLVM emits kernel symbols to .dynsym
#add_comgr_test(symbols_test)
# FIXME: Add this test again when upstream LLVM emits kernel symbols to .dynsym
#add_comgr_test(symbols_iterate_test)
add_comgr_test(compile_test)
add_comgr_test(compile_minimal_test)
add_comgr_test(compile_log_test)
if (AMDDeviceLibs_FOUND)
  add_comgr_test(compile_device_libs_test)
endif()
add_comgr_test(assemble_test)
add_comgr_test(link_test)
add_comgr_test(isa_name_parsing_test)
add_comgr_test(get_data_isa_name_test)
add_comgr_test(include_subdirectory_test)
