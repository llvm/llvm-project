set(TEST_INPUT_BINARIES)
# Create target ${name} which depends on a clang command to compile ${input} to
# ${output}, with any additional arguments from ${ARGN}, and add it to the
# TEST_INPUT_BINARIES target list.
macro(add_test_input_binary name input output)
  add_custom_command(
    OUTPUT "${output}"
    COMMAND "$<TARGET_FILE:clang>" --target=amdgcn-amd-amdhsa -mcpu=gfx803 -nogpulib
    ${ARGN} "${CMAKE_CURRENT_SOURCE_DIR}/${input}" -o "${output}"
    VERBATIM
    DEPENDS clang lld "${input}")
  add_custom_target("${name}"
    DEPENDS "${output}"
    SOURCES "${input}")
  list(APPEND TEST_INPUT_BINARIES "${name}")
endmacro()

# Creates target ${name} and output ${output} by linking two object files.
# ${target1} and ${target2} should refer to the targets created in the above
# add_test_input_binary() macro, and ${input1} and ${input2} should refer
# to the associated object files built by the same macro.
macro(add_test_shared_binary name target1 input1 target2 input2 output)
  add_custom_command(
    OUTPUT "${output}"
    COMMAND "$<TARGET_FILE:clang>" --target=amdgcn-amd-amdhsa
      "${input1}" "${input2}" -o "${output}"
    VERBATIM
    DEPENDS clang lld ${target1} ${target2})
  add_custom_target("${name}"
    DEPENDS "${output}"
    SOURCES "${input1}" "${input2}")
  list(APPEND TEST_INPUT_BINARIES "${name}")
endmacro()

add_test_input_binary(reloc1 reloc1.cl reloc1.o -c -mcode-object-version=2)
add_test_input_binary(reloc2 reloc2.cl reloc2.o -c -mcode-object-version=2)
add_test_input_binary(reloc-asm reloc-asm.s reloc-asm.o -c -mcode-object-version=2)
add_test_input_binary(shared shared.cl shared.so -mcode-object-version=2)
add_test_input_binary(shared-v3 shared.cl shared-v3.so -mcode-object-version=3)

add_test_input_binary(source1-v2 source1-v2.s source1-v2.o -c -mcode-object-version=2)
add_test_input_binary(source2-v2 source2-v2.s source2-v2.o -c -mcode-object-version=2)
add_test_input_binary(source3-v2 source3-v2.s source3-v2.o -c -mcode-object-version=2)
add_test_input_binary(source4-v2 source4-v2.s source4-v2.o -c -mcode-object-version=2)

add_test_input_binary(source1-v3 source1-v3.s source1-v3.o -c -mcode-object-version=3)
add_test_input_binary(source2-v3 source2-v3.s source2-v3.o -c -mcode-object-version=3)
add_test_input_binary(source3-v3 source3-v3.s source3-v3.o -c -mcode-object-version=3)
add_test_input_binary(source4-v3 source4-v3.s source4-v3.o -c -mcode-object-version=3)

add_test_shared_binary(shared12-v2 source1-v2 source1-v2.o source2-v2 source2-v2.o shared12-v2.so)
add_test_shared_binary(shared14-v2 source1-v2 source1-v2.o source4-v2 source4-v2.o shared14-v2.so)
add_test_shared_binary(shared23-v2 source2-v2 source2-v2.o source3-v2 source3-v2.o shared23-v2.so)

add_test_shared_binary(shared12-v3 source1-v3 source1-v3.o source2-v3 source2-v3.o shared12-v3.so)
add_test_shared_binary(shared14-v3 source1-v3 source1-v3.o source4-v3 source4-v3.o shared14-v3.so)
add_test_shared_binary(shared23-v3 source2-v3 source2-v3.o source3-v3 source3-v3.o shared23-v3.so)

configure_file("source1.cl" "source1.cl" COPYONLY)
configure_file("source1-device-libs.cl" "source1-device-libs.cl" COPYONLY)
configure_file("source2.cl" "source2.cl" COPYONLY)
configure_file("include-a.h" "include-a.h" COPYONLY)
configure_file("source1.s" "source1.s" COPYONLY)
configure_file("shared.cl" "shared.cl" COPYONLY)
configure_file("source1.hip" "source1.hip" COPYONLY)
configure_file("source2.hip" "source2.hip" COPYONLY)

configure_file("source1-v2.s" "source1-v2.s" COPYONLY)
configure_file("source2-v2.s" "source2-v2.s" COPYONLY)
configure_file("source3-v2.s" "source3-v2.s" COPYONLY)
configure_file("source4-v2.s" "source4-v2.s" COPYONLY)
configure_file("source1-v3.s" "source1-v3.s" COPYONLY)
configure_file("source2-v3.s" "source2-v3.s" COPYONLY)
configure_file("source3-v3.s" "source3-v3.s" COPYONLY)
configure_file("source4-v3.s" "source4-v3.s" COPYONLY)

# Create executable ${name} and accompanying test ${name} built from
# test/${name}.cl
macro(add_comgr_test name)
  set(test_name "comgr_${name}")
  add_executable("${name}" "${name}.c")
  set_target_properties("${name}" PROPERTIES
    C_STANDARD 99
    C_STANDARD_REQUIRED Yes
    C_EXTENSIONS No)
  target_compile_definitions("${name}"
    PRIVATE -DTEST_OBJ_DIR=\"${CMAKE_CURRENT_BINARY_DIR}\")
if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  target_compile_definitions("${name}"
    PRIVATE -D_CRT_SECURE_NO_WARNINGS)
endif()

if (DEFINED HIP_COMPILER AND "${HIP_COMPILER}" STREQUAL "clang")
  target_compile_definitions("${name}"
    PRIVATE -DHIP_COMPILER=${HIP_COMPILER})
endif()
  target_link_libraries("${name}"
    amd_comgr)
  add_dependencies("${name}"
    ${TEST_INPUT_BINARIES})
  add_test(NAME ${test_name}
    COMMAND "${name}")
  add_dependencies(check-comgr ${name})
  # Windows binaries have no equivalent to RPATH, so we must set their PATH to
  # include the .lib/.dll directory.
  if (NOT(UNIX))
    set_tests_properties(${test_name}
      PROPERTIES ENVIRONMENT "PATH=$<TARGET_LINKER_FILE_DIR:amd_comgr>")
  endif()
endmacro()

find_package(hip CONFIG PATHS ${ROCM_INSTALL_PATH}/hip QUIET)

add_comgr_test(data_test)
add_comgr_test(disasm_llvm_reloc_test)
add_comgr_test(disasm_llvm_so_test)
add_comgr_test(disasm_instr_test)
add_comgr_test(disasm_options_test)
add_comgr_test(metadata_tp_test)
add_comgr_test(metadata_yaml_test)
add_comgr_test(metadata_msgpack_test)
add_comgr_test(metadata_merge_test)
add_comgr_test(symbols_test)
add_comgr_test(symbols_iterate_test)
add_comgr_test(compile_test)
add_comgr_test(compile_minimal_test)
add_comgr_test(compile_log_test)
add_comgr_test(compile_log_remarks_test)
add_comgr_test(compile_device_libs_test)
add_comgr_test(compile_source_with_device_libs_to_bc_test)
add_comgr_test(assemble_test)
add_comgr_test(link_test)
add_comgr_test(isa_name_parsing_test)
add_comgr_test(get_data_isa_name_test)
add_comgr_test(include_subdirectory_test)
add_comgr_test(options_test)
add_comgr_test(demangle_test)
add_comgr_test(file_map)
add_comgr_test(lookup_code_object_test)
add_comgr_test(symbolize_test)

# Test : Compile HIP tests only if HIP-Clang is installed.
if (DEFINED HIP_COMPILER AND "${HIP_COMPILER}" STREQUAL "clang")
  add_comgr_test(compile_hip_test)
  add_comgr_test(compile_hip_test_in_process)
endif()
