set(TEST_INPUT_BINARIES)
set(TEST_INPUT_BITCODES)
set(TEST_INPUT_BUNDLES)
set(TEST_INPUT_ARCHIVES)
# Create target ${name} which depends on a clang command to compile ${input} to
# ${output}, with any additional arguments from ${ARGN}, and add it to the
# TEST_INPUT_BINARIES target list.
macro(add_test_input_binary name input output)
  add_custom_command(
    OUTPUT "${output}"
    COMMAND "$<TARGET_FILE:clang>" --target=amdgcn-amd-amdhsa -mcpu=gfx900 -nogpulib
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

# Creates target ${name} which depends on a clang command to compile ${input} to
# ${output}, with any additional arguments from ${ARGN}, and add it to the
# TEST_INPUT_BITCODES target list.
macro(add_test_input_bitcode name input output)
  add_custom_command(
    OUTPUT "${output}"
    COMMAND "$<TARGET_FILE:clang>" -c -emit-llvm -target amdgcn-amd-amdhsa
    -mcpu=gfx900
    ${ARGN} "${CMAKE_CURRENT_SOURCE_DIR}/${input}"
    -o "${output}"
    VERBATIM
    DEPENDS clang lld "${input}")
  add_custom_target("${name}"
    DEPENDS "${output}"
    SOURCES "${input}")
  list(APPEND TEST_INPUT_BITCODES "${name}")
endmacro()

# Creates target ${name} which depends on a clang command to compile ${input} to
# ${output}, with any additional arguments from ${ARGN}, and add it to the
# TEST_INPUT_BUNDLES target list.
macro(add_test_input_bundle name input output)
  add_custom_command(
    OUTPUT "${output}"
    COMMAND "$<TARGET_FILE:clang>" -c --offload-arch=gfx900 -emit-llvm -fgpu-rdc
    --gpu-bundle-output ${ARGN} "${CMAKE_CURRENT_SOURCE_DIR}/${input}"
    -o "${output}"
    VERBATIM
    DEPENDS clang lld "${input}")
  add_custom_target("${name}"
    DEPENDS "${output}"
    SOURCES "${input}")
  list(APPEND TEST_INPUT_BUNDLES "${name}")
endmacro()

# Creates target ${name} and output ${output} by archiving a file.
# ${target} should refer to the a target created in the above
# add_test_input_bitcode() macro, and ${input} should refer
# to the associated bitcode file built by the same macro.
macro(add_test_archive name target input output)
  add_custom_command(
    OUTPUT "${output}"
    COMMAND "$<TARGET_FILE:llvm-ar>" rc "${output}" "${input}"
    VERBATIM
    DEPENDS clang lld ${target})
  add_custom_target("${name}"
    DEPENDS "${output}"
    SOURCES "${input}")
  list(APPEND TEST_INPUT_ARCHIVES "${name}")
endmacro()

add_test_input_binary(reloc1 source/reloc1.cl source/reloc1.o -c -mcode-object-version=4)
add_test_input_binary(reloc2 source/reloc2.cl source/reloc2.o -c -mcode-object-version=4)
add_test_input_binary(reloc-asm source/reloc-asm.s source/reloc-asm.o -c -mcode-object-version=4)
add_test_input_binary(shared source/shared.cl source/shared.so -mcode-object-version=4)
add_test_input_binary(shared-debug source/shared.cl source/shared-debug.so -g -mcode-object-version=4)

configure_file("source/source1.cl" "source/source1.cl" COPYONLY)
configure_file("source/source2.cl" "source/source2.cl" COPYONLY)
configure_file("source/nested-kernel1.cl" "source/nested-kernel1.cl" COPYONLY)
configure_file("source/nested-kernel2.cl" "source/nested-kernel2.cl" COPYONLY)
configure_file("source/shared.cl" "source/shared.cl" COPYONLY)
configure_file("source/device_libs.cl" "source/device_libs.cl" COPYONLY)
configure_file("source/include-macro.h" "source/include-macro.h" COPYONLY)
configure_file("source/include-nested.h" "source/include-nested.h" COPYONLY)
configure_file("source/source1.s" "source/source1.s" COPYONLY)
configure_file("source/source1.hip" "source/source1.hip" COPYONLY)
configure_file("source/source2.hip" "source/source2.hip" COPYONLY)
configure_file("source/name-expression.hip" "source/name-expression.hip" COPYONLY)
configure_file("source/rocm56slice.b" "source/rocm56slice.b" COPYONLY)
configure_file("source/rocm57slice.b" "source/rocm57slice.b" COPYONLY)

configure_file("source/square.hip" "source/square.hip" COPYONLY)
configure_file("source/double.hip" "source/double.hip" COPYONLY)
configure_file("source/cube.hip" "source/cube.hip" COPYONLY)

# We no longer support emission of code object v2/v3. The runtime however
# can still load them so we need to test them using prebuilt binaries.
configure_file("source/legacy/shared-v2.so" "source/shared-v2.so" COPYONLY)
configure_file("source/legacy/shared12-v2.so" "source/shared12-v2.so" COPYONLY)
configure_file("source/legacy/shared14-v2.so" "source/shared14-v2.so" COPYONLY)
configure_file("source/legacy/shared23-v2.so" "source/shared23-v2.so" COPYONLY)
configure_file("source/legacy/source1-v2.o" "source/source1-v2.o" COPYONLY)
configure_file("source/legacy/source2-v2.o" "source/source2-v2.o" COPYONLY)
configure_file("source/legacy/source3-v2.o" "source/source3-v2.o" COPYONLY)
configure_file("source/legacy/source4-v2.o" "source/source4-v2.o" COPYONLY)

configure_file("source/legacy/shared-v3.so" "source/shared-v3.so" COPYONLY)
configure_file("source/legacy/shared12-v3.so" "source/shared12-v3.so" COPYONLY)
configure_file("source/legacy/shared14-v3.so" "source/shared14-v3.so" COPYONLY)
configure_file("source/legacy/shared23-v3.so" "source/shared23-v3.so" COPYONLY)
configure_file("source/legacy/source1-v3.o" "source/source1-v3.o" COPYONLY)
configure_file("source/legacy/source2-v3.o" "source/source2-v3.o" COPYONLY)
configure_file("source/legacy/source3-v3.o" "source/source3-v3.o" COPYONLY)
configure_file("source/legacy/source4-v3.o" "source/source4-v3.o" COPYONLY)

# Creates executable ${name} and accompanying test ${name} built from
# test/${name}.cl
macro(add_comgr_test name lang)
  set(test_name "comgr_${name}")
  add_executable("${name}" "${name}.${lang}")
  set_target_properties("${name}" PROPERTIES
    C_STANDARD 99
    C_STANDARD_REQUIRED Yes
    C_EXTENSIONS No)
  target_compile_definitions("${name}"
    PRIVATE -DTEST_OBJ_DIR=\"${CMAKE_CURRENT_BINARY_DIR}/source\")
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
    ${TEST_INPUT_BINARIES};${TEST_INPUT_BITCODES};${TEST_INPUT_BUNDLES};
    ${TEST_INPUT_ARCHIVES})
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

add_comgr_test(data_test c)
add_comgr_test(disasm_llvm_reloc_test c)
add_comgr_test(disasm_llvm_so_test c)
add_comgr_test(disasm_instr_test c)
add_comgr_test(disasm_options_test c)
add_comgr_test(metadata_tp_test c)
add_comgr_test(metadata_yaml_test c)
add_comgr_test(metadata_msgpack_test c)
add_comgr_test(metadata_merge_test c)
add_comgr_test(symbols_test c)
add_comgr_test(symbols_iterate_test c)
add_comgr_test(compile_test c)
add_comgr_test(compile_minimal_test c)
add_comgr_test(compile_log_test c)
add_comgr_test(compile_log_remarks_test c)
add_comgr_test(compile_device_libs_test c)
add_comgr_test(compile_source_with_device_libs_to_bc_test c)
add_comgr_test(assemble_test c)
add_comgr_test(link_test c)
add_comgr_test(isa_name_parsing_test c)
add_comgr_test(get_data_isa_name_test c)
add_comgr_test(include_subdirectory_test c)
add_comgr_test(options_test c)
add_comgr_test(demangle_test c)
add_comgr_test(file_map c)
add_comgr_test(lookup_code_object_test c)
add_comgr_test(symbolize_test c)
add_comgr_test(mangled_names_test c)
add_comgr_test(multithread_test cpp)
add_comgr_test(name_expression_map_test c)
add_comgr_test(nested_kernel_test c)
add_comgr_test(map_elf_virtual_address_test c)

# Test : Compile HIP tests only if HIP-Clang is installed.
if (DEFINED HIP_COMPILER AND "${HIP_COMPILER}" STREQUAL "clang")
  add_test_input_bundle(cube source/cube.hip source/cube.bc)
  add_test_input_bundle(square source/square.hip source/square.bc)
  add_test_input_bundle(double source/double.hip source/double.bc)

  add_test_archive(cube_archive cube source/cube.bc source/cube.a)

  add_comgr_test(compile_hip_test c)
  add_comgr_test(compile_hip_test_in_process c)
  add_comgr_test(unbundle_hip_test c)
  add_comgr_test(compile_hip_to_relocatable c)
  add_comgr_test(mangled_names_hip_test c)
endif()
