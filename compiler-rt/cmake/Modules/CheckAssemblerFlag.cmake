# Helper function to find out whether the assembler supports a particular
# command-line flag. You'd like to use the standard check_compiler_flag(), but
# that only supports a fixed list of languages, and ASM isn't one of them. So
# we do it ourselves, by trying to assemble an empty source file.

function(check_assembler_flag outvar flag)
  if(NOT DEFINED "${outvar}")
    if(NOT CMAKE_REQUIRED_QUIET)
      message(CHECK_START "Checking for assembler flag ${flag}")
    endif()

    # Stop try_compile from attempting to link the result of the assembly, so
    # that we don't depend on having a working linker, and also don't have to
    # figure out what special symbol like _start needs to be defined in the
    # test input.
    #
    # This change is made within the dynamic scope of this function, so
    # CMAKE_TRY_COMPILE_TARGET_TYPE will be restored to its previous value on
    # return.
    set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

    # Try to assemble an empty file with a .S name, using the provided flag.
    set(asm_source_file
      ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CheckAssemblerFlag.S)
    write_file(${asm_source_file} "")
    try_compile(${outvar}
      ${CMAKE_BINARY_DIR}
      SOURCES ${asm_source_file}
      COMPILE_DEFINITIONS ${flag})

    if(NOT CMAKE_REQUIRED_QUIET)
      if(${outvar})
        message(CHECK_PASS "Accepted")
      else()
        message(CHECK_FAIL "Not accepted")
      endif()
    endif()
  endif()
endfunction()
