set(CMAKE_CLC_OUTPUT_EXTENSION .o)
set(CMAKE_INCLUDE_FLAG_CLC "-I")

set(CMAKE_CLC_DEPFILE_FORMAT gcc)
set(CMAKE_DEPFILE_FLAGS_CLC "-MD -MT <DEP_TARGET> -MF <DEP_FILE>")

cmake_initialize_per_config_variable(CMAKE_CLC_FLAGS "Flags used by the CLC compiler")

if(NOT CMAKE_CLC_COMPILE_OBJECT)
  set(CMAKE_CLC_COMPILE_OBJECT
    "<CMAKE_CLC_COMPILER> -x cl <DEFINES> <INCLUDES> <FLAGS> -c -o <OBJECT> <SOURCE>")
endif()

# Finds a required LLVM tool by searching the CLC compiler directory first.
function(find_llvm_tool name out_var)
  cmake_path(GET CMAKE_CLC_COMPILER PARENT_PATH llvm_bin_dir)
  find_program(${out_var}
    NAMES ${name}
    HINTS "${llvm_bin_dir}"
    DOC "libclc: path to the ${name} tool"
  )
  if(NOT ${out_var})
    message(FATAL_ERROR "${name} not found for libclc build.")
  endif()
endfunction()

find_llvm_tool(llvm-ar CLC_AR)
find_llvm_tool(llvm-ranlib CLC_RANLIB)

if(NOT DEFINED CMAKE_CLC_ARCHIVE_CREATE)
  set(CMAKE_CLC_ARCHIVE_CREATE "${CLC_AR} qc <TARGET> <OBJECTS>")
endif()

if(NOT DEFINED CMAKE_CLC_ARCHIVE_APPEND)
  set(CMAKE_CLC_ARCHIVE_APPEND "${CLC_AR} q <TARGET> <OBJECTS>")
endif()

if(NOT DEFINED CMAKE_CLC_ARCHIVE_FINISH)
  set(CMAKE_CLC_ARCHIVE_FINISH "${CLC_RANLIB} <TARGET>")
endif()

set(CMAKE_CLC_USE_LINKER_INFORMATION FALSE)

set(CMAKE_CLC_INFORMATION_LOADED 1)
