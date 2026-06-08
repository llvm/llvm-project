option(LLVM_ENABLE_DOXYGEN "Use Doxygen to generate API documentation." OFF)


# available programs checks
function(llvm_find_program name)
  string(TOUPPER ${name} NAME)
  string(REGEX REPLACE "\\." "_" NAME ${NAME})

  find_program(LLVM_PATH_${NAME} NAMES ${ARGV})
  mark_as_advanced(LLVM_PATH_${NAME})
  if(LLVM_PATH_${NAME})
    set(HAVE_${NAME} 1 CACHE INTERNAL "Is ${name} available ?")
    mark_as_advanced(HAVE_${NAME})
  else(LLVM_PATH_${NAME})
    set(HAVE_${NAME} "" CACHE INTERNAL "Is ${name} available ?")
  endif(LLVM_PATH_${NAME})
endfunction()

if (LLVM_ENABLE_DOXYGEN)
  llvm_find_program(dot)
  find_package(Doxygen REQUIRED)
  message(STATUS "Doxygen enabled (${DOXYGEN_VERSION}).")

  if (DOXYGEN_FOUND)
    # If we find doxygen and we want to enable doxygen by default create a
    # global aggregate doxygen target for generating llvm and any/all
    # subprojects doxygen documentation.
    if (LLVM_BUILD_DOCS)
      add_custom_target(doxygen ALL)
    endif()

    option(LLVM_DOXYGEN_EXTERNAL_SEARCH "Enable doxygen external search." OFF)
    if (LLVM_DOXYGEN_EXTERNAL_SEARCH)
      set(LLVM_DOXYGEN_SEARCHENGINE_URL "" CACHE STRING "URL to use for external search.")
      set(LLVM_DOXYGEN_SEARCH_MAPPINGS "" CACHE STRING "Doxygen Search Mappings")
    endif()
  endif()

  # Uses all CPUs for doxygen >= 1.17 where multi-threading is fast, and
  # single-threaded (1) for older versions where multi-threading is slower than
  # single-threaded.
  if (DOXYGEN_VERSION VERSION_GREATER_EQUAL "1.17")
    set(DOXYGEN_NUM_PROC_THREADS 0)
  else()
    set(DOXYGEN_NUM_PROC_THREADS 1)
  endif()
else()
  message(STATUS "Doxygen disabled.")
endif()

# Configure doxygen.cfg.in -> doxygen.cfg, using a macro here to ensure that
# DOXYGEN_NUM_PROC_THREADS has been set.
macro(llvm_configure_doxygen)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doxygen.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/doxygen.cfg @ONLY)
endmacro()
