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
  message(STATUS "Doxygen enabled.")
  llvm_find_program(dot)
  find_package(Doxygen REQUIRED)

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
else()
  message(STATUS "Doxygen disabled.")
endif()
