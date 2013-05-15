# If we are not building as a part of LLVM, build LLDB as an
# standalone project, using LLVM as an external library:
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(lldb)
  cmake_minimum_required(VERSION 2.8)

  set(LLDB_PATH_TO_LLVM_SOURCE "" CACHE PATH
    "Path to LLVM source code. Not necessary if using an installed LLVM.")
  set(LLDB_PATH_TO_LLVM_BUILD "" CACHE PATH
    "Path to the directory where LLVM was built or installed.")
    
  set(LLDB_PATH_TO_CLANG_SOURCE "" CACHE PATH
    "Path to Clang source code. Not necessary if using an installed Clang.")
  set(LLDB_PATH_TO_CLANG_BUILD "" CACHE PATH
    "Path to the directory where Clang was built or installed.")    

  set(LLDB_DISABLE_PYTHON 1 BOOL "Disables the Python scripting integration.")

  if (LLDB_PATH_TO_LLVM_SOURCE)
    if (NOT EXISTS "${LLDB_PATH_TO_LLVM_SOURCE}/cmake/config-ix.cmake")
      message(FATAL_ERROR "Please set LLDB_PATH_TO_LLVM_SOURCE to the root "
              "directory of LLVM source code.")
    else()
      get_filename_component(LLVM_MAIN_SRC_DIR ${LLDB_PATH_TO_LLVM_SOURCE}
                             ABSOLUTE)
      list(APPEND CMAKE_MODULE_PATH "${LLVM_MAIN_SRC_DIR}/cmake/modules")
    endif()
  endif()
  
  if (LLDB_PATH_TO_CLANG_SOURCE)
      get_filename_component(CLANG_MAIN_SRC_DIR ${LLDB_PATH_TO_CLANG_SOURCE}
                             ABSOLUTE)
  endif()  

  list(APPEND CMAKE_MODULE_PATH "${LLDB_PATH_TO_LLVM_BUILD}/share/llvm/cmake")

  get_filename_component(PATH_TO_LLVM_BUILD ${LLDB_PATH_TO_LLVM_BUILD}
                         ABSOLUTE)
                         
  get_filename_component(PATH_TO_CLANG_BUILD ${LLDB_PATH_TO_CLANG_BUILD}
                         ABSOLUTE)                         
                         
  include(AddLLVM)
  include("${LLDB_PATH_TO_LLVM_BUILD}/share/llvm/cmake/LLVMConfig.cmake")
  include(HandleLLVMOptions)

  set(PACKAGE_VERSION "${LLVM_PACKAGE_VERSION}")

  set(LLVM_MAIN_INCLUDE_DIR "${LLVM_MAIN_SRC_DIR}/include")
  set(LLVM_BINARY_DIR ${CMAKE_BINARY_DIR})
  
  set(CLANG_MAIN_INCLUDE_DIR "${CLANG_MAIN_SRC_DIR}/include")

  set(CMAKE_INCLUDE_CURRENT_DIR ON)
  include_directories("${PATH_TO_LLVM_BUILD}/include"
                      "${LLVM_MAIN_INCLUDE_DIR}"
                      "${PATH_TO_CLANG_BUILD}/include"
                      "${CLANG_MAIN_INCLUDE_DIR}"
                      "${CMAKE_CURRENT_SOURCE_DIR}/source")
  link_directories("${PATH_TO_LLVM_BUILD}/lib"
                   "${PATH_TO_CLANG_BUILD}/lib")

  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

  set(LLDB_BUILT_STANDALONE 1)
  
  if (LLDB_DISABLE_PYTHON)
    add_definitions( -DLLDB_DISABLE_PYTHON )
  endif() 
endif()

macro(add_lldb_definitions)
  # We don't want no semicolons on LLDB_DEFINITIONS:
  foreach(arg ${ARGN})
    set(LLDB_DEFINITIONS "${LLVM_DEFINITIONS} ${arg}")
  endforeach(arg)
  add_definitions( ${ARGN} )
endmacro(add_lldb_definitions)

include_directories(/usr/include/python2.7)
include_directories(../clang/include)
include_directories("${CMAKE_CURRENT_BINARY_DIR}/../clang/include")

# lldb requires c++11 to build. Make sure that we have a compiler and standard
# library combination that can do that.
if (MSVC11)
  # Do nothing, we're good.
elseif (NOT MSVC)
  # gcc and clang require the -std=c++0x or -std=c++11 flag.
  if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU" OR
      "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    if (NOT ("${CMAKE_CXX_FLAGS}" MATCHES "-std=c\\+\\+0x" OR
             "${CMAKE_CXX_FLAGS}" MATCHES "-std=gnu\\+\\+0x" OR
             "${CMAKE_CXX_FLAGS}" MATCHES "-std=c\\+\\+11" OR
             "${CMAKE_CXX_FLAGS}" MATCHES "-std=gnu\\+\\+11"))
      if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS "4.7")
          set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
        else()
          set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
        endif()
      else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
      endif()
    endif()
  endif()
else()
  message(FATAL_ERROR "The selected compiler does not support c++11 which is "
          "required to build lldb.")
endif()

# Disable Clang warnings
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  add_lldb_definitions(
    -Wno-deprecated-declarations # Suppress "deprecated auto_ptr" warnings
  )
endif()

# Disable MSVC warnings
if( MSVC )
  add_lldb_definitions(
    -wd4018 # Suppress 'warning C4018: '>=' : signed/unsigned mismatch'
    -wd4068 # Suppress 'warning C4068: unknown pragma'
    -wd4150 # Suppress 'warning C4150: deletion of pointer to incomplete type'
    -wd4521 # Suppress 'warning C4521: 'type' : multiple copy constructors specified'
  )
endif() 

# If building on a 32-bit system, make sure off_t can store offsets > 2GB
if( CMAKE_SIZEOF_VOID_P EQUAL 4 )
  add_lldb_definitions(
    -D_LARGEFILE_SOURCE
    -D_FILE_OFFSET_BITS=64
    )
endif()

set(LLDB_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(LLDB_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "In-source builds are not allowed. CMake would overwrite "
"the makefiles distributed with LLDB. Please create a directory and run cmake "
"from there, passing the path to this source directory as the last argument. "
"This process created the file `CMakeCache.txt' and the directory "
"`CMakeFiles'. Please delete them.")
endif()

macro(add_lldb_library name)
  llvm_process_sources(srcs ${ARGN})
  if (MSVC_IDE OR XCODE)
    string(REGEX MATCHALL "/[^/]+" split_path ${CMAKE_CURRENT_SOURCE_DIR})
    list(GET split_path -1 dir)
    file(GLOB_RECURSE headers
      ../../include/lldb${dir}/*.h)
    set(srcs ${srcs} ${headers})
  endif()
  if (MODULE)
    set(libkind MODULE)
  elseif (SHARED_LIBRARY)
    set(libkind SHARED)
  else()
    set(libkind STATIC)
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
  add_library(${name} ${libkind} ${srcs})
  #if (LLVM_COMMON_DEPENDS)
  ##add_dependencies(${name} ${LLVM_COMMON_DEPENDS})
  #endif()

  if(LLDB_USED_LIBS)
    if (CMAKE_SYSTEM_NAME MATCHES "Linux")
      target_link_libraries(${name} -Wl,--start-group ${LLDB_USED_LIBS} -Wl,--end-group)
    else()
      target_link_libraries(${name} ${LLDB_USED_LIBS})
    endif()
  endif()
  target_link_libraries(${name} ${CLANG_USED_LIBS})
  target_link_libraries(${name} ${LLVM_USED_LIBS})
  llvm_config(${name} ${LLVM_LINK_COMPONENTS})
  target_link_libraries(${name} ${LLVM_COMMON_LIBS})
  link_system_libs(${name})
  if (LLVM_COMMON_DEPENDS)
    add_dependencies(${name} ${LLVM_COMMON_DEPENDS})
  endif()

  # Hack: only some LLDB libraries depend on the clang autogenerated headers,
  # but it is simple enough to make all of LLDB depend on some of those
  # headers without negatively impacting much of anything.
  set (LLDB_DEPENDENCIES
    libclang
    )
  add_dependencies(${name} ${LLDB_DEPENDENCIES})

  install(TARGETS ${name}
    LIBRARY DESTINATION lib${LLVM_LIBDIR_SUFFIX}
    ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX})
  set_target_properties(${name} PROPERTIES FOLDER "lldb libraries")
endmacro(add_lldb_library)

macro(add_lldb_executable name)
  #add_llvm_executable(${name} ${ARGN})
  llvm_process_sources( ALL_FILES ${ARGN} )
  add_executable(${name} ${ALL_FILES})
  #target_link_libraries(${name} ${CLANG_USED_LIBS})
  #llvm_config( ${name} ${LLVM_LINK_COMPONENTS} )
  #link_system_libs( ${name} )
  #if (LLVM_COMMON_DEPENDS)
  #add_dependencies(${name} ${LLVM_COMMON_DEPENDS})
  #endif()
  set_target_properties(${name} PROPERTIES FOLDER "lldb executables")
endmacro(add_lldb_executable)

include_directories(BEFORE
  ${CMAKE_CURRENT_BINARY_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  )

install(DIRECTORY include/
  DESTINATION include
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN ".svn" EXCLUDE
  )


# Find libraries or frameworks that may be needed
if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
  find_library(CARBON_LIBRARY Carbon)
  find_library(FOUNDATION_LIBRARY Foundation)
  find_library(CORE_FOUNDATION_LIBRARY CoreFoundation)
  find_library(CORE_SERVICES_LIBRARY CoreServices)
  find_library(SECURITY_LIBRARY Security)
  find_library(DEBUG_SYMBOLS_LIBRARY DebugSymbols PATHS "/System/Library/PrivateFrameworks")

  set(LIBXML2_INCLUDE_DIR "/usr/include/libxml2")
  list(APPEND system_libs xml2)
  list(APPEND system_libs ${CARBON_LIBRARY} ${FOUNDATION_LIBRARY}
  ${CORE_FOUNDATION_LIBRARY} ${CORE_SERVICES_LIBRARY} ${SECURITY_LIBRARY}
  ${DEBUG_SYMBOLS_LIBRARY})
endif()

#add_subdirectory(include)
add_subdirectory(scripts)
add_subdirectory(source)
add_subdirectory(test)
add_subdirectory(tools)
