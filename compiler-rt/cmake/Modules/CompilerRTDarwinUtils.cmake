include(CMakeParseArguments)

# On OS X SDKs can be installed anywhere on the base system and xcode-select can
# set the default Xcode to use. This function finds the SDKs that are present in
# the current Xcode.
function(find_darwin_sdk_dir var sdk_name)
  # Let's first try the internal SDK, otherwise use the public SDK.
  execute_process(
    COMMAND xcodebuild -version -sdk ${sdk_name}.internal Path
    OUTPUT_VARIABLE var_internal
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_FILE /dev/null
  )
  if("" STREQUAL "${var_internal}")
    execute_process(
      COMMAND xcodebuild -version -sdk ${sdk_name} Path
      OUTPUT_VARIABLE var_internal
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_FILE /dev/null
    )
  else()
    set(${var}_INTERNAL ${var_internal} PARENT_SCOPE)
  endif()
  set(${var} ${var_internal} PARENT_SCOPE)
endfunction()

# There isn't a clear mapping of what architectures are supported with a given
# target platform, but ld's version output does list the architectures it can
# link for.
function(darwin_get_toolchain_supported_archs output_var)
  execute_process(
    COMMAND ld -v
    ERROR_VARIABLE LINKER_VERSION)

  string(REGEX MATCH "configured to support archs: ([^\n]+)"
         ARCHES_MATCHED "${LINKER_VERSION}")
  if(ARCHES_MATCHED)
    set(ARCHES "${CMAKE_MATCH_1}")
    message(STATUS "Got ld supported ARCHES: ${ARCHES}")
    string(REPLACE " " ";" ARCHES ${ARCHES})
  else()
    # If auto-detecting fails, fall back to a default set
    message(WARNING "Detecting supported architectures from 'ld -v' failed. Returning default set.")
    set(ARCHES "i386;x86_64;armv7;armv7s;arm64")
  endif()
  
  set(${output_var} ${ARCHES} PARENT_SCOPE)
endfunction()

# This function takes an OS and a list of architectures and identifies the
# subset of the architectures list that the installed toolchain can target.
function(darwin_test_archs os valid_archs)
  if(${valid_archs})
    message(STATUS "Using cached valid architectures for ${os}.")
    return()
  endif()

  set(archs ${ARGN})
  if(NOT TEST_COMPILE_ONLY)
    message(STATUS "Finding valid architectures for ${os}...")
    set(SIMPLE_C ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/src.c)
    file(WRITE ${SIMPLE_C} "#include <stdio.h>\nint main() { printf(__FILE__); return 0; }\n")
  
    set(os_linker_flags)
    foreach(flag ${DARWIN_${os}_LINKFLAGS})
      set(os_linker_flags "${os_linker_flags} ${flag}")
    endforeach()
  endif()

  # The simple program will build for x86_64h on the simulator because it is 
  # compatible with x86_64 libraries (mostly), but since x86_64h isn't actually
  # a valid or useful architecture for the iOS simulator we should drop it.
  if(${os} MATCHES "^(iossim|tvossim|watchossim)$")
    list(REMOVE_ITEM archs "x86_64h")
  endif()

  set(working_archs)
  foreach(arch ${archs})
   
    set(arch_linker_flags "-arch ${arch} ${os_linker_flags}")
    if(TEST_COMPILE_ONLY)
      try_compile_only(CAN_TARGET_${os}_${arch} -v -arch ${arch} ${DARWIN_${os}_CFLAGS})
    else()
      try_compile(CAN_TARGET_${os}_${arch} ${CMAKE_BINARY_DIR} ${SIMPLE_C}
                  COMPILE_DEFINITIONS "-v -arch ${arch}" ${DARWIN_${os}_CFLAGS}
                  CMAKE_FLAGS "-DCMAKE_EXE_LINKER_FLAGS=${arch_linker_flags}"
                  OUTPUT_VARIABLE TEST_OUTPUT)
    endif()
    if(${CAN_TARGET_${os}_${arch}})
      list(APPEND working_archs ${arch})
    else()
      file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeError.log
        "Testing compiler for supporting ${os}-${arch}:\n"
        "${TEST_OUTPUT}\n")
    endif()
  endforeach()
  set(${valid_archs} ${working_archs}
    CACHE STRING "List of valid architectures for platform ${os}.")
endfunction()

# This function checks the host cpusubtype to see if it is post-haswell. Haswell
# and later machines can run x86_64h binaries. Haswell is cpusubtype 8.
function(darwin_filter_host_archs input output)
  list_intersect(tmp_var DARWIN_osx_ARCHS ${input})
  execute_process(
    COMMAND sysctl hw.cpusubtype
    OUTPUT_VARIABLE SUBTYPE)

  string(REGEX MATCH "hw.cpusubtype: ([0-9]*)"
         SUBTYPE_MATCHED "${SUBTYPE}")
  set(HASWELL_SUPPORTED Off)
  if(SUBTYPE_MATCHED)
    if(${CMAKE_MATCH_1} GREATER 7)
      set(HASWELL_SUPPORTED On)
    endif()
  endif()
  if(NOT HASWELL_SUPPORTED)
    list(REMOVE_ITEM tmp_var x86_64h)
  endif()
  set(${output} ${tmp_var} PARENT_SCOPE)
endfunction()

# Read and process the exclude file into a list of symbols
function(darwin_read_list_from_file output_var file)
  if(EXISTS ${file})
    file(READ ${file} EXCLUDES)
    string(REPLACE "\n" ";" EXCLUDES ${EXCLUDES})
    set(${output_var} ${EXCLUDES} PARENT_SCOPE)
  endif()
endfunction()

# this function takes an OS, architecture and minimum version and provides a
# list of builtin functions to exclude
function(darwin_find_excluded_builtins_list output_var)
  cmake_parse_arguments(LIB
    ""
    "OS;ARCH;MIN_VERSION"
    ""
    ${ARGN})

  if(NOT LIB_OS OR NOT LIB_ARCH)
    message(FATAL_ERROR "Must specify OS and ARCH to darwin_find_excluded_builtins_list!")
  endif()

  darwin_read_list_from_file(${LIB_OS}_BUILTINS
    ${DARWIN_EXCLUDE_DIR}/${LIB_OS}.txt)
  darwin_read_list_from_file(${LIB_OS}_${LIB_ARCH}_BASE_BUILTINS
    ${DARWIN_EXCLUDE_DIR}/${LIB_OS}-${LIB_ARCH}.txt)

  if(LIB_MIN_VERSION)
    file(GLOB builtin_lists ${DARWIN_EXCLUDE_DIR}/${LIB_OS}*-${LIB_ARCH}.txt)
    foreach(builtin_list ${builtin_lists})
      string(REGEX MATCH "${LIB_OS}([0-9\\.]*)-${LIB_ARCH}.txt" VERSION_MATCHED "${builtin_list}")
      if (VERSION_MATCHED AND NOT CMAKE_MATCH_1 VERSION_LESS LIB_MIN_VERSION)
        if(NOT smallest_version)
          set(smallest_version ${CMAKE_MATCH_1})
        elseif(CMAKE_MATCH_1 VERSION_LESS smallest_version)
          set(smallest_version ${CMAKE_MATCH_1})
        endif()
      endif()
    endforeach()

    if(smallest_version)
      darwin_read_list_from_file(${LIB_ARCH}_${LIB_OS}_BUILTINS
        ${DARWIN_EXCLUDE_DIR}/${LIB_OS}${smallest_version}-${LIB_ARCH}.txt)
    endif()
  endif()
  
  set(${output_var}
      ${${LIB_ARCH}_${LIB_OS}_BUILTINS}
      ${${LIB_OS}_${LIB_ARCH}_BASE_BUILTINS}
      ${${LIB_OS}_BUILTINS} PARENT_SCOPE)
endfunction()

# adds a single builtin library for a single OS & ARCH
macro(darwin_add_builtin_library name suffix)
  cmake_parse_arguments(LIB
    ""
    "PARENT_TARGET;OS;ARCH"
    "SOURCES;CFLAGS;DEFS"
    ${ARGN})
  set(libname "${name}.${suffix}_${LIB_ARCH}_${LIB_OS}")
  add_library(${libname} STATIC ${LIB_SOURCES})
  if(DARWIN_${LIB_OS}_SYSROOT)
    set(sysroot_flag -isysroot ${DARWIN_${LIB_OS}_SYSROOT})
  endif()
  set_target_compile_flags(${libname}
    ${sysroot_flag}
    ${DARWIN_${LIB_OS}_BUILTIN_MIN_VER_FLAG}
    ${LIB_CFLAGS})
  set_property(TARGET ${libname} APPEND PROPERTY
      COMPILE_DEFINITIONS ${LIB_DEFS})
  set_target_properties(${libname} PROPERTIES
      OUTPUT_NAME ${libname}${COMPILER_RT_OS_SUFFIX})
  set_target_properties(${libname} PROPERTIES
    OSX_ARCHITECTURES ${LIB_ARCH})

  if(LIB_PARENT_TARGET)
    add_dependencies(${LIB_PARENT_TARGET} ${libname})
  endif()

  list(APPEND ${LIB_OS}_${suffix}_libs ${libname})
  list(APPEND ${LIB_OS}_${suffix}_lipo_flags -arch ${arch} $<TARGET_FILE:${libname}>)
endmacro()

function(darwin_lipo_libs name)
  cmake_parse_arguments(LIB
    ""
    "PARENT_TARGET;OUTPUT_DIR;INSTALL_DIR"
    "LIPO_FLAGS;DEPENDS"
    ${ARGN})
  if(LIB_DEPENDS AND LIB_LIPO_FLAGS)
    add_custom_command(OUTPUT ${LIB_OUTPUT_DIR}/lib${name}.a
      COMMAND ${CMAKE_COMMAND} -E make_directory ${LIB_OUTPUT_DIR}
      COMMAND lipo -output
              ${LIB_OUTPUT_DIR}/lib${name}.a
              -create ${LIB_LIPO_FLAGS}
      DEPENDS ${LIB_DEPENDS}
      )
    add_custom_target(${name}
      DEPENDS ${LIB_OUTPUT_DIR}/lib${name}.a)
    add_dependencies(${LIB_PARENT_TARGET} ${name})
    install(FILES ${LIB_OUTPUT_DIR}/lib${name}.a
      DESTINATION ${LIB_INSTALL_DIR})
  else()
    message(WARNING "Not generating lipo target for ${name} because no input libraries exist.")
  endif()
endfunction()

# Filter out generic versions of routines that are re-implemented in
# architecture specific manner.  This prevents multiple definitions of the
# same symbols, making the symbol selection non-deterministic.
function(darwin_filter_builtin_sources output_var exclude_or_include excluded_list)
  if(exclude_or_include STREQUAL "EXCLUDE")
    set(filter_action GREATER)
    set(filter_value -1)
  elseif(exclude_or_include STREQUAL "INCLUDE")
    set(filter_action LESS)
    set(filter_value 0)
  else()
    message(FATAL_ERROR "darwin_filter_builtin_sources called without EXCLUDE|INCLUDE")
  endif()

  set(intermediate ${ARGN})
  foreach (_file ${intermediate})
    get_filename_component(_name_we ${_file} NAME_WE)
    list(FIND ${excluded_list} ${_name_we} _found)
    if(_found ${filter_action} ${filter_value})
      list(REMOVE_ITEM intermediate ${_file})
    elseif(${_file} MATCHES ".*/.*\\.S" OR ${_file} MATCHES ".*/.*\\.c")
      get_filename_component(_name ${_file} NAME)
      string(REPLACE ".S" ".c" _cname "${_name}")
      list(REMOVE_ITEM intermediate ${_cname})
    endif ()
  endforeach ()
  set(${output_var} ${intermediate} PARENT_SCOPE)
endfunction()

function(darwin_add_eprintf_library)
  cmake_parse_arguments(LIB
    ""
    ""
    "CFLAGS"
    ${ARGN})

  add_library(clang_rt.eprintf STATIC eprintf.c)
  set_target_compile_flags(clang_rt.eprintf
    -isysroot ${DARWIN_osx_SYSROOT}
    ${DARWIN_osx_BUILTIN_MIN_VER_FLAG}
    -arch i386
    ${LIB_CFLAGS})
  set_target_properties(clang_rt.eprintf PROPERTIES
      OUTPUT_NAME clang_rt.eprintf${COMPILER_RT_OS_SUFFIX})
  set_target_properties(clang_rt.eprintf PROPERTIES
    OSX_ARCHITECTURES i386)
  add_dependencies(builtins clang_rt.eprintf)
  set_target_properties(clang_rt.eprintf PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY ${COMPILER_RT_LIBRARY_OUTPUT_DIR})
  install(TARGETS clang_rt.eprintf
      ARCHIVE DESTINATION ${COMPILER_RT_LIBRARY_INSTALL_DIR})
endfunction()

# Generates builtin libraries for all operating systems specified in ARGN. Each
# OS library is constructed by lipo-ing together single-architecture libraries.
macro(darwin_add_builtin_libraries)
  set(DARWIN_EXCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Darwin-excludes)

  set(CFLAGS "-fPIC -O3 -fvisibility=hidden -DVISIBILITY_HIDDEN -Wall -fomit-frame-pointer")
  set(CMAKE_C_FLAGS "")
  set(CMAKE_CXX_FLAGS "")
  set(CMAKE_ASM_FLAGS "")

  set(PROFILE_SOURCES ../profile/InstrProfiling 
                      ../profile/InstrProfilingBuffer
                      ../profile/InstrProfilingPlatformDarwin
                      ../profile/InstrProfilingWriter)
  foreach (os ${ARGN})
    list_intersect(DARWIN_BUILTIN_ARCHS DARWIN_${os}_ARCHS BUILTIN_SUPPORTED_ARCH)
    foreach (arch ${DARWIN_BUILTIN_ARCHS})
      darwin_find_excluded_builtins_list(${arch}_${os}_EXCLUDED_BUILTINS
                              OS ${os}
                              ARCH ${arch}
                              MIN_VERSION ${DARWIN_${os}_BUILTIN_MIN_VER})

      darwin_filter_builtin_sources(filtered_sources
        EXCLUDE ${arch}_${os}_EXCLUDED_BUILTINS
        ${${arch}_SOURCES})

      darwin_add_builtin_library(clang_rt builtins
                              OS ${os}
                              ARCH ${arch}
                              SOURCES ${filtered_sources}
                              CFLAGS ${CFLAGS} -arch ${arch}
                              PARENT_TARGET builtins)
    endforeach()

    # Don't build cc_kext libraries for simulator platforms
    if(NOT DARWIN_${os}_SKIP_CC_KEXT)
      foreach (arch ${DARWIN_BUILTIN_ARCHS})
        # By not specifying MIN_VERSION this only reads the OS and OS-arch lists.
        # We don't want to filter out the builtins that are present in libSystem
        # because kexts can't link libSystem.
        darwin_find_excluded_builtins_list(${arch}_${os}_EXCLUDED_BUILTINS
                              OS ${os}
                              ARCH ${arch})

        darwin_filter_builtin_sources(filtered_sources
          EXCLUDE ${arch}_${os}_EXCLUDED_BUILTINS
          ${${arch}_SOURCES})

        # In addition to the builtins cc_kext includes some profile sources
        darwin_add_builtin_library(clang_rt cc_kext
                                OS ${os}
                                ARCH ${arch}
                                SOURCES ${filtered_sources} ${PROFILE_SOURCES}
                                CFLAGS ${CFLAGS} -arch ${arch} -mkernel
                                DEFS KERNEL_USE
                                PARENT_TARGET builtins)
      endforeach()
      set(archive_name clang_rt.cc_kext_${os})
      if(${os} STREQUAL "osx")
        set(archive_name clang_rt.cc_kext)
      endif()
      darwin_lipo_libs(${archive_name}
                      PARENT_TARGET builtins
                      LIPO_FLAGS ${${os}_cc_kext_lipo_flags}
                      DEPENDS ${${os}_cc_kext_libs}
                      OUTPUT_DIR ${COMPILER_RT_LIBRARY_OUTPUT_DIR}
                      INSTALL_DIR ${COMPILER_RT_LIBRARY_INSTALL_DIR})
    endif()
  endforeach()

  darwin_add_eprintf_library(CFLAGS ${CFLAGS})

  # We put the x86 sim slices into the archives for their base OS
  foreach (os ${ARGN})
    if(NOT ${os} MATCHES ".*sim$")
      darwin_lipo_libs(clang_rt.${os}
                        PARENT_TARGET builtins
                        LIPO_FLAGS ${${os}_builtins_lipo_flags} ${${os}sim_builtins_lipo_flags}
                        DEPENDS ${${os}_builtins_libs} ${${os}sim_builtins_libs}
                        OUTPUT_DIR ${COMPILER_RT_LIBRARY_OUTPUT_DIR}
                        INSTALL_DIR ${COMPILER_RT_LIBRARY_INSTALL_DIR})
    endif()
  endforeach()
  darwin_add_embedded_builtin_libraries()
endmacro()

macro(darwin_add_embedded_builtin_libraries)
  # this is a hacky opt-out. If you can't target both intel and arm
  # architectures we bail here.
  set(DARWIN_SOFT_FLOAT_ARCHS armv6m armv7m armv7em armv7)
  set(DARWIN_HARD_FLOAT_ARCHS armv7em armv7)
  if(COMPILER_RT_SUPPORTED_ARCH MATCHES ".*armv.*")
    list(FIND COMPILER_RT_SUPPORTED_ARCH i386 i386_idx)
    if(i386_idx GREATER -1)
      list(APPEND DARWIN_HARD_FLOAT_ARCHS i386)
    endif()

    list(FIND COMPILER_RT_SUPPORTED_ARCH x86_64 x86_64_idx)
    if(x86_64_idx GREATER -1)
      list(APPEND DARWIN_HARD_FLOAT_ARCHS x86_64)
    endif()

    set(MACHO_SYM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/macho_embedded)

    set(CFLAGS "-Oz -Wall -fomit-frame-pointer -ffreestanding")
    set(CMAKE_C_FLAGS "")
    set(CMAKE_CXX_FLAGS "")
    set(CMAKE_ASM_FLAGS "")

    set(SOFT_FLOAT_FLAG -mfloat-abi=soft)
    set(HARD_FLOAT_FLAG -mfloat-abi=hard)

    set(ENABLE_PIC Off)
    set(PIC_FLAG -fPIC)
    set(STATIC_FLAG -static)

    set(DARWIN_macho_embedded_ARCHS armv6m armv7m armv7em armv7 i386 x86_64)

    set(DARWIN_macho_embedded_LIBRARY_OUTPUT_DIR
      ${COMPILER_RT_OUTPUT_DIR}/lib/macho_embedded)
    set(DARWIN_macho_embedded_LIBRARY_INSTALL_DIR
      ${COMPILER_RT_INSTALL_PATH}/lib/macho_embedded)
      
    set(CFLAGS_armv7 "-target thumbv7-apple-darwin-eabi")
    set(CFLAGS_i386 "-march=pentium")

    darwin_read_list_from_file(common_FUNCTIONS ${MACHO_SYM_DIR}/common.txt)
    darwin_read_list_from_file(thumb2_FUNCTIONS ${MACHO_SYM_DIR}/thumb2.txt)
    darwin_read_list_from_file(thumb2_64_FUNCTIONS ${MACHO_SYM_DIR}/thumb2-64.txt)
    darwin_read_list_from_file(arm_FUNCTIONS ${MACHO_SYM_DIR}/arm.txt)
    darwin_read_list_from_file(i386_FUNCTIONS ${MACHO_SYM_DIR}/i386.txt)


    set(armv6m_FUNCTIONS ${common_FUNCTIONS} ${arm_FUNCTIONS})
    set(armv7m_FUNCTIONS ${common_FUNCTIONS} ${arm_FUNCTIONS} ${thumb2_FUNCTIONS})
    set(armv7em_FUNCTIONS ${common_FUNCTIONS} ${arm_FUNCTIONS} ${thumb2_FUNCTIONS})
    set(armv7_FUNCTIONS ${common_FUNCTIONS} ${arm_FUNCTIONS} ${thumb2_FUNCTIONS} ${thumb2_64_FUNCTIONS})
    set(i386_FUNCTIONS ${common_FUNCTIONS} ${i386_FUNCTIONS})
    set(x86_64_FUNCTIONS ${common_FUNCTIONS})

    foreach(arch ${DARWIN_macho_embedded_ARCHS})
      darwin_filter_builtin_sources(${arch}_filtered_sources
        INCLUDE ${arch}_FUNCTIONS
        ${${arch}_SOURCES})
      if(NOT ${arch}_filtered_sources)
        message("${arch}_SOURCES: ${${arch}_SOURCES}")
        message("${arch}_FUNCTIONS: ${${arch}_FUNCTIONS}")
        message(FATAL_ERROR "Empty filtered sources!")
      endif()
    endforeach()

    foreach(float_type SOFT HARD)
      foreach(type PIC STATIC)
        string(TOLOWER "${float_type}_${type}" lib_suffix)
        foreach(arch ${DARWIN_${float_type}_FLOAT_ARCHS})
          set(DARWIN_macho_embedded_SYSROOT ${DARWIN_osx_SYSROOT})
          set(float_flag)
          if(${arch} MATCHES "^arm")
            # x86 targets are hard float by default, but the complain about the
            # float ABI flag, so don't pass it unless we're targeting arm.
            set(float_flag ${${float_type}_FLOAT_FLAG})
          endif()
          darwin_add_builtin_library(clang_rt ${lib_suffix}
                                OS macho_embedded
                                ARCH ${arch}
                                SOURCES ${${arch}_filtered_sources}
                                CFLAGS ${CFLAGS} -arch ${arch} ${${type}_FLAG} ${float_flag} ${CFLAGS_${arch}}
                                PARENT_TARGET builtins)
        endforeach()
        foreach(lib ${macho_embedded_${lib_suffix}_libs})
          set_target_properties(${lib} PROPERTIES LINKER_LANGUAGE C)
        endforeach()
        darwin_lipo_libs(clang_rt.${lib_suffix}
                      PARENT_TARGET builtins
                      LIPO_FLAGS ${macho_embedded_${lib_suffix}_lipo_flags}
                      DEPENDS ${macho_embedded_${lib_suffix}_libs}
                      OUTPUT_DIR ${DARWIN_macho_embedded_LIBRARY_OUTPUT_DIR}
                      INSTALL_DIR ${DARWIN_macho_embedded_LIBRARY_INSTALL_DIR})
      endforeach()
    endforeach()
  endif()
endmacro()
