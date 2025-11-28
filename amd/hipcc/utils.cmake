## Configure Copyright File for Debian Package
function( configure_pkg PACKAGE_NAME_T COMPONENT_NAME_T PACKAGE_VERSION_T MAINTAINER_NM_T MAINTAINER_EMAIL_T)
    # Check If Debian Platform
    find_file (DEBIAN debian_version debconf.conf PATHS /etc)
    if(DEBIAN)
      set( BUILD_DEBIAN_PKGING_FLAG ON CACHE BOOL "Internal Status Flag to indicate Debian Packaging Build" FORCE )
      set_debian_pkg_cmake_flags( ${PACKAGE_NAME_T} ${PACKAGE_VERSION_T}
                                  ${MAINTAINER_NM_T} ${MAINTAINER_EMAIL_T} )

      set(DEB_SOURCE_DIR "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/DEBIAN")
      set(DEB_BUILD_DIR "${CMAKE_BINARY_DIR}/DEBIAN")
      # Create debian directory in build tree
      file(MAKE_DIRECTORY "${DEB_BUILD_DIR}")

      # Configure the copyright file
      configure_file(
	"${DEB_SOURCE_DIR}/copyright.in"
	"${DEB_BUILD_DIR}/copyright"
        @ONLY
      )

      # Install copyright file
      install ( FILES "${CMAKE_BINARY_DIR}/DEBIAN/copyright"
	        DESTINATION "${CMAKE_INSTALL_DOCDIR}"
	        COMPONENT ${COMPONENT_NAME_T} )

      # Configure the changelog file
      configure_file(
	"${DEB_SOURCE_DIR}/changelog.in"
	"${DEB_BUILD_DIR}/changelog.Debian"
        @ONLY
      )

      # Install Change Log 
      find_program ( DEB_GZIP_EXEC gzip )
      if(EXISTS "${DEB_BUILD_DIR}/changelog.Debian" )
        execute_process(
          COMMAND ${DEB_GZIP_EXEC} -f -n -9 "${DEB_BUILD_DIR}/changelog.Debian"
	  WORKING_DIRECTORY "${DEB_BUILD_DIR}"
          RESULT_VARIABLE result
          OUTPUT_VARIABLE output
          ERROR_VARIABLE error
        )
        if(NOT ${result} EQUAL 0)
          message(FATAL_ERROR "Failed to compress: ${error}")
        endif()
	install ( FILES "${DEB_BUILD_DIR}/${DEB_CHANGELOG_INSTALL_FILENM}"
                  DESTINATION ${CMAKE_INSTALL_DOCDIR}
                  COMPONENT ${COMPONENT_NAME_T})
      endif()

    endif()
endfunction()

# Set variables for changelog and copyright
# For Debian specific Packages 
function( set_debian_pkg_cmake_flags DEB_PACKAGE_NAME_T DEB_PACKAGE_VERSION_T DEB_MAINTAINER_NM_T DEB_MAINTAINER_EMAIL_T )
    # Setting configure flags
    set( DEB_PACKAGE_NAME             "${DEB_PACKAGE_NAME_T}" CACHE STRING "Debian Package Name" FORCE )
    set( DEB_PACKAGE_VERSION          "${DEB_PACKAGE_VERSION_T}" CACHE STRING "Debian Package Version String" )
    set( DEB_MAINTAINER_NAME          "${DEB_MAINTAINER_NM_T}" CACHE STRING "Debian Package Maintainer Name" )
    set( DEB_MAINTAINER_EMAIL         "${DEB_MAINTAINER_EMAIL_T}" CACHE STRING "Debian Package Maintainer Email" )
    set( DEB_COPYRIGHT_YEAR           "2025" CACHE STRING "Debian Package Copyright Year" )
    set( DEB_LICENSE                  "MIT" CACHE STRING "Debian Package License Type" )
    set( DEB_CHANGELOG_INSTALL_FILENM "changelog.Debian.gz" CACHE STRING "Debian Package ChangeLog File Name" ) 

    if( BUILD_ENABLE_LINTIAN_OVERRIDES )
      set( DEB_OVERRIDES_INSTALL_FILENM "${DEB_PACKAGE_NAME}" CACHE STRING "Debian Package Lintian Override File Name" FORCE )
      set( DEB_OVERRIDES_INSTALL_PATH   "/usr/share/lintian/overrides/" CACHE STRING "Deb Pkg Lintian Override Install Loc" )
    endif()

    # Get TimeStamp
    find_program( DEB_DATE_TIMESTAMP_EXEC date )
    set ( DEB_TIMESTAMP_FORMAT_OPTION "-R" )
    execute_process (
        COMMAND ${DEB_DATE_TIMESTAMP_EXEC} ${DEB_TIMESTAMP_FORMAT_OPTION}
        OUTPUT_VARIABLE TIMESTAMP_T
    )
    set( DEB_TIMESTAMP                "${TIMESTAMP_T}" CACHE STRING "Current Time Stamp for Copyright/Changelog" )

    message(STATUS "DEB_PACKAGE_NAME             : ${DEB_PACKAGE_NAME}" )
    message(STATUS "DEB_PACKAGE_VERSION          : ${DEB_PACKAGE_VERSION}" )
    message(STATUS "DEB_MAINTAINER_NAME          : ${DEB_MAINTAINER_NAME}" )
    message(STATUS "DEB_MAINTAINER_EMAIL         : ${DEB_MAINTAINER_EMAIL}" )
    message(STATUS "DEB_COPYRIGHT_YEAR           : ${DEB_COPYRIGHT_YEAR}" )
    message(STATUS "DEB_LICENSE                  : ${DEB_LICENSE}" )
    message(STATUS "DEB_TIMESTAMP                : ${DEB_TIMESTAMP}" )
    message(STATUS "DEB_CHANGELOG_INSTALL_FILENM : ${DEB_CHANGELOG_INSTALL_FILENM}" )
endfunction()

