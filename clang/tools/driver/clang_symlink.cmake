# We need to execute this script at installation time because the
# DESTDIR environment variable may be unset at configuration time.
# See PR8397.

if(UNIX)
  set(CLANGXX_LINK_OR_COPY create_symlink)
  set(CLANGXX_DESTDIR $ENV{DESTDIR})
else()
  set(CLANGXX_LINK_OR_COPY copy)
endif()

# CMAKE_EXECUTABLE_SUFFIX is undefined on cmake scripts. See PR9286.
if( WIN32 )
  set(EXECUTABLE_SUFFIX ".exe")
else()
  set(EXECUTABLE_SUFFIX "")
endif()

set(bindir "${CLANGXX_DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/")
set(clang "clang${EXECUTABLE_SUFFIX}")
set(clangxx "clang++${EXECUTABLE_SUFFIX}")

message("Creating clang++ executable based on ${clang}")

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E ${CLANGXX_LINK_OR_COPY} "${clang}" "${clangxx}"
  WORKING_DIRECTORY "${bindir}")
