include(${CMAKE_CURRENT_LIST_DIR}/Apple-lldb-base.cmake)

# Begin Swift Mods
set(LLDB_ALLOW_STATIC_BINDINGS ON CACHE BOOL "")
# End Swift Mods

set(LLVM_DISTRIBUTION_COMPONENTS
  lldb
  liblldb
  lldb-argdumper
  lldb-server
  CACHE STRING "")
