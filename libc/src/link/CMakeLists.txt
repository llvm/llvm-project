add_entrypoint_object(
  _r_debug
  SRCS
    _r_debug.cpp
  HDRS
    _r_debug.h
  DEPENDS
    libc.hdr.types.struct_r_debug
    libc.src.__support.common
    libc.src.__support.macros.config
)

# The lock that dl_iterate_phdr takes belongs to dlfcn, which only exists
# where there is a loader to take it against.
set(dl_iterate_phdr_lock)
if(TARGET libc.src.dlfcn.dl_internal)
  set(dl_iterate_phdr_lock libc.src.dlfcn.dl_internal)
endif()

add_entrypoint_object(
  dl_iterate_phdr
  SRCS
    dl_iterate_phdr.cpp
  HDRS
    dl_iterate_phdr.h
  DEPENDS
    libc.src.__support.CPP.mutex
    libc.src.__support.elf.passive_abi
    ${dl_iterate_phdr_lock}
    libc.hdr.stdint_proxy
    libc.include.elf
    libc.include.llvm-libc-macros.link_macros
    libc.src.__support.OSUtil.linux.auxv
)
