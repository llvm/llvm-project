add_entrypoint_object(
  dlclose
  SRCS
    dlclose.cpp
  HDRS
    dlclose.h
)

add_entrypoint_object(
  dlerror
  SRCS
    dlerror.cpp
  HDRS
    dlerror.h
  DEPENDS
    libc.include.dlfcn
)

add_entrypoint_object(
  dlopen
  SRCS
    dlopen.cpp
  HDRS
    dlopen.h
  DEPENDS
    libc.include.dlfcn
)

add_entrypoint_object(
  dlsym
  SRCS
    dlsym.cpp
  HDRS
    dlsym.h
  DEPENDS
    libc.include.dlfcn
)

add_entrypoint_object(
  dlinfo
  SRCS
    dlinfo.cpp
  HDRS
    dlinfo.h
  DEPENDS
    libc.include.dlfcn
)

add_entrypoint_object(
  dladdr
  SRCS
    dladdr.cpp
  HDRS
    dladdr.h
  DEPENDS
    libc.include.dlfcn
)
