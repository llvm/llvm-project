add_object_library(
  locale
  SRCS
    locale.cpp
  HDRS
    locale.h
  DEPENDS
    libc.include.locale
)

add_entrypoint_object(
  localeconv
  SRCS
    localeconv.cpp
  HDRS
    localeconv.h
  DEPENDS
    libc.include.locale
  CXX_STANDARD
    20 # For designated initializers
)

add_entrypoint_object(
  newlocale
  SRCS
    newlocale.cpp
  HDRS
    newlocale.h
  DEPENDS
    libc.include.locale
    .locale
)

add_entrypoint_object(
  duplocale
  SRCS
    duplocale.cpp
  HDRS
    duplocale.h
  DEPENDS
    libc.include.locale
    .locale
)

add_entrypoint_object(
  setlocale
  SRCS
    setlocale.cpp
  HDRS
    setlocale.h
  DEPENDS
    libc.include.locale
    .locale
)

add_entrypoint_object(
  uselocale
  SRCS
    uselocale.cpp
  HDRS
    uselocale.h
  DEPENDS
    libc.include.locale
    .locale
)

add_entrypoint_object(
  freelocale
  SRCS
    freelocale.cpp
  HDRS
    freelocale.h
  DEPENDS
    libc.include.locale
    .locale
)
