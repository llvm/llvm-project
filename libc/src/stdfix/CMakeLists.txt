if(NOT LIBC_COMPILER_HAS_FIXED_POINT)
  return()
endif()

foreach(suffix IN ITEMS hr r lr hk k lk)
  add_entrypoint_object(
    abs${suffix}
    HDRS
      abs${suffix}.h
    SRCS
      abs${suffix}.cpp
    COMPILE_OPTIONS
      ${libc_opt_high_flag}
    DEPENDS
      libc.src.__support.fixed_point.fx_bits
  )
endforeach()

foreach(suffix IN ITEMS uhr ur ulr uhk uk)
  add_entrypoint_object(
    sqrt${suffix}
    HDRS
      sqrt${suffix}.h
    SRCS
      sqrt${suffix}.cpp
    COMPILE_OPTIONS
      ${libc_opt_high_flag}
    DEPENDS
      libc.src.__support.fixed_point.sqrt
  )
endforeach()

foreach(suffix IN ITEMS hr r lr hk k lk uhr ur ulr uhk uk ulk)
  add_entrypoint_object(
    round${suffix}
    HDRS
      round${suffix}.h
    SRCS
      round${suffix}.cpp
    COMPILE_OPTIONS
      ${libc_opt_high_flag}
    DEPENDS
      libc.src.__support.fixed_point.fx_bits
  )

  add_entrypoint_object(
    ${suffix}bits
    HDRS
      ${suffix}bits.h
    SRCS
      ${suffix}bits.cpp
    DEPENDS
      libc.src.__support.CPP.bit
      libc.src.__support.fixed_point.fx_bits
  )
endforeach()

add_entrypoint_object(
  uhksqrtus
  HDRS
    uhksqrtus.h
  SRCS
    uhksqrtus.cpp
  COMPILE_OPTIONS
    ${libc_opt_high_flag}
  DEPENDS
    libc.src.__support.fixed_point.sqrt
)

add_entrypoint_object(
  uksqrtui
  HDRS
    uksqrtui.h
  SRCS
    uksqrtui.cpp
  COMPILE_OPTIONS
    ${libc_opt_high_flag}
  DEPENDS
    libc.src.__support.fixed_point.sqrt
)

add_entrypoint_object(
  exphk
  HDRS
    exphk.h
  SRCS
    exphk.cpp
  COMPILE_OPTIONS
    ${libc_opt_high_flag}
  DEPENDS
    libc.src.__support.fixed_point.fx_rep
    libc.src.__support.CPP.bit
)

add_entrypoint_object(
  expk
  HDRS
    expk.h
  SRCS
    expk.cpp
  COMPILE_OPTIONS
    ${libc_opt_high_flag}
  DEPENDS
    libc.src.__support.fixed_point.fx_rep
    libc.src.__support.CPP.bit
)
