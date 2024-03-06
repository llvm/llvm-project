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
      -O3
      -ffixed-point
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
      -O3
      -ffixed-point
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
      -O3
      -ffixed-point
    DEPENDS
      libc.src.__support.fixed_point.fx_bits
  )
endforeach()
