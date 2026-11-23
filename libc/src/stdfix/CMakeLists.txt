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
    DEPENDS
      libc.src.__support.fixed_point.sqrt
  )
endforeach()

foreach(suffix IN ITEMS hr r lr hk k lk uhr ur ulr uhk uk ulk)
  if(suffix MATCHES "^u")
    set(int_type uint_${suffix}_t)
  else()
    set(int_type int_${suffix}_t)
  endif()

  add_entrypoint_object(
    round${suffix}
    HDRS
      round${suffix}.h
    SRCS
      round${suffix}.cpp
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
      libc.include.llvm-libc-macros.stdfix_macros
      libc.include.llvm-libc-types.${int_type}
      libc.src.__support.CPP.bit
  )

  add_entrypoint_object(
    bits${suffix}
    HDRS
      bits${suffix}.h
    SRCS
      bits${suffix}.cpp
    DEPENDS
      libc.include.llvm-libc-macros.stdfix_macros
      libc.include.llvm-libc-types.${int_type}
      libc.src.__support.fixed_point.fx_bits
  )

  add_entrypoint_object(
    countls${suffix}
    HDRS
      countls${suffix}.h
    SRCS
      countls${suffix}.cpp
    DEPENDS
      libc.src.__support.fixed_point.fx_bits
  )
endforeach()

foreach(suffix IN ITEMS r lr k lk ur ulr uk ulk)
  add_entrypoint_object(
    idiv${suffix}
    HDRS
      idiv${suffix}.h
    SRCS
      idiv${suffix}.cpp
    DEPENDS
      libc.src.__support.fixed_point.fx_bits
  )
endforeach()

foreach(suffix IN ITEMS r lr k lk ur ulr uk ulk)
  add_entrypoint_object(
    ${suffix}divi
    HDRS
      ${suffix}divi.h
    SRCS
      ${suffix}divi.cpp
    DEPENDS
      libc.src.__support.fixed_point.fx_bits
  )
endforeach()

foreach(suffix IN ITEMS r lr k lk ur ulr uk ulk)
  add_entrypoint_object(
    divi${suffix}
    HDRS
      divi${suffix}.h
    SRCS
      divi${suffix}.cpp
    DEPENDS
      libc.src.__support.fixed_point.fx_bits
  )
endforeach()

foreach(suffix IN ITEMS r lr k lk ur ulr uk ulk)
  add_entrypoint_object(
    muli${suffix}
    HDRS
      muli${suffix}.h
    SRCS
      muli${suffix}.cpp
    DEPENDS
      libc.src.__support.fixed_point.fx_bits
  )
endforeach()

add_entrypoint_object(
  uhksqrtus
  HDRS
    uhksqrtus.h
  SRCS
    uhksqrtus.cpp
  DEPENDS
    libc.src.__support.fixed_point.sqrt
)

add_entrypoint_object(
  uksqrtui
  HDRS
    uksqrtui.h
  SRCS
    uksqrtui.cpp
  DEPENDS
    libc.src.__support.fixed_point.sqrt
)

foreach(suffix IN ITEMS hk k uhk uk)
  add_entrypoint_object(
    exp${suffix}
    HDRS
      exp${suffix}.h
    SRCS
      exp${suffix}.cpp
    DEPENDS
      libc.src.__support.CPP.bit
      libc.src.__support.fixed_point.fx_rep
      libc.src.__support.macros.optimization
  )
endforeach()
