function(declare_dependencies prefixes dependencies)
  set(suffixes c s i l ll)
  foreach(prefix ${prefixes})
    foreach(suffix IN LISTS suffixes)
      add_entrypoint_object(
        stdc_${prefix}_u${suffix}
        SRCS
          stdc_${prefix}_u${suffix}.cpp
        HDRS
          stdc_${prefix}_u${suffix}.h
        DEPENDS
          ${dependencies}
      )
    endforeach()
  endforeach()
endfunction()


set(prefixes
  leading_zeros
  leading_ones
  trailing_zeros
  trailing_ones
  count_ones
  has_single_bit
  bit_width
  bit_floor
  bit_ceil
)
declare_dependencies("${prefixes}" libc.src.__support.CPP.bit)
set(prefixes
  first_leading_zero
  first_leading_one
  first_trailing_zero
  first_trailing_one
  count_zeros
)
declare_dependencies("${prefixes}" libc.src.__support.math_extras)
