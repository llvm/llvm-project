set(prefixes
  leading_zeros
  leading_ones
  trailing_zeros
  trailing_ones
)
set(suffixes c s i l ll)
foreach(prefix IN LISTS prefixes)
  foreach(suffix IN LISTS suffixes)
    add_entrypoint_object(
      stdc_${prefix}_u${suffix}
      SRCS
        stdc_${prefix}_u${suffix}.cpp
      HDRS
        stdc_${prefix}_u${suffix}.h
      DEPENDS
        libc.src.__support.CPP.bit
    )
  endforeach()
endforeach()
