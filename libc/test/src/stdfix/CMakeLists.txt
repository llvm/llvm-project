if(NOT LIBC_COMPILER_HAS_FIXED_POINT)
  return()
endif()

add_libc_testsuite(libc-stdfix-tests)

foreach(suffix IN ITEMS hr r lr hk k lk)
  add_libc_test(
    abs${suffix}_test
    SUITE
      libc-stdfix-tests
    HDRS
      AbsTest.h
    SRCS
      abs${suffix}_test.cpp
    DEPENDS
      libc.src.stdfix.abs${suffix}
      libc.src.__support.fixed_point.fx_rep
  )
endforeach()

foreach(suffix IN ITEMS uhr ur ulr uhk uk)
  add_libc_test(
    sqrt${suffix}_test
    SUITE
      libc-stdfix-tests
    HDRS
      SqrtTest.h
    SRCS
      sqrt${suffix}_test.cpp
    DEPENDS
      libc.src.stdfix.sqrt${suffix}
      libc.src.__support.CPP.bit
      libc.src.__support.fixed_point.fx_rep
      libc.src.__support.FPUtil.basic_operations
      libc.src.__support.FPUtil.sqrt
  )
endforeach()

foreach(suffix IN ITEMS hr r lr hk k lk uhr ur ulr uhk uk ulk)
  if(suffix MATCHES "^u")
    set(int_type uint_${suffix}_t)
  else()
    set(int_type int_${suffix}_t)
  endif()

  add_libc_test(
    round${suffix}_test
    SUITE
      libc-stdfix-tests
    HDRS
      RoundTest.h
    SRCS
      round${suffix}_test.cpp
    DEPENDS
      libc.src.stdfix.round${suffix}
      libc.src.__support.fixed_point.fx_rep
  )

  add_libc_test(
    ${suffix}bits_test
    SUITE
      libc-stdfix-tests
    HDRS
      FxBitsTest.h
    SRCS
      ${suffix}bits_test.cpp
    DEPENDS
      libc.src.stdfix.${suffix}bits
      libc.src.__support.fixed_point.fx_rep
      libc.src.__support.math_extras
  )

  add_libc_test(
    bits${suffix}_test
    SUITE
      libc-stdfix-tests
    HDRS
      BitsFxTest.h
    SRCS
      bits${suffix}_test.cpp
    DEPENDS
      libc.include.llvm-libc-types.${int_type}
      libc.src.stdfix.bits${suffix}
      libc.src.__support.fixed_point.fx_rep
  )

  add_libc_test(
    countls${suffix}_test
    SUITE
      libc-stdfix-tests
    HDRS
      CountlsTest.h
    SRCS
      countls${suffix}_test.cpp
    DEPENDS
      libc.src.stdfix.countls${suffix}
      libc.src.__support.fixed_point.fx_rep
  )
endforeach()

foreach(suffix IN ITEMS r lr k lk ur ulr uk ulk)
  add_libc_test(
    idiv${suffix}_test
    SUITE
      libc-stdfix-tests
    HDRS
      IdivFxTest.h
    SRCS
      idiv${suffix}_test.cpp
    DEPENDS
      libc.hdr.signal_macros
      libc.include.llvm-libc-macros.stdfix_macros
      libc.src.stdfix.idiv${suffix}
      libc.src.__support.fixed_point.fx_rep
  )
endforeach()

foreach(suffix IN ITEMS r lr k lk ur ulr uk ulk)
  add_libc_test(
    ${suffix}divi_test
    SUITE
      libc-stdfix-tests
    HDRS
      FxDiviTest.h
    SRCS
      ${suffix}divi_test.cpp
    DEPENDS
      libc.hdr.signal_macros
      libc.include.llvm-libc-macros.stdfix_macros
      libc.src.stdfix.${suffix}divi
      libc.src.__support.CPP.limits
      libc.src.__support.CPP.type_traits
      libc.src.__support.fixed_point.fx_rep
  )
endforeach()

foreach(suffix IN ITEMS r lr k lk ur ulr uk ulk)
  add_libc_test(
    divi${suffix}_test
    SUITE
      libc-stdfix-tests
    HDRS
      DiviFxTest.h
    SRCS
      divi${suffix}_test.cpp
    DEPENDS
      libc.hdr.signal_macros
      libc.include.llvm-libc-macros.stdfix_macros
      libc.src.stdfix.divi${suffix}
      libc.src.__support.CPP.limits
      libc.src.__support.fixed_point.fx_rep
  )
endforeach()

foreach(suffix IN ITEMS r lr k lk ur ulr uk ulk)
  add_libc_test(
    muli${suffix}_test
    SUITE
      libc-stdfix-tests
    HDRS
      MuliFxTest.h
    SRCS
      muli${suffix}_test.cpp
    DEPENDS
      libc.include.llvm-libc-macros.stdfix_macros
      libc.src.stdfix.muli${suffix}
      libc.src.__support.CPP.limits
      libc.src.__support.fixed_point.fx_rep
  )
endforeach()

add_libc_test(
  uhksqrtus_test
  SUITE
    libc-stdfix-tests
  HDRS
    ISqrtTest.h
  SRCS
    uhksqrtus_test.cpp
  DEPENDS
    libc.src.stdfix.uhksqrtus
    libc.src.__support.fixed_point.fx_rep
    libc.src.__support.fixed_point.sqrt
    libc.src.__support.FPUtil.basic_operations
    libc.src.__support.FPUtil.sqrt
)

add_libc_test(
  uksqrtui_test
  SUITE
    libc-stdfix-tests
  HDRS
    ISqrtTest.h
  SRCS
    uksqrtui_test.cpp
  DEPENDS
    libc.src.stdfix.uksqrtui
    libc.src.__support.fixed_point.fx_rep
    libc.src.__support.fixed_point.sqrt
    libc.src.__support.FPUtil.basic_operations
    libc.src.__support.FPUtil.sqrt
)

foreach(suffix IN ITEMS hk k uhk uk)
  add_libc_test(
    exp${suffix}_test
    SUITE
      libc-stdfix-tests
    HDRS
      ExpTest.h
    SRCS
      exp${suffix}_test.cpp
    DEPENDS
      libc.src.stdfix.exp${suffix}
      libc.src.math.exp
      libc.src.__support.fixed_point.fx_rep
      libc.src.__support.FPUtil.basic_operations
  )
endforeach()

set(macros_depends)

foreach(s IN ITEMS r hr lr k hk lk)
  list(APPEND macros_depends
    libc.src.stdfix.abs${s}
    libc.src.stdfix.countls${s}
    libc.src.stdfix.countlsu${s}
    libc.src.stdfix.round${s}
    libc.src.stdfix.roundu${s}
  )
endforeach()

add_libc_test(
  macros_test
  SUITE libc-stdfix-tests
  SRCS macros_test.cpp
  DEPENDS
    libc.include.llvm-libc-macros.stdfix_macros
    ${macros_depends}
)
