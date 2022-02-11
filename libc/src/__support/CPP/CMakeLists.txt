# TODO(michaelrj): separate the standalone_cpp library into individual targets.
add_header_library(
  standalone_cpp
  HDRS
    Array.h
    ArrayRef.h
    Bit.h
    Bitset.h
    Functional.h
    Limits.h
    StringView.h
    TypeTraits.h
)

if(LLVM_LIBC_INCLUDE_SCUDO OR NOT LLVM_LIBC_FULL_BUILD)
  add_header_library(
    vector
    HDRS
      vector.h
    DEPENDS
      libc.include.stdlib
  )
endif()
