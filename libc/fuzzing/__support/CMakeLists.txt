add_libc_fuzzer(
  uint_fuzz
  SRCS
    uint_fuzz.cpp
  DEPENDS
    libc.src.__support.big_int
)

add_libc_fuzzer(
  hashtable_fuzz
  SRCS
    hashtable_fuzz.cpp
  DEPENDS
    libc.src.__support.HashTable.table
    libc.src.string.memcpy
)
