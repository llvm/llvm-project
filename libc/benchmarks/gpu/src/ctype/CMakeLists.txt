add_libc_testsuite(libc-gpu-ctype-benchmarks PARENT gpu-benchmark)

add_benchmark(
  isalnum_benchmark
  SUITE
    libc-gpu-ctype-benchmarks
  SRCS
    isalnum_benchmark.cpp
  DEPENDS
    libc.src.ctype.isalnum
  LOADER_ARGS
    --threads 64
)

add_benchmark(
  isalpha_benchmark
  SUITE
    libc-gpu-ctype-benchmarks
  SRCS
    isalpha_benchmark.cpp
  DEPENDS
    libc.src.ctype.isalpha
)
