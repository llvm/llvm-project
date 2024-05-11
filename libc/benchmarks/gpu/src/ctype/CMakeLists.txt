add_custom_target(libc-gpu-ctype-benchmarks)

add_gpu_benchmark(
    isalnum_benchmark
    SUITE
        libc-gpu-ctype-benchmarks
    SRCS
        isalnum_benchmark.cpp
    DEPENDS
        libc.src.ctype.isalnum
)

add_gpu_benchmark(
    isalpha_benchmark
    SUITE
        libc-gpu-ctype-benchmarks
    SRCS
        isalpha_benchmark.cpp
    DEPENDS
        libc.src.ctype.isalpha
)
