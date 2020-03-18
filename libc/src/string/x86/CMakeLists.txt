add_memcpy("memcpy_${LIBC_TARGET_MACHINE}_opt_none" REJECT "${ALL_CPU_FEATURES}")
add_memcpy("memcpy_${LIBC_TARGET_MACHINE}_opt_sse" REQUIRE "SSE" REJECT "SSE2")
add_memcpy("memcpy_${LIBC_TARGET_MACHINE}_opt_avx" REQUIRE "AVX" REJECT "AVX2")
add_memcpy("memcpy_${LIBC_TARGET_MACHINE}_opt_avx512f" REQUIRE "AVX512F")
