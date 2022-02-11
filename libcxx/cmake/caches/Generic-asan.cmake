set(LLVM_USE_SANITIZER "Address" CACHE STRING "")
# This is a temporary (hopefully) workaround for an ASan issue (see https://llvm.org/D119410).
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mllvm -asan-use-private-alias=1" CACHE INTERNAL "")
