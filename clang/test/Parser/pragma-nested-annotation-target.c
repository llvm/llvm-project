// RUN: %clang_cc1 -triple riscv64 -fsyntax-only -fms-extensions -DTEST_RISCV -verify=riscv %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x cuda -fsyntax-only -fms-extensions -DTEST_CUDA -verify=cuda %s

#ifdef TEST_RISCV
#pragma clang riscv _Pragma("weak foobar") intrinsic vector // riscv-warning {{unexpected argument to '#pragma riscv'; expected 'intrinsic'}}
#pragma clang riscv intrinsic _Pragma("weak foobar") vector // riscv-warning {{unexpected argument to '#pragma riscv'; expected 'vector', 'sifive_vector' or 'andes_vector'}}

// Invalid ordinary tokens keep their spelling.
#pragma clang riscv 42 // riscv-warning {{unexpected argument '42' to '#pragma riscv'; expected 'intrinsic'}}
#pragma clang riscv intrinsic wrong // riscv-warning {{unexpected argument 'wrong' to '#pragma riscv'; expected 'vector', 'sifive_vector' or 'andes_vector'}}
#endif

#ifdef TEST_CUDA
#pragma clang force_cuda_host_device _Pragma("weak foobar") // cuda-warning {{incorrect use of #pragma clang force_cuda_host_device begin|end}}

#pragma clang force_cuda_host_device begin
void host_device_function(void) {}
#pragma clang force_cuda_host_device end
#endif

int after_pragmas;
int use_after_pragmas(void) { return after_pragmas; }
