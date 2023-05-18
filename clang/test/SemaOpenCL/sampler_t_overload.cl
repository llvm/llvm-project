// RUN: %clang_cc1 %s

void __attribute__((overloadable)) foo(sampler_t, read_only image1d_t);
void __attribute__((overloadable)) foo(sampler_t, read_only image2d_t);

constant sampler_t glb_smp = 5;

void kernel ker(read_only image1d_t src1, read_only image2d_t src2) {
  const sampler_t smp = 10;
  foo(glb_smp, src1);
  foo(smp, src2);
}
