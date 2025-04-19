extern const __constant bool __oclc_finite_only_opt;
extern const __constant bool __oclc_unsafe_math_opt;
extern const __constant bool __oclc_wavefrontsize64;
extern const __constant int __oclc_ISA_version;
extern const __constant int __oclc_ABI_version;

void kernel device_libs(__global float *status) {

  if (__oclc_finite_only_opt)
    status[0] = 1.0;
  if (__oclc_unsafe_math_opt)
    status[1] = 1.0;
  if (__oclc_wavefrontsize64)
    status[4] = 1.0;
  if (__oclc_ISA_version)
    status[5] = 1.0;
  if (__oclc_ABI_version)
    status[6] = 1.0;

  // Math functions to test AMDGPULibCalls Folding optimizations
  // fold_sincos()
  float x = 0.25;
  status[7] = sin(x) + cos(x);
  status[8] = cos(x) + sin(x);

  // fold_rootn()
  float y = 725.0;
  status[9] = rootn(y, 3);
  status[10] = rootn(y, -1);
  status[11] = rootn(y, -2);

  // fold_pow()
  float z = 12.16;
  status[12] = pow(z, (float)0.5);
  status[13] = powr(y, (float)7.23);

  // printf()
  printf("testy\n");
}
