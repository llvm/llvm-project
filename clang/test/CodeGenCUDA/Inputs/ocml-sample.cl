#pragma OPENCL EXTENSION cl_khr_fp16 : enable

half do_f16_stuff(half a, half b, half c) {
  return __builtin_fmaf16(a, b, c) + 4.0h;
}

float do_f32_stuff(float a, float b, float c) {
  return __builtin_fmaf(a, b, c) + 4.0f;
}

double do_f64_stuff(double a, double b, double c) {
  return __builtin_fma(a, b, c) + 4.0;
}

__attribute__((weak))
float weak_do_f32_stuff(float a, float b, float c) {
  return c * (a / b);
}
