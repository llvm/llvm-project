// RUN: %clang_cc1 -include __clang_hip_runtime_wrapper.h      \
// RUN:   -internal-isystem %S/../../lib/Headers/cuda_wrappers \
// RUN:   -internal-isystem %S/Inputs/include \
// RUN:   -triple amdgcn-amd-amdhsa -aux-triple x86_64-unknown-unknown \
// RUN:   -target-cpu gfx906 %s -fcuda-is-device -fsyntax-only -o -

template<typename A, typename B>
struct is_same {
    static constexpr bool value = false;
};

template<typename A>
struct is_same<A, A> {
    static constexpr bool value = true;
};

__device__ void test_abs()
{
    static_assert(is_same<decltype(abs((float)0)), float>::value, "");
    static_assert(is_same<decltype(abs((double)0)), double>::value, "");
}

__device__ void test_acos()
{
    static_assert(is_same<decltype(acos((float)0)), float>::value, "");
    static_assert(is_same<decltype(acos((bool)0)), double>::value, "");
    static_assert(is_same<decltype(acos((unsigned short)0)), double>::value, "");
    static_assert(is_same<decltype(acos((int)0)), double>::value, "");
    static_assert(is_same<decltype(acos((unsigned int)0)), double>::value, "");
    static_assert(is_same<decltype(acos((long)0)), double>::value, "");
    static_assert(is_same<decltype(acos((unsigned long)0)), double>::value, "");
    static_assert(is_same<decltype(acos((long long)0)), double>::value, "");
    static_assert(is_same<decltype(acos((unsigned long long)0)), double>::value, "");
    static_assert(is_same<decltype(acos((double)0)), double>::value, "");
    static_assert(is_same<decltype(acosf(0)), float>::value, "");
}

__device__ void test_asin()
{
    static_assert(is_same<decltype(asin((float)0)), float>::value, "");
    static_assert(is_same<decltype(asin((bool)0)), double>::value, "");
    static_assert(is_same<decltype(asin((unsigned short)0)), double>::value, "");
    static_assert(is_same<decltype(asin((int)0)), double>::value, "");
    static_assert(is_same<decltype(asin((unsigned int)0)), double>::value, "");
    static_assert(is_same<decltype(asin((long)0)), double>::value, "");
    static_assert(is_same<decltype(asin((unsigned long)0)), double>::value, "");
    static_assert(is_same<decltype(asin((long long)0)), double>::value, "");
    static_assert(is_same<decltype(asin((unsigned long long)0)), double>::value, "");
    static_assert(is_same<decltype(asin((double)0)), double>::value, "");
    static_assert(is_same<decltype(asinf(0)), float>::value, "");
}

__device__ void test_atan()
{
    static_assert(is_same<decltype(atan((float)0)), float>::value, "");
    static_assert(is_same<decltype(atan((bool)0)), double>::value, "");
    static_assert(is_same<decltype(atan((unsigned short)0)), double>::value, "");
    static_assert(is_same<decltype(atan((int)0)), double>::value, "");
    static_assert(is_same<decltype(atan((unsigned int)0)), double>::value, "");
    static_assert(is_same<decltype(atan((long)0)), double>::value, "");
    static_assert(is_same<decltype(atan((unsigned long)0)), double>::value, "");
    static_assert(is_same<decltype(atan((long long)0)), double>::value, "");
    static_assert(is_same<decltype(atan((unsigned long long)0)), double>::value, "");
    static_assert(is_same<decltype(atan((double)0)), double>::value, "");
    static_assert(is_same<decltype(atanf(0)), float>::value, "");
}

__device__ void test_atan2()
{
    static_assert(is_same<decltype(atan2((float)0, (float)0)), float>::value, "");
    static_assert(is_same<decltype(atan2((bool)0, (float)0)), double>::value, "");
    static_assert(is_same<decltype(atan2((unsigned short)0, (double)0)), double>::value, "");
    static_assert(is_same<decltype(atan2((float)0, (unsigned int)0)), double>::value, "");
    static_assert(is_same<decltype(atan2((double)0, (long)0)), double>::value, "");
    static_assert(is_same<decltype(atan2((int)0, (long long)0)), double>::value, "");
    static_assert(is_same<decltype(atan2((int)0, (unsigned long long)0)), double>::value, "");
    static_assert(is_same<decltype(atan2((double)0, (double)0)), double>::value, "");
    static_assert(is_same<decltype(atan2((float)0, (double)0)), double>::value, "");
    static_assert(is_same<decltype(atan2f(0,0)), float>::value, "");
    static_assert(is_same<decltype(atan2((int)0, (int)0)), double>::value, "");
}

__device__ void test_ceil()
{
    static_assert(is_same<decltype(ceil((float)0)), float>::value, "");
    static_assert(is_same<decltype(ceil((bool)0)), double>::value, "");
    static_assert(is_same<decltype(ceil((unsigned short)0)), double>::value, "");
    static_assert(is_same<decltype(ceil((int)0)), double>::value, "");
    static_assert(is_same<decltype(ceil((unsigned int)0)), double>::value, "");
    static_assert(is_same<decltype(ceil((long)0)), double>::value, "");
    static_assert(is_same<decltype(ceil((unsigned long)0)), double>::value, "");
    static_assert(is_same<decltype(ceil((long long)0)), double>::value, "");
    static_assert(is_same<decltype(ceil((unsigned long long)0)), double>::value, "");
    static_assert(is_same<decltype(ceil((double)0)), double>::value, "");
    static_assert(is_same<decltype(ceilf(0)), float>::value, "");
}

__device__ void test_cos()
{
    static_assert(is_same<decltype(cos((float)0)), float>::value, "");
    static_assert(is_same<decltype(cos((bool)0)), double>::value, "");
    static_assert(is_same<decltype(cos((unsigned short)0)), double>::value, "");
    static_assert(is_same<decltype(cos((int)0)), double>::value, "");
    static_assert(is_same<decltype(cos((unsigned int)0)), double>::value, "");
    static_assert(is_same<decltype(cos((long)0)), double>::value, "");
    static_assert(is_same<decltype(cos((unsigned long)0)), double>::value, "");
    static_assert(is_same<decltype(cos((long long)0)), double>::value, "");
    static_assert(is_same<decltype(cos((unsigned long long)0)), double>::value, "");
    static_assert(is_same<decltype(cos((double)0)), double>::value, "");
    static_assert(is_same<decltype(cosf(0)), float>::value, "");
}

__device__ void test_cosh()
{
    static_assert(is_same<decltype(cosh((float)0)), float>::value, "");
    static_assert(is_same<decltype(cosh((bool)0)), double>::value, "");
    static_assert(is_same<decltype(cosh((unsigned short)0)), double>::value, "");
    static_assert(is_same<decltype(cosh((int)0)), double>::value, "");
    static_assert(is_same<decltype(cosh((unsigned int)0)), double>::value, "");
    static_assert(is_same<decltype(cosh((long)0)), double>::value, "");
    static_assert(is_same<decltype(cosh((unsigned long)0)), double>::value, "");
    static_assert(is_same<decltype(cosh((long long)0)), double>::value, "");
    static_assert(is_same<decltype(cosh((unsigned long long)0)), double>::value, "");
    static_assert(is_same<decltype(cosh((double)0)), double>::value, "");
    static_assert(is_same<decltype(coshf(0)), float>::value, "");
}

__device__ void test_exp()
{
    static_assert(is_same<decltype(exp((float)0)), float>::value, "");
    static_assert(is_same<decltype(exp((bool)0)), double>::value, "");
    static_assert(is_same<decltype(exp((unsigned short)0)), double>::value, "");
    static_assert(is_same<decltype(exp((int)0)), double>::value, "");
    static_assert(is_same<decltype(exp((unsigned int)0)), double>::value, "");
    static_assert(is_same<decltype(exp((long)0)), double>::value, "");
    static_assert(is_same<decltype(exp((unsigned long)0)), double>::value, "");
    static_assert(is_same<decltype(exp((long long)0)), double>::value, "");
    static_assert(is_same<decltype(exp((unsigned long long)0)), double>::value, "");
    static_assert(is_same<decltype(exp((double)0)), double>::value, "");
    static_assert(is_same<decltype(expf(0)), float>::value, "");
}

__device__ void test_fabs()
{
    static_assert(is_same<decltype(fabs((float)0)), float>::value, "");
    static_assert(is_same<decltype(fabs((bool)0)), double>::value, "");
    static_assert(is_same<decltype(fabs((unsigned short)0)), double>::value, "");
    static_assert(is_same<decltype(fabs((int)0)), double>::value, "");
    static_assert(is_same<decltype(fabs((unsigned int)0)), double>::value, "");
    static_assert(is_same<decltype(fabs((long)0)), double>::value, "");
    static_assert(is_same<decltype(fabs((unsigned long)0)), double>::value, "");
    static_assert(is_same<decltype(fabs((long long)0)), double>::value, "");
    static_assert(is_same<decltype(fabs((unsigned long long)0)), double>::value, "");
    static_assert(is_same<decltype(fabs((double)0)), double>::value, "");
    static_assert(is_same<decltype(fabsf(0.0f)), float>::value, "");
}

__device__ void test_floor()
{
    static_assert(is_same<decltype(floor((float)0)), float>::value, "");
    static_assert(is_same<decltype(floor((bool)0)), double>::value, "");
    static_assert(is_same<decltype(floor((unsigned short)0)), double>::value, "");
    static_assert(is_same<decltype(floor((int)0)), double>::value, "");
    static_assert(is_same<decltype(floor((unsigned int)0)), double>::value, "");
    static_assert(is_same<decltype(floor((long)0)), double>::value, "");
    static_assert(is_same<decltype(floor((unsigned long)0)), double>::value, "");
    static_assert(is_same<decltype(floor((long long)0)), double>::value, "");
    static_assert(is_same<decltype(floor((unsigned long long)0)), double>::value, "");
    static_assert(is_same<decltype(floor((double)0)), double>::value, "");
    static_assert(is_same<decltype(floorf(0)), float>::value, "");
}

__device__ void test_fmod()
{
    static_assert(is_same<decltype(fmod((float)0, (float)0)), float>::value, "");
    static_assert(is_same<decltype(fmod((bool)0, (float)0)), double>::value, "");
    static_assert(is_same<decltype(fmod((unsigned short)0, (double)0)), double>::value, "");
    static_assert(is_same<decltype(fmod((float)0, (unsigned int)0)), double>::value, "");
    static_assert(is_same<decltype(fmod((double)0, (long)0)), double>::value, "");
    static_assert(is_same<decltype(fmod((int)0, (long long)0)), double>::value, "");
    static_assert(is_same<decltype(fmod((int)0, (unsigned long long)0)), double>::value, "");
    static_assert(is_same<decltype(fmod((double)0, (double)0)), double>::value, "");
    static_assert(is_same<decltype(fmod((float)0, (double)0)), double>::value, "");
    static_assert(is_same<decltype(fmodf(0,0)), float>::value, "");
    static_assert(is_same<decltype(fmod((int)0, (int)0)), double>::value, "");
}

__device__ void test_frexp()
{
    int ip;
    static_assert(is_same<decltype(frexp((float)0, &ip)), float>::value, "");
    static_assert(is_same<decltype(frexp((bool)0, &ip)), double>::value, "");
    static_assert(is_same<decltype(frexp((unsigned short)0, &ip)), double>::value, "");
    static_assert(is_same<decltype(frexp((int)0, &ip)), double>::value, "");
    static_assert(is_same<decltype(frexp((unsigned int)0, &ip)), double>::value, "");
    static_assert(is_same<decltype(frexp((long)0, &ip)), double>::value, "");
    static_assert(is_same<decltype(frexp((unsigned long)0, &ip)), double>::value, "");
    static_assert(is_same<decltype(frexp((long long)0, &ip)), double>::value, "");
    static_assert(is_same<decltype(frexp((unsigned long long)0, &ip)), double>::value, "");
    static_assert(is_same<decltype(frexp((double)0, &ip)), double>::value, "");
    static_assert(is_same<decltype(frexpf(0, &ip)), float>::value, "");
}

__device__ void test_ldexp()
{
    int ip = 1;
    static_assert(is_same<decltype(ldexp((float)0, ip)), float>::value, "");
    static_assert(is_same<decltype(ldexp((bool)0, ip)), double>::value, "");
    static_assert(is_same<decltype(ldexp((unsigned short)0, ip)), double>::value, "");
    static_assert(is_same<decltype(ldexp((int)0, ip)), double>::value, "");
    static_assert(is_same<decltype(ldexp((unsigned int)0, ip)), double>::value, "");
    static_assert(is_same<decltype(ldexp((long)0, ip)), double>::value, "");
    static_assert(is_same<decltype(ldexp((unsigned long)0, ip)), double>::value, "");
    static_assert(is_same<decltype(ldexp((long long)0, ip)), double>::value, "");
    static_assert(is_same<decltype(ldexp((unsigned long long)0, ip)), double>::value, "");
    static_assert(is_same<decltype(ldexp((double)0, ip)), double>::value, "");
    static_assert(is_same<decltype(ldexpf(0, ip)), float>::value, "");
}

__device__ void test_log()
{
    static_assert(is_same<decltype(log((float)0)), float>::value, "");
    static_assert(is_same<decltype(log((bool)0)), double>::value, "");
    static_assert(is_same<decltype(log((unsigned short)0)), double>::value, "");
    static_assert(is_same<decltype(log((int)0)), double>::value, "");
    static_assert(is_same<decltype(log((unsigned int)0)), double>::value, "");
    static_assert(is_same<decltype(log((long)0)), double>::value, "");
    static_assert(is_same<decltype(log((unsigned long)0)), double>::value, "");
    static_assert(is_same<decltype(log((long long)0)), double>::value, "");
    static_assert(is_same<decltype(log((unsigned long long)0)), double>::value, "");
    static_assert(is_same<decltype(log((double)0)), double>::value, "");
    static_assert(is_same<decltype(logf(0)), float>::value, "");
}

__device__ void test_log10()
{
    static_assert(is_same<decltype(log10((float)0)), float>::value, "");
    static_assert(is_same<decltype(log10((bool)0)), double>::value, "");
    static_assert(is_same<decltype(log10((unsigned short)0)), double>::value, "");
    static_assert(is_same<decltype(log10((int)0)), double>::value, "");
    static_assert(is_same<decltype(log10((unsigned int)0)), double>::value, "");
    static_assert(is_same<decltype(log10((long)0)), double>::value, "");
    static_assert(is_same<decltype(log10((unsigned long)0)), double>::value, "");
    static_assert(is_same<decltype(log10((long long)0)), double>::value, "");
    static_assert(is_same<decltype(log10((unsigned long long)0)), double>::value, "");
    static_assert(is_same<decltype(log10((double)0)), double>::value, "");
    static_assert(is_same<decltype(log10f(0)), float>::value, "");
}

__device__ void test_modf()
{
    static_assert(is_same<decltype(modf((float)0, (float*)0)), float>::value, "");
    static_assert(is_same<decltype(modf((double)0, (double*)0)), double>::value, "");
    static_assert(is_same<decltype(modff(0, (float*)0)), float>::value, "");
    double i;
}

__device__ void test_pow()
{
    static_assert(is_same<decltype(pow((float)0, (float)0)), float>::value, "");
    static_assert(is_same<decltype(pow((bool)0, (float)0)), double>::value, "");
    static_assert(is_same<decltype(pow((unsigned short)0, (double)0)), double>::value, "");
    static_assert(is_same<decltype(pow((float)0, (unsigned int)0)), double>::value, "");
    static_assert(is_same<decltype(pow((double)0, (long)0)), double>::value, "");
    static_assert(is_same<decltype(pow((int)0, (long long)0)), double>::value, "");
    static_assert(is_same<decltype(pow((int)0, (unsigned long long)0)), double>::value, "");
    static_assert(is_same<decltype(pow((double)0, (double)0)), double>::value, "");
    static_assert(is_same<decltype(pow((float)0, (double)0)), double>::value, "");
    static_assert(is_same<decltype(powf(0,0)), float>::value, "");
    static_assert(is_same<decltype(pow((int)0, (int)0)), double>::value, "");
}

__device__ void test_sin()
{
    static_assert(is_same<decltype(sin((float)0)), float>::value, "");
    static_assert(is_same<decltype(sin((bool)0)), double>::value, "");
    static_assert(is_same<decltype(sin((unsigned short)0)), double>::value, "");
    static_assert(is_same<decltype(sin((int)0)), double>::value, "");
    static_assert(is_same<decltype(sin((unsigned int)0)), double>::value, "");
    static_assert(is_same<decltype(sin((long)0)), double>::value, "");
    static_assert(is_same<decltype(sin((unsigned long)0)), double>::value, "");
    static_assert(is_same<decltype(sin((long long)0)), double>::value, "");
    static_assert(is_same<decltype(sin((unsigned long long)0)), double>::value, "");
    static_assert(is_same<decltype(sin((double)0)), double>::value, "");
    static_assert(is_same<decltype(sinf(0)), float>::value, "");
}

__device__ void test_sinh()
{
    static_assert(is_same<decltype(sinh((float)0)), float>::value, "");
    static_assert(is_same<decltype(sinh((bool)0)), double>::value, "");
    static_assert(is_same<decltype(sinh((unsigned short)0)), double>::value, "");
    static_assert(is_same<decltype(sinh((int)0)), double>::value, "");
    static_assert(is_same<decltype(sinh((unsigned int)0)), double>::value, "");
    static_assert(is_same<decltype(sinh((long)0)), double>::value, "");
    static_assert(is_same<decltype(sinh((unsigned long)0)), double>::value, "");
    static_assert(is_same<decltype(sinh((long long)0)), double>::value, "");
    static_assert(is_same<decltype(sinh((unsigned long long)0)), double>::value, "");
    static_assert(is_same<decltype(sinh((double)0)), double>::value, "");
    static_assert(is_same<decltype(sinhf(0)), float>::value, "");
}

__device__ void test_sqrt()
{
    static_assert(is_same<decltype(sqrt((float)0)), float>::value, "");
    static_assert(is_same<decltype(sqrt((bool)0)), double>::value, "");
    static_assert(is_same<decltype(sqrt((unsigned short)0)), double>::value, "");
    static_assert(is_same<decltype(sqrt((int)0)), double>::value, "");
    static_assert(is_same<decltype(sqrt((unsigned int)0)), double>::value, "");
    static_assert(is_same<decltype(sqrt((long)0)), double>::value, "");
    static_assert(is_same<decltype(sqrt((unsigned long)0)), double>::value, "");
    static_assert(is_same<decltype(sqrt((long long)0)), double>::value, "");
    static_assert(is_same<decltype(sqrt((unsigned long long)0)), double>::value, "");
    static_assert(is_same<decltype(sqrt((double)0)), double>::value, "");
    static_assert(is_same<decltype(sqrtf(0)), float>::value, "");
}

__device__ void test_tan()
{
    static_assert(is_same<decltype(tan((float)0)), float>::value, "");
    static_assert(is_same<decltype(tan((bool)0)), double>::value, "");
    static_assert(is_same<decltype(tan((unsigned short)0)), double>::value, "");
    static_assert(is_same<decltype(tan((int)0)), double>::value, "");
    static_assert(is_same<decltype(tan((unsigned int)0)), double>::value, "");
    static_assert(is_same<decltype(tan((long)0)), double>::value, "");
    static_assert(is_same<decltype(tan((unsigned long)0)), double>::value, "");
    static_assert(is_same<decltype(tan((long long)0)), double>::value, "");
    static_assert(is_same<decltype(tan((unsigned long long)0)), double>::value, "");
    static_assert(is_same<decltype(tan((double)0)), double>::value, "");
    static_assert(is_same<decltype(tanf(0)), float>::value, "");
}

__device__ void test_tanh()
{
    static_assert(is_same<decltype(tanh((float)0)), float>::value, "");
    static_assert(is_same<decltype(tanh((bool)0)), double>::value, "");
    static_assert(is_same<decltype(tanh((unsigned short)0)), double>::value, "");
    static_assert(is_same<decltype(tanh((int)0)), double>::value, "");
    static_assert(is_same<decltype(tanh((unsigned int)0)), double>::value, "");
    static_assert(is_same<decltype(tanh((long)0)), double>::value, "");
    static_assert(is_same<decltype(tanh((unsigned long)0)), double>::value, "");
    static_assert(is_same<decltype(tanh((long long)0)), double>::value, "");
    static_assert(is_same<decltype(tanh((unsigned long long)0)), double>::value, "");
    static_assert(is_same<decltype(tanh((double)0)), double>::value, "");
    static_assert(is_same<decltype(tanhf(0)), float>::value, "");
}

__device__ void test_signbit()
{
    static_assert(is_same<decltype(signbit((float)0)), bool>::value, "");
    static_assert(is_same<decltype(signbit((double)0)), bool>::value, "");
    static_assert(is_same<decltype(signbit(0)), bool>::value, "");
}

__device__ void test_fpclassify()
{
    static_assert(is_same<decltype(fpclassify((float)0)), int>::value, "");
    static_assert(is_same<decltype(fpclassify((double)0)), int>::value, "");
    static_assert(is_same<decltype(fpclassify(0)), int>::value, "");
}

__device__ void test_isfinite()
{
    static_assert(is_same<decltype(isfinite((float)0)), bool>::value, "");
    static_assert(is_same<decltype(isfinite((double)0)), bool>::value, "");
    static_assert(is_same<decltype(isfinite(0)), bool>::value, "");
}

__device__ void test_isnormal()
{
    static_assert(is_same<decltype(isnormal((float)0)), bool>::value, "");
    static_assert(is_same<decltype(isnormal((double)0)), bool>::value, "");
    static_assert(is_same<decltype(isnormal(0)), bool>::value, "");
}

__device__ void test_isgreater()
{
    static_assert(is_same<decltype(isgreater((float)0, (float)0)), bool>::value, "");
    static_assert(is_same<decltype(isgreater((float)0, (double)0)), bool>::value, "");
    static_assert(is_same<decltype(isgreater((double)0, (float)0)), bool>::value, "");
    static_assert(is_same<decltype(isgreater((double)0, (double)0)), bool>::value, "");
    static_assert(is_same<decltype(isgreater(0, (double)0)), bool>::value, "");
}

__device__ void test_isgreaterequal()
{
    static_assert(is_same<decltype(isgreaterequal((float)0, (float)0)), bool>::value, "");
    static_assert(is_same<decltype(isgreaterequal((float)0, (double)0)), bool>::value, "");
    static_assert(is_same<decltype(isgreaterequal((double)0, (float)0)), bool>::value, "");
    static_assert(is_same<decltype(isgreaterequal((double)0, (double)0)), bool>::value, "");
    static_assert(is_same<decltype(isgreaterequal(0, (double)0)), bool>::value, "");
}

__device__ void test_isinf()
{
    static_assert(is_same<decltype(isinf((float)0)), bool>::value, "");
    static_assert(is_same<decltype(isinf((double)0)), bool>::value, "");
}

__device__ void test_isless()
{
    static_assert(is_same<decltype(isless((float)0, (float)0)), bool>::value, "");
    static_assert(is_same<decltype(isless((float)0, (double)0)), bool>::value, "");
    static_assert(is_same<decltype(isless((double)0, (float)0)), bool>::value, "");
    static_assert(is_same<decltype(isless((double)0, (double)0)), bool>::value, "");
    static_assert(is_same<decltype(isless(0, (double)0)), bool>::value, "");
}

__device__ void test_islessequal()
{
    static_assert(is_same<decltype(islessequal((float)0, (float)0)), bool>::value, "");
    static_assert(is_same<decltype(islessequal((float)0, (double)0)), bool>::value, "");
    static_assert(is_same<decltype(islessequal((double)0, (float)0)), bool>::value, "");
    static_assert(is_same<decltype(islessequal((double)0, (double)0)), bool>::value, "");
    static_assert(is_same<decltype(islessequal(0, (double)0)), bool>::value, "");
}

__device__ void test_islessgreater()
{
    static_assert(is_same<decltype(islessgreater((float)0, (float)0)), bool>::value, "");
    static_assert(is_same<decltype(islessgreater((float)0, (double)0)), bool>::value, "");
    static_assert(is_same<decltype(islessgreater((double)0, (float)0)), bool>::value, "");
    static_assert(is_same<decltype(islessgreater((double)0, (double)0)), bool>::value, "");
    static_assert(is_same<decltype(islessgreater(0, (double)0)), bool>::value, "");
}

__device__ void test_isnan()
{
    static_assert(is_same<decltype(isnan((float)0)), bool>::value, "");
    static_assert(is_same<decltype(isnan((double)0)), bool>::value, "");
}

__device__ void test_isunordered()
{
    static_assert(is_same<decltype(isunordered((float)0, (float)0)), bool>::value, "");
    static_assert(is_same<decltype(isunordered((float)0, (double)0)), bool>::value, "");
    static_assert(is_same<decltype(isunordered((double)0, (float)0)), bool>::value, "");
    static_assert(is_same<decltype(isunordered((double)0, (double)0)), bool>::value, "");
    static_assert(is_same<decltype(isunordered(0, (double)0)), bool>::value, "");
}

__device__ void test_acosh()
{
    static_assert(is_same<decltype(acosh((float)0)), float>::value, "");
    static_assert(is_same<decltype(acosh((bool)0)), double>::value, "");
    static_assert(is_same<decltype(acosh((unsigned short)0)), double>::value, "");
    static_assert(is_same<decltype(acosh((int)0)), double>::value, "");
    static_assert(is_same<decltype(acosh((unsigned int)0)), double>::value, "");
    static_assert(is_same<decltype(acosh((long)0)), double>::value, "");
    static_assert(is_same<decltype(acosh((unsigned long)0)), double>::value, "");
    static_assert(is_same<decltype(acosh((long long)0)), double>::value, "");
    static_assert(is_same<decltype(acosh((unsigned long long)0)), double>::value, "");
    static_assert(is_same<decltype(acosh((double)0)), double>::value, "");
    static_assert(is_same<decltype(acoshf(0)), float>::value, "");
}

__device__ void test_asinh()
{
    static_assert(is_same<decltype(asinh((float)0)), float>::value, "");
    static_assert(is_same<decltype(asinh((bool)0)), double>::value, "");
    static_assert(is_same<decltype(asinh((unsigned short)0)), double>::value, "");
    static_assert(is_same<decltype(asinh((int)0)), double>::value, "");
    static_assert(is_same<decltype(asinh((unsigned int)0)), double>::value, "");
    static_assert(is_same<decltype(asinh((long)0)), double>::value, "");
    static_assert(is_same<decltype(asinh((unsigned long)0)), double>::value, "");
    static_assert(is_same<decltype(asinh((long long)0)), double>::value, "");
    static_assert(is_same<decltype(asinh((unsigned long long)0)), double>::value, "");
    static_assert(is_same<decltype(asinh((double)0)), double>::value, "");
    static_assert(is_same<decltype(asinhf(0)), float>::value, "");
}

__device__ void test_atanh()
{
    static_assert(is_same<decltype(atanh((float)0)), float>::value, "");
    static_assert(is_same<decltype(atanh((bool)0)), double>::value, "");
    static_assert(is_same<decltype(atanh((unsigned short)0)), double>::value, "");
    static_assert(is_same<decltype(atanh((int)0)), double>::value, "");
    static_assert(is_same<decltype(atanh((unsigned int)0)), double>::value, "");
    static_assert(is_same<decltype(atanh((long)0)), double>::value, "");
    static_assert(is_same<decltype(atanh((unsigned long)0)), double>::value, "");
    static_assert(is_same<decltype(atanh((long long)0)), double>::value, "");
    static_assert(is_same<decltype(atanh((unsigned long long)0)), double>::value, "");
    static_assert(is_same<decltype(atanh((double)0)), double>::value, "");
    static_assert(is_same<decltype(atanhf(0)), float>::value, "");
}

__device__ void test_cbrt()
{
    static_assert(is_same<decltype(cbrt((float)0)), float>::value, "");
    static_assert(is_same<decltype(cbrt((bool)0)), double>::value, "");
    static_assert(is_same<decltype(cbrt((unsigned short)0)), double>::value, "");
    static_assert(is_same<decltype(cbrt((int)0)), double>::value, "");
    static_assert(is_same<decltype(cbrt((unsigned int)0)), double>::value, "");
    static_assert(is_same<decltype(cbrt((long)0)), double>::value, "");
    static_assert(is_same<decltype(cbrt((unsigned long)0)), double>::value, "");
    static_assert(is_same<decltype(cbrt((long long)0)), double>::value, "");
    static_assert(is_same<decltype(cbrt((unsigned long long)0)), double>::value, "");
    static_assert(is_same<decltype(cbrt((double)0)), double>::value, "");
    static_assert(is_same<decltype(cbrtf(0)), float>::value, "");
}

__device__ void test_copysign()
{
    static_assert(is_same<decltype(copysign((float)0, (float)0)), float>::value, "");
    static_assert(is_same<decltype(copysign((bool)0, (float)0)), double>::value, "");
    static_assert(is_same<decltype(copysign((unsigned short)0, (double)0)), double>::value, "");
    static_assert(is_same<decltype(copysign((float)0, (unsigned int)0)), double>::value, "");
    static_assert(is_same<decltype(copysign((double)0, (long)0)), double>::value, "");
    static_assert(is_same<decltype(copysign((int)0, (long long)0)), double>::value, "");
    static_assert(is_same<decltype(copysign((int)0, (unsigned long long)0)), double>::value, "");
    static_assert(is_same<decltype(copysign((double)0, (double)0)), double>::value, "");
    static_assert(is_same<decltype(copysign((float)0, (double)0)), double>::value, "");
    static_assert(is_same<decltype(copysignf(0,0)), float>::value, "");
    static_assert(is_same<decltype(copysign((int)0, (int)0)), double>::value, "");
}

__device__ void test_erf()
{
    static_assert(is_same<decltype(erf((float)0)), float>::value, "");
    static_assert(is_same<decltype(erf((bool)0)), double>::value, "");
    static_assert(is_same<decltype(erf((unsigned short)0)), double>::value, "");
    static_assert(is_same<decltype(erf((int)0)), double>::value, "");
    static_assert(is_same<decltype(erf((unsigned int)0)), double>::value, "");
    static_assert(is_same<decltype(erf((long)0)), double>::value, "");
    static_assert(is_same<decltype(erf((unsigned long)0)), double>::value, "");
    static_assert(is_same<decltype(erf((long long)0)), double>::value, "");
    static_assert(is_same<decltype(erf((unsigned long long)0)), double>::value, "");
    static_assert(is_same<decltype(erf((double)0)), double>::value, "");
    static_assert(is_same<decltype(erff(0)), float>::value, "");
}

__device__ void test_erfc()
{
    static_assert(is_same<decltype(erfc((float)0)), float>::value, "");
    static_assert(is_same<decltype(erfc((bool)0)), double>::value, "");
    static_assert(is_same<decltype(erfc((unsigned short)0)), double>::value, "");
    static_assert(is_same<decltype(erfc((int)0)), double>::value, "");
    static_assert(is_same<decltype(erfc((unsigned int)0)), double>::value, "");
    static_assert(is_same<decltype(erfc((long)0)), double>::value, "");
    static_assert(is_same<decltype(erfc((unsigned long)0)), double>::value, "");
    static_assert(is_same<decltype(erfc((long long)0)), double>::value, "");
    static_assert(is_same<decltype(erfc((unsigned long long)0)), double>::value, "");
    static_assert(is_same<decltype(erfc((double)0)), double>::value, "");
    static_assert(is_same<decltype(erfcf(0)), float>::value, "");
}

__device__ void test_exp2()
{
    static_assert(is_same<decltype(exp2((float)0)), float>::value, "");
    static_assert(is_same<decltype(exp2((bool)0)), double>::value, "");
    static_assert(is_same<decltype(exp2((unsigned short)0)), double>::value, "");
    static_assert(is_same<decltype(exp2((int)0)), double>::value, "");
    static_assert(is_same<decltype(exp2((unsigned int)0)), double>::value, "");
    static_assert(is_same<decltype(exp2((long)0)), double>::value, "");
    static_assert(is_same<decltype(exp2((unsigned long)0)), double>::value, "");
    static_assert(is_same<decltype(exp2((long long)0)), double>::value, "");
    static_assert(is_same<decltype(exp2((unsigned long long)0)), double>::value, "");
    static_assert(is_same<decltype(exp2((double)0)), double>::value, "");
    static_assert(is_same<decltype(exp2f(0)), float>::value, "");
}

__device__ void test_expm1()
{
    static_assert(is_same<decltype(expm1((float)0)), float>::value, "");
    static_assert(is_same<decltype(expm1((bool)0)), double>::value, "");
    static_assert(is_same<decltype(expm1((unsigned short)0)), double>::value, "");
    static_assert(is_same<decltype(expm1((int)0)), double>::value, "");
    static_assert(is_same<decltype(expm1((unsigned int)0)), double>::value, "");
    static_assert(is_same<decltype(expm1((long)0)), double>::value, "");
    static_assert(is_same<decltype(expm1((unsigned long)0)), double>::value, "");
    static_assert(is_same<decltype(expm1((long long)0)), double>::value, "");
    static_assert(is_same<decltype(expm1((unsigned long long)0)), double>::value, "");
    static_assert(is_same<decltype(expm1((double)0)), double>::value, "");
    static_assert(is_same<decltype(expm1f(0)), float>::value, "");
}

__device__ void test_fdim()
{
    static_assert(is_same<decltype(fdim((float)0, (float)0)), float>::value, "");
    static_assert(is_same<decltype(fdim((bool)0, (float)0)), double>::value, "");
    static_assert(is_same<decltype(fdim((unsigned short)0, (double)0)), double>::value, "");
    static_assert(is_same<decltype(fdim((float)0, (unsigned int)0)), double>::value, "");
    static_assert(is_same<decltype(fdim((double)0, (long)0)), double>::value, "");
    static_assert(is_same<decltype(fdim((int)0, (long long)0)), double>::value, "");
    static_assert(is_same<decltype(fdim((int)0, (unsigned long long)0)), double>::value, "");
    static_assert(is_same<decltype(fdim((double)0, (double)0)), double>::value, "");
    static_assert(is_same<decltype(fdim((float)0, (double)0)), double>::value, "");
    static_assert(is_same<decltype(fdimf(0,0)), float>::value, "");
    static_assert(is_same<decltype(fdim((int)0, (int)0)), double>::value, "");
}

__device__ void test_fma()
{
    static_assert(is_same<decltype(fma((bool)0, (float)0, (float)0)), double>::value, "");
    static_assert(is_same<decltype(fma((char)0, (float)0, (float)0)), double>::value, "");
    static_assert(is_same<decltype(fma((unsigned)0, (float)0, (float)0)), double>::value, "");
    static_assert(is_same<decltype(fma((float)0, (int)0, (float)0)), double>::value, "");
    static_assert(is_same<decltype(fma((float)0, (long)0, (float)0)), double>::value, "");
    static_assert(is_same<decltype(fma((float)0, (float)0, (unsigned long long)0)), double>::value, "");
    static_assert(is_same<decltype(fma((float)0, (float)0, (double)0)), double>::value, "");
    static_assert(is_same<decltype(fma((float)0, (float)0, (float)0)), float>::value, "");

    static_assert(is_same<decltype(fma((bool)0, (double)0, (double)0)), double>::value, "");
    static_assert(is_same<decltype(fma((char)0, (double)0, (double)0)), double>::value, "");
    static_assert(is_same<decltype(fma((unsigned)0, (double)0, (double)0)), double>::value, "");
    static_assert(is_same<decltype(fma((double)0, (int)0, (double)0)), double>::value, "");
    static_assert(is_same<decltype(fma((double)0, (long)0, (double)0)), double>::value, "");
    static_assert(is_same<decltype(fma((double)0, (double)0, (unsigned long long)0)), double>::value, "");
    static_assert(is_same<decltype(fma((double)0, (double)0, (float)0)), double>::value, "");
    static_assert(is_same<decltype(fma((double)0, (double)0,  (double)0)), double>::value, "");

    static_assert(is_same<decltype(fmaf(0,0,0)), float>::value, "");
}

__device__ void test_fmax()
{
    static_assert(is_same<decltype(fmax((float)0, (float)0)), float>::value, "");
    static_assert(is_same<decltype(fmax((bool)0, (float)0)), double>::value, "");
    static_assert(is_same<decltype(fmax((unsigned short)0, (double)0)), double>::value, "");
    static_assert(is_same<decltype(fmax((float)0, (unsigned int)0)), double>::value, "");
    static_assert(is_same<decltype(fmax((double)0, (long)0)), double>::value, "");
    static_assert(is_same<decltype(fmax((int)0, (long long)0)), double>::value, "");
    static_assert(is_same<decltype(fmax((int)0, (unsigned long long)0)), double>::value, "");
    static_assert(is_same<decltype(fmax((double)0, (double)0)), double>::value, "");
    static_assert(is_same<decltype(fmax((float)0, (double)0)), double>::value, "");
    static_assert(is_same<decltype(fmaxf(0,0)), float>::value, "");
    static_assert(is_same<decltype(fmax((int)0, (int)0)), double>::value, "");
}

__device__ void test_fmin()
{
    static_assert(is_same<decltype(fmin((float)0, (float)0)), float>::value, "");
    static_assert(is_same<decltype(fmin((bool)0, (float)0)), double>::value, "");
    static_assert(is_same<decltype(fmin((unsigned short)0, (double)0)), double>::value, "");
    static_assert(is_same<decltype(fmin((float)0, (unsigned int)0)), double>::value, "");
    static_assert(is_same<decltype(fmin((double)0, (long)0)), double>::value, "");
    static_assert(is_same<decltype(fmin((int)0, (long long)0)), double>::value, "");
    static_assert(is_same<decltype(fmin((int)0, (unsigned long long)0)), double>::value, "");
    static_assert(is_same<decltype(fmin((double)0, (double)0)), double>::value, "");
    static_assert(is_same<decltype(fmin((float)0, (double)0)), double>::value, "");
    static_assert(is_same<decltype(fminf(0,0)), float>::value, "");
    static_assert(is_same<decltype(fmin((int)0, (int)0)), double>::value, "");
}

__device__ void test_hypot()
{
    static_assert(is_same<decltype(hypot((float)0, (float)0)), float>::value, "");
    static_assert(is_same<decltype(hypot((bool)0, (float)0)), double>::value, "");
    static_assert(is_same<decltype(hypot((unsigned short)0, (double)0)), double>::value, "");
    static_assert(is_same<decltype(hypot((float)0, (unsigned int)0)), double>::value, "");
    static_assert(is_same<decltype(hypot((double)0, (long)0)), double>::value, "");
    static_assert(is_same<decltype(hypot((int)0, (long long)0)), double>::value, "");
    static_assert(is_same<decltype(hypot((int)0, (unsigned long long)0)), double>::value, "");
    static_assert(is_same<decltype(hypot((double)0, (double)0)), double>::value, "");
    static_assert(is_same<decltype(hypot((float)0, (double)0)), double>::value, "");
    static_assert(is_same<decltype(hypotf(0,0)), float>::value, "");
    static_assert(is_same<decltype(hypot((int)0, (int)0)), double>::value, "");
}

__device__ void test_ilogb()
{
    static_assert(is_same<decltype(ilogb((float)0)), int>::value, "");
    static_assert(is_same<decltype(ilogb((bool)0)), int>::value, "");
    static_assert(is_same<decltype(ilogb((unsigned short)0)), int>::value, "");
    static_assert(is_same<decltype(ilogb((int)0)), int>::value, "");
    static_assert(is_same<decltype(ilogb((unsigned int)0)), int>::value, "");
    static_assert(is_same<decltype(ilogb((long)0)), int>::value, "");
    static_assert(is_same<decltype(ilogb((unsigned long)0)), int>::value, "");
    static_assert(is_same<decltype(ilogb((long long)0)), int>::value, "");
    static_assert(is_same<decltype(ilogb((unsigned long long)0)), int>::value, "");
    static_assert(is_same<decltype(ilogb((double)0)), int>::value, "");
    static_assert(is_same<decltype(ilogbf(0)), int>::value, "");
}

__device__ void test_lgamma()
{
    static_assert(is_same<decltype(lgamma((float)0)), float>::value, "");
    static_assert(is_same<decltype(lgamma((bool)0)), double>::value, "");
    static_assert(is_same<decltype(lgamma((unsigned short)0)), double>::value, "");
    static_assert(is_same<decltype(lgamma((int)0)), double>::value, "");
    static_assert(is_same<decltype(lgamma((unsigned int)0)), double>::value, "");
    static_assert(is_same<decltype(lgamma((long)0)), double>::value, "");
    static_assert(is_same<decltype(lgamma((unsigned long)0)), double>::value, "");
    static_assert(is_same<decltype(lgamma((long long)0)), double>::value, "");
    static_assert(is_same<decltype(lgamma((unsigned long long)0)), double>::value, "");
    static_assert(is_same<decltype(lgamma((double)0)), double>::value, "");
    static_assert(is_same<decltype(lgammaf(0)), float>::value, "");
}

__device__ void test_llrint()
{
    static_assert(is_same<decltype(llrint((float)0)), long long>::value, "");
    static_assert(is_same<decltype(llrint((bool)0)), long long>::value, "");
    static_assert(is_same<decltype(llrint((unsigned short)0)), long long>::value, "");
    static_assert(is_same<decltype(llrint((int)0)), long long>::value, "");
    static_assert(is_same<decltype(llrint((unsigned int)0)), long long>::value, "");
    static_assert(is_same<decltype(llrint((long)0)), long long>::value, "");
    static_assert(is_same<decltype(llrint((unsigned long)0)), long long>::value, "");
    static_assert(is_same<decltype(llrint((long long)0)), long long>::value, "");
    static_assert(is_same<decltype(llrint((unsigned long long)0)), long long>::value, "");
    static_assert(is_same<decltype(llrint((double)0)), long long>::value, "");
    static_assert(is_same<decltype(llrintf(0)), long long>::value, "");
}

__device__ void test_llround()
{
    static_assert(is_same<decltype(llround((float)0)), long long>::value, "");
    static_assert(is_same<decltype(llround((bool)0)), long long>::value, "");
    static_assert(is_same<decltype(llround((unsigned short)0)), long long>::value, "");
    static_assert(is_same<decltype(llround((int)0)), long long>::value, "");
    static_assert(is_same<decltype(llround((unsigned int)0)), long long>::value, "");
    static_assert(is_same<decltype(llround((long)0)), long long>::value, "");
    static_assert(is_same<decltype(llround((unsigned long)0)), long long>::value, "");
    static_assert(is_same<decltype(llround((long long)0)), long long>::value, "");
    static_assert(is_same<decltype(llround((unsigned long long)0)), long long>::value, "");
    static_assert(is_same<decltype(llround((double)0)), long long>::value, "");
    static_assert(is_same<decltype(llroundf(0)), long long>::value, "");
}

__device__ void test_log1p()
{
    static_assert(is_same<decltype(log1p((float)0)), float>::value, "");
    static_assert(is_same<decltype(log1p((bool)0)), double>::value, "");
    static_assert(is_same<decltype(log1p((unsigned short)0)), double>::value, "");
    static_assert(is_same<decltype(log1p((int)0)), double>::value, "");
    static_assert(is_same<decltype(log1p((unsigned int)0)), double>::value, "");
    static_assert(is_same<decltype(log1p((long)0)), double>::value, "");
    static_assert(is_same<decltype(log1p((unsigned long)0)), double>::value, "");
    static_assert(is_same<decltype(log1p((long long)0)), double>::value, "");
    static_assert(is_same<decltype(log1p((unsigned long long)0)), double>::value, "");
    static_assert(is_same<decltype(log1p((double)0)), double>::value, "");
    static_assert(is_same<decltype(log1pf(0)), float>::value, "");
}

__device__ void test_log2()
{
    static_assert(is_same<decltype(log2((float)0)), float>::value, "");
    static_assert(is_same<decltype(log2((bool)0)), double>::value, "");
    static_assert(is_same<decltype(log2((unsigned short)0)), double>::value, "");
    static_assert(is_same<decltype(log2((int)0)), double>::value, "");
    static_assert(is_same<decltype(log2((unsigned int)0)), double>::value, "");
    static_assert(is_same<decltype(log2((long)0)), double>::value, "");
    static_assert(is_same<decltype(log2((unsigned long)0)), double>::value, "");
    static_assert(is_same<decltype(log2((long long)0)), double>::value, "");
    static_assert(is_same<decltype(log2((unsigned long long)0)), double>::value, "");
    static_assert(is_same<decltype(log2((double)0)), double>::value, "");
    static_assert(is_same<decltype(log2f(0)), float>::value, "");
}

__device__ void test_logb()
{
    static_assert(is_same<decltype(logb((float)0)), float>::value, "");
    static_assert(is_same<decltype(logb((bool)0)), double>::value, "");
    static_assert(is_same<decltype(logb((unsigned short)0)), double>::value, "");
    static_assert(is_same<decltype(logb((int)0)), double>::value, "");
    static_assert(is_same<decltype(logb((unsigned int)0)), double>::value, "");
    static_assert(is_same<decltype(logb((long)0)), double>::value, "");
    static_assert(is_same<decltype(logb((unsigned long)0)), double>::value, "");
    static_assert(is_same<decltype(logb((long long)0)), double>::value, "");
    static_assert(is_same<decltype(logb((unsigned long long)0)), double>::value, "");
    static_assert(is_same<decltype(logb((double)0)), double>::value, "");
    static_assert(is_same<decltype(logbf(0)), float>::value, "");
}

__device__ void test_lrint()
{
    static_assert(is_same<decltype(lrint((float)0)), long>::value, "");
    static_assert(is_same<decltype(lrint((bool)0)), long>::value, "");
    static_assert(is_same<decltype(lrint((unsigned short)0)), long>::value, "");
    static_assert(is_same<decltype(lrint((int)0)), long>::value, "");
    static_assert(is_same<decltype(lrint((unsigned int)0)), long>::value, "");
    static_assert(is_same<decltype(lrint((long)0)), long>::value, "");
    static_assert(is_same<decltype(lrint((unsigned long)0)), long>::value, "");
    static_assert(is_same<decltype(lrint((long long)0)), long>::value, "");
    static_assert(is_same<decltype(lrint((unsigned long long)0)), long>::value, "");
    static_assert(is_same<decltype(lrint((double)0)), long>::value, "");
    static_assert(is_same<decltype(lrintf(0)), long>::value, "");
}

__device__ void test_lround()
{
    static_assert(is_same<decltype(lround((float)0)), long>::value, "");
    static_assert(is_same<decltype(lround((bool)0)), long>::value, "");
    static_assert(is_same<decltype(lround((unsigned short)0)), long>::value, "");
    static_assert(is_same<decltype(lround((int)0)), long>::value, "");
    static_assert(is_same<decltype(lround((unsigned int)0)), long>::value, "");
    static_assert(is_same<decltype(lround((long)0)), long>::value, "");
    static_assert(is_same<decltype(lround((unsigned long)0)), long>::value, "");
    static_assert(is_same<decltype(lround((long long)0)), long>::value, "");
    static_assert(is_same<decltype(lround((unsigned long long)0)), long>::value, "");
    static_assert(is_same<decltype(lround((double)0)), long>::value, "");
    static_assert(is_same<decltype(lroundf(0)), long>::value, "");
}

__device__ void test_nan()
{
    static_assert(is_same<decltype(nan("")), double>::value, "");
    static_assert(is_same<decltype(nanf("")), float>::value, "");
}

__device__ void test_nearbyint()
{
    static_assert(is_same<decltype(nearbyint((float)0)), float>::value, "");
    static_assert(is_same<decltype(nearbyint((bool)0)), double>::value, "");
    static_assert(is_same<decltype(nearbyint((unsigned short)0)), double>::value, "");
    static_assert(is_same<decltype(nearbyint((int)0)), double>::value, "");
    static_assert(is_same<decltype(nearbyint((unsigned int)0)), double>::value, "");
    static_assert(is_same<decltype(nearbyint((long)0)), double>::value, "");
    static_assert(is_same<decltype(nearbyint((unsigned long)0)), double>::value, "");
    static_assert(is_same<decltype(nearbyint((long long)0)), double>::value, "");
    static_assert(is_same<decltype(nearbyint((unsigned long long)0)), double>::value, "");
    static_assert(is_same<decltype(nearbyint((double)0)), double>::value, "");
    static_assert(is_same<decltype(nearbyintf(0)), float>::value, "");
}

__device__ void test_nextafter()
{
    static_assert(is_same<decltype(nextafter((float)0, (float)0)), float>::value, "");
    static_assert(is_same<decltype(nextafter((bool)0, (float)0)), double>::value, "");
    static_assert(is_same<decltype(nextafter((unsigned short)0, (double)0)), double>::value, "");
    static_assert(is_same<decltype(nextafter((float)0, (unsigned int)0)), double>::value, "");
    static_assert(is_same<decltype(nextafter((double)0, (long)0)), double>::value, "");
    static_assert(is_same<decltype(nextafter((int)0, (long long)0)), double>::value, "");
    static_assert(is_same<decltype(nextafter((int)0, (unsigned long long)0)), double>::value, "");
    static_assert(is_same<decltype(nextafter((double)0, (double)0)), double>::value, "");
    static_assert(is_same<decltype(nextafter((float)0, (double)0)), double>::value, "");
    static_assert(is_same<decltype(nextafterf(0,0)), float>::value, "");
    static_assert(is_same<decltype(nextafter((int)0, (int)0)), double>::value, "");
}

__device__ void test_remainder()
{
    static_assert(is_same<decltype(remainder((float)0, (float)0)), float>::value, "");
    static_assert(is_same<decltype(remainder((bool)0, (float)0)), double>::value, "");
    static_assert(is_same<decltype(remainder((unsigned short)0, (double)0)), double>::value, "");
    static_assert(is_same<decltype(remainder((float)0, (unsigned int)0)), double>::value, "");
    static_assert(is_same<decltype(remainder((double)0, (long)0)), double>::value, "");
    static_assert(is_same<decltype(remainder((int)0, (long long)0)), double>::value, "");
    static_assert(is_same<decltype(remainder((int)0, (unsigned long long)0)), double>::value, "");
    static_assert(is_same<decltype(remainder((double)0, (double)0)), double>::value, "");
    static_assert(is_same<decltype(remainder((float)0, (double)0)), double>::value, "");
    static_assert(is_same<decltype(remainderf(0,0)), float>::value, "");
    static_assert(is_same<decltype(remainder((int)0, (int)0)), double>::value, "");
}

__device__ void test_remquo()
{
    int ip;
    static_assert(is_same<decltype(remquo((float)0, (float)0, &ip)), float>::value, "");
    static_assert(is_same<decltype(remquo((bool)0, (float)0, &ip)), double>::value, "");
    static_assert(is_same<decltype(remquo((unsigned short)0, (double)0, &ip)), double>::value, "");
    static_assert(is_same<decltype(remquo((float)0, (unsigned int)0, &ip)), double>::value, "");
    static_assert(is_same<decltype(remquo((double)0, (long)0, &ip)), double>::value, "");
    static_assert(is_same<decltype(remquo((int)0, (long long)0, &ip)), double>::value, "");
    static_assert(is_same<decltype(remquo((int)0, (unsigned long long)0, &ip)), double>::value, "");
    static_assert(is_same<decltype(remquo((double)0, (double)0, &ip)), double>::value, "");
    static_assert(is_same<decltype(remquo((float)0, (double)0, &ip)), double>::value, "");
    static_assert(is_same<decltype(remquof(0,0, &ip)), float>::value, "");
    static_assert(is_same<decltype(remquo((int)0, (int)0, &ip)), double>::value, "");
}

__device__ void test_rint()
{
    static_assert(is_same<decltype(rint((float)0)), float>::value, "");
    static_assert(is_same<decltype(rint((bool)0)), double>::value, "");
    static_assert(is_same<decltype(rint((unsigned short)0)), double>::value, "");
    static_assert(is_same<decltype(rint((int)0)), double>::value, "");
    static_assert(is_same<decltype(rint((unsigned int)0)), double>::value, "");
    static_assert(is_same<decltype(rint((long)0)), double>::value, "");
    static_assert(is_same<decltype(rint((unsigned long)0)), double>::value, "");
    static_assert(is_same<decltype(rint((long long)0)), double>::value, "");
    static_assert(is_same<decltype(rint((unsigned long long)0)), double>::value, "");
    static_assert(is_same<decltype(rint((double)0)), double>::value, "");
    static_assert(is_same<decltype(rintf(0)), float>::value, "");
}

__device__ void test_round()
{
    static_assert(is_same<decltype(round((float)0)), float>::value, "");
    static_assert(is_same<decltype(round((bool)0)), double>::value, "");
    static_assert(is_same<decltype(round((unsigned short)0)), double>::value, "");
    static_assert(is_same<decltype(round((int)0)), double>::value, "");
    static_assert(is_same<decltype(round((unsigned int)0)), double>::value, "");
    static_assert(is_same<decltype(round((long)0)), double>::value, "");
    static_assert(is_same<decltype(round((unsigned long)0)), double>::value, "");
    static_assert(is_same<decltype(round((long long)0)), double>::value, "");
    static_assert(is_same<decltype(round((unsigned long long)0)), double>::value, "");
    static_assert(is_same<decltype(round((double)0)), double>::value, "");
    static_assert(is_same<decltype(roundf(0)), float>::value, "");
}

__device__ void test_scalbln()
{
    static_assert(is_same<decltype(scalbln((float)0, (long)0)), float>::value, "");
    static_assert(is_same<decltype(scalbln((bool)0, (long)0)), double>::value, "");
    static_assert(is_same<decltype(scalbln((unsigned short)0, (long)0)), double>::value, "");
    static_assert(is_same<decltype(scalbln((int)0, (long)0)), double>::value, "");
    static_assert(is_same<decltype(scalbln((unsigned int)0, (long)0)), double>::value, "");
    static_assert(is_same<decltype(scalbln((long)0, (long)0)), double>::value, "");
    static_assert(is_same<decltype(scalbln((unsigned long)0, (long)0)), double>::value, "");
    static_assert(is_same<decltype(scalbln((long long)0, (long)0)), double>::value, "");
    static_assert(is_same<decltype(scalbln((unsigned long long)0, (long)0)), double>::value, "");
    static_assert(is_same<decltype(scalbln((double)0, (long)0)), double>::value, "");
    static_assert(is_same<decltype(scalblnf(0, (long)0)), float>::value, "");
}

__device__ void test_scalbn()
{
    static_assert(is_same<decltype(scalbn((float)0, (int)0)), float>::value, "");
    static_assert(is_same<decltype(scalbn((bool)0, (int)0)), double>::value, "");
    static_assert(is_same<decltype(scalbn((unsigned short)0, (int)0)), double>::value, "");
    static_assert(is_same<decltype(scalbn((int)0, (int)0)), double>::value, "");
    static_assert(is_same<decltype(scalbn((unsigned int)0, (int)0)), double>::value, "");
    static_assert(is_same<decltype(scalbn((long)0, (int)0)), double>::value, "");
    static_assert(is_same<decltype(scalbn((unsigned long)0, (int)0)), double>::value, "");
    static_assert(is_same<decltype(scalbn((long long)0, (int)0)), double>::value, "");
    static_assert(is_same<decltype(scalbn((unsigned long long)0, (int)0)), double>::value, "");
    static_assert(is_same<decltype(scalbn((double)0, (int)0)), double>::value, "");
    static_assert(is_same<decltype(scalbnf(0, (int)0)), float>::value, "");
}

__device__ void test_tgamma()
{
    static_assert(is_same<decltype(tgamma((float)0)), float>::value, "");
    static_assert(is_same<decltype(tgamma((bool)0)), double>::value, "");
    static_assert(is_same<decltype(tgamma((unsigned short)0)), double>::value, "");
    static_assert(is_same<decltype(tgamma((int)0)), double>::value, "");
    static_assert(is_same<decltype(tgamma((unsigned int)0)), double>::value, "");
    static_assert(is_same<decltype(tgamma((long)0)), double>::value, "");
    static_assert(is_same<decltype(tgamma((unsigned long)0)), double>::value, "");
    static_assert(is_same<decltype(tgamma((long long)0)), double>::value, "");
    static_assert(is_same<decltype(tgamma((unsigned long long)0)), double>::value, "");
    static_assert(is_same<decltype(tgamma((double)0)), double>::value, "");
    static_assert(is_same<decltype(tgammaf(0)), float>::value, "");
}

__device__ void test_trunc()
{
    static_assert(is_same<decltype(trunc((float)0)), float>::value, "");
    static_assert(is_same<decltype(trunc((bool)0)), double>::value, "");
    static_assert(is_same<decltype(trunc((unsigned short)0)), double>::value, "");
    static_assert(is_same<decltype(trunc((int)0)), double>::value, "");
    static_assert(is_same<decltype(trunc((unsigned int)0)), double>::value, "");
    static_assert(is_same<decltype(trunc((long)0)), double>::value, "");
    static_assert(is_same<decltype(trunc((unsigned long)0)), double>::value, "");
    static_assert(is_same<decltype(trunc((long long)0)), double>::value, "");
    static_assert(is_same<decltype(trunc((unsigned long long)0)), double>::value, "");
    static_assert(is_same<decltype(trunc((double)0)), double>::value, "");
    static_assert(is_same<decltype(truncf(0)), float>::value, "");
}

__global__ void tests()
{
    test_abs();
    test_acos();
    test_asin();
    test_atan();
    test_atan2();
    test_ceil();
    test_cos();
    test_cosh();
    test_exp();
    test_fabs();
    test_floor();
    test_fmod();
    test_frexp();
    test_ldexp();
    test_log();
    test_log10();
    test_modf();
    test_pow();
    test_sin();
    test_sinh();
    test_sqrt();
    test_tan();
    test_tanh();
    test_signbit();
    test_fpclassify();
    test_isfinite();
    test_isnormal();
    test_isgreater();
    test_isgreaterequal();
    test_isinf();
    test_isless();
    test_islessequal();
    test_islessgreater();
    test_isnan();
    test_isunordered();
    test_acosh();
    test_asinh();
    test_atanh();
    test_cbrt();
    test_copysign();
    test_erf();
    test_erfc();
    test_exp2();
    test_expm1();
    test_fdim();
    test_fma();
    test_fmax();
    test_fmin();
    test_hypot();
    test_ilogb();
    test_lgamma();
    test_llrint();
    test_llround();
    test_log1p();
    test_log2();
    test_logb();
    test_lrint();
    test_lround();
    test_nan();
    test_nearbyint();
    test_nextafter();
    test_remainder();
    test_remquo();
    test_rint();
    test_round();
    test_scalbln();
    test_scalbn();
    test_tgamma();
    test_trunc();
}
