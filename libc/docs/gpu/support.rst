.. _libc_gpu_support:

===================
Supported Functions
===================

.. include:: ../check.rst

.. contents:: Table of Contents
  :depth: 4
  :local:

The following functions and headers are supported at least partially on the
device. Some functions are implemented fully on the GPU, while others require a
`remote procedure call <libc_gpu_rpc>`_.

ctype.h
-------

=============  =========  ============
Function Name  Available  RPC Required
=============  =========  ============
isalnum        |check|
isalpha        |check|
isascii        |check|
isblank        |check|
iscntrl        |check|
isdigit        |check|
isgraph        |check|
islower        |check|
isprint        |check|
ispunct        |check|
isspace        |check|
isupper        |check|
isxdigit       |check|
toascii        |check|
tolower        |check|
toupper        |check|
=============  =========  ============

string.h
--------

=============  =========  ============
Function Name  Available  RPC Required
=============  =========  ============
bcmp           |check|
bcopy          |check|
bzero          |check|
memccpy        |check|
memchr         |check|
memcmp         |check|
memcpy         |check|
memmem         |check|
memmove        |check|
mempcpy        |check|
memrchr        |check|
memset         |check|
stpcpy         |check|
stpncpy        |check|
strcat         |check|
strchr         |check|
strchrnul      |check|
strcmp         |check|
strcoll        |check|
strcpy         |check|
strcspn        |check|
strdup         |check|
strerror       |check|
strlcat        |check|
strlcpy        |check|
strlen         |check|
strncat        |check|
strncmp        |check|
strncpy        |check|
strndup        |check|
strnlen        |check|
strpbrk        |check|
strrchr        |check|
strsep         |check|
strspn         |check|
strstr         |check|
strtok         |check|
strtok_r       |check|
strxfrm        |check|
=============  =========  ============

strings.h
---------

=============  =========  ============
Function Name  Available  RPC Required
=============  =========  ============
bcmp           |check|
bcopy          |check|
bzero          |check|
strcasecmp     |check|
strcasestr     |check|
index          |check|
rindex         |check|
=============  =========  ============

stdbit.h
--------

============================  =========  ============
Function Name                 Available  RPC Required
============================  =========  ============
stdc_leading_zeros_uc         |check|
stdc_leading_zeros_us         |check|
stdc_leading_zeros_ui         |check|
stdc_leading_zeros_ul         |check|
stdc_leading_zeros_ull        |check|
stdc_trailing_zeros_uc        |check|
stdc_trailing_zeros_us        |check|
stdc_trailing_zeros_ui        |check|
stdc_trailing_zeros_ul        |check|
stdc_trailing_zeros_ull       |check|
stdc_trailing_ones_uc         |check|
stdc_trailing_ones_us         |check|
stdc_trailing_ones_ui         |check|
stdc_trailing_ones_ul         |check|
stdc_trailing_ones_ull        |check|
stdc_first_leading_zero_uc    |check|
stdc_first_leading_zero_us    |check|
stdc_first_leading_zero_ui    |check|
stdc_first_leading_zero_ul    |check|
stdc_first_leading_zero_ull   |check|
stdc_first_leading_one_uc     |check|
stdc_first_leading_one_us     |check|
stdc_first_leading_one_ui     |check|
stdc_first_leading_one_ul     |check|
stdc_first_leading_one_ull    |check|
stdc_first_trailing_zero_uc   |check|
stdc_first_trailing_zero_us   |check|
stdc_first_trailing_zero_ui   |check|
stdc_first_trailing_zero_ul   |check|
stdc_first_trailing_zero_ull  |check|
stdc_first_trailing_one_uc    |check|
stdc_first_trailing_one_us    |check|
stdc_first_trailing_one_ui    |check|
stdc_first_trailing_one_ul    |check|
stdc_first_trailing_one_ull   |check|
stdc_count_zeros_uc           |check|
stdc_count_zeros_us           |check|
stdc_count_zeros_ui           |check|
stdc_count_zeros_ul           |check|
stdc_count_zeros_ull          |check|
stdc_count_ones_uc            |check|
stdc_count_ones_us            |check|
stdc_count_ones_ui            |check|
stdc_count_ones_ul            |check|
stdc_count_ones_ull           |check|
stdc_has_single_bit_uc        |check|
stdc_has_single_bit_us        |check|
stdc_has_single_bit_ui        |check|
stdc_has_single_bit_ul        |check|
stdc_has_single_bit_ull       |check|
stdc_bit_width_uc             |check|
stdc_bit_width_us             |check|
stdc_bit_width_ui             |check|
stdc_bit_width_ul             |check|
stdc_bit_width_ull            |check|
stdc_bit_floor_uc             |check|
stdc_bit_floor_us             |check|
stdc_bit_floor_ui             |check|
stdc_bit_floor_ul             |check|
stdc_bit_floor_ull            |check|
stdc_bit_ceil_uc              |check|
stdc_bit_ceil_us              |check|
stdc_bit_ceil_ui              |check|
stdc_bit_ceil_ul              |check|
stdc_bit_ceil_ull             |check|
============================  =========  ============

stdlib.h
--------

=============  =========  ============
Function Name  Available  RPC Required
=============  =========  ============
abs            |check|
atoi           |check|
atof           |check|
atol           |check|
atoll          |check|
exit           |check|    |check|
abort          |check|    |check|
system         |check|    |check|
labs           |check|
llabs          |check|
div            |check|
ldiv           |check|
lldiv          |check|
bsearch        |check|
qsort          |check|
qsort_r        |check|
strtod         |check|
strtof         |check|
strtol         |check|
strtold        |check|
strtoll        |check|
strtoul        |check|
strtoull       |check|
srand          |check|
rand           |check|
=============  =========  ============

inttypes.h
----------

=============  =========  ============
Function Name  Available  RPC Required
=============  =========  ============
imaxabs        |check|
imaxdiv        |check|
strtoimax      |check|
strtoumax      |check|
=============  =========  ============

stdio.h
-------

=============  =========  ============
Function Name  Available  RPC Required
=============  =========  ============
feof           |check|    |check|
ferror         |check|    |check|
clearerr       |check|    |check|
fseek          |check|    |check|
ftell          |check|    |check|
fflush         |check|    |check|
fgetc          |check|    |check|
fgets          |check|    |check|
ungetc         |check|    |check|
getc           |check|    |check|
getchar        |check|    |check|
puts           |check|    |check|
fputs          |check|    |check|
fputc          |check|    |check|
fwrite         |check|    |check|
remove         |check|    |check|
rename         |check|    |check|
putc           |check|    |check|
printf         |check|    |check|
vprintf        |check|    |check|
fprintf        |check|    |check|
vfprintf       |check|    |check|
sprintf        |check|
snprintf       |check|
vsprintf       |check|
vsnprintf      |check|
sscanf         |check|
scanf          |check|    |check|
fscanf         |check|    |check|
putchar        |check|    |check|
fclose         |check|    |check|
fopen          |check|    |check|
fread          |check|    |check|
=============  =========  ============

time.h
------

=============  =========  ============
Function Name  Available  RPC Required
=============  =========  ============
clock          |check|
clock_gettime  |check|
nanosleep      |check|
strftime       |check|
strftime_l     |check|
=============  =========  ============

assert.h
--------

=============  =========  ============
Function Name  Available  RPC Required
=============  =========  ============
assert         |check|    |check|
__assert_fail  |check|    |check|
=============  =========  ============

math.h
------

The following table presents the conformance test results for higher math functions on the GPU. The results show the maximum observed ULP (Units in the Last Place) distance when comparing the GPU implementation against a correctly rounded reference computed on the host CPU. In addition to the C standard math library (LLVM-libm), these tests are conducted against CUDA Math and HIP Math, for comparison only.

+------------------------+-------------+---------------+-----------------------------------------------------------------------------------+
| Function               | Test Method | ULP Tolerance | Max ULP Distance                                                                  |
|                        |             |               +--------------------+--------------------+--------------------+--------------------+
|                        |             |               | llvm-libm          | llvm-libm          | cuda-math          | hip-math           |
|                        |             |               | (AMDGPU)           | (CUDA)             | (CUDA)             | (AMDGPU)           |
+========================+=============+===============+====================+====================+====================+====================+
| acos                   | Randomized  | 4             | 6 (FAILED)         | 6 (FAILED)         | 1                  | 1                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| acosf                  | Exhaustive  | 4             | 1                  | 1                  | 1                  | 1                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| acosf16                | Exhaustive  | 2             | 1                  | 1                  |                    | 1                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| acoshf                 | Exhaustive  | 4             | 1                  | 1                  | 2                  | 1                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| acoshf16               | Exhaustive  | 2             | 0                  | 0                  |                    | 0                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| acospif16              | Exhaustive  | 2             | 0                  | 0                  |                    |                    |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| asin                   | Randomized  | 4             | 6 (FAILED)         | 6 (FAILED)         | 2                  | 1                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| asinf                  | Exhaustive  | 4             | 1                  | 1                  | 1                  | 3                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| asinf16                | Exhaustive  | 2             | 0                  | 0                  |                    | 2                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| asinhf                 | Exhaustive  | 4             | 1                  | 1                  | 2                  | 1                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| asinhf16               | Exhaustive  | 2             | 1                  | 1                  |                    | 1                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| atanf                  | Exhaustive  | 5             | 0                  | 0                  | 1                  | 2                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| atanf16                | Exhaustive  | 2             | 1                  | 1                  |                    | 1                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| atan2f                 | Randomized  | 6             | 1                  | 1                  | 2                  | 3                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| atanhf                 | Exhaustive  | 5             | 0                  | 0                  | 3                  | 1                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| atanhf16               | Exhaustive  | 2             | 0                  | 0                  |                    | 1                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| cbrt                   | Randomized  | 2             | 1                  | 1                  | 1                  | 1                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| cbrtf                  | Exhaustive  | 2             | 0                  | 0                  | 1                  | 1                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| cos                    | Randomized  | 4             | 1                  | 1                  | 2                  | 1                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| cosf                   | Exhaustive  | 4             | 1                  | 1                  | 2                  | 2                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| cosf16                 | Exhaustive  | 2             | 1                  | 1                  |                    | 1                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| coshf                  | Exhaustive  | 4             | 0                  | 0                  | 2                  | 1                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| coshf16                | Exhaustive  | 2             | 1                  | 0                  |                    | 1                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| cospif                 | Exhaustive  | 4             | 0                  | 0                  | 1                  | 1                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| cospif16               | Exhaustive  | 2             | 0                  | 0                  |                    |                    |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| erff                   | Exhaustive  | 16            | 0                  | 0                  | 1                  | 2                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| exp                    | Randomized  | 3             | 1                  | 1                  | 1                  | 1                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| expf                   | Exhaustive  | 3             | 0                  | 0                  | 2                  | 1                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| expf16                 | Exhaustive  | 2             | 1                  | 1                  |                    | 1                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| exp10                  | Randomized  | 3             | 1                  | 1                  | 1                  | 1                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| exp10f                 | Exhaustive  | 3             | 0                  | 0                  | 2                  | 1                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| exp10f16               | Exhaustive  | 2             | 1                  | 1                  |                    | 1                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| exp2                   | Randomized  | 3             | 1                  | 1                  | 1                  | 1                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| exp2f                  | Exhaustive  | 3             | 1                  | 1                  | 2                  | 1                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| exp2f16                | Exhaustive  | 2             | 1                  | 1                  |                    | 0                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| expm1                  | Randomized  | 3             | 0                  | 0                  | 1                  | 2                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| expm1f                 | Exhaustive  | 3             | 1                  | 1                  | 1                  | 1                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| expm1f16               | Exhaustive  | 2             | 1                  | 1                  |                    | 1                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| hypot                  | Randomized  | 4             | 0                  | 0                  | 2                  | 1                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| hypotf                 | Randomized  | 4             | 0                  | 0                  | 1                  | 2                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| hypotf16               | Exhaustive  | 2             | 0                  | 0                  |                    |                    |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| log                    | Randomized  | 3             | 1                  | 1                  | 1                  | 1                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| logf                   | Exhaustive  | 3             | 1                  | 1                  | 1                  | 2                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| logf16                 | Exhaustive  | 2             | 1                  | 1                  |                    | 1                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| log10                  | Randomized  | 3             | 1                  | 1                  | 1                  | 1                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| log10f                 | Exhaustive  | 3             | 1                  | 1                  | 2                  | 2                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| log10f16               | Exhaustive  | 2             | 1                  | 1                  |                    | 1                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| log1p                  | Randomized  | 2             | 1                  | 1                  | 1                  | 1                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| log1pf                 | Exhaustive  | 2             | 1                  | 1                  | 1                  | 1                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| log2                   | Randomized  | 3             | 1                  | 1                  | 1                  | 1                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| log2f                  | Exhaustive  | 3             | 0                  | 0                  | 1                  | 1                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| log2f16                | Exhaustive  | 2             | 1                  | 1                  |                    | 0                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| powf (integer exp.)    | Randomized  | 16            | 0                  | 0                  | 2                  | 1                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| powf (real exp.)       | Randomized  | 16            | 0                  | 0                  | 2                  | 1                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| sin                    | Randomized  | 4             | 1                  | 1                  | 1                  | 1                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| sinf                   | Exhaustive  | 4             | 1                  | 1                  | 1                  | 2                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| sinf16                 | Exhaustive  | 2             | 1                  | 1                  |                    | 1                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| sincos (cos part)      | Randomized  | 4             | 1                  | 1                  | 2                  | 1                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| sincos (sin part)      | Randomized  | 4             | 1                  | 1                  | 1                  | 1                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| sincosf (cos part)     | Exhaustive  | 4             | 1                  | 1                  | 2                  | 2                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| sincosf (sin part)     | Exhaustive  | 4             | 1                  | 1                  | 1                  | 2                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| sinhf                  | Exhaustive  | 4             | 1                  | 1                  | 3                  | 1                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| sinhf16                | Exhaustive  | 2             | 1                  | 1                  |                    | 1                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| sinpif                 | Exhaustive  | 4             | 0                  | 0                  | 1                  | 1                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| sinpif16               | Exhaustive  | 2             | 0                  | 0                  |                    |                    |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| tan                    | Randomized  | 5             | 2                  | 2                  | 2                  | 1                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| tanf                   | Exhaustive  | 5             | 0                  | 0                  | 3                  | 2                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| tanf16                 | Exhaustive  | 2             | 1                  | 1                  |                    | 2                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| tanhf                  | Exhaustive  | 5             | 0                  | 0                  | 2                  | 1                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| tanhf16                | Exhaustive  | 2             | 0                  | 0                  |                    | 1                  |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| tanpif                 | Exhaustive  | 6             | 0                  | 0                  |                    |                    |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+
| tanpif16               | Exhaustive  | 2             | 1                  | 1                  |                    |                    |
+------------------------+-------------+---------------+--------------------+--------------------+--------------------+--------------------+

**Notes on Conformance Test Results:**

* **Test Method**:
    * **Exhaustive**: Every representable point in the input space is tested. This method is used for half-precision functions and single-precision univariate functions.
    * **Randomized**: A large, deterministic subset of the input space is tested, typically using 2\ :sup:`32` samples. This method is used for functions with larger input spaces, such as single-precision bivariate and double-precision functions.
* ULP tolerances are based on *The Khronos Group, The OpenCL C Specification v3.0.19, Sec. 7.4, Khronos Registry [July 10, 2025]*.
* The AMD GPU used for testing is *gfx1030*.
* The NVIDIA GPU used for testing is *NVIDIA RTX 4000 SFF Ada Generation*.
* For more details on the tests, please refer to the `GPU Math Conformance Tests <https://github.com/llvm/llvm-project/tree/main/offload/unittests/Conformance>`_.
