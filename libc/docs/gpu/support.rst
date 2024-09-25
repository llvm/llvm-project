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
=============  =========  ============

assert.h
--------

=============  =========  ============
Function Name  Available  RPC Required
=============  =========  ============
assert         |check|    |check|
__assert_fail  |check|    |check|
=============  =========  ============
