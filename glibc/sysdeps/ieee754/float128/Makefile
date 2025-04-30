ifeq ($(enable-float128),yes)
ifeq ($(subdir),stdlib)
routines += float1282mpn strfromf128
routines += strtof128 strtof128_l strtof128_nan mpn2float128

# The strfrom class of functions call __printf_fp in order to convert the
# floating-point value to characters.  This requires the value of IO_MTSAFE_IO.
CFLAGS-strfromf128.c += $(libio-mtsafe)
endif

ifeq ($(subdir),wcsmbs)
routines += wcstof128_l wcstof128 wcstof128_nan
endif
endif # enable-float128
