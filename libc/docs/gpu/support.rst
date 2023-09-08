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
bzero          |check|
memccpy        |check|
memchr         
memcmp         |check|
memcpy         |check|
memmove        |check|
mempcpy        |check|
memrchr        |check|
memset         |check|
stpcpy         |check|
stpncpy        |check|
strcat         |check|
strchr         
strcmp         |check|
strcpy         |check|
strcspn        |check|
strlcat        |check|
strlcpy        |check|
strlen         |check|
strncat        |check|
strncmp        |check|
strncpy        |check|
strnlen        |check|
strpbrk        
strrchr        
strspn         |check|
strstr         
strtok         |check|
strtok_r       |check|
strdup
strndup
=============  =========  ============

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
labs           |check|
llabs          |check|
div            |check|
ldiv           |check|
lldiv          |check|
strtod         |check|
strtof         |check|
strtol         |check|
strtold        |check|
strtoll        |check|
strtoul        |check|
strtoull       |check|
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
--------

=============  =========  ============
Function Name  Available  RPC Required
=============  =========  ============
puts           |check|    |check|
fputs          |check|    |check|
fclose         |check|    |check|
fopen          |check|    |check|
fread          |check|    |check|
=============  =========  ============

time.h
--------

=============  =========  ============
Function Name  Available  RPC Required
=============  =========  ============
clock          |check|
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
