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
`remote procedure call <libc_gpu_rpc>`.

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
memchr         |check|
memcmp         |check|
memcpy         |check|
memmove        |check|
mempcpy        |check|
memrchr        |check|
memset         |check|
stpcpy         |check|
stpncpy        |check|
strcat         |check|
strchr         |check|
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
strpbrk        |check|
strrchr        |check|
strspn         |check|
strstr         |check|
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
atoi           
atof           
atol           
atoll          
labs           |check|
llabs          |check|
strtod         
strtof         
strtol         
strtold        
strtoll        
strtoul        
strtoull       
=============  =========  ============

stdio.h
--------

=============  =========  ============
Function Name  Available  RPC Required
=============  =========  ============
puts           |check|    |check|
fputs          |check|    |check|
=============  =========  ============
