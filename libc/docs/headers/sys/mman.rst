.. include:: ../../check.rst

==========
sys/mman.h
==========

Macros
======

.. list-table::
  :widths: auto
  :align: center
  :header-rows: 1

  * - Macro
    - Implemented
    - C23 Standard Section
    - POSIX.1-2024 Standard Section
  * - MAP_ANON
    -
    -
    - 
  * - MAP_ANONYMOUS
    -
    -
    - 
  * - MAP_FAILED
    - |check|
    -
    - 
  * - MAP_FIXED
    -
    -
    - 
  * - MAP_PRIVATE
    -
    -
    - 
  * - MAP_SHARED
    -
    -
    - 
  * - MCL_CURRENT
    -
    -
    - 
  * - MCL_FUTURE
    -
    -
    - 
  * - MS_ASYNC
    -
    -
    - 
  * - MS_INVALIDATE
    -
    -
    - 
  * - MS_SYNC
    -
    -
    - 
  * - POSIX_MADV_DONTNEED
    - |check|
    -
    - 
  * - POSIX_MADV_NORMAL
    - |check|
    -
    - 
  * - POSIX_MADV_RANDOM
    - |check|
    -
    - 
  * - POSIX_MADV_SEQUENTIAL
    - |check|
    -
    - 
  * - POSIX_MADV_WILLNEED
    - |check|
    -
    - 
  * - POSIX_TYPED_MEM_ALLOCATE
    -
    -
    - 
  * - POSIX_TYPED_MEM_ALLOCATE_CONTIG
    -
    -
    - 
  * - POSIX_TYPED_MEM_MAP_ALLOCATABLE
    -
    -
    - 
  * - PROT_EXEC
    -
    -
    - 
  * - PROT_NONE
    -
    -
    - 
  * - PROT_READ
    -
    -
    - 
  * - PROT_WRITE
    -
    -
    - 

Functions
=========

.. list-table::
  :widths: auto
  :align: center
  :header-rows: 1

  * - Function
    - Implemented
    - C23 Standard Section
    - POSIX.1-2024 Standard Section
  * - mlock
    - |check|
    -
    - 
  * - mlockall
    - |check|
    -
    - 
  * - mmap
    - |check|
    -
    - 
  * - mprotect
    - |check|
    -
    - 
  * - msync
    - |check|
    -
    - 
  * - munlock
    - |check|
    -
    - 
  * - munlockall
    - |check|
    -
    - 
  * - munmap
    - |check|
    -
    - 
  * - posix_madvise
    - |check|
    -
    - 
  * - posix_mem_offset
    -
    -
    - 
  * - posix_typed_mem_get_info
    -
    -
    - 
  * - posix_typed_mem_open
    -
    -
    - 
  * - shm_open
    - |check|
    -
    - 
  * - shm_unlink
    - |check|
    -
    - 
