/* As default architectures with sizeof (off_t) < sizeof (off64_t) the mmap is
   implemented with __SYS_mmap2 syscall and the offset is represented in
   multiples of page size.  For offset larger than
   '1 << (page_shift + 8 * sizeof (off_t))' (that is, 1<<44 on system with
   page size of 4096 bytes) the system call silently truncates the offset.
   For this case, glibc mmap implementation returns EINVAL.  */

/* Return the maximum value expected as offset argument in mmap64 call.  */
static inline uint64_t
mmap64_maximum_offset (long int page_shift)
{
  if (sizeof (off_t) < sizeof (off64_t))
    return (UINT64_C(1) << (page_shift + (8 * sizeof (off_t)))) - 1;
  else
    return UINT64_MAX;
}
