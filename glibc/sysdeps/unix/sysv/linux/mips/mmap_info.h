/* mips64n32 uses __NR_mmap for mmap64 while still having sizeof (off_t)
   smaller than sizeof (off64_t).  So it allows mapping large offsets
   using mmap64 than 32-bit archs which uses __NR_mmap2.  */

static inline uint64_t
mmap64_maximum_offset (long int page_shift)
{
#if _MIPS_SIM == _ABIN32 || _MIPS_SIM == _ABI64
  return UINT64_MAX;
#else
  return (UINT64_C(1) << (page_shift + (8 * sizeof (off_t)))) - 1;
#endif
}
