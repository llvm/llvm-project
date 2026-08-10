#if defined(__MINGW32__)
void *__mingw_aligned_malloc(size_t, size_t);
void __mingw_aligned_free(void *);
#elif defined(_WIN32)
void *_aligned_malloc(size_t, size_t);
void _aligned_free(void *);
#endif
