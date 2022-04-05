// The stub symbols library used for testing purposes

void *memcpy(void *dest, const void *src, unsigned long n) { return 0; }
void *memset(void *dest, int c, unsigned long n) { return 0; }
int printf(const char *format, ...) { return 0; }
void exit(int status) {}

void *__gxx_personality_v0;
void *__cxa_allocate_exception;
void *_ZTIi;
void *__cxa_throw;
void *__cxa_begin_catch;
void *__cxa_end_catch;
