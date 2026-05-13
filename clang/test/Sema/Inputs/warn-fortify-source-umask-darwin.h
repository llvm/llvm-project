#pragma GCC system_header

// Darwin and the BSDs spell mode_t as a 16-bit type (__uint16_t), unlike
// glibc's unsigned int. umask is recognized by builtin name with no prototype
// to match, so the width the libc picks for mode_t does not matter.

#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned short __uint16_t;
typedef __uint16_t mode_t;
mode_t umask(mode_t);

#ifdef __cplusplus
}
#endif
