#pragma GCC system_header

#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned mode_t;
mode_t umask(mode_t cmask);

#ifdef __cplusplus
}
#endif
