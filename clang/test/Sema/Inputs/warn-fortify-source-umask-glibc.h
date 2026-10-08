#pragma GCC system_header

// Mimic glibc's typedef chain: the prototype is written in terms of
// __mode_t, an internal typedef that itself aliases unsigned int.
// `mode_t` is then defined as an alias of __mode_t. This shape would
// reject any check that walks one layer of typedef sugar and matches
// on the typedef name `mode_t`.

#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned int __mode_t;
typedef __mode_t mode_t;
extern __mode_t umask(__mode_t __mask);

#ifdef __cplusplus
}
#endif
