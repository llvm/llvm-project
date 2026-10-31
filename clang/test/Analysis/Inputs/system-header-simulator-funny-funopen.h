#pragma clang system_header

// A non-standard function named "funopen" with four parameters:
void funopen(int *, int *, int *, int *);

// Note that standard funopen() has five parameters:
//FILE *funopen(const void *cookie,
//              int (*readfn)(void *, char *, int),
//              int (*writefn)(void *, const char *, int),
//              off_t (*seekfn)(void *, off_t, int),
//              int (*closefn)(void *));
