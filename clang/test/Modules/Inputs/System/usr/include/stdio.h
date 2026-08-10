typedef struct { int id; } FILE;
int fprintf(FILE*restrict, const char* restrict format, ...);
extern FILE *__stderrp;
