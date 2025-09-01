// RUN: %clang_cc1 -fsyntax-only %s -Weverything

void free(void *);
typedef void (*set_free_func)(void *);
struct Method {
  int nparams;
  int *param;
};
void selelem_free_method(struct Method* method, void* data) {
    set_free_func free_func = 0;
    for (int i = 0; i < method->nparams; ++i)
        free(&method->param[i]);
    if (data && free_func)
        free_func(data);
}
