// RUN: %clang_cc1 -emit-llvm < %s

// PR2414
struct mad_frame{};
enum mad_flow {ont};

typedef enum mad_flow filter_func_t(void *, struct mad_frame *);

filter_func_t mono_filter;

void addfilter2(filter_func_t *func){}

void setup_filters()
{
  addfilter2( mono_filter);
}
