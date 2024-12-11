#pragma clang system_header

void foo(int *__counted_by(len) p, int len);
void foo(int *__null_terminated p, int len);

void bar(int *__null_terminated p, int len);
void bar(int *__counted_by(len) p, int len);
