#ifdef __cplusplus
int base(int *p) {
  int x = p[5];
#pragma clang unsafe_buffer_usage begin
  int y = p[5];
#pragma clang unsafe_buffer_usage end
  return x + y;
}
#endif
