#pragma clang system_header

inline void FUnspecified(int *x) {
  int *y = x;
  y = y + 1;
  int **z = &y;
}
