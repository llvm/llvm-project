#include_next <math.h>
template<typename T> T abs(T t) { return (t < 0) ? -t : t; }
