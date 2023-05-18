#pragma GCC system_header
#pragma once

inline void f() { register int k; }
#define to_int(x) ({ register int n = (x); n; })
