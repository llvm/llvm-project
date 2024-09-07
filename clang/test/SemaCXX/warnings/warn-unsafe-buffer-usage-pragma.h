#ifdef _INCLUDE_NO_WARN
// the snippet will be included in an opt-out region
p1++;

#undef _INCLUDE_NO_WARN

#elif defined(_INCLUDE_WARN)
// the snippet will be included in a location where warnings are expected
p2++; // expected-note{{used in pointer arithmetic here}}
#undef _INCLUDE_WARN

#else

#endif
