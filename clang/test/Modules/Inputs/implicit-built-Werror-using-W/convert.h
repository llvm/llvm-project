#ifdef USE_PRAGMA
#pragma clang diagnostic push
#pragma clang diagnostic warning "-Wshorten-64-to-32"
#endif
template <class T> int convert(T V) { return V; }
#ifdef USE_PRAGMA
#pragma clang diagnostic pop
#endif
