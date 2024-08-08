#ifdef USE_PRAGMA
#pragma clang diagnostic push
#if USE_PRAGMA == 1
#pragma clang diagnostic warning "-Wshorten-64-to-32"
#else
#pragma clang diagnostic error "-Wshorten-64-to-32"
#endif
#endif
template <class T> int convert(T V) { return V; }
#ifdef USE_PRAGMA
#pragma clang diagnostic pop
#endif
