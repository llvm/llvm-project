// Produce a crash if CRASH is defined.
#ifdef CRASH
#  pragma clang __debug crash
#endif

const char *getCrashString();
