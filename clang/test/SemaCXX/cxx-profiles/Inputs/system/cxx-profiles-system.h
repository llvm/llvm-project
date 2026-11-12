// System headers are exempt from all profiles.
inline char *system_cast(int *p) { return reinterpret_cast<char *>(p); }
