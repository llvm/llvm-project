#define pragma _Pragma("clang diagnostic push") _Pragma("clang diagnostic error \"-Wpadded\"") _Pragma("clang diagnostic pop")
pragma

struct Padded { char x; int y; };
