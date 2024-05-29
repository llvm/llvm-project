struct ImmortalRefType {};

struct RefCountedType { int value; };

inline void RCRetain(RefCountedType *x) { x->value++; }
inline void RCRelease(RefCountedType *x) { x->value--; }

struct NonCopyableType { int value; };
struct CopyableType { int value; };
