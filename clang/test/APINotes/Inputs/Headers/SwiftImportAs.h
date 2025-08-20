struct ImmortalRefType {
    ImmortalRefType * methodReturningFrt__(void);
    ImmortalRefType * methodReturningFrt_returns_unretained(void);
    ImmortalRefType * methodReturningFrt_returns_retained(void);
};

ImmortalRefType * functionReturningFrt__(void);
ImmortalRefType * functionReturningFrt_returns_unretained(void);
ImmortalRefType * functionReturningFrt_returns_retained(void);


struct RefCountedType { int value; };

inline void RCRetain(RefCountedType *x) { x->value++; }
inline void RCRelease(RefCountedType *x) { x->value--; }

struct NonCopyableType { int value; };
struct CopyableType { int value; };

struct NonEscapableType { int value; };
struct EscapableType { int value; };

struct RefCountedTypeWithDefaultConvention {};
inline void retain(RefCountedType *x) {}
inline void release(RefCountedType *x) {}

struct OpaqueRefCountedType;
struct OpaqueRefCountedType; // redeclaration

inline void ORCRetain(struct OpaqueRefCountedType *x);
inline void ORCRelease(struct OpaqueRefCountedType *x);

typedef unsigned WrappedOptions;

struct NoncopyableWithDestroyType {
};

void NCDDestroy(NoncopyableWithDestroyType instance);
