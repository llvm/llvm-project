#ifdef USE_PRAGMA_BEFORE
#pragma intrinsic(_InterlockedOr64)
#endif

#define MACRO(x,y) _InterlockedOr64(x,y);

#ifdef USE_PRAGMA_AFTER
#pragma intrinsic(_InterlockedOr64)
#endif
