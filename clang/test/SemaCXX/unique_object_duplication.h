/**
 * When building shared libraries, hidden objects which are defined in header
 * files will be duplicated, with one copy in each shared library. If the object
 * was meant to be globally unique (one copy per program), this can cause very
 * subtle bugs. This file contains tests for the -Wunique-object-duplication
 * warning, which is meant to detect this.
 * 
 * Roughly, an object might be incorrectly duplicated if:
 * - Is defined in a header (so it might appear in multiple TUs), and
 * - Has external linkage (otherwise it's supposed to be duplicated), and
 * - Has hidden visibility (or else the dynamic linker will handle it)
 * 
 * Duplication becomes an issue only if one of the following is true:
 * - The object is mutable (the copies won't be in sync), or
 * - Its initialization may has side effects (it may now run more than once), or
 * - The value of its address is used.
 * 
 * Currently, we only detect the first two, and only warn on effectful
 * initialization if we're certain there are side effects. Warning if the
 * address is taken is prone to false positives, so we don't warn for now.
 * 
 * The check is also disabled on Windows for now, since it uses 
 * dllimport/dllexport instead of visibility.
 */

#define HIDDEN __attribute__((visibility("hidden")))
#define DEFAULT __attribute__((visibility("default")))

// Helper functions
constexpr int init_constexpr(int x) { return x; };
extern double init_dynamic(int);

/******************************************************************************
 * Case one: Static local variables in an externally-visible function
 ******************************************************************************/
namespace StaticLocalTest {

inline void has_static_locals_external() {
  // Mutable
  static int disallowedStatic1 = 0; // hidden-warning {{'disallowedStatic1' is mutable, has hidden visibility, and external linkage; it may be duplicated when built into a shared library}}
  // Initialization might run more than once
  static const double disallowedStatic2 = disallowedStatic1++; // hidden-warning {{'disallowedStatic2' has hidden visibility, and external linkage; its initialization may run more than once when built into a shared library}}
  
  // OK, because immutable and compile-time-initialized
  static constexpr int allowedStatic1 = 0;
  static const float allowedStatic2 = 1;
  static constexpr int allowedStatic3 = init_constexpr(2);
  static const int allowedStatic4 = init_constexpr(3);
}

// Don't warn for non-inline functions, since they can't (legally) appear
// in more than one TU in the first place.
void has_static_locals_non_inline() {
  // Mutable
  static int allowedStatic1 = 0;
  // Initialization might run more than once
  static const double allowedStatic2 = allowedStatic1++;
}

// Everything in this function is OK because the function is TU-local
static void has_static_locals_internal() {
  static int allowedStatic1 = 0;
  static double allowedStatic2 = init_dynamic(2);
  static char allowedStatic3 = []() { return allowedStatic1++; }();

  static constexpr int allowedStatic4 = 0;
  static const float allowedStatic5 = 1;
  static constexpr int allowedStatic6 = init_constexpr(2);
  static const int allowedStatic7 = init_constexpr(3);
}

namespace {

// Everything in this function is OK because the function is also TU-local
void has_static_locals_anon() {
  static int allowedStatic1 = 0;
  static double allowedStatic2 = init_dynamic(2);
  static char allowedStatic3 = []() { return allowedStatic1++; }();

  static constexpr int allowedStatic4 = 0;
  static const float allowedStatic5 = 1;
  static constexpr int allowedStatic6 = init_constexpr(2);
  static const int allowedStatic7 = init_constexpr(3);
} 

} // Anonymous namespace

HIDDEN inline void static_local_always_hidden() {
    static int disallowedStatic1 = 3; // hidden-warning {{'disallowedStatic1' is mutable, has hidden visibility, and external linkage; it may be duplicated when built into a shared library}}
                                      // expected-warning@-1 {{'disallowedStatic1' is mutable, has hidden visibility, and external linkage; it may be duplicated when built into a shared library}}
    {
      static int disallowedStatic2 = 3; // hidden-warning {{'disallowedStatic2' is mutable, has hidden visibility, and external linkage; it may be duplicated when built into a shared library}}
                                        // expected-warning@-1 {{'disallowedStatic2' is mutable, has hidden visibility, and external linkage; it may be duplicated when built into a shared library}}
    }

    auto lmb = []() {
      static int disallowedStatic3 = 3; // hidden-warning {{'disallowedStatic3' is mutable, has hidden visibility, and external linkage; it may be duplicated when built into a shared library}}
                                        // expected-warning@-1 {{'disallowedStatic3' is mutable, has hidden visibility, and external linkage; it may be duplicated when built into a shared library}}
    };
}

DEFAULT void static_local_never_hidden() {
    static int allowedStatic1 = 3; 

    {
      static int allowedStatic2 = 3; 
    }

    auto lmb = []() {
      static int allowedStatic3 = 3;
    };
}

// Don't warn on this because it's not in a function
const int setByLambda = ([]() { static int x = 3; return x++; })();

inline void has_extern_local() {
  extern int allowedAddressExtern; // Not a definition
}

inline void has_regular_local() {
  int allowedAddressLocal = 0;
}

inline void has_thread_local() {
  // thread_local variables are static by default
  thread_local int disallowedThreadLocal = 0; // hidden-warning {{'disallowedThreadLocal' is mutable, has hidden visibility, and external linkage; it may be duplicated when built into a shared library}}
}

} // namespace StaticLocalTest

/******************************************************************************
 * Case two: Globals with external linkage
 ******************************************************************************/
namespace GlobalTest {
  // Mutable
  inline float disallowedGlobal1 = 3.14; // hidden-warning {{'disallowedGlobal1' is mutable, has hidden visibility, and external linkage; it may be duplicated when built into a shared library}}
  // Same as above, but explicitly marked inline
  inline float disallowedGlobal4 = 3.14; // hidden-warning {{'disallowedGlobal4' is mutable, has hidden visibility, and external linkage; it may be duplicated when built into a shared library}}
  
  // Initialization might run more than once
  inline const double disallowedGlobal5 = disallowedGlobal1++; // hidden-warning {{'disallowedGlobal5' has hidden visibility, and external linkage; its initialization may run more than once when built into a shared library}}

  // OK because internal linkage, so duplication is intended
  static float allowedGlobal1 = 3.14;
  const double allowedGlobal2 = init_dynamic(2);
  static const char allowedGlobal3 = []() { return disallowedGlobal1++; }();
  static inline double allowedGlobal4 = init_dynamic(2);

  // OK, because immutable and compile-time-initialized
  constexpr int allowedGlobal5 = 0;
  const float allowedGlobal6 = 1;
  constexpr int allowedGlobal7 = init_constexpr(2);
  const int allowedGlobal8 = init_constexpr(3);

  // We don't warn on this because non-inline variables can't (legally) appear
  // in more than one TU.
  float allowedGlobal9 = 3.14;
  
  // Pointers need to be double-const-qualified
  inline float& nonConstReference = disallowedGlobal1; // hidden-warning {{'nonConstReference' is mutable, has hidden visibility, and external linkage; it may be duplicated when built into a shared library}}
  const inline int& constReference = allowedGlobal5;

  inline int* nonConstPointerToNonConst = nullptr; // hidden-warning {{'nonConstPointerToNonConst' is mutable, has hidden visibility, and external linkage; it may be duplicated when built into a shared library}}
  inline int const* nonConstPointerToConst = nullptr; // hidden-warning {{'nonConstPointerToConst' is mutable, has hidden visibility, and external linkage; it may be duplicated when built into a shared library}}
  inline int* const constPointerToNonConst = nullptr; // hidden-warning {{'constPointerToNonConst' is mutable, has hidden visibility, and external linkage; it may be duplicated when built into a shared library}}
  inline int const* const constPointerToConst = nullptr;
  // Don't warn on new because it tends to generate false positives
  inline int const* const constPointerToConstNew = new int(7);

  inline int const * const * const * const nestedConstPointer = nullptr;
  inline int const * const ** const * const nestedNonConstPointer = nullptr; // hidden-warning {{'nestedNonConstPointer' is mutable, has hidden visibility, and external linkage; it may be duplicated when built into a shared library}}

  struct Test {
    static inline float disallowedStaticMember1; // hidden-warning {{'disallowedStaticMember1' is mutable, has hidden visibility, and external linkage; it may be duplicated when built into a shared library}}       
    // Defined below, in the header file
    static float disallowedStaticMember2;                                       
    // Defined in the cpp file, so won't get duplicated
    static float allowedStaticMember1;

    // Tests here are sparse because the AddrTest case below will define plenty
    // more, which aren't problematic to define (because they're immutable), but
    // may still cause problems if their address is taken.
  };

  inline float Test::disallowedStaticMember2 = 2.3; // hidden-warning {{'disallowedStaticMember2' is mutable, has hidden visibility, and external linkage; it may be duplicated when built into a shared library}}
} // namespace GlobalTest