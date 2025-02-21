/**
 * This file contains tests for the -Wunique_object_duplication warning.
 * See the warning's documentation for more information.
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
  static int disallowedStatic1 = 0; // hidden-warning {{'disallowedStatic1' may be duplicated when built into a shared library: it is mutable, has hidden visibility, and external linkage}}
  // Initialization might run more than once
  static const double disallowedStatic2 = disallowedStatic1++; // hidden-warning {{initializeation of 'disallowedStatic2' may run twice when built into a shared library: it has hidden visibility and external linkage}}
  
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
}

namespace {

// Everything in this function is OK because the function is also TU-local
void has_static_locals_anon() {
  static int allowedStatic1 = 0;
  static double allowedStatic2 = init_dynamic(2);
  static char allowedStatic3 = []() { return allowedStatic1++; }();
  static constexpr int allowedStatic4 = init_constexpr(3);
} 

} // Anonymous namespace

HIDDEN inline void static_local_always_hidden() {
    static int disallowedStatic1 = 3; // hidden-warning {{'disallowedStatic1' may be duplicated when built into a shared library: it is mutable, has hidden visibility, and external linkage}}
                                      // expected-warning@-1 {{'disallowedStatic1' may be duplicated when built into a shared library: it is mutable, has hidden visibility, and external linkage}}
    {
      static int disallowedStatic2 = 3; // hidden-warning {{'disallowedStatic2' may be duplicated when built into a shared library: it is mutable, has hidden visibility, and external linkage}}
                                        // expected-warning@-1 {{'disallowedStatic2' may be duplicated when built into a shared library: it is mutable, has hidden visibility, and external linkage}}
    }

    auto lmb = []() {
      static int disallowedStatic3 = 3; // hidden-warning {{'disallowedStatic3' may be duplicated when built into a shared library: it is mutable, has hidden visibility, and external linkage}}
                                        // expected-warning@-1 {{'disallowedStatic3' may be duplicated when built into a shared library: it is mutable, has hidden visibility, and external linkage}}
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
  thread_local int disallowedThreadLocal = 0; // hidden-warning {{'disallowedThreadLocal' may be duplicated when built into a shared library: it is mutable, has hidden visibility, and external linkage}}
}

// Functions themselves are always immutable, so referencing them is okay
inline auto& allowedFunctionReference = has_static_locals_external;

} // namespace StaticLocalTest

/******************************************************************************
 * Case two: Globals with external linkage
 ******************************************************************************/
namespace GlobalTest {
  // Mutable
  inline float disallowedGlobal1 = 3.14; // hidden-warning {{'disallowedGlobal1' may be duplicated when built into a shared library: it is mutable, has hidden visibility, and external linkage}}
  
  // Initialization might run more than once
  inline const double disallowedGlobal5 = disallowedGlobal1++; // hidden-warning {{initializeation of 'disallowedGlobal5' may run twice when built into a shared library: it has hidden visibility and external linkage}}

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
  inline float& nonConstReference = disallowedGlobal1; // hidden-warning {{'nonConstReference' may be duplicated when built into a shared library: it is mutable, has hidden visibility, and external linkage}}
  const inline int& constReference = allowedGlobal5;

  inline int* nonConstPointerToNonConst = nullptr; // hidden-warning {{'nonConstPointerToNonConst' may be duplicated when built into a shared library: it is mutable, has hidden visibility, and external linkage}}
  inline int const* nonConstPointerToConst = nullptr; // hidden-warning {{'nonConstPointerToConst' may be duplicated when built into a shared library: it is mutable, has hidden visibility, and external linkage}}
  inline int* const constPointerToNonConst = nullptr; // hidden-warning {{'constPointerToNonConst' may be duplicated when built into a shared library: it is mutable, has hidden visibility, and external linkage}}
  inline int const* const constPointerToConst = nullptr;
  // Don't warn on new because it tends to generate false positives
  inline int const* const constPointerToConstNew = new int(7);

  inline int const * const * const * const nestedConstPointer = nullptr;
  inline int const * const ** const * const nestedNonConstPointer = nullptr; // hidden-warning {{'nestedNonConstPointer' may be duplicated when built into a shared library: it is mutable, has hidden visibility, and external linkage}}

  struct Test {
    static inline float disallowedStaticMember1; // hidden-warning {{'disallowedStaticMember1' may be duplicated when built into a shared library: it is mutable, has hidden visibility, and external linkage}}       
    // Defined below, in the header file
    static float disallowedStaticMember2;                                       
    // Defined in the cpp file, so won't get duplicated
    static float allowedStaticMember1;

    // Tests here are sparse because the AddrTest case below will define plenty
    // more, which aren't problematic to define (because they're immutable), but
    // may still cause problems if their address is taken.
  };

  inline float Test::disallowedStaticMember2 = 2.3; // hidden-warning {{'disallowedStaticMember2' may be duplicated when built into a shared library: it is mutable, has hidden visibility, and external linkage}}
} // namespace GlobalTest

/******************************************************************************
 * Case three: Inside templates
 ******************************************************************************/

namespace TemplateTest {

template <typename T>
int disallowedTemplate1 = 0; // hidden-warning {{'disallowedTemplate1<int>' may be duplicated when built into a shared library: it is mutable, has hidden visibility, and external linkage}}

template int disallowedTemplate1<int>; // hidden-note {{in instantiation of}}


// Should work for implicit instantiation as well
template <typename T>
int disallowedTemplate2 = 0; // hidden-warning {{'disallowedTemplate2<int>' may be duplicated when built into a shared library: it is mutable, has hidden visibility, and external linkage}}

int implicit_instantiate() {
  return disallowedTemplate2<int>; // hidden-note {{in instantiation of}}
}


// Ensure we only get warnings for templates that are actually instantiated
template <typename T>
int maybeAllowedTemplate = 0; // Not instantiated, so no warning here

template <typename T>
int maybeAllowedTemplate<T*> = 1; // hidden-warning {{'maybeAllowedTemplate<int *>' may be duplicated when built into a shared library: it is mutable, has hidden visibility, and external linkage}}

template <>
int maybeAllowedTemplate<bool> = 2; // hidden-warning {{'maybeAllowedTemplate<bool>' may be duplicated when built into a shared library: it is mutable, has hidden visibility, and external linkage}}

template int maybeAllowedTemplate<int*>; // hidden-note {{in instantiation of}}



// Should work the same for static class members
template <typename T>
struct S {
  static int staticMember;
};

template <typename T>
int S<T>::staticMember = 0; // Never instantiated

// T* specialization
template <typename T>
struct S<T*> {
  static int staticMember;
};

template <typename T>
int S<T*>::staticMember = 1; // hidden-warning {{'staticMember' may be duplicated when built into a shared library: it is mutable, has hidden visibility, and external linkage}}

template class S<int*>; // hidden-note {{in instantiation of}}

// T& specialization, implicitly instantiated
template <typename T>
struct S<T&> {
  static int staticMember;
};

template <typename T>
int S<T&>::staticMember = 2; // hidden-warning {{'staticMember' may be duplicated when built into a shared library: it is mutable, has hidden visibility, and external linkage}}

int implicit_instantiate2() {
  return S<bool&>::staticMember; // hidden-note {{in instantiation of}}
}


// Should work for static locals as well
template <typename T>
int* wrapper() {
  static int staticLocal; // hidden-warning {{'staticLocal' may be duplicated when built into a shared library: it is mutable, has hidden visibility, and external linkage}}
  return &staticLocal;
}

template <>
int* wrapper<int*>() {
  static int staticLocal; // hidden-warning {{'staticLocal' may be duplicated when built into a shared library: it is mutable, has hidden visibility, and external linkage}}
  return &staticLocal;
}

auto dummy = wrapper<bool>(); // hidden-note {{in instantiation of}}
} // namespace TemplateTest