/**
 * This file contains tests for the -Wunique_object_duplication warning.
 * See the warning's documentation for more information.
 */

#ifdef WINDOWS_TEST
#define HIDDEN
// dllimport also suffices for visibility, but those can't have definitions
#define VISIBLE __declspec(dllexport)
#else
#define HIDDEN __attribute__((visibility("hidden")))
#define VISIBLE __attribute__((visibility("default")))
#endif

// Helper functions
constexpr int init_constexpr(int x) { return x; };
extern double init_dynamic(int);

/******************************************************************************
 * Case one: Static local variables in an externally-visible function
 ******************************************************************************/
namespace StaticLocalTest {

inline void has_static_locals_external() {
  // Mutable
  static int disallowedStatic1 = 0; // hidden-warning {{'disallowedStatic1' may be duplicated when built into a shared library: it is mutable, with external linkage and hidden visibility}}
                                    // windows-warning@-1 {{'disallowedStatic1' may be duplicated when built into a shared library: it is mutable, with external linkage and no import/export annotation}}
  // Initialization might run more than once
  static const double disallowedStatic2 = disallowedStatic1++; // hidden-warning {{initialization of 'disallowedStatic2' may run twice when built into a shared library: it has external linkage and hidden visibility}}
                                                               // windows-warning@-1 {{initialization of 'disallowedStatic2' may run twice when built into a shared library: it has external linkage and no import/export annotation}}
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
    static int disallowedStatic1 = 3; // hidden-warning {{'disallowedStatic1' may be duplicated when built into a shared library: it is mutable, with external linkage and hidden visibility}}
                                      // expected-warning@-1 {{'disallowedStatic1' may be duplicated when built into a shared library: it is mutable, with external linkage and hidden visibility}}
                                      // windows-warning@-2 {{'disallowedStatic1' may be duplicated when built into a shared library: it is mutable, with external linkage and no import/export annotation}}
    {
      static int disallowedStatic2 = 3; // hidden-warning {{'disallowedStatic2' may be duplicated when built into a shared library: it is mutable, with external linkage and hidden visibility}}
                                        // expected-warning@-1 {{'disallowedStatic2' may be duplicated when built into a shared library: it is mutable, with external linkage and hidden visibility}}
                                        // windows-warning@-2 {{'disallowedStatic2' may be duplicated when built into a shared library: it is mutable, with external linkage and no import/export annotation}}
    }

    auto lmb = []() {
      static int disallowedStatic3 = 3; // hidden-warning {{'disallowedStatic3' may be duplicated when built into a shared library: it is mutable, with external linkage and hidden visibility}}
                                        // expected-warning@-1 {{'disallowedStatic3' may be duplicated when built into a shared library: it is mutable, with external linkage and hidden visibility}}
                                        // windows-warning@-2 {{'disallowedStatic3' may be duplicated when built into a shared library: it is mutable, with external linkage and no import/export annotation}}
    };
}

// Always visible
VISIBLE void static_local_never_hidden() {
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
  thread_local int disallowedThreadLocal = 0; // hidden-warning {{'disallowedThreadLocal' may be duplicated when built into a shared library: it is mutable, with external linkage and hidden visibility}}
                                              // windows-warning@-1 {{'disallowedThreadLocal' may be duplicated when built into a shared library: it is mutable, with external linkage and no import/export annotation}}
}

// Functions themselves are always immutable, so referencing them is okay
inline auto& allowedFunctionReference = has_static_locals_external;

} // namespace StaticLocalTest

/******************************************************************************
 * Case two: Globals with external linkage
 ******************************************************************************/
namespace GlobalTest {
  // Mutable
  inline float disallowedGlobal1 = 3.14; // hidden-warning {{'disallowedGlobal1' may be duplicated when built into a shared library: it is mutable, with external linkage and hidden visibility}}
                                         // windows-warning@-1 {{'disallowedGlobal1' may be duplicated when built into a shared library: it is mutable, with external linkage and no import/export annotation}}


  // Initialization might run more than once
  inline const double disallowedGlobal5 = disallowedGlobal1++; // hidden-warning {{initialization of 'disallowedGlobal5' may run twice when built into a shared library: it has external linkage and hidden visibility}}
                                                               // windows-warning@-1 {{initialization of 'disallowedGlobal5' may run twice when built into a shared library: it has external linkage and no import/export annotation}}
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
  inline float& nonConstReference = disallowedGlobal1; // hidden-warning {{'nonConstReference' may be duplicated when built into a shared library: it is mutable, with external linkage and hidden visibility}}
                                                       // windows-warning@-1 {{'nonConstReference' may be duplicated when built into a shared library: it is mutable, with external linkage and no import/export annotation}}
  const inline int& constReference = allowedGlobal5;

  inline int* nonConstPointerToNonConst = nullptr; // hidden-warning {{'nonConstPointerToNonConst' may be duplicated when built into a shared library: it is mutable, with external linkage and hidden visibility}}
                                                   // windows-warning@-1 {{'nonConstPointerToNonConst' may be duplicated when built into a shared library: it is mutable, with external linkage and no import/export annotation}}
  inline int const* nonConstPointerToConst = nullptr; // hidden-warning {{'nonConstPointerToConst' may be duplicated when built into a shared library: it is mutable, with external linkage and hidden visibility}}
                                                      // windows-warning@-1 {{'nonConstPointerToConst' may be duplicated when built into a shared library: it is mutable, with external linkage and no import/export annotation}}
  inline int* const constPointerToNonConst = nullptr; // hidden-warning {{'constPointerToNonConst' may be duplicated when built into a shared library: it is mutable, with external linkage and hidden visibility}}
                                                      // windows-warning@-1 {{'constPointerToNonConst' may be duplicated when built into a shared library: it is mutable, with external linkage and no import/export annotation}}
  inline int const* const constPointerToConst = nullptr;
  // Don't warn on new because it tends to generate false positives
  inline int const* const constPointerToConstNew = new int(7);

  inline int const * const * const * const nestedConstPointer = nullptr;
  inline int const * const ** const * const nestedNonConstPointer = nullptr; // hidden-warning {{'nestedNonConstPointer' may be duplicated when built into a shared library: it is mutable, with external linkage and hidden visibility}}
                                                                             // windows-warning@-1 {{'nestedNonConstPointer' may be duplicated when built into a shared library: it is mutable, with external linkage and no import/export annotation}}

  struct Test {
    static inline float disallowedStaticMember1; // hidden-warning {{'disallowedStaticMember1' may be duplicated when built into a shared library: it is mutable, with external linkage and hidden visibility}}
                                                 // windows-warning@-1 {{'disallowedStaticMember1' may be duplicated when built into a shared library: it is mutable, with external linkage and no import/export annotation}}
    // Defined below, in the header file
    static float disallowedStaticMember2;
    // Defined in the cpp file, so won't get duplicated
    static float allowedStaticMember1;

    // Always visible
    VISIBLE static inline float allowedStaticMember2 = 0.0;

    // Tests here are sparse because the AddrTest case below will define plenty
    // more, which aren't problematic to define (because they're immutable), but
    // may still cause problems if their address is taken.
  };

  inline float Test::disallowedStaticMember2 = 2.3; // hidden-warning {{'disallowedStaticMember2' may be duplicated when built into a shared library: it is mutable, with external linkage and hidden visibility}}
                                                    // windows-warning@-1 {{'disallowedStaticMember2' may be duplicated when built into a shared library: it is mutable, with external linkage and no import/export annotation}}

  // This is always visible, so nothing inside it will get duplicated
  struct VISIBLE NeverHidden {
    static inline float allowedStaticMember3;
    static float allowedStaticMember4;
  };

  inline float NeverHidden::allowedStaticMember4 = 3.4;
} // namespace GlobalTest

/******************************************************************************
 * Case three: Inside templates
 ******************************************************************************/

namespace TemplateTest {

// We never warn inside templates because it's usually infeasible to actually
// fix the warning.

template <typename T>
int allowedTemplate1 = 0;

template int allowedTemplate1<int>;

template <typename T>
inline int allowedTemplate2 = 0;

template int allowedTemplate2<int>;

} // namespace TemplateTest
