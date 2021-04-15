// RUN: %check_clang_tidy -std=c++14 %s bugprone-unhandled-exception-at-new %t -- -- -fexceptions

namespace std {
typedef __typeof__(sizeof(0)) size_t;
enum class align_val_t : std::size_t {};
class exception {};
class bad_alloc : public exception {};
class bad_array_new_length : public bad_alloc {};
struct nothrow_t {};
extern const nothrow_t nothrow;
} // namespace std

void *operator new(std::size_t, const std::nothrow_t &) noexcept;
void *operator new(std::size_t, std::align_val_t, const std::nothrow_t &) noexcept;
void *operator new(std::size_t, void *) noexcept;

class A {};
typedef std::bad_alloc badalloc1;
using badalloc2 = std::bad_alloc;
using badalloc3 = std::bad_alloc &;

void *operator new(std::size_t, int, int);
void *operator new(std::size_t, int, int, int) noexcept;

struct ClassSpecificNew {
  void *operator new(std::size_t);
  void *operator new(std::size_t, std::align_val_t);
  void *operator new(std::size_t, int, int) noexcept;
  void *operator new(std::size_t, int, int, int);
};

void f1() noexcept {
  int *I1 = new int;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: missing exception handler for allocation failure at 'new'
  try {
    int *I2 = new int;
    try {
      int *I3 = new int;
    } catch (A) {
    }
  } catch (std::bad_alloc) {
  }

  try {
    int *I = new int;
  } catch (std::bad_alloc &) {
  }

  try {
    int *I = new int;
  } catch (const std::bad_alloc &) {
  }

  try {
    int *I = new int;
  } catch (badalloc1) {
  }

  try {
    int *I = new int;
  } catch (badalloc1 &) {
  }

  try {
    int *I = new int;
  } catch (const badalloc1 &) {
  }

  try {
    int *I = new int;
  } catch (badalloc2) {
  }

  try {
    int *I = new int;
  } catch (badalloc3) {
  }

  try {
    int *I = new int;
    // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: missing exception handler for allocation failure at 'new'
  } catch (std::bad_alloc *) {
  }

  try {
    int *I = new int;
    // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: missing exception handler for allocation failure at 'new'
  } catch (A) {
  }
}

void f2() noexcept {
  try {
    int *I = new int;
  } catch (A) {
  } catch (std::bad_alloc) {
  }

  try {
    int *I = new int;
  } catch (...) {
  }

  try {
    int *I = new int;
  } catch (const std::exception &) {
  }

  try {
    int *I = new int;
    // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: missing exception handler for allocation failure at 'new'
  } catch (const std::bad_array_new_length &) {
  }
}

void f_new_nothrow() noexcept {
  int *I1 = new (std::nothrow) int;
  int *I2 = new (static_cast<std::align_val_t>(1), std::nothrow) int;
}

void f_new_placement() noexcept {
  char buf[100];
  int *I = new (buf) int;
}

void f_new_user_defined() noexcept {
  int *I1 = new (1, 2) int;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: missing exception handler for allocation failure at 'new'
  int *I2 = new (1, 2, 3) int;
}

void f_class_specific() noexcept {
  ClassSpecificNew *N1 = new ClassSpecificNew;
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: missing exception handler for allocation failure at 'new'
  ClassSpecificNew *N2 = new (static_cast<std::align_val_t>(1)) ClassSpecificNew;
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: missing exception handler for allocation failure at 'new'
  ClassSpecificNew *N3 = new (1, 2) ClassSpecificNew;
  ClassSpecificNew *N4 = new (1, 2, 3) ClassSpecificNew;
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: missing exception handler for allocation failure at 'new'
}

void f_est_none() {
  int *I = new int;
}

void f_est_noexcept_false() noexcept(false) {
  int *I = new int;
}

void f_est_noexcept_true() noexcept(true) {
  int *I = new int;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: missing exception handler for allocation failure at 'new'
}

void f_est_dynamic_none() throw() {
  int *I = new int;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: missing exception handler for allocation failure at 'new'
}

void f_est_dynamic_1() throw(std::bad_alloc) {
  int *I = new int;
}

void f_est_dynamic_2() throw(A) {
  // the exception specification list is not checked
  int *I = new int;
}

void f_try() noexcept try {
  int *I = new int;
} catch (const std::bad_alloc &) {
}

void f_try_bad() noexcept try {
  int *I = new int;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: missing exception handler for allocation failure at 'new'
} catch (const A &) {
}

void f_embedded_try() noexcept {
  try {
    try {
      int *I = new int;
    } catch (const A &) {
    }
  } catch (const std::bad_alloc &) {
  }
}

template <bool P>
void f_est_noexcept_dependent_unused() noexcept(P) {
  int *I = new int;
}

template <bool P>
void f_est_noexcept_dependent_used() noexcept(P) {
  int *I = new int;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: missing exception handler for allocation failure at 'new'
}

template <class T>
void f_dependent_new() {
  T *X = new T;
}

void f_1() {
  f_est_noexcept_dependent_used<true>();
}
