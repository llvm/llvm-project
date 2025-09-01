// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 %s

static_assert(!__builtin_lt_synthesises_from_spaceship()); // expected-error {{expected a type}}
static_assert(!__builtin_lt_synthesises_from_spaceship(int)); // expected-error {{type trait requires 2 arguments; have 1 argument}}
static_assert(!__builtin_lt_synthesises_from_spaceship(int, int, int)); // expected-error {{type trait requires 2 arguments; have 3 argument}}
static_assert(!__builtin_lt_synthesises_from_spaceship(int, 0)); // expected-error {{expected a type}}

static_assert(!__builtin_le_synthesises_from_spaceship()); // expected-error {{expected a type}}
static_assert(!__builtin_le_synthesises_from_spaceship(int)); // expected-error {{type trait requires 2 arguments; have 1 argument}}
static_assert(!__builtin_le_synthesises_from_spaceship(int, int, int)); // expected-error {{type trait requires 2 arguments; have 3 argument}}
static_assert(!__builtin_le_synthesises_from_spaceship(int, 0)); // expected-error {{expected a type}}

static_assert(!__builtin_gt_synthesises_from_spaceship()); // expected-error {{expected a type}}
static_assert(!__builtin_gt_synthesises_from_spaceship(int)); // expected-error {{type trait requires 2 arguments; have 1 argument}}
static_assert(!__builtin_gt_synthesises_from_spaceship(int, int, int)); // expected-error {{type trait requires 2 arguments; have 3 argument}}
static_assert(!__builtin_gt_synthesises_from_spaceship(int, 0)); // expected-error {{expected a type}}

static_assert(!__builtin_ge_synthesises_from_spaceship()); // expected-error {{expected a type}}
static_assert(!__builtin_ge_synthesises_from_spaceship(int)); // expected-error {{type trait requires 2 arguments; have 1 argument}}
static_assert(!__builtin_ge_synthesises_from_spaceship(int, int, int)); // expected-error {{type trait requires 2 arguments; have 3 argument}}
static_assert(!__builtin_ge_synthesises_from_spaceship(int, 0)); // expected-error {{expected a type}}

namespace std {
  struct strong_ordering {
    int n;
    constexpr operator int() const { return n; }
    static const strong_ordering less, equal, greater;
  };
  constexpr strong_ordering strong_ordering::less = {-1};
  constexpr strong_ordering strong_ordering::equal = {0};
  constexpr strong_ordering strong_ordering::greater = {1};
}

struct DefaultSpaceship {
  friend auto operator<=>(DefaultSpaceship, DefaultSpaceship) = default;
};

static_assert(__builtin_lt_synthesises_from_spaceship(const DefaultSpaceship&, const DefaultSpaceship&));
static_assert(__builtin_le_synthesises_from_spaceship(const DefaultSpaceship&, const DefaultSpaceship&));
static_assert(__builtin_gt_synthesises_from_spaceship(const DefaultSpaceship&, const DefaultSpaceship&));
static_assert(__builtin_ge_synthesises_from_spaceship(const DefaultSpaceship&, const DefaultSpaceship&));

struct CustomSpaceship {
  int i;

  friend auto operator<=>(CustomSpaceship lhs, CustomSpaceship rhs) {
    return rhs.i <=> lhs.i;
  }
};

static_assert(__builtin_lt_synthesises_from_spaceship(const CustomSpaceship&, const CustomSpaceship&));
static_assert(__builtin_le_synthesises_from_spaceship(const CustomSpaceship&, const CustomSpaceship&));
static_assert(__builtin_gt_synthesises_from_spaceship(const CustomSpaceship&, const CustomSpaceship&));
static_assert(__builtin_ge_synthesises_from_spaceship(const CustomSpaceship&, const CustomSpaceship&));

struct CustomLT {
  int i;

  friend auto operator<(CustomLT lhs, CustomLT rhs) {
    return rhs.i < lhs.i;
  }
};

static_assert(!__builtin_lt_synthesises_from_spaceship(const CustomLT&, const CustomLT&));
static_assert(!__builtin_le_synthesises_from_spaceship(const CustomLT&, const CustomLT&));
static_assert(!__builtin_gt_synthesises_from_spaceship(const CustomLT&, const CustomLT&));
static_assert(!__builtin_ge_synthesises_from_spaceship(const CustomLT&, const CustomLT&));

struct CustomLE {
  int i;

  friend auto operator<=(CustomLE lhs, CustomLE rhs) {
    return rhs.i < lhs.i;
  }
};

static_assert(!__builtin_lt_synthesises_from_spaceship(const CustomLE&, const CustomLE&));
static_assert(!__builtin_le_synthesises_from_spaceship(const CustomLE&, const CustomLE&));
static_assert(!__builtin_gt_synthesises_from_spaceship(const CustomLE&, const CustomLE&));
static_assert(!__builtin_ge_synthesises_from_spaceship(const CustomLE&, const CustomLE&));

struct CustomGT {
  int i;

  friend auto operator>(CustomGT lhs, CustomGT rhs) {
    return rhs.i < lhs.i;
  }
};

static_assert(!__builtin_lt_synthesises_from_spaceship(const CustomGT&, const CustomGT&));
static_assert(!__builtin_le_synthesises_from_spaceship(const CustomGT&, const CustomGT&));
static_assert(!__builtin_gt_synthesises_from_spaceship(const CustomGT&, const CustomGT&));
static_assert(!__builtin_ge_synthesises_from_spaceship(const CustomGT&, const CustomGT&));

struct CustomGE {
  int i;

  friend auto operator>=(CustomGE lhs, CustomGE rhs) {
    return rhs.i < lhs.i;
  }
};

static_assert(!__builtin_lt_synthesises_from_spaceship(const CustomGE&, const CustomGE&));
static_assert(!__builtin_le_synthesises_from_spaceship(const CustomGE&, const CustomGE&));
static_assert(!__builtin_gt_synthesises_from_spaceship(const CustomGE&, const CustomGE&));
static_assert(!__builtin_ge_synthesises_from_spaceship(const CustomGE&, const CustomGE&));

struct CustomLTAndSpaceship {
  int i;

  friend auto operator<=>(CustomLTAndSpaceship lhs, CustomLTAndSpaceship rhs) {
    return rhs.i <=> lhs.i;
  }

  friend auto operator<(CustomLTAndSpaceship lhs, CustomLTAndSpaceship rhs) {
    return rhs.i < lhs.i;
  }
};

static_assert(!__builtin_lt_synthesises_from_spaceship(const CustomLTAndSpaceship&, const CustomLTAndSpaceship&));
static_assert(__builtin_le_synthesises_from_spaceship(const CustomLTAndSpaceship&, const CustomLTAndSpaceship&));
static_assert(__builtin_gt_synthesises_from_spaceship(const CustomLTAndSpaceship&, const CustomLTAndSpaceship&));
static_assert(__builtin_ge_synthesises_from_spaceship(const CustomLTAndSpaceship&, const CustomLTAndSpaceship&));

struct CustomLEAndSpaceship {
  int i;

  friend auto operator<=>(CustomLEAndSpaceship lhs, CustomLEAndSpaceship rhs) {
    return rhs.i <=> lhs.i;
  }

  friend auto operator<=(CustomLEAndSpaceship lhs, CustomLEAndSpaceship rhs) {
    return rhs.i < lhs.i;
  }
};

static_assert(__builtin_lt_synthesises_from_spaceship(const CustomLEAndSpaceship&, const CustomLEAndSpaceship&));
static_assert(!__builtin_le_synthesises_from_spaceship(const CustomLEAndSpaceship&, const CustomLEAndSpaceship&));
static_assert(__builtin_gt_synthesises_from_spaceship(const CustomLEAndSpaceship&, const CustomLEAndSpaceship&));
static_assert(__builtin_ge_synthesises_from_spaceship(const CustomLEAndSpaceship&, const CustomLEAndSpaceship&));

struct CustomGTAndSpaceship {
  int i;

  friend auto operator<=>(CustomGTAndSpaceship lhs, CustomGTAndSpaceship rhs) {
    return rhs.i <=> lhs.i;
  }

  friend auto operator>(CustomGTAndSpaceship lhs, CustomGTAndSpaceship rhs) {
    return rhs.i < lhs.i;
  }
};

static_assert(__builtin_lt_synthesises_from_spaceship(const CustomGTAndSpaceship&, const CustomGTAndSpaceship&));
static_assert(__builtin_le_synthesises_from_spaceship(const CustomGTAndSpaceship&, const CustomGTAndSpaceship&));
static_assert(!__builtin_gt_synthesises_from_spaceship(const CustomGTAndSpaceship&, const CustomGTAndSpaceship&));
static_assert(__builtin_ge_synthesises_from_spaceship(const CustomGTAndSpaceship&, const CustomGTAndSpaceship&));

struct CustomGEAndSpaceship {
  int i;

  friend auto operator<=>(CustomGEAndSpaceship lhs, CustomGEAndSpaceship rhs) {
    return rhs.i <=> lhs.i;
  }

  friend auto operator>=(CustomGEAndSpaceship lhs, CustomGEAndSpaceship rhs) {
    return rhs.i < lhs.i;
  }
};

static_assert(__builtin_lt_synthesises_from_spaceship(const CustomGEAndSpaceship&, const CustomGEAndSpaceship&));
static_assert(__builtin_le_synthesises_from_spaceship(const CustomGEAndSpaceship&, const CustomGEAndSpaceship&));
static_assert(__builtin_gt_synthesises_from_spaceship(const CustomGEAndSpaceship&, const CustomGEAndSpaceship&));
static_assert(!__builtin_ge_synthesises_from_spaceship(const CustomGEAndSpaceship&, const CustomGEAndSpaceship&));

struct DefaultedCmpAndSpaceship {
  int i;

  friend auto operator<=>(DefaultedCmpAndSpaceship lhs, DefaultedCmpAndSpaceship rhs) {
    return rhs.i <=> lhs.i;
  }

  friend bool operator<(DefaultedCmpAndSpaceship lhs, DefaultedCmpAndSpaceship rhs) = default;
  friend bool operator<=(DefaultedCmpAndSpaceship lhs, DefaultedCmpAndSpaceship rhs) = default;
  friend bool operator>(DefaultedCmpAndSpaceship lhs, DefaultedCmpAndSpaceship rhs) = default;
  friend bool operator>=(DefaultedCmpAndSpaceship lhs, DefaultedCmpAndSpaceship rhs) = default;
};

// TODO: This should probably return true
static_assert(!__builtin_lt_synthesises_from_spaceship(const DefaultedCmpAndSpaceship&, const DefaultedCmpAndSpaceship&));
static_assert(!__builtin_le_synthesises_from_spaceship(const DefaultedCmpAndSpaceship&, const DefaultedCmpAndSpaceship&));
static_assert(!__builtin_gt_synthesises_from_spaceship(const DefaultedCmpAndSpaceship&, const DefaultedCmpAndSpaceship&));
static_assert(!__builtin_ge_synthesises_from_spaceship(const DefaultedCmpAndSpaceship&, const DefaultedCmpAndSpaceship&));

struct DifferentTypes {
  int i;

  friend auto operator<=>(DifferentTypes lhs, int rhs) {
    return rhs <=> lhs.i;
  }
};

static_assert(__builtin_lt_synthesises_from_spaceship(const DifferentTypes&, const int&));
static_assert(__builtin_le_synthesises_from_spaceship(const DifferentTypes&, const int&));
static_assert(__builtin_gt_synthesises_from_spaceship(const DifferentTypes&, const int&));
static_assert(__builtin_ge_synthesises_from_spaceship(const DifferentTypes&, const int&));

// TODO: Should this return true? It's technically not synthesized from spaceship, but it behaves exactly as-if it was
static_assert(!__builtin_lt_synthesises_from_spaceship(int, int));
static_assert(!__builtin_le_synthesises_from_spaceship(int, int));
static_assert(!__builtin_gt_synthesises_from_spaceship(int, int));
static_assert(!__builtin_ge_synthesises_from_spaceship(int, int));
