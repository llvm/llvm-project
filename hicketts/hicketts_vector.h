#ifndef HICKETTS_VECTOR_H_
#define HICKETTS_VECTOR_H_

/// A minimal std::vector-like container exercising two families of attribute,
/// sharing ONE predicate vocabulary with hicketts_optional_hybrid.h.
///
///   * REAL, shipping-today attributes (inline; drive analysis now):
///       [[gsl::Owner]] / [[gsl::Pointer]]  -> -Wdangling lifetime analysis
///       [[clang::lifetimebound]]           -> return tied to *this
///       [[clang::reinitializes]]           -> "returns object to valid state"
///
///   * PROPOSED predicate roles (do NOT exist yet; guarded by -DHV_ROLES):
///     the SAME four-verb vocabulary as the optional fixture, over a single
///     model-opaque predicate. The predicate is unnamed -- it is inert to the
///     model ("empty" == "disengaged" == set-false), so a name would only matter
///     if a type had >1 predicate. What the vector adds -- and the optional
///     fixture cannot show -- is POLARITY:
///       engaged          push_back             set the bit true
///       disengaged       clear                 set the bit false
///       assume_engaged   front/back/pop_back   precondition: bit true
///       test_disengaged  empty()               returns true iff bit FALSE  <-- negative
///     Compare optional's has_value(), a test_engaged (positive). Same bit,
///     opposite-polarity query -- which is exactly why "non_empty" never needed
///     to be a distinct predicate from "engaged": only the method polarity differs.
///
/// Out of scope (deliberately NOT predicate roles):
///   * operator[](i) -- a NUMERIC invariant (i < size), not a per-object bit.
///   * iterator invalidation -- a RELATIONAL hazard, stays with Owner/Pointer.
namespace mylib {

// Proposed predicate roles -- shared spelling with hicketts_optional_hybrid.h.
// The only guarded layer, because these attributes do not exist yet. Each
// fixture aliases only the subset of verbs it uses; the vocabulary is shared.
#ifdef HV_ROLES
#define HV_ENGAGED         [[clang::engaged]]
#define HV_DISENGAGED      [[clang::disengaged]]
#define HV_ASSUME_ENGAGED  [[clang::assume_engaged]]
#define HV_TEST_DISENGAGED [[clang::test_disengaged]]
#else
#define HV_ENGAGED
#define HV_DISENGAGED
#define HV_ASSUME_ENGAGED
#define HV_TEST_DISENGAGED
#endif

template <typename T>
class [[gsl::Owner]] HickettsVector {
  // Tiny fixed buffer keeps the fixture simple (no allocator); irrelevant to the
  // static analysis anyway.
  T buf_[16] = {};
  unsigned size_ = 0;

public:
  // A pointer-like handle INTO the container. gsl::Pointer lets the lifetime
  // analysis know it can dangle once the owning vector dies.
  class [[gsl::Pointer]] iterator {
    T *p_ = nullptr;

  public:
    iterator() = default;
    explicit iterator(T *p) : p_(p) {}
    T &operator*() const { return *p_; }
    iterator &operator++() {
      ++p_;
      return *this;
    }
    bool operator==(const iterator &o) const { return p_ == o.p_; }
    bool operator!=(const iterator &o) const { return p_ != o.p_; }
  };

  HickettsVector() = default;

  // --- Element access -------------------------------------------------------
  // assume_engaged: precondition that the vector is non-empty (front/back on an
  // empty vector is UB) -- the SAME role as optional's value(). lifetimebound is
  // the orthogonal RELATIONAL hazard (handle tied to *this).
  HV_ASSUME_ENGAGED T &front() [[clang::lifetimebound]] { return buf_[0]; }
  HV_ASSUME_ENGAGED T &back() [[clang::lifetimebound]] { return buf_[size_ - 1]; }

  // operator[]: NO predicate role. Its precondition is numeric (i < size), which
  // a single-bit predicate model does not track. lifetimebound still applies.
  T &operator[](unsigned i) [[clang::lifetimebound]] { return buf_[i]; }

  // Iterators: no predicate role (begin() == end() on an empty vector is
  // well-defined); lifetimebound handles the dangling hazard.
  iterator begin() [[clang::lifetimebound]] { return iterator(buf_); }
  iterator end() [[clang::lifetimebound]] { return iterator(buf_ + size_); }

  // --- State transitions ---------------------------------------------------
  // clear(): disengaged (proposed predicate role) + reinitializes (real, ships
  // today, consumed by other checks). Two different consumers -- NOT redundant.
  HV_DISENGAGED [[clang::reinitializes]] void clear() { size_ = 0; }

  // push_back -> engaged (the bit becomes true).
  HV_ENGAGED void push_back(const T &v) { buf_[size_++] = v; }

  // pop_back: assume_engaged (precondition non-empty) but leaves the bit UNKNOWN
  // afterwards -- it may or may not still be non-empty -- so it carries no set
  // role. (In a real vector this also invalidates handles: a RELATIONAL hazard
  // modelled by Owner/Pointer, not by any per-object bit.)
  HV_ASSUME_ENGAGED void pop_back() { --size_; }

  // --- Queries -------------------------------------------------------------
  unsigned size() const { return size_; } // numeric, no role

  // empty(): test_disengaged -- returns true iff the bit is FALSE. This is the
  // negative-polarity query that optional's has_value() (test_engaged) is not.
  HV_TEST_DISENGAGED bool empty() const { return size_ == 0; }
};

} // namespace mylib

#endif // HICKETTS_VECTOR_H_
