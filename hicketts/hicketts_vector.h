#ifndef HICKETTS_VECTOR_H_
#define HICKETTS_VECTOR_H_

/// A minimal std::vector-like container for exercising two families of
/// attribute at once:
///
///   * EXISTING, working attributes (compile today, drive real analysis):
///       [[gsl::Owner]] / [[gsl::Pointer]]  -> -Wdangling lifetime analysis
///       [[clang::lifetimebound]]           -> return tied to *this
///       [[clang::reinitializes]]           -> "returns object to valid state"
///
///   * PROPOSED per-object-state role attributes (do NOT exist yet; shown
///     commented so the header stays buildable). These illustrate the closed
///     role vocabulary discussed in architecture.md section 4.
///
/// The attributes are macro-guarded so the SAME fixture can be compiled two
/// ways, for a clean before/after:
///     -DHICKETTS_VECTOR_NO_ATTRS   -> baseline, no attributes
///     (default)                    -> attributes on
namespace mylib {

#ifdef HICKETTS_VECTOR_NO_ATTRS
#define HV_OWNER
#define HV_POINTER
#define HV_LIFETIMEBOUND
#define HV_REINITIALIZES
#else
#define HV_OWNER [[gsl::Owner]]
#define HV_POINTER [[gsl::Pointer]]
#define HV_LIFETIMEBOUND [[clang::lifetimebound]]
#define HV_REINITIALIZES [[clang::reinitializes]]
#endif

template <typename T>
class HV_OWNER HickettsVector {
  // Tiny fixed buffer keeps the fixture simple (no allocator); big enough for
  // small tests, and irrelevant to the static lifetime analysis anyway.
  T buf_[16] = {};
  unsigned size_ = 0;

public:
  // A pointer-like handle INTO the container. Marked gsl::Pointer so the
  // lifetime analysis knows it can dangle once the owning vector dies.
  class HV_POINTER iterator {
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
  // lifetimebound: the returned reference/iterator is tied to *this, so
  // -Wdangling fires when *this is a temporary. This is the RELATIONAL hazard
  // (container <-> derived handle) and is fully covered by Owner/Pointer +
  // lifetimebound -- no per-method role needed.
  T &front() HV_LIFETIMEBOUND { return buf_[0]; }
  T &back() HV_LIFETIMEBOUND { return buf_[size_ - 1]; }
  T &operator[](unsigned i) HV_LIFETIMEBOUND { return buf_[i]; }

  iterator begin() HV_LIFETIMEBOUND { return iterator(buf_); }
  iterator end() HV_LIFETIMEBOUND { return iterator(buf_ + size_); }

  // --- State transitions ---------------------------------------------------
  // reinitializes: clear() returns the object to a defined (empty) state. This
  // already applies to both vector::clear() and optional::reset(), and is a
  // real "makes valid" role attribute shipping today.
  HV_REINITIALIZES void clear() { size_ = 0; }

  // Mutators. In a real std::vector these INVALIDATE existing iterators and
  // references -- a relational hazard that Owner/Pointer models via lifetime,
  // but which the per-object role attributes below CANNOT express (there is no
  // single per-object bit meaning "every handle I handed out is now stale").
  void push_back(const T &v) { buf_[size_++] = v; }
  void pop_back() { --size_; }

  // --- Queries -------------------------------------------------------------
  unsigned size() const { return size_; }
  bool empty() const { return size_ == 0; }

  // --- PROPOSED per-object-state role attributes (NOT YET IMPLEMENTED) ------
  // Shown commented; enabling them requires adding the attributes first.
  // Spellings are illustrative only -- see architecture.md section 4 for the
  // "closed role vocabulary" vs "capability-style" options still open.
  //
  // front()/back()/pop_back() carry a precondition: the vector is non-empty.
  //   [[clang::requires_state("non_empty")]] T &front() ...
  //   [[clang::requires_state("non_empty")]] void pop_back() ...
  //
  // Transitions that establish a state:
  //   [[clang::sets_state("empty")]]     void clear() ...
  //   [[clang::sets_state("non_empty")]] void push_back(const T &) ...
  //
  // WHY THIS IS THE INTERESTING TEST:
  //   * empty / non-empty is a SINGLE per-object predicate -- exactly the shape
  //     of optional's has_value -- so it fits the capability/role model, and a
  //     "requires non_empty" on front() is the direct analog of value()
  //     requiring engaged.
  //   * iterator invalidation is RELATIONAL, so it does NOT fit a per-object
  //     bit and stays with Owner/Pointer. That boundary is the constraint we
  //     wanted to surface: role attributes generalise to the state-predicate
  //     slice of a container, not to its aliasing hazards.
};

} // namespace mylib

#endif // HICKETTS_VECTOR_H_
