// RUN: %clang_cc1 -fexperimental-late-parse-attributes -fsyntax-only -verify %s

// Test that counted_by works correctly with late parsing when ActOnFields
// is called before late-parsed attributes are evaluated. This allows offset
// expressions in counted_by to be evaluated.

#define __counted_by(f)  __attribute__((counted_by(f)))

struct size_known {
  int field;
};

//==============================================================================
// Verify anonymous struct handling works correctly under current ordering.
// GetEnclosingNamedOrTopAnonRecord must correctly walk through anonymous
// structs when the struct is already marked complete.
//==============================================================================

// count in outer struct, buf in anonymous struct
struct on_pointer_anon_buf {
  int count;
  struct {
    struct size_known *buf __counted_by(count);
  };
};

// both count and buf in anonymous struct
struct on_pointer_anon_both {
  struct {
    int count;
    struct size_known *buf __counted_by(count);
  };
};

// nested anonymous structs
struct on_pointer_nested_anon {
  int count;
  struct {
    struct {
      struct size_known *buf __counted_by(count);
    };
  };
};

//==============================================================================
// Verify non-anonymous unnamed structs correctly reject counted_by if it
// references fields in the outer struct.
//==============================================================================

// count in outer, buf in non-anonymous unnamed struct: should reject
struct on_pointer_named_inner {
  int count; // expected-note{{'count' declared here}}
  struct {
    // expected-error@+1{{'counted_by' field 'count' isn't within the same struct as the annotated pointer}}
    struct size_known *buf __counted_by(count);
  } inner;
};

// both in non-anonymous unnamed struct: should accept
struct on_pointer_named_inner_both {
  struct {
    int count;
    struct size_known *buf __counted_by(count);
  } inner;
};

//==============================================================================
// TODO: allow future sizeof, offsetof expressions in the new ordering
// of ActOnFields
//==============================================================================

#define offsetof(t, d) __builtin_offsetof(t, d)

struct pointer_sizeof {
  // TODO: Allow this
  // expected-error@+1{{'counted_by' argument must be a simple declaration reference}}
  struct size_known *p __counted_by(sizeof(struct pointer_sizeof));
};

struct pointer_offsetof {
  // TODO: Allow this
  // expected-error@+1{{'counted_by' argument must be a simple declaration reference}}
  struct size_known *q __counted_by(offsetof(struct pointer_offsetof, q));
};
